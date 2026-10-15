// The tensor compiler and its executor: operations at the same depth share a
// layer, which gathers its inputs, evaluates one sparse affine projection, then
// applies gather reads, index reads, ReGLU gates and boolean rounding. A port of
// compileTensorProgram and executeTensorProgram in
// src/runtime/numeric-ui-core.ts; docs/execution-model.md describes the layout.

use std::collections::HashMap;

use crate::circuit::{
    evaluate_boolean, resolve_index_position, AtomKind, Circuit, IndexRead, OneHotGroup, OutOfRange, Term, Wire,
};
use crate::fp::{fround, relu_js};

#[derive(Copy, Clone, Debug, Default)]
pub struct CompileOptions {
    /// Place each live operation one layer after the deepest value it reads in
    /// the compiled program, instead of at its build-time depth.
    pub recompute_depths: bool,
}

/// One resolved one-hot group inside a layer: the executor takes the dot of
/// its packed bits with 0..N-1 once, shared by every read of the group.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorGatherCompute {
    pub slot: usize,
    pub bit_positions: Vec<u32>,
}

/// One lowered masked sum: the selected candidate's packed value is added onto
/// the row's bias, times the group's mass for a group that may be all zero.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorGatherRead {
    pub row: usize,
    pub slot: usize,
    pub masked: bool,
    pub candidate_positions: Vec<u32>,
}

/// One index read inside a layer. When `candidate_positions` is empty,
/// candidate k sits at `candidate_base + k × candidate_stride`.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorIndexRead {
    pub row: usize,
    pub index_row: usize,
    pub below_zero: bool,
    pub above_zero: bool,
    pub candidate_count: usize,
    pub candidate_base: i64,
    pub candidate_stride: i64,
    pub candidate_positions: Vec<u32>,
}

/// One layer. Projection rows: one per output, then one gate row per ReGLU,
/// then one index row per index read.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorLayer {
    pub input_indices: Vec<u32>,
    pub input_size: usize,
    pub output_size: usize,
    pub bias: Vec<f32>,
    pub gate_rows: Vec<u32>,
    pub boolean_rows: Vec<u32>,
    pub row_offsets: Vec<u32>,
    pub columns: Vec<u32>,
    pub weights: Vec<f32>,
    pub gather_computes: Vec<TensorGatherCompute>,
    pub gather_reads: Vec<TensorGatherRead>,
    pub index_reads: Vec<TensorIndexRead>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorProgram {
    pub layers: Vec<TensorLayer>,
    pub roots: Vec<u32>,
    pub source_size: usize,
    /// Source cells plus live operations.
    pub value_count: usize,
    /// `wire_order[packed]` is the wire a packed value holds.
    pub wire_order: Vec<u32>,
    pub width: usize,
    pub gather_slot_count: usize,
}

// A recognized one-hot-masked sum: the affine's terms are products of the
// group's bits against these candidate value wires, ordered by group position.
#[derive(Clone, Debug)]
struct GatherPlan {
    group_index: usize,
    candidate_wires: Vec<Wire>,
}

struct PackedCandidateTable {
    candidate_count: usize,
    candidate_base: i64,
    candidate_stride: i64,
    candidate_positions: Vec<u32>,
}

/// The buffers one evaluation computes in, reused across evaluations.
#[derive(Clone, Debug)]
pub struct TensorWorkspace {
    /// After an evaluation, `packed_values[i]` is the value of wire
    /// `program.wire_order[i]`.
    pub packed_values: Vec<f32>,
    plans: Vec<LayerPlan>,
    /// A layer's gate and index rows; its output rows go straight to
    /// `packed_values`.
    scratch: Vec<f32>,
    gather_indices: Vec<i32>,
    gather_masses: Vec<f32>,
}

impl TensorWorkspace {
    pub fn new(program: &TensorProgram) -> TensorWorkspace {
        let scratch = program.layers.iter().map(|layer| layer.bias.len() - layer.output_size).max().unwrap_or(0);
        let plans = program.layers.iter().map(LayerPlan::new).collect();
        TensorWorkspace {
            packed_values: vec![0.0; program.value_count],
            plans,
            scratch: vec![0.0; scratch],
            gather_indices: vec![0; program.gather_slot_count],
            gather_masses: vec![0.0; program.gather_slot_count],
        }
    }
}

// A layer's projection rows, read straight from the packed values. Rows of
// one layer are independent, so they run grouped by shape: a single-term row
// as one flat entry, the others with their terms. Output rows land in the
// layer's packed values, gate and index rows in the scratch rows.
#[derive(Clone, Debug, Default)]
struct LayerPlan {
    output_singles: Vec<SingleRow>,
    output_rows: Vec<MultiRow>,
    scratch_singles: Vec<SingleRow>,
    scratch_rows: Vec<MultiRow>,
    /// (packed position, weight) of every multi-term row.
    terms: Vec<(u32, f32)>,
}

#[derive(Copy, Clone, Debug)]
struct SingleRow {
    row: u32,
    column: u32,
    weight: f32,
    bias: f32,
}

#[derive(Copy, Clone, Debug)]
struct MultiRow {
    row: u32,
    bias: f32,
    start: u32,
    end: u32,
}

impl LayerPlan {
    fn new(layer: &TensorLayer) -> LayerPlan {
        let mut plan = LayerPlan::default();
        for row in 0..layer.bias.len() {
            let (start, end) = (layer.row_offsets[row] as usize, layer.row_offsets[row + 1] as usize);
            let is_output = row < layer.output_size;
            let row_index = if is_output { row } else { row - layer.output_size } as u32;
            let bias = layer.bias[row];
            if end - start == 1 {
                let single = SingleRow {
                    row: row_index,
                    column: layer.input_indices[layer.columns[start] as usize],
                    weight: layer.weights[start],
                    bias,
                };
                if is_output { plan.output_singles.push(single) } else { plan.scratch_singles.push(single) }
            } else {
                let term_start = plan.terms.len() as u32;
                for index in start..end {
                    plan.terms.push((layer.input_indices[layer.columns[index] as usize], layer.weights[index]));
                }
                let multi = MultiRow { row: row_index, bias, start: term_start, end: plan.terms.len() as u32 };
                if is_output { plan.output_rows.push(multi) } else { plan.scratch_rows.push(multi) }
            }
        }
        plan
    }

    // value = bias + w0 x0 + w1 x1 + ..., in FP32, in the program's term order.
    fn run(&self, earlier: &[f32], outputs: &mut [f32], scratch: &mut [f32]) {
        for (singles, rows, target) in [(&self.output_singles, &self.output_rows, outputs), (&self.scratch_singles, &self.scratch_rows, scratch)] {
            for single in singles {
                target[single.row as usize] = single.bias + single.weight * earlier[single.column as usize];
            }
            for row in rows {
                let mut value = row.bias;
                for &(column, weight) in &self.terms[row.start as usize..row.end as usize] {
                    value += weight * earlier[column as usize];
                }
                target[row.row as usize] = value;
            }
        }
    }
}

/// A compiled program of some roots, with its own workspace, e.g. for a host
/// that evaluates the same circuit every frame.
pub struct TensorEvaluator {
    pub program: TensorProgram,
    pub workspace: TensorWorkspace,
    source_ranges: Vec<crate::circuit::Range>,
    positions: Vec<usize>,
    outputs: Vec<f32>,
}

impl TensorEvaluator {
    /// The roots' values. The returned slice is overwritten on the next call.
    pub fn evaluate(&mut self, source: &[f32]) -> &[f32] {
        validate_source(&self.source_ranges, source);
        self.workspace.packed_values[..source.len()].copy_from_slice(source);
        self.evaluate_current()
    }

    /// Write one source cell for `evaluate_current`, checked against its range.
    pub fn set_source(&mut self, wire: usize, value: f32) {
        let range = self.source_ranges[wire];
        if !(value as f64 >= range.minimum && value as f64 <= range.maximum) {
            panic!("source wire {wire} is outside its assumed range");
        }
        self.workspace.packed_values[wire] = value;
    }

    /// Write source cells for `evaluate_current` without checking them, e.g.
    /// cells that `set_source` overwrites before the next evaluation.
    pub fn set_source_unchecked(&mut self, start: usize, values: &[f32]) {
        self.workspace.packed_values[start..start + values.len()].copy_from_slice(values);
    }

    /// Check the source cells already written, e.g. after `set_source_unchecked`.
    pub fn check_source(&self, wires: impl IntoIterator<Item = usize>) {
        for wire in wires {
            let value = self.workspace.packed_values[wire] as f64;
            let range = self.source_ranges[wire];
            if !(value >= range.minimum && value <= range.maximum) {
                panic!("source wire {wire} is outside its assumed range");
            }
        }
    }

    /// The roots' values for the source cells written so far, e.g. by a host
    /// that changes a few cells of a large source table between evaluations.
    pub fn evaluate_current(&mut self) -> &[f32] {
        run_layers(&self.program, &mut self.workspace);
        for (output, &position) in self.outputs.iter_mut().zip(&self.positions) {
            *output = self.workspace.packed_values[position];
        }
        &self.outputs
    }
}

fn validate_source(ranges: &[crate::circuit::Range], source: &[f32]) {
    assert!(source.len() == ranges.len(), "source table has the wrong length");
    for (wire, (&value, range)) in source.iter().zip(ranges).enumerate() {
        let value = value as f64;
        if !(value >= range.minimum && value <= range.maximum) {
            panic!("source wire {wire} is outside its assumed range");
        }
    }
}

impl Circuit {
    /// Compile the operations that `roots` read, every operation without
    /// roots. With `lower_gathers`, one-hot-masked sums become gather reads.
    pub fn compile_tensor_program(
        &self,
        roots: Option<&[Wire]>,
        lower_gathers: bool,
        options: CompileOptions,
    ) -> TensorProgram {
        let width = self.width();
        let normalized_roots: Vec<Wire> = match roots {
            Some(roots) => roots.to_vec(),
            None => (0..self.operations.len()).map(|index| Wire::from_position(self.source_size + index)).collect(),
        };
        let gather_plans = if lower_gathers { plan_gather_lowering(self) } else { HashMap::new() };
        let live = find_live_operations(self, &normalized_roots, &gather_plans);
        let compiled_depths = options.recompute_depths.then(|| compute_compiled_depths(self, &live, &gather_plans));
        let mut operations_by_layer: Vec<Vec<usize>> = Vec::new();
        for (operation_index, operation) in self.operations.iter().enumerate() {
            if !live[operation_index] {
                continue;
            }
            let depth = match &compiled_depths {
                None => operation.depth,
                Some(depths) => depths[operation_index],
            } as usize;
            if operations_by_layer.len() < depth {
                operations_by_layer.resize(depth, Vec::new());
            }
            operations_by_layer[depth - 1].push(operation_index);
        }

        let live_count = live.iter().filter(|&&live| live).count();
        let value_count = self.source_size + live_count;
        let mut wire_order = vec![0u32; value_count];
        let mut packed_index_by_wire = vec![-1i64; width];
        for wire in 0..self.source_size {
            wire_order[wire] = wire as u32;
            packed_index_by_wire[wire] = wire as i64;
        }
        let mut available = self.source_size;
        let mut input_column_by_wire = vec![-1i64; width];
        let mut slot_by_group: HashMap<usize, usize> = HashMap::new();
        let mut packed_tables: HashMap<usize, PackedCandidateTable> = HashMap::new();
        let mut layers = Vec::new();
        for operation_indices in &operations_by_layer {
            if operation_indices.is_empty() {
                continue;
            }
            let output_size = operation_indices.len();
            // Insertion-ordered, like a JavaScript Set.
            let mut input_wires: Vec<Wire> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for &operation_index in operation_indices {
                // Gather rows read packed values directly by position.
                if gather_plans.contains_key(&operation_index) {
                    continue;
                }
                let operation = &self.operations[operation_index];
                for &(wire, _) in operation.left_terms.iter().chain(&operation.gate_terms) {
                    if seen.insert(wire) {
                        input_wires.push(wire);
                    }
                }
            }
            input_wires.sort_by_key(|wire| packed_index_by_wire[wire.index()]);
            let input_size = input_wires.len();
            let mut gate_count = 0;
            let mut index_read_count = 0;
            for &operation_index in operation_indices {
                match self.operations[operation_index].kind {
                    AtomKind::Reglu => gate_count += 1,
                    AtomKind::Index => index_read_count += 1,
                    _ => {}
                }
            }
            let input_indices: Vec<u32> = input_wires
                .iter()
                .map(|wire| {
                    let position = packed_index_by_wire[wire.index()];
                    assert!(position >= 0, "index must be a nonnegative integer");
                    position as u32
                })
                .collect();
            for (column, wire) in input_wires.iter().enumerate() {
                input_column_by_wire[wire.index()] = column as i64;
            }
            let projection_size = output_size + gate_count + index_read_count;
            let mut bias = vec![0f32; projection_size];
            let mut gate_rows = Vec::new();
            let mut boolean_rows = Vec::new();
            let mut projection_terms: Vec<&[Term]> = vec![&[]; projection_size];
            let mut pending_gather_reads: Vec<(usize, &GatherPlan)> = Vec::new();
            let mut pending_index_reads: Vec<(usize, usize, IndexRead)> = Vec::new();
            let mut gate_row = 0;
            let mut index_read_row = 0;
            for (row, &operation_index) in operation_indices.iter().enumerate() {
                let operation = &self.operations[operation_index];
                let output_wire = self.source_size + operation_index;
                let packed_output = available + row;
                wire_order[packed_output] = output_wire as u32;
                packed_index_by_wire[output_wire] = packed_output as i64;
                bias[row] = fround(operation.left_bias) as f32;
                if let Some(plan) = gather_plans.get(&operation_index) {
                    // The lowered sum keeps its bias and stores no weights.
                    pending_gather_reads.push((row, plan));
                    continue;
                }
                projection_terms[row] = &operation.left_terms;
                match operation.kind {
                    AtomKind::Reglu => {
                        let projection_row = output_size + gate_row;
                        bias[projection_row] = fround(operation.gate_bias) as f32;
                        gate_rows.push(row as u32);
                        projection_terms[projection_row] = &operation.gate_terms;
                        gate_row += 1;
                    }
                    AtomKind::Boolean => boolean_rows.push(row as u32),
                    AtomKind::Index => {
                        // The output row keeps zero bias and no weights.
                        let index_row = output_size + gate_count + index_read_row;
                        bias[index_row] = fround(operation.gate_bias) as f32;
                        projection_terms[index_row] = &operation.gate_terms;
                        pending_index_reads.push((row, index_row, operation.index_read.unwrap()));
                        index_read_row += 1;
                    }
                    AtomKind::Affine => {}
                }
            }
            let mut gather_computes = Vec::new();
            let mut gather_reads = Vec::new();
            for (row, plan) in pending_gather_reads {
                let group = &self.one_hot_groups[plan.group_index];
                let slot = match slot_by_group.get(&plan.group_index) {
                    Some(&slot) => slot,
                    None => {
                        // First read of this group: resolve its index here, once.
                        let slot = slot_by_group.len();
                        slot_by_group.insert(plan.group_index, slot);
                        let bit_positions = group
                            .members
                            .iter()
                            .map(|member| available_position(&packed_index_by_wire, *member, available, "gather bit"))
                            .collect();
                        gather_computes.push(TensorGatherCompute { slot, bit_positions });
                        slot
                    }
                };
                let candidate_positions = plan
                    .candidate_wires
                    .iter()
                    .map(|candidate| available_position(&packed_index_by_wire, *candidate, available, "gather candidate"))
                    .collect();
                gather_reads.push(TensorGatherRead { row, slot, masked: group.allow_all_zero, candidate_positions });
            }
            let mut index_reads = Vec::new();
            for (row, index_row, read) in pending_index_reads {
                let packed = packed_tables
                    .entry(read.table)
                    .or_insert_with(|| pack_candidate_table(self, read.table, &packed_index_by_wire, available));
                index_reads.push(TensorIndexRead {
                    row,
                    index_row,
                    below_zero: read.below == OutOfRange::Zero,
                    above_zero: read.above == OutOfRange::Zero,
                    candidate_count: packed.candidate_count,
                    candidate_base: packed.candidate_base,
                    candidate_stride: packed.candidate_stride,
                    candidate_positions: packed.candidate_positions.clone(),
                });
            }
            let mut row_offsets = vec![0u32; projection_size + 1];
            for row in 0..projection_size {
                row_offsets[row + 1] = row_offsets[row] + projection_terms[row].len() as u32;
            }
            let mut columns = Vec::with_capacity(row_offsets[projection_size] as usize);
            let mut weights = Vec::with_capacity(row_offsets[projection_size] as usize);
            for terms in &projection_terms {
                for &(wire, coefficient) in terms.iter() {
                    let column = input_column_by_wire[wire.index()];
                    assert!(column >= 0, "tensor dependency is not available");
                    columns.push(column as u32);
                    weights.push(fround(coefficient) as f32);
                }
            }
            for wire in &input_wires {
                input_column_by_wire[wire.index()] = -1;
            }
            layers.push(TensorLayer {
                input_indices,
                input_size,
                output_size,
                bias,
                gate_rows,
                boolean_rows,
                row_offsets,
                columns,
                weights,
                gather_computes,
                gather_reads,
                index_reads,
            });
            available += output_size;
        }
        TensorProgram {
            layers,
            roots: normalized_roots.iter().map(|root| root.index() as u32).collect(),
            source_size: self.source_size,
            value_count,
            wire_order,
            width,
            gather_slot_count: slot_by_group.len(),
        }
    }

    /// Every wire's value from a compiled program, with 0 for the wires that
    /// the program removed. Without a program, compiles every operation.
    pub fn evaluate_tensor(&self, source: &[f32], program: Option<&TensorProgram>) -> Vec<f32> {
        self.validate_source_ranges(source);
        let compiled;
        let program = match program {
            Some(program) => program,
            None => {
                compiled = self.compile_tensor_program(None, true, CompileOptions::default());
                &compiled
            }
        };
        let mut workspace = TensorWorkspace::new(program);
        execute_tensor_program(source, program, &mut workspace);
        let mut values = vec![0f32; program.width];
        for (packed, &wire) in program.wire_order.iter().enumerate() {
            values[wire as usize] = workspace.packed_values[packed];
        }
        values
    }

    /// An evaluator of a program of `roots` that returns the roots' values.
    pub fn create_tensor_evaluator(&self, roots: &[Wire], options: CompileOptions) -> TensorEvaluator {
        self.create_tensor_program_evaluator(self.compile_tensor_program(Some(roots), true, options))
    }

    /// An evaluator of a program that `compile_tensor_program` already returned.
    pub fn create_tensor_program_evaluator(&self, program: TensorProgram) -> TensorEvaluator {
        let positions = program
            .roots
            .iter()
            .map(|&root| program.wire_order.iter().position(|&wire| wire == root).expect("root is in the program"))
            .collect::<Vec<_>>();
        TensorEvaluator {
            workspace: TensorWorkspace::new(&program),
            outputs: vec![0.0; positions.len()],
            positions,
            source_ranges: self.ranges[..self.source_size].to_vec(),
            program,
        }
    }
}

fn available_position(packed_index_by_wire: &[i64], wire: Wire, available: usize, label: &str) -> u32 {
    let position = packed_index_by_wire[wire.index()];
    assert!(position >= 0 && (position as usize) < available, "{label} is not available");
    position as u32
}

// Store a candidate table as a base position and a stride when the candidates
// sit at evenly spaced packed positions, e.g. a run of source cells; otherwise
// one position per candidate.
fn pack_candidate_table(
    circuit: &Circuit,
    table: usize,
    packed_index_by_wire: &[i64],
    available: usize,
) -> PackedCandidateTable {
    let positions: Vec<u32> = circuit.candidate_tables[table]
        .candidates
        .iter()
        .map(|candidate| available_position(packed_index_by_wire, *candidate, available, "index read candidate"))
        .collect();
    let base = positions[0] as i64;
    let stride = if positions.len() > 1 { positions[1] as i64 - base } else { 0 };
    let evenly_spaced = positions.iter().enumerate().skip(2).all(|(index, &position)| position as i64 == base + index as i64 * stride);
    PackedCandidateTable {
        candidate_count: positions.len(),
        candidate_base: base,
        candidate_stride: stride,
        candidate_positions: if evenly_spaced { Vec::new() } else { positions },
    }
}

fn find_live_operations(circuit: &Circuit, roots: &[Wire], gather_plans: &HashMap<usize, GatherPlan>) -> Vec<bool> {
    let mut live = vec![false; circuit.operations.len()];
    let mut pending = roots.to_vec();
    // A table read by many operations contributes its candidates once.
    let mut visited_tables = std::collections::HashSet::new();
    while let Some(wire) = pending.pop() {
        assert!(wire.index() < circuit.width(), "wire is outside the compiled circuit");
        if wire.index() < circuit.source_size {
            continue;
        }
        let operation_index = wire.index() - circuit.source_size;
        if live[operation_index] {
            continue;
        }
        live[operation_index] = true;
        if let Some(plan) = gather_plans.get(&operation_index) {
            // A lowered sum reads the group's bits and every candidate directly.
            pending.extend(&circuit.one_hot_groups[plan.group_index].members);
            pending.extend(&plan.candidate_wires);
            continue;
        }
        let operation = &circuit.operations[operation_index];
        pending.extend(operation.left_terms.iter().map(|&(wire, _)| wire));
        pending.extend(operation.gate_terms.iter().map(|&(wire, _)| wire));
        if let Some(read) = &operation.index_read
            && visited_tables.insert(read.table)
        {
            pending.extend(&circuit.candidate_tables[read.table].candidates);
        }
    }
    live
}

// The depth of every live operation from what it reads in the compiled program.
fn compute_compiled_depths(circuit: &Circuit, live: &[bool], gather_plans: &HashMap<usize, GatherPlan>) -> Vec<u32> {
    let mut depths = vec![0u32; circuit.operations.len()];
    let depth_of = |depths: &[u32], wire: Wire| {
        if wire.index() < circuit.source_size { 0 } else { depths[wire.index() - circuit.source_size] }
    };
    let mut table_depths: HashMap<usize, u32> = HashMap::new();
    for operation_index in 0..circuit.operations.len() {
        if !live[operation_index] {
            continue;
        }
        let mut depth = 0;
        match gather_plans.get(&operation_index) {
            None => {
                let operation = &circuit.operations[operation_index];
                for &(wire, _) in operation.left_terms.iter().chain(&operation.gate_terms) {
                    depth = depth.max(depth_of(&depths, wire));
                }
                if let Some(read) = &operation.index_read {
                    let table_depth = *table_depths.entry(read.table).or_insert_with(|| {
                        circuit.candidate_tables[read.table]
                            .candidates
                            .iter()
                            .map(|&candidate| depth_of(&depths, candidate))
                            .max()
                            .unwrap_or(0)
                    });
                    depth = depth.max(table_depth);
                }
            }
            Some(plan) => {
                for &wire in circuit.one_hot_groups[plan.group_index].members.iter().chain(&plan.candidate_wires) {
                    depth = depth.max(depth_of(&depths, wire));
                }
            }
        }
        depths[operation_index] = depth + 1;
    }
    depths
}

#[derive(Copy, Clone, Debug)]
struct MaskedProduct {
    coefficient: f64,
    left_wire: Wire,
    gate_wire: Wire,
    gate_coefficient: f64,
    exact_multiply: bool,
}

// Undo the builder's affine composition for one side of a ReGLU: the builder's
// deduplication map recovers the wire whose definition a consumer copied in.
// Sign -1 means the side equals the negated wire.
fn resolve_affine_side(circuit: &Circuit, terms: &[Term], bias: f64, allow_negated: bool) -> Option<(Wire, f64)> {
    if bias == 0.0 && terms.len() == 1 {
        let (wire, coefficient) = terms[0];
        if coefficient == 1.0 {
            return Some((wire, 1.0));
        }
        if coefficient == -1.0 && allow_negated {
            return Some((wire, -1.0));
        }
    }
    if terms.is_empty() {
        return None;
    }
    if let Some(direct) = circuit.find_affine(terms, bias) {
        return Some((direct, 1.0));
    }
    if !allow_negated {
        return None;
    }
    let negated: Vec<Term> = terms.iter().map(|&(wire, coefficient)| (wire, -coefficient)).collect();
    circuit.find_affine(&negated, -bias).map(|wire| (wire, -1.0))
}

// Parse one term of a candidate masked sum: a ReGLU whose left side is some
// wire and whose gate side is a wire or its negation, the shapes multiply
// builds for bit × value.
fn parse_masked_product(circuit: &Circuit, wire: Wire, coefficient: f64) -> Option<MaskedProduct> {
    if wire.index() < circuit.source_size {
        return None;
    }
    let operation = &circuit.operations[wire.index() - circuit.source_size];
    if let Some(crate::circuit::Macro::Multiply { left, right }) = operation.macro_ {
        return Some(MaskedProduct { coefficient, left_wire: left, gate_wire: right, gate_coefficient: 1.0, exact_multiply: true });
    }
    if operation.kind != AtomKind::Reglu {
        return None;
    }
    let (left_wire, _) = resolve_affine_side(circuit, &operation.left_terms, operation.left_bias, false)?;
    let (gate_wire, sign) = resolve_affine_side(circuit, &operation.gate_terms, operation.gate_bias, true)?;
    Some(MaskedProduct { coefficient, left_wire, gate_wire, gate_coefficient: sign, exact_multiply: false })
}

// Recognize affines whose terms are products of a declared one-hot group
// against value wires, covering every group position exactly once. Such a sum
// equals an indexed read of the selected candidate, bit for bit.
fn plan_gather_lowering(circuit: &Circuit) -> HashMap<usize, GatherPlan> {
    let mut plans = HashMap::new();
    if circuit.one_hot_groups.is_empty() {
        return plans;
    }
    let mut groups_by_wire: HashMap<Wire, Vec<usize>> = HashMap::new();
    let positions_by_group: Vec<HashMap<Wire, usize>> = circuit
        .one_hot_groups
        .iter()
        .enumerate()
        .map(|(group_index, group)| {
            let mut positions = HashMap::new();
            for (position, &member) in group.members.iter().enumerate() {
                positions.insert(member, position);
                groups_by_wire.entry(member).or_default().push(group_index);
            }
            positions
        })
        .collect();
    for (operation_index, operation) in circuit.operations.iter().enumerate() {
        if operation.kind != AtomKind::Affine || operation.left_terms.is_empty() {
            continue;
        }
        let mut products = Vec::with_capacity(operation.left_terms.len());
        let mut parsed = true;
        for &(wire, coefficient) in &operation.left_terms {
            if coefficient != 1.0 && coefficient != -1.0 {
                parsed = false;
                break;
            }
            match parse_masked_product(circuit, wire, coefficient) {
                Some(product) => products.push(product),
                None => {
                    parsed = false;
                    break;
                }
            }
        }
        if !parsed {
            continue;
        }
        // Every term of a matching sum references the group, so the first
        // product's groups are the only ones to try, in insertion order.
        let first = products[0];
        let mut candidate_groups: Vec<usize> = Vec::new();
        for wire in [first.gate_wire, first.left_wire] {
            for &group in groups_by_wire.get(&wire).map(Vec::as_slice).unwrap_or(&[]) {
                if !candidate_groups.contains(&group) {
                    candidate_groups.push(group);
                }
            }
        }
        for group_index in candidate_groups {
            if let Some(candidate_wires) = attempt_gather_match(
                circuit,
                &products,
                &circuit.one_hot_groups[group_index],
                &positions_by_group[group_index],
            ) {
                plans.insert(operation_index, GatherPlan { group_index, candidate_wires });
                break;
            }
        }
    }
    plans
}

// Interpret the products as one complete read of the group: every position
// contributes exactly one value wire, and every term provably equals that
// masked read in FP32. Value wires need finite declared ranges, since a
// non-selected f(Infinity × 0) would poison the dense sum with NaN.
fn attempt_gather_match(
    circuit: &Circuit,
    products: &[MaskedProduct],
    group: &OneHotGroup,
    position_by_member: &HashMap<Wire, usize>,
) -> Option<Vec<Wire>> {
    // The negative arms of mixed-sign multiplies, each of which must find its
    // positive partner.
    let mut negative_arms: HashMap<(Wire, Wire), bool> = HashMap::new();
    for product in products {
        if product.coefficient == -1.0
            && product.gate_coefficient == -1.0
            && position_by_member.contains_key(&product.left_wire)
            && !position_by_member.contains_key(&product.gate_wire)
            && negative_arms.insert((product.left_wire, product.gate_wire), false).is_some()
        {
            return None;
        }
    }
    let mut options_by_term: Vec<Vec<(usize, Wire)>> = Vec::new();
    for product in products {
        let MaskedProduct { coefficient, left_wire, gate_wire, gate_coefficient, exact_multiply } = *product;
        if coefficient == 1.0 && gate_coefficient == 1.0 {
            let mut options = Vec::new();
            // Bit in the gate: f(value × max(0, bit)) selects any finite value.
            if let Some(&gate_position) = position_by_member.get(&gate_wire)
                && circuit.has_finite_range(left_wire)
            {
                options.push((gate_position, left_wire));
            }
            if let Some(&left_position) = position_by_member.get(&left_wire) {
                let arm = negative_arms.get_mut(&(left_wire, gate_wire));
                let gate_range = circuit.ranges[gate_wire.index()];
                if exact_multiply && circuit.has_finite_range(gate_wire) {
                    options.push((left_position, gate_wire));
                } else if let Some(used) = arm.filter(|used| !**used && circuit.has_finite_range(gate_wire)) {
                    *used = true;
                    options.push((left_position, gate_wire));
                } else if gate_range.minimum >= 0.0 && gate_range.maximum.is_finite() {
                    // Bit on the left: f(bit × max(0, value)) reads the value
                    // only for a non-negative value.
                    options.push((left_position, gate_wire));
                }
            }
            if options.is_empty() {
                return None;
            }
            options_by_term.push(options);
        } else if gate_coefficient == -1.0 && position_by_member.contains_key(&gate_wire) {
            // A bit behind a negated gate contributes a signed zero, unless the
            // left value could be infinite.
            if !circuit.has_finite_range(left_wire) {
                return None;
            }
        } else if coefficient == -1.0 && gate_coefficient == -1.0 && position_by_member.contains_key(&left_wire) {
            // The negative arm of a mixed-sign pair, verified as used below.
        } else {
            return None;
        }
    }
    if negative_arms.values().any(|used| !used) {
        return None;
    }
    // Fixed terms first, then terms whose bit could sit on either side take
    // whichever position is still open.
    let mut candidate_wires: Vec<Option<Wire>> = vec![None; group.members.len()];
    let mut flexible = Vec::new();
    for options in options_by_term {
        if options.len() == 1 {
            let (position, value_wire) = options[0];
            if candidate_wires[position].is_some() {
                return None;
            }
            candidate_wires[position] = Some(value_wire);
        } else {
            flexible.push(options);
        }
    }
    for options in flexible {
        let open = options.iter().find(|(position, _)| candidate_wires[*position].is_none())?;
        candidate_wires[open.0] = Some(open.1);
    }
    candidate_wires.into_iter().collect()
}

/// Run a compiled program on one source table into the workspace.
pub fn execute_tensor_program(source: &[f32], program: &TensorProgram, workspace: &mut TensorWorkspace) {
    assert!(source.len() == program.source_size, "source table has the wrong length");
    workspace.packed_values[..source.len()].copy_from_slice(source);
    run_layers(program, workspace);
}

// The layers on the source cells already in `packed_values`. A layer only reads
// values before its own, so its output rows are computed in place.
fn run_layers(program: &TensorProgram, workspace: &mut TensorWorkspace) {
    let TensorWorkspace { packed_values, plans, scratch, gather_indices, gather_masses } = workspace;
    let mut available = program.source_size;
    for (layer, plan) in program.layers.iter().zip(plans.iter()) {
        let output_size = layer.output_size;
        let (earlier, rest) = packed_values.split_at_mut(available);
        let outputs = &mut rest[..output_size];
        let earlier = &*earlier;
        plan.run(earlier, outputs, scratch);
        if !layer.gather_reads.is_empty() {
            for compute in &layer.gather_computes {
                // The dot of exact 0/1 bits with 0..N-1 is an exact small integer.
                let mut index = 0f64;
                let mut mass = 0f64;
                for (position, &bit_position) in compute.bit_positions.iter().enumerate() {
                    let bit = earlier[bit_position as usize] as f64;
                    index += position as f64 * bit;
                    mass += bit;
                }
                gather_indices[compute.slot] = to_int32(index);
                gather_masses[compute.slot] = mass as f32;
            }
            for read in &layer.gather_reads {
                // Clamp, so a violated one-hot contract reads a wrong value
                // instead of reading out of bounds.
                let limit = read.candidate_positions.len() as i32 - 1;
                let index = gather_indices[read.slot].max(0).min(limit);
                let mut contribution = earlier[read.candidate_positions[index as usize] as usize];
                if read.masked {
                    contribution *= gather_masses[read.slot];
                }
                *row_mut(outputs, scratch, read.row) += contribution;
            }
        }
        for read in &layer.index_reads {
            let index = *row_mut(outputs, scratch, read.index_row);
            let position = resolve_index_position(index as f64, read.candidate_count, read.below_zero, read.above_zero);
            *row_mut(outputs, scratch, read.row) = if position >= 0 {
                let packed = if read.candidate_positions.is_empty() {
                    (read.candidate_base + position * read.candidate_stride) as usize
                } else {
                    read.candidate_positions[position as usize] as usize
                };
                earlier[packed]
            } else if position == -1 {
                0.0
            } else {
                f32::NAN
            };
        }
        for (gate_row, &output_row) in layer.gate_rows.iter().enumerate() {
            let gate = scratch[gate_row] as f64;
            let output = row_mut(outputs, scratch, output_row as usize);
            *output = (*output as f64 * relu_js(gate)) as f32;
        }
        for &boolean_row in &layer.boolean_rows {
            let value = row_mut(outputs, scratch, boolean_row as usize);
            *value = evaluate_boolean(*value as f64) as f32;
        }
        available += output_size;
    }
}

// A projection row: an output row in the layer's packed values, or a gate or
// index row in the scratch rows after them.
fn row_mut<'a>(outputs: &'a mut [f32], scratch: &'a mut [f32], row: usize) -> &'a mut f32 {
    let output_size = outputs.len();
    if row < output_size { &mut outputs[row] } else { &mut scratch[row - output_size] }
}

// JavaScript's ToInt32, which an Int32Array store applies.
fn to_int32(value: f64) -> i32 {
    if !value.is_finite() {
        return 0;
    }
    (value.trunc().rem_euclid(4_294_967_296.0) as u64 as u32) as i32
}
