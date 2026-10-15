// Regex search as the work of one code point, a port of
// src/json-search/search-step.ts: the class of the code point, one step of a
// reverse pass, and one step of a forward pass. See that file for the passes.

use super::kinds::{display_kind, DISPLAY_KINDS};
use super::loop_program::{
    create_stage, Binding, Buffer, Field, GroupKind, LoopProgram, LoopResult, OutputGroup, Register, Repeat, Write,
};
use super::regex::{
    and_not, bit_wire, declare_data_search_cells, emit_classes, none_of, not, or_exclusive, write_pattern, Bit, CompiledPattern,
    DataSearchCells, SearchCapacity, StepAnd,
};
use crate::cells::{input, input_bit, SourceTable};
use crate::circuit::{Circuit, OutOfRange, Wire};

pub const KIND_COUNT: usize = DISPLAY_KINDS.len();
pub const CLASS_CAPACITY: usize = 16;
pub const BOUNDARY_CAPACITY: usize = 32;

/// Which display kinds a masked search reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchScope {
    Everything,
    Strings,
    Keys,
    StringsAndKeys,
    Kinds(Vec<usize>),
}

pub fn searchable_kinds(scope: &SearchScope) -> Vec<usize> {
    match scope {
        SearchScope::Everything => (0..KIND_COUNT).collect(),
        SearchScope::Strings => vec![display_kind::STRING],
        SearchScope::Keys => vec![display_kind::KEY],
        SearchScope::StringsAndKeys => vec![display_kind::STRING, display_kind::KEY],
        SearchScope::Kinds(kinds) => kinds.clone(),
    }
}

fn emit_masked_code_point(circuit: &mut Circuit, code_point: Wire, kind: Wire, searchable: &[Wire]) -> Wire {
    let bit = circuit.index_read_affine(&[(kind, 1.0)], 0.0, searchable, OutOfRange::Clamp, OutOfRange::Clamp, None);
    circuit.assume_grid(bit, 1.0);
    let kept = circuit.reglu(&[(code_point, 1.0)], 1.0, &[(bit, 1.0)], 0.0);
    let searched = circuit.round_affine(&[(kept, 1.0)], -1.0);
    assert!(circuit.grids[searched.index()] == Some(1.0), "the searched code point is on a grid of 1");
    searched
}

#[derive(Copy, Clone, Debug)]
pub struct SearchLoopOptions {
    pub positions: usize,
    pub cells_per_document: usize,
    pub masked: bool,
    pub stop_at_length: bool,
    pub bit_fields: bool,
}

pub struct SearchLoop {
    pub program: LoopProgram,
    pub capacity: SearchCapacity,
    pub options: SearchLoopOptions,
    data_cells: DataSearchCells,
    searchable_cells: Option<Vec<Wire>>,
    /// How many shared cells a new pattern writes.
    pub pattern_cells: usize,
}

impl SearchLoop {
    /// Writes a pattern's cells into a copy of the shared values.
    pub fn write_pattern(&self, shared: &mut [f32], compiled: &CompiledPattern) -> Result<(), String> {
        write_pattern(shared, &self.data_cells, self.capacity, compiled)
    }

    /// Writes which kinds a masked search reads.
    pub fn write_scope(&self, shared: &mut [f32], scope: &SearchScope) {
        let cells = self.searchable_cells.as_ref().expect("only a masked search has a scope");
        let kinds = searchable_kinds(scope);
        for (kind, wire) in cells.iter().enumerate() {
            shared[wire.index()] = if kinds.contains(&kind) { 1.0 } else { 0.0 };
        }
    }
}

pub fn build_search_loop(capacity: SearchCapacity, options: SearchLoopOptions) -> SearchLoop {
    let states = capacity.states;
    let SearchLoopOptions { positions, masked, .. } = options;
    assert!(positions < options.cells_per_document, "the text's length sits after the code points");
    let mut table = SourceTable::new();
    let pattern = declare_data_search_cells(&mut table, "search", capacity, 1);
    let mask = masked.then(|| {
        let kind = table.cell(
            "mask[0].kind",
            input((0.0, KIND_COUNT as f64 - 1.0)).initial(display_kind::PAST_END as f64).grid(1.0),
        );
        let searchable: Vec<Wire> =
            (0..KIND_COUNT).map(|index| table.cell(format!("mask[0].searchable[{index}]"), input_bit(1.0))).collect();
        (kind, searchable)
    });
    let not_finish_cells: Vec<Wire> =
        (0..states).map(|state| table.cell(format!("notFinish[{state}]"), input_bit(1.0))).collect();
    let classes = capacity.classes as f64;
    let code_point_class = table.cell("codePointClass", input((0.0, classes)).initial(classes).grid(1.0));
    let not_reach_cells: Vec<Wire> = (0..states).map(|state| table.cell(format!("notReach[{state}]"), input_bit(1.0))).collect();
    let reached_cells: Vec<Wire> = (0..states).map(|state| table.cell(format!("reached[{state}]"), input_bit(0.0))).collect();
    let match_count = table.cell("matchCount", input((0.0, positions as f64)).grid(1.0));
    let mut circuit = table.create_circuit();
    let c = &mut circuit;
    let code_point = pattern.text[0];
    let step_and = StepAnd::new(&pattern);

    let start_not_finish: Vec<Wire> = pattern
        .accepting
        .iter()
        .map(|&accepting| {
            let bit = not(c, Bit::Wire(accepting));
            bit_wire(c, bit)
        })
        .collect();

    let searched = match &mask {
        None => code_point,
        Some((kind, searchable)) => emit_masked_code_point(c, code_point, *kind, searchable),
    };
    let masked_cells = DataSearchCells { text: vec![searched], ..pattern.clone() };
    let class_of = emit_classes(c, &masked_cells)[0];
    let not_finish: Vec<Bit> = not_finish_cells.iter().map(|&wire| Bit::Wire(wire)).collect();
    let not_reach: Vec<Wire> = (0..states)
        .map(|from| {
            let steps: Vec<Bit> = (0..states)
                .map(|to| {
                    let finish = not(c, not_finish[to]);
                    step_and.apply(c, class_of, from, to, finish)
                })
                .collect();
            let none = none_of(c, &steps);
            bit_wire(c, none)
        })
        .collect();
    let next_not_finish: Vec<Wire> = (0..states)
        .map(|state| {
            let bit = and_not(c, Bit::Wire(not_reach[state]), Bit::Wire(pattern.accepting[state]));
            bit_wire(c, bit)
        })
        .collect();

    let continuing: Vec<Bit> = (0..states)
        .map(|state| and_not(c, Bit::Wire(reached_cells[state]), Bit::Wire(not_reach_cells[state])))
        .collect();
    let not_continuing = none_of(c, &continuing);
    let start = and_not(c, not_continuing, Bit::Wire(not_reach_cells[0]));
    let live: Vec<Bit> = continuing
        .iter()
        .enumerate()
        .map(|(state, &bit)| if state == 0 { or_exclusive(c, bit, start) } else { bit })
        .collect();
    let any_continuing = not(c, not_continuing);
    let highlighted = or_exclusive(c, any_continuing, start);
    let highlighted_wire = bit_wire(c, highlighted);
    let start_wire = bit_wire(c, start);
    let mark = c.affine(&[(highlighted_wire, 1.0), (start_wire, 1.0)], 0.0);
    let next_reached: Vec<Wire> = (0..states)
        .map(|to| {
            let steps: Vec<Bit> = (0..states).map(|from| step_and.apply(c, code_point_class, from, to, live[from])).collect();
            let none = none_of(c, &steps);
            let any = not(c, none);
            bit_wire(c, any)
        })
        .collect();
    let start_wire = bit_wire(c, start);
    let next_match_count = c.affine(&[(match_count, 1.0), (start_wire, 1.0)], 0.0);

    let finished = table.finish(&mut circuit);
    let mut registers: Vec<Register> =
        (0..states).map(|state| Register { name: format!("notFinish[{state}]"), initial: 1.0 }).collect();
    registers.extend((0..states).map(|state| Register { name: format!("reached[{state}]"), initial: 0.0 }));
    registers.push(Register { name: "matchCount".into(), initial: 0.0 });
    let not_finish_register = |state: usize| state;
    let reached_register = |state: usize| states + state;
    let match_count_register = 2 * states;
    let class_field = 0;
    let not_reach_field = 1;
    let mut item_bindings = vec![(code_point, Binding::Item { buffer: 0 })];
    if let Some((kind, _)) = &mask {
        item_bindings.push((*kind, Binding::Item { buffer: 1 }));
    }
    let mut buffers = vec![Buffer { name: "documents".into(), cells_per_document: options.cells_per_document }];
    if masked {
        buffers.push(Buffer { name: "kinds".into(), cells_per_document: positions });
    }
    let start_stage = create_stage(
        0,
        &circuit,
        "start",
        Repeat::Once,
        0,
        Vec::new(),
        start_not_finish
            .iter()
            .enumerate()
            .map(|(state, &wire)| (wire, Write::Register { register: not_finish_register(state) }))
            .collect(),
    );
    let mut reverse_bindings = item_bindings;
    reverse_bindings.extend(
        not_finish_cells.iter().enumerate().map(|(state, &wire)| (wire, Binding::Register { register: not_finish_register(state) })),
    );
    let mut reverse_outputs = vec![(class_of, Write::Field { field: class_field, component: 0 })];
    reverse_outputs.extend(
        not_reach.iter().enumerate().map(|(state, &wire)| (wire, Write::Field { field: not_reach_field, component: state })),
    );
    reverse_outputs.extend(
        next_not_finish.iter().enumerate().map(|(state, &wire)| (wire, Write::Register { register: not_finish_register(state) })),
    );
    let reverse_stage = create_stage(0, &circuit, "reverse", Repeat::Backward, 0, reverse_bindings, reverse_outputs);
    let mut forward_bindings = vec![(code_point_class, Binding::Field { field: class_field, component: 0 })];
    forward_bindings.extend(
        not_reach_cells.iter().enumerate().map(|(state, &wire)| (wire, Binding::Field { field: not_reach_field, component: state })),
    );
    forward_bindings.extend(
        reached_cells.iter().enumerate().map(|(state, &wire)| (wire, Binding::Register { register: reached_register(state) })),
    );
    forward_bindings.push((match_count, Binding::Register { register: match_count_register }));
    let mut forward_outputs = vec![(mark, Write::Position { group: 1 })];
    forward_outputs.extend(
        next_reached.iter().enumerate().map(|(state, &wire)| (wire, Write::Register { register: reached_register(state) })),
    );
    forward_outputs.push((next_match_count, Write::Register { register: match_count_register }));
    let forward_stage = create_stage(0, &circuit, "forward", Repeat::Forward, 0, forward_bindings, forward_outputs);
    let program = LoopProgram {
        positions,
        buffers,
        registers,
        fields: vec![
            Field { name: "codePointClass".into(), width: 1, bits: false },
            Field { name: "notReach".into(), width: states, bits: options.bit_fields },
        ],
        output_groups: vec![
            OutputGroup { name: "matchCount".into(), kind: GroupKind::Document, values_per_document: 1, fill: 0.0 },
            OutputGroup { name: "marks".into(), kind: GroupKind::Position, values_per_document: positions, fill: 0.0 },
        ],
        circuits: vec![circuit],
        stages: vec![start_stage, reverse_stage, forward_stage],
        results: vec![LoopResult { group: 0, column: 0, register: match_count_register }],
        shared_values: finished.initial_source,
        length_cell: options.stop_at_length.then_some((0, options.cells_per_document - 1)),
    };
    let pattern_cells = capacity.boundaries + capacity.boundaries + 1 + capacity.classes * states * states + states;
    SearchLoop { program, capacity, options, data_cells: pattern, searchable_cells: mask.map(|(_, searchable)| searchable), pattern_cells }
}
