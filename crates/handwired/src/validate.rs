// Checks for tests and error messages: a circuit's structure, a compiled
// program's layout, and the promises behind declarations against evaluated
// values. A port of the validate* functions in src/runtime/numeric-ui-core.ts.
// Each returns the TypeScript runtime's error message on failure.

use crate::circuit::{evaluate_terms, AtomKind, Circuit, Macro, Operation, Term, Wire};
use crate::compile::{CompileOptions, TensorProgram};
use crate::fp::js_number;

const INDEX_MAGNITUDE_LIMIT: f64 = 16_777_216.0;

fn check(condition: bool, message: &str) -> Result<(), String> {
    if condition { Ok(()) } else { Err(message.to_string()) }
}

fn validate_terms(terms: &[Term], wire_limit: usize) -> Result<(), String> {
    for &(wire, coefficient) in terms {
        check(wire.index() < wire_limit, "wire is outside the compiled circuit")?;
        check(coefficient.is_finite(), "weight must be finite")?;
    }
    Ok(())
}

/// Check that every operation reads earlier wires with finite weights, and
/// that index reads are well formed. Returns the wire count.
pub fn validate_circuit(circuit: &Circuit) -> Result<usize, String> {
    let wire_count = circuit.width();
    for (operation_index, operation) in circuit.operations.iter().enumerate() {
        check(operation.left_bias.is_finite(), "left bias must be finite")?;
        check(operation.gate_bias.is_finite(), "gate bias must be finite")?;
        validate_terms(&operation.left_terms, circuit.source_size + operation_index)?;
        validate_terms(&operation.gate_terms, circuit.source_size + operation_index)?;
        check(
            (operation.kind == AtomKind::Index) == operation.index_read.is_some(),
            "only index reads carry a candidate table",
        )?;
        let Some(read) = operation.index_read else { continue };
        check(
            operation.left_terms.is_empty() && operation.left_bias == 0.0,
            "an index read keeps its index form on the gate side",
        )?;
        check(!operation.gate_terms.is_empty(), "an index read needs a runtime index")?;
        let range = read.index_range;
        check(
            range.minimum > -INDEX_MAGNITUDE_LIMIT && range.minimum <= range.maximum && range.maximum < INDEX_MAGNITUDE_LIMIT,
            "an index read needs an accepted index range strictly inside ±2^24",
        )?;
        let candidates = &circuit.candidate_tables[read.table].candidates;
        check(!candidates.is_empty(), "an index read needs at least one candidate")?;
        let largest = candidates.iter().map(|candidate| candidate.index()).max().unwrap();
        check(largest < wire_count, "wire is outside the compiled circuit")?;
        check(largest < circuit.source_size + operation_index, "index read candidates must be earlier wires")?;
    }
    Ok(wire_count)
}

/// Check a compiled program's layout against its circuit.
pub fn validate_tensor_program(circuit: &Circuit, program: &TensorProgram) -> Result<(), String> {
    let width = program.width;
    check(program.source_size == circuit.source_size, "tensor source size mismatch")?;
    check(width == circuit.width(), "tensor width mismatch")?;
    check(program.value_count <= width, "tensor value count exceeds circuit width")?;
    check(program.wire_order.len() == program.value_count, "tensor wire order size mismatch")?;
    let mut seen_wires = vec![false; width];
    for wire in 0..circuit.source_size {
        check(program.wire_order[wire] as usize == wire, "tensor source order mismatch")?;
        seen_wires[wire] = true;
    }
    let mut resolved_slots = vec![false; program.gather_slot_count];
    let mut available = circuit.source_size;
    for layer in &program.layers {
        let output_size = layer.output_size;
        check(output_size > 0, "tensor layer must produce at least one value")?;
        check(layer.input_indices.len() == layer.input_size, "tensor input index size mismatch")?;
        for &input in &layer.input_indices {
            check((input as usize) < available, "tensor layer input is not available")?;
        }
        let gate_size = layer.gate_rows.len();
        let projection_size = output_size + gate_size + layer.index_reads.len();
        check(layer.bias.len() == projection_size, "projection bias shape mismatch")?;
        let mut gated = vec![false; output_size];
        for &row in &layer.gate_rows {
            check((row as usize) < output_size, "gate output row is invalid")?;
            check(!gated[row as usize], "gate output row must be unique")?;
            gated[row as usize] = true;
        }
        let mut boolean = vec![false; output_size];
        for &row in &layer.boolean_rows {
            check((row as usize) < output_size, "boolean output row is invalid")?;
            check(!gated[row as usize], "boolean output row cannot also be gated")?;
            check(!boolean[row as usize], "boolean output row must be unique")?;
            boolean[row as usize] = true;
        }
        let offsets = &layer.row_offsets;
        check(offsets.len() == projection_size + 1, "sparse row offset shape mismatch")?;
        check(offsets[0] == 0, "sparse row offsets must start at zero")?;
        check(layer.columns.len() == layer.weights.len(), "sparse column shape mismatch")?;
        check(
            offsets[projection_size] as usize == layer.weights.len(),
            "sparse row offsets must end at the stored weight count",
        )?;
        for row in 0..projection_size {
            check(offsets[row] <= offsets[row + 1], "sparse row offsets must be ordered")?;
        }
        for &column in &layer.columns {
            check((column as usize) < layer.input_size, "sparse matrix column is invalid")?;
        }
        for compute in &layer.gather_computes {
            check(compute.slot < program.gather_slot_count, "gather slot is invalid")?;
            check(!resolved_slots[compute.slot], "gather slot must be resolved exactly once")?;
            resolved_slots[compute.slot] = true;
            for &position in &compute.bit_positions {
                check((position as usize) < available, "gather bit is not available")?;
            }
        }
        let mut read_rows = vec![false; output_size];
        for read in &layer.gather_reads {
            check(read.row < output_size, "gather output row is invalid")?;
            check(!gated[read.row], "gather output row cannot be gated")?;
            check(!boolean[read.row], "gather output row cannot be boolean")?;
            check(offsets[read.row] == offsets[read.row + 1], "gather output row must store no weights")?;
            check(
                read.slot < program.gather_slot_count && resolved_slots[read.slot],
                "gather read must follow its slot resolution",
            )?;
            check(!read.candidate_positions.is_empty(), "gather read needs at least one candidate")?;
            for &position in &read.candidate_positions {
                check((position as usize) < available, "gather candidate is not available")?;
            }
            read_rows[read.row] = true;
        }
        let mut index_rows = vec![false; projection_size];
        for read in &layer.index_reads {
            check(read.row < output_size, "index read output row is invalid")?;
            check(!gated[read.row], "index read output row cannot be gated")?;
            check(!boolean[read.row], "index read output row cannot be boolean")?;
            check(!read_rows[read.row], "index read output row must be unique")?;
            read_rows[read.row] = true;
            check(
                offsets[read.row] == offsets[read.row + 1] && layer.bias[read.row] == 0.0,
                "index read output row must store no weights and no bias",
            )?;
            check(
                read.index_row >= output_size + gate_size && read.index_row < projection_size,
                "index row must follow the gate rows",
            )?;
            check(!index_rows[read.index_row], "index row must be unique")?;
            index_rows[read.index_row] = true;
            check(offsets[read.index_row] < offsets[read.index_row + 1], "index row must read a runtime index")?;
            check(read.candidate_count > 0, "index read needs at least one candidate")?;
            if read.candidate_positions.is_empty() {
                // Evenly spaced positions lie between the first and the last.
                let last = read.candidate_base + (read.candidate_count as i64 - 1) * read.candidate_stride;
                check(
                    read.candidate_base >= 0
                        && (read.candidate_base as usize) < available
                        && last >= 0
                        && (last as usize) < available,
                    "evenly spaced index read candidates are not available",
                )?;
            } else {
                check(read.candidate_positions.len() == read.candidate_count, "index read candidate table size mismatch")?;
                for &position in &read.candidate_positions {
                    check((position as usize) < available, "index read candidate is not available")?;
                }
            }
        }
        check(layer.bias.iter().all(|value| value.is_finite()), "projection bias contains a non-finite number")?;
        check(layer.weights.iter().all(|value| value.is_finite()), "projection matrix contains a non-finite number")?;
        for row in 0..output_size {
            let wire = program.wire_order[available + row] as usize;
            check(wire >= circuit.source_size && wire < width, "tensor output wire is invalid")?;
            check(!seen_wires[wire], "tensor wire must be written exactly once")?;
            seen_wires[wire] = true;
        }
        available += output_size;
    }
    check(available == program.value_count, "tensor program output size mismatch")?;
    check(resolved_slots.iter().all(|&resolved| resolved), "every gather slot must be resolved")?;
    for &root in &program.roots {
        check(seen_wires[root as usize], "every tensor root must be present")?;
    }
    Ok(())
}

/// Check every declared one-hot group against evaluated values.
pub fn validate_one_hot_groups(circuit: &Circuit, values: &[f32]) -> Result<(), String> {
    check(values.len() == circuit.width(), "one-hot validation needs the full wire values")?;
    for group in &circuit.one_hot_groups {
        let mut mass = 0f64;
        for member in &group.members {
            let value = values[member.index()] as f64;
            if !(value == 0.0 || value == 1.0) {
                return Err(format!(
                    "one-hot member wire {} must be exactly 0 or 1, got {}",
                    member.index(),
                    js_number(value)
                ));
            }
            mass += value;
        }
        if !(mass == 1.0 || (group.allow_all_zero && mass == 0.0)) {
            return Err(format!("one-hot group must have exactly one active member, got mass {}", js_number(mass)));
        }
    }
    Ok(())
}

/// Check that every index read's index form holds an integer within the range
/// the builder accepted.
pub fn validate_index_reads(circuit: &Circuit, values: &[f32]) -> Result<(), String> {
    check(values.len() == circuit.width(), "index read validation needs the full wire values")?;
    for (operation_index, operation) in circuit.operations.iter().enumerate() {
        if operation.kind != AtomKind::Index {
            continue;
        }
        let index = evaluate_terms(values, &operation.gate_terms, operation.gate_bias);
        let range = operation.index_read.unwrap().index_range;
        if !(index.is_finite() && index.fract() == 0.0 && index >= range.minimum && index <= range.maximum) {
            return Err(format!(
                "index read wire {} needs an integer index within {}..{}, got {}",
                circuit.source_size + operation_index,
                js_number(range.minimum),
                js_number(range.maximum),
                js_number(index),
            ));
        }
    }
    Ok(())
}

/// Check that every wire with a declared or inferred grid holds a finite
/// multiple of it.
pub fn validate_grids(circuit: &Circuit, values: &[f32]) -> Result<(), String> {
    check(values.len() == circuit.width(), "grid validation needs the full wire values")?;
    for (wire, &value) in values.iter().enumerate() {
        let Some(grid) = circuit.grids[wire] else { continue };
        let value = value as f64;
        let quotient = value / grid;
        if !(value.is_finite() && quotient.is_finite() && quotient.fract() == 0.0) {
            return Err(format!("wire {wire} must be a multiple of {}, got {}", js_number(grid), js_number(value)));
        }
    }
    Ok(())
}

/// Check that no evaluated wire is NaN. With `roots`, only the roots and the
/// wires they read are checked.
pub fn validate_no_nan(circuit: &Circuit, values: &[f32], roots: Option<&[Wire]>) -> Result<(), String> {
    match describe_first_nan(circuit, values, roots) {
        Some(message) => Err(message),
        None => Ok(()),
    }
}

/// The operation that produced the first NaN, and the multiply or select that
/// contains it, or None when no checked wire is NaN.
pub fn describe_first_nan(circuit: &Circuit, values: &[f32], roots: Option<&[Wire]>) -> Option<String> {
    assert!(values.len() == circuit.width(), "NaN validation needs the full wire values");
    let checked = roots.map(|roots| mark_wires_read(circuit, roots));
    let is_checked = |wire: usize| checked.as_ref().is_none_or(|checked| checked[wire]);
    for wire in 0..circuit.source_size {
        if values[wire].is_nan() && is_checked(wire) {
            return Some(format!("source cell {wire} is NaN"));
        }
    }
    let value = |wire: Wire| js_number(values[wire.index()] as f64);
    for (operation_index, operation) in circuit.operations.iter().enumerate() {
        let wire = circuit.source_size + operation_index;
        if !values[wire].is_nan() || !is_checked(wire) {
            continue;
        }
        let describe_terms =
            |terms: &[Term]| terms.iter().map(|&(input, _)| format!("wire {} = {}", input.index(), value(input))).collect::<Vec<_>>().join(", ");
        let description = match operation.kind {
            AtomKind::Reglu => format!(
                "f(left × max(0, gate)) with left = {} and gate = {}",
                js_number(evaluate_terms(values, &operation.left_terms, operation.left_bias)),
                js_number(evaluate_terms(values, &operation.gate_terms, operation.gate_bias)),
            ),
            AtomKind::Index => format!("an index read whose index form is NaN, from {}", describe_terms(&operation.gate_terms)),
            _ => format!("a sum of {}", describe_terms(&operation.left_terms)),
        };
        let mut context = String::new();
        // The multiply and select that contain the operation: their result reads
        // it and none of their operands does.
        let mut reading = mark_wires_reading(circuit, wire);
        for reader in operation_index..circuit.operations.len() {
            let Some(macro_) = circuit.operations[reader].macro_ else { continue };
            if !reading[circuit.source_size + reader] {
                continue;
            }
            let operands: Vec<Wire> = match macro_ {
                Macro::Multiply { left, right } => vec![left, right],
                Macro::Select { gate, off, on } => vec![gate, off, on],
            };
            if operands.iter().any(|operand| reading[operand.index()]) {
                continue;
            }
            match macro_ {
                Macro::Multiply { left, right } => {
                    context += &format!(
                        ", in multiply(wire {} = {}, wire {} = {})",
                        left.index(),
                        value(left),
                        right.index(),
                        value(right)
                    );
                    reading = mark_wires_reading(circuit, circuit.source_size + reader);
                }
                Macro::Select { gate, off, on } => {
                    context += &format!(
                        ", in select(gate wire {} = {}, off wire {} = {}, on wire {} = {}). Both sides of a select always \
                         compute, so a side the gate doesn't pick must stay finite",
                        gate.index(),
                        value(gate),
                        off.index(),
                        value(off),
                        on.index(),
                        value(on),
                    );
                    break;
                }
            }
        }
        return Some(format!("wire {wire} is the first NaN: {description}{context}"));
    }
    None
}

// Marks the roots, and every wire they read directly or through other operations.
fn mark_wires_read(circuit: &Circuit, roots: &[Wire]) -> Vec<bool> {
    let mut read = vec![false; circuit.width()];
    for root in roots {
        assert!(root.index() < read.len(), "wire is outside the compiled circuit");
        read[root.index()] = true;
    }
    for operation_index in (0..circuit.operations.len()).rev() {
        if read[circuit.source_size + operation_index] {
            for_each_input(circuit, &circuit.operations[operation_index], |input| read[input.index()] = true);
        }
    }
    read
}

// Marks `wire`, and every operation that reads it directly or through other operations.
fn mark_wires_reading(circuit: &Circuit, wire: usize) -> Vec<bool> {
    let mut reading = vec![false; circuit.width()];
    reading[wire] = true;
    let first = (wire + 1).saturating_sub(circuit.source_size);
    for operation_index in first..circuit.operations.len() {
        let mut reads = false;
        for_each_input(circuit, &circuit.operations[operation_index], |input| reads |= reading[input.index()]);
        if reads {
            reading[circuit.source_size + operation_index] = true;
        }
    }
    reading
}

// Visits every wire an operation reads: its terms, and an index read's candidates.
fn for_each_input(circuit: &Circuit, operation: &Operation, mut visit: impl FnMut(Wire)) {
    for &(input, _) in operation.left_terms.iter().chain(&operation.gate_terms) {
        visit(input);
    }
    if let Some(read) = &operation.index_read {
        for &candidate in &circuit.candidate_tables[read.table].candidates {
            visit(candidate);
        }
    }
}

/// Check a compiled evaluation against the reference evaluator on every wire
/// the default program computes.
pub fn validate_tensor_evaluation(source: &[f32], circuit: &Circuit, tensor_values: &[f32]) -> Result<(), String> {
    let expected = circuit.evaluate_reference(source);
    check(expected.len() == tensor_values.len(), "tensor result width mismatch")?;
    let program = circuit.compile_tensor_program(None, true, CompileOptions::default());
    for &wire in &program.wire_order {
        let wire = wire as usize;
        // JavaScript's ===: NaN never matches, and -0 matches +0.
        if tensor_values[wire] != expected[wire] {
            return Err(format!("tensor wire {wire} does not match the reference FP32 atom program"));
        }
    }
    Ok(())
}
