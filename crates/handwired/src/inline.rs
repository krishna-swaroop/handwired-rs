// Copy one circuit's operations into another, e.g. a reusable layout circuit
// into an app's circuit. A port of inlineCircuit in src/runtime/numeric-ui-core.ts.

use std::collections::HashMap;

use crate::circuit::{AtomKind, Circuit, DeclarationOptions, Macro, Term, Wire};

/// Build every operation of `source` in `target`, reading `source_wires` for
/// the source's cells, and carry over its declarations. Returns the target
/// wire of every source wire. `options` applies to declarations that land on
/// target wires that operations already read.
pub fn inline_circuit(target: &mut Circuit, source: &Circuit, source_wires: &[Wire], options: DeclarationOptions) -> Vec<Wire> {
    assert!(source_wires.len() == source.source_size, "inlined source mapping has the wrong length");
    let mut wire_map = source_wires.to_vec();
    for wire in 0..source.source_size {
        let range = source.ranges[wire];
        target.assume_range_with(source_wires[wire], range.minimum, range.maximum, options);
        if let Some(grid) = source.grids[wire] {
            target.assume_grid_with(source_wires[wire], grid, options);
        }
        if source.exact_bits[wire] {
            target.assume_bit_with(source_wires[wire], options);
        }
    }
    // Map each distinct table once.
    let mut mapped_tables: HashMap<usize, Vec<Wire>> = HashMap::new();
    for (operation_index, operation) in source.operations.iter().enumerate() {
        let map_terms =
            |terms: &[Term], wire_map: &[Wire]| -> Vec<Term> { terms.iter().map(|&(wire, coefficient)| (wire_map[wire.index()], coefficient)).collect() };
        let wire = match (operation.macro_, operation.kind) {
            (Some(Macro::Multiply { left, right }), _) => target.multiply(wire_map[left.index()], wire_map[right.index()]),
            (Some(Macro::Select { gate, off, on }), _) => {
                target.select(wire_map[gate.index()], wire_map[off.index()], wire_map[on.index()])
            }
            (None, AtomKind::Reglu) => target.append(
                AtomKind::Reglu,
                map_terms(&operation.left_terms, &wire_map),
                operation.left_bias,
                map_terms(&operation.gate_terms, &wire_map),
                operation.gate_bias,
                operation.preserve_rounding,
                None,
            ),
            (None, AtomKind::Index) => {
                let read = operation.index_read.unwrap();
                let candidates = mapped_tables
                    .entry(read.table)
                    .or_insert_with(|| {
                        source.candidate_tables[read.table].candidates.iter().map(|candidate| wire_map[candidate.index()]).collect()
                    })
                    .clone();
                // The accepted index range carries over.
                target.index_read_affine(
                    &map_terms(&operation.gate_terms, &wire_map),
                    operation.gate_bias,
                    &candidates,
                    read.below,
                    read.above,
                    Some(read.index_range),
                )
            }
            (None, AtomKind::Boolean) if operation.preserve_rounding => {
                // emit_compare's atom on grid operands reads its score terms directly.
                target.append(AtomKind::Boolean, map_terms(&operation.left_terms, &wire_map), operation.left_bias, Vec::new(), 0.0, true, None)
            }
            (None, kind) => {
                let terms = map_terms(&operation.left_terms, &wire_map);
                let value = if operation.preserve_rounding {
                    target.round_affine(&terms, operation.left_bias)
                } else {
                    target.affine(&terms, operation.left_bias)
                };
                if kind == AtomKind::Boolean { target.boolean(value) } else { value }
            }
        };
        let source_wire = source.source_size + operation_index;
        let range = source.ranges[source_wire];
        target.assume_range_with(wire, range.minimum, range.maximum, options);
        if let Some(grid) = source.grids[source_wire] {
            target.assume_grid_with(wire, grid, options);
        }
        // A marked affine can fold into a source cell in the target, which has
        // no operation to record.
        if wire.index() >= target.source_size {
            if let Some(kind) = operation.rounding_point {
                target.record_rounding_point(wire, kind);
            }
            if operation.comparison_bit {
                let index = wire.index() - target.source_size;
                target.operations[index].comparison_bit = true;
            }
        }
        wire_map.push(wire);
    }
    // Declared one-hot groups survive inlining.
    for group in &source.one_hot_groups {
        let members: Vec<Wire> = group.members.iter().map(|member| wire_map[member.index()]).collect();
        target.assume_one_hot(&members, group.allow_all_zero, options);
    }
    wire_map
}
