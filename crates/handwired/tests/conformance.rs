// Builds every scenario of conformance/generate.ts with the Rust port and
// checks it against what the TypeScript runtime produced, bit for bit: every
// operation, range and grid, the reference evaluator's values, and its
// gradients. Regenerate the fixtures with
//
//   node crates/handwired/conformance/generate.ts

use handwired::*;
use serde_json::Value;

fn compare_all(circuit: &mut Circuit, threshold: f64) {
    let a = circuit.source(0);
    let b = circuit.source(1);
    let constant = circuit.constant(threshold);
    for end_with_boolean in [false, true] {
        let options = CompareOptions { end_with_boolean };
        for relation in [Relation::Greater, Relation::AtLeast, Relation::Less, Relation::AtMost, Relation::Equal] {
            emit_compare(circuit, a, relation, b, options);
            emit_compare(circuit, a, relation, constant, options);
            emit_compare(circuit, constant, relation, b, options);
            emit_compare(circuit, a, relation, a, options);
        }
    }
}

fn compare_scenario(circuit: &mut Circuit, minimum: f64, maximum: f64, grids: [Option<f64>; 2], threshold: f64) {
    for (index, grid) in grids.into_iter().enumerate() {
        let wire = circuit.source(index);
        circuit.assume_range(wire, minimum, maximum);
        if let Some(grid) = grid {
            circuit.assume_grid(wire, grid);
        }
    }
    compare_all(circuit, threshold);
}

fn sources(circuit: &Circuit) -> Vec<Wire> {
    (0..circuit.source_size).map(|index| circuit.source(index)).collect()
}

fn one_hot_cells(c: &Circuit, slot_count: usize, extra_count: usize) -> (Vec<Wire>, Vec<Wire>, Vec<Wire>) {
    let cells = sources(c);
    (
        cells[..slot_count].to_vec(),
        cells[slot_count..slot_count * 2].to_vec(),
        cells[slot_count * 2..slot_count * 2 + extra_count].to_vec(),
    )
}

// The gather_* scenarios: slot_count one-hot bits and slot_count values,
// declared before or after the products, and the reads made from them.
fn gather_scenario(c: &mut Circuit, name: &str) -> Option<Vec<Wire>> {
    let unbounded = (f64::NEG_INFINITY, f64::INFINITY);
    let (slot_count, value_range, declare_bits_first, extra_count, extra_range) = match name {
        "gather_basic" => (4, (0.0, 4096.0), true, 0, (0.0, 100.0)),
        "gather_signed_pair" => (6, (-4096.0, 4096.0), false, 0, (0.0, 100.0)),
        "gather_bit_in_gate" => (5, (-1e6, 1e6), true, 0, (0.0, 100.0)),
        "gather_not_pure" => (4, (0.0, 4096.0), true, 1, (0.0, 4096.0)),
        "gather_unbounded_left" => (2, (0.0, f64::INFINITY), true, 0, (0.0, 100.0)),
        "gather_unbounded_gate" => (2, unbounded, true, 0, (0.0, 100.0)),
        "gather_unbounded_pair" => (2, unbounded, false, 0, (0.0, 100.0)),
        "gather_filler_finite" => (2, (0.0, 100.0), true, 1, (0.0, 100.0)),
        "gather_filler_unbounded" => (2, (0.0, 100.0), true, 1, (0.0, f64::INFINITY)),
        "gather_unused_negative_arm" => (3, (-100.0, 100.0), false, 0, (0.0, 100.0)),
        _ => return None,
    };
    let (bits, values, extras) = one_hot_cells(c, slot_count, extra_count);
    if declare_bits_first {
        c.assume_one_hot(&bits, false, DeclarationOptions::default());
    }
    for &value in &values {
        c.assume_range(value, value_range.0, value_range.1);
    }
    for &extra in &extras {
        c.assume_range(extra, extra_range.0, extra_range.1);
    }
    let products: Vec<Term> = bits.iter().zip(&values).map(|(&bit, &value)| (emit_multiply(c, bit, value), 1.0)).collect();
    let roots = match name {
        "gather_basic" => vec![c.affine(&products, 0.1), c.affine(&products, 0.0)],
        "gather_not_pure" => {
            let scaled: Vec<Term> =
                products.iter().enumerate().map(|(slot, &(wire, _))| (wire, if slot == 1 { 2.0 } else { 1.0 })).collect();
            let scaled_read = c.affine(&scaled, 0.0);
            let missing_read = c.affine(&products[..products.len() - 1], 0.0);
            let shifted = c.affine(&[(extras[0], 1.0)], 5.0);
            let extra_read = c.affine(&[products.clone(), vec![(shifted, 1.0)]].concat(), 0.0);
            vec![scaled_read, missing_read, extra_read]
        }
        "gather_filler_finite" | "gather_filler_unbounded" => {
            let first_bit = c.source(0);
            let filler = c.reglu(&[(extras[0], 1.0)], 0.0, &[(first_bit, -1.0)], 0.0);
            vec![c.affine(&[products.clone(), vec![(filler, 1.0)]].concat(), 0.0)]
        }
        "gather_unused_negative_arm" => {
            let negative_arm = Wire::from_position(products[0].0.index() - 1);
            vec![c.affine(&[products.clone(), vec![(negative_arm, -1.0)]].concat(), 0.0)]
        }
        _ => vec![c.affine(&products, 0.0)],
    };
    if !declare_bits_first {
        c.assume_one_hot(&bits, false, DeclarationOptions { allow_existing_readers: true });
    }
    Some(roots)
}


// The scenario's circuit and the roots of its roots-only program, the last
// operation by default.
fn build(name: &str, source_size: usize) -> (Circuit, Vec<Wire>) {
    let mut circuit = Circuit::new(source_size);
    let c = &mut circuit;
    let mut roots = Vec::new();
    match name {
        "readme" => {
            let a = c.source(0);
            c.assume_range(a, -100.0, 100.0);
            c.assume_grid(a, 1.0);
            let four = c.constant(4.0);
            let above4 = emit_compare(c, a, Relation::Greater, four, CompareOptions::default());
            let twelve = c.constant(12.0);
            let ten = c.constant(10.0);
            emit_select(c, above4, twelve, ten);
        }
        "compare_grid_integer" => compare_scenario(c, -1000.0, 1000.0, [Some(1.0), Some(1.0)], 3.0),
        "compare_grid_fine" => compare_scenario(c, 0.0, 4096.0, [Some(1.0 / 64.0), Some(1.0 / 64.0)], 3.25),
        "compare_grid_mixed" => compare_scenario(c, -64.0, 64.0, [Some(0.25), Some(1.0)], 3.25),
        "compare_grid_too_wide" => compare_scenario(c, -(2f64.powi(25)), 2f64.powi(25), [Some(1.0), Some(1.0)], 3.0),
        "compare_cascade" => compare_scenario(c, -16384.0, 16384.0, [None, None], 0.1),
        "compare_cascade_wide" => compare_scenario(c, -(2f64.powi(100)), 2f64.powi(100), [None, None], 0.1),
        "compare_cascade_tiny" => compare_scenario(c, -(2f64.powi(-100)), 2f64.powi(-100), [None, None], 2f64.powi(-120)),
        "minmax" => {
            let [a, b, cc, d, u, v] = sources(c)[..] else { unreachable!() };
            c.assume_range(a, 0.0, 1000.0);
            c.assume_range(b, 0.0, 1000.0);
            c.assume_range(cc, -50.0, 50.0);
            c.assume_range(d, -50.0, 50.0);
            c.assume_range(u, 0.0, 1.0);
            c.assume_range(v, 0.0, 1.0);
            emit_minimum(c, a, b);
            emit_maximum(c, a, b);
            emit_minimum(c, cc, d);
            emit_maximum(c, cc, d);
            emit_maximum(c, u, v);
            emit_minimum(c, u, v);
            emit_minimum_exact(c, a, b);
            emit_maximum_exact(c, cc, d);
            let five_hundred = c.constant(500.0);
            emit_clamp_exact(c, a, b, five_hundred);
            let ten = c.constant(10.0);
            emit_clamp_exact(c, cc, d, ten);
        }
        "multiply_select" => {
            let [x, y, g, z] = sources(c)[..] else { unreachable!() };
            c.assume_range(x, -10.0, 10.0);
            c.assume_range(y, 0.0, 5.0);
            c.assume_bit(g);
            emit_multiply(c, x, y);
            emit_multiply(c, y, x);
            emit_multiply(c, x, x);
            emit_multiply(c, x, z);
            emit_select(c, g, x, y);
            emit_select(c, g, x, z);
            emit_relu(c, x);
        }
        "index_read" => {
            let index = c.source(0);
            c.assume_range(index, -3.0, 12.0);
            let table: Vec<Wire> = (1..=5).map(|cell| c.source(cell)).collect();
            c.index_read(index, &table, OutOfRange::Clamp);
            c.index_read(index, &table, OutOfRange::Zero);
            c.index_read(index, &table.clone(), OutOfRange::Zero);
            c.index_read_affine(&[(index, 1.0)], 2.0, &table, OutOfRange::Clamp, OutOfRange::Zero, None);
            let two = c.affine(&[], 2.0);
            c.index_read(two, &table, OutOfRange::Clamp);
            let minus_seven = c.affine(&[], -7.0);
            c.index_read(minus_seven, &table, OutOfRange::Zero);
            let shifted = c.affine(&[(index, 1.0)], -1.0);
            c.index_read(shifted, &[table[4], table[0], table[2]], OutOfRange::Clamp);
        }
        "compose" => {
            let [a, b, cc, d, e] = sources(c)[..] else { unreachable!() };
            for wire in [a, b, cc, d] {
                c.assume_range(wire, -100.0, 100.0);
            }
            c.assume_range(e, 0.0, 1.0);
            let t1 = c.affine(&[(a, 2.0), (b, 3.0)], 1.0);
            let t2 = c.affine(&[(t1, 0.5), (a, -1.0)], 0.0);
            let k = c.constant(3.0);
            c.affine(&[(k, 2.0), (a, 1.0)], 0.0);
            c.affine(&[(a, 1.0)], 0.0);
            c.affine(&[(b, 3.0), (a, 2.0)], 1.0);
            let wide = c.affine(&[(a, 1.0), (b, 1.0), (cc, 1.0), (d, 1.0)], 0.0);
            c.affine(&[(wide, 2.0), (a, 1.0)], 0.0);
            c.reglu(&[(a, 1.0)], 0.0, &[], 2.0);
            c.reglu(&[], 0.0, &[(a, 1.0)], 0.0);
            c.reglu(&[(t1, 1.0)], 0.5, &[(t2, 1.0), (b, -1.0)], 0.25);
            c.boolean(e);
            let half_e = c.affine(&[(e, 0.5)], 0.25);
            c.boolean(half_e);
            let rounded = c.round_affine(&[(a, 1.0), (a, 1.0), (k, 1.0)], 0.5);
            c.mark_rounding_point(rounded);
            c.affine(&[(rounded, 0.1), (cc, 1.0 / 3.0), (d, -1e-7)], 1e10);
        }
        name if name.starts_with("gather_") && !matches!(name, "gather_two_groups" | "gather_chain" | "gather_bit_in_left_signed") => {
            roots = gather_scenario(c, name).expect("a gather scenario");
        }
        "gather_bit_in_left_signed" => {
            let (bits, values, _) = one_hot_cells(c, 2, 0);
            c.assume_one_hot(&bits, false, DeclarationOptions::default());
            for &value in &values {
                c.assume_range(value, -100.0, 100.0);
            }
            let products: Vec<Term> =
                bits.iter().zip(&values).map(|(&bit, &value)| (c.reglu(&[(bit, 1.0)], 0.0, &[(value, 1.0)], 0.0), 1.0)).collect();
            roots = vec![c.affine(&products, 0.0)];
        }
        "gather_two_groups" => {
            let cells = sources(c);
            c.assume_one_hot(&cells[0..2], false, DeclarationOptions::default());
            c.assume_one_hot(&cells[2..4], false, DeclarationOptions::default());
            for &value in &cells[4..8] {
                c.assume_range(value, 0.0, 100.0);
            }
            let products: Vec<Term> = (0..4).map(|bit| (emit_multiply(c, cells[bit], cells[4 + bit]), 1.0)).collect();
            roots = vec![c.affine(&products, 0.0)];
        }
        "gather_chain" => {
            let cells = sources(c);
            let (first, second, values) = (&cells[0..3], &cells[3..6], &cells[6..10]);
            c.assume_one_hot(first, false, DeclarationOptions::default());
            c.assume_one_hot(second, false, DeclarationOptions::default());
            for &value in values {
                c.assume_range(value, -50.0, 50.0);
            }
            let inner_products: Vec<Term> =
                first.iter().zip(values).map(|(&bit, &value)| (emit_multiply(c, bit, value), 1.0)).collect();
            let inner = c.affine(&inner_products, 0.0);
            let outer_values = [inner, values[3], values[0]];
            let outer_products: Vec<Term> =
                second.iter().zip(outer_values).map(|(&bit, value)| (emit_multiply(c, bit, value), 1.0)).collect();
            let outer = c.affine(&outer_products, 0.0);
            roots = vec![outer, emit_minimum(c, outer, values[1])];
        }
        "argmax" => {
            let cells = sources(c);
            let (scores, values, unit) = (&cells[0..4], &cells[4..8], cells[8]);
            for &score in scores {
                c.assume_range(score, 0.0, 1.0);
            }
            for &value in values {
                c.assume_range(value, -50.0, 50.0);
            }
            c.assume_range(unit, 1.0, 1.0);
            let argmax = emit_argmax(c, scores, unit);
            let products: Vec<Term> =
                argmax.selected.iter().zip(values).map(|(&bit, &value)| (emit_multiply(c, bit, value), 1.0)).collect();
            let read = c.affine(&products, 0.0);
            roots = [vec![read, argmax.score], argmax.selected].concat();
        }
        "first_active_bits" => {
            let cells = sources(c);
            let (xs, values, gate) = (&cells[0..4], &cells[4..8], cells[8]);
            for &x in xs.iter().chain([&gate]) {
                c.assume_range(x, 0.0, 10.0);
                c.assume_grid(x, 1.0);
            }
            for &value in values {
                c.assume_range(value, 0.0, 100.0);
            }
            let five = c.constant(5.0);
            let candidates: Vec<Wire> =
                xs.iter().map(|&x| emit_compare(c, x, Relation::AtLeast, five, CompareOptions::default())).collect();
            let zero = c.constant(0.0);
            let active = emit_compare(c, gate, Relation::Greater, zero, CompareOptions::default());
            let selected = emit_first_active(c, active, &candidates);
            let products: Vec<Term> =
                selected.iter().zip(values).map(|(&bit, &value)| (emit_multiply(c, bit, value), 1.0)).collect();
            let read = c.affine(&products, 0.0);
            roots = [vec![read], selected].concat();
        }
        "first_active_fractional" => {
            let cells = sources(c);
            let (candidates, values, active) = (&cells[0..4], &cells[4..8], cells[8]);
            for &wire in candidates.iter().chain([&active]) {
                c.assume_range(wire, 0.0, 1.0);
            }
            for &value in values {
                c.assume_range(value, 0.0, 100.0);
            }
            let selected = emit_first_active(c, active, candidates);
            let products: Vec<Term> =
                selected.iter().zip(values).map(|(&bit, &value)| (emit_multiply(c, bit, value), 1.0)).collect();
            roots = vec![c.affine(&products, 0.0)];
        }
        "first_active_shared_constant" => {
            let unit = c.source(0);
            c.assume_range(unit, 1.0, 1.0);
            let zero = c.constant(0.0);
            roots = emit_first_active(c, unit, &[zero, zero]);
        }
        "spring" => {
            let cells = sources(c);
            let ranges = [
                (-1000.0, 1000.0), (-1000.0, 1000.0), (-5000.0, 5000.0), (0.0, 1000.0), (0.0, 100.0), (0.0, 0.05),
                (0.0, 1.0), (0.0, 1.0), (0.0, 10000.0), (0.0, 10000.0), (1.0, 1.0),
            ];
            for (&cell, (minimum, maximum)) in cells.iter().zip(ranges) {
                c.assume_range(cell, minimum, maximum);
            }
            let spring = emit_spring(c, SpringWires {
                position: cells[0],
                destination: cells[1],
                velocity: cells[2],
                stiffness: cells[3],
                damping: cells[4],
                timestep: cells[5],
                settle_distance: cells[6],
                settle_speed: cells[7],
                settle_distance_scale: cells[8],
                settle_speed_scale: cells[9],
                unit: cells[10],
            });
            roots = vec![spring.position, spring.velocity, spring.active];
        }
        "animated_object" => {
            let cells = sources(c);
            for &cell in &cells[0..8] {
                c.assume_range(cell, -2.0, 2.0);
            }
            for &cell in &cells[8..15] {
                c.assume_range(cell, -1000.0, 1000.0);
            }
            c.assume_bit(cells[15]);
            c.assume_bit(cells[16]);
            let object = emit_animated_object(
                c,
                &[
                    AnimatedSpring {
                        matrix: [cells[0], cells[1], cells[2], cells[3]],
                        displacement: cells[8],
                        velocity: cells[9],
                        destination: cells[12],
                        destination_during_steps: Some(cells[14]),
                    },
                    AnimatedSpring {
                        matrix: [cells[4], cells[5], cells[6], cells[7]],
                        displacement: cells[10],
                        velocity: cells[11],
                        destination: cells[13],
                        destination_during_steps: None,
                    },
                ],
                AnimatedObjectOptions {
                    settle_check: cells[15],
                    snap: Some(cells[16]),
                    settle_distance: Some(0.01),
                    settle_speed: Some(0.02),
                },
            );
            roots = [object.positions, vec![object.settled, object.active]].concat();
        }
        "inline" => {
            let mut part = Circuit::new(3);
            let [a, b, index] = sources(&part)[..] else { unreachable!() };
            part.assume_range(a, 0.0, 100.0);
            part.assume_range(b, 0.0, 100.0);
            part.assume_grid(a, 1.0);
            part.assume_grid(b, 1.0);
            part.assume_range(index, 0.0, 2.0);
            let greater = emit_compare(&mut part, a, Relation::Greater, b, CompareOptions::default());
            let picked = emit_select(&mut part, greater, b, a);
            let smaller = emit_minimum_exact(&mut part, a, b);
            let read = part.index_read(index, &[a, b, picked], OutOfRange::Zero);
            let rounded = part.round_affine(&[(read, 1.0), (smaller, 1.0)], 0.5);
            part.mark_rounding_point(rounded);
            let cells = sources(c);
            let first = inline_circuit(c, &part, &[cells[0], cells[1], cells[2]], DeclarationOptions::default());
            let second = inline_circuit(c, &part, &[cells[1], cells[0], cells[3]], DeclarationOptions::default());
            roots = vec![*first.last().unwrap(), *second.last().unwrap()];
        }
        _ => panic!("no Rust builder for scenario {name}"),
    }
    if roots.is_empty() {
        roots.push(Wire::from_position(circuit.width() - 1));
    }
    (circuit, roots)
}

fn hex(value: &Value) -> u64 {
    u64::from_str_radix(value.as_str().unwrap(), 16).unwrap()
}

fn bits32(value: &Value) -> Vec<u32> {
    value.as_array().unwrap().iter().map(|bits| bits.as_u64().unwrap() as u32).collect()
}

fn f32s(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&bits| f32::from_bits(bits)).collect()
}

// Bit equality, except that every NaN matches every NaN.
fn same_f32(left: u32, right: u32) -> bool {
    left == right || (f32::from_bits(left).is_nan() && f32::from_bits(right).is_nan())
}

fn same_f64(left: f64, right: u64) -> bool {
    left.to_bits() == right || (left.is_nan() && f64::from_bits(right).is_nan())
}

fn terms_match(terms: &[Term], expected: &Value) -> bool {
    let expected = expected.as_array().unwrap();
    terms.len() == expected.len()
        && terms.iter().zip(expected).all(|(&(wire, coefficient), term)| {
            wire.index() as u64 == term[0].as_u64().unwrap() && same_f64(coefficient, hex(&term[1]))
        })
}

fn kind_name(kind: AtomKind) -> &'static str {
    match kind {
        AtomKind::Affine => "affine",
        AtomKind::Reglu => "reglu",
        AtomKind::Boolean => "boolean",
        AtomKind::Index => "index",
    }
}

fn rounding_point_name(kind: Option<RoundingPointKind>) -> Value {
    match kind {
        None => Value::Null,
        Some(RoundingPointKind::Boolean) => "boolean".into(),
        Some(RoundingPointKind::IntegerBit) => "integer bit".into(),
        Some(RoundingPointKind::SignCascade) => "sign cascade".into(),
        Some(RoundingPointKind::Marked) => "marked".into(),
    }
}

fn out_of_range_name(rule: OutOfRange) -> &'static str {
    match rule {
        OutOfRange::Clamp => "clamp",
        OutOfRange::Zero => "zero",
    }
}

fn check_structure(name: &str, circuit: &Circuit, expected: &Value) {
    let operations = expected["operations"].as_array().unwrap();
    assert_eq!(circuit.operations.len(), operations.len(), "{name}: operation count");
    for (index, (operation, expected)) in circuit.operations.iter().zip(operations).enumerate() {
        let wire = circuit.source_size + index;
        let context = format!("{name}: operation {index} (wire {wire}): {operation:?}\nexpected {expected}");
        assert_eq!(kind_name(operation.kind), expected["kind"], "{context}");
        assert!(terms_match(&operation.left_terms, &expected["leftTerms"]), "left terms differ: {context}");
        assert!(same_f64(operation.left_bias, hex(&expected["leftBias"])), "left bias differs: {context}");
        assert!(terms_match(&operation.gate_terms, &expected["gateTerms"]), "gate terms differ: {context}");
        assert!(same_f64(operation.gate_bias, hex(&expected["gateBias"])), "gate bias differs: {context}");
        assert_eq!(operation.depth as u64, expected["depth"].as_u64().unwrap(), "depth differs: {context}");
        assert_eq!(operation.preserve_rounding, expected["preserveRounding"], "{context}");
        assert_eq!(rounding_point_name(operation.rounding_point), expected["roundingPoint"], "{context}");
        assert_eq!(operation.comparison_bit, expected["comparisonBit"], "{context}");
        let macro_ = match operation.macro_ {
            None => Value::Null,
            Some(Macro::Multiply { left, right }) => serde_json::json!(["multiply", left.index(), right.index()]),
            Some(Macro::Select { gate, off, on }) => {
                serde_json::json!(["select", gate.index(), off.index(), on.index()])
            }
        };
        assert_eq!(macro_, expected["macro"], "{context}");
        let read = operation.index_read.map_or(Value::Null, |read| {
            serde_json::json!({
                "table": read.table,
                "below": out_of_range_name(read.below),
                "above": out_of_range_name(read.above),
                "indexRange": [f64_hex(read.index_range.minimum), f64_hex(read.index_range.maximum)],
            })
        });
        assert_eq!(read, expected["indexRead"], "{context}");
    }
    for (wire, (range, expected)) in circuit.ranges.iter().zip(expected["ranges"].as_array().unwrap()).enumerate() {
        assert!(
            same_f64(range.minimum, hex(&expected[0])) && same_f64(range.maximum, hex(&expected[1])),
            "{name}: range of wire {wire} is {range:?}, expected {:?}..{:?}",
            f64::from_bits(hex(&expected[0])),
            f64::from_bits(hex(&expected[1])),
        );
    }
    for (wire, (grid, expected)) in circuit.grids.iter().zip(expected["grids"].as_array().unwrap()).enumerate() {
        let matches = match (grid, expected.is_null()) {
            (None, true) => true,
            (Some(grid), false) => same_f64(*grid, hex(expected)),
            _ => false,
        };
        assert!(matches, "{name}: grid of wire {wire} is {grid:?}, expected {expected}");
    }
}

fn check_evaluations(name: &str, circuit: &Circuit, expected: &Value) -> usize {
    let evaluations = expected["evaluations"].as_array().unwrap();
    for evaluation in evaluations {
        let source = f32s(&bits32(&evaluation["source"]));
        let values = circuit.evaluate_reference(&source);
        let expected_values = bits32(&evaluation["values"]);
        assert_eq!(values.len(), expected_values.len(), "{name}: value count");
        for (wire, (value, &expected)) in values.iter().zip(&expected_values).enumerate() {
            assert!(
                same_f32(value.to_bits(), expected),
                "{name}: wire {wire} on source {source:?} is {value:?}, expected {:?}",
                f32::from_bits(expected),
            );
        }
    }
    evaluations.len()
}

fn check_gradients(name: &str, circuit: &Circuit, expected: &Value) -> usize {
    let cases = expected["gradients"].as_array().unwrap();
    for case in cases {
        let source = f32s(&bits32(&case["source"]));
        let values = circuit.evaluate_reference(&source);
        let output_gradients = f32s(&bits32(&case["outputGradients"]));
        let options = BackpropagateOptions {
            boolean_gradient: match case["booleanGradient"].as_str().unwrap() {
                "straight-through" => BooleanGradient::StraightThrough,
                _ => BooleanGradient::Reject,
            },
            index_gradient: match case["indexGradient"].as_str().unwrap() {
                "zero" => IndexGradient::Zero,
                "backward-difference" => IndexGradient::BackwardDifference,
                _ => IndexGradient::Reject,
            },
        };
        let result = circuit.backpropagate_reference(&values, &output_gradients, options);
        match (&result, &case["gradients"]) {
            (Err(error), Value::String(message)) => assert_eq!(&error.to_string(), message, "{name}"),
            (Ok(gradients), expected @ Value::Array(_)) => {
                for (wire, (gradient, expected)) in gradients.iter().zip(bits32(expected)).enumerate() {
                    assert!(
                        same_f32(gradient.to_bits(), expected),
                        "{name}: gradient of wire {wire} on source {source:?} is {gradient:?}, expected {:?}",
                        f32::from_bits(expected),
                    );
                }
            }
            (result, expected) => panic!("{name}: backpropagation gave {result:?}, expected {expected}"),
        }
    }
    cases.len()
}

fn f64_hex(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn check_declarations(name: &str, circuit: &Circuit, roots: &[Wire], expected: &Value) {
    let root_positions: Vec<usize> = roots.iter().map(|root| root.index()).collect();
    assert_eq!(serde_json::json!(root_positions), expected["roots"], "{name}: roots");
    assert_eq!(serde_json::json!(circuit.exact_bits), expected["exactBits"], "{name}: exact bits");
    let groups: Vec<Value> = circuit
        .one_hot_groups
        .iter()
        .map(|group| {
            let members: Vec<usize> = group.members.iter().map(|member| member.index()).collect();
            serde_json::json!({"members": members, "allowAllZero": group.allow_all_zero})
        })
        .collect();
    assert_eq!(Value::Array(groups), expected["oneHotGroups"], "{name}: one-hot groups");
    assert_eq!(outcome(validate_circuit(circuit).map(|count| serde_json::json!(count))), expected["validCircuit"], "{name}");
}

// A validator's result as the generator records it: "ok", a returned value,
// or {"error": message}.
fn outcome(result: Result<Value, String>) -> Value {
    match result {
        Ok(value) => value,
        Err(message) => serde_json::json!({"error": message}),
    }
}

fn ok(result: Result<(), String>) -> Value {
    outcome(result.map(|()| Value::from("ok")))
}

fn serialize_program(program: &TensorProgram) -> Value {
    let layers: Vec<Value> = program
        .layers
        .iter()
        .map(|layer| {
            serde_json::json!({
                "inputIndices": layer.input_indices,
                "inputSize": layer.input_size,
                "outputSize": layer.output_size,
                "bias": f32_bits(&layer.bias),
                "gateRows": layer.gate_rows,
                "booleanRows": layer.boolean_rows,
                "rowOffsets": layer.row_offsets,
                "columns": layer.columns,
                "weights": f32_bits(&layer.weights),
                "gatherComputes": layer.gather_computes.iter().map(|compute| serde_json::json!({
                    "slot": compute.slot, "bitPositions": compute.bit_positions,
                })).collect::<Vec<_>>(),
                "gatherReads": layer.gather_reads.iter().map(|read| serde_json::json!({
                    "row": read.row, "slot": read.slot, "masked": read.masked, "candidatePositions": read.candidate_positions,
                })).collect::<Vec<_>>(),
                "indexReads": layer.index_reads.iter().map(|read| serde_json::json!({
                    "row": read.row,
                    "indexRow": read.index_row,
                    "belowZero": read.below_zero,
                    "aboveZero": read.above_zero,
                    "candidateCount": read.candidate_count,
                    "candidateBase": read.candidate_base,
                    "candidateStride": read.candidate_stride,
                    "candidatePositions": read.candidate_positions,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    serde_json::json!({
        "layers": layers,
        "roots": program.roots,
        "valueCount": program.value_count,
        "wireOrder": program.wire_order,
        "width": program.width,
        "gatherSlotCount": program.gather_slot_count,
    })
}

// FNV-1a over the FP32 bits, every NaN as one NaN, like the generator's hashValues.
fn hash_values(values: &[f32]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for value in values {
        let word = if value.is_nan() { 0x7fc0_0000 } else { value.to_bits() };
        for byte in word.to_le_bytes() {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
    }
    hash
}

// The samples the generator ran the programs and validators on.
fn program_samples(expected: &Value) -> Vec<usize> {
    expected["validations"].as_array().unwrap().iter().map(|validation| validation["sample"].as_u64().unwrap() as usize).collect()
}

fn program_sources(expected: &Value) -> Vec<Vec<f32>> {
    let evaluations = expected["evaluations"].as_array().unwrap();
    program_samples(expected).into_iter().map(|sample| f32s(&bits32(&evaluations[sample]["source"]))).collect()
}

fn check_programs(name: &str, circuit: &Circuit, roots: &[Wire], expected: &Value) -> usize {
    let sources = program_sources(expected);
    let recompute = CompileOptions { recompute_depths: true };
    let programs = [
        ("lowered", circuit.compile_tensor_program(None, true, CompileOptions::default())),
        ("plain", circuit.compile_tensor_program(None, false, CompileOptions::default())),
        ("recomputed", circuit.compile_tensor_program(None, true, recompute)),
        ("roots", circuit.compile_tensor_program(Some(roots), true, recompute)),
    ];
    for (label, program) in &programs {
        let expected = &expected["programs"][label];
        let actual = serialize_program(program);
        if actual != expected["program"] {
            let actual_layers = actual["layers"].as_array().unwrap();
            let expected_layers = expected["program"]["layers"].as_array().unwrap();
            for (index, (actual, expected)) in actual_layers.iter().zip(expected_layers).enumerate() {
                assert_eq!(actual, expected, "{name}: {label} program layer {index}");
            }
            assert_eq!(actual, expected["program"], "{name}: {label} program");
        }
        assert_eq!(ok(validate_tensor_program(circuit, program)), expected["valid"], "{name}: {label} program validity");
        for (source, expected_hash) in sources.iter().zip(expected["valueHashes"].as_array().unwrap()) {
            let values = circuit.evaluate_tensor(source, Some(program));
            assert_eq!(
                hash_values(&values) as u64,
                expected_hash.as_u64().unwrap(),
                "{name}: {label} program on source {source:?} gave {values:?}"
            );
        }
    }
    sources.len() * programs.len()
}

fn check_validations(name: &str, circuit: &Circuit, roots: &[Wire], expected: &Value) {
    let mut evaluator = circuit.create_tensor_evaluator(roots, CompileOptions { recompute_depths: true });
    let validations = expected["validations"].as_array().unwrap();
    let evaluations = expected["evaluations"].as_array().unwrap();
    for ((source, expected), sample) in program_sources(expected).iter().zip(validations).zip(program_samples(expected)) {
        let values = f32s(&bits32(&evaluations[sample]["values"]));
        let root_values = evaluator.evaluate(source);
        assert!(
            bits32(&expected["rootValues"]).iter().zip(root_values).all(|(&expected, value)| same_f32(value.to_bits(), expected)),
            "{name}: evaluator roots on {source:?} gave {root_values:?}, expected {}",
            expected["rootValues"]
        );
        let context = format!("{name} on source {source:?}");
        assert_eq!(ok(validate_one_hot_groups(circuit, &values)), expected["oneHotGroups"], "{context}");
        assert_eq!(ok(validate_index_reads(circuit, &values)), expected["indexReads"], "{context}");
        assert_eq!(ok(validate_grids(circuit, &values)), expected["grids"], "{context}");
        assert_eq!(serde_json::json!(describe_first_nan(circuit, &values, None)), expected["firstNaN"], "{context}");
        assert_eq!(
            serde_json::json!(describe_first_nan(circuit, &values, Some(roots))),
            expected["firstNaNOfRoots"],
            "{context}"
        );
        let tensor = circuit.evaluate_tensor(source, None);
        assert_eq!(ok(validate_tensor_evaluation(source, circuit, &tensor)), expected["tensorEvaluation"], "{context}");
    }
}

#[test]
fn matches_typescript_runtime() {
    let fixtures: Value =
        serde_json::from_str(include_str!("../conformance/fixtures.json")).expect("fixtures.json is valid JSON");
    let scenarios = fixtures.as_object().unwrap();
    assert!(!scenarios.is_empty());
    for (name, expected) in scenarios {
        let (circuit, roots) = build(name, expected["sourceSize"].as_u64().unwrap() as usize);
        check_structure(name, &circuit, expected);
        check_declarations(name, &circuit, &roots, expected);
        let evaluations = check_evaluations(name, &circuit, expected);
        let gradients = check_gradients(name, &circuit, expected);
        let tensor_evaluations = check_programs(name, &circuit, &roots, expected);
        check_validations(name, &circuit, &roots, expected);
        println!(
            "{name}: {} operations, {evaluations} evaluations, {gradients} gradients, {tensor_evaluations} tensor evaluations",
            circuit.operations.len()
        );
    }
}
