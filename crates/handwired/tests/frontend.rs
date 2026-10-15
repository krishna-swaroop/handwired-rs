// The frontend spike (FRONTEND.md, M0): plain functions under
// #[handwired::step], checked four ways over every input in their ranges: the
// original Rust, the macro's translation as numbers, the circuit's reference
// evaluator, and its compiled tensor program.

use handwired::{CompareOptions, Native, Relation, StepCircuit, emit_compare, emit_select};

#[handwired::step]
fn above(#[range(-100, 100)] a: i32) -> i32 {
    if a > 4 { 10 } else { 12 }
}

#[handwired::step]
fn score(#[range(0, 50)] x: i32, #[range(-20, 20)] y: i32, armed: bool) -> i32 {
    let mut total = x * 3 - y;
    total = if armed && x >= y { total + x * y } else { total };
    let near = x - y < 5 && y - x < 5;
    if near {
        -total
    } else if total > 40 || !armed {
        40
    } else {
        total
    }
}

// The network's value of `step` on `source`, from the reference evaluator and
// from the compiled program, which must agree.
fn network_values(step: &StepCircuit, source: &[f32]) -> f64 {
    let reference = step.circuit.evaluate_reference(source)[step.output.index()];
    let program = step.circuit.compile_tensor_program(Some(&[step.output]), true, Default::default());
    let tensor = step.circuit.evaluate_tensor(source, Some(&program))[step.output.index()];
    assert_eq!(reference.to_bits(), tensor.to_bits(), "reference and tensor program differ on {source:?}");
    reference as f64
}

#[test]
fn above_matches_everywhere() {
    let step = above_circuit();
    handwired::validate_circuit(&step.circuit).unwrap();
    for a in -100..=100 {
        let expected = above(a) as f64;
        assert_eq!(above_in(&mut Native, a as f64), expected, "translation at a = {a}");
        assert_eq!(network_values(&step, &[a as f32]), expected, "network at a = {a}");
    }
}

#[test]
fn above_builds_the_hand_written_circuit() {
    // lib.rs's doc example, by hand, with the macro's range.
    let mut circuit = handwired::Circuit::new(1);
    let a = circuit.source(0);
    circuit.assume_range(a, -100.0, 100.0);
    circuit.assume_grid(a, 1.0);
    let four = circuit.constant(4.0);
    let above4 = emit_compare(&mut circuit, a, Relation::Greater, four, CompareOptions::default());
    let (ten, twelve) = (circuit.constant(10.0), circuit.constant(12.0));
    let result = emit_select(&mut circuit, above4, twelve, ten);

    let step = above_circuit();
    assert_eq!(step.output, result);
    assert_eq!(format!("{:?}", step.circuit.operations), format!("{:?}", circuit.operations));
}

#[test]
fn score_matches_everywhere() {
    let step = score_circuit();
    handwired::validate_circuit(&step.circuit).unwrap();
    let mut checked = 0;
    for x in 0..=50 {
        for y in -20..=20 {
            for armed in [false, true] {
                let expected = score(x, y, armed) as f64;
                let bit = if armed { 1.0 } else { 0.0 };
                assert_eq!(score_in(&mut Native, x as f64, y as f64, bit), expected, "translation at {x}, {y}, {armed}");
                assert_eq!(network_values(&step, &[x as f32, y as f32, bit as f32]), expected, "network at {x}, {y}, {armed}");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 51 * 41 * 2);
}
