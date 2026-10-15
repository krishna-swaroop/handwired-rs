// Builder functions that replace familiar expressions, e.g. `a >= b` or
// `Math.min(a, b)`. A port of the emit* functions in
// src/runtime/numeric-ui-core.ts; docs/expressions.md has each construction's
// cost and the inputs it's exact for.

use crate::circuit::{AtomKind, Circuit, Range, RoundingPointKind, Term, Wire};
use crate::fp::{fround, lowest_power_of_two, pow2, EXACT_GRID_STEPS, LARGEST_GRID, SMALLEST_GRID};

pub fn emit_relu(circuit: &mut Circuit, value: Wire) -> Wire {
    circuit.reglu(&[], 1.0, &[(value, 1.0)], 0.0)
}

pub fn emit_multiply(circuit: &mut Circuit, a: Wire, b: Wire) -> Wire {
    circuit.multiply(a, b)
}

/// b + relu(a - b). Exact for integer operands below 2^23; fractional operands
/// drift, see `emit_maximum_exact`.
pub fn emit_maximum(circuit: &mut Circuit, a: Wire, b: Wire) -> Wire {
    let difference = circuit.round_affine(&[(a, 1.0), (b, -1.0)], 0.0);
    let relu = emit_relu(circuit, difference);
    let result = circuit.round_affine(&[(b, 1.0), (relu, 1.0)], 0.0);
    let a_range = circuit.ranges[a.index()];
    let b_range = circuit.ranges[b.index()];
    // In [0,1], subtraction error is below the midpoint above 1, so the
    // reconstruction cannot round above 1. This does not imply exact max.
    if a_range.minimum >= 0.0 && a_range.maximum <= 1.0 && b_range.minimum >= 0.0 && b_range.maximum <= 1.0 {
        circuit.assume_range(result, 0.0, 1.0);
    }
    result
}

/// a - relu(a - b).
pub fn emit_minimum(circuit: &mut Circuit, a: Wire, b: Wire) -> Wire {
    let difference = circuit.round_affine(&[(a, 1.0), (b, -1.0)], 0.0);
    let relu = emit_relu(circuit, difference);
    let result = circuit.round_affine(&[(a, 1.0), (relu, -1.0)], 0.0);
    // For finite nonnegative operands, rounded max(0, a-b) is within [0,a].
    if circuit.is_non_negative(a) && circuit.is_non_negative(b) && circuit.has_finite_range(a) && circuit.has_finite_range(b)
    {
        let maximum = fround(circuit.ranges[a.index()].maximum);
        circuit.assume_range(result, 0.0, maximum);
    }
    result
}

pub fn emit_select(circuit: &mut Circuit, gate: Wire, off: Wire, on: Wire) -> Wire {
    circuit.select(gate, off, on)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Relation {
    Greater,
    AtLeast,
    Less,
    AtMost,
    Equal,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct CompareOptions {
    /// End with a boolean atom instead of a ReLU stage. Needs an executor that
    /// keeps FP32 rounding at the atom.
    pub end_with_boolean: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SignTest {
    Greater,
    AtLeast,
    Equal,
}

// One side of a comparison. A constant has no wire, a range of one value, and
// the grid of its value, or Infinity for 0, which lies on every grid.
#[derive(Copy, Clone)]
struct CompareOperand {
    wire: Option<Wire>,
    value: f64,
    range: Range,
    grid: Option<f64>,
}

/// [a relation b] as an exact bit, equal to the comparison of the two wires'
/// FP32 values for every pair of values within the declared ranges. Both
/// operands need finite declared ranges. On grids, the bit is built from the
/// integer (a - b) / grid; otherwise a sign cascade lifts every nonzero
/// difference to at least 1.
pub fn emit_compare(circuit: &mut Circuit, a: Wire, relation: Relation, b: Wire, options: CompareOptions) -> Wire {
    let mut left = compare_operand(circuit, a);
    let mut right = compare_operand(circuit, b);
    if matches!(relation, Relation::Less | Relation::AtMost) {
        std::mem::swap(&mut left, &mut right);
    }
    let test = match relation {
        Relation::Greater | Relation::Less => SignTest::Greater,
        Relation::Equal => SignTest::Equal,
        Relation::AtLeast | Relation::AtMost => SignTest::AtLeast,
    };
    if left.wire.is_none() && right.wire.is_none() {
        let holds = match test {
            SignTest::Greater => left.value > right.value,
            SignTest::AtLeast => left.value >= right.value,
            SignTest::Equal => left.value == right.value,
        };
        return circuit.affine(&[], if holds { 1.0 } else { 0.0 });
    }
    for operand in [left, right] {
        let Range { minimum, maximum } = operand.range;
        if !(minimum.is_finite() && maximum.is_finite()) {
            panic!("emit_compare needs finite declared ranges on both operands, got {minimum}..{maximum}");
        }
    }
    // A wire equals itself for every value its finite declared range allows.
    if left.wire == right.wire {
        return circuit.affine(&[], if test == SignTest::Greater { 0.0 } else { 1.0 });
    }
    let difference_bound = (left.range.maximum - right.range.minimum)
        .abs()
        .max((left.range.minimum - right.range.maximum).abs());
    assert!(
        fround(difference_bound).is_finite(),
        "the difference of the declared operand ranges can overflow FP32"
    );
    let mut terms: Vec<Term> = Vec::new();
    if let Some(wire) = left.wire {
        terms.push((wire, 1.0));
    }
    if let Some(wire) = right.wire {
        terms.push((wire, -1.0));
    }
    // At most one side is a constant, so the bias is exactly that constant or its negation.
    let bias = left.value - right.value;
    let end_with_boolean = options.end_with_boolean;
    let bit = match emit_grid_compare(circuit, &left, &right, &terms, bias, test, end_with_boolean) {
        Some(bit) => bit,
        None => emit_cascade_compare(circuit, &terms, bias, fround(difference_bound), test, end_with_boolean),
    };
    circuit.assume_range(bit, 0.0, 1.0);
    circuit.assume_grid(bit, 1.0);
    if bit.index() >= circuit.source_size {
        let index = bit.index() - circuit.source_size;
        circuit.operations[index].comparison_bit = true;
    }
    bit
}

fn compare_operand(circuit: &Circuit, operand: Wire) -> CompareOperand {
    assert!(operand.index() < circuit.width(), "wire is outside the compiled circuit");
    if let Some(constant) = circuit.constants[operand.index()] {
        // A constant wire holds f(bias).
        let value = fround(constant);
        assert!(value.is_finite(), "a constant comparison operand must be finite");
        return constant_operand(value);
    }
    CompareOperand {
        wire: Some(operand),
        value: 0.0,
        range: circuit.ranges[operand.index()],
        grid: circuit.grids[operand.index()],
    }
}

fn constant_operand(value: f64) -> CompareOperand {
    // -0 and +0 compare equal, so a zero constant becomes +0.
    let normalized = if value == 0.0 { 0.0 } else { value };
    CompareOperand {
        wire: None,
        value: normalized,
        range: Range::new(normalized, normalized),
        grid: Some(if normalized == 0.0 { f64::INFINITY } else { lowest_power_of_two(normalized) }),
    }
}

// Integer bits for operands on grids. D = (a - b) / grid is an integer, and
// every product and partial sum is an integer number of grid steps within 2^24,
// so nothing rounds. None when an operand has no grid or the ranges are too
// wide.
fn emit_grid_compare(
    circuit: &mut Circuit,
    left: &CompareOperand,
    right: &CompareOperand,
    terms: &[Term],
    bias: f64,
    test: SignTest,
    end_with_boolean: bool,
) -> Option<Wire> {
    let grid = left.grid?.min(right.grid?);
    // The largest partial sum of sign × D + offset in stored order, bias first,
    // in grid steps.
    let largest_steps = |circuit: &Circuit, sign: f64, offset: f64| {
        let scaled: Vec<Term> = terms.iter().map(|&(wire, coefficient)| (wire, sign * coefficient / grid)).collect();
        largest_partial_sum(&scaled, sign * bias / grid + offset, &circuit.ranges)
    };
    let mut emitted = Vec::new();
    let bit;
    if end_with_boolean {
        // The score 0.5 + (D - offset) × 2^-24 is a multiple of 2^-24 within
        // 0.25..0.75 while every partial sum stays within 2^22 grid steps, so
        // D = 0 scores exactly 0.5 and rounds to 1.
        let score_scale = pow2(-24) / grid;
        let limit = pow2(22);
        let fits = if test == SignTest::Equal {
            largest_steps(circuit, 1.0, 0.0) <= limit && largest_steps(circuit, -1.0, 0.0) <= limit
        } else {
            largest_steps(circuit, 1.0, if test == SignTest::Greater { -1.0 } else { 0.0 }) <= limit
        };
        if !(fits && (SMALLEST_GRID..=LARGEST_GRID).contains(&score_scale)) {
            return None;
        }
        let mut emit_atom = |circuit: &mut Circuit, sign: f64, offset: f64| {
            let atom = circuit.append(
                AtomKind::Boolean,
                terms.iter().map(|&(wire, coefficient)| (wire, sign * coefficient * score_scale)).collect(),
                0.5 + (sign * bias / grid - offset) * pow2(-24),
                Vec::new(),
                0.0,
                true,
                None,
            );
            emitted.push(atom);
            atom
        };
        if test == SignTest::Equal {
            let above = emit_atom(circuit, 1.0, 0.0);
            let below = emit_atom(circuit, -1.0, 0.0);
            bit = circuit.reglu(&[], 1.0, &[(above, 1.0), (below, 1.0)], -1.0);
            emitted.push(bit);
        } else {
            bit = emit_atom(circuit, 1.0, if test == SignTest::Greater { 1.0 } else { 0.0 });
        }
    } else {
        let scale = 1.0 / grid;
        // The bit's own sum, 1 - ramp, adds one more step.
        let fits = if test == SignTest::Equal {
            largest_steps(circuit, 1.0, 0.0) < EXACT_GRID_STEPS && largest_steps(circuit, -1.0, 0.0) < EXACT_GRID_STEPS
        } else {
            largest_steps(circuit, -1.0, if test == SignTest::Greater { 1.0 } else { 0.0 }) < EXACT_GRID_STEPS
        };
        if !(fits && scale <= LARGEST_GRID) {
            return None;
        }
        // relu(sign × D + offset), reading the operand wires as written.
        bit = emit_integer_bit(circuit, test, &mut emitted, |circuit, sign, offset| {
            circuit.append(
                AtomKind::Reglu,
                Vec::new(),
                1.0,
                terms.iter().map(|&(wire, coefficient)| (wire, sign * coefficient * scale)).collect(),
                sign * bias / grid + offset,
                true,
                None,
            )
        });
    }
    for wire in emitted {
        circuit.record_rounding_point(wire, RoundingPointKind::IntegerBit);
    }
    Some(bit)
}

// The largest magnitude of any product or partial sum of bias + Σ coefficient ×
// operand, added in stored order, from the declared operand ranges.
fn largest_partial_sum(terms: &[Term], bias: f64, ranges: &[Range]) -> f64 {
    let mut minimum = bias;
    let mut maximum = bias;
    let mut largest = bias.abs();
    for &(wire, coefficient) in terms {
        let range = ranges[wire.index()];
        let low = if coefficient >= 0.0 { coefficient * range.minimum } else { coefficient * range.maximum };
        let high = if coefficient >= 0.0 { coefficient * range.maximum } else { coefficient * range.minimum };
        minimum += low;
        maximum += high;
        largest = largest.max(low.abs()).max(high.abs()).max(minimum.abs()).max(maximum.abs());
    }
    largest
}

// The exact bit of D > 0, D >= 0 or D === 0, when every value of D is 0 or at
// least 1 away from 0. emit_ramp(sign, offset) builds relu(sign × D + offset):
// - D >= 0 is relu(1 - relu(-D)).
// - D > 0 is relu(1 - relu(1 - D)).
// - D === 0 is relu(1 - relu(D) - relu(-D)).
fn emit_integer_bit(
    circuit: &mut Circuit,
    test: SignTest,
    emitted: &mut Vec<Wire>,
    mut emit_ramp: impl FnMut(&mut Circuit, f64, f64) -> Wire,
) -> Wire {
    let ramps = if test == SignTest::Equal {
        vec![emit_ramp(circuit, 1.0, 0.0), emit_ramp(circuit, -1.0, 0.0)]
    } else {
        vec![emit_ramp(circuit, -1.0, if test == SignTest::Greater { 1.0 } else { 0.0 })]
    };
    let gate_terms: Vec<Term> = ramps.iter().map(|&ramp| (ramp, -1.0)).collect();
    let bit = circuit.reglu(&[], 1.0, &gate_terms, 1.0);
    emitted.extend(ramps);
    emitted.push(bit);
    bit
}

// Every nonzero FP32 difference, down to 2^-149, lifted to at least 1.
const FULL_LIFT_EXPONENT: i32 = 149;
// Every nonzero FP32 difference lifted to at least 2^-45, which a final
// rounding stage with scale 2^22 resolves.
const ROUNDING_STAGE_LIFT_EXPONENT: i32 = 104;

#[derive(Copy, Clone, Debug)]
struct CascadeStage {
    scale: f64,
    limit: f64,
}

// Stages of min(limit, scale × input). Multiplying by a power of two only
// changes the exponent, and min(limit, x) = x - relu(x - limit) is exact while
// x is at most 2^24 × limit. The first stage multiplies as far as the
// difference bound allows, each later stage by up to 2^24, until the product of
// the scales reaches 2^lift_exponent.
fn plan_sign_cascade(difference_bound: f64, lift_exponent: i32) -> Vec<CascadeStage> {
    let bound_exponent = power_of_two_exponent_at_least(difference_bound.max(SMALLEST_GRID));
    let mut stages = Vec::new();
    let mut limit_exponent = bound_exponent;
    let mut gain_exponent = 0;
    while stages.is_empty() || limit_exponent > 0 || gain_exponent < lift_exponent {
        let next_limit_exponent = (limit_exponent - 24).max(0);
        // A coefficient above 2^127 isn't an FP32 value.
        let room = 127.min(24 + next_limit_exponent - limit_exponent);
        let scale_exponent = room.min((lift_exponent - gain_exponent).max(0));
        stages.push(CascadeStage { scale: pow2(scale_exponent), limit: pow2(next_limit_exponent) });
        limit_exponent = next_limit_exponent;
        gain_exponent += scale_exponent;
    }
    stages
}

// The smallest integer e with 2^e >= value, for a finite positive value.
fn power_of_two_exponent_at_least(value: f64) -> i32 {
    let mut exponent = value.log2().ceil() as i32;
    while pow2(exponent) < value {
        exponent += 1;
    }
    while pow2(exponent - 1) >= value {
        exponent -= 1;
    }
    exponent
}

// After clamp stage k, the output is min(limit_k, scale_1 × ... × scale_k ×
// relu(sign × d)).
fn emit_clamp_stages(
    circuit: &mut Circuit,
    emitted: &mut Vec<Wire>,
    difference: Wire,
    difference_sign: f64,
    sign: f64,
    clamp_stages: &[CascadeStage],
) -> Wire {
    let mut stage_output = difference;
    for (index, &CascadeStage { scale, limit }) in clamp_stages.iter().enumerate() {
        let scaled_term: Term;
        let above_limit;
        if index == 0 {
            // The first stage reads the signed difference as written, and keeps
            // only its positive side.
            let gate_terms = vec![(difference, sign * difference_sign * scale)];
            let scaled_positive = circuit.append(AtomKind::Reglu, Vec::new(), 1.0, gate_terms.clone(), 0.0, true, None);
            above_limit = circuit.append(AtomKind::Reglu, Vec::new(), 1.0, gate_terms, -limit, true, None);
            scaled_term = (scaled_positive, 1.0);
            emitted.push(scaled_positive);
        } else {
            above_limit = circuit.reglu(&[], 1.0, &[(stage_output, scale)], -limit);
            scaled_term = (stage_output, scale);
        }
        // Its own operation: copied into the next stage, the regrouped partial
        // sums would round.
        stage_output = circuit.round_affine(&[scaled_term, (above_limit, -1.0)], 0.0);
        circuit.assume_range(stage_output, 0.0, limit);
        emitted.push(above_limit);
        emitted.push(stage_output);
    }
    stage_output
}

// The sign cascade on d = f(a - b). For finite FP32 operands, f(a - b) is 0
// exactly when a === b, and otherwise has the sign of a - b, since subnormal
// values fill the gap next to 0.
fn emit_cascade_compare(
    circuit: &mut Circuit,
    terms: &[Term],
    bias: f64,
    difference_bound: f64,
    test: SignTest,
    end_with_boolean: bool,
) -> Wire {
    let mut emitted = Vec::new();
    // A comparison against 0 reads its operand directly.
    let (difference, difference_sign) = if bias == 0.0 && terms.len() == 1 {
        terms[0]
    } else {
        let difference = circuit.round_affine(terms, bias);
        emitted.push(difference);
        (difference, 1.0)
    };
    let stages = plan_sign_cascade(
        difference_bound,
        if end_with_boolean { ROUNDING_STAGE_LIFT_EXPONENT } else { FULL_LIFT_EXPONENT },
    );
    let bit;
    if end_with_boolean {
        // The clamp stages lift every positive difference to at least 2^-45, and
        // 2^22 × 2^-45 moves a score next to 0.5 across 0.5.
        let score_scale = pow2(22);
        if test == SignTest::Greater {
            let stage_output = emit_clamp_stages(circuit, &mut emitted, difference, difference_sign, 1.0, &stages);
            // The FP32 value just below 0.5, so d = 0 rounds to 0.
            let just_below_half = 0.5 - pow2(-25);
            let score_above_zero = circuit.reglu(&[], 1.0, &[(stage_output, score_scale)], just_below_half);
            let score_above_one = circuit.reglu(&[], 1.0, &[(stage_output, score_scale)], just_below_half - 1.0);
            let score = circuit.affine(&[(score_above_zero, 1.0), (score_above_one, -1.0)], 0.0);
            circuit.assume_range(score, 0.0, 1.0);
            emitted.push(score_above_zero);
            emitted.push(score_above_one);
            bit = circuit.boolean(score);
        } else {
            // a >= b scores 0.5 - 2^22 × (the stages lifting b - a), and equality
            // also subtracts the stages lifting a - b.
            let below = emit_clamp_stages(circuit, &mut emitted, difference, difference_sign, -1.0, &stages);
            let mut gate_terms = vec![(below, -score_scale)];
            if test == SignTest::Equal {
                let above = emit_clamp_stages(circuit, &mut emitted, difference, difference_sign, 1.0, &stages);
                gate_terms.push((above, -score_scale));
            }
            let score = circuit.reglu(&[], 1.0, &gate_terms, 0.5);
            emitted.push(score);
            bit = circuit.boolean(score);
        }
        emitted.push(bit);
    } else {
        let final_scale = stages.last().unwrap().scale;
        let clamp_stages = &stages[..stages.len() - 1];
        if clamp_stages.is_empty() {
            // One stage lifts the signed difference itself.
            bit = emit_integer_bit(circuit, test, &mut emitted, |circuit, sign, offset| {
                circuit.append(
                    AtomKind::Reglu,
                    Vec::new(),
                    1.0,
                    vec![(difference, sign * difference_sign * final_scale)],
                    offset,
                    true,
                    None,
                )
            });
        } else if test == SignTest::Greater {
            // scale × stage is 0 when d <= 0, and at least 1 when d > 0.
            let stage = emit_clamp_stages(circuit, &mut emitted, difference, difference_sign, 1.0, clamp_stages);
            let not_greater = circuit.reglu(&[], 1.0, &[(stage, -final_scale)], 1.0);
            bit = circuit.reglu(&[], 1.0, &[(not_greater, -1.0)], 1.0);
            emitted.push(not_greater);
            emitted.push(bit);
        } else {
            let below = emit_clamp_stages(circuit, &mut emitted, difference, difference_sign, -1.0, clamp_stages);
            let mut gate_terms = vec![(below, -final_scale)];
            if test == SignTest::Equal {
                let above = emit_clamp_stages(circuit, &mut emitted, difference, difference_sign, 1.0, clamp_stages);
                gate_terms.push((above, -final_scale));
            }
            bit = circuit.reglu(&[], 1.0, &gate_terms, 1.0);
            emitted.push(bit);
        }
    }
    for wire in emitted {
        circuit.record_rounding_point(wire, RoundingPointKind::SignCascade);
    }
    bit
}

/// min(a, b) as exactly one of the operands. Ties return a. Needs finite
/// declared ranges, like `emit_compare`.
pub fn emit_minimum_exact(circuit: &mut Circuit, a: Wire, b: Wire) -> Wire {
    let a_is_smaller = emit_compare(circuit, a, Relation::AtMost, b, CompareOptions::default());
    let a_range = compare_operand(circuit, a).range;
    let b_range = compare_operand(circuit, b).range;
    emit_select_operand(
        circuit,
        a_is_smaller,
        b,
        a,
        a_range.minimum.min(b_range.minimum),
        a_range.maximum.min(b_range.maximum),
    )
}

/// max(a, b) as exactly one of the operands. Ties return a.
pub fn emit_maximum_exact(circuit: &mut Circuit, a: Wire, b: Wire) -> Wire {
    let a_is_larger = emit_compare(circuit, a, Relation::AtLeast, b, CompareOptions::default());
    let a_range = compare_operand(circuit, a).range;
    let b_range = compare_operand(circuit, b).range;
    emit_select_operand(
        circuit,
        a_is_larger,
        b,
        a,
        a_range.minimum.max(b_range.minimum),
        a_range.maximum.max(b_range.maximum),
    )
}

/// min(max(value, minimum), maximum).
pub fn emit_clamp_exact(circuit: &mut Circuit, value: Wire, minimum: Wire, maximum: Wire) -> Wire {
    let at_least_minimum = emit_maximum_exact(circuit, value, minimum);
    emit_minimum_exact(circuit, at_least_minimum, maximum)
}

// on × relu(bit) + off × relu(1 - bit) for an exact bit. The unselected product
// is ±0, so the sum returns the selected operand exactly, up to the sign of
// zero.
fn emit_select_operand(circuit: &mut Circuit, bit: Wire, off: Wire, on: Wire, minimum: f64, maximum: f64) -> Wire {
    let selected_on = circuit.reglu(&[(on, 1.0)], 0.0, &[(bit, 1.0)], 0.0);
    let selected_off = circuit.reglu(&[(off, 1.0)], 0.0, &[(bit, -1.0)], 1.0);
    let result = circuit.affine(&[(selected_on, 1.0), (selected_off, 1.0)], 0.0);
    circuit.assume_range(result, minimum, maximum);
    result
}
