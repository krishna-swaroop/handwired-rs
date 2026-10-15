// Springs: a stepped spring with a settle bit, a host-written step matrix, and
// an animated object whose springs settle together. A port of emitSpring,
// emitApplySpringMatrix and emitAnimatedObject in src/runtime/numeric-ui-core.ts.

use crate::circuit::{AtomKind, Circuit, Term, Wire};
use crate::emit::{emit_multiply, emit_relu, emit_select};
use crate::fp::{fround, pow2};

/// The wires `emit_spring` reads.
#[derive(Copy, Clone, Debug)]
pub struct SpringWires {
    pub position: Wire,
    pub destination: Wire,
    pub velocity: Wire,
    pub stiffness: Wire,
    pub damping: Wire,
    pub timestep: Wire,
    pub settle_distance: Wire,
    pub settle_speed: Wire,
    pub settle_distance_scale: Wire,
    pub settle_speed_scale: Wire,
    /// A wire that holds 1.
    pub unit: Wire,
}

#[derive(Copy, Clone, Debug)]
pub struct Spring {
    pub position: Wire,
    pub velocity: Wire,
    pub active: Wire,
}

/// Four explicit Euler steps, then a clamped 0..1 activity from how far the
/// spring is from rest, which snaps it onto its destination once it settles.
pub fn emit_spring(circuit: &mut Circuit, wires: SpringWires) -> Spring {
    let SpringWires { destination, stiffness, damping, timestep, unit, .. } = wires;
    let mut position = wires.position;
    let mut velocity = wires.velocity;
    for _ in 0..4 {
        let displacement = circuit.affine(&[(position, 1.0), (destination, -1.0)], 0.0);
        let spring_force = emit_multiply(circuit, stiffness, displacement);
        let damping_force = emit_multiply(circuit, damping, velocity);
        let acceleration = circuit.affine(&[(spring_force, -1.0), (damping_force, -1.0)], 0.0);
        let velocity_step = emit_multiply(circuit, acceleration, timestep);
        velocity = circuit.affine(&[(velocity, 1.0), (velocity_step, 1.0)], 0.0);
        let position_step = emit_multiply(circuit, velocity, timestep);
        position = circuit.affine(&[(position, 1.0), (position_step, 1.0)], 0.0);
    }
    let offset = circuit.affine(&[(position, 1.0), (destination, -1.0)], 0.0);
    let distance = emit_absolute(circuit, offset);
    let distance_excess = circuit.affine(&[(distance, 1.0), (wires.settle_distance, -1.0)], 0.0);
    let distance_above_threshold = emit_relu(circuit, distance_excess);
    let speed = emit_absolute(circuit, velocity);
    let speed_excess = circuit.affine(&[(speed, 1.0), (wires.settle_speed, -1.0)], 0.0);
    let speed_above_threshold = emit_relu(circuit, speed_excess);
    let distance_term = emit_multiply(circuit, distance_above_threshold, wires.settle_distance_scale);
    let speed_term = emit_multiply(circuit, speed_above_threshold, wires.settle_speed_scale);
    let activity = circuit.affine(&[(distance_term, 1.0), (speed_term, 1.0)], 0.0);
    let active = emit_clamp_zero_one(circuit, activity, unit);
    Spring {
        position: emit_select(circuit, active, destination, position),
        velocity: emit_multiply(circuit, active, velocity),
        active,
    }
}

fn emit_absolute(circuit: &mut Circuit, value: Wire) -> Wire {
    let positive = emit_relu(circuit, value);
    let negated = circuit.affine(&[(value, -1.0)], 0.0);
    let negative = emit_relu(circuit, negated);
    circuit.affine(&[(positive, 1.0), (negative, 1.0)], 0.0)
}

fn emit_clamp_zero_one(circuit: &mut Circuit, value: Wire, unit: Wire) -> Wire {
    let positive = emit_relu(circuit, value);
    let excess = circuit.affine(&[(value, 1.0), (unit, -1.0)], 0.0);
    let above_one = emit_relu(circuit, excess);
    let result = circuit.affine(&[(positive, 1.0), (above_one, -1.0)], 0.0);
    circuit.assume_range(result, 0.0, 1.0);
    result
}

/// The 4 cells of a spring matrix that the host writes before each evaluation:
/// displacement' = m[0] × displacement + m[1] × velocity, and
/// velocity' = m[2] × displacement + m[3] × velocity.
pub type SpringMatrixWires = [Wire; 4];

#[derive(Copy, Clone, Debug)]
pub struct SpringState {
    pub displacement: Wire,
    pub velocity: Wire,
}

/// One spring's displacement and velocity after every step the matrix
/// describes. The matrix cells need finite declared ranges. 14 operations in 3
/// layers.
pub fn emit_apply_spring_matrix(
    circuit: &mut Circuit,
    matrix: SpringMatrixWires,
    displacement: Wire,
    velocity: Wire,
) -> SpringState {
    for entry in matrix {
        assert!(entry.index() < circuit.width(), "wire is outside the compiled circuit");
        assert!(
            circuit.has_finite_range(entry),
            "spring matrix wires need finite declared ranges, e.g. ±springMatrixBound(constants)"
        );
    }
    let a = circuit.multiply(matrix[0], displacement);
    let b = circuit.multiply(matrix[1], velocity);
    let next_displacement = circuit.round_affine(&[(a, 1.0), (b, 1.0)], 0.0);
    let c = circuit.multiply(matrix[2], displacement);
    let d = circuit.multiply(matrix[3], velocity);
    let next_velocity = circuit.round_affine(&[(c, 1.0), (d, 1.0)], 0.0);
    SpringState { displacement: next_displacement, velocity: next_velocity }
}

#[derive(Copy, Clone, Debug)]
pub struct AnimatedSpring {
    pub matrix: SpringMatrixWires,
    /// The displacement from `destination_during_steps`, usually a state cell.
    pub displacement: Wire,
    pub velocity: Wire,
    /// The destination from this evaluation's time on, after its events.
    pub destination: Wire,
    /// The destination in effect during this evaluation's steps. None when
    /// this evaluation can't change the destination.
    pub destination_during_steps: Option<Wire>,
}

#[derive(Copy, Clone, Debug)]
pub struct AnimatedObjectOptions {
    /// An exact bit: 1 when this evaluation ends at a fixed settle time.
    pub settle_check: Wire,
    /// An exact bit that pins every spring onto its destination.
    pub snap: Option<Wire>,
    /// 0.01 by default; at least 2^-22.
    pub settle_distance: Option<f64>,
    pub settle_speed: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct AnimatedObject {
    pub displacements: Vec<Wire>,
    pub velocities: Vec<Wire>,
    pub positions: Vec<Wire>,
    pub settled: Wire,
    pub active: Wire,
}

/// Springs of one animated object stepped by their matrices and settled
/// together with one exact bit. See emitAnimatedObject in the TypeScript
/// runtime for why every step is exact.
pub fn emit_animated_object(circuit: &mut Circuit, springs: &[AnimatedSpring], options: AnimatedObjectOptions) -> AnimatedObject {
    assert!(!springs.is_empty(), "an animated object needs at least one spring");
    let width = circuit.width();
    let settle_distance = fround(options.settle_distance.unwrap_or(0.01));
    let settle_speed = fround(options.settle_speed.unwrap_or(0.01));
    for threshold in [settle_distance, settle_speed] {
        assert!(
            threshold.is_finite() && threshold >= pow2(-22),
            "settle thresholds must be finite and at least 2^-22, got {threshold}"
        );
    }
    for bit in std::iter::once(options.settle_check).chain(options.snap) {
        assert!(bit.index() < width, "wire is outside the compiled circuit");
        let range = circuit.ranges[bit.index()];
        assert!(range.minimum >= 0.0 && range.maximum <= 1.0, "settleCheck and snap must be declared within 0..1");
    }
    let mut exceedance_terms: Vec<Term> = Vec::new();
    let mut rest_terms: Vec<Term> = Vec::new();
    let mut stepped = Vec::with_capacity(springs.len());
    for spring in springs {
        let next = emit_apply_spring_matrix(circuit, spring.matrix, spring.displacement, spring.velocity);
        let displacement = match spring.destination_during_steps {
            Some(during) if during != spring.destination => {
                circuit.round_affine(&[(during, 1.0), (spring.destination, -1.0), (next.displacement, 1.0)], 0.0)
            }
            _ => next.displacement,
        };
        for (value, threshold) in [(displacement, settle_distance), (next.velocity, settle_speed)] {
            let positive_part = circuit.reglu(&[], 1.0, &[(value, 1.0)], 0.0);
            let negative_part = circuit.reglu(&[], 1.0, &[(value, -1.0)], 0.0);
            // relu(|value| - threshold), with |value| = relu(value) + relu(-value).
            let exceedance = circuit.reglu(&[], 1.0, &[(positive_part, 1.0), (negative_part, 1.0)], -threshold);
            exceedance_terms.push((exceedance, 1.0));
            rest_terms.push((positive_part, pow2(127)));
            rest_terms.push((negative_part, pow2(127)));
        }
        stepped.push(SpringState { displacement, velocity: next.velocity });
    }
    let exceedance = circuit.round_affine(&exceedance_terms, 0.0);
    // Each lifted term is 0, at least 2^-22, or Infinity, so the lifted sum is 0
    // exactly when every displacement and velocity is ±0.
    let lifted_rest = circuit.append(AtomKind::Reglu, Vec::new(), 1.0, rest_terms, 0.0, true, None);
    let at_rest = circuit.reglu(&[], 1.0, &[(lifted_rest, -pow2(22))], 1.0);
    let settled_score = circuit.append(
        AtomKind::Reglu,
        Vec::new(),
        1.0,
        vec![(options.settle_check, 0.5), (at_rest, 0.5), (exceedance, -pow2(20))],
        0.0,
        true,
        None,
    );
    circuit.mark_rounding_point(settled_score);
    let settled = circuit.boolean(settled_score);
    let not_settled = circuit.round_affine(&[(settled, -1.0)], 1.0);
    let active = match options.snap {
        None => not_settled,
        Some(snap) => {
            let not_snapped = circuit.round_affine(&[(snap, -1.0)], 1.0);
            circuit.multiply(not_settled, not_snapped)
        }
    };
    let displacements: Vec<Wire> = stepped.iter().map(|state| circuit.multiply(state.displacement, active)).collect();
    let velocities = stepped.iter().map(|state| circuit.multiply(state.velocity, active)).collect();
    let positions = springs
        .iter()
        .zip(&displacements)
        .map(|(spring, &displacement)| circuit.round_affine(&[(spring.destination, 1.0), (displacement, 1.0)], 0.0))
        .collect();
    AnimatedObject { displacements, velocities, positions, settled, active }
}
