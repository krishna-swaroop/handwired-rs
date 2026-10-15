// The vocabulary `#[handwired::step]` translates a function into, and its
// interpretations. See FRONTEND.md: the macro writes the function once against
// `Domain`, and each implementation gives it a meaning, the way Elliott's
// compiling to categories interprets one program in many categories.

use crate::circuit::{Circuit, Wire};
use crate::emit::{emit_compare, emit_select, CompareOptions, Relation};
use crate::fp::fround;

/// The operations a step function is made of. Bits are values that hold
/// exactly 0 or 1.
pub trait Domain {
    type Value: Copy;
    fn constant(&mut self, value: f64) -> Self::Value;
    fn add(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
    fn sub(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
    fn neg(&mut self, a: Self::Value) -> Self::Value;
    fn mul(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
    /// The bit of `a relation b`.
    fn compare(&mut self, a: Self::Value, relation: Relation, b: Self::Value) -> Self::Value;
    fn not(&mut self, bit: Self::Value) -> Self::Value;
    fn and(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
    fn or(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
    /// `on` when `gate` is 1, `off` when it is 0.
    fn select(&mut self, gate: Self::Value, off: Self::Value, on: Self::Value) -> Self::Value;
}

/// The macro's own semantics as numbers, rounded to FP32 after every
/// operation like the circuit's, to check the translation against the
/// original Rust function apart from the circuit.
pub struct Native;

impl Domain for Native {
    type Value = f64;
    fn constant(&mut self, value: f64) -> f64 {
        fround(value)
    }
    fn add(&mut self, a: f64, b: f64) -> f64 {
        fround(a + b)
    }
    fn sub(&mut self, a: f64, b: f64) -> f64 {
        fround(a - b)
    }
    fn neg(&mut self, a: f64) -> f64 {
        -a
    }
    fn mul(&mut self, a: f64, b: f64) -> f64 {
        fround(a * b)
    }
    fn compare(&mut self, a: f64, relation: Relation, b: f64) -> f64 {
        let holds = match relation {
            Relation::Greater => a > b,
            Relation::AtLeast => a >= b,
            Relation::Less => a < b,
            Relation::AtMost => a <= b,
            Relation::Equal => a == b,
        };
        if holds { 1.0 } else { 0.0 }
    }
    fn not(&mut self, bit: f64) -> f64 {
        1.0 - bit
    }
    fn and(&mut self, a: f64, b: f64) -> f64 {
        a * b
    }
    fn or(&mut self, a: f64, b: f64) -> f64 {
        a + b - a * b
    }
    fn select(&mut self, gate: f64, off: f64, on: f64) -> f64 {
        if gate == 1.0 { on } else { off }
    }
}

/// The network: every operation is built in `circuit`.
pub struct CircuitDomain<'a> {
    pub circuit: &'a mut Circuit,
}

impl CircuitDomain<'_> {
    // The value of a wire whose declared range is one point, e.g. a constant.
    fn known(&self, wire: Wire) -> Option<f64> {
        let range = self.circuit.ranges[wire.index()];
        (range.minimum == range.maximum).then_some(range.minimum)
    }
}

impl Domain for CircuitDomain<'_> {
    type Value = Wire;
    fn constant(&mut self, value: f64) -> Wire {
        self.circuit.constant(value)
    }
    fn add(&mut self, a: Wire, b: Wire) -> Wire {
        self.circuit.affine(&[(a, 1.0), (b, 1.0)], 0.0)
    }
    fn sub(&mut self, a: Wire, b: Wire) -> Wire {
        self.circuit.affine(&[(a, 1.0), (b, -1.0)], 0.0)
    }
    fn neg(&mut self, a: Wire) -> Wire {
        self.circuit.affine(&[(a, -1.0)], 0.0)
    }
    fn mul(&mut self, a: Wire, b: Wire) -> Wire {
        // A product with a known factor is affine; only two variables need a ReGLU.
        match (self.known(a), self.known(b)) {
            (Some(factor), _) => self.circuit.affine(&[(b, factor)], 0.0),
            (_, Some(factor)) => self.circuit.affine(&[(a, factor)], 0.0),
            _ => self.circuit.multiply(a, b),
        }
    }
    fn compare(&mut self, a: Wire, relation: Relation, b: Wire) -> Wire {
        emit_compare(self.circuit, a, relation, b, CompareOptions::default())
    }
    fn not(&mut self, bit: Wire) -> Wire {
        self.circuit.affine(&[(bit, -1.0)], 1.0)
    }
    fn and(&mut self, a: Wire, b: Wire) -> Wire {
        self.circuit.multiply(a, b)
    }
    fn or(&mut self, a: Wire, b: Wire) -> Wire {
        let both = self.circuit.multiply(a, b);
        self.circuit.affine(&[(a, 1.0), (b, 1.0), (both, -1.0)], 0.0)
    }
    fn select(&mut self, gate: Wire, off: Wire, on: Wire) -> Wire {
        emit_select(self.circuit, gate, off, on)
    }
}

/// The circuit `#[handwired::step]` builds for a function: one source cell per
/// parameter, in order, and the result.
pub struct StepCircuit {
    pub circuit: Circuit,
    pub inputs: Vec<Wire>,
    pub output: Wire,
}
