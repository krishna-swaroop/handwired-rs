//! Exact, differentiable circuits of affine sums and ReGLU products, built from
//! ordinary program logic. A Rust port of
//! [chenglou/handwired](https://github.com/chenglou/handwired)'s runtime: the
//! same operations evaluate to the same FP32 bits as the TypeScript reference
//! evaluator, which `tests/conformance.rs` checks.
//!
//! ```
//! use handwired::{emit_compare, emit_select, Circuit, CompareOptions, Relation};
//!
//! // const result = a > 4 ? 10 : 12
//! let mut circuit = Circuit::new(1);
//! let a = circuit.source(0);
//! circuit.assume_range(a, -100.0, 100.0);
//! circuit.assume_grid(a, 1.0);
//! let four = circuit.constant(4.0);
//! let above4 = emit_compare(&mut circuit, a, Relation::Greater, four, CompareOptions::default());
//! let (ten, twelve) = (circuit.constant(10.0), circuit.constant(12.0));
//! let result = emit_select(&mut circuit, above4, twelve, ten);
//! assert_eq!(circuit.evaluate_reference(&[5.0])[result.index()], 10.0);
//! assert_eq!(circuit.evaluate_reference(&[4.0])[result.index()], 12.0);
//! ```

// The NaN-aware `!(a <= b)` checks mirror the TypeScript runtime on purpose.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
// Index loops keep the ported code line-for-line comparable with the TypeScript runtime.
#![allow(clippy::needless_range_loop)]

pub mod cells;
mod circuit;
mod compile;
mod domain;
mod emit;
pub mod fp;
pub mod handwiredos;
pub mod host_timing;
mod inline;
pub mod json_search;
mod motion;
pub mod prefix;
mod selection;
mod validate;

pub use circuit::*;
pub use compile::*;
pub use domain::*;
pub use emit::*;
pub use inline::*;
pub use motion::*;
pub use selection::*;
pub use validate::*;

/// Translates a function into the `Domain` vocabulary; see FRONTEND.md.
///
/// ```
/// #[handwired::step]
/// fn above(#[range(-100, 100)] a: i32) -> i32 {
///     if a > 4 { 10 } else { 12 }
/// }
///
/// let step = above_circuit();
/// assert_eq!(step.circuit.evaluate_reference(&[5.0])[step.output.index()], 10.0);
/// assert_eq!(above_in(&mut handwired::Native, 4.0), 12.0);
/// ```
///
/// Anything outside the subset is a compile error, e.g. division:
///
/// ```compile_fail
/// #[handwired::step]
/// fn half(#[range(0, 10)] a: i32) -> i32 {
///     a / 2
/// }
/// ```
pub use handwired_macros::step;

// Lets the macro's `::handwired` paths resolve inside this crate too.
extern crate self as handwired;
