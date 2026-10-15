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

mod circuit;
mod compile;
mod emit;
pub mod fp;
mod inline;
mod motion;
mod selection;
mod validate;

pub use circuit::*;
pub use compile::*;
pub use emit::*;
pub use inline::*;
pub use motion::*;
pub use selection::*;
pub use validate::*;
