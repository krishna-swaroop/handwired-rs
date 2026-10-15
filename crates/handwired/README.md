# handwired (Rust)

A Rust port of [chenglou/handwired](https://github.com/chenglou/handwired)'s runtime. A circuit built here holds the same operations, terms, ranges and grids as one built with the TypeScript runtime in `src/runtime/numeric-ui-core.ts`, and evaluates to the same FP32 bits.

```rust
use handwired::{emit_compare, emit_select, Circuit, CompareOptions, Relation};

// const result = a > 4 ? 10 : 12
let mut circuit = Circuit::new(1);
let a = circuit.source(0);
circuit.assume_range(a, -100.0, 100.0);
circuit.assume_grid(a, 1.0);
let four = circuit.constant(4.0);
let above4 = emit_compare(&mut circuit, a, Relation::Greater, four, CompareOptions::default());
let (ten, twelve) = (circuit.constant(10.0), circuit.constant(12.0));
let result = emit_select(&mut circuit, above4, twelve, ten);
assert_eq!(circuit.evaluate_reference(&[5.0])[result.index()], 10.0);
```

The docs in [`docs/`](../../docs) apply as written. Names are snake_case (`emitCompare` is `emit_compare`, `assumeRange` is `assume_range`), and builder misuse panics where the TypeScript runtime throws.

## Ported

Everything in `src/runtime/numeric-ui-core.ts` except its small `assert*` helpers. One difference in shape: the TypeScript circuit caches its default compiled program in `circuit.tensorProgram`, while here `evaluate_tensor` takes a program, or compiles one when given none.

- `Circuit`: the affine, ReGLU, boolean and index read atoms, the multiply and select macros, term normalization, constant folding, small-affine composition, deduplication, depths, declarations (range, grid, bit, one-hot), inferred ranges and grids, and rounding points.
- `evaluate_reference` and `backpropagate_reference`, including the straight-through boolean gradient and the zero and backward-difference index gradients.
- The tensor compiler: `compile_tensor_program` with gather lowering of one-hot-masked sums, liveness from roots, `recompute_depths`, and packed candidate tables. The layered executor behind `evaluate_tensor`, `create_tensor_evaluator` and `create_tensor_program_evaluator`.
- Emitters: `emit_relu`, `emit_multiply`, `emit_minimum`, `emit_maximum`, `emit_select`, `emit_compare` (integer bits on grids and the sign cascade, each with or without a closing boolean atom), `emit_minimum_exact`, `emit_maximum_exact`, `emit_clamp_exact`, `emit_argmax`, `emit_first_active`, `emit_spring`, `emit_apply_spring_matrix` and `emit_animated_object`.
- `inline_circuit`.
- Validators: `validate_circuit`, `validate_tensor_program`, `validate_one_hot_groups`, `validate_index_reads`, `validate_grids`, `validate_no_nan`, `describe_first_nan` and `validate_tensor_evaluation`. They return the TypeScript runtime's error message as `Err`, with numbers printed the way JavaScript prints them (`fp::js_number`).

## Conformance

[`conformance/generate.ts`](conformance/generate.ts) builds 32 scenarios with the TypeScript runtime and writes `conformance/fixtures.json`. [`tests/conformance.rs`](tests/conformance.rs) builds the same scenarios in Rust and compares, bit for bit:

- every operation, range, grid, exact-bit declaration, one-hot group and accepted index range
- the reference evaluator's values on seeded inputs, including one-FP32-step pairs, zeros of both signs, subnormals, infinities and broken one-hot contracts
- gradients, and the error a rejected backward pass reports
- four compiled programs per scenario (gathers lowered, not lowered, recomputed depths, and only the scenario's roots): every layer's inputs, weights, gate, boolean, gather and index read rows, and the hash of every tensor evaluation
- every validator's verdict and error message, and the roots a tensor evaluator returns

The gather scenarios follow `tests/runtime/gather-lowering.test.ts`: sums that must lower (with a bias, a signed pair, the bit in the gate, a gather read by another gather, argmax and first-active selections) and sums that must stay dense (scaled, missing or extra terms, two groups, a bit-in-left product over a signed value, unbounded values, an unused negative arm, fractional first-active selections).

```bash
node crates/handwired/conformance/generate.ts
```

```bash
cargo test
```

Plain Node 23.6 or later runs the generator directly. Regenerate the fixtures after changing a scenario or pulling upstream runtime changes.

## Toward demo parity

1. **Done: the runtime.** Every builder function, the tensor compiler and the validators match the TypeScript runtime.
2. **Demo circuits, run without a UI**: port the circuits behind `src/handwiredos` and `src/json-search` (and the `src/runtime/cells.ts` and `prefix-networks.ts` helpers they use), then compare operation counts, layer counts and values against the TypeScript originals on recorded inputs.
3. **GPU backend**: a `wgpu` executor for the compiled program, checked against the reference evaluator at the rounding points.
