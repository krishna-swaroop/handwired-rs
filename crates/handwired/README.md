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

The demo circuits are ported too: HandwiredOS's window manager, Photos and Monitor's list (`handwired::handwiredos`), and JSON search's highlighter and regex search as loop programs (`handwired::json_search`). They build the same operations as the TypeScript originals and step through the same states, see [Demo parity](#demo-parity).

The docs in [`docs/`](../../docs) apply as written. Names are snake_case (`emitCompare` is `emit_compare`, `assumeRange` is `assume_range`), and builder misuse panics where the TypeScript runtime throws.

## Ported

Everything in `src/runtime/numeric-ui-core.ts` except its small `assert*` helpers. One difference in shape: the TypeScript circuit caches its default compiled program in `circuit.tensorProgram`, while here `evaluate_tensor` takes a program, or compiles one when given none.

- `Circuit`: the affine, ReGLU, boolean and index read atoms, the multiply and select macros, term normalization, constant folding, small-affine composition, deduplication, depths, declarations (range, grid, bit, one-hot), inferred ranges and grids, and rounding points.
- `evaluate_reference` and `backpropagate_reference`, including the straight-through boolean gradient and the zero and backward-difference index gradients.
- The tensor compiler: `compile_tensor_program` with gather lowering of one-hot-masked sums, liveness from roots, `recompute_depths`, and packed candidate tables. The layered executor behind `evaluate_tensor`, `create_tensor_evaluator` and `create_tensor_program_evaluator`.
- Emitters: `emit_relu`, `emit_multiply`, `emit_minimum`, `emit_maximum`, `emit_select`, `emit_compare` (integer bits on grids and the sign cascade, each with or without a closing boolean atom), `emit_minimum_exact`, `emit_maximum_exact`, `emit_clamp_exact`, `emit_argmax`, `emit_first_active`, `emit_spring`, `emit_apply_spring_matrix` and `emit_animated_object`.
- `inline_circuit`.
- Validators: `validate_circuit`, `validate_tensor_program`, `validate_one_hot_groups`, `validate_index_reads`, `validate_grids`, `validate_no_nan`, `describe_first_nan` and `validate_tensor_evaluation`. They return the TypeScript runtime's error message as `Err`, with numbers printed the way JavaScript prints them (`fp::js_number`).

## Ported: runtime helpers and demo circuits

- `cells`: `src/runtime/cells.ts`. `SourceTable` declares cells one at a time instead of as a nested object, in the same order and under the same names, e.g. `windows[2].left`, so positions, the circuit's declarations, the validator's messages and Monitor's State rows match. `TableEvaluator` is `createEvaluator`.
- `prefix`: `sklansky_prefix` and `brent_kung_prefix` from `src/runtime/prefix-networks.ts`.
- `host_timing`: all of `src/runtime/host-timing.ts`, including `advance_frame`, which plans a frame's settle checks, events, snaps and frame evaluation.
- `handwiredos`: the circuit-building code of `src/handwiredos`, without the DOM:
  - `window_manager`: the cells, `build_window_manager`, `WindowManagerMachine` (`createWindowManagerMachine`: event queueing, hover coalescing, write-back, the scroll request), the named outputs, the output checks and `note_focus_change`.
  - `photo_gallery`: `build_photos`, the standalone circuit and `PhotoGalleryMachine`.
  - `value_list`: `build_value_list`, the standalone circuit, Monitor's Operations and State rows and `format_value`.
  - `arrangement`: `arrange_windows` and `arrange_initial_windows`.
  - `monitor_footer`: the program's size, the frame-time averages, the chart's scale and SVG paths, and the footer's text per frame (`MonitorFooter`), with JavaScript's `toFixed`.
  - `wiring`: the wallpaper's arcs, as a list instead of canvas strokes, and their opacity.
- `json_search`: the core of `src/json-search`, without the page, the workers and the GPU host:
  - `tables`, `kinds`, `json_step`: the highlighter's tables, built from the same transition functions, and its two stages.
  - `regex`: the pattern language and its compilation (`pattern.ts`), known bits (`bits.ts`), and the pattern as cells (`data-program.ts`). `search_step`: the search's three stages, masked or not.
  - `loop_program`: loop programs, stages, `describe_loop` and the CPU runner on the executor.
  - `emit`, `loop_emit`: a program as rows, and the WGSL shader of a loop program (`emitLoopWgsl`), statement for statement.
  - `program`: a document's cells. `table_loop`: the plain table-loop highlighter.

Two small additions to the runtime serve them: `Circuit::reglu_preserving`, the TypeScript runtime's `append('reglu', ..., {preserveRounding: true})` that Photos' zoom uses, and `fp::js_round`, JavaScript's `Math.round`.

## Conformance

[`conformance/generate.ts`](conformance/generate.ts) builds 32 scenarios with the TypeScript runtime and writes `conformance/fixtures.json`. [`tests/conformance.rs`](tests/conformance.rs) builds the same scenarios in Rust and compares, bit for bit:

- every operation, range, grid, exact-bit declaration, one-hot group and accepted index range
- the reference evaluator's values on seeded inputs, including one-FP32-step pairs, zeros of both signs, subnormals, infinities and broken one-hot contracts
- gradients, and the error a rejected backward pass reports
- four compiled programs per scenario (gathers lowered, not lowered, recomputed depths, and only the scenario's roots): every layer's inputs, weights, gate, boolean, gather and index read rows, and the hash of every tensor evaluation
- every validator's verdict and error message, and the roots a tensor evaluator returns

The gather scenarios follow `tests/runtime/gather-lowering.test.ts`: sums that must lower (with a bias, a signed pair, the bit in the gate, a gather read by another gather, argmax and first-active selections) and sums that must stay dense (scaled, missing or extra terms, two groups, a bit-in-left product over a signed value, unbounded values, an unused negative arm, fractional first-active selections).

The generator also builds and drives the demo circuits with the TypeScript originals ([`conformance/demos.ts`](conformance/demos.ts)) and writes `conformance/demos.json`, which [`tests/demos.rs`](tests/demos.rs) replays in Rust, see [Demo parity](#demo-parity).

```bash
node crates/handwired/conformance/generate.ts
```

```bash
cargo test
```

Plain Node 23.6 or later runs the generator directly, in about 40 s. Regenerate the fixtures after changing a scenario or pulling upstream changes. The workspace compiles tests with `opt-level = 2`, since the demo tests run thousands of evaluations; `cargo test` takes about 10 s after the build.

## Demo parity

Every demo circuit below is built in Rust and compared, bit for bit, with the same circuit built by the TypeScript original. For each circuit: every operation (kind, terms, biases, depth, macro, index read, rounding point, comparison bit), the ranges, grids, exact bits, one-hot groups and rounding points, `validateCircuit`; for each compiled program: every layer's arrays, the roots, `wireOrder`, the value count and `validateTensorProgram`. Operations and layers are compared through one hash each, so a failure names the first operation or layer that differs, and `wireOrder` and the one-hot groups are compared in full.

| Circuit | Size | Driven through |
| --- | --- | --- |
| HandwiredOS's window manager | 234 cells, 5,028 operations, 4,646 live in 130 layers, 9,909 weights | 428 frames, 591 evaluations |
| Photos alone | 98 cells, 2,189 operations, 2,026 live in 67 layers | 137 frames, 175 evaluations |
| Monitor's list alone | 21 cells, 800 operations, 733 live in 67 layers | 23 evaluations with write-back |
| JSON highlighter | 37 operations per position, 17 at the end, 18,961 per document | 84 texts |
| Regex search, 4 configurations | 8, 16 and 32 states, masked or not, stopping at the text's end or not: 122,888 to 1,167,392 operations per document | 117 pattern and scope runs over 8 to 24 documents |

What matches, all bit for bit:

- **The window manager's trace**: the windows arranged for a 1280 × 800 viewport, then idle hover moves that coalesce, a title-bar drag in coalesced samples, a resize by the grip, maximize and restore mid-animation, minimize to the dock and back, Photos' grid, lightbox, next and previous keys and close, a photo opened while another returns, Monitor's list keys (arrows, page keys, Home, End), scroll positions, sidebar locations and row clicks, viewports of a tablet, a phone upright and sideways and back, a close with a 2 s stall that snaps, a reopen from the dock, a drag ended by a pointercancel, a press released off its button, events without timestamps, and a resize event between frames. Per frame: every planned evaluation (kind, time, steps, write-back, events), the hash of the values and of the table after each evaluation, the window outputs in full, the values, the stored table, the packed values, the spring clock and the scroll request.
- **Photos alone and Monitor's list alone**, on their own traces, including present and interactive changes, a resize during a zoom and a viewport of 33 × 339 px; the list over keys, clicks, locations, viewport heights and scroll positions, with the reference evaluator on every table.
- **The page's own numbers**: the windows' first rectangles at 408 workspace sizes, Monitor's footer text over 90 frames, the program's size in bytes, Monitor's Operations and State rows, the named outputs, the wallpaper's 9,909 arcs at two canvas sizes, `noteFocusChange` for all 64 inputs and `formatValue`.
- **JSON search**: the 12 highlighter tables, the highlighter's summary and display kinds on 84 texts (generated documents with defects, every error kind, the 511 to 513 code point limits, nesting 16 and 17 deep, surrogate pairs and a lone surrogate), the table-loop baseline on the same texts, 41 patterns' automata, character classes, capacity errors and syntax errors with positions, and the search's match counts and per-position marks in every scope. The WGSL shader of every loop program, in use order and in layer order, matches statement for statement (compared by hash).
- **The runtime helpers**: both prefix networks on 0 to 19 items, spring step matrices and every power up to 300 steps for 4 springs, their bounds, frame timing, step counts, `advance_frame` over a 16-frame script under both write-back policies, and a source table with every kind of cell: names, policies, the initial table, the validator's message for 12 broken tables, and `TableEvaluator` over 4 frames.

The tests were checked with deliberate mutations, e.g. Photos' fade rate, a WGSL loop header, one tokenizer transition, the pointer bound, a preserved sum in Monitor's list, the arrangement's gap, the settle check boundary and the class parser's merge of adjacent intervals: each one fails a test.

What isn't ported, and why:

- **The DOM, rendering and listeners** of both demos: the renderers of Photos and Monitor's list, `projectWindowElements`, `mountWindowManager`, the footer's DOM writes and the wallpaper's canvas and redraw timing. Their numbers are ported (`MonitorFooter`, `wiring_arcs`); the operation kinds' sentences and CSS classes are not.
- **`bit-emit.ts`**, the second WGSL writer, which writes each operation in the cheapest exact form (bits, i32, packed words). It is an optimization of the same loop programs; `loop-emit.ts`'s shader is ported and matches.
- **`generated-js.ts`** and the loop runner's `'generated'` backend, which write the program as JavaScript. The Rust runner uses the executor, the repo's definition of a program's values.
- **`searcher.ts`, `gpu/host.ts`, `demo*.ts`, `demo-worker.ts`**: the WebGPU host, the page and its workers. Running the shaders is step 3.
- **`plain-search.ts`**, the RegExp baseline, which needs JavaScript's RegExp semantics.
- **`documents.ts`**, the document generator. The test texts are generated by it and recorded in the fixture; porting it needs `JSON.stringify`'s output and UTF-16 strings.
- `format_value` searches the shortest decimal with Rust's formatting, which rounds a tie to even where `toPrecision` rounds it up; the two agree on the values checked.

## Toward demo parity

1. **Done: the runtime.** Every builder function, the tensor compiler and the validators match the TypeScript runtime.
2. **Done: the demo circuits, run without a UI.** HandwiredOS's window manager, Photos and Monitor's list, and JSON search's highlighter and regex search, with the runtime helpers they use, match the TypeScript originals on scripted traces, see [Demo parity](#demo-parity).
3. **GPU backend**: a `wgpu` executor for the compiled program, checked against the reference evaluator at the rounding points. The loop programs' WGSL from `json_search::loop_emit` is ready to run.
