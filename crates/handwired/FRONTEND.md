# A Rust frontend: plain Rust to a handwired network

Status, October 16, 2026: M0 is built: `crates/handwired-macros`, `src/domain.rs` and `tests/frontend.rs`. Everything after M0 is design.

Today every network in this crate is written against the builder by hand: `emit_compare` for a `>`, `emit_select` for an `if`, `index_read` for a table lookup, with ranges declared per wire. Upstream works the same way. The frontend is the missing step: write a step function in ordinary Rust, get the circuit automatically, and check that the circuit computes what the Rust computes.

## Prior art: compiling to categories

Conal Elliott's [Compiling to categories](http://conal.net/papers/compiling-to-categories/) (ICFP 2017) does this for Haskell. A GHC plugin translates a typed function into a small vocabulary, the operations of a cartesian closed category: identity, composition, the projections `exl`/`exr`, pairing (`fork`, `△`), `apply`, `curry`, constants, plus primitive classes for numbers, booleans and conditionals. Each *interpretation* is then just an instance of those classes, written outside the compiler: ordinary functions, hardware circuits, automatic differentiation, interval analysis, incremental computation. One source program, many meanings.

What we take from it:

1. **Translate once, interpret many times.** The macro does not emit circuits. It emits the program against an abstract vocabulary, and the circuit is one interpretation among several.
2. **Two of Elliott's example interpretations are our two hard problems.** Interval analysis is our range inference, and automatic differentiation is `backpropagate_reference`. The circuit builder already tracks a range per wire, so for us the interval interpretation comes nearly free with the circuit one.
3. **Primitives are a closed list.** Anything outside the vocabulary is a compile error at the macro, not a silent fallback.

What we don't take:

- **The closed part (`curry`, `apply`).** Step functions are first-order: no closures, no higher-order functions. A cartesian category is enough, and a first-order program is already a dataflow graph, so the macro can stay close to the source instead of producing point-free combinators.
- **The plugin mechanism.** Rust has no stable compiler plugins. A proc macro over `syn` sees the function's syntax, which for a first-order subset is all the translation needs. A MIR pass could come later if the subset outgrows syntax.

## The shape: a step function and a domain trait

The unit of compilation is a step: `fn step(state, input) -> state` plus outputs, which is what HandwiredOS's machines and json-search's loop stages already are. The macro rewrites the body into a function generic over a `Domain`, in the tagless-final style, which is a shallow embedding of the same category vocabulary:

```rust
#[handwired::step]
fn above(#[range(0, 1000)] a: i32) -> i32 {
    if a > 4 { 10 } else { 12 }
}
```

The macro keeps `above` as written, and adds:

```rust
fn above_in<D: Domain>(d: &mut D, a: D::Value) -> D::Value {
    let c0 = d.constant(4.0);
    let t0 = d.compare(a, Relation::Greater, c0);
    let c1 = d.constant(10.0);
    let c2 = d.constant(12.0);
    d.select(t0, c2, c1)
}
```

Domains:

| Domain | `Value` | What it is for |
|---|---|---|
| native | `f64` | the macro's own semantics, checked against the original Rust function |
| circuit | `Wire` | the network: emits through `Circuit`, `emit_compare`, `emit_select`, `index_read` |
| interval (later) | `Range` | range errors before building, e.g. an index that can leave its table |
| graph (later) | node id | a picture of the program, like Elliott's graph category |

The native domain is a quiet but important member: comparing `above(a)` with `above_in(&mut Native, a)` over inputs tests the macro, and comparing either with the compiled tensor program tests the lowering. Those are separate failures with separate fixes.

## The subset

| Rust | Vocabulary | Circuit lowering |
|---|---|---|
| `let`, `let mut` with reassignment | SSA values | none, wires are values |
| `+`, `-`, `*` by a constant, unary `-` | affine | `affine` / `round_affine` |
| `*` of two values | multiply | `multiply` macro |
| `<`, `<=`, `>`, `>=`, `==` | compare | `emit_compare`, integer bits on grids |
| `&&`, `\|\|`, `!` on bools | bit arithmetic | `multiply`, affine |
| `if c { a } else { b }` | select | `emit_select`; both branches are evaluated, so branches must be pure |
| `match` on an integer or fieldless enum | one-hot select | compare per arm, `assume_one_hot`, a select tree |
| `TABLE[i]` on a `const` array | index read | `index_read` with `OutOfRange` from the declared range |
| `min`, `max`, `clamp`, `abs` | primitives | `emit_minimum_exact`, `emit_maximum_exact`, `emit_clamp_exact` |
| `for i in 0..N` with constant `N` | unroll | none |
| `while` / `loop` | a loop stage | a `LoopProgram` stage with `Repeat::Forward` and a declared maximum, plus a halt bit |
| struct of numbers | product | a bundle of wires, like `SpringWires` |

Rejected with a pointed error: division by a non-constant, floats with unbounded ranges, recursion, heap allocation, references into dynamic arrays, function pointers, unbounded loops, bit operations on values (see below), and calls to functions that aren't themselves `#[handwired::step]` or a primitive.

Ranges come from `#[range(lo, hi)]` on inputs and state fields and are propagated by the circuit builder. Every intermediate must stay in FP32's exact integer range, 2^24, or be on a declared grid; the macro reports the first operation that leaves it, with its Rust source span.

## What the frontend will not do: change representations

The plain highlighter (`json_search::table_loop`) keeps the open containers as a bit stack, `container_bits`, and reads the top with `(container_bits >> (depth - 1)) & 1`. The hand-wired highlighter (`json_search::json_step`) has no bit stack. It keeps a number, `code`, adds a table-driven `object_weight` per push and pop, and asks whether the top container is an object with one comparison, `code >= level_weight[depth]`. That is a change of representation that relies on knowing the stack discipline, not a lowering of the shift, and no syntax-directed compiler will find it.

So the contract is: the frontend compiles the program as written, efficiently and exactly, and rejects what it can't lower. Getting a small network is the programmer's job, done in Rust, by writing the step in a network-friendly form: tables instead of branches on many states, one comparison instead of a bit extraction. That is the same division of labour as Elliott's, where the category instance is fixed and the source is what you change.

## Milestones

Each milestone ends with a differential test: the original Rust, the native domain, and the compiled tensor program agree bit for bit on the same inputs.

- **M0, spike (done).** A `handwired-macros` crate: `#[handwired::step]` on the README's `if a > 4 { 10 } else { 12 }`, with the native and circuit domains. The generated circuit has exactly the hand-written circuit's operations, from `lib.rs`'s doc example. A second function with `let mut`, `&&`, `||`, `!`, a product of two variables and nested `if`s agrees four ways on all 4,182 inputs in its ranges.
- **M1, scalar subset.** Everything in the table above except loops. Tests on small programs, e.g. clamp, argmax over a fixed list, a lookup into a `const` table.
- **M2, the highlighter.** A network-friendly plain step of the highlighter (`code` and tables, as described above), compiled into a `LoopProgram`, checked against `TableLoop::highlight` on the 84 conformance texts, and compared in size and speed with `build_json_loop`.
- **M3, search.** The regex automaton step from `search_step`, in plain Rust, against `json_search_matches`'s fixtures.
- **M4, a HandwiredOS piece.** A window hit test or the value list's scroll logic, plain, against the hand-wired emitters.
- **Then, the GPU executor**, now that every frontend program is a `TensorProgram`.

## Open questions

- **How much of the step's state should be a struct the macro understands**, versus registers declared by hand like `json_step`'s `REGISTER_NAMES`?
- **Should the interval domain exist at all,** given that the circuit builder already tracks ranges, or only as a way to report errors with source spans before building?
- **Grids for non-integers.** Springs and positions are on grids or are rounding points. A `#[grid(...)]` attribute is the obvious form; it needs a real example first.
