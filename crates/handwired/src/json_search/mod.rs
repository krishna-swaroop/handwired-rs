//! JSON search's two programs without a page: the JSON highlighter and the
//! regex search, as loop programs over a document's positions, with their
//! tables, the pattern compiler, the CPU runner, the WGSL writer of
//! loop-emit.ts and the plain table-loop baseline. A port of src/json-search's
//! core; the page, the workers, the GPU host, bit-emit.ts, generated-js.ts and
//! the document generator are left out.

pub mod emit;
pub mod json_step;
pub mod kinds;
pub mod loop_emit;
pub mod loop_program;
pub mod program;
pub mod regex;
pub mod search_step;
pub mod table_loop;
pub mod tables;
