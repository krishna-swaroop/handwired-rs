// A document's cells, a port of src/json-search/program.ts: the first 512 code
// points, -1 past the text, and the text's length in code points.

use super::kinds::CAPACITY;

pub const CELLS_PER_DOCUMENT: usize = CAPACITY + 1;
const LENGTH_LIMIT: usize = 1_048_576;

/// Writes one document's cells at `offset` from the text's UTF-16 code units.
/// A surrogate pair is one code point, and a lone surrogate is its own code point.
pub fn write_document_cells(cells: &mut [f32], offset: usize, text: &[u16]) {
    let unit_count = text.len();
    let mut length = 0;
    let mut index = 0;
    while index < unit_count {
        let mut code_point = text[index] as u32;
        if (0xd800..=0xdbff).contains(&code_point) && index + 1 < unit_count {
            let low = text[index + 1] as u32;
            if (0xdc00..=0xdfff).contains(&low) {
                code_point = (code_point - 0xd800) * 0x400 + (low - 0xdc00) + 0x10000;
                index += 1;
            }
        }
        if length < CAPACITY {
            cells[offset + length] = code_point as f32;
        }
        length += 1;
        index += 1;
    }
    for position in length.min(CAPACITY)..CAPACITY {
        cells[offset + position] = -1.0;
    }
    cells[offset + CAPACITY] = length.min(LENGTH_LIMIT) as f32;
}

/// The code units of a Rust string, for `write_document_cells`.
pub fn utf16(text: &str) -> Vec<u16> {
    text.encode_utf16().collect()
}
