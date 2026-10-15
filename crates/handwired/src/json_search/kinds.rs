// Names and numbers shared by the highlighter and the code that shows its
// outputs, a port of src/json-search/highlighter/kinds.ts.

/// The program reads up to this many code points of a text.
pub const CAPACITY: usize = 512;
/// Containers nest up to this depth.
pub const MAXIMUM_DEPTH: usize = 16;

/// What a character is shown as.
pub const DISPLAY_KINDS: [&str; 9] =
    ["whitespace", "punctuation", "string", "escape", "number", "literal", "error", "pastEnd", "key"];

pub mod display_kind {
    pub const WHITESPACE: usize = 0;
    pub const PUNCTUATION: usize = 1;
    pub const STRING: usize = 2;
    pub const ESCAPE: usize = 3;
    pub const NUMBER: usize = 4;
    pub const LITERAL: usize = 5;
    pub const ERROR: usize = 6;
    pub const PAST_END: usize = 7;
    pub const KEY: usize = 8;
}

/// The kinds of the first error.
pub const ERROR_KINDS: [&str; 16] = [
    "none",
    "unexpectedCharacter",
    "controlCharacterInString",
    "badEscape",
    "badUnicodeEscape",
    "badNumber",
    "badLiteral",
    "expectedValue",
    "expectedKey",
    "expectedColon",
    "expectedCommaOrClose",
    "mismatchedBracket",
    "extraAfterValue",
    "tooDeep",
    "unexpectedEnd",
    "tooLong",
];

pub fn error_kind(name: &str) -> usize {
    ERROR_KINDS.iter().position(|&kind| kind == name).unwrap_or_else(|| panic!("no error kind {name}"))
}
