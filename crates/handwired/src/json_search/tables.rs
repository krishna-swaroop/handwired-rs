// The highlighter's tables, a port of src/json-search/highlighter/tables.ts:
// a character class per code point, the tokenizer's transition and event
// tables, small tables per event, and the grammar's transition and error
// tables. Each comes from the same transition functions, with the same names.

use super::kinds::{display_kind, error_kind, MAXIMUM_DEPTH};

pub const CHARACTER_CLASSES: [&str; 33] = [
    "end", "truncatedEnd", "space", "tabOrCarriageReturn", "lineFeed", "control", "quote", "backslash", "slash",
    "openBrace", "closeBrace", "openBracket", "closeBracket", "colon", "comma", "minus", "plus", "dot", "zero", "digit",
    "a", "b", "e", "capitalE", "f", "l", "n", "r", "s", "t", "u", "otherHexLetter", "other",
];
pub const CLASS_COUNT: usize = CHARACTER_CLASSES.len();

pub const LEXER_STATES: [&str; 26] = [
    "between", "string", "escape", "hex4", "hex3", "hex2", "hex1", "minus", "zero", "integer", "dot", "fraction",
    "exponent", "exponentSign", "exponentDigits", "t", "tr", "tru", "f", "fa", "fal", "fals", "n", "nu", "nul", "errorLine",
];
pub const LEXER_STATE_COUNT: usize = LEXER_STATES.len();

pub const LEXER_EVENTS: [&str; 25] = [
    "whitespace", "stringStart", "stringContent", "stringEnd", "escape", "numberStart", "numberContent", "literalStart",
    "literalContent", "openObject", "openArray", "closeObject", "closeArray", "colon", "comma", "unexpectedCharacter",
    "controlCharacterInString", "badEscape", "badUnicodeEscape", "badNumber", "badLiteral", "errorContinuation", "end",
    "endInsideToken", "truncatedEnd",
];
pub const EVENT_COUNT: usize = LEXER_EVENTS.len();

pub const GRAMMAR_TOKENS: [&str; 11] =
    ["none", "string", "scalar", "openObject", "openArray", "closeObject", "closeArray", "colon", "comma", "lexerError", "end"];

pub const GRAMMAR_STATES: [&str; 8] = ["value", "valueOrClose", "keyOrClose", "key", "colon", "commaOrClose", "end", "error"];
pub const GRAMMAR_STATE_COUNT: usize = GRAMMAR_STATES.len();
const DEPTH_CLASS_COUNT: usize = 4;
pub const CONTEXT_COUNT: usize = DEPTH_CLASS_COUNT * 2;
pub const SYMBOL_COUNT: usize = GRAMMAR_TOKENS.len() * CONTEXT_COUNT;

fn index_of(names: &[&str], name: &str) -> usize {
    names.iter().position(|&candidate| candidate == name).unwrap_or_else(|| panic!("{name} is not listed"))
}

fn class_of_code_point(code_point: i64) -> &'static str {
    if code_point == -1 {
        return "end";
    }
    if code_point == 0x20 {
        return "space";
    }
    if code_point == 0x09 || code_point == 0x0d {
        return "tabOrCarriageReturn";
    }
    if code_point == 0x0a {
        return "lineFeed";
    }
    if code_point < 0x20 {
        return "control";
    }
    if (0x30..=0x39).contains(&code_point) {
        return if code_point == 0x30 { "zero" } else { "digit" };
    }
    let character = char::from_u32(code_point as u32).unwrap();
    match character {
        '"' => return "quote",
        '\\' => return "backslash",
        '/' => return "slash",
        '{' => return "openBrace",
        '}' => return "closeBrace",
        '[' => return "openBracket",
        ']' => return "closeBracket",
        ':' => return "colon",
        ',' => return "comma",
        '-' => return "minus",
        '+' => return "plus",
        '.' => return "dot",
        'E' => return "capitalE",
        _ => {}
    }
    if let Some(index) = "abeflnrstu".find(character) {
        return CHARACTER_CLASSES[index_of(&CHARACTER_CLASSES, &"abeflnrstu"[index..index + 1])];
    }
    if "cdABCDF".contains(character) {
        return "otherHexLetter";
    }
    "other"
}

pub fn class_table() -> Vec<f64> {
    (0..129).map(|entry| index_of(&CHARACTER_CLASSES, class_of_code_point(entry as i64 - 1)) as f64).collect()
}

type Step = (&'static str, &'static str);

fn is_digit(class: &str) -> bool {
    class == "zero" || class == "digit"
}

fn is_hex_digit(class: &str) -> bool {
    is_digit(class) || ["a", "b", "e", "capitalE", "f", "otherHexLetter"].contains(&class)
}

fn lexer_error(class: &str, event: &'static str) -> Step {
    (if class == "lineFeed" { "between" } else { "errorLine" }, event)
}

fn start_token(class: &str) -> Step {
    match class {
        "space" | "tabOrCarriageReturn" | "lineFeed" => ("between", "whitespace"),
        "quote" => ("string", "stringStart"),
        "openBrace" => ("between", "openObject"),
        "closeBrace" => ("between", "closeObject"),
        "openBracket" => ("between", "openArray"),
        "closeBracket" => ("between", "closeArray"),
        "colon" => ("between", "colon"),
        "comma" => ("between", "comma"),
        "minus" => ("minus", "numberStart"),
        "zero" => ("zero", "numberStart"),
        "digit" => ("integer", "numberStart"),
        "t" => ("t", "literalStart"),
        "f" => ("f", "literalStart"),
        "n" => ("n", "literalStart"),
        "end" => ("between", "end"),
        "truncatedEnd" => ("between", "truncatedEnd"),
        _ => lexer_error(class, "unexpectedCharacter"),
    }
}

fn continue_or(class: &str, continued: Option<Step>, error: &'static str) -> Step {
    if let Some(step) = continued {
        return step;
    }
    if class == "end" {
        return ("between", "endInsideToken");
    }
    if class == "truncatedEnd" {
        return ("between", "truncatedEnd");
    }
    lexer_error(class, error)
}

fn literal_letter(class: &str, expected: &str, next: &'static str) -> Step {
    continue_or(class, (class == expected).then_some((next, "literalContent")), "badLiteral")
}

fn lexer_step(state: &str, class: &str) -> Step {
    let digit = is_digit(class);
    let exponent_letter = class == "e" || class == "capitalE";
    match state {
        "between" => start_token(class),
        "string" => {
            if class == "quote" {
                return ("between", "stringEnd");
            }
            if class == "backslash" {
                return ("escape", "escape");
            }
            let is_control = class == "control" || class == "tabOrCarriageReturn" || class == "lineFeed";
            let blocked = is_control || class == "end" || class == "truncatedEnd";
            continue_or(class, (!blocked).then_some(("string", "stringContent")), "controlCharacterInString")
        }
        "escape" => {
            let simple = ["quote", "backslash", "slash", "b", "f", "n", "r", "t"].contains(&class);
            let continued = if simple {
                Some(("string", "escape"))
            } else if class == "u" {
                Some(("hex4", "escape"))
            } else {
                None
            };
            continue_or(class, continued, "badEscape")
        }
        "hex4" | "hex3" | "hex2" | "hex1" => {
            let next = match state {
                "hex4" => "hex3",
                "hex3" => "hex2",
                "hex2" => "hex1",
                _ => "string",
            };
            continue_or(class, is_hex_digit(class).then_some((next, "escape")), "badUnicodeEscape")
        }
        "minus" => {
            let continued = if class == "zero" {
                Some(("zero", "numberContent"))
            } else if class == "digit" {
                Some(("integer", "numberContent"))
            } else {
                None
            };
            continue_or(class, continued, "badNumber")
        }
        "zero" => {
            if digit {
                return lexer_error(class, "badNumber");
            }
            if class == "dot" {
                return ("dot", "numberContent");
            }
            if exponent_letter {
                return ("exponent", "numberContent");
            }
            start_token(class)
        }
        "integer" => {
            if digit {
                return ("integer", "numberContent");
            }
            if class == "dot" {
                return ("dot", "numberContent");
            }
            if exponent_letter {
                return ("exponent", "numberContent");
            }
            start_token(class)
        }
        "dot" => continue_or(class, digit.then_some(("fraction", "numberContent")), "badNumber"),
        "fraction" => {
            if digit {
                return ("fraction", "numberContent");
            }
            if exponent_letter {
                return ("exponent", "numberContent");
            }
            start_token(class)
        }
        "exponent" => {
            let sign = class == "plus" || class == "minus";
            let continued = if sign {
                Some(("exponentSign", "numberContent"))
            } else if digit {
                Some(("exponentDigits", "numberContent"))
            } else {
                None
            };
            continue_or(class, continued, "badNumber")
        }
        "exponentSign" => continue_or(class, digit.then_some(("exponentDigits", "numberContent")), "badNumber"),
        "exponentDigits" => {
            if digit {
                ("exponentDigits", "numberContent")
            } else {
                start_token(class)
            }
        }
        "t" => literal_letter(class, "r", "tr"),
        "tr" => literal_letter(class, "u", "tru"),
        "tru" => literal_letter(class, "e", "between"),
        "f" => literal_letter(class, "a", "fa"),
        "fa" => literal_letter(class, "l", "fal"),
        "fal" => literal_letter(class, "s", "fals"),
        "fals" => literal_letter(class, "e", "between"),
        "n" => literal_letter(class, "u", "nu"),
        "nu" => literal_letter(class, "l", "nul"),
        "nul" => literal_letter(class, "l", "between"),
        "errorLine" => {
            if class == "lineFeed" {
                return ("between", "whitespace");
            }
            if class == "end" {
                return ("between", "end");
            }
            if class == "truncatedEnd" {
                return ("between", "truncatedEnd");
            }
            ("errorLine", "errorContinuation")
        }
        _ => unreachable!("no lexer state {state}"),
    }
}

pub fn lexer_next_table() -> Vec<f64> {
    LEXER_STATES
        .iter()
        .flat_map(|&state| CHARACTER_CLASSES.iter().map(move |&class| index_of(&LEXER_STATES, lexer_step(state, class).0) as f64))
        .collect()
}

pub fn lexer_event_table() -> Vec<f64> {
    LEXER_STATES
        .iter()
        .flat_map(|&state| CHARACTER_CLASSES.iter().map(move |&class| index_of(&LEXER_EVENTS, lexer_step(state, class).1) as f64))
        .collect()
}

fn grammar_token_of(event: &str) -> &'static str {
    match event {
        "whitespace" | "stringContent" | "stringEnd" | "escape" | "numberContent" | "literalContent" => "none",
        "stringStart" => "string",
        "numberStart" | "literalStart" => "scalar",
        "openObject" => "openObject",
        "openArray" => "openArray",
        "closeObject" => "closeObject",
        "closeArray" => "closeArray",
        "colon" => "colon",
        "comma" => "comma",
        "end" => "end",
        _ => "lexerError",
    }
}

fn lexer_error_kind_of(event: &str) -> usize {
    match event {
        "unexpectedCharacter" | "controlCharacterInString" | "badEscape" | "badUnicodeEscape" | "badNumber"
        | "badLiteral" => error_kind(event),
        "endInsideToken" => error_kind("unexpectedEnd"),
        "truncatedEnd" => error_kind("tooLong"),
        _ => error_kind("none"),
    }
}

pub fn grammar_token_table() -> Vec<f64> {
    LEXER_EVENTS.iter().map(|&event| index_of(&GRAMMAR_TOKENS, grammar_token_of(event)) as f64).collect()
}

pub fn lexer_error_kind_table() -> Vec<f64> {
    LEXER_EVENTS.iter().map(|&event| lexer_error_kind_of(event) as f64).collect()
}

pub fn depth_change_table() -> Vec<f64> {
    LEXER_EVENTS
        .iter()
        .map(|&event| match event {
            "openObject" | "openArray" => 1.0,
            "closeObject" | "closeArray" => -1.0,
            _ => 0.0,
        })
        .collect()
}

pub fn object_weight_table() -> Vec<f64> {
    (0..(MAXIMUM_DEPTH + 1) * EVENT_COUNT)
        .map(|entry| {
            let depth_before = entry / EVENT_COUNT;
            let event = LEXER_EVENTS[entry % EVENT_COUNT];
            if event == "openObject" && depth_before < MAXIMUM_DEPTH {
                return 2f64.powi(depth_before as i32);
            }
            if event == "closeObject" && depth_before >= 1 {
                return -(2f64.powi(depth_before as i32 - 1));
            }
            0.0
        })
        .collect()
}

pub fn level_weight_table() -> Vec<f64> {
    (0..MAXIMUM_DEPTH).map(|level| 2f64.powi(level as i32)).collect()
}

pub fn depth_class_table() -> Vec<f64> {
    (0..=MAXIMUM_DEPTH)
        .map(|depth| match depth {
            0 => 0.0,
            1 => 1.0,
            _ if depth == MAXIMUM_DEPTH => 3.0,
            _ => 2.0,
        })
        .collect()
}

fn grammar_step(state: &str, token: &str, depth_class: usize, in_object: bool) -> (&'static str, usize) {
    let none = error_kind("none");
    if state == "error" {
        return ("error", none);
    }
    if token == "none" {
        return (GRAMMAR_STATES[index_of(&GRAMMAR_STATES, state)], none);
    }
    if token == "lexerError" {
        return ("error", none);
    }
    if token == "end" {
        return if state == "end" { ("end", none) } else { ("error", error_kind("unexpectedEnd")) };
    }
    let unexpected = || -> (&'static str, usize) {
        match state {
            "value" | "valueOrClose" => ("error", error_kind("expectedValue")),
            "keyOrClose" | "key" => ("error", error_kind("expectedKey")),
            "colon" => ("error", error_kind("expectedColon")),
            "commaOrClose" => ("error", error_kind("expectedCommaOrClose")),
            _ => ("error", error_kind("extraAfterValue")),
        }
    };
    let expects_value = state == "value" || state == "valueOrClose";
    let after_value = (if depth_class == 0 { "end" } else { "commaOrClose" }, none);
    match token {
        "string" => {
            if state == "keyOrClose" || state == "key" {
                return ("colon", none);
            }
            if expects_value { after_value } else { unexpected() }
        }
        "scalar" => {
            if expects_value {
                after_value
            } else {
                unexpected()
            }
        }
        "openObject" | "openArray" => {
            if !expects_value {
                return unexpected();
            }
            if depth_class == 3 {
                return ("error", error_kind("tooDeep"));
            }
            (if token == "openObject" { "keyOrClose" } else { "valueOrClose" }, none)
        }
        "closeObject" | "closeArray" => {
            let closes_object = token == "closeObject";
            let may_close = if closes_object {
                state == "keyOrClose" || state == "commaOrClose"
            } else {
                state == "valueOrClose" || state == "commaOrClose"
            };
            let other_kind_may_close = if closes_object {
                state == "valueOrClose" || state == "commaOrClose"
            } else {
                state == "keyOrClose" || state == "commaOrClose"
            };
            if may_close && in_object == closes_object {
                return (if depth_class == 1 { "end" } else { "commaOrClose" }, none);
            }
            if other_kind_may_close {
                return ("error", error_kind("mismatchedBracket"));
            }
            unexpected()
        }
        "colon" => {
            if state == "colon" {
                ("value", none)
            } else {
                unexpected()
            }
        }
        "comma" => {
            if state == "commaOrClose" {
                (if in_object { "key" } else { "value" }, none)
            } else {
                unexpected()
            }
        }
        _ => unreachable!("no grammar token {token}"),
    }
}

fn grammar_table(field: usize) -> Vec<f64> {
    let mut table = Vec::new();
    for &state in &GRAMMAR_STATES {
        for &token in &GRAMMAR_TOKENS {
            for context in 0..CONTEXT_COUNT {
                let step = grammar_step(state, token, context / 2, context % 2 == 1);
                table.push(if field == 0 { index_of(&GRAMMAR_STATES, step.0) as f64 } else { step.1 as f64 });
            }
        }
    }
    table
}

pub fn grammar_next_table() -> Vec<f64> {
    grammar_table(0)
}

pub fn grammar_error_table() -> Vec<f64> {
    grammar_table(1)
}

fn display_kind_of(event: &str, grammar_after: &str) -> usize {
    match event {
        "whitespace" => display_kind::WHITESPACE,
        "stringStart" | "stringContent" | "stringEnd" => {
            if grammar_after == "colon" {
                display_kind::KEY
            } else {
                display_kind::STRING
            }
        }
        "escape" => display_kind::ESCAPE,
        "numberStart" | "numberContent" => display_kind::NUMBER,
        "literalStart" | "literalContent" => display_kind::LITERAL,
        "openObject" | "openArray" | "closeObject" | "closeArray" | "colon" | "comma" => display_kind::PUNCTUATION,
        "end" | "endInsideToken" | "truncatedEnd" => display_kind::PAST_END,
        _ => display_kind::ERROR,
    }
}

pub fn display_table() -> Vec<f64> {
    LEXER_EVENTS
        .iter()
        .flat_map(|&event| GRAMMAR_STATES.iter().map(move |&state| display_kind_of(event, state) as f64))
        .collect()
}

/// Every table by name, in json-step.ts's declaration order.
pub fn all_tables() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("classes", class_table()),
        ("lexerNext", lexer_next_table()),
        ("lexerEvent", lexer_event_table()),
        ("grammarToken", grammar_token_table()),
        ("lexerErrorKind", lexer_error_kind_table()),
        ("depthChange", depth_change_table()),
        ("objectWeight", object_weight_table()),
        ("levelWeight", level_weight_table()),
        ("depthClass", depth_class_table()),
        ("grammarNext", grammar_next_table()),
        ("grammarError", grammar_error_table()),
        ("display", display_table()),
    ]
}
