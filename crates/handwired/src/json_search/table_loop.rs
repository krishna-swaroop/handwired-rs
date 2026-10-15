// The highlighter as one plain loop over the program's own tables, a port of
// src/json-search/table-loop.ts: the fastest plain baseline for a display kind
// per character and the first error.

use super::kinds::{display_kind, CAPACITY};
use super::tables::{
    class_table, depth_change_table, depth_class_table, display_table, grammar_error_table, grammar_next_table,
    grammar_token_table, lexer_error_kind_table, lexer_event_table, lexer_next_table, object_weight_table, CLASS_COUNT,
    CONTEXT_COUNT, EVENT_COUNT, GRAMMAR_STATES, GRAMMAR_STATE_COUNT, SYMBOL_COUNT,
};

/// tableLoopResult: firstErrorPosition is -1 for a valid text.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TableLoopResult {
    pub first_error_position: i64,
    pub first_error_kind: usize,
    pub truncated: usize,
}

/// The tables as small integers, built once.
pub struct TableLoop {
    classes: Vec<usize>,
    lexer_next: Vec<usize>,
    lexer_event: Vec<usize>,
    grammar_token: Vec<usize>,
    lexer_error_kind: Vec<usize>,
    depth_change: Vec<i64>,
    depth_class: Vec<usize>,
    grammar_next: Vec<usize>,
    grammar_error: Vec<usize>,
    display: Vec<usize>,
    error_state: usize,
    open_object_event: usize,
}

fn small(values: Vec<f64>) -> Vec<usize> {
    values.into_iter().map(|value| value as usize).collect()
}

impl Default for TableLoop {
    fn default() -> TableLoop {
        TableLoop::new()
    }
}

impl TableLoop {
    pub fn new() -> TableLoop {
        let object_weight = object_weight_table();
        let open_object_event = object_weight[..EVENT_COUNT].iter().position(|&weight| weight == 1.0).expect("the tables have the form this loop expects");
        TableLoop {
            classes: small(class_table()),
            lexer_next: small(lexer_next_table()),
            lexer_event: small(lexer_event_table()),
            grammar_token: small(grammar_token_table()),
            lexer_error_kind: small(lexer_error_kind_table()),
            depth_change: depth_change_table().into_iter().map(|value| value as i64).collect(),
            depth_class: small(depth_class_table()),
            grammar_next: small(grammar_next_table()),
            grammar_error: small(grammar_error_table()),
            display: small(display_table()),
            error_state: GRAMMAR_STATES.iter().position(|&state| state == "error").unwrap(),
            open_object_event,
        }
    }

    /// Writes the display kinds of the text's first 512 code points into `kinds`.
    pub fn highlight(&self, text: &[u16], kinds: &mut [u8]) -> TableLoopResult {
        let other_class = self.classes[128];
        let unit_count = text.len();
        let mut unit = 0;
        let mut lexer_state = 0;
        let mut grammar_state = 0;
        let mut depth: i64 = 0;
        let mut container_bits: u32 = 0;
        let mut first_error_position = -1;
        let mut first_error_kind = 0;
        let mut truncated = 0;
        let mut position = 0;
        loop {
            let mut character_class = 0;
            let ended = unit >= unit_count;
            if !ended {
                if position == CAPACITY {
                    character_class = 1;
                    truncated = 1;
                } else {
                    let code_unit = text[unit] as usize;
                    unit += 1;
                    if code_unit < 128 {
                        character_class = self.classes[code_unit + 1];
                    } else {
                        character_class = other_class;
                        if (0xd800..=0xdbff).contains(&code_unit) && unit < unit_count {
                            let low = text[unit];
                            if (0xdc00..=0xdfff).contains(&low) {
                                unit += 1;
                            }
                        }
                    }
                }
            }
            let lexer_index = lexer_state * CLASS_COUNT + character_class;
            let event = self.lexer_event[lexer_index];
            lexer_state = self.lexer_next[lexer_index];
            if grammar_state != self.error_state {
                let in_object = if depth > 0 { ((container_bits >> (depth - 1)) & 1) as usize } else { 0 };
                let grammar_index = grammar_state * SYMBOL_COUNT
                    + self.grammar_token[event] * CONTEXT_COUNT
                    + self.depth_class[depth as usize] * 2
                    + in_object;
                grammar_state = self.grammar_next[grammar_index];
                if grammar_state == self.error_state {
                    first_error_position = position as i64;
                    first_error_kind = self.lexer_error_kind[event] + self.grammar_error[grammar_index];
                } else {
                    let change = self.depth_change[event];
                    if change == 1 {
                        container_bits = if event == self.open_object_event {
                            container_bits | (1 << depth)
                        } else {
                            container_bits & !(1 << depth)
                        };
                    }
                    depth += change;
                }
            }
            if ended || position == CAPACITY {
                break;
            }
            kinds[position] = self.display[event * GRAMMAR_STATE_COUNT + grammar_state] as u8;
            position += 1;
        }
        for kind in kinds.iter_mut().take(CAPACITY).skip(position) {
            *kind = display_kind::PAST_END as u8;
        }
        TableLoopResult { first_error_position, first_error_kind, truncated }
    }
}
