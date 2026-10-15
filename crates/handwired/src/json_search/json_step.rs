// The JSON highlighter as the work of one position, a port of
// src/json-search/json-step.ts, see that file for what each part computes.

use super::kinds::{display_kind, CAPACITY, ERROR_KINDS, MAXIMUM_DEPTH};
use super::loop_program::{create_stage, Binding, Buffer, GroupKind, LoopProgram, OutputGroup, Register, Repeat, Write};
use super::tables::{
    all_tables, CLASS_COUNT, CONTEXT_COUNT, EVENT_COUNT, GRAMMAR_STATES, GRAMMAR_STATE_COUNT, LEXER_STATE_COUNT, SYMBOL_COUNT,
};
use crate::cells::{constant, input, SourceTable};
use crate::circuit::{Circuit, OutOfRange, Term, Wire};
use crate::emit::{emit_compare, CompareOptions, Relation};

const LENGTH_LIMIT: f64 = 1_048_576.0;
pub const STEP_SUMMARY_NAMES: [&str; 4] = ["valid", "firstErrorPosition", "truncated", "firstErrorKind"];
pub const REGISTER_NAMES: [&str; 6] = ["lexerState", "depth", "code", "grammarState", "errorCount", "firstErrorKind"];

fn read(circuit: &mut Circuit, terms: &[Term], bias: f64, candidates: &[Wire], out_of_range: OutOfRange) -> Wire {
    circuit.index_read_affine(terms, bias, candidates, out_of_range, out_of_range, None)
}

fn on_grid(circuit: &mut Circuit, wire: Wire) -> Wire {
    circuit.assume_grid(wire, 1.0);
    wire
}

pub struct JsonLoop {
    pub program: LoopProgram,
}

/// `cells_per_document` is the layout of buffer 0: 512 code points, then the text's length.
pub fn build_json_loop(cells_per_document: usize) -> JsonLoop {
    let mut table = SourceTable::new();
    let positions = (CAPACITY + 1) as f64;
    let code_limit = (CAPACITY as f64 - 1.0) * 2f64.powi(MAXIMUM_DEPTH as i32 - 1);
    let code_point = table.cell("codePoint", input((-1.0, 1_114_111.0)).initial(-1.0).grid(1.0));
    let text_length = table.cell("textLength", input((0.0, LENGTH_LIMIT)).grid(1.0));
    let lexer_state = table.cell("lexerState", input((0.0, LEXER_STATE_COUNT as f64 - 1.0)).grid(1.0));
    let depth = table.cell("depth", input((-(CAPACITY as f64), CAPACITY as f64)).grid(1.0));
    let code = table.cell("code", input((-code_limit, code_limit)).grid(1.0));
    let grammar_state = table.cell("grammarState", input((0.0, GRAMMAR_STATE_COUNT as f64 - 1.0)).grid(1.0));
    let error_count = table.cell("errorCount", input((0.0, positions)).grid(1.0));
    let first_error_kind = table.cell("firstErrorKind", input((0.0, ERROR_KINDS.len() as f64 - 1.0)).grid(1.0));
    let tables: Vec<Vec<Wire>> = all_tables()
        .into_iter()
        .map(|(name, values)| {
            let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            values
                .iter()
                .enumerate()
                .map(|(index, &value)| table.cell(format!("tables.{name}[{index}]"), constant(value, (minimum, maximum)).grid(1.0)))
                .collect()
        })
        .collect();
    let [classes, lexer_next_cells, lexer_event_cells, grammar_token_cells, lexer_error_kind_cells, depth_change_cells, object_weight_cells, level_weight_cells, depth_class_cells, grammar_next_cells, grammar_error_cells, display_cells] =
        <[Vec<Wire>; 12]>::try_from(tables).unwrap();
    let mut circuit = table.create_circuit();
    let c = &mut circuit;
    let error_state = GRAMMAR_STATES.iter().position(|&state| state == "error").unwrap() as f64;
    let in_error = |c: &mut Circuit, state: Wire| c.reglu(&[], 1.0, &[(state, 1.0)], 1.0 - error_state);
    let error_before = in_error(c, grammar_state);
    let clamp = OutOfRange::Clamp;

    struct Position {
        event: Wire,
        lexer_next: Wire,
        grammar_next: Wire,
        error_after: Wire,
        first_error_kind: Wire,
    }
    let emit_position = |c: &mut Circuit, character_class: &[Term], in_object: Option<Wire>| -> Position {
        let mut lexer_terms = vec![(lexer_state, CLASS_COUNT as f64)];
        lexer_terms.extend_from_slice(character_class);
        let event = read(c, &lexer_terms, 0.0, &lexer_event_cells, clamp);
        let lexer_next = read(c, &lexer_terms, 0.0, &lexer_next_cells, clamp);
        let depth_class = read(c, &[(depth, 1.0)], 0.0, &depth_class_cells, clamp);
        let grammar_token = read(c, &[(event, 1.0)], 0.0, &grammar_token_cells, clamp);
        let mut grammar_terms = vec![(grammar_state, SYMBOL_COUNT as f64), (grammar_token, CONTEXT_COUNT as f64), (depth_class, 2.0)];
        if let Some(in_object) = in_object {
            grammar_terms.push((in_object, 1.0));
        }
        let grammar_next = read(c, &grammar_terms, 0.0, &grammar_next_cells, clamp);
        let lexer_error_kind = read(c, &[(event, 1.0)], 0.0, &lexer_error_kind_cells, clamp);
        let grammar_error = read(c, &grammar_terms, 0.0, &grammar_error_cells, clamp);
        let error_kind_here = c.round_affine(&[(lexer_error_kind, 1.0), (grammar_error, 1.0)], 0.0);
        let error_after = in_error(c, grammar_next);
        let entered = c.reglu(&[(error_kind_here, 1.0)], 0.0, &[(error_after, 1.0), (error_before, -1.0)], 0.0);
        let first_error_kind = c.round_affine(&[(first_error_kind, 1.0), (entered, 1.0)], 0.0);
        Position { event, lexer_next, grammar_next, error_after, first_error_kind }
    };

    let character_class = read(c, &[(code_point, 1.0)], 1.0, &classes, clamp);
    let level_weight = read(c, &[(depth, 1.0)], -1.0, &level_weight_cells, clamp);
    let level_weight = on_grid(c, level_weight);
    let in_object = emit_compare(c, code, Relation::AtLeast, level_weight, CompareOptions::default());
    let step = emit_position(c, &[(character_class, 1.0)], Some(in_object));
    let depth_change = read(c, &[(step.event, 1.0)], 0.0, &depth_change_cells, clamp);
    let depth_change = on_grid(c, depth_change);
    let object_weight =
        read(c, &[(depth, EVENT_COUNT as f64), (step.event, 1.0)], 0.0, &object_weight_cells, OutOfRange::Zero);
    let object_weight = on_grid(c, object_weight);
    let next_depth = c.round_affine(&[(depth, 1.0), (depth_change, 1.0)], 0.0);
    let next_code = c.round_affine(&[(code, 1.0), (object_weight, 1.0)], 0.0);
    let next_error_count = c.round_affine(&[(error_count, 1.0), (step.error_after, 1.0)], 0.0);
    let display_kind_wire =
        read(c, &[(step.event, GRAMMAR_STATE_COUNT as f64), (step.grammar_next, 1.0)], 0.0, &display_cells, clamp);
    let step_registers = [step.lexer_next, next_depth, next_code, step.grammar_next, next_error_count, step.first_error_kind];

    let capacity = c.constant(CAPACITY as f64);
    let truncated = emit_compare(c, text_length, Relation::Greater, capacity, CompareOptions::default());
    let end = emit_position(c, &[(truncated, 1.0)], None);
    let error_free_count = c.round_affine(&[(error_count, -1.0), (end.error_after, -1.0)], positions);
    let valid = c.round_affine(&[(end.error_after, -1.0)], 1.0);
    let first_error_position = c.round_affine(&[(error_free_count, 1.0), (valid, -(positions + 1.0))], 0.0);
    let truncated_output = c.round_affine(&[(truncated, 1.0)], 0.0);
    let end_outputs = [valid, first_error_position, truncated_output, end.first_error_kind];

    let finished = table.finish(&mut circuit);
    let register_cells = [lexer_state, depth, code, grammar_state, error_count, first_error_kind];
    let register_bindings: Vec<(Wire, Binding)> =
        register_cells.iter().enumerate().map(|(register, &wire)| (wire, Binding::Register { register })).collect();
    let mut step_bindings = vec![(code_point, Binding::Item { buffer: 0 })];
    step_bindings.extend(&register_bindings);
    let mut step_outputs = vec![(display_kind_wire, Write::Position { group: 1 })];
    step_outputs.extend(step_registers.iter().enumerate().map(|(register, &wire)| (wire, Write::Register { register })));
    let step_stage = create_stage(0, &circuit, "step", Repeat::Forward, 0, step_bindings, step_outputs);
    let mut end_bindings = vec![(text_length, Binding::Cell { buffer: 0, column: cells_per_document - 1 })];
    end_bindings.extend(&register_bindings);
    let end_writes =
        end_outputs.iter().enumerate().map(|(column, &wire)| (wire, Write::Document { group: 0, column })).collect();
    let end_stage = create_stage(0, &circuit, "end", Repeat::Once, 0, end_bindings, end_writes);
    let program = LoopProgram {
        positions: CAPACITY,
        buffers: vec![Buffer { name: "documents".into(), cells_per_document }],
        registers: REGISTER_NAMES.iter().map(|&name| Register { name: name.into(), initial: 0.0 }).collect(),
        fields: Vec::new(),
        output_groups: vec![
            OutputGroup {
                name: "summary".into(),
                kind: GroupKind::Document,
                values_per_document: STEP_SUMMARY_NAMES.len(),
                fill: 0.0,
            },
            OutputGroup {
                name: "displayKinds".into(),
                kind: GroupKind::Position,
                values_per_document: CAPACITY,
                fill: display_kind::PAST_END as f64,
            },
        ],
        circuits: vec![circuit],
        stages: vec![step_stage, end_stage],
        results: Vec::new(),
        shared_values: finished.initial_source,
        length_cell: None,
    };
    JsonLoop { program }
}
