// One WGSL compute shader from a loop program, a port of
// src/json-search/loop-emit.ts: one invocation runs one document, and each
// looped stage's operations are written once, inside a `for` loop over the
// document's positions.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};

use super::emit::{flatten_program, positions_of_wires, wgsl_float, wgsl_int, Row, RowCounts, RowWriter};
use super::loop_program::{Binding, GroupKind, LoopProgram, Repeat, Write};
use crate::circuit::Wire;

#[derive(Clone, Debug, PartialEq)]
pub struct LoopWgslModule {
    pub code: String,
    pub workgroup_size: usize,
    pub statement_count: usize,
    pub bytes: usize,
    pub shared_length: usize,
    pub cells_per_instance: usize,
    pub input_cells: Vec<usize>,
    pub output_counts: Vec<usize>,
    pub windowed_groups: Vec<bool>,
    pub row_counts: RowCounts,
}

fn order_rows_by_use(rows: &[Row], source_size: usize, output_positions: &[usize]) -> Vec<Row> {
    let row_at: HashMap<usize, &Row> = rows.iter().map(|row| (row.position(), row)).collect();
    let mut ordered = Vec::new();
    let mut visited = HashSet::new();
    fn visit(position: usize, source_size: usize, row_at: &HashMap<usize, &Row>, visited: &mut HashSet<usize>, ordered: &mut Vec<Row>) {
        if position < source_size || !visited.insert(position) {
            return;
        }
        let row = row_at[&position];
        for term in row.read_terms() {
            visit(term.position, source_size, row_at, visited, ordered);
        }
        ordered.push(row.clone());
    }
    for &position in output_positions {
        visit(position, source_size, &row_at, &mut visited, &mut ordered);
    }
    assert!(ordered.len() == rows.len(), "every row of a stage leads to one of its outputs");
    ordered
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RowOrder {
    Use,
    Layers,
}

fn buffer_name(buffer: usize) -> String {
    if buffer == 0 { "instance_sources".to_string() } else { format!("instance_inputs_{buffer}") }
}

fn base_name(buffer: usize) -> String {
    if buffer == 0 { "source_base".to_string() } else { format!("input_base_{buffer}") }
}

/// Insertion-ordered map of hoisted cell reads, like a JavaScript Map.
#[derive(Default)]
struct OrderedCells {
    order: Vec<String>,
    lines: HashMap<String, String>,
}

impl OrderedCells {
    fn set(&mut self, name: String, line: String) {
        if !self.lines.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.lines.insert(name, line);
    }
}

pub fn emit_loop_wgsl(program: &LoopProgram, workgroup_size: usize, order: RowOrder) -> LoopWgslModule {
    let positions = program.positions;
    let hoisted_shared: RefCell<BTreeSet<usize>> = RefCell::new(BTreeSet::new());
    let hoisted_cells: RefCell<OrderedCells> = RefCell::new(OrderedCells::default());
    let cell_name = |buffer: usize, column: usize| -> String {
        let name = format!("d{buffer}_{column}");
        let line = format!("let {name}: f32 = {}[{} + {column}u];", buffer_name(buffer), base_name(buffer));
        hoisted_cells.borrow_mut().set(name.clone(), line);
        name
    };
    let mut row_counts = RowCounts::default();
    let mut stage_lines: Vec<String> = Vec::new();

    for (stage_index, stage) in program.stages.iter().enumerate() {
        let stage_program = &stage.program;
        let prefix = format!("{}_", (b'a' + stage_index as u8) as char);
        let per_position: RefCell<Vec<String>> = RefCell::new(Vec::new());
        let names: RefCell<HashMap<usize, String>> = RefCell::new(HashMap::new());
        let words_read: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
        let source = |position: usize| -> String {
            if let Some(existing) = names.borrow().get(&position) {
                return existing.clone();
            }
            let name = match stage.binding(Wire::from_position(position)) {
                None => {
                    hoisted_shared.borrow_mut().insert(stage.shared_offset + position);
                    format!("s{}", stage.shared_offset + position)
                }
                Some(Binding::Register { register }) => format!("r{register}"),
                Some(Binding::Cell { buffer, column }) => cell_name(buffer, column),
                Some(Binding::Item { buffer }) => {
                    let name = format!("{prefix}c{position}");
                    per_position
                        .borrow_mut()
                        .push(format!("let {name}: f32 = {}[{} + j];", buffer_name(buffer), base_name(buffer)));
                    name
                }
                Some(Binding::Field { field, component }) => {
                    let declared = &program.fields[field];
                    let name = format!("{prefix}f{position}");
                    if declared.bits {
                        let word = format!("{prefix}w{field}");
                        if words_read.borrow_mut().insert(field) {
                            per_position.borrow_mut().push(format!("let {word}: u32 = field{field}[j];"));
                        }
                        per_position.borrow_mut().push(format!("let {name}: f32 = f32(({word} >> {component}u) & 1u);"));
                    } else {
                        per_position
                            .borrow_mut()
                            .push(format!("let {name}: f32 = field{field}[j * {}u + {component}u];", declared.width));
                    }
                    name
                }
            };
            names.borrow_mut().insert(position, name.clone());
            name
        };
        let candidate = |row: &Row, clamped: &str| -> String {
            let Row::Index { candidate_count, candidate_base, candidate_stride, .. } = row else { unreachable!() };
            for candidate in 0..*candidate_count as i64 {
                let wire = Wire::from_position((candidate_base + candidate * candidate_stride) as usize);
                assert!(stage.binding(wire).is_none(), "in a loop program, an index read's candidates are shared cells");
            }
            format!(
                "shared_values[u32({}i + {clamped} * {})]",
                stage.shared_offset as i64 + candidate_base,
                wgsl_int(*candidate_stride)
            )
        };
        let mut writer = RowWriter::new(stage_program.source_size, &prefix, source, candidate);
        let output_wires: Vec<Wire> = stage.outputs.iter().map(|&(wire, _)| wire).collect();
        let output_positions = positions_of_wires(stage_program, &output_wires);
        let rows = flatten_program(stage_program);
        let rows = if order == RowOrder::Layers {
            rows
        } else {
            order_rows_by_use(&rows, stage_program.source_size, &output_positions)
        };
        for row in &rows {
            writer.write(row);
        }
        row_counts.affine += writer.row_counts.affine;
        row_counts.reglu += writer.row_counts.reglu;
        row_counts.boolean += writer.row_counts.boolean;
        row_counts.index += writer.row_counts.index;

        let mut field_writes = Vec::new();
        let mut window_writes = Vec::new();
        let mut document_writes = Vec::new();
        let mut register_writes = Vec::new();
        let mut bits_of_field: Vec<(usize, Vec<String>)> = Vec::new();
        for (index, &(_, write)) in stage.outputs.iter().enumerate() {
            let value = writer.name(output_positions[index]);
            match write {
                Write::Register { register } => register_writes.push(format!("r{register} = {value};")),
                Write::Field { field, component } => {
                    let declared = &program.fields[field];
                    if declared.bits {
                        let slot = match bits_of_field.iter().position(|(existing, _)| *existing == field) {
                            Some(slot) => slot,
                            None => {
                                bits_of_field.push((field, vec![String::new(); declared.width]));
                                bits_of_field.len() - 1
                            }
                        };
                        bits_of_field[slot].1[component] =
                            if component == 0 { format!("u32({value})") } else { format!("(u32({value}) << {component}u)") };
                    } else {
                        field_writes.push(format!("field{field}[j * {}u + {component}u] = {value};", declared.width));
                    }
                }
                Write::Position { group } => {
                    window_writes.push(format!("instance_outputs_{group}[window_base_{group} + j] = {value};"));
                }
                Write::Document { group, column } => {
                    document_writes.push(format!("instance_outputs_{group}[output_base_{group} + {column}u] = {value};"));
                }
            }
        }
        for (field, bits) in &bits_of_field {
            assert!(bits.iter().all(|bit| !bit.is_empty()), "a stage that writes a field of bits writes every bit of it");
            field_writes.push(format!("field{field}[j] = {};", bits.join(" | ")));
        }
        let body_lines = std::mem::take(&mut writer.body);
        drop(writer);
        let per_position = per_position.into_inner();
        let mut body = per_position.clone();
        body.extend(body_lines);
        body.extend(field_writes);
        if !window_writes.is_empty() {
            body.push("if (in_window) {".to_string());
            body.extend(window_writes);
            body.push("}".to_string());
        }
        body.extend(document_writes);
        body.extend(register_writes);
        match stage.repeat {
            Repeat::Once => {
                assert!(per_position.is_empty(), "a stage that runs once reads no item and no field");
                stage_lines.extend(body);
            }
            Repeat::Forward => {
                stage_lines.push("for (var j: u32 = 0u; j < positions; j++) {".to_string());
                stage_lines.extend(body);
                stage_lines.push("}".to_string());
            }
            Repeat::Backward => {
                stage_lines.push("for (var back: u32 = 0u; back < positions; back++) {".to_string());
                stage_lines.push("let j: u32 = positions - 1u - back;".to_string());
                stage_lines.extend(body);
                stage_lines.push("}".to_string());
            }
        }
    }

    let groups = &program.output_groups;
    if program.length_cell.is_some() {
        let fills: Vec<String> = groups
            .iter()
            .enumerate()
            .filter(|(_, group)| group.kind == GroupKind::Position)
            .map(|(index, group)| format!("instance_outputs_{index}[window_base_{index} + j] = {};", wgsl_float(group.fill)))
            .collect();
        if !fills.is_empty() {
            stage_lines.push("if (in_window) {".to_string());
            stage_lines.push(format!("for (var j: u32 = positions; j < {positions}u; j++) {{"));
            stage_lines.extend(fills);
            stage_lines.push("}".to_string());
            stage_lines.push("}".to_string());
        }
    }
    for result in &program.results {
        stage_lines.push(format!(
            "instance_outputs_{}[output_base_{} + {}u] = r{};",
            result.group, result.group, result.column, result.register
        ));
    }

    let length_name = program.length_cell.map(|(buffer, column)| cell_name(buffer, column));
    let mut main = vec![
        "let i: u32 = id.x;".to_string(),
        "if (i >= params.count) {".to_string(),
        "return;".to_string(),
        "}".to_string(),
        "_ = arrayLength(&shared_values);".to_string(),
    ];
    main.extend((0..program.buffers.len()).map(|buffer| format!("_ = arrayLength(&{});", buffer_name(buffer))));
    main.push("let row: u32 = params.first + i;".to_string());
    main.extend(
        program
            .buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| format!("let {}: u32 = row * {}u;", base_name(index), buffer.cells_per_document)),
    );
    main.extend(groups.iter().enumerate().filter(|(_, group)| group.kind == GroupKind::Document).map(|(index, group)| {
        format!("let output_base_{index}: u32 = row * {}u;", group.values_per_document)
    }));
    main.push(
        "let in_window: bool = row >= params.window_first && row - params.window_first < params.window_count;".to_string(),
    );
    main.extend(
        groups
            .iter()
            .enumerate()
            .filter(|(_, group)| group.kind == GroupKind::Position)
            .map(|(index, _)| format!("let window_base_{index}: u32 = (row - params.window_first) * {positions}u;")),
    );
    main.extend(hoisted_shared.borrow().iter().map(|position| format!("let s{position}: f32 = shared_values[{position}u];")));
    {
        let cells = hoisted_cells.borrow();
        main.extend(cells.order.iter().map(|name| cells.lines[name].clone()));
    }
    main.push(match &length_name {
        None => format!("let positions: u32 = {positions}u;"),
        Some(name) => format!("let positions: u32 = min(u32({name}), {positions}u);"),
    });
    main.extend(program.registers.iter().enumerate().map(|(index, register)| format!("var r{index}: f32 = {};", wgsl_float(register.initial))));
    main.extend(program.fields.iter().enumerate().map(|(index, field)| {
        if field.bits {
            format!("var field{index}: array<u32, {positions}>;")
        } else {
            format!("var field{index}: array<f32, {}>;", positions * field.width)
        }
    }));
    main.extend(stage_lines);
    for field in &program.fields {
        assert!(!field.bits || field.width <= 32, "a field of bits holds at most 32 bits per position");
    }
    let mut code_lines = vec![
        "struct Params {".to_string(),
        "  count: u32,".to_string(),
        "  window_first: u32,".to_string(),
        "  window_count: u32,".to_string(),
        "  first: u32,".to_string(),
        "}".to_string(),
        "@group(0) @binding(0) var<uniform> params: Params;".to_string(),
        "@group(0) @binding(1) var<storage, read> shared_values: array<f32>;".to_string(),
    ];
    code_lines.extend(
        (0..program.buffers.len())
            .map(|buffer| format!("@group(0) @binding({}) var<storage, read> {}: array<f32>;", 2 + buffer, buffer_name(buffer))),
    );
    code_lines.extend((0..groups.len()).map(|group| {
        format!(
            "@group(0) @binding({}) var<storage, read_write> instance_outputs_{group}: array<f32>;",
            2 + program.buffers.len() + group
        )
    }));
    code_lines.push(String::new());
    code_lines.push(format!("@compute @workgroup_size({workgroup_size})"));
    code_lines.push("fn main(@builtin(global_invocation_id) id: vec3<u32>) {".to_string());
    let statement_count = main.len();
    code_lines.extend(main);
    code_lines.push("}".to_string());
    code_lines.push(String::new());
    let code = code_lines.join("\n");
    LoopWgslModule {
        bytes: code.encode_utf16().count(),
        code,
        workgroup_size,
        statement_count,
        shared_length: program.shared_values.len(),
        cells_per_instance: program.buffers[0].cells_per_document,
        input_cells: program.buffers[1..].iter().map(|buffer| buffer.cells_per_document).collect(),
        output_counts: groups.iter().map(|group| group.values_per_document).collect(),
        windowed_groups: groups.iter().map(|group| group.kind == GroupKind::Position).collect(),
        row_counts,
    }
}
