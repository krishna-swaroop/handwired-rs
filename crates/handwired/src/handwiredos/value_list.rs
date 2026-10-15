// The list of the program's own values, which fills HandwiredOS's Monitor
// window, a port of the circuit-building parts of src/handwiredos/value-list.ts:
// the cells, buildValueList, the standalone circuit, the three locations' rows
// and the output checks. The DOM renderer is left out.

use crate::cells::{event, input, input_bit, state, state_bit, FinishedTable, SourceTable};
use crate::circuit::{Circuit, Term, Wire};
use crate::compile::TensorProgram;
use crate::emit::{emit_clamp_exact, emit_compare, emit_maximum_exact, emit_minimum_exact, emit_multiply, emit_select, CompareOptions, Relation};
use crate::fp::{fround, js_number};

pub const ROW_HEIGHT: f64 = 24.0;
pub const SLOT_COUNT: usize = 64;
pub const ROW_CAPACITY: f64 = 100_000.0;
pub const SCROLL_GRID: f64 = 0.25;
pub const MAXIMUM_SCROLL_TOP: f64 = ROW_CAPACITY * ROW_HEIGHT;
const PIXEL_GRID: f64 = 1.0 / 64.0;
const MAXIMUM_VIEWPORT_HEIGHT: f64 = 16_384.0;
pub const LOCATION_COUNT: usize = 3;
const INITIAL_LOCATION: usize = 1;

/// Monitor's layout inside its content box, in px.
pub struct MonitorLayout {
    pub sidebar_width: f64,
    pub header_height: f64,
    pub row_height: f64,
    pub location_inset: f64,
    pub location_top: f64,
    pub location_height: f64,
    pub scrollbar_strip: f64,
    pub footer_height: f64,
}

pub const MONITOR_LAYOUT: MonitorLayout = MonitorLayout {
    sidebar_width: 152.0,
    header_height: 28.0,
    row_height: ROW_HEIGHT,
    location_inset: 8.0,
    location_top: 28.0,
    location_height: ROW_HEIGHT,
    scrollbar_strip: 16.0,
    footer_height: 32.0,
};

pub const KEY_NAMES: [&str; 6] = ["up", "down", "pageUp", "pageDown", "home", "end"];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ValueListKey {
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
}

/// The list's own cells, in valueListCells' order.
#[derive(Clone, Debug)]
pub struct ValueListCells {
    pub scroll_top: Wire,
    pub row_counts: Vec<Wire>,
    /// up, down, pageUp, pageDown, home, end.
    pub keys: [Wire; 6],
    pub location: Vec<Wire>,
    pub selected: Wire,
}

impl ValueListCells {
    pub fn key(&self, key: ValueListKey) -> Wire {
        self.keys[key as usize]
    }
}

fn path(prefix: &str, field: &str) -> String {
    if prefix.is_empty() { field.to_string() } else { format!("{prefix}.{field}") }
}

/// Declares valueListCells under `prefix`, e.g. 'valueList'.
pub fn declare_value_list_cells(table: &mut SourceTable, prefix: &str) -> ValueListCells {
    let scroll_top = table.cell(path(prefix, "scrollTop"), input((0.0, MAXIMUM_SCROLL_TOP)).grid(SCROLL_GRID));
    let row_counts = (0..LOCATION_COUNT)
        .map(|index| table.cell(format!("{}[{index}]", path(prefix, "rowCounts")), input((0.0, ROW_CAPACITY)).grid(1.0)))
        .collect();
    let keys = KEY_NAMES.map(|name| table.cell(format!("{}.{name}", path(prefix, "keys")), event().grid(1.0)));
    let members: Vec<_> = (0..LOCATION_COUNT)
        .map(|index| state_bit(if index == INITIAL_LOCATION { 1.0 } else { 0.0 }).grid(1.0))
        .collect();
    let location = table.one_hot(&path(prefix, "location"), &members, false, false);
    let selected = table.cell(path(prefix, "selected"), state((-1.0, ROW_CAPACITY - 1.0)).initial(-1.0).grid(1.0));
    ValueListCells { scroll_top, row_counts, keys, location, selected }
}

/// The list's key cells, for the at-most-one rule.
pub fn value_list_keys(cells: &ValueListCells) -> Vec<Wire> {
    cells.keys.to_vec()
}

/// The host's wires that the list reads.
pub struct ValueListHost {
    pub viewport_height: Wire,
    pub focused: Wire,
    pub list_click_active: Wire,
    pub list_click_y: Wire,
    pub location_clicks: Vec<Wire>,
}

/// Per DOM slot: row, top, present, selected.
pub const SLOT_FIELD_SIZE: usize = 4;
pub const SLOT_FIELD_NAMES: [&str; 4] = ["row", "top", "present", "selected"];

#[derive(Clone, Debug)]
pub struct ValueListOutputs {
    pub total_height: Wire,
    pub scroll: Wire,
    pub scroll_target: Wire,
    pub scroll_request: Wire,
    pub slot_overflow: Wire,
    pub selected: Wire,
    pub locations: Vec<Wire>,
    pub slots: Vec<Wire>,
}

const ROUNDING_BASE: f64 = 12_582_912.0; // 2^23 + 2^22

fn emit_round(circuit: &mut Circuit, value: Wire, minimum: f64, maximum: f64) -> Wire {
    let shifted = circuit.round_affine(&[(value, 1.0)], ROUNDING_BASE);
    let rounded = circuit.round_affine(&[(shifted, 1.0)], -ROUNDING_BASE);
    circuit.mark_rounding_point(shifted);
    circuit.mark_rounding_point(rounded);
    circuit.assume_range(rounded, minimum, maximum);
    circuit.assume_grid(rounded, 1.0);
    rounded
}

fn emit_floor_divide(circuit: &mut Circuit, value: Wire, divisor: f64, maximum: f64) -> Wire {
    assert!(divisor.fract() == 0.0 && divisor >= 1.0, "the divisor must be a whole number");
    let approximate = circuit.round_affine(&[(value, 1.0 / divisor)], 0.0);
    let rounded = emit_round(circuit, approximate, 0.0, (maximum / divisor).floor() + 1.0);
    let product = circuit.affine(&[(rounded, divisor)], 0.0);
    let too_large = emit_compare(circuit, product, Relation::Greater, value, CompareOptions::default());
    let floor = circuit.round_affine(&[(rounded, 1.0), (too_large, -1.0)], 0.0);
    circuit.assume_range(floor, 0.0, (maximum / divisor).floor());
    circuit.assume_grid(floor, 1.0);
    floor
}

fn emit_floor_shift(circuit: &mut Circuit, value: Wire, bits: i32, maximum: f64) -> Wire {
    let shifted = circuit.round_affine(&[(value, 2f64.powi(-bits))], 2f64.powi(-(bits + 1)) - 0.5);
    emit_round(circuit, shifted, 0.0, (maximum * 2f64.powi(-bits)).floor())
}

fn emit_all(circuit: &mut Circuit, bits: &[Wire]) -> Wire {
    let terms: Vec<Term> = bits.iter().map(|&bit| (bit, 1.0)).collect();
    let all = circuit.reglu(&[], 1.0, &terms, 1.0 - bits.len() as f64);
    circuit.assume_range(all, 0.0, 1.0);
    all
}

fn compare(circuit: &mut Circuit, a: Wire, relation: Relation, b: Wire) -> Wire {
    emit_compare(circuit, a, relation, b, CompareOptions::default())
}

/// Builds the list into the host's circuit, pairs the list's state cells with
/// their next values, and returns the list's outputs.
pub fn build_value_list(circuit: &mut Circuit, table: &mut SourceTable, cells: &ValueListCells, host: &ValueListHost) -> ValueListOutputs {
    assert!(host.location_clicks.len() == LOCATION_COUNT, "one click per location");
    circuit.assume_range(host.viewport_height, 0.0, MAXIMUM_VIEWPORT_HEIGHT);
    circuit.assume_grid(host.viewport_height, PIXEL_GRID);
    for &bit in [host.focused, host.list_click_active].iter().chain(&host.location_clicks) {
        circuit.assume_range(bit, 0.0, 1.0);
        circuit.assume_grid(bit, 1.0);
    }
    let zero = circuit.constant(0.0);
    let one = circuit.constant(1.0);

    let click_terms: Vec<Term> = host.location_clicks.iter().map(|&click| (click, 1.0)).collect();
    let any_location_click = circuit.affine(&click_terms, 0.0);
    let next_location: Vec<Wire> = cells
        .location
        .iter()
        .enumerate()
        .map(|(index, &stored)| {
            let kept = circuit.reglu(&[(stored, 1.0)], 0.0, &[(any_location_click, -1.0)], 1.0);
            let next = circuit.round_affine(&[(kept, 1.0), (host.location_clicks[index], 1.0)], 0.0);
            circuit.assume_range(next, 0.0, 1.0);
            next
        })
        .collect();
    let changed_terms: Vec<Term> = cells
        .location
        .iter()
        .enumerate()
        .map(|(index, &stored)| (circuit.reglu(&[], 1.0, &[(host.location_clicks[index], 1.0), (stored, -1.0)], 0.0), 1.0))
        .collect();
    let location_changed = circuit.round_affine(&changed_terms, 0.0);
    circuit.assume_range(location_changed, 0.0, 1.0);
    let row_terms: Vec<Term> = next_location
        .iter()
        .enumerate()
        .map(|(index, &bit)| (emit_multiply(circuit, bit, cells.row_counts[index]), 1.0))
        .collect();
    let row_count = circuit.round_affine(&row_terms, 0.0);
    circuit.assume_range(row_count, 0.0, ROW_CAPACITY);

    let total_height = circuit.round_affine(&[(row_count, ROW_HEIGHT)], 0.0);
    let floor_shifted = circuit.round_affine(&[(host.viewport_height, 1.0)], PIXEL_GRID / 2.0 - 0.5);
    let viewport_floor = emit_round(circuit, floor_shifted, 0.0, MAXIMUM_VIEWPORT_HEIGHT);
    let ceil_shifted = circuit.round_affine(&[(host.viewport_height, 1.0)], 0.5 - PIXEL_GRID / 2.0);
    let viewport_ceil = emit_round(circuit, ceil_shifted, 0.0, MAXIMUM_VIEWPORT_HEIGHT);
    let maximum_scroll = circuit.reglu(&[], 1.0, &[(total_height, 1.0), (viewport_floor, -1.0)], 0.0);
    let scroll = emit_minimum_exact(circuit, cells.scroll_top, maximum_scroll);
    circuit.assume_grid(scroll, SCROLL_GRID);

    let first_visible = emit_floor_divide(circuit, scroll, ROW_HEIGHT, MAXIMUM_SCROLL_TOP);
    let offset_in_row = circuit.round_affine(&[(scroll, 1.0), (first_visible, -ROW_HEIGHT)], 0.0);
    circuit.assume_range(offset_in_row, 0.0, ROW_HEIGHT - SCROLL_GRID);
    circuit.assume_grid(offset_in_row, SCROLL_GRID);
    let view_end = circuit.round_affine(&[(scroll, 1.0), (viewport_ceil, 1.0)], ROW_HEIGHT - SCROLL_GRID);
    let end_divided =
        emit_floor_divide(circuit, view_end, ROW_HEIGHT, MAXIMUM_SCROLL_TOP + MAXIMUM_VIEWPORT_HEIGHT + ROW_HEIGHT);
    let end_visible = emit_minimum_exact(circuit, end_divided, row_count);
    circuit.assume_grid(end_visible, 1.0);
    let spare_slots = circuit.reglu(&[], 1.0, &[(first_visible, 1.0), (end_visible, -1.0)], SLOT_COUNT as f64);
    circuit.assume_range(spare_slots, 0.0, SLOT_COUNT as f64);
    let half_spare = emit_floor_shift(circuit, spare_slots, 1, SLOT_COUNT as f64);
    let first_drawn = circuit.reglu(&[], 1.0, &[(first_visible, 1.0), (half_spare, -1.0)], 0.0);

    let maximum_click = circuit.constant(MAXIMUM_VIEWPORT_HEIGHT);
    let click_y = emit_clamp_exact(circuit, host.list_click_y, zero, maximum_click);
    let click_offset = circuit.round_affine(&[(click_y, 1.0), (offset_in_row, 1.0)], 0.0);
    let click_rows = emit_floor_divide(circuit, click_offset, ROW_HEIGHT, MAXIMUM_VIEWPORT_HEIGHT + ROW_HEIGHT);
    let clicked_row = circuit.round_affine(&[(first_visible, 1.0), (click_rows, 1.0)], 0.0);
    let clicked_in_list = compare(circuit, clicked_row, Relation::Less, row_count);
    let click_selects = emit_all(circuit, &[host.list_click_active, clicked_in_list]);
    let after_click = emit_select(circuit, click_selects, cells.selected, clicked_row);
    circuit.assume_grid(after_click, 1.0);

    let page_divided = emit_floor_divide(circuit, viewport_floor, ROW_HEIGHT, MAXIMUM_VIEWPORT_HEIGHT);
    let page_rows = emit_maximum_exact(circuit, page_divided, one);
    circuit.assume_grid(page_rows, 1.0);
    let key_terms: Vec<Term> = cells.keys.iter().map(|&key| (key, 1.0)).collect();
    let any_key = circuit.affine(&key_terms, 0.0);
    circuit.assume_range(any_key, 0.0, 1.0);
    let has_rows = compare(circuit, row_count, Relation::AtLeast, one);
    let key_active = emit_all(circuit, &[any_key, host.focused, has_rows]);
    let last_row = circuit.round_affine(&[(row_count, 1.0)], -1.0);
    let [up, down, page_up, page_down, home, end] = cells.keys;
    let has_selection = compare(circuit, after_click, Relation::AtLeast, zero);
    let page_down_rows = emit_multiply(circuit, page_down, page_rows);
    let page_up_rows = emit_multiply(circuit, page_up, page_rows);
    let stepped = circuit.round_affine(
        &[(after_click, 1.0), (down, 1.0), (up, -1.0), (page_down_rows, 1.0), (page_up_rows, -1.0)],
        0.0,
    );
    let moved = emit_select(circuit, has_selection, first_visible, stepped);
    let with_home = emit_select(circuit, home, moved, zero);
    let with_end = emit_select(circuit, end, with_home, last_row);
    circuit.assume_grid(with_end, 1.0);
    let clamped = emit_clamp_exact(circuit, with_end, zero, last_row);
    let after_keys = emit_select(circuit, key_active, after_click, clamped);
    circuit.assume_range(after_keys, -1.0, ROW_CAPACITY - 1.0);
    circuit.assume_grid(after_keys, 1.0);
    let selected_in_range = emit_minimum_exact(circuit, after_keys, last_row);
    circuit.assume_range(selected_in_range, -1.0, ROW_CAPACITY - 1.0);
    circuit.assume_grid(selected_in_range, 1.0);
    let kept_selection = circuit.reglu(&[(selected_in_range, 1.0)], 1.0, &[(location_changed, -1.0)], 1.0);
    let next_selected = circuit.round_affine(&[(kept_selection, 1.0)], -1.0);
    circuit.assume_range(next_selected, -1.0, ROW_CAPACITY - 1.0);
    circuit.assume_grid(next_selected, 1.0);

    let selected_top = circuit.round_affine(&[(after_keys, ROW_HEIGHT)], 0.0);
    let lowest = circuit.round_affine(&[(selected_top, 1.0), (viewport_floor, -1.0)], ROW_HEIGHT);
    let at_least_lowest = emit_maximum_exact(circuit, scroll, lowest);
    circuit.assume_grid(at_least_lowest, SCROLL_GRID);
    let revealed = emit_minimum_exact(circuit, at_least_lowest, selected_top);
    circuit.assume_grid(revealed, SCROLL_GRID);
    let key_target = emit_select(circuit, key_active, scroll, revealed);
    circuit.assume_range(key_target, 0.0, MAXIMUM_SCROLL_TOP);
    circuit.assume_grid(key_target, SCROLL_GRID);
    let scroll_target = circuit.reglu(&[(key_target, 1.0)], 0.0, &[(location_changed, -1.0)], 1.0);
    circuit.assume_range(scroll_target, 0.0, MAXIMUM_SCROLL_TOP);
    circuit.assume_grid(scroll_target, SCROLL_GRID);
    let unchanged = compare(circuit, scroll_target, Relation::Equal, scroll);
    let scroll_request = circuit.round_affine(&[(unchanged, -1.0)], 1.0);
    circuit.assume_range(scroll_request, 0.0, 1.0);

    let block = emit_floor_shift(circuit, first_drawn, SLOT_COUNT.trailing_zeros() as i32, ROW_CAPACITY);
    let offset = circuit.round_affine(&[(first_drawn, 1.0), (block, -(SLOT_COUNT as f64))], 0.0);
    circuit.assume_range(offset, 0.0, SLOT_COUNT as f64 - 1.0);
    let mut slots = Vec::with_capacity(SLOT_COUNT * SLOT_FIELD_SIZE);
    for slot in 0..SLOT_COUNT {
        let slot_constant = circuit.constant(slot as f64);
        let wraps = compare(circuit, offset, Relation::Greater, slot_constant);
        let row = circuit.round_affine(&[(first_drawn, 1.0), (offset, -1.0), (wraps, SLOT_COUNT as f64)], slot as f64);
        circuit.assume_range(row, 0.0, ROW_CAPACITY + SLOT_COUNT as f64 - 1.0);
        let top = circuit.round_affine(&[(row, ROW_HEIGHT)], 0.0);
        let present = compare(circuit, row, Relation::Less, row_count);
        let selected = compare(circuit, row, Relation::Equal, next_selected);
        slots.extend([row, top, present, selected]);
    }
    let visible_rows = circuit.round_affine(&[(end_visible, 1.0), (first_visible, -1.0)], 0.0);
    let slot_count = circuit.constant(SLOT_COUNT as f64);
    let slot_overflow = compare(circuit, visible_rows, Relation::Greater, slot_count);

    table.next(circuit, &[cells.selected], &[next_selected]);
    table.next(circuit, &cells.location, &next_location);
    ValueListOutputs {
        total_height,
        scroll,
        scroll_target,
        scroll_request,
        slot_overflow,
        selected: next_selected,
        locations: next_location,
        slots,
    }
}

/// The host-provided cells of createValueListCircuit, then the list's own.
pub struct StandaloneValueListCells {
    pub viewport_height: Wire,
    pub focused: Wire,
    pub list_click_active: Wire,
    pub list_click_y: Wire,
    pub location_clicks: Vec<Wire>,
    pub value_list: ValueListCells,
}

/// The list on its own, in a table that also holds the cells a host provides.
pub fn create_value_list_circuit() -> (Circuit, SourceTable, StandaloneValueListCells, ValueListOutputs, FinishedTable) {
    let mut table = SourceTable::new();
    let viewport_height =
        table.cell("viewportHeight", input((0.0, MAXIMUM_VIEWPORT_HEIGHT)).initial(266.0).grid(PIXEL_GRID));
    let focused = table.cell("focused", input_bit(1.0));
    let list_click_active = table.cell("listClick.active", event().grid(1.0));
    let list_click_y = table.cell("listClick.y", input((-49_152.0, 49_152.0)).grid(PIXEL_GRID));
    let location_clicks: Vec<Wire> =
        (0..LOCATION_COUNT).map(|index| table.cell(format!("locationClicks[{index}]"), event().grid(1.0))).collect();
    let value_list = declare_value_list_cells(&mut table, "valueList");
    table.at_most_one("value list keys", &value_list_keys(&value_list));
    table.at_most_one("location clicks", &location_clicks);
    let mut circuit = table.create_circuit();
    let host = ValueListHost {
        viewport_height,
        focused,
        list_click_active,
        list_click_y,
        location_clicks: location_clicks.clone(),
    };
    let output = build_value_list(&mut circuit, &mut table, &value_list, &host);
    let finished = table.finish(&mut circuit);
    let cells = StandaloneValueListCells { viewport_height, focused, list_click_active, list_click_y, location_clicks, value_list };
    (circuit, table, cells, output, finished)
}

/// An FP32 value as the shortest decimal that reads back as the same FP32
/// value, e.g. '0.1' for f(0.1). A whole number below 10^21 is shown exactly.
pub fn format_value(value: f64) -> String {
    if !value.is_finite() || (value.fract() == 0.0 && value.abs() < 1e21) {
        return js_number(value);
    }
    for digits in 1..=9 {
        let rounded: f64 = format!("{:.*e}", digits - 1, value).parse().unwrap();
        if fround(rounded) == value {
            return js_number(rounded);
        }
    }
    js_number(value)
}

/// The operation kinds Monitor lists, by the index describe_operations stores.
pub const OPERATION_KIND_NAMES: [&str; 4] = ["affine", "ReGLU", "boolean", "index read"];

/// One row of Operations: the operation's wire, its kind, its layer counted
/// from 1, and its number of stored weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationRow {
    pub wire: u32,
    pub kind: u8,
    pub layer: u32,
    pub weights: u32,
}

/// The rows of a compiled program's live operations, in packed order, as
/// describeOperations computes them.
pub fn describe_operations(program: &TensorProgram) -> Vec<OperationRow> {
    let row_count = program.value_count - program.source_size;
    let mut kinds = vec![0u8; row_count];
    let mut layers = vec![0u32; row_count];
    let mut weights = vec![0u32; row_count];
    let mut first_row = 0;
    for (layer_index, layer) in program.layers.iter().enumerate() {
        let stored = |projection_row: usize| layer.row_offsets[projection_row + 1] - layer.row_offsets[projection_row];
        for output in 0..layer.output_size {
            layers[first_row + output] = layer_index as u32 + 1;
            weights[first_row + output] = stored(output);
        }
        for (gate, &gate_row) in layer.gate_rows.iter().enumerate() {
            let row = first_row + gate_row as usize;
            kinds[row] = 1;
            weights[row] += stored(layer.output_size + gate);
        }
        for &boolean_row in &layer.boolean_rows {
            kinds[first_row + boolean_row as usize] = 2;
        }
        for read in &layer.index_reads {
            let row = first_row + read.row;
            kinds[row] = 3;
            weights[row] += stored(read.index_row);
        }
        first_row += layer.output_size;
    }
    assert!(first_row == row_count, "the layers hold every live operation");
    (0..row_count)
        .map(|row| OperationRow {
            wire: program.wire_order[program.source_size + row],
            kind: kinds[row],
            layer: layers[row],
            weights: weights[row],
        })
        .collect()
}

/// A cell's name as the State location shows it, with dots: 'windows.2.left'
/// for the table's 'windows[2].left'.
pub fn state_row_name(name: &str) -> String {
    let mut text = String::with_capacity(name.len());
    let mut rest = name;
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        match after.find(']') {
            Some(close) if close > 0 && after[..close].bytes().all(|byte| byte.is_ascii_digit()) => {
                text.push_str(&rest[..open]);
                text.push('.');
                text.push_str(&after[..close]);
                rest = &after[close + 1..];
            }
            _ => {
                text.push_str(&rest[..=open]);
                rest = after;
            }
        }
    }
    text.push_str(rest);
    text
}

/// validateValueListOutput: the checks the machine runs after every evaluation.
pub fn validate_value_list_output(values: &[f32], output: &ValueListOutputs) -> Result<(), String> {
    let value = |wire: Wire| values[wire.index()] as f64;
    let is_bit = |x: f64| x == 0.0 || x == 1.0;
    let total_height = value(output.total_height);
    if !(total_height >= 0.0 && (total_height / ROW_HEIGHT).fract() == 0.0) {
        return Err(format!("the list's total height must be a whole number of rows, got {}", js_number(total_height)));
    }
    for wire in [output.scroll, output.scroll_target] {
        if !(value(wire) >= 0.0 && value(wire) <= MAXIMUM_SCROLL_TOP) {
            return Err(format!(
                "a scroll position of the list must be within 0..{}, got {}",
                js_number(MAXIMUM_SCROLL_TOP),
                js_number(value(wire))
            ));
        }
    }
    if !is_bit(value(output.scroll_request)) || !is_bit(value(output.slot_overflow)) {
        return Err("the list's scroll request and slot overflow must equal zero or one".to_string());
    }
    let selected = value(output.selected);
    if !(selected.fract() == 0.0 && selected >= -1.0) {
        return Err(format!("the list's selected row must be a row or -1, got {}", js_number(selected)));
    }
    let mut mass = 0.0;
    for &wire in &output.locations {
        if !is_bit(value(wire)) {
            return Err(format!("a location bit of the list must equal zero or one, got {}", js_number(value(wire))));
        }
        mass += value(wire);
    }
    if mass != 1.0 {
        return Err(format!("the list must show exactly one location, got {}", js_number(mass)));
    }
    for slot in 0..SLOT_COUNT {
        let start = slot * SLOT_FIELD_SIZE;
        let row = value(output.slots[start]);
        if !(row.fract() == 0.0 && row >= 0.0 && row % SLOT_COUNT as f64 == slot as f64) {
            return Err(format!("slot {slot} of the list must show a row whose index % 64 is {slot}, got {}", js_number(row)));
        }
        if value(output.slots[start + 1]) != row * ROW_HEIGHT {
            return Err(format!("slot {slot} of the list must be at its row's top"));
        }
        if !is_bit(value(output.slots[start + 2])) || !is_bit(value(output.slots[start + 3])) {
            return Err(format!("slot {slot} of the list: present and selected must equal zero or one"));
        }
    }
    Ok(())
}
