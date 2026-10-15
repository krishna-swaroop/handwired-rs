// Photos, a port of the circuit-building parts of
// src/handwiredos/photo-gallery.ts: the cells, buildPhotos, the standalone
// circuit and its machine, and the output checks. The DOM renderer is left out.

use crate::cells::{constant, event, input, input_bit, state, state_bit, FinishedTable, SourceTable, ANY_NUMBER, UNIT_INTERVAL};
use crate::circuit::{Circuit, Term, Wire};
use crate::compile::{CompileOptions, TensorEvaluator, TensorProgram};
use crate::emit::{emit_maximum, emit_minimum, emit_multiply, emit_relu};
use crate::host_timing::{
    advance_frame, spring_matrix_bound, spring_matrix_power, spring_step_matrix, EvaluationKind, PlannedEvaluation,
    SpringClock, SpringConstants, SpringMatrix, TimedEvent, WriteBack, DEFAULT_SPRING_CLOCK_CONFIG, STOPPED_SPRING_CLOCK,
};
use crate::motion::{emit_animated_object, AnimatedObjectOptions, AnimatedSpring, SpringMatrixWires};
use crate::selection::emit_first_active;
use crate::validate::validate_circuit;
use crate::validate::validate_tensor_program;

pub const ITEM_COUNT: usize = 12;
pub const COLUMN_COUNT: usize = 4;
pub const ROW_COUNT: usize = 3;
pub const MAXIMUM_PHOTO_TALLNESS: f64 = 1.25;
pub const MAXIMUM_PHOTO_WIDENESS: f64 = 2.0;
pub const MAXIMUM_OPEN_PHOTO_TALLNESS: f64 = 2.0;
pub const MAXIMUM_OPEN_PHOTO_WIDENESS: f64 = 2.5;
pub const SPRING_CONSTANTS: SpringConstants = SpringConstants { stiffness: 650.0, damping: 45.0 };
const ZOOM_SETTLE_THRESHOLD: f64 = 1.0 / 65_536.0;
pub const GRID_FADE_RATE: f64 = 1.0 + 1.0 / 1024.0;

pub fn matrix_bound() -> f64 {
    spring_matrix_bound(SPRING_CONSTANTS, &DEFAULT_SPRING_CLOCK_CONFIG)
}

/// What the host writes into the 4 matrix cells for each step count, 0 to 300.
pub fn spring_matrix_powers() -> Vec<SpringMatrix> {
    let step = spring_step_matrix(SPRING_CONSTANTS, DEFAULT_SPRING_CLOCK_CONFIG.step_ms);
    (0..=DEFAULT_SPRING_CLOCK_CONFIG.max_steps_per_frame).map(|steps| spring_matrix_power(&step, steps)).collect()
}

#[derive(Clone, Debug)]
pub struct ThumbnailSpringCells {
    pub center_x_displacement: Wire,
    pub center_x_velocity: Wire,
    pub scale_displacement: Wire,
    pub scale_velocity: Wire,
}

#[derive(Clone, Debug)]
pub struct ZoomSpringCells {
    pub displacement: Wire,
    pub velocity: Wire,
}

/// Photos' own cells.
#[derive(Clone, Debug)]
pub struct PhotoCells {
    pub gap: Wire,
    pub main_inset: Wire,
    pub thumbnail_height: Wire,
    pub control_size: Wire,
    pub click_active: Wire,
    pub click_x: Wire,
    pub click_y: Wire,
    pub previous: Wire,
    pub next: Wire,
    pub close: Wire,
    pub focus: Vec<Wire>,
    pub thumbnails: Vec<ThumbnailSpringCells>,
    pub zoom_springs: Vec<ZoomSpringCells>,
    pub open_spring: Wire,
    pub closed_last: Vec<Wire>,
}

fn path(prefix: &str, field: &str) -> String {
    if prefix.is_empty() { field.to_string() } else { format!("{prefix}.{field}") }
}

/// photoCells up to and including the focus cells.
pub struct PhotoCellsHead {
    pub gap: Wire,
    pub main_inset: Wire,
    pub thumbnail_height: Wire,
    pub control_size: Wire,
    pub click_active: Wire,
    pub click_x: Wire,
    pub click_y: Wire,
    pub previous: Wire,
    pub next: Wire,
    pub close: Wire,
    pub focus: Vec<Wire>,
}

pub fn declare_photo_cells_head(table: &mut SourceTable, prefix: &str) -> PhotoCellsHead {
    let gap = table.cell(path(prefix, "gap"), constant(18.0, (0.0, 1024.0)));
    let main_inset = table.cell(path(prefix, "mainInset"), constant(92.0, (0.0, 1024.0)));
    let thumbnail_height = table.cell(path(prefix, "thumbnailHeight"), constant(82.0, (0.0, 1024.0)));
    let control_size = table.cell(path(prefix, "controlSize"), constant(52.0, (0.0, 1024.0)));
    let click_active = table.cell(path(prefix, "click.active"), event());
    let click_x = table.cell(path(prefix, "click.x"), input(ANY_NUMBER));
    let click_y = table.cell(path(prefix, "click.y"), input(ANY_NUMBER));
    let previous = table.cell(path(prefix, "previous"), event());
    let next = table.cell(path(prefix, "next"), event());
    let close = table.cell(path(prefix, "close"), event());
    let focus = table.one_hot(&path(prefix, "focus"), &[state_bit(0.0); ITEM_COUNT], true, false);
    PhotoCellsHead { gap, main_inset, thumbnail_height, control_size, click_active, click_x, click_y, previous, next, close, focus }
}

pub fn declare_thumbnail_cells(table: &mut SourceTable, prefix: &str) -> Vec<ThumbnailSpringCells> {
    (0..ITEM_COUNT)
        .map(|item| {
            let name = |field: &str| format!("{}[{item}].{field}", path(prefix, "thumbnails"));
            ThumbnailSpringCells {
                center_x_displacement: table.cell(name("centerXDisplacement"), state(ANY_NUMBER)),
                center_x_velocity: table.cell(name("centerXVelocity"), state(ANY_NUMBER)),
                scale_displacement: table.cell(name("scaleDisplacement"), state(ANY_NUMBER)),
                scale_velocity: table.cell(name("scaleVelocity"), state(ANY_NUMBER)),
            }
        })
        .collect()
}

pub fn declare_zoom_cells(table: &mut SourceTable, prefix: &str) -> (Vec<ZoomSpringCells>, Wire, Vec<Wire>) {
    let zoom = path(prefix, "zoom");
    let springs = (0..2)
        .map(|index| ZoomSpringCells {
            displacement: table.cell(format!("{zoom}.springs[{index}].displacement"), state(ANY_NUMBER)),
            velocity: table.cell(format!("{zoom}.springs[{index}].velocity"), state(ANY_NUMBER)),
        })
        .collect();
    let open_spring = table.cell(format!("{zoom}.openSpring"), state_bit(0.0));
    let closed_last = table.one_hot(&format!("{zoom}.closedLast"), &[state_bit(0.0); ITEM_COUNT], true, false);
    (springs, open_spring, closed_last)
}

/// Declares photoCells in their own order under `prefix`, e.g. 'photos'.
pub fn declare_photo_cells(table: &mut SourceTable, prefix: &str) -> PhotoCells {
    let head = declare_photo_cells_head(table, prefix);
    let thumbnails = declare_thumbnail_cells(table, prefix);
    let (zoom_springs, open_spring, closed_last) = declare_zoom_cells(table, prefix);
    assemble(head, thumbnails, zoom_springs, open_spring, closed_last)
}

fn assemble(
    head: PhotoCellsHead,
    thumbnails: Vec<ThumbnailSpringCells>,
    zoom_springs: Vec<ZoomSpringCells>,
    open_spring: Wire,
    closed_last: Vec<Wire>,
) -> PhotoCells {
    PhotoCells {
        gap: head.gap,
        main_inset: head.main_inset,
        thumbnail_height: head.thumbnail_height,
        control_size: head.control_size,
        click_active: head.click_active,
        click_x: head.click_x,
        click_y: head.click_y,
        previous: head.previous,
        next: head.next,
        close: head.close,
        focus: head.focus,
        thumbnails,
        zoom_springs,
        open_spring,
        closed_last,
    }
}

/// Photos' event cells: click, previous, next, close.
pub fn photo_events(cells: &PhotoCells) -> Vec<Wire> {
    vec![cells.click_active, cells.previous, cells.next, cells.close]
}

pub const PHOTO_FIELD_SIZE: usize = 7;
pub const PHOTO_FIELD_NAMES: [&str; 7] = ["left", "top", "width", "height", "opacity", "zoom", "stack"];
pub const THUMBNAIL_FIELD_SIZE: usize = 6;
pub const THUMBNAIL_FIELD_NAMES: [&str; 6] = ["left", "top", "width", "height", "opacity", "focused"];
pub const CONTROL_FIELD_SIZE: usize = 5;
pub const CONTROL_FIELD_NAMES: [&str; 5] = ["left", "top", "width", "height", "opacity"];
pub const PHOTO_STACK_RETURNING: f64 = 2.0;
pub const PHOTO_STACK_OPEN: f64 = 3.0;

#[derive(Clone, Debug)]
pub struct PhotoOutputs {
    pub backdrop_opacity: Wire,
    pub animation_active: Wire,
    pub photos: Vec<Wire>,
    pub thumbnails: Vec<Wire>,
    pub previous_control: Vec<Wire>,
    pub next_control: Vec<Wire>,
}

/// The host's wires that Photos reads.
pub struct PhotoHost {
    pub viewport_width: Wire,
    pub viewport_height: Wire,
    pub click_x: Wire,
    pub click_y: Wire,
    pub present: Wire,
    pub interactive: Wire,
    pub unit: Wire,
    pub spring_matrix: SpringMatrixWires,
    pub settle_check: Wire,
    pub snap: Wire,
}

#[derive(Copy, Clone)]
struct Geometry {
    left: Wire,
    top: Wire,
    width: Wire,
    height: Wire,
}

#[derive(Copy, Clone)]
struct ThumbnailTarget {
    center_x: Wire,
    scale: Wire,
}

fn sum(circuit: &mut Circuit, terms: &[Term]) -> Wire {
    circuit.affine(terms, 0.0)
}

fn thumbnail_falloff(distance: f64) -> f64 {
    0f64.max(1.0 - 0.22 * distance)
}

fn thumbnail_falloff_mass() -> f64 {
    let mut mass = thumbnail_falloff(0.0);
    for distance in 1..5 {
        mass += 2.0 * thumbnail_falloff(distance as f64);
    }
    mass
}

fn thumbnail_spread(distance: f64) -> f64 {
    let mut spread = 0.0;
    let mut step = 1.0;
    while step <= distance {
        spread += (thumbnail_falloff(step - 1.0) + thumbnail_falloff(step)) / 2.0;
        step += 1.0;
    }
    spread
}

fn js_sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        value
    }
}

fn emit_proportion_excess(circuit: &mut Circuit, width: Wire, height: Wire, maximum_wideness: f64, maximum_tallness: f64) -> (Wire, Wire) {
    let wide = sum(circuit, &[(width, 1.0), (height, -maximum_wideness)]);
    let excess_width = emit_relu(circuit, wide);
    let tall = sum(circuit, &[(height, 1.0), (width, -maximum_tallness)]);
    let excess_height = emit_relu(circuit, tall);
    (excess_width, excess_height)
}

fn emit_without_excess(circuit: &mut Circuit, size: Wire, excess: Wire) -> Wire {
    let result = sum(circuit, &[(size, 1.0), (excess, -1.0)]);
    circuit.assume_range(result, 0.0, 16_384.0);
    result
}

fn emit_rectangle_hit(circuit: &mut Circuit, click: (Wire, Wire), geometry: Geometry) -> Wire {
    let horizontal = emit_interval_hit(circuit, click.0, geometry.left, geometry.width);
    let vertical = emit_interval_hit(circuit, click.1, geometry.top, geometry.height);
    emit_minimum(circuit, horizontal, vertical)
}

fn emit_interval_hit(circuit: &mut Circuit, position: Wire, start: Wire, size: Wire) -> Wire {
    let distance = emit_interval_distance(circuit, position, start, size);
    let clamped = emit_clamp_zero_one(circuit, distance);
    circuit.boolean(clamped)
}

/// How far the position lies inside the interval from start to start + size.
pub fn emit_interval_distance(circuit: &mut Circuit, position: Wire, start: Wire, size: Wire) -> Wire {
    let end = sum(circuit, &[(start, 1.0), (size, 1.0)]);
    let from_start = sum(circuit, &[(position, 1.0), (start, -1.0)]);
    let to_end = sum(circuit, &[(end, 1.0), (position, -1.0)]);
    emit_minimum(circuit, from_start, to_end)
}

fn emit_clamp_zero_one(circuit: &mut Circuit, value: Wire) -> Wire {
    let positive = emit_relu(circuit, value);
    let shifted = circuit.affine(&[(value, 1.0)], -1.0);
    let above_one = emit_relu(circuit, shifted);
    let result = sum(circuit, &[(positive, 1.0), (above_one, -1.0)]);
    circuit.assume_range(result, 0.0, 1.0);
    result
}

fn emit_both(circuit: &mut Circuit, first: Wire, second: Wire) -> Wire {
    circuit.reglu(&[], 1.0, &[(first, 1.0), (second, 1.0)], -1.0)
}

fn emit_first_only(circuit: &mut Circuit, first: Wire, second: Wire) -> Wire {
    circuit.reglu(&[], 1.0, &[(first, 1.0), (second, -1.0)], 0.0)
}

/// Builds Photos into the host's circuit, pairs Photos' state cells with their
/// next values, and returns Photos' outputs.
pub fn build_photos(circuit: &mut Circuit, table: &mut SourceTable, cells: &PhotoCells, host: &PhotoHost) -> PhotoOutputs {
    let matrix_bound = matrix_bound();
    circuit.assume_range(host.viewport_width, 0.0, 16_384.0);
    circuit.assume_range(host.viewport_height, 0.0, 16_384.0);
    circuit.assume_range(host.present, 0.0, 1.0);
    circuit.assume_range(host.interactive, 0.0, 1.0);
    circuit.assume_range(host.unit, 1.0, 1.0);
    for entry in host.spring_matrix {
        circuit.assume_range(entry, -matrix_bound, matrix_bound);
    }
    circuit.assume_range(host.settle_check, 0.0, 1.0);
    circuit.assume_range(host.snap, 0.0, 1.0);
    let (viewport_width, viewport_height, present, interactive, unit) =
        (host.viewport_width, host.viewport_height, host.present, host.interactive, host.unit);
    let click = (host.click_x, host.click_y);
    let events = photo_events(cells);
    let zero = circuit.constant(0.0);
    let columns = COLUMN_COUNT as f64;
    let rows = ROW_COUNT as f64;
    let slot_width_sum = sum(circuit, &[(viewport_width, 1.0 / columns), (cells.gap, -(columns + 1.0) / columns)]);
    let slot_width = emit_relu(circuit, slot_width_sum);
    let slot_height_sum = sum(circuit, &[(viewport_height, 1.0 / rows), (cells.gap, -(rows + 1.0) / rows)]);
    let slot_height = emit_relu(circuit, slot_height_sum);
    let slot_excess =
        emit_proportion_excess(circuit, slot_width, slot_height, MAXIMUM_PHOTO_WIDENESS, MAXIMUM_PHOTO_TALLNESS);
    let grid_width = emit_without_excess(circuit, slot_width, slot_excess.0);
    let grid_height = emit_without_excess(circuit, slot_height, slot_excess.1);
    let grid_origin_x = sum(circuit, &[(slot_excess.0, columns / 2.0)]);
    let grid_origin_y = sum(circuit, &[(slot_excess.1, rows / 2.0)]);

    let grid_left: Vec<Wire> = (0..COLUMN_COUNT)
        .map(|column| {
            let column = column as f64;
            sum(circuit, &[(grid_origin_x, 1.0), (cells.gap, 1.0 + column), (grid_width, column)])
        })
        .collect();
    let grid_top: Vec<Wire> = (0..ROW_COUNT)
        .map(|row| {
            let row = row as f64;
            sum(circuit, &[(grid_origin_y, 1.0), (cells.gap, 1.0 + row), (grid_height, row)])
        })
        .collect();
    let grid_column_hit: Vec<Wire> =
        grid_left.iter().map(|&left| emit_interval_hit(circuit, click.0, left, grid_width)).collect();
    let grid_row_hit: Vec<Wire> = grid_top.iter().map(|&top| emit_interval_hit(circuit, click.1, top, grid_height)).collect();
    let grid_hit: Vec<Wire> = (0..ITEM_COUNT)
        .map(|item| emit_minimum(circuit, grid_column_hit[item % COLUMN_COUNT], grid_row_hit[item / COLUMN_COUNT]))
        .collect();

    let sixth = sum(circuit, &[(viewport_width, 1.0 / 6.0)]);
    let quarter_height = sum(circuit, &[(viewport_height, 0.24)]);
    let smaller = emit_minimum(circuit, sixth, quarter_height);
    let thumbnail_base_size = emit_minimum(circuit, cells.thumbnail_height, smaller);
    let inset_share = sum(circuit, &[(viewport_width, 0.08)]);
    let main_inset = emit_minimum(circuit, cells.main_inset, inset_share);
    let main_box_width_sum = sum(circuit, &[(viewport_width, 1.0), (main_inset, -2.0)]);
    let main_box_width = emit_relu(circuit, main_box_width_sum);
    let main_box_height_sum = sum(circuit, &[(viewport_height, 1.0), (thumbnail_base_size, -1.0), (cells.gap, -3.0)]);
    let main_box_height = emit_relu(circuit, main_box_height_sum);
    let main_excess =
        emit_proportion_excess(circuit, main_box_width, main_box_height, MAXIMUM_OPEN_PHOTO_WIDENESS, MAXIMUM_OPEN_PHOTO_TALLNESS);
    let main_left = sum(circuit, &[(main_inset, 1.0), (main_excess.0, 0.5)]);
    let main_top = sum(circuit, &[(cells.gap, 1.0), (main_excess.1, 0.5)]);
    let main_width = emit_without_excess(circuit, main_box_width, main_excess.0);
    let main_height = emit_without_excess(circuit, main_box_height, main_excess.1);
    let main = Geometry { left: main_left, top: main_top, width: main_width, height: main_height };
    let main_hit = emit_rectangle_hit(circuit, click, main);

    let focus_terms: Vec<Term> = cells.focus.iter().map(|&wire| (wire, 1.0)).collect();
    let previous_focus_mass = sum(circuit, &focus_terms);
    circuit.assume_range(previous_focus_mass, 0.0, 1.0);
    let previous_grid_mass = circuit.affine(&[(previous_focus_mass, -1.0)], 1.0);

    let half_width = sum(circuit, &[(viewport_width, 0.5)]);
    let thumbnail_outer_inset = emit_minimum(circuit, cells.gap, half_width);
    let thumbnail_image_span = sum(circuit, &[(thumbnail_base_size, thumbnail_falloff_mass())]);
    let twelve = circuit.constant(12.0);
    let gap_room = sum(
        circuit,
        &[(viewport_width, 1.0 / 8.0), (thumbnail_outer_inset, -2.0 / 8.0), (thumbnail_image_span, -1.0 / 8.0)],
    );
    let gap_room = emit_relu(circuit, gap_room);
    let thumbnail_gap = emit_minimum(circuit, twelve, gap_room);
    let thumbnail_center_y = sum(circuit, &[(viewport_height, 1.0), (cells.gap, -1.0), (thumbnail_base_size, -0.5)]);

    let project_thumbnail_targets = |circuit: &mut Circuit, focus: &[Wire], focus_mass: Wire| -> Vec<ThumbnailTarget> {
        (0..ITEM_COUNT)
            .map(|item| {
                let mut offset_terms = Vec::new();
                let mut distance_terms = Vec::new();
                let mut spread_terms = Vec::new();
                for (focused_item, &focused) in focus.iter().enumerate() {
                    let mut offset = item as f64 - focused_item as f64;
                    if offset > ITEM_COUNT as f64 / 2.0 {
                        offset -= ITEM_COUNT as f64;
                    }
                    if offset < -(ITEM_COUNT as f64) / 2.0 {
                        offset += ITEM_COUNT as f64;
                    }
                    let visible_offset = if offset.abs() < 5.0 { offset } else { 0.0 };
                    offset_terms.push((focused, visible_offset));
                    distance_terms.push((focused, offset.abs()));
                    spread_terms.push((focused, js_sign(visible_offset) * thumbnail_spread(visible_offset.abs())));
                }
                let offset = sum(circuit, &offset_terms);
                let distance = sum(circuit, &distance_terms);
                let spread = sum(circuit, &spread_terms);
                let falloff_sum = sum(circuit, &[(focus_mass, 1.0), (distance, -0.22)]);
                let falloff = emit_relu(circuit, falloff_sum);
                let spread_px = emit_multiply(circuit, thumbnail_base_size, spread);
                let offset_px = emit_multiply(circuit, thumbnail_gap, offset);
                let center_x = sum(circuit, &[(viewport_width, 0.5), (spread_px, 1.0), (offset_px, 1.0)]);
                ThumbnailTarget { center_x, scale: falloff }
            })
            .collect()
    };

    let project_thumbnail = |circuit: &mut Circuit, target: ThumbnailTarget| -> (Geometry, Wire) {
        let visible_scale = emit_multiply(circuit, present, target.scale);
        let size = emit_multiply(circuit, thumbnail_base_size, visible_scale);
        let left = sum(circuit, &[(target.center_x, 1.0), (size, -0.5)]);
        let top_sum = sum(circuit, &[(thumbnail_center_y, 1.0), (size, -0.5)]);
        let top = emit_relu(circuit, top_sum);
        (Geometry { left, top, width: size, height: size }, visible_scale)
    };

    let current_thumbnail_targets = project_thumbnail_targets(circuit, &cells.focus, previous_focus_mass);
    let thumbnail_hit: Vec<Wire> = current_thumbnail_targets
        .iter()
        .map(|&target| {
            let (geometry, _) = project_thumbnail(circuit, target);
            emit_rectangle_hit(circuit, click, geometry)
        })
        .collect();

    let control_top_sum = sum(circuit, &[(cells.gap, 1.0), (main_box_height, 0.5), (cells.control_size, -0.5)]);
    let control_top = emit_relu(circuit, control_top_sum);
    let control_inset = sum(circuit, &[(cells.gap, 0.5)]);
    let previous_control =
        Geometry { left: control_inset, top: control_top, width: cells.control_size, height: cells.control_size };
    let next_left_sum = sum(circuit, &[(viewport_width, 1.0), (control_inset, -1.0), (cells.control_size, -1.0)]);
    let next_left = emit_relu(circuit, next_left_sum);
    let next_control = Geometry { left: next_left, top: control_top, width: cells.control_size, height: cells.control_size };
    let control_vertical_hit = emit_interval_hit(circuit, click.1, control_top, cells.control_size);
    let previous_horizontal = emit_interval_hit(circuit, click.0, previous_control.left, previous_control.width);
    let previous_control_hit = emit_minimum(circuit, previous_horizontal, control_vertical_hit);
    let next_horizontal = emit_interval_hit(circuit, click.0, next_control.left, next_control.width);
    let next_control_hit = emit_minimum(circuit, next_horizontal, control_vertical_hit);

    let effective_events: Vec<Wire> = events.iter().map(|&wire| emit_multiply(circuit, interactive, wire)).collect();
    let event_terms: Vec<Term> = effective_events.iter().map(|&wire| (wire, 1.0)).collect();
    let event_mass = sum(circuit, &event_terms);
    circuit.assume_range(event_mass, 0.0, 1.0);
    let event_inactive = circuit.affine(&[(event_mass, -1.0)], 1.0);
    let click_active = effective_events[0];
    let click_grid_gate = emit_multiply(circuit, click_active, previous_grid_mass);
    let focused_click = emit_multiply(circuit, click_active, previous_focus_mass);
    circuit.assume_range(focused_click, 0.0, 1.0);
    let mut surfaces = vec![previous_control_hit, next_control_hit];
    surfaces.extend(&thumbnail_hit);
    surfaces.push(main_hit);
    let selected = emit_first_active(circuit, focused_click, &surfaces);
    let previous_control_click = selected[0];
    let next_control_click = selected[1];
    let thumbnail_clicks = &selected[2..2 + ITEM_COUNT];
    let click_main_gate = selected[2 + ITEM_COUNT];
    let current_focus_gate = sum(circuit, &[(event_inactive, 1.0), (click_main_gate, 1.0)]);
    let previous_focus_gate = sum(circuit, &[(effective_events[1], 1.0), (previous_control_click, 1.0)]);
    let next_focus_gate = sum(circuit, &[(effective_events[2], 1.0), (next_control_click, 1.0)]);
    let next_focus: Vec<Wire> = (0..ITEM_COUNT)
        .map(|item| {
            let previous_shift = cells.focus[(item + 1) % ITEM_COUNT];
            let next_shift = cells.focus[(item + ITEM_COUNT - 1) % ITEM_COUNT];
            let kept = emit_multiply(circuit, current_focus_gate, cells.focus[item]);
            let clicked = emit_multiply(circuit, click_grid_gate, grid_hit[item]);
            let previous = emit_multiply(circuit, previous_focus_gate, previous_shift);
            let next = emit_multiply(circuit, next_focus_gate, next_shift);
            sum(circuit, &[(kept, 1.0), (clicked, 1.0), (thumbnail_clicks[item], 1.0), (previous, 1.0), (next, 1.0)])
        })
        .collect();
    let next_focus_terms: Vec<Term> = next_focus.iter().map(|&wire| (wire, 1.0)).collect();
    let next_focus_mass = sum(circuit, &next_focus_terms);
    circuit.assume_range(next_focus_mass, 0.0, 1.0);

    // The zoom, see "The zoom" in photo-gallery.ts.
    let closing = emit_first_only(circuit, previous_focus_mass, next_focus_mass);
    let closed_last_terms: Vec<Term> = next_focus
        .iter()
        .enumerate()
        .map(|(item, &wire)| (emit_both(circuit, wire, cells.closed_last[item]), 1.0))
        .collect();
    let closed_last_is_open = sum(circuit, &closed_last_terms);
    circuit.assume_range(closed_last_is_open, 0.0, 1.0);
    let reopening = emit_first_only(circuit, closed_last_is_open, previous_focus_mass);
    let swap = sum(circuit, &[(closing, 1.0), (reopening, 1.0)]);
    circuit.assume_range(swap, 0.0, 1.0);
    let open_without_swap = emit_first_only(circuit, cells.open_spring, swap);
    let swap_without_open = emit_first_only(circuit, swap, cells.open_spring);
    let next_open_spring = sum(circuit, &[(open_without_swap, 1.0), (swap_without_open, 1.0)]);
    circuit.assume_range(next_open_spring, 0.0, 1.0);
    table.next(circuit, &[cells.open_spring], &[next_open_spring]);
    let stepped_onto_closed_last = emit_both(circuit, previous_focus_mass, closed_last_is_open);
    let other_spring_stops = sum(circuit, &[(closing, 1.0), (stepped_onto_closed_last, 1.0)]);
    circuit.assume_range(other_spring_stops, 0.0, 1.0);
    struct ZoomSpring {
        is_open_spring: Wire,
        zoom: Wire,
        active: Wire,
    }
    let zoom_springs: Vec<ZoomSpring> = cells
        .zoom_springs
        .iter()
        .enumerate()
        .map(|(index, spring)| {
            let was_open_spring =
                if index == 0 { circuit.affine(&[(cells.open_spring, -1.0)], 1.0) } else { cells.open_spring };
            let is_open_spring = if index == 0 { circuit.affine(&[(next_open_spring, -1.0)], 1.0) } else { next_open_spring };
            let stops = emit_first_only(circuit, other_spring_stops, was_open_spring);
            let keeps_moving = circuit.reglu(&[], 1.0, &[(host.snap, -1.0), (stops, -1.0)], 1.0);
            let snap = circuit.affine(&[(keeps_moving, -1.0)], 1.0);
            circuit.assume_range(snap, 0.0, 1.0);
            let destination = emit_both(circuit, is_open_spring, next_focus_mass);
            let destination_during_steps = emit_both(circuit, was_open_spring, previous_focus_mass);
            let object = emit_animated_object(
                circuit,
                &[AnimatedSpring {
                    matrix: host.spring_matrix,
                    displacement: spring.displacement,
                    velocity: spring.velocity,
                    destination,
                    destination_during_steps: Some(destination_during_steps),
                }],
                AnimatedObjectOptions {
                    settle_check: host.settle_check,
                    snap: Some(snap),
                    settle_distance: Some(ZOOM_SETTLE_THRESHOLD),
                    settle_speed: Some(ZOOM_SETTLE_THRESHOLD),
                },
            );
            table.next(
                circuit,
                &[spring.displacement, spring.velocity],
                &[object.displacements[0], object.velocities[0]],
            );
            let zoom = emit_clamp_zero_one(circuit, object.positions[0]);
            ZoomSpring { is_open_spring, zoom, active: object.active }
        })
        .collect();
    let (first, second) = (&zoom_springs[0], &zoom_springs[1]);
    let first_open = emit_multiply(circuit, first.zoom, first.is_open_spring);
    let second_open = emit_multiply(circuit, second.zoom, second.is_open_spring);
    let open_zoom = sum(circuit, &[(first_open, 1.0), (second_open, 1.0)]);
    circuit.assume_range(open_zoom, 0.0, 1.0);
    let first_return = emit_multiply(circuit, first.zoom, second.is_open_spring);
    let second_return = emit_multiply(circuit, second.zoom, first.is_open_spring);
    let return_zoom = sum(circuit, &[(first_return, 1.0), (second_return, 1.0)]);
    circuit.assume_range(return_zoom, 0.0, 1.0);
    let first_returning = emit_both(circuit, first.active, second.is_open_spring);
    let second_returning = emit_both(circuit, second.active, first.is_open_spring);
    let returning = sum(circuit, &[(first_returning, 1.0), (second_returning, 1.0)]);
    circuit.assume_range(returning, 0.0, 1.0);
    let next_closed_last: Vec<Wire> = cells
        .closed_last
        .iter()
        .enumerate()
        .map(|(item, &closed_last)| {
            let added = circuit.reglu(&[(cells.focus[item], 1.0), (closed_last, -1.0)], 0.0, &[(closing, 1.0)], 0.0);
            let next = sum(circuit, &[(closed_last, 1.0), (added, 1.0)]);
            circuit.assume_range(next, 0.0, 1.0);
            next
        })
        .collect();
    table.next(circuit, &cells.closed_last, &next_closed_last);
    let zoom_sum = sum(circuit, &[(open_zoom, 1.0), (return_zoom, 1.0)]);
    let zoom_excess_sum = circuit.affine(&[(zoom_sum, 1.0)], -1.0);
    let zoom_excess = emit_relu(circuit, zoom_excess_sum);
    let lightbox_opacity = sum(circuit, &[(zoom_sum, 1.0), (zoom_excess, -1.0)]);
    circuit.assume_range(lightbox_opacity, 0.0, 1.0);
    let returning_share = sum(circuit, &[(lightbox_opacity, 1.0), (open_zoom, -1.0)]);
    circuit.assume_range(returning_share, 0.0, 1.0);
    let lightbox_present = emit_multiply(circuit, present, lightbox_opacity);

    let mut photos = Vec::with_capacity(ITEM_COUNT * PHOTO_FIELD_SIZE);
    let mut photo_zooms = Vec::with_capacity(ITEM_COUNT);
    for item in 0..ITEM_COUNT {
        let column = item % COLUMN_COUNT;
        let row = item / COLUMN_COUNT;
        let focused = next_focus[item];
        let closed_last = next_closed_last[item];
        let zoom_as_open_photo = emit_multiply(circuit, open_zoom, focused);
        let returned = circuit.reglu(&[(return_zoom, 1.0)], 0.0, &[(closed_last, 1.0), (focused, -1.0)], 0.0);
        let zoom = sum(circuit, &[(zoom_as_open_photo, 1.0), (returned, 1.0)]);
        circuit.assume_range(zoom, 0.0, 1.0);
        let between = |circuit: &mut Circuit, cell_value: Wire, open_value: Wire| {
            let kept = circuit.reglu_preserving(&[(cell_value, 1.0)], 0.0, &[(zoom, -1.0)], 1.0);
            let moved = emit_multiply(circuit, open_value, zoom);
            sum(circuit, &[(kept, 1.0), (moved, 1.0)])
        };
        // The fades come before the rectangle, as in photo-gallery.ts.
        let faded_by_open = circuit.reglu(&[], 1.0, &[(open_zoom, 1.0), (focused, -1.0)], 0.0);
        let faded_by_returning =
            circuit.reglu(&[], 1.0, &[(returning_share, 1.0), (zoom_as_open_photo, -1.0), (closed_last, -1.0)], 0.0);
        let left = between(circuit, grid_left[column], main.left);
        let top = between(circuit, grid_top[row], main.top);
        let width = between(circuit, grid_width, main.width);
        let height = between(circuit, grid_height, main.height);
        let opacity = circuit.reglu(
            &[(present, 1.0)],
            0.0,
            &[(faded_by_open, -GRID_FADE_RATE), (faded_by_returning, -GRID_FADE_RATE)],
            1.0,
        );
        let raised = circuit.reglu(&[], 1.0, &[(closed_last, 1.0), (returning, 1.0), (focused, -1.0)], -1.0);
        let stack = sum(circuit, &[(focused, PHOTO_STACK_OPEN), (raised, PHOTO_STACK_RETURNING)]);
        photos.extend([left, top, width, height, opacity, zoom, stack]);
        photo_zooms.push(zoom);
    }

    let thumbnail_targets = project_thumbnail_targets(circuit, &next_focus, next_focus_mass);
    let mut thumbnails = Vec::with_capacity(ITEM_COUNT * THUMBNAIL_FIELD_SIZE);
    let mut activity_terms: Vec<Term> = vec![(first.active, 1.0), (second.active, 1.0)];
    for item in 0..ITEM_COUNT {
        let before = current_thumbnail_targets[item];
        let target = thumbnail_targets[item];
        let springs = &cells.thumbnails[item];
        let thumbnail = emit_animated_object(
            circuit,
            &[
                AnimatedSpring {
                    matrix: host.spring_matrix,
                    displacement: springs.center_x_displacement,
                    velocity: springs.center_x_velocity,
                    destination: target.center_x,
                    destination_during_steps: Some(before.center_x),
                },
                AnimatedSpring {
                    matrix: host.spring_matrix,
                    displacement: springs.scale_displacement,
                    velocity: springs.scale_velocity,
                    destination: target.scale,
                    destination_during_steps: Some(before.scale),
                },
            ],
            AnimatedObjectOptions { settle_check: host.settle_check, snap: Some(host.snap), settle_distance: None, settle_speed: None },
        );
        let center_above = emit_maximum(circuit, zero, thumbnail.positions[0]);
        let center_x = emit_minimum(circuit, center_above, viewport_width);
        let scale_above = emit_maximum(circuit, zero, thumbnail.positions[1]);
        let scale = emit_minimum(circuit, scale_above, unit);
        table.next(
            circuit,
            &[springs.center_x_displacement, springs.center_x_velocity, springs.scale_displacement, springs.scale_velocity],
            &[thumbnail.displacements[0], thumbnail.velocities[0], thumbnail.displacements[1], thumbnail.velocities[1]],
        );
        let (geometry, opacity) = project_thumbnail(circuit, ThumbnailTarget { center_x, scale });
        thumbnails.extend([geometry.left, geometry.top, geometry.width, geometry.height, opacity, photo_zooms[item]]);
        activity_terms.push((thumbnail.active, 1.0));
    }
    let activity = sum(circuit, &activity_terms);
    let animation_active = emit_clamp_zero_one(circuit, activity);

    table.next(circuit, &cells.focus, &next_focus);
    let previous_control_output = vec![
        previous_control.left,
        previous_control.top,
        previous_control.width,
        previous_control.height,
        lightbox_present,
    ];
    let next_control_output =
        vec![next_control.left, next_control.top, next_control.width, next_control.height, lightbox_present];
    PhotoOutputs {
        backdrop_opacity: lightbox_present,
        animation_active,
        photos,
        thumbnails,
        previous_control: previous_control_output,
        next_control: next_control_output,
    }
}

/// The standalone gallery's cells besides Photos' own.
#[derive(Clone, Debug)]
pub struct PhotoGalleryCells {
    pub viewport_width: Wire,
    pub viewport_height: Wire,
    pub photos: PhotoCells,
    pub present: Wire,
    pub interactive: Wire,
    pub unit: Wire,
    pub spring_matrix: SpringMatrixWires,
    pub settle_check: Wire,
    pub snap: Wire,
}

pub fn declare_spring_matrix(table: &mut SourceTable, prefix: &str, bound: f64) -> SpringMatrixWires {
    let name = path(prefix, "springMatrix");
    let range = (-bound, bound);
    [
        table.cell(format!("{name}[0]"), state(range).initial(1.0)),
        table.cell(format!("{name}[1]"), state(range)),
        table.cell(format!("{name}[2]"), state(range)),
        table.cell(format!("{name}[3]"), state(range).initial(1.0)),
    ]
}

/// The standalone gallery's circuit, see createPhotoGalleryCircuit.
pub struct PhotoGalleryCircuit {
    pub circuit: Circuit,
    pub table: SourceTable,
    pub cells: PhotoGalleryCells,
    pub output: PhotoOutputs,
    pub finished: FinishedTable,
}

pub fn create_photo_gallery_circuit() -> PhotoGalleryCircuit {
    let mut table = SourceTable::new();
    let viewport_width = table.cell("viewportWidth", input((0.0, 16_384.0)));
    let viewport_height = table.cell("viewportHeight", input((0.0, 16_384.0)));
    let head = declare_photo_cells_head(&mut table, "");
    let present = table.cell("present", input(UNIT_INTERVAL).initial(1.0));
    let interactive = table.cell("interactive", input_bit(1.0));
    let unit = table.cell("unit", constant(1.0, (1.0, 1.0)));
    let spring_matrix = declare_spring_matrix(&mut table, "", matrix_bound());
    let settle_check = table.cell("settleCheck", state_bit(0.0));
    let snap = table.cell("snap", state_bit(0.0));
    let thumbnails = declare_thumbnail_cells(&mut table, "");
    let (zoom_springs, open_spring, closed_last) = declare_zoom_cells(&mut table, "");
    let photos = assemble(head, thumbnails, zoom_springs, open_spring, closed_last);
    table.at_most_one("events", &photo_events(&photos));
    let mut circuit = table.create_circuit();
    let host = PhotoHost {
        viewport_width,
        viewport_height,
        click_x: photos.click_x,
        click_y: photos.click_y,
        present,
        interactive,
        unit,
        spring_matrix,
        settle_check,
        snap,
    };
    let output = build_photos(&mut circuit, &mut table, &photos, &host);
    let zero = circuit.constant(0.0);
    table.next(
        &circuit,
        &[spring_matrix[0], spring_matrix[1], spring_matrix[2], spring_matrix[3], settle_check, snap],
        &[unit, zero, zero, unit, zero, zero],
    );
    let finished = table.finish(&mut circuit);
    let cells = PhotoGalleryCells {
        viewport_width,
        viewport_height,
        photos,
        present,
        interactive,
        unit,
        spring_matrix,
        settle_check,
        snap,
    };
    PhotoGalleryCircuit { circuit, table, cells, output, finished }
}

fn check(condition: bool, message: impl FnOnce() -> String) -> Result<(), String> {
    if condition { Ok(()) } else { Err(message()) }
}

pub(crate) fn assert_unit_interval(value: f32, label: &str, tolerance: f64) -> Result<(), String> {
    let value = value as f64;
    check(value.is_finite() && value >= -tolerance && value <= 1.0 + tolerance, || {
        format!("{label} must be within zero and one, got {}", crate::fp::js_number(value))
    })
}

pub(crate) fn assert_non_negative_finite(value: f32, label: &str, tolerance: f64) -> Result<(), String> {
    let value = value as f64;
    check(value.is_finite() && value >= -tolerance, || {
        format!("{label} must be non-negative and finite, got {}", crate::fp::js_number(value))
    })
}

/// validatePhotoGalleryOutput.
pub fn validate_photo_gallery_output(values: &[f32], output: &PhotoOutputs, tolerance: f64) -> Result<(), String> {
    let value = |wire: Wire| values[wire.index()];
    assert_unit_interval(value(output.backdrop_opacity), "backdrop opacity", tolerance)?;
    assert_unit_interval(value(output.animation_active), "animation active", tolerance)?;
    for item in 0..ITEM_COUNT {
        let start = item * PHOTO_FIELD_SIZE;
        for field in 0..4 {
            assert_non_negative_finite(value(output.photos[start + field]), &format!("photo {}", PHOTO_FIELD_NAMES[field]), 0.0)?;
        }
        assert_unit_interval(value(output.photos[start + 4]), "photo opacity", tolerance)?;
        assert_unit_interval(value(output.photos[start + 5]), "photo zoom", tolerance)?;
        let stack = value(output.photos[start + 6]) as f64;
        check(stack == 0.0 || stack == PHOTO_STACK_RETURNING || stack == PHOTO_STACK_OPEN, || {
            format!("photo stack must be 0, 2 or 3, got {}", crate::fp::js_number(stack))
        })?;
    }
    for item in 0..ITEM_COUNT {
        let start = item * THUMBNAIL_FIELD_SIZE;
        for field in 0..4 {
            let field_value = value(output.thumbnails[start + field]);
            if field == 0 {
                check(field_value.is_finite(), || "thumbnail left must be finite".to_string())?;
            } else {
                assert_non_negative_finite(field_value, &format!("thumbnail {}", THUMBNAIL_FIELD_NAMES[field]), 0.0)?;
            }
        }
        assert_unit_interval(value(output.thumbnails[start + 4]), "thumbnail opacity", tolerance)?;
        assert_unit_interval(value(output.thumbnails[start + 5]), "thumbnail focus", tolerance)?;
    }
    for control in [&output.previous_control, &output.next_control] {
        for field in 0..4 {
            assert_non_negative_finite(value(control[field]), &format!("control {}", CONTROL_FIELD_NAMES[field]), 0.0)?;
        }
        assert_unit_interval(value(control[4]), "control opacity", tolerance)?;
    }
    Ok(())
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PhotoGalleryEventKind {
    Mount,
    Resize,
    Click,
    Previous,
    Next,
    Close,
}

impl PhotoGalleryEventKind {
    pub fn name(self) -> &'static str {
        match self {
            PhotoGalleryEventKind::Mount => "mount",
            PhotoGalleryEventKind::Resize => "resize",
            PhotoGalleryEventKind::Click => "click",
            PhotoGalleryEventKind::Previous => "previous",
            PhotoGalleryEventKind::Next => "next",
            PhotoGalleryEventKind::Close => "close",
        }
    }
}

#[derive(Clone, Debug)]
pub struct PhotoGalleryEvent {
    pub kind: PhotoGalleryEventKind,
    pub time_stamp: f64,
    /// The cell writes the event makes, in order.
    pub writes: Vec<(Wire, f32)>,
    /// Clears every event cell before the writes.
    pub clears_events: bool,
}

impl TimedEvent for PhotoGalleryEvent {
    fn time_stamp(&self) -> f64 {
        self.time_stamp
    }
}

/// One evaluation of a step, as the machine records it.
#[derive(Clone, Debug, PartialEq)]
pub struct MachineEvaluation {
    pub kind: EvaluationKind,
    pub time_ms: f64,
    pub steps: u32,
    pub writes_back: bool,
    pub events: Vec<&'static str>,
}

/// Called with the table just before each evaluation.
pub type BeforeEvaluate<Event> = Box<dyn FnMut(&[f32], &PlannedEvaluation<Event>)>;
/// Called with the values, indexed by wire, and the table after each evaluation.
pub type AfterEvaluate<Event> = Box<dyn FnMut(&[f32], &[f32], &PlannedEvaluation<Event>)>;

/// createPhotoGalleryMachine: Photos alone, driven by advance_frame.
pub struct PhotoGalleryMachine {
    pub circuit: Circuit,
    pub cells: PhotoGalleryCells,
    pub output: PhotoOutputs,
    pub source: Vec<f32>,
    pub tensor_program: TensorProgram,
    pub roots: Vec<Wire>,
    pub values: Vec<f32>,
    pub clock: SpringClock,
    pub evaluations: Vec<MachineEvaluation>,
    validator: crate::cells::TableValidator,
    evaluator: TensorEvaluator,
    write_back: WriteBack,
    queued: Vec<(PhotoGalleryEvent, bool)>,
    mounted: bool,
    queued_viewport: (f64, f64),
    previous_frame_time: f64,
    present: f32,
    interactive: f32,
    powers: Vec<SpringMatrix>,
    stored_after_last_step: Vec<f32>,
    /// Called with the table just before each evaluation.
    pub before_evaluate: Option<BeforeEvaluate<PhotoGalleryEvent>>,
    /// Called with the values and the table after each evaluation.
    pub after_evaluate: Option<AfterEvaluate<PhotoGalleryEvent>>,
}

impl PhotoGalleryMachine {
    pub fn new(write_back: WriteBack) -> PhotoGalleryMachine {
        let PhotoGalleryCircuit { circuit, cells, output, finished, .. } = create_photo_gallery_circuit();
        let wire_count = validate_circuit(&circuit).expect("the gallery's circuit is valid");
        let mut roots = finished.next_source.clone();
        roots.push(output.backdrop_opacity);
        roots.push(output.animation_active);
        roots.extend(&output.photos);
        roots.extend(&output.thumbnails);
        roots.extend(&output.previous_control);
        roots.extend(&output.next_control);
        let tensor_program = circuit.compile_tensor_program(Some(&roots), true, CompileOptions::default());
        validate_tensor_program(&circuit, &tensor_program).expect("the gallery's program is valid");
        let evaluator = circuit.create_tensor_program_evaluator(tensor_program.clone());
        let source = finished.initial_source.clone();
        PhotoGalleryMachine {
            circuit,
            cells,
            output,
            stored_after_last_step: source.clone(),
            source,
            tensor_program,
            roots,
            values: vec![0.0; wire_count],
            clock: STOPPED_SPRING_CLOCK,
            evaluations: Vec::new(),
            validator: finished.validator,
            evaluator,
            write_back,
            queued: Vec::new(),
            mounted: false,
            queued_viewport: (0.0, 0.0),
            previous_frame_time: 0.0,
            present: 1.0,
            interactive: 1.0,
            powers: spring_matrix_powers(),
            before_evaluate: None,
            after_evaluate: None,
        }
    }

    fn queue(&mut self, event: PhotoGalleryEvent, has_time_stamp: bool) {
        if event.kind == PhotoGalleryEventKind::Mount {
            self.queued.insert(0, (event, has_time_stamp));
        } else {
            self.queued.push((event, has_time_stamp));
        }
    }

    fn queue_viewport(&mut self, kind: PhotoGalleryEventKind, width: f64, height: f64, time_stamp: Option<f64>) {
        self.queued_viewport = (width, height);
        let writes = vec![(self.cells.viewport_width, width as f32), (self.cells.viewport_height, height as f32)];
        self.queue(
            PhotoGalleryEvent { kind, time_stamp: time_stamp.unwrap_or(0.0), writes, clears_events: false },
            time_stamp.is_some(),
        );
    }

    fn receive(&mut self, kind: PhotoGalleryEventKind, field: Wire, time_stamp: Option<f64>, point: Option<(f64, f64)>) {
        let mut writes = vec![(field, 1.0)];
        if let Some((x, y)) = point {
            writes.push((self.cells.photos.click_x, x as f32));
            writes.push((self.cells.photos.click_y, y as f32));
        }
        self.queue(
            PhotoGalleryEvent { kind, time_stamp: time_stamp.unwrap_or(0.0), writes, clears_events: true },
            time_stamp.is_some(),
        );
    }

    pub fn resize(&mut self, width: f64, height: f64, time_stamp: Option<f64>) {
        self.queue_viewport(PhotoGalleryEventKind::Resize, width, height, time_stamp);
    }

    pub fn click(&mut self, x: f64, y: f64, time_stamp: Option<f64>) {
        let field = self.cells.photos.click_active;
        self.receive(PhotoGalleryEventKind::Click, field, time_stamp, Some((x, y)));
    }

    pub fn previous(&mut self, time_stamp: Option<f64>) {
        let field = self.cells.photos.previous;
        self.receive(PhotoGalleryEventKind::Previous, field, time_stamp, None);
    }

    pub fn next(&mut self, time_stamp: Option<f64>) {
        let field = self.cells.photos.next;
        self.receive(PhotoGalleryEventKind::Next, field, time_stamp, None);
    }

    pub fn close(&mut self, time_stamp: Option<f64>) {
        let field = self.cells.photos.close;
        self.receive(PhotoGalleryEventKind::Close, field, time_stamp, None);
    }

    fn evaluate(&mut self, evaluation: &PlannedEvaluation<PhotoGalleryEvent>, scratch: &mut Vec<f32>) -> bool {
        let mut table = if evaluation.writes_back { std::mem::take(&mut self.source) } else { self.source.clone() };
        let matrix = self.powers[evaluation.steps as usize];
        for (entry, &wire) in matrix.iter().zip(&self.cells.spring_matrix) {
            table[wire.index()] = *entry as f32;
        }
        table[self.cells.settle_check.index()] = if evaluation.settle_check { 1.0 } else { 0.0 };
        table[self.cells.snap.index()] = if evaluation.snap { 1.0 } else { 0.0 };
        table[self.cells.present.index()] = self.present;
        table[self.cells.interactive.index()] = self.interactive;
        let events = photo_events(&self.cells.photos);
        for event in &evaluation.events {
            if event.clears_events {
                for &field in &events {
                    table[field.index()] = 0.0;
                }
            }
            for &(wire, value) in &event.writes {
                table[wire.index()] = value;
            }
        }
        if let Err(message) = self.validator.check(&table) {
            panic!("{message}");
        }
        if let Some(observer) = self.before_evaluate.as_mut() {
            observer(&table, evaluation);
        }
        let root_values = self.evaluator.evaluate(&table).to_vec();
        for (&root, &value) in self.roots.iter().zip(&root_values) {
            self.values[root.index()] = value;
        }
        let next_source = &root_values[..table.len()];
        if let Err(message) = self.validator.check(next_source) {
            panic!("{message}");
        }
        assert!(root_values.iter().all(|value| value.is_finite()), "tensor output contains a non-finite number");
        if let Err(message) = validate_photo_gallery_output(&self.values, &self.output, 1e-5) {
            panic!("{message}");
        }
        if evaluation.writes_back {
            table.copy_from_slice(next_source);
        }
        self.evaluations.push(MachineEvaluation {
            kind: evaluation.kind,
            time_ms: evaluation.time_ms,
            steps: evaluation.steps,
            writes_back: evaluation.writes_back,
            events: evaluation.events.iter().map(|event| event.kind.name()).collect(),
        });
        if let Some(observer) = self.after_evaluate.as_mut() {
            observer(&self.values, &table, evaluation);
        }
        let animating = self.values[self.output.animation_active.index()] > 0.0;
        if evaluation.writes_back {
            self.source = table;
        } else {
            *scratch = table;
        }
        animating
    }

    /// Runs the frame at `frame_time`, by default 16 ms after the previous
    /// frame, and returns the values of its 'frame' evaluation, indexed by wire.
    pub fn step(&mut self, viewport_width: f64, viewport_height: f64, present: f64, interactive: f64, frame_time: Option<f64>) -> &[f32] {
        let frame_time = frame_time.unwrap_or(self.previous_frame_time + 16.0);
        assert!(frame_time.is_finite(), "frame time must be finite");
        if self.mounted {
            for (field, (&now, &stored)) in self.source.iter().zip(&self.stored_after_last_step).enumerate() {
                assert!(now.to_bits() == stored.to_bits(), "source cell {field} was written between steps");
            }
        }
        self.present = present as f32;
        self.interactive = interactive as f32;
        if !self.mounted {
            self.mounted = true;
            let earliest = self
                .queued
                .iter()
                .fold(frame_time, |time, (event, has)| time.min(if *has { event.time_stamp } else { frame_time }));
            self.queue_viewport(PhotoGalleryEventKind::Mount, viewport_width, viewport_height, Some(earliest));
        } else if viewport_width != self.queued_viewport.0 || viewport_height != self.queued_viewport.1 {
            self.queue_viewport(PhotoGalleryEventKind::Resize, viewport_width, viewport_height, Some(frame_time));
        }
        let events: Vec<PhotoGalleryEvent> = std::mem::take(&mut self.queued)
            .into_iter()
            .map(|(mut event, has)| {
                if !has {
                    event.time_stamp = frame_time;
                }
                event
            })
            .collect();
        self.evaluations.clear();
        let mut scratch = Vec::new();
        let clock = self.clock;
        let write_back = self.write_back;
        self.clock = advance_frame(
            clock,
            frame_time,
            &events,
            &mut |evaluation| self.evaluate(evaluation, &mut scratch),
            write_back,
            &DEFAULT_SPRING_CLOCK_CONFIG,
        );
        self.previous_frame_time = frame_time;
        self.stored_after_last_step.copy_from_slice(&self.source);
        &self.values
    }
}
