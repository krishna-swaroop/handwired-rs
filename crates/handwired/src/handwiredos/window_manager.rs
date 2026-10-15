// HandwiredOS's window manager, a port of the circuit-building parts of
// src/handwiredos/window-manager.ts: the cells, buildWindowManager, the
// machine that drives the program with advance_frame, the names of its
// outputs, the output checks and noteFocusChange. The DOM projection, the
// render loop and the event listeners are left out.

use super::photo_gallery::{
    build_photos, declare_photo_cells, declare_spring_matrix, emit_interval_distance, matrix_bound, photo_events,
    spring_matrix_powers, validate_photo_gallery_output, PhotoCells, PhotoHost, PhotoOutputs, CONTROL_FIELD_NAMES,
    PHOTO_FIELD_NAMES, THUMBNAIL_FIELD_NAMES,
};
use super::photo_gallery::{assert_non_negative_finite, assert_unit_interval, MachineEvaluation};
use super::value_list::{
    build_value_list, declare_value_list_cells, value_list_keys, ValueListCells, ValueListHost, ValueListKey,
    ValueListOutputs, LOCATION_COUNT, MAXIMUM_SCROLL_TOP, MONITOR_LAYOUT, ROW_CAPACITY, SCROLL_GRID, SLOT_FIELD_NAMES,
};
use crate::cells::{constant, event, input, state, state_bit, CellPolicy, CellSpec, SourceTable, TableValidator, ANY_NUMBER, UNIT_INTERVAL};
use crate::circuit::{Circuit, Term, Wire};
use crate::compile::{CompileOptions, TensorEvaluator, TensorProgram};
use crate::emit::{emit_compare, emit_maximum, emit_minimum, emit_multiply, emit_relu, emit_select, CompareOptions, Relation};
use crate::fp::js_round;
use crate::host_timing::{
    advance_frame, quantize_time_ms, PlannedEvaluation, SpringClock, SpringMatrix, TimedEvent, WriteBack,
    DEFAULT_SPRING_CLOCK_CONFIG, STOPPED_SPRING_CLOCK,
};
use crate::motion::{emit_animated_object, AnimatedObjectOptions, AnimatedSpring, SpringMatrixWires};
use crate::validate::{validate_circuit, validate_tensor_program};

pub const WINDOW_COUNT: usize = 3;
pub const WINDOW_OUTPUT_FIELD_SIZE: usize = 13;
pub const WINDOW_OUTPUT_FIELD_NAMES: [&str; 13] = [
    "translateX",
    "translateY",
    "width",
    "height",
    "scale",
    "opacity",
    "focused",
    "stack",
    "maximized",
    "contentWidth",
    "contentHeight",
    "destinationWidth",
    "destinationHeight",
];
pub const DOCK_OUTPUT_FIELD_SIZE: usize = 7;
pub const DOCK_OUTPUT_FIELD_NAMES: [&str; 7] = ["left", "top", "width", "height", "inactive", "focused", "open"];
pub mod window_output_field {
    pub const TRANSLATE_X: usize = 0;
    pub const TRANSLATE_Y: usize = 1;
    pub const WIDTH: usize = 2;
    pub const HEIGHT: usize = 3;
    pub const SCALE: usize = 4;
    pub const OPACITY: usize = 5;
    pub const FOCUSED: usize = 6;
    pub const STACK: usize = 7;
    pub const MAXIMIZED: usize = 8;
}

const PIXEL_GRID: f64 = 1.0 / 64.0;
const POINTER_COORDINATE_BOUND: f64 = 32_768.0;
const PIXEL: (f64, f64) = (0.0, 16_384.0);
const INITIAL_GEOMETRY: [[f64; 4]; 3] = [[54.0, 54.0, 540.0, 340.0], [410.0, 104.0, 500.0, 366.0], [650.0, 190.0, 560.0, 430.0]];
pub const MONITOR_WINDOW_INDEX: usize = 0;
pub const NOTES_WINDOW_INDEX: usize = 1;
pub const PHOTOS_WINDOW_INDEX: usize = WINDOW_COUNT - 1;
const ANIMATION_FIELDS: [&str; 5] = ["centerX", "centerY", "width", "height", "scale"];

fn coordinate() -> CellSpec {
    input((-POINTER_COORDINATE_BOUND, POINTER_COORDINATE_BOUND)).grid(PIXEL_GRID)
}

fn pixel_constant(value: f64) -> CellSpec {
    constant(value, (0.0, 1024.0)).grid(PIXEL_GRID)
}

#[derive(Copy, Clone, Debug)]
pub struct Latch {
    pub active: Wire,
    pub x: Wire,
    pub y: Wire,
}

#[derive(Copy, Clone, Debug)]
pub struct WindowCells {
    pub left: Wire,
    pub top: Wire,
    pub width: Wire,
    pub height: Wire,
    pub open: Wire,
    pub minimized: Wire,
    pub maximized: Wire,
    pub present: Wire,
}

impl WindowCells {
    /// The cells in windowStateField's order.
    pub fn fields(&self) -> [Wire; 8] {
        [self.left, self.top, self.width, self.height, self.open, self.minimized, self.maximized, self.present]
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SpringCells {
    pub displacement: Wire,
    pub velocity: Wire,
    pub destination: Wire,
}

#[derive(Clone, Debug)]
pub struct ArmedCells {
    pub close: Vec<Wire>,
    pub minimize: Vec<Wire>,
    pub maximize: Vec<Wire>,
    pub dock_item: Vec<Wire>,
    pub photos_content: Wire,
    pub value_list: Wire,
    pub value_list_location: Vec<Wire>,
}

impl ArmedCells {
    pub fn all(&self) -> Vec<Wire> {
        let mut all = Vec::new();
        all.extend(&self.close);
        all.extend(&self.minimize);
        all.extend(&self.maximize);
        all.extend(&self.dock_item);
        all.push(self.photos_content);
        all.push(self.value_list);
        all.extend(&self.value_list_location);
        all
    }
}

/// windowManagerCells, in declaration order.
#[derive(Clone, Debug)]
pub struct WindowManagerCells {
    pub viewport_width: Wire,
    pub viewport_height: Wire,
    pub unit: Wire,
    pub margin: Wire,
    pub title_height: Wire,
    pub min_width: Wire,
    pub min_height: Wire,
    pub resize_grip: Wire,
    pub button_size: Wire,
    pub button_gap: Wire,
    pub dock_height: Wire,
    pub dock_gap: Wire,
    pub dock_item_width: Wire,
    pub down: Latch,
    pub move_: Latch,
    pub up: Latch,
    pub cancel: Wire,
    pub pointer_x: Wire,
    pub pointer_y: Wire,
    pub system_bar_height: Wire,
    pub spring_matrix: SpringMatrixWires,
    pub settle_check: Wire,
    pub snap: Wire,
    pub windows: Vec<WindowCells>,
    pub drag: Vec<Wire>,
    pub resize: Vec<Wire>,
    pub armed: ArmedCells,
    pub stack: Vec<Vec<Wire>>,
    /// springs[window][field] for centerX, centerY, width, height, scale.
    pub springs: Vec<[SpringCells; 5]>,
    pub value_list: ValueListCells,
    pub photos: PhotoCells,
}

pub fn declare_window_manager_cells(table: &mut SourceTable) -> WindowManagerCells {
    let bound = matrix_bound();
    let viewport_width = table.cell("viewportWidth", input(PIXEL).grid(PIXEL_GRID));
    let viewport_height = table.cell("viewportHeight", input(PIXEL).grid(PIXEL_GRID));
    let unit = table.cell("unit", constant(1.0, (1.0, 1.0)).grid(1.0));
    let margin = table.cell("margin", pixel_constant(12.0));
    let title_height = table.cell("titleHeight", pixel_constant(44.0));
    let min_width = table.cell("minWidth", pixel_constant(280.0));
    let min_height = table.cell("minHeight", pixel_constant(190.0));
    let resize_grip = table.cell("resizeGrip", pixel_constant(24.0));
    let button_size = table.cell("buttonSize", pixel_constant(18.0));
    let button_gap = table.cell("buttonGap", pixel_constant(10.0));
    let dock_height = table.cell("dockHeight", pixel_constant(76.0));
    let dock_gap = table.cell("dockGap", pixel_constant(12.0));
    let dock_item_width = table.cell("dockItemWidth", pixel_constant(50.0));
    let latch = |table: &mut SourceTable, name: &str| Latch {
        active: table.cell(format!("{name}.active"), event().grid(1.0)),
        x: table.cell(format!("{name}.x"), coordinate()),
        y: table.cell(format!("{name}.y"), coordinate()),
    };
    let down = latch(table, "down");
    let move_ = latch(table, "move");
    let up = latch(table, "up");
    let cancel = table.cell("cancel", event().grid(1.0));
    let pointer_range = (-POINTER_COORDINATE_BOUND, POINTER_COORDINATE_BOUND);
    let pointer_x = table.cell("pointer.x", state(pointer_range).grid(PIXEL_GRID));
    let pointer_y = table.cell("pointer.y", state(pointer_range).grid(PIXEL_GRID));
    let system_bar_height = table.cell("systemBarHeight", pixel_constant(36.0));
    let spring_matrix = declare_spring_matrix(table, "", bound);
    let settle_check = table.cell("settleCheck", state_bit(0.0));
    let snap = table.cell("snap", state_bit(0.0));
    let windows = (0..WINDOW_COUNT)
        .map(|index| {
            let [left, top, width, height] = INITIAL_GEOMETRY[index];
            let name = |field: &str| format!("windows[{index}].{field}");
            WindowCells {
                left: table.cell(name("left"), state(PIXEL).initial(left).grid(PIXEL_GRID)),
                top: table.cell(name("top"), state(PIXEL).initial(top).grid(PIXEL_GRID)),
                width: table.cell(name("width"), state(PIXEL).initial(width).grid(PIXEL_GRID)),
                height: table.cell(name("height"), state(PIXEL).initial(height).grid(PIXEL_GRID)),
                open: table.cell(name("open"), state_bit(1.0).grid(1.0)),
                minimized: table.cell(name("minimized"), state_bit(0.0).grid(1.0)),
                maximized: table.cell(name("maximized"), state_bit(0.0).grid(1.0)),
                present: table.cell(name("present"), state_bit(1.0).grid(1.0)),
            }
        })
        .collect();
    let bits = [state_bit(0.0).grid(1.0); WINDOW_COUNT];
    let drag = table.one_hot("drag", &bits, true, false);
    let resize = table.one_hot("resize", &bits, true, false);
    let armed_list = |table: &mut SourceTable, field: &str, count: usize| -> Vec<Wire> {
        (0..count).map(|index| table.cell(format!("armed.{field}[{index}]"), state_bit(0.0).grid(1.0))).collect()
    };
    let close = armed_list(table, "close", WINDOW_COUNT);
    let minimize = armed_list(table, "minimize", WINDOW_COUNT);
    let maximize = armed_list(table, "maximize", WINDOW_COUNT);
    let dock_item = armed_list(table, "dockItem", WINDOW_COUNT);
    let photos_content = table.cell("armed.photosContent", state_bit(0.0).grid(1.0));
    let value_list = table.cell("armed.valueList", state_bit(0.0).grid(1.0));
    let value_list_location = armed_list(table, "valueListLocation", LOCATION_COUNT);
    let armed = ArmedCells { close, minimize, maximize, dock_item, photos_content, value_list, value_list_location };
    let order: Vec<usize> = (0..WINDOW_COUNT).collect();
    let stack = table.permutation("stack", &order, state_bit(0.0).grid(1.0));
    let springs = (0..WINDOW_COUNT)
        .map(|index| {
            let [left, top, width, height] = INITIAL_GEOMETRY[index];
            let ranges = [(-16_384.0, 32_768.0), (-16_384.0, 32_768.0), PIXEL, PIXEL, UNIT_INTERVAL];
            let initials = [left + width / 2.0, top + height / 2.0, width, height, 1.0];
            std::array::from_fn(|field| {
                let name = |part: &str| format!("springs[{index}].{}.{part}", ANIMATION_FIELDS[field]);
                SpringCells {
                    displacement: table.cell(name("displacement"), state(ANY_NUMBER)),
                    velocity: table.cell(name("velocity"), state(ANY_NUMBER)),
                    destination: table.cell(name("destination"), state(ranges[field]).initial(initials[field])),
                }
            })
        })
        .collect();
    let value_list = declare_value_list_cells(table, "valueList");
    let photos = declare_photo_cells(table, "photos");
    WindowManagerCells {
        viewport_width,
        viewport_height,
        unit,
        margin,
        title_height,
        min_width,
        min_height,
        resize_grip,
        button_size,
        button_gap,
        dock_height,
        dock_gap,
        dock_item_width,
        down,
        move_,
        up,
        cancel,
        pointer_x,
        pointer_y,
        system_bar_height,
        spring_matrix,
        settle_check,
        snap,
        windows,
        drag,
        resize,
        armed,
        stack,
        springs,
        value_list,
        photos,
    }
}

#[derive(Copy, Clone, Debug)]
struct Geometry {
    left: Wire,
    top: Wire,
    width: Wire,
    height: Wire,
}

#[derive(Clone, Debug)]
pub struct WindowManagerOutput {
    pub next_source: Vec<Wire>,
    pub windows: Vec<Wire>,
    pub dock_items: Vec<Wire>,
    pub animation_active: Wire,
    pub photos: PhotoOutputs,
    pub value_list: ValueListOutputs,
}

pub struct BuiltWindowManager {
    pub circuit: Circuit,
    pub cells: WindowManagerCells,
    pub output: WindowManagerOutput,
    /// Each source cell's name and policy, in table order.
    pub source_cells: Vec<(String, CellPolicy)>,
    pub initial_source: Vec<f32>,
    pub validator: TableValidator,
}

fn affine(circuit: &mut Circuit, terms: &[Term]) -> Wire {
    circuit.affine(terms, 0.0)
}

fn emit_rectangle_hit(circuit: &mut Circuit, x: Wire, y: Wire, geometry: Geometry) -> Wire {
    let half = circuit.constant(0.5);
    let distances = [
        affine(circuit, &[(x, 1.0), (geometry.left, -1.0)]),
        affine(circuit, &[(geometry.left, 1.0), (geometry.width, 1.0), (x, -1.0)]),
        affine(circuit, &[(y, 1.0), (geometry.top, -1.0)]),
        affine(circuit, &[(geometry.top, 1.0), (geometry.height, 1.0), (y, -1.0)]),
    ];
    let edge_terms: Vec<Term> = distances
        .iter()
        .map(|&distance| (emit_compare(circuit, distance, Relation::AtLeast, half, CompareOptions::default()), 1.0))
        .collect();
    let hit = circuit.reglu(&[], 1.0, &edge_terms, -3.0);
    circuit.assume_range(hit, 0.0, 1.0);
    hit
}

fn emit_dock_item_hit(circuit: &mut Circuit, x: Wire, y: Wire, geometry: Geometry, unit: Wire) -> Wire {
    let horizontal = emit_interval_distance(circuit, x, geometry.left, geometry.width);
    let vertical = emit_interval_distance(circuit, y, geometry.top, geometry.height);
    let distance = emit_minimum(circuit, horizontal, vertical);
    let score = emit_clamp_zero_one(circuit, distance, unit);
    circuit.boolean(score)
}

fn emit_all(circuit: &mut Circuit, bits: &[Wire]) -> Wire {
    let terms: Vec<Term> = bits.iter().map(|&bit| (bit, 1.0)).collect();
    let all = circuit.reglu(&[], 1.0, &terms, 1.0 - bits.len() as f64);
    circuit.assume_range(all, 0.0, 1.0);
    all
}

fn emit_clamp_zero_one(circuit: &mut Circuit, value: Wire, unit: Wire) -> Wire {
    let positive = emit_relu(circuit, value);
    let shifted = affine(circuit, &[(value, 1.0), (unit, -1.0)]);
    let above_one = emit_relu(circuit, shifted);
    let result = affine(circuit, &[(positive, 1.0), (above_one, -1.0)]);
    circuit.assume_range(result, 0.0, 1.0);
    result
}

fn emit_clamp_dynamic(circuit: &mut Circuit, value: Wire, minimum: Wire, maximum: Wire) -> Wire {
    let above = emit_maximum(circuit, value, minimum);
    emit_minimum(circuit, above, maximum)
}

fn emit_complement(circuit: &mut Circuit, value: Wire, unit: Wire) -> Wire {
    affine(circuit, &[(unit, 1.0), (value, -1.0)])
}

fn emit_frontmost(circuit: &mut Circuit, stack: &[Vec<Wire>], candidates: &[Wire], unit: Wire) -> Vec<Wire> {
    let mut selected_rank = [unit; WINDOW_COUNT];
    let mut remaining = unit;
    for rank in (0..WINDOW_COUNT).rev() {
        let terms: Vec<Term> =
            (0..WINDOW_COUNT).map(|window| (emit_multiply(circuit, stack[rank][window], candidates[window]), 1.0)).collect();
        let sum = affine(circuit, &terms);
        let rank_hit = emit_clamp_zero_one(circuit, sum, unit);
        selected_rank[rank] = emit_multiply(circuit, remaining, rank_hit);
        let left = affine(circuit, &[(remaining, 1.0), (rank_hit, -1.0)]);
        remaining = emit_relu(circuit, left);
    }
    (0..WINDOW_COUNT)
        .map(|window| {
            let terms: Vec<Term> =
                (0..WINDOW_COUNT).map(|rank| (emit_multiply(circuit, stack[rank][window], selected_rank[rank]), 1.0)).collect();
            let sum = affine(circuit, &terms);
            emit_clamp_zero_one(circuit, sum, unit)
        })
        .collect()
}

pub fn build_window_manager() -> BuiltWindowManager {
    let mut table = SourceTable::new();
    let cells = declare_window_manager_cells(&mut table);
    table.at_most_one("photos events", &photo_events(&cells.photos));
    table.at_most_one("value list keys", &value_list_keys(&cells.value_list));
    table.at_most_one("armed control", &cells.armed.all());
    let mut circuit = table.create_circuit();
    let c = &mut circuit;

    let unit = cells.unit;
    let zero = c.constant(0.0);
    let effective_dock_height = emit_minimum(c, cells.dock_height, cells.viewport_height);
    let dock_top = affine(c, &[(cells.viewport_height, 1.0), (effective_dock_height, -1.0)]);
    let workspace_top = emit_minimum(c, cells.system_bar_height, dock_top);
    let workspace_height = affine(c, &[(dock_top, 1.0), (workspace_top, -1.0)]);
    let half_viewport = affine(c, &[(cells.viewport_width, 0.5)]);
    let maximum_left = emit_minimum(c, cells.margin, half_viewport);
    let half_workspace = affine(c, &[(workspace_height, 0.5)]);
    let maximum_top_inset = emit_minimum(c, cells.margin, half_workspace);
    let maximum_geometry_top = affine(c, &[(workspace_top, 1.0), (maximum_top_inset, 1.0)]);
    let maximum_width_sum = affine(c, &[(cells.viewport_width, 1.0), (maximum_left, -2.0)]);
    let maximum_width = emit_relu(c, maximum_width_sum);
    let maximum_height_sum = affine(c, &[(workspace_height, 1.0), (maximum_top_inset, -2.0)]);
    let maximum_height = emit_relu(c, maximum_height_sum);
    let maximum_geometry =
        Geometry { left: maximum_left, top: maximum_geometry_top, width: maximum_width, height: maximum_height };
    let minimum_width = emit_minimum(c, cells.min_width, cells.viewport_width);
    let minimum_height = emit_minimum(c, cells.min_height, workspace_height);
    let viewport_width = cells.viewport_width;
    let fit_into_workspace = |c: &mut Circuit, rectangle: Geometry| -> Geometry {
        let in_viewport = |c: &mut Circuit, wire: Wire| {
            c.assume_range(wire, PIXEL.0, PIXEL.1);
            c.assume_grid(wire, PIXEL_GRID);
            wire
        };
        let width = emit_clamp_dynamic(c, rectangle.width, minimum_width, viewport_width);
        let width = in_viewport(c, width);
        let height = emit_clamp_dynamic(c, rectangle.height, minimum_height, workspace_height);
        let height = in_viewport(c, height);
        let left_room = affine(c, &[(viewport_width, 1.0), (width, -1.0)]);
        let maximum_left = emit_relu(c, left_room);
        let top_room = affine(c, &[(dock_top, 1.0), (height, -1.0)]);
        let maximum_top = emit_relu(c, top_room);
        let left = emit_clamp_dynamic(c, rectangle.left, zero, maximum_left);
        let left = in_viewport(c, left);
        let top = emit_clamp_dynamic(c, rectangle.top, workspace_top, maximum_top);
        let top = in_viewport(c, top);
        Geometry { left, top, width, height }
    };

    let mut fitted_geometry = Vec::new();
    let mut window_hit = Vec::new();
    let mut close_hit = Vec::new();
    let mut minimize_hit = Vec::new();
    let mut maximize_hit = Vec::new();
    let mut drag_hit = Vec::new();
    let mut resize_hit = Vec::new();
    let mut close_hit_at_up = Vec::new();
    let mut minimize_hit_at_up = Vec::new();
    let mut maximize_hit_at_up = Vec::new();
    let mut photos_content_hit = zero;
    let mut photos_content_hit_at_up = zero;
    let mut photos_grip_hit_at_up = zero;
    let mut value_list_hit = zero;
    let mut value_list_hit_at_up = zero;
    let mut monitor_grip_hit_at_up = zero;
    let mut value_list_top = zero;
    let mut value_list_location_hit = Vec::new();
    let mut value_list_location_hit_at_up = Vec::new();
    let (down, up) = (cells.down, cells.up);

    for window in 0..WINDOW_COUNT {
        let current = cells.windows[window];
        let not_minimized = emit_complement(c, current.minimized, unit);
        let visible = emit_multiply(c, current.open, not_minimized);
        let fitted = fit_into_workspace(
            c,
            Geometry { left: current.left, top: current.top, width: current.width, height: current.height },
        );
        fitted_geometry.push(fitted);
        let geometry = Geometry {
            left: emit_select(c, current.maximized, fitted.left, maximum_geometry.left),
            top: emit_select(c, current.maximized, fitted.top, maximum_geometry.top),
            width: emit_select(c, current.maximized, fitted.width, maximum_geometry.width),
            height: emit_select(c, current.maximized, fitted.height, maximum_geometry.height),
        };
        let hit = emit_rectangle_hit(c, down.x, down.y, geometry);
        window_hit.push(emit_multiply(c, visible, hit));

        let button_top = affine(c, &[(geometry.top, 1.0), (cells.title_height, 0.5), (cells.button_size, -0.5)]);
        let close_left = affine(
            c,
            &[(geometry.left, 1.0), (geometry.width, 1.0), (cells.button_gap, -1.0), (cells.button_size, -1.0)],
        );
        let maximize_left = affine(c, &[(close_left, 1.0), (cells.button_gap, -1.0), (cells.button_size, -1.0)]);
        let minimize_left = affine(c, &[(maximize_left, 1.0), (cells.button_gap, -1.0), (cells.button_size, -1.0)]);
        let button_hit_inset = affine(c, &[(cells.button_gap, 0.5)]);
        let button_hit_size = affine(c, &[(cells.button_size, 1.0), (cells.button_gap, 1.0)]);
        let button_hit_geometry = |c: &mut Circuit, left: Wire| Geometry {
            left: affine(c, &[(left, 1.0), (button_hit_inset, -1.0)]),
            top: affine(c, &[(button_top, 1.0), (button_hit_inset, -1.0)]),
            width: button_hit_size,
            height: button_hit_size,
        };
        let close_rectangle = button_hit_geometry(c, close_left);
        let maximize_rectangle = button_hit_geometry(c, maximize_left);
        let minimize_rectangle = button_hit_geometry(c, minimize_left);
        close_hit.push(emit_rectangle_hit(c, down.x, down.y, close_rectangle));
        maximize_hit.push(emit_rectangle_hit(c, down.x, down.y, maximize_rectangle));
        minimize_hit.push(emit_rectangle_hit(c, down.x, down.y, minimize_rectangle));
        close_hit_at_up.push(emit_rectangle_hit(c, up.x, up.y, close_rectangle));
        maximize_hit_at_up.push(emit_rectangle_hit(c, up.x, up.y, maximize_rectangle));
        minimize_hit_at_up.push(emit_rectangle_hit(c, up.x, up.y, minimize_rectangle));
        let title_width_sum = affine(c, &[(minimize_left, 1.0), (cells.button_gap, -1.0), (geometry.left, -1.0)]);
        let title_width = emit_relu(c, title_width_sum);
        drag_hit.push(emit_rectangle_hit(
            c,
            down.x,
            down.y,
            Geometry { left: geometry.left, top: geometry.top, width: title_width, height: cells.title_height },
        ));
        let grip_rectangle = Geometry {
            left: affine(c, &[(geometry.left, 1.0), (geometry.width, 1.0), (cells.resize_grip, -1.0)]),
            top: affine(c, &[(geometry.top, 1.0), (geometry.height, 1.0), (cells.resize_grip, -1.0)]),
            width: cells.resize_grip,
            height: cells.resize_grip,
        };
        resize_hit.push(emit_rectangle_hit(c, down.x, down.y, grip_rectangle));
        if window == MONITOR_WINDOW_INDEX {
            let layout = &MONITOR_LAYOUT;
            let list_rectangle = Geometry {
                left: c.affine(&[(geometry.left, 1.0), (unit, 1.0)], layout.sidebar_width),
                top: c.affine(&[(geometry.top, 1.0), (cells.title_height, 1.0), (unit, 1.0)], layout.header_height),
                width: c.affine(&[(geometry.width, 1.0), (unit, -2.0)], -layout.sidebar_width - layout.scrollbar_strip),
                height: c.affine(
                    &[(geometry.height, 1.0), (cells.title_height, -1.0), (unit, -2.0)],
                    -layout.header_height - layout.footer_height,
                ),
            };
            value_list_hit = emit_rectangle_hit(c, down.x, down.y, list_rectangle);
            value_list_hit_at_up = emit_rectangle_hit(c, up.x, up.y, list_rectangle);
            monitor_grip_hit_at_up = emit_rectangle_hit(c, up.x, up.y, grip_rectangle);
            value_list_top = list_rectangle.top;
            for location in 0..LOCATION_COUNT {
                let location_rectangle = Geometry {
                    left: c.affine(&[(geometry.left, 1.0), (unit, 1.0)], layout.location_inset),
                    top: c.affine(
                        &[(geometry.top, 1.0), (cells.title_height, 1.0), (unit, 1.0)],
                        layout.location_top + location as f64 * layout.location_height,
                    ),
                    width: c.constant(layout.sidebar_width - 2.0 * layout.location_inset),
                    height: c.constant(layout.location_height),
                };
                value_list_location_hit.push(emit_rectangle_hit(c, down.x, down.y, location_rectangle));
                value_list_location_hit_at_up.push(emit_rectangle_hit(c, up.x, up.y, location_rectangle));
            }
        }
        if window == PHOTOS_WINDOW_INDEX {
            let content_rectangle = Geometry {
                left: affine(c, &[(geometry.left, 1.0), (unit, 1.0)]),
                top: affine(c, &[(geometry.top, 1.0), (cells.title_height, 1.0), (unit, 1.0)]),
                width: affine(c, &[(geometry.width, 1.0), (unit, -2.0)]),
                height: affine(c, &[(geometry.height, 1.0), (cells.title_height, -1.0), (unit, -2.0)]),
            };
            photos_content_hit = emit_rectangle_hit(c, down.x, down.y, content_rectangle);
            photos_content_hit_at_up = emit_rectangle_hit(c, up.x, up.y, content_rectangle);
            photos_grip_hit_at_up = emit_rectangle_hit(c, up.x, up.y, grip_rectangle);
        }
    }

    let selected_canvas = emit_frontmost(c, &cells.stack, &window_hit, unit);

    let windows = WINDOW_COUNT as f64;
    let gap_share = affine(c, &[(cells.viewport_width, 1.0 / (windows - 1.0))]);
    let height_share = affine(c, &[(effective_dock_height, 0.5)]);
    let smaller_share = emit_minimum(c, gap_share, height_share);
    let dock_gap = emit_minimum(c, cells.dock_gap, smaller_share);
    let horizontal_sum = affine(c, &[(cells.viewport_width, 1.0), (dock_gap, -(windows - 1.0))]);
    let dock_horizontal_space = emit_relu(c, horizontal_sum);
    let vertical_sum = affine(c, &[(effective_dock_height, 1.0), (dock_gap, -2.0)]);
    let dock_vertical_space = emit_relu(c, vertical_sum);
    let item_share = affine(c, &[(dock_horizontal_space, 1.0 / windows)]);
    let smaller_space = emit_minimum(c, item_share, dock_vertical_space);
    let dock_item_size = emit_minimum(c, cells.dock_item_width, smaller_space);
    let dock_total_width = affine(c, &[(dock_item_size, windows), (dock_gap, windows - 1.0)]);
    let dock_left = affine(c, &[(cells.viewport_width, 0.5), (dock_total_width, -0.5)]);
    let dock_item_top = affine(c, &[(dock_top, 1.0), (effective_dock_height, 0.5), (dock_item_size, -0.5)]);
    let mut dock_geometry = Vec::new();
    let mut dock_hit = Vec::new();
    let mut dock_hit_at_up = Vec::new();
    for window in 0..WINDOW_COUNT {
        let index = window as f64;
        let geometry = Geometry {
            left: affine(c, &[(dock_left, 1.0), (dock_item_size, index), (dock_gap, index)]),
            top: dock_item_top,
            width: dock_item_size,
            height: dock_item_size,
        };
        dock_geometry.push(geometry);
        dock_hit.push(emit_dock_item_hit(c, down.x, down.y, geometry, unit));
        dock_hit_at_up.push(emit_dock_item_hit(c, up.x, up.y, geometry, unit));
    }

    let canvas_terms: Vec<Term> = selected_canvas.iter().map(|&wire| (wire, 1.0)).collect();
    let window_pressed = affine(c, &canvas_terms);
    c.assume_range(window_pressed, 0.0, 1.0);
    let release_sum = affine(c, &[(up.active, 1.0), (cells.cancel, -1.0)]);
    let release = emit_relu(c, release_sum);
    let pointer_still_down = emit_complement(c, up.active, unit);
    let next_armed_for = |c: &mut Circuit, armed: Wire| emit_all(c, &[armed, pointer_still_down]);
    let mut next_armed_close = Vec::new();
    let mut next_armed_minimize = Vec::new();
    let mut next_armed_maximize = Vec::new();
    let mut next_armed_dock_item = Vec::new();
    let mut canvas_press = Vec::new();
    let mut selected_window = Vec::new();
    let mut close_click = Vec::new();
    let mut minimize_click = Vec::new();
    let mut maximize_click = Vec::new();
    let mut dock_click = Vec::new();
    let mut drag_press = Vec::new();
    let mut resize_press = Vec::new();
    for window in 0..WINDOW_COUNT {
        let canvas = emit_multiply(c, down.active, selected_canvas[window]);
        canvas_press.push(canvas);
        let normal_window = emit_complement(c, cells.windows[window].maximized, unit);
        let dragged = emit_multiply(c, canvas, drag_hit[window]);
        drag_press.push(emit_multiply(c, dragged, normal_window));
        let resized = emit_multiply(c, canvas, resize_hit[window]);
        resize_press.push(emit_multiply(c, resized, normal_window));
        let armed_after_down = |c: &mut Circuit, stored: Wire, pressed: Wire| emit_select(c, down.active, stored, pressed);
        let pressed_close = emit_multiply(c, canvas, close_hit[window]);
        let armed_close = armed_after_down(c, cells.armed.close[window], pressed_close);
        let pressed_minimize = emit_multiply(c, canvas, minimize_hit[window]);
        let armed_minimize = armed_after_down(c, cells.armed.minimize[window], pressed_minimize);
        let pressed_maximize = emit_multiply(c, canvas, maximize_hit[window]);
        let armed_maximize = armed_after_down(c, cells.armed.maximize[window], pressed_maximize);
        let no_window = emit_complement(c, window_pressed, unit);
        let pressed_dock = emit_all(c, &[down.active, dock_hit[window], no_window]);
        let armed_dock_item = armed_after_down(c, cells.armed.dock_item[window], pressed_dock);
        close_click.push(emit_all(c, &[release, armed_close, close_hit_at_up[window]]));
        minimize_click.push(emit_all(c, &[release, armed_minimize, minimize_hit_at_up[window]]));
        maximize_click.push(emit_all(c, &[release, armed_maximize, maximize_hit_at_up[window]]));
        dock_click.push(emit_all(c, &[release, armed_dock_item, dock_hit_at_up[window]]));
        next_armed_close.push(next_armed_for(c, armed_close));
        next_armed_minimize.push(next_armed_for(c, armed_minimize));
        next_armed_maximize.push(next_armed_for(c, armed_maximize));
        next_armed_dock_item.push(next_armed_for(c, armed_dock_item));
        let selection = affine(c, &[(canvas, 1.0), (dock_click[window], 1.0)]);
        selected_window.push(emit_clamp_zero_one(c, selection, unit));
    }
    let photos_window = PHOTOS_WINDOW_INDEX;
    let not_resizing_photos = emit_complement(c, resize_press[photos_window], unit);
    let pressed_photos = emit_all(c, &[canvas_press[photos_window], photos_content_hit, not_resizing_photos]);
    let armed_photos_content = emit_select(c, down.active, cells.armed.photos_content, pressed_photos);
    let photos_normal = emit_complement(c, cells.windows[photos_window].maximized, unit);
    let photos_grip_at_up = emit_all(c, &[photos_grip_hit_at_up, photos_normal]);
    let off_photos_grip = emit_complement(c, photos_grip_at_up, unit);
    let photos_content_click = emit_all(c, &[release, armed_photos_content, photos_content_hit_at_up, off_photos_grip]);
    let monitor = MONITOR_WINDOW_INDEX;
    let not_resizing_monitor = emit_complement(c, resize_press[monitor], unit);
    let pressed_list = emit_all(c, &[canvas_press[monitor], value_list_hit, not_resizing_monitor]);
    let armed_value_list = emit_select(c, down.active, cells.armed.value_list, pressed_list);
    let monitor_normal = emit_complement(c, cells.windows[monitor].maximized, unit);
    let monitor_grip_at_up = emit_all(c, &[monitor_grip_hit_at_up, monitor_normal]);
    let off_monitor_grip = emit_complement(c, monitor_grip_at_up, unit);
    let value_list_click = emit_all(c, &[release, armed_value_list, value_list_hit_at_up, off_monitor_grip]);
    let armed_value_list_location: Vec<Wire> = value_list_location_hit
        .iter()
        .enumerate()
        .map(|(location, &hit)| {
            let not_resizing = emit_complement(c, resize_press[monitor], unit);
            let pressed = emit_all(c, &[canvas_press[monitor], hit, not_resizing]);
            emit_select(c, down.active, cells.armed.value_list_location[location], pressed)
        })
        .collect();
    let value_list_location_click: Vec<Wire> = armed_value_list_location
        .iter()
        .enumerate()
        .map(|(location, &armed)| {
            let off_grip = emit_complement(c, monitor_grip_at_up, unit);
            emit_all(c, &[release, armed, value_list_location_hit_at_up[location], off_grip])
        })
        .collect();

    let baseline_x = emit_select(c, down.active, cells.pointer_x, down.x);
    let baseline_y = emit_select(c, down.active, cells.pointer_y, down.y);
    let moved_x = emit_select(c, cells.move_.active, baseline_x, cells.move_.x);
    let moved_y = emit_select(c, cells.move_.active, baseline_y, cells.move_.y);
    let pointer_x = emit_select(c, up.active, moved_x, up.x);
    let pointer_y = emit_select(c, up.active, moved_y, up.y);
    let delta_x = affine(c, &[(pointer_x, 1.0), (baseline_x, -1.0)]);
    let delta_y = affine(c, &[(pointer_y, 1.0), (baseline_y, -1.0)]);

    let grid_steps_per_pixel = 1.0 / PIXEL_GRID;
    let right = emit_relu(c, delta_x);
    let negated_x = affine(c, &[(delta_x, -1.0)]);
    let left = emit_relu(c, negated_x);
    let below = emit_relu(c, delta_y);
    let negated_y = affine(c, &[(delta_y, -1.0)]);
    let above = emit_relu(c, negated_y);
    let moved_sum = affine(
        c,
        &[(right, grid_steps_per_pixel), (left, grid_steps_per_pixel), (below, grid_steps_per_pixel), (above, grid_steps_per_pixel)],
    );
    let pointer_moved = emit_clamp_zero_one(c, moved_sum, unit);
    let mut next_normal_geometry = Vec::new();
    let mut next_stored_geometry = Vec::new();
    let mut next_open = Vec::new();
    let mut next_minimized = Vec::new();
    let mut next_maximized = Vec::new();
    let mut next_drag = Vec::new();
    let mut next_resize = Vec::new();
    let mut direct_manipulation = Vec::new();
    for window in 0..WINDOW_COUNT {
        let drag_for_motion = emit_select(c, down.active, cells.drag[window], drag_press[window]);
        let resize_for_motion = emit_select(c, down.active, cells.resize[window], resize_press[window]);
        next_drag.push(emit_multiply(c, drag_for_motion, pointer_still_down));
        next_resize.push(emit_multiply(c, resize_for_motion, pointer_still_down));
        let manipulated = affine(c, &[(drag_for_motion, 1.0), (resize_for_motion, 1.0)]);
        direct_manipulation.push(emit_clamp_zero_one(c, manipulated, unit));

        let current = cells.windows[window];
        let fitted = fitted_geometry[window];
        let dragged_x = emit_multiply(c, drag_for_motion, delta_x);
        let left = affine(c, &[(fitted.left, 1.0), (dragged_x, 1.0)]);
        let dragged_y = emit_multiply(c, drag_for_motion, delta_y);
        let top = affine(c, &[(fitted.top, 1.0), (dragged_y, 1.0)]);
        let resized_x = emit_multiply(c, resize_for_motion, delta_x);
        let width = affine(c, &[(fitted.width, 1.0), (resized_x, 1.0)]);
        let resized_y = emit_multiply(c, resize_for_motion, delta_y);
        let height = affine(c, &[(fitted.height, 1.0), (resized_y, 1.0)]);
        let normal = fit_into_workspace(c, Geometry { left, top, width, height });
        next_normal_geometry.push(normal);
        let stores_position = emit_multiply(c, direct_manipulation[window], pointer_moved);
        let stores_size = emit_multiply(c, resize_for_motion, pointer_moved);
        next_stored_geometry.push(Geometry {
            left: emit_select(c, stores_position, current.left, normal.left),
            top: emit_select(c, stores_position, current.top, normal.top),
            width: emit_select(c, stores_size, current.width, normal.width),
            height: emit_select(c, stores_size, current.height, normal.height),
        });

        let after_close = emit_select(c, close_click[window], current.open, zero);
        next_open.push(emit_select(c, dock_click[window], after_close, unit));
        let after_minimize = emit_select(c, minimize_click[window], current.minimized, unit);
        let after_dock = emit_select(c, dock_click[window], after_minimize, zero);
        next_minimized.push(emit_select(c, close_click[window], after_dock, zero));
        let flip = affine(c, &[(unit, 1.0), (current.maximized, -2.0)]);
        let toggle = emit_multiply(c, maximize_click[window], flip);
        let toggled_maximized = affine(c, &[(current.maximized, 1.0), (toggle, 1.0)]);
        next_maximized.push(emit_select(c, close_click[window], toggled_maximized, zero));
    }

    let selection_terms: Vec<Term> = selected_window.iter().map(|&wire| (wire, 1.0)).collect();
    let selection_sum = affine(c, &selection_terms);
    let selection_mass = emit_clamp_zero_one(c, selection_sum, unit);
    let selection_inactive = emit_complement(c, selection_mass, unit);
    let selected_rank: Vec<Wire> = (0..WINDOW_COUNT)
        .map(|rank| {
            let terms: Vec<Term> = (0..WINDOW_COUNT)
                .map(|window| (emit_multiply(c, cells.stack[rank][window], selected_window[window]), 1.0))
                .collect();
            let sum = affine(c, &terms);
            emit_clamp_zero_one(c, sum, unit)
        })
        .collect();
    let mut next_stack = vec![vec![zero; WINDOW_COUNT]; WINDOW_COUNT];
    for rank in 0..WINDOW_COUNT {
        for window in 0..WINDOW_COUNT {
            let mut moved_terms: Vec<Term> = Vec::new();
            if rank == WINDOW_COUNT - 1 {
                moved_terms.push((selected_window[window], 1.0));
            } else {
                for selected_rank_index in 0..WINDOW_COUNT {
                    let source_rank = if rank < selected_rank_index { rank } else { rank + 1 };
                    moved_terms.push((emit_multiply(c, selected_rank[selected_rank_index], cells.stack[source_rank][window]), 1.0));
                }
            }
            let kept = emit_multiply(c, selection_inactive, cells.stack[rank][window]);
            let moved = affine(c, &moved_terms);
            next_stack[rank][window] = affine(c, &[(kept, 1.0), (moved, 1.0)]);
        }
    }

    let mut next_visible = Vec::new();
    let mut effective_geometry = Vec::new();
    for window in 0..WINDOW_COUNT {
        let not_minimized = emit_complement(c, next_minimized[window], unit);
        next_visible.push(emit_multiply(c, next_open[window], not_minimized));
        let normal = next_normal_geometry[window];
        let maximized = next_maximized[window];
        effective_geometry.push(Geometry {
            left: emit_select(c, maximized, normal.left, maximum_geometry.left),
            top: emit_select(c, maximized, normal.top, maximum_geometry.top),
            width: emit_select(c, maximized, normal.width, maximum_geometry.width),
            height: emit_select(c, maximized, normal.height, maximum_geometry.height),
        });
    }
    let window_border_span = affine(c, &[(unit, 2.0)]);
    let destination_content_width: Vec<Wire> = effective_geometry
        .iter()
        .map(|geometry| {
            let sum = affine(c, &[(geometry.width, 1.0), (window_border_span, -1.0)]);
            emit_relu(c, sum)
        })
        .collect();
    let destination_content_height: Vec<Wire> = effective_geometry
        .iter()
        .map(|geometry| {
            let sum = affine(c, &[(geometry.height, 1.0), (cells.title_height, -1.0), (window_border_span, -1.0)]);
            emit_relu(c, sum)
        })
        .collect();
    let focused_window = emit_frontmost(c, &next_stack, &next_visible, unit);

    struct WindowAnimation {
        width: Wire,
        height: Wire,
        translate_x: Wire,
        translate_y: Wire,
        scale: Wire,
        active: Wire,
    }
    let mut window_animations = Vec::new();
    for window in 0..WINDOW_COUNT {
        let visible_target = effective_geometry[window];
        let dock_target = dock_geometry[window];
        let visible_center_x = affine(c, &[(visible_target.left, 1.0), (visible_target.width, 0.5)]);
        let visible_center_y = affine(c, &[(visible_target.top, 1.0), (visible_target.height, 0.5)]);
        let dock_center_x = affine(c, &[(dock_target.left, 1.0), (dock_target.width, 0.5)]);
        let dock_center_y = affine(c, &[(dock_target.top, 1.0), (dock_target.height, 0.5)]);
        let target = [
            emit_select(c, next_visible[window], dock_center_x, visible_center_x),
            emit_select(c, next_visible[window], dock_center_y, visible_center_y),
            visible_target.width,
            visible_target.height,
            next_visible[window],
        ];
        let absent = emit_complement(c, cells.windows[window].present, unit);
        let hidden = emit_complement(c, next_visible[window], unit);
        let hidden_and_absent = emit_multiply(c, absent, hidden);
        let direct_sum = affine(c, &[(direct_manipulation[window], 1.0), (hidden_and_absent, 1.0)]);
        let direct = emit_clamp_zero_one(c, direct_sum, unit);
        let springs: Vec<AnimatedSpring> = (0..5)
            .map(|field| AnimatedSpring {
                matrix: cells.spring_matrix,
                displacement: cells.springs[window][field].displacement,
                velocity: cells.springs[window][field].velocity,
                destination: target[field],
                destination_during_steps: Some(cells.springs[window][field].destination),
            })
            .collect();
        let snap_sum = affine(c, &[(direct, 1.0), (cells.snap, 1.0)]);
        let snap = emit_clamp_zero_one(c, snap_sum, unit);
        let object = emit_animated_object(
            c,
            &springs,
            AnimatedObjectOptions { settle_check: cells.settle_check, snap: Some(snap), settle_distance: None, settle_speed: None },
        );
        let mut stored = Vec::new();
        let mut next = Vec::new();
        for field in 0..5 {
            let spring = cells.springs[window][field];
            stored.extend([spring.displacement, spring.velocity, spring.destination]);
            next.extend([object.displacements[field], object.velocities[field], target[field]]);
        }
        table.next(c, &stored, &next);
        let width = emit_clamp_dynamic(c, object.positions[2], zero, cells.viewport_width);
        let height = emit_clamp_dynamic(c, object.positions[3], zero, cells.viewport_height);
        let scale = emit_clamp_dynamic(c, object.positions[4], zero, unit);
        let half_width = affine(c, &[(width, 0.5)]);
        let half_height = affine(c, &[(height, 0.5)]);
        let scaled_half_width = emit_multiply(c, half_width, scale);
        let scaled_half_height = emit_multiply(c, half_height, scale);
        let right_limit = affine(c, &[(cells.viewport_width, 1.0), (scaled_half_width, -1.0)]);
        let center_x = emit_clamp_dynamic(c, object.positions[0], scaled_half_width, right_limit);
        let bottom_limit = affine(c, &[(cells.viewport_height, 1.0), (scaled_half_height, -1.0)]);
        let center_y = emit_clamp_dynamic(c, object.positions[1], scaled_half_height, bottom_limit);
        let translate_x = affine(c, &[(center_x, 1.0), (half_width, -1.0)]);
        let translate_y = affine(c, &[(center_y, 1.0), (half_height, -1.0)]);
        window_animations.push(WindowAnimation { width, height, translate_x, translate_y, scale, active: object.active });
    }
    let activity_terms: Vec<Term> = window_animations.iter().map(|animation| (animation.active, 1.0)).collect();
    let activity = affine(c, &activity_terms);
    let window_animation_active = emit_clamp_zero_one(c, activity, unit);
    let next_present: Vec<Wire> = (0..WINDOW_COUNT)
        .map(|window| {
            let still = emit_multiply(c, cells.windows[window].present, window_animations[window].active);
            let sum = affine(c, &[(next_visible[window], 1.0), (still, 1.0)]);
            emit_clamp_zero_one(c, sum, unit)
        })
        .collect();

    table.next(c, &[cells.pointer_x, cells.pointer_y], &[pointer_x, pointer_y]);
    let matrix = cells.spring_matrix;
    table.next(
        c,
        &[matrix[0], matrix[1], matrix[2], matrix[3], cells.settle_check, cells.snap],
        &[unit, zero, zero, unit, zero, zero],
    );
    for window in 0..WINDOW_COUNT {
        let geometry = next_stored_geometry[window];
        let current = cells.windows[window];
        table.next(
            c,
            &current.fields(),
            &[
                geometry.left,
                geometry.top,
                geometry.width,
                geometry.height,
                next_open[window],
                next_minimized[window],
                next_maximized[window],
                next_present[window],
            ],
        );
    }
    table.next(c, &cells.drag, &next_drag);
    table.next(c, &cells.resize, &next_resize);
    let next_photos_content = next_armed_for(c, armed_photos_content);
    let next_value_list = next_armed_for(c, armed_value_list);
    let next_value_list_location: Vec<Wire> =
        armed_value_list_location.iter().map(|&armed| next_armed_for(c, armed)).collect();
    let mut next_armed = Vec::new();
    next_armed.extend(&next_armed_close);
    next_armed.extend(&next_armed_minimize);
    next_armed.extend(&next_armed_maximize);
    next_armed.extend(&next_armed_dock_item);
    next_armed.push(next_photos_content);
    next_armed.push(next_value_list);
    next_armed.extend(&next_value_list_location);
    table.next(c, &cells.armed.all(), &next_armed);
    let stored_stack: Vec<Wire> = cells.stack.iter().flatten().copied().collect();
    let next_stack_flat: Vec<Wire> = next_stack.iter().flatten().copied().collect();
    table.next(c, &stored_stack, &next_stack_flat);

    let mut windows = Vec::new();
    let mut dock_items = Vec::new();
    for window in 0..WINDOW_COUNT {
        let animation = &window_animations[window];
        let mut stack_terms: Vec<Term> = (0..WINDOW_COUNT).map(|rank| (next_stack[rank][window], rank as f64 + 1.0)).collect();
        stack_terms.push((animation.active, 40.0));
        let stack = affine(c, &stack_terms);
        windows.extend([
            animation.translate_x,
            animation.translate_y,
            animation.width,
            animation.height,
            animation.scale,
            next_present[window],
            focused_window[window],
            stack,
            next_maximized[window],
            destination_content_width[window],
            destination_content_height[window],
            effective_geometry[window].width,
            effective_geometry[window].height,
        ]);
        let dock = dock_geometry[window];
        let inactive = emit_complement(c, next_visible[window], unit);
        dock_items.extend([dock.left, dock.top, dock.width, dock.height, inactive, focused_window[window], next_open[window]]);
    }

    let photos_output_start = PHOTOS_WINDOW_INDEX * WINDOW_OUTPUT_FIELD_SIZE;
    let photos_destination = effective_geometry[PHOTOS_WINDOW_INDEX];
    let photos_focused = windows[photos_output_start + window_output_field::FOCUSED];
    let blocked_sum = affine(c, &[(cells.photos.click_active, 1.0), (photos_content_click, -1.0)]);
    let photos_click_blocked = emit_relu(c, blocked_sum);
    let interactive_sum = affine(c, &[(photos_focused, 1.0), (photos_click_blocked, -1.0)]);
    let photos_interactive = emit_relu(c, interactive_sum);
    let click_x = affine(c, &[(up.x, 1.0), (photos_destination.left, -1.0), (unit, -1.0)]);
    let click_y = affine(c, &[(up.y, 1.0), (photos_destination.top, -1.0), (cells.title_height, -1.0), (unit, -1.0)]);
    let photos = build_photos(
        c,
        &mut table,
        &cells.photos,
        &PhotoHost {
            viewport_width: destination_content_width[PHOTOS_WINDOW_INDEX],
            viewport_height: destination_content_height[PHOTOS_WINDOW_INDEX],
            click_x,
            click_y,
            present: windows[photos_output_start + window_output_field::OPACITY],
            interactive: photos_interactive,
            unit,
            spring_matrix: cells.spring_matrix,
            settle_check: cells.settle_check,
            snap: cells.snap,
        },
    );
    let all_activity = affine(c, &[(window_animation_active, 1.0), (photos.animation_active, 1.0)]);
    let animation_active = emit_clamp_zero_one(c, all_activity, unit);

    let list_height = c.affine(
        &[(destination_content_height[MONITOR_WINDOW_INDEX], 1.0)],
        -MONITOR_LAYOUT.header_height - MONITOR_LAYOUT.footer_height,
    );
    let list_viewport_height = emit_relu(c, list_height);
    let list_click_y = affine(c, &[(up.y, 1.0), (value_list_top, -1.0)]);
    let value_list = build_value_list(
        c,
        &mut table,
        &cells.value_list,
        &ValueListHost {
            viewport_height: list_viewport_height,
            focused: focused_window[MONITOR_WINDOW_INDEX],
            list_click_active: value_list_click,
            list_click_y,
            location_clicks: value_list_location_click,
        },
    );

    let finished = table.finish(&mut circuit);
    let output = WindowManagerOutput {
        next_source: finished.next_source,
        windows,
        dock_items,
        animation_active,
        photos,
        value_list,
    };
    let source_cells = (0..table.size())
        .map(|index| {
            let wire = circuit.source(index);
            (table.name_of(wire), table.policy_of(wire).expect("every source wire is a declared cell"))
        })
        .collect();
    BuiltWindowManager {
        circuit,
        cells,
        output,
        source_cells,
        initial_source: finished.initial_source,
        validator: finished.validator,
    }
}

fn name_output_view(prefix: &str, wires: &[Wire], fields: &[&str], names: &mut Vec<(Wire, String)>) {
    let single = wires.len() == fields.len();
    for (index, &wire) in wires.iter().enumerate() {
        let name = if single {
            format!("{prefix}.{}", fields[index])
        } else {
            format!("{prefix}.{}.{}", index / fields.len(), fields[index % fields.len()])
        };
        names.push((wire, name));
    }
}

/// Every output that the page draws from, with its name, in the order of the
/// compiled program's roots after the next-state values.
pub fn name_window_manager_outputs(output: &WindowManagerOutput) -> Vec<(Wire, String)> {
    let mut names = Vec::new();
    name_output_view("windows", &output.windows, &WINDOW_OUTPUT_FIELD_NAMES, &mut names);
    name_output_view("dockItems", &output.dock_items, &DOCK_OUTPUT_FIELD_NAMES, &mut names);
    names.push((output.animation_active, "animationActive".to_string()));
    let photos = &output.photos;
    names.push((photos.backdrop_opacity, "photos.backdropOpacity".to_string()));
    name_output_view("photos.photos", &photos.photos, &PHOTO_FIELD_NAMES, &mut names);
    name_output_view("photos.thumbnails", &photos.thumbnails, &THUMBNAIL_FIELD_NAMES, &mut names);
    name_output_view("photos.previousControl", &photos.previous_control, &CONTROL_FIELD_NAMES, &mut names);
    name_output_view("photos.nextControl", &photos.next_control, &CONTROL_FIELD_NAMES, &mut names);
    let list = &output.value_list;
    names.push((list.total_height, "valueList.totalHeight".to_string()));
    names.push((list.scroll, "valueList.scroll".to_string()));
    names.push((list.scroll_target, "valueList.scrollTarget".to_string()));
    names.push((list.scroll_request, "valueList.scrollRequest".to_string()));
    names.push((list.slot_overflow, "valueList.slotOverflow".to_string()));
    names.push((list.selected, "valueList.selected".to_string()));
    for (location, &wire) in list.locations.iter().enumerate() {
        names.push((wire, format!("valueList.locations.{location}")));
    }
    name_output_view("valueList.slots", &list.slots, &SLOT_FIELD_NAMES, &mut names);
    names
}

/// validateWindowManagerOutput.
pub fn validate_window_manager_output(values: &[f32], output: &WindowManagerOutput, tolerance: f64) -> Result<(), String> {
    validate_photo_gallery_output(values, &output.photos, tolerance)?;
    super::value_list::validate_value_list_output(values, &output.value_list)?;
    let value = |wire: Wire| values[wire.index()];
    assert_unit_interval(value(output.animation_active), "animation active", tolerance)?;
    for window in 0..WINDOW_COUNT {
        let start = window * WINDOW_OUTPUT_FIELD_SIZE;
        for field in [2, 3, 9, 10, 11, 12] {
            assert_non_negative_finite(value(output.windows[start + field]), "projected window geometry", tolerance)?;
        }
        for field in [4, 5, 6, 8] {
            assert_unit_interval(value(output.windows[start + field]), "projected window flag", tolerance)?;
        }
        for field in [0, 1] {
            if !value(output.windows[start + field]).is_finite() {
                return Err("projected window translation must be finite".to_string());
            }
        }
        assert_non_negative_finite(value(output.windows[start + 7]), "projected stack", tolerance)?;
    }
    let photos_focused =
        value(output.windows[PHOTOS_WINDOW_INDEX * WINDOW_OUTPUT_FIELD_SIZE + window_output_field::FOCUSED]);
    if photos_focused != 0.0 && photos_focused != 1.0 {
        return Err(format!("Photos interactive must equal zero or one, got {photos_focused}"));
    }
    for window in 0..WINDOW_COUNT {
        let start = window * DOCK_OUTPUT_FIELD_SIZE;
        for field in 0..4 {
            assert_non_negative_finite(value(output.dock_items[start + field]), "projected dock geometry", tolerance)?;
        }
        for field in 4..7 {
            assert_unit_interval(value(output.dock_items[start + field]), "projected dock flag", tolerance)?;
        }
    }
    Ok(())
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WindowManagerEventKind {
    Mount,
    Resize,
    PointerDown,
    PointerMove,
    PointerUp,
    PointerCancel,
    PhotoPrevious,
    PhotoNext,
    PhotoClose,
    ValueListKey,
}

impl WindowManagerEventKind {
    pub fn name(self) -> &'static str {
        match self {
            WindowManagerEventKind::Mount => "mount",
            WindowManagerEventKind::Resize => "resize",
            WindowManagerEventKind::PointerDown => "pointer down",
            WindowManagerEventKind::PointerMove => "pointer move",
            WindowManagerEventKind::PointerUp => "pointer up",
            WindowManagerEventKind::PointerCancel => "pointer cancel",
            WindowManagerEventKind::PhotoPrevious => "photo previous",
            WindowManagerEventKind::PhotoNext => "photo next",
            WindowManagerEventKind::PhotoClose => "photo close",
            WindowManagerEventKind::ValueListKey => "value list key",
        }
    }
}

/// What an event writes into the table when it applies.
#[derive(Clone, Debug)]
enum EventWrite {
    Viewport { width: f32, height: f32 },
    Pointer { latch: Latch, x: f32, y: f32 },
    Photo { field: Wire },
    Key { key: ValueListKey },
}

#[derive(Clone, Debug)]
pub struct WindowManagerEvent {
    pub kind: WindowManagerEventKind,
    pub time_stamp: f64,
    write: EventWrite,
}

impl TimedEvent for WindowManagerEvent {
    fn time_stamp(&self) -> f64 {
        self.time_stamp
    }
}

/// The nearest multiple of the pixel grid.
fn quantize_pixels(value: f64) -> f64 {
    assert!(value.is_finite(), "a pixel coordinate or size must be finite");
    js_round(value / PIXEL_GRID) * PIXEL_GRID
}

/// A viewport size on the 1/64 px grid, clamped into the viewport cells' range.
pub fn quantize_viewport(value: f64) -> f64 {
    PIXEL.1.min(PIXEL.0.max(quantize_pixels(value)))
}

fn quantize_scroll_top(value: f64) -> f64 {
    assert!(value.is_finite(), "a scroll position must be finite");
    MAXIMUM_SCROLL_TOP.min(0f64.max((value / SCROLL_GRID).floor() * SCROLL_GRID))
}

/// createWindowManagerMachine.
pub struct WindowManagerMachine {
    pub circuit: Circuit,
    pub cells: WindowManagerCells,
    pub output: WindowManagerOutput,
    pub source: Vec<f32>,
    pub tensor_program: TensorProgram,
    pub source_cells: Vec<(String, CellPolicy)>,
    pub named_outputs: Vec<(Wire, String)>,
    pub roots: Vec<Wire>,
    /// Every root's value in the last evaluation, indexed by wire.
    pub values: Vec<f32>,
    pub clock: SpringClock,
    pub evaluations: Vec<MachineEvaluation>,
    /// The scroll position the last step asks the host to give Monitor's list.
    pub value_list_scroll_to: Option<f64>,
    validator: TableValidator,
    evaluator: TensorEvaluator,
    write_back: WriteBack,
    queued: Vec<(WindowManagerEvent, bool)>,
    mounted: bool,
    queued_viewport: (f64, f64),
    value_list_scroll_top: f64,
    powers: Vec<SpringMatrix>,
    stored_after_last_step: Vec<f32>,
    /// Called with the values and the table after each evaluation.
    pub after_evaluate: Option<super::photo_gallery::AfterEvaluate<WindowManagerEvent>>,
}

impl WindowManagerMachine {
    pub fn new(write_back: WriteBack) -> WindowManagerMachine {
        let BuiltWindowManager { circuit, cells, output, source_cells, initial_source, validator } = build_window_manager();
        let mut source = initial_source;
        let wire_count = validate_circuit(&circuit).expect("the window manager's circuit is valid");
        let named_outputs = name_window_manager_outputs(&output);
        let mut roots = output.next_source.clone();
        roots.extend(named_outputs.iter().map(|&(wire, _)| wire));
        let tensor_program = circuit.compile_tensor_program(Some(&roots), true, CompileOptions::default());
        let row_counts = [
            tensor_program.source_size,
            tensor_program.value_count - tensor_program.source_size,
            named_outputs.len(),
        ];
        for (location, &row_count) in row_counts.iter().enumerate() {
            source[cells.value_list.row_counts[location].index()] = ROW_CAPACITY.min(row_count as f64) as f32;
        }
        validate_tensor_program(&circuit, &tensor_program).expect("the window manager's program is valid");
        let evaluator = circuit.create_tensor_program_evaluator(tensor_program.clone());
        WindowManagerMachine {
            circuit,
            cells,
            output,
            stored_after_last_step: source.clone(),
            source,
            tensor_program,
            source_cells,
            named_outputs,
            roots,
            values: vec![0.0; wire_count],
            clock: STOPPED_SPRING_CLOCK,
            evaluations: Vec::new(),
            value_list_scroll_to: None,
            validator,
            evaluator,
            write_back,
            queued: Vec::new(),
            mounted: false,
            queued_viewport: (0.0, 0.0),
            value_list_scroll_top: 0.0,
            powers: spring_matrix_powers(),
            after_evaluate: None,
        }
    }

    /// The packed values of the last evaluation: every source cell and live
    /// operation, in the program's packed order.
    pub fn packed_values(&self) -> &[f32] {
        &self.evaluator.workspace.packed_values
    }

    fn queue(&mut self, kind: WindowManagerEventKind, time_stamp: Option<f64>, write: EventWrite) {
        assert!(time_stamp.is_none_or(f64::is_finite), "an event timestamp must be finite");
        let event = (WindowManagerEvent { kind, time_stamp: time_stamp.unwrap_or(0.0), write }, time_stamp.is_some());
        if kind == WindowManagerEventKind::Mount {
            self.queued.insert(0, event);
        } else {
            self.queued.push(event);
        }
    }

    fn queue_viewport(&mut self, kind: WindowManagerEventKind, width: f64, height: f64, time_stamp: Option<f64>) {
        self.queued_viewport = (width, height);
        self.queue(kind, time_stamp, EventWrite::Viewport { width: width as f32, height: height as f32 });
    }

    fn receive_pointer(&mut self, kind: WindowManagerEventKind, latch: Latch, x: f64, y: f64, time_stamp: Option<f64>) {
        assert!(x.is_finite() && y.is_finite(), "pointer coordinates must be finite");
        let bound = POINTER_COORDINATE_BOUND;
        let x = bound.min((-bound).max(quantize_pixels(x)));
        let y = bound.min((-bound).max(quantize_pixels(y)));
        self.queue(kind, time_stamp, EventWrite::Pointer { latch, x: x as f32, y: y as f32 });
    }

    pub fn resize(&mut self, width: f64, height: f64, time_stamp: Option<f64>) {
        self.queue_viewport(WindowManagerEventKind::Resize, quantize_viewport(width), quantize_viewport(height), time_stamp);
    }

    pub fn pointer_down(&mut self, x: f64, y: f64, time_stamp: Option<f64>) {
        self.receive_pointer(WindowManagerEventKind::PointerDown, self.cells.down, x, y, time_stamp);
    }

    pub fn pointer_move(&mut self, x: f64, y: f64, time_stamp: Option<f64>) {
        self.receive_pointer(WindowManagerEventKind::PointerMove, self.cells.move_, x, y, time_stamp);
    }

    pub fn pointer_up(&mut self, x: f64, y: f64, time_stamp: Option<f64>) {
        self.receive_pointer(WindowManagerEventKind::PointerUp, self.cells.up, x, y, time_stamp);
    }

    pub fn pointer_cancel(&mut self, x: f64, y: f64, time_stamp: Option<f64>) {
        self.receive_pointer(WindowManagerEventKind::PointerCancel, self.cells.up, x, y, time_stamp);
    }

    pub fn photo_previous(&mut self, time_stamp: Option<f64>) {
        let field = self.cells.photos.previous;
        self.queue(WindowManagerEventKind::PhotoPrevious, time_stamp, EventWrite::Photo { field });
    }

    pub fn photo_next(&mut self, time_stamp: Option<f64>) {
        let field = self.cells.photos.next;
        self.queue(WindowManagerEventKind::PhotoNext, time_stamp, EventWrite::Photo { field });
    }

    pub fn photo_close(&mut self, time_stamp: Option<f64>) {
        let field = self.cells.photos.close;
        self.queue(WindowManagerEventKind::PhotoClose, time_stamp, EventWrite::Photo { field });
    }

    pub fn value_list_key(&mut self, key: ValueListKey, time_stamp: Option<f64>) {
        self.queue(WindowManagerEventKind::ValueListKey, time_stamp, EventWrite::Key { key });
    }

    fn apply(&self, event: &WindowManagerEvent, table: &mut [f32]) {
        let photo_events = photo_events(&self.cells.photos);
        match event.write {
            EventWrite::Viewport { width, height } => {
                table[self.cells.viewport_width.index()] = width;
                table[self.cells.viewport_height.index()] = height;
            }
            EventWrite::Pointer { latch, x, y } => {
                table[latch.active.index()] = 1.0;
                table[latch.x.index()] = x;
                table[latch.y.index()] = y;
                if event.kind == WindowManagerEventKind::PointerCancel {
                    table[self.cells.cancel.index()] = 1.0;
                }
                if event.kind == WindowManagerEventKind::PointerUp
                    && photo_events.iter().all(|field| table[field.index()] == 0.0)
                {
                    table[self.cells.photos.click_active.index()] = 1.0;
                }
            }
            EventWrite::Photo { field } => {
                for event_field in &photo_events {
                    table[event_field.index()] = 0.0;
                }
                table[field.index()] = 1.0;
            }
            EventWrite::Key { key } => {
                for cell in value_list_keys(&self.cells.value_list) {
                    table[cell.index()] = 0.0;
                }
                table[self.cells.value_list.key(key).index()] = 1.0;
            }
        }
    }

    fn held_press_cells(&self) -> Vec<Wire> {
        let mut cells = Vec::new();
        cells.extend(&self.cells.drag);
        cells.extend(&self.cells.resize);
        cells.extend(self.cells.armed.all());
        cells
    }

    fn coalesce_hover_moves(&self, events: Vec<WindowManagerEvent>, frame_time: f64) -> Vec<WindowManagerEvent> {
        if self.clock.animated_until_ms.is_some()
            || events.len() < 2
            || events.iter().any(|event| event.kind != WindowManagerEventKind::PointerMove)
        {
            return events;
        }
        if self.held_press_cells().iter().any(|cell| self.source[cell.index()] != 0.0) {
            return events;
        }
        let config = &DEFAULT_SPRING_CLOCK_CONFIG;
        let frame_time_ms = quantize_time_ms(frame_time, config);
        let apply_time = |event: &WindowManagerEvent| frame_time_ms.min(quantize_time_ms(event.time_stamp, config));
        let mut latest = events[0].clone();
        for event in &events {
            if apply_time(event) >= apply_time(&latest) {
                latest = event.clone();
            }
        }
        vec![latest]
    }

    fn evaluate(&mut self, evaluation: &PlannedEvaluation<WindowManagerEvent>) -> bool {
        let mut table = self.source.clone();
        let matrix = self.powers[evaluation.steps as usize];
        for (entry, &wire) in matrix.iter().zip(&self.cells.spring_matrix) {
            table[wire.index()] = *entry as f32;
        }
        table[self.cells.settle_check.index()] = if evaluation.settle_check { 1.0 } else { 0.0 };
        table[self.cells.snap.index()] = if evaluation.snap { 1.0 } else { 0.0 };
        table[self.cells.value_list.scroll_top.index()] = self.value_list_scroll_top as f32;
        for event in &evaluation.events {
            self.apply(event, &mut table);
        }
        if let Err(message) = self.validator.check(&table) {
            panic!("{message}");
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
        if let Err(message) = validate_window_manager_output(&self.values, &self.output, 1e-5) {
            panic!("{message}");
        }
        if evaluation.writes_back {
            table.copy_from_slice(next_source);
        }
        if self.values[self.output.value_list.scroll_request.index()] == 1.0 {
            self.value_list_scroll_top = self.values[self.output.value_list.scroll_target.index()] as f64;
            self.value_list_scroll_to = Some(self.value_list_scroll_top);
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
        if evaluation.writes_back {
            self.source = table;
        }
        self.values[self.output.animation_active.index()] > 0.0
    }

    /// Runs the frame at `frame_time` and returns the values of its 'frame'
    /// evaluation, indexed by wire.
    pub fn step(&mut self, viewport_width: f64, viewport_height: f64, frame_time: f64, scroll_top_now: Option<f64>) -> &[f32] {
        assert!(frame_time.is_finite() && frame_time >= 0.0, "frame time must be nonnegative");
        if let Some(scroll_top) = scroll_top_now {
            self.value_list_scroll_top = quantize_scroll_top(scroll_top);
        }
        self.value_list_scroll_to = None;
        if self.mounted {
            for (field, (&now, &stored)) in self.source.iter().zip(&self.stored_after_last_step).enumerate() {
                assert!(now.to_bits() == stored.to_bits(), "source cell {field} was written between steps");
            }
        }
        let width = quantize_viewport(viewport_width);
        let height = quantize_viewport(viewport_height);
        if !self.mounted {
            self.mounted = true;
            let earliest = self
                .queued
                .iter()
                .fold(frame_time, |time, (event, has)| time.min(if *has { event.time_stamp } else { frame_time }));
            self.queue_viewport(WindowManagerEventKind::Mount, width, height, Some(earliest));
        } else if width != self.queued_viewport.0 || height != self.queued_viewport.1 {
            self.queue_viewport(WindowManagerEventKind::Resize, width, height, Some(frame_time));
        }
        let events: Vec<WindowManagerEvent> = std::mem::take(&mut self.queued)
            .into_iter()
            .map(|(mut event, has)| {
                if !has {
                    event.time_stamp = frame_time;
                }
                event
            })
            .collect();
        let events = self.coalesce_hover_moves(events, frame_time);
        self.evaluations.clear();
        let clock = self.clock;
        let write_back = self.write_back;
        self.clock = advance_frame(
            clock,
            frame_time,
            &events,
            &mut |evaluation| self.evaluate(evaluation),
            write_back,
            &DEFAULT_SPRING_CLOCK_CONFIG,
        );
        self.stored_after_last_step.copy_from_slice(&self.source);
        &self.values
    }
}

/// What a frame does to the keyboard focus of Notes' text field.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NoteFocusChange {
    Focus,
    FocusWithCaretAtEnd,
    Blur,
}

pub struct NoteFocusFrame {
    pub notes_is_focused_window: bool,
    pub note_has_keyboard_focus: bool,
    pub note_has_had_keyboard_focus: bool,
    pub pressed_or_released: bool,
    pub last_press_by_mouse: bool,
    pub last_press_outside_every_window: bool,
}

/// noteFocusChange.
pub fn note_focus_change(frame: &NoteFocusFrame) -> Option<NoteFocusChange> {
    if !frame.notes_is_focused_window {
        return frame.note_has_keyboard_focus.then_some(NoteFocusChange::Blur);
    }
    if frame.note_has_keyboard_focus
        || !frame.pressed_or_released
        || !frame.last_press_by_mouse
        || frame.last_press_outside_every_window
    {
        return None;
    }
    Some(if frame.note_has_had_keyboard_focus { NoteFocusChange::Focus } else { NoteFocusChange::FocusWithCaretAtEnd })
}
