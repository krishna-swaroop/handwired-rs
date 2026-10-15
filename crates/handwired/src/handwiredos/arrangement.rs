// The rectangles that HandwiredOS's three windows open at, chosen for the
// viewport, a port of src/handwiredos/arrangement.ts. Plain f64 host code that
// writes the table before the machine's first step.

use super::photo_gallery::MAXIMUM_PHOTO_TALLNESS;
use super::value_list::{LOCATION_COUNT, MONITOR_LAYOUT};
use super::window_manager::WindowManagerCells;
use crate::fp::js_round;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rectangle {
    pub left: f64,
    pub top: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ArrangementMetrics {
    pub title_height: f64,
    pub photos_gap: f64,
    pub maximized_margin: f64,
}

const WINDOW_COUNT: usize = 3;
const BORDER: f64 = 1.0;
const NOTE_SHOWN: f64 = 136.0;

fn monitor_shown_for(available: f64, whole: f64, metrics: &ArrangementMetrics) -> f64 {
    if available >= whole {
        return whole;
    }
    let header = BORDER + metrics.title_height + MONITOR_LAYOUT.header_height;
    if available < header {
        return metrics.title_height;
    }
    header
        + (LOCATION_COUNT as f64).min(((available - header) / MONITOR_LAYOUT.row_height).floor()) * MONITOR_LAYOUT.row_height
}

fn photos_tallest_for(width: f64, metrics: &ArrangementMetrics) -> f64 {
    let photo_width = (width - 2.0 * BORDER - 5.0 * metrics.photos_gap) / 4.0;
    metrics.title_height + 2.0 * BORDER + 4.0 * metrics.photos_gap + 3.0 * MAXIMUM_PHOTO_TALLNESS * photo_width
}

fn clamp(value: f64, minimum: f64, maximum: f64) -> f64 {
    value.max(minimum).min(maximum)
}

pub fn arrange_windows(workspace: Rectangle, minimum: Size, natural: &[Size], metrics: &ArrangementMetrics) -> Vec<Rectangle> {
    assert!(natural.len() == WINDOW_COUNT, "the arrangement needs one size per window");
    let sizes = [natural[0], natural[1], natural[2]];
    let fits_two_rows = workspace.height >= 2.0 * minimum.height + 48.0;
    let rectangles = if !fits_two_rows {
        arrange_row(workspace, minimum, sizes, metrics)
    } else if workspace.width >= 1000.0 {
        arrange_desk(workspace, minimum, sizes, metrics)
    } else if workspace.width >= 640.0 {
        arrange_columns(workspace, minimum, sizes, metrics)
    } else {
        arrange_column(workspace, minimum, sizes, metrics)
    };
    rectangles.into_iter().map(|rectangle| fit_into(rectangle, workspace, minimum)).collect()
}

fn fit_into(rectangle: Rectangle, workspace: Rectangle, minimum: Size) -> Rectangle {
    let width = clamp(js_round(rectangle.width), minimum.width.min(workspace.width), workspace.width);
    let height = clamp(js_round(rectangle.height), minimum.height.min(workspace.height), workspace.height);
    Rectangle {
        left: clamp(js_round(rectangle.left), workspace.left, workspace.left + workspace.width - width),
        top: clamp(js_round(rectangle.top), workspace.top, workspace.top + workspace.height - height),
        width,
        height,
    }
}

fn arrange_desk(workspace: Rectangle, minimum: Size, [monitor, notes, photos]: [Size; 3], metrics: &ArrangementMetrics) -> Vec<Rectangle> {
    let gap = 24.0;
    let margin = 24.0;
    let photos_offset_x = 240.0;
    let photos_offset_y = 126.0;
    let full_width = monitor.width + gap + photos_offset_x + photos.width;
    let width = full_width.min(workspace.width - 2.0 * margin);
    let origin_x = workspace.left + (workspace.width - width) / 2.0;
    let missing = 0f64.max(full_width - width - 120.0);
    let monitor_width = minimum.width.max(monitor.width - (missing / 2.0).ceil());
    let notes_left = monitor_width + gap;
    let photos_left = notes_left + photos_offset_x;
    let photos_width = clamp(width - photos_left, minimum.width, photos.width);
    let height = (photos_offset_y + photos.height.min(photos_tallest_for(photos_width, metrics))).min(workspace.height - 2.0 * margin);
    let photos_height = clamp(height - photos_offset_y, minimum.height, photos.height);
    let origin_y = workspace.top + (workspace.height - height) / 2.0;
    vec![
        Rectangle { left: origin_x, top: origin_y, width: monitor_width, height },
        Rectangle { left: origin_x + notes_left, top: origin_y, width: notes.width, height: notes.height.min(height) },
        Rectangle {
            left: origin_x + photos_left,
            top: origin_y + photos_offset_y.min(height - photos_height),
            width: photos_width,
            height: photos_height,
        },
    ]
}

fn arrange_columns(workspace: Rectangle, minimum: Size, [_, notes, photos]: [Size; 3], metrics: &ArrangementMetrics) -> Vec<Rectangle> {
    let gap = 16.0;
    let margin = 16.0;
    let column_width = ((workspace.width - 2.0 * margin - gap) / 2.0).floor();
    let available = workspace.height - 2.0 * margin;
    let notes_needs = minimum.height.max(2.0 * BORDER + metrics.title_height + NOTE_SHOWN);
    let photos_height = clamp(
        (available - notes_needs - gap).min(photos_tallest_for(column_width, metrics)),
        minimum.height,
        photos.height,
    );
    let notes_height = clamp(available - photos_height - gap, minimum.height, notes.height);
    let height = notes_height + gap + photos_height;
    let origin_y = workspace.top + margin + (available - height) / 2.0;
    let left_column = workspace.left + margin;
    let right_column = left_column + column_width + gap;
    vec![
        Rectangle { left: left_column, top: origin_y, width: column_width, height },
        Rectangle { left: right_column, top: origin_y, width: column_width, height: notes_height },
        Rectangle { left: right_column, top: origin_y + notes_height + gap, width: column_width, height: photos_height },
    ]
}

fn arrange_column(workspace: Rectangle, minimum: Size, [monitor, notes, photos]: [Size; 3], metrics: &ArrangementMetrics) -> Vec<Rectangle> {
    let margin = metrics.maximized_margin;
    let title_bar = metrics.title_height;
    let width = (workspace.width - 2.0 * margin).min(monitor.width);
    let left = workspace.left + (workspace.width - width) / 2.0;
    let available = workspace.height - 2.0 * margin;
    let notes_least = BORDER + title_bar + NOTE_SHOWN;
    let photo_size = (width - 2.0 * BORDER - 5.0 * metrics.photos_gap) / 4.0;
    let photos_height = clamp(
        (title_bar + 2.0 * BORDER + 4.0 * metrics.photos_gap + 3.0 * photo_size)
            .min(available * 0.45)
            .min(available - notes_least - title_bar),
        minimum.height,
        photos.height,
    );
    let monitor_shown = monitor_shown_for(available - photos_height - notes_least, monitor.height + margin, metrics);
    let notes_shown = title_bar.max((available - photos_height - monitor_shown).min(notes.height + margin));
    let height = monitor_shown + notes_shown + photos_height;
    let monitor_top = workspace.top + margin + 0f64.max((available - height) / 2.0);
    let notes_top = monitor_top + monitor_shown;
    let photos_top = notes_top + notes_shown;
    let bottom = photos_top + photos_height;
    vec![
        Rectangle { left, top: monitor_top, width, height: monitor.height.min(bottom - monitor_top) },
        Rectangle { left, top: notes_top, width, height: notes.height.min(bottom - notes_top) },
        Rectangle { left, top: photos_top, width, height: photos_height },
    ]
}

fn arrange_row(workspace: Rectangle, minimum: Size, [monitor, notes, photos]: [Size; 3], metrics: &ArrangementMetrics) -> Vec<Rectangle> {
    let margin = metrics.maximized_margin;
    let available = workspace.width - 2.0 * margin;
    let height = (workspace.height - 2.0 * margin).min(monitor.height);
    let top = workspace.top + (workspace.height - height) / 2.0;
    let shown_at_least: f64 = 88.0;
    let photos_width = minimum.width.min(available);
    let notes_width = clamp(available - photos_width - (monitor.width + margin), minimum.width, notes.width);
    let notes_shown = shown_at_least.max(notes_width.min(available - photos_width - shown_at_least));
    let monitor_shown = clamp(available - photos_width - notes_shown, shown_at_least, monitor.width + margin);
    let width = monitor_shown + notes_shown + photos_width;
    let monitor_left = workspace.left + margin + 0f64.max((available - width) / 2.0);
    let notes_left = monitor_left + monitor_shown;
    let photos_left = notes_left + notes_shown;
    let right = photos_left + photos_width;
    vec![
        Rectangle { left: monitor_left, top, width: monitor.width.min(right - monitor_left), height },
        Rectangle { left: notes_left, top, width: notes_width, height },
        Rectangle { left: photos_left, top, width: photos.width.min(photos_width), height },
    ]
}

/// Chooses the windows' rectangles for the viewport and writes them into the
/// window manager's table, with the destinations of their springs.
pub fn arrange_initial_windows(source: &mut [f32], cells: &WindowManagerCells, viewport_width: f64, viewport_height: f64) -> Vec<Rectangle> {
    let read = |wire: crate::circuit::Wire| source[wire.index()] as f64;
    let dock_top = viewport_height - read(cells.dock_height).min(viewport_height);
    let workspace_top = read(cells.system_bar_height).min(dock_top);
    let workspace = Rectangle { left: 0.0, top: workspace_top, width: viewport_width, height: dock_top - workspace_top };
    let minimum = Size { width: read(cells.min_width), height: read(cells.min_height) };
    let natural: Vec<Size> =
        cells.windows.iter().map(|window| Size { width: read(window.width), height: read(window.height) }).collect();
    let metrics = ArrangementMetrics {
        title_height: read(cells.title_height),
        photos_gap: read(cells.photos.gap),
        maximized_margin: read(cells.margin),
    };
    let rectangles = arrange_windows(workspace, minimum, &natural, &metrics);
    for (index, rectangle) in rectangles.iter().enumerate() {
        let window = cells.windows[index];
        let springs = &cells.springs[index];
        source[window.left.index()] = rectangle.left as f32;
        source[window.top.index()] = rectangle.top as f32;
        source[window.width.index()] = rectangle.width as f32;
        source[window.height.index()] = rectangle.height as f32;
        source[springs[0].destination.index()] = (rectangle.left + rectangle.width / 2.0) as f32;
        source[springs[1].destination.index()] = (rectangle.top + rectangle.height / 2.0) as f32;
        source[springs[2].destination.index()] = rectangle.width as f32;
        source[springs[3].destination.index()] = rectangle.height as f32;
    }
    rectangles
}
