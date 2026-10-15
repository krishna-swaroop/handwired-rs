// The desk's wallpaper, a port of the geometry of src/handwiredos/wiring.ts:
// one arc per stored weight, from the value that the weight reads to the
// operation that reads it. Instead of stroking a canvas, `wiring_arcs` returns
// the arcs in the order drawWiring strokes them.

use crate::compile::TensorProgram;

/// One stroked arc: the upper half of an ellipse.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Arc {
    /// Drawn with the color of the weights that subtract.
    pub subtracts: bool,
    pub center_x: f64,
    pub center_y: f64,
    pub radius_x: f64,
    pub radius_y: f64,
}

/// The canvas's pixel size and the arcs drawWiring strokes on a picture of
/// `width` × `height` CSS px at `pixel_ratio`.
pub fn wiring_arcs(program: &TensorProgram, width: f64, height: f64, pixel_ratio: f64) -> ((f64, f64), Vec<Arc>) {
    let canvas = (crate::fp::js_round(width * pixel_ratio), crate::fp::js_round(height * pixel_ratio));
    let mut arcs = Vec::new();
    if width <= 0.0 || height <= 0.0 {
        return (canvas, arcs);
    }
    let column_width = width / (program.layers.len() as f64 + 1.0);
    let mut x = vec![0f32; program.value_count];
    for cell in 0..program.source_size {
        x[cell] = (cell as f64 / program.source_size as f64 * column_width) as f32;
    }
    let mut layer_start = program.source_size;
    for (layer_index, layer) in program.layers.iter().enumerate() {
        for row in 0..layer.output_size {
            x[layer_start + row] = ((layer_index as f64 + 1.0 + row as f64 / layer.output_size as f64) * column_width) as f32;
        }
        layer_start += layer.output_size;
    }
    let baseline = height - 0.5 / pixel_ratio;
    let height_per_radius = (baseline / (width / 2.0)).min(1.25);
    for subtracts in [true, false] {
        layer_start = program.source_size;
        for layer in &program.layers {
            let row_count = layer.row_offsets.len() - 1;
            for row in 0..row_count {
                let mut operation_row = row;
                if row >= layer.output_size {
                    let extra = row - layer.output_size;
                    operation_row = if extra < layer.gate_rows.len() {
                        layer.gate_rows[extra] as usize
                    } else {
                        layer.index_reads[extra - layer.gate_rows.len()].row
                    };
                }
                let to = x[layer_start + operation_row] as f64;
                for stored in layer.row_offsets[row] as usize..layer.row_offsets[row + 1] as usize {
                    if (layer.weights[stored] < 0.0) != subtracts {
                        continue;
                    }
                    let from = x[layer.input_indices[layer.columns[stored] as usize] as usize] as f64;
                    let radius = (to - from) / 2.0;
                    if radius <= 0.0 {
                        continue;
                    }
                    arcs.push(Arc {
                        subtracts,
                        center_x: to - radius,
                        center_y: baseline,
                        radius_x: radius,
                        radius_y: radius * height_per_radius,
                    });
                }
            }
            layer_start += layer.output_size;
        }
    }
    (canvas, arcs)
}

/// The opacity of one arc for a program that stores `weight_count` weights.
pub fn arc_opacity(weight_count: usize) -> f64 {
    1f64.min(2_600.0 / weight_count as f64)
}

/// The canvas's rectangle on the desktop for a viewport and the system bar's height.
pub fn wiring_rectangle(viewport_width: f64, viewport_height: f64, system_bar_height: f64) -> [f64; 4] {
    [24.0, system_bar_height + 12.0, 0f64.max(viewport_width - 48.0), 0f64.max(viewport_height - system_bar_height - 18.0)]
}

/// The canvas pixels per CSS px for a picture of `width` × `height` CSS px.
pub fn wiring_pixel_ratio(width: f64, height: f64) -> f64 {
    2f64.min((16_000_000.0 / (width * height).max(1.0)).sqrt())
}
