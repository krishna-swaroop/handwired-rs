// The footer of the Monitor window, a port of the numbers and text of
// src/handwiredos/monitor-footer.ts: how large the compiled program is, and
// the chart of frame times. The DOM writes are left out.

use std::collections::HashSet;

use crate::compile::TensorProgram;

pub const FRAME_CAPACITY: usize = 60;
pub const AVERAGED_FRAMES: usize = 10;

// compactUnsignedIndices' choice of Uint8Array, Uint16Array or Uint32Array.
fn compact_bytes(values: &[u32]) -> usize {
    let maximum = values.iter().copied().max().unwrap_or(0);
    let width = if maximum <= 0xff {
        1
    } else if maximum <= 0xffff {
        2
    } else {
        4
    };
    values.len() * width
}

/// The stored weights, and the bytes of every array the program holds, with
/// the TypeScript runtime's compact index arrays. Index reads of one table
/// share one array of positions, counted once.
pub fn measure_program(program: &TensorProgram) -> (usize, usize) {
    let mut weights = 0;
    let mut bytes = compact_bytes(&program.roots) + compact_bytes(&program.wire_order);
    let mut index_tables: HashSet<Vec<u32>> = HashSet::new();
    for layer in &program.layers {
        weights += layer.weights.len();
        bytes += layer.weights.len() * 4
            + layer.bias.len() * 4
            + compact_bytes(&layer.input_indices)
            + compact_bytes(&layer.gate_rows)
            + compact_bytes(&layer.boolean_rows)
            + compact_bytes(&layer.row_offsets)
            + compact_bytes(&layer.columns);
        for compute in &layer.gather_computes {
            bytes += compact_bytes(&compute.bit_positions);
        }
        for read in &layer.gather_reads {
            bytes += compact_bytes(&read.candidate_positions);
        }
        for read in &layer.index_reads {
            index_tables.insert(read.candidate_positions.clone());
        }
    }
    for table in &index_tables {
        bytes += compact_bytes(table);
    }
    (weights, bytes)
}

/// One point per frame for the last `point_count` frames of `times`: the
/// average of the `window` frames up to the point's frame.
pub fn average_frame_times(times: &[f64], point_count: usize, window: usize) -> Vec<f64> {
    let mut points = Vec::new();
    for last in times.len().saturating_sub(point_count)..times.len() {
        let first = (last + 1).saturating_sub(window);
        let mut sum = 0.0;
        for &time in &times[first..=last] {
            sum += time;
        }
        points.push(sum / (last - first + 1) as f64);
    }
    points
}

/// The top of the chart's scale for the highest point, both in µs.
pub fn frame_time_scale(highest_microseconds: f64) -> f64 {
    let mut magnitude = 100.0;
    loop {
        for step in [1.0, 2.0, 5.0] {
            if step * magnitude >= highest_microseconds {
                return step * magnitude;
            }
        }
        magnitude *= 10.0;
    }
}

/// `Number.prototype.toFixed` for finite values below 1e21: the exact decimal
/// value rounded to `digits` places, a tie rounded away from zero.
pub fn js_to_fixed(value: f64, digits: usize) -> String {
    let negative = value < 0.0;
    let exact = format!("{:.1100}", value.abs());
    let (whole, fraction) = exact.split_once('.').unwrap();
    let kept = &fraction[..digits];
    let rest = &fraction[digits..];
    let mut number: Vec<u8> = format!("{whole}{kept}").into_bytes();
    let round_up = rest.as_bytes().first().is_some_and(|&digit| digit >= b'5');
    if round_up {
        let mut index = number.len();
        loop {
            if index == 0 {
                number.insert(0, b'1');
                break;
            }
            index -= 1;
            if number[index] == b'9' {
                number[index] = b'0';
            } else {
                number[index] += 1;
                break;
            }
        }
    }
    let text = String::from_utf8(number).unwrap();
    let split = text.len() - digits;
    let body = if digits == 0 { text } else { format!("{}.{}", &text[..split], &text[split..]) };
    let is_zero = body.bytes().all(|byte| byte == b'0' || byte == b'.');
    if negative && !is_zero { format!("-{body}") } else { body }
}

/// The chart's line as an SVG path in a box `capacity - 1` wide and 1 tall.
pub fn frame_times_path(times: &[f64], capacity: usize, scale: f64) -> String {
    let first = capacity as i64 - times.len() as i64;
    let points: Vec<String> = times
        .iter()
        .enumerate()
        .map(|(index, &time)| format!("{} {}", first + index as i64, js_to_fixed(1.0 - (time / scale).min(1.0), 3)))
        .collect();
    format!("M{}", points.join("L"))
}

/// A count as the footer shows it, e.g. '4,254'.
pub fn format_count(count: f64) -> String {
    let rounded = crate::fp::js_round(count.abs() * 1000.0) / 1000.0;
    let whole = rounded.trunc() as u64;
    let digits = whole.to_string();
    let mut grouped = String::new();
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(character);
    }
    let fraction = rounded - rounded.trunc();
    if fraction > 0.0 {
        let decimals = js_to_fixed(fraction, 3);
        grouped.push_str(decimals.trim_start_matches('0').trim_end_matches('0'));
    }
    if count < 0.0 { format!("-{grouped}") } else { grouped }
}

/// 1 KB is 1,000 bytes, e.g. 386,312 bytes is ('386', 'KB').
pub fn format_bytes(bytes: f64) -> (String, &'static str) {
    if bytes < 999_500.0 {
        (crate::fp::js_number(crate::fp::js_round(bytes / 1_000.0)), "KB")
    } else {
        (js_to_fixed(bytes / 1_000_000.0, 1), "MB")
    }
}

/// A time in ms with three digits under 100 ms, e.g. '0.16' or '12.3'.
pub fn format_milliseconds(value: f64) -> String {
    if value < 10.0 {
        js_to_fixed(value, 2)
    } else if value < 100.0 {
        js_to_fixed(value, 1)
    } else {
        format_count(crate::fp::js_round(value))
    }
}

/// The footer's counts: layers, weights and the program's size.
pub fn footer_counts(program: &TensorProgram) -> [(String, String); 3] {
    let (weights, bytes) = measure_program(program);
    let (size, unit) = format_bytes(bytes as f64);
    [
        (format_count(program.layers.len() as f64), "layers".to_string()),
        (format_count(weights as f64), "weights".to_string()),
        (size, unit.to_string()),
    ]
}

/// The footer's state between frames: the last frames' times in whole µs.
#[derive(Default)]
pub struct MonitorFooter {
    times: Vec<f64>,
}

/// What one update draws.
#[derive(Clone, Debug, PartialEq)]
pub struct FooterFrame {
    pub frame_time_text: String,
    pub scale: f64,
    pub line: String,
    pub area: String,
    pub end: String,
}

impl MonitorFooter {
    pub fn update(&mut self, frame_ms: f64) -> FooterFrame {
        if self.times.len() == FRAME_CAPACITY + AVERAGED_FRAMES - 1 {
            self.times.remove(0);
        }
        self.times.push(crate::fp::js_round(frame_ms * 1000.0));
        let points = average_frame_times(&self.times, FRAME_CAPACITY, AVERAGED_FRAMES);
        let last_point = *points.last().unwrap();
        let scale = frame_time_scale(points.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        let line = frame_times_path(&points, FRAME_CAPACITY, scale);
        let area = format!("{line}V1H{}Z", FRAME_CAPACITY - points.len());
        let end = format!("{}h0", frame_times_path(&[last_point], FRAME_CAPACITY, scale));
        FooterFrame { frame_time_text: format_milliseconds(last_point / 1000.0), scale, line, area, end }
    }
}
