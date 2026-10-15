// Builds and drives the demo circuits of HandwiredOS and JSON search with the
// Rust port and checks them against conformance/demos.json, which
// conformance/demos.ts records from the TypeScript originals: every
// operation, every compiled layer and wireOrder, and every value along the
// scripted traces, bit for bit. Regenerate the fixtures with
//
//   node crates/handwired/conformance/generate.ts

use std::sync::OnceLock;

use handwired::cells::*;
use handwired::handwiredos::arrangement::{arrange_initial_windows, arrange_windows, ArrangementMetrics, Rectangle, Size};
use handwired::handwiredos::monitor_footer::{average_frame_times, footer_counts, frame_time_scale, frame_times_path, measure_program, MonitorFooter};
use handwired::handwiredos::photo_gallery::{spring_matrix_powers, PhotoGalleryMachine, PHOTO_FIELD_SIZE, THUMBNAIL_FIELD_SIZE};
use handwired::handwiredos::value_list::{create_value_list_circuit, describe_operations, format_value, state_row_name, OPERATION_KIND_NAMES};
use handwired::handwiredos::window_manager::{
    note_focus_change, NoteFocusChange, NoteFocusFrame, WindowManagerMachine, WINDOW_OUTPUT_FIELD_SIZE,
};
use handwired::handwiredos::wiring::{arc_opacity, wiring_arcs};
use handwired::host_timing::*;
use handwired::json_search::json_step::build_json_loop;
use handwired::json_search::loop_emit::{emit_loop_wgsl, RowOrder};
use handwired::json_search::loop_program::{describe_loop, operations_per_document, Binding, LoopProgram, LoopRunner, Write};
use handwired::json_search::program::{write_document_cells, CELLS_PER_DOCUMENT};
use handwired::json_search::regex::{capacity_errors, compile_pattern, SearchCapacity};
use handwired::json_search::search_step::{build_search_loop, SearchLoopOptions, SearchScope};
use handwired::json_search::table_loop::TableLoop;
use handwired::json_search::tables::all_tables;
use handwired::prefix::{brent_kung_prefix, sklansky_prefix};
use handwired::*;
use serde_json::{json, Value};

fn fixtures() -> &'static Value {
    static FIXTURES: OnceLock<Value> = OnceLock::new();
    FIXTURES.get_or_init(|| serde_json::from_str(include_str!("../conformance/demos.json")).expect("demos.json is valid JSON"))
}

// ---------------------------------------------------------------------------
// Exact bits and hashes, as demos.ts computes them.

fn f64_hex(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn hex(value: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(value.as_str().unwrap(), 16).unwrap())
}

fn hexes(values: &[f64]) -> Value {
    json!(values.iter().map(|&value| f64_hex(value)).collect::<Vec<_>>())
}

fn opt_hex(value: Option<f64>) -> Value {
    value.map_or(Value::Null, |value| json!(f64_hex(value)))
}

fn hash_words(words: impl IntoIterator<Item = u32>) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
    }
    hash
}

fn f64_words(value: f64) -> [u32; 2] {
    let bits = if value.is_nan() { 0x7ff8_0000_0000_0000 } else { value.to_bits() };
    [bits as u32, (bits >> 32) as u32]
}

fn f32_words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| if value.is_nan() { 0x7fc0_0000 } else { value.to_bits() }).collect()
}

fn hash_f32(values: &[f32]) -> u32 {
    hash_words(f32_words(values))
}

fn hash_f64(values: &[f64]) -> u32 {
    hash_words(values.iter().flat_map(|&value| f64_words(value)))
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|number| number.as_f64().unwrap()).collect()
}

fn as_u32(value: &Value) -> u32 {
    value.as_u64().unwrap() as u32
}

fn units_string(value: &Value) -> Vec<u16> {
    value.as_array().unwrap().iter().map(|unit| unit.as_u64().unwrap() as u16).collect()
}

fn outcome<T: Into<Value>>(result: Result<T, String>) -> Value {
    match result {
        Ok(value) => value.into(),
        Err(message) => json!({"error": message}),
    }
}

// ---------------------------------------------------------------------------
// Circuits and programs.

fn term_words(terms: &[Term], words: &mut Vec<u32>) {
    words.push(terms.len() as u32);
    for &(wire, coefficient) in terms {
        words.push(wire.index() as u32);
        words.extend(f64_words(coefficient));
    }
}

fn rounding_code(kind: Option<RoundingPointKind>) -> u32 {
    match kind {
        None => 0,
        Some(RoundingPointKind::Boolean) => 1,
        Some(RoundingPointKind::IntegerBit) => 2,
        Some(RoundingPointKind::SignCascade) => 3,
        Some(RoundingPointKind::Marked) => 4,
    }
}

fn operation_hash(operation: &Operation) -> u32 {
    let mut words = vec![match operation.kind {
        AtomKind::Affine => 0,
        AtomKind::Reglu => 1,
        AtomKind::Boolean => 2,
        AtomKind::Index => 3,
    }];
    term_words(&operation.left_terms, &mut words);
    words.extend(f64_words(operation.left_bias));
    term_words(&operation.gate_terms, &mut words);
    words.extend(f64_words(operation.gate_bias));
    words.push(operation.depth);
    words.push(operation.preserve_rounding as u32);
    match operation.macro_ {
        None => words.push(0),
        Some(Macro::Multiply { left, right }) => words.extend([1, left.index() as u32, right.index() as u32]),
        Some(Macro::Select { gate, off, on }) => words.extend([2, gate.index() as u32, off.index() as u32, on.index() as u32]),
    }
    match operation.index_read {
        None => words.push(0),
        Some(read) => {
            let code = |rule: OutOfRange| if rule == OutOfRange::Zero { 1 } else { 0 };
            words.extend([1, read.table as u32, code(read.below), code(read.above)]);
            words.extend(f64_words(read.index_range.minimum));
            words.extend(f64_words(read.index_range.maximum));
        }
    }
    words.push(rounding_code(operation.rounding_point));
    words.push(operation.comparison_bit as u32);
    hash_words(words)
}

fn layer_hash(layer: &TensorLayer) -> u32 {
    let mut words = Vec::new();
    let list = |values: &[u32], words: &mut Vec<u32>| {
        words.push(values.len() as u32);
        words.extend_from_slice(values);
    };
    list(&layer.input_indices, &mut words);
    words.push(layer.input_size as u32);
    words.push(layer.output_size as u32);
    list(&f32_words(&layer.bias), &mut words);
    list(&layer.gate_rows, &mut words);
    list(&layer.boolean_rows, &mut words);
    list(&layer.row_offsets, &mut words);
    list(&layer.columns, &mut words);
    list(&f32_words(&layer.weights), &mut words);
    words.push(layer.gather_computes.len() as u32);
    for compute in &layer.gather_computes {
        words.push(compute.slot as u32);
        list(&compute.bit_positions, &mut words);
    }
    words.push(layer.gather_reads.len() as u32);
    for read in &layer.gather_reads {
        words.extend([read.row as u32, read.slot as u32, read.masked as u32]);
        list(&read.candidate_positions, &mut words);
    }
    words.push(layer.index_reads.len() as u32);
    for read in &layer.index_reads {
        words.extend([
            read.row as u32,
            read.index_row as u32,
            read.below_zero as u32,
            read.above_zero as u32,
            read.candidate_count as u32,
            read.candidate_base as i32 as u32,
            read.candidate_stride as i32 as u32,
        ]);
        list(&read.candidate_positions, &mut words);
    }
    hash_words(words)
}

fn check_circuit(name: &str, circuit: &Circuit, expected: &Value) {
    assert_eq!(circuit.source_size as u64, expected["sourceSize"].as_u64().unwrap(), "{name}: source size");
    assert_eq!(circuit.operations.len() as u64, expected["operationCount"].as_u64().unwrap(), "{name}: operation count");
    for (index, (operation, hash)) in circuit.operations.iter().zip(expected["operationHashes"].as_array().unwrap()).enumerate() {
        assert_eq!(
            operation_hash(operation),
            as_u32(hash),
            "{name}: operation {index} (wire {}) differs: {operation:?}",
            circuit.source_size + index
        );
    }
    let ranges: Vec<f64> = circuit.ranges.iter().flat_map(|range| [range.minimum, range.maximum]).collect();
    assert_eq!(hash_f64(&ranges), as_u32(&expected["rangesHash"]), "{name}: ranges");
    let grids: Vec<f64> = circuit.grids.iter().map(|grid| grid.unwrap_or(-1.0)).collect();
    assert_eq!(hash_f64(&grids), as_u32(&expected["gridsHash"]), "{name}: grids");
    assert_eq!(hash_words(circuit.exact_bits.iter().map(|&bit| bit as u32)), as_u32(&expected["exactBitsHash"]), "{name}: exact bits");
    let groups: Vec<Value> = circuit
        .one_hot_groups
        .iter()
        .map(|group| {
            json!({"members": group.members.iter().map(|member| member.index()).collect::<Vec<_>>(), "allowAllZero": group.allow_all_zero})
        })
        .collect();
    assert_eq!(Value::Array(groups), expected["oneHotGroups"], "{name}: one-hot groups");
    let points: Vec<Value> =
        circuit.rounding_points().iter().map(|&(wire, kind)| json!([wire.index(), rounding_code(Some(kind))])).collect();
    assert_eq!(Value::Array(points), expected["roundingPoints"], "{name}: rounding points");
    assert_eq!(outcome(validate_circuit(circuit).map(|count| json!(count))), expected["validCircuit"], "{name}: validity");
}

fn check_program(name: &str, circuit: &Circuit, program: &TensorProgram, expected: &Value) {
    assert_eq!(program.layers.len() as u64, expected["layerCount"].as_u64().unwrap(), "{name}: layer count");
    for (index, (layer, hash)) in program.layers.iter().zip(expected["layerHashes"].as_array().unwrap()).enumerate() {
        assert_eq!(layer_hash(layer), as_u32(hash), "{name}: compiled layer {index} differs");
    }
    assert_eq!(json!(program.roots), expected["roots"], "{name}: roots");
    assert_eq!(program.source_size as u64, expected["sourceSize"].as_u64().unwrap(), "{name}: program source size");
    assert_eq!(program.value_count as u64, expected["valueCount"].as_u64().unwrap(), "{name}: value count");
    assert_eq!(json!(program.wire_order), expected["wireOrder"], "{name}: wireOrder");
    assert_eq!(program.width as u64, expected["width"].as_u64().unwrap(), "{name}: width");
    assert_eq!(program.gather_slot_count as u64, expected["gatherSlotCount"].as_u64().unwrap(), "{name}: gather slots");
    assert_eq!(outcome(validate_tensor_program(circuit, program).map(|()| json!("ok"))), expected["valid"], "{name}: program validity");
}

// ---------------------------------------------------------------------------
// Runtime helpers.

#[derive(Clone, Debug)]
struct LabeledEvent {
    time_stamp: f64,
    label: &'static str,
}

impl TimedEvent for LabeledEvent {
    fn time_stamp(&self) -> f64 {
        self.time_stamp
    }
}

fn advance_trace(write_back: WriteBack) -> Value {
    let mut clock = STOPPED_SPRING_CLOCK;
    let mut animate_until = f64::NEG_INFINITY;
    let event = |time_stamp: f64, label: &'static str| LabeledEvent { time_stamp, label };
    let script: Vec<(f64, Vec<LabeledEvent>)> = vec![
        (16.0, vec![event(10.0, "a")]),
        (33.0, vec![]),
        (50.0, vec![event(40.0, "b"), event(38.0, "c"), event(40.0, "d")]),
        (66.0, vec![]),
        (74.0, vec![]),
        (83.0, vec![]),
        (400.0, vec![event(120.0, "e")]),
        (2500.0, vec![]),
        (2516.0, vec![event(2600.0, "f")]),
        (2533.0, vec![]),
        (2600.0, vec![]),
        (2617.0, vec![event(2610.0, "g")]),
        (4000.0, vec![]),
        (4016.0, vec![]),
        (4100.5, vec![event(4100.25, "h")]),
        (4200.0, vec![]),
    ];
    let mut frames = Vec::new();
    for (frame_time, events) in script {
        let mut evaluations = Vec::new();
        clock = advance_frame(
            clock,
            frame_time,
            &events,
            &mut |evaluation: &PlannedEvaluation<LabeledEvent>| {
                for _ in &evaluation.events {
                    animate_until = animate_until.max(evaluation.time_ms + 150.0);
                }
                let animating = evaluation.simulated_time_ms < animate_until && !evaluation.snap;
                evaluations.push(json!([
                    evaluation.kind.name(),
                    f64_hex(evaluation.time_ms),
                    f64_hex(evaluation.simulated_time_ms),
                    f64_hex(evaluation.duration_ms),
                    evaluation.steps,
                    evaluation.settle_check,
                    evaluation.snap,
                    evaluation.events.iter().map(|event| event.label).collect::<Vec<_>>(),
                    evaluation.writes_back,
                    animating,
                ]));
                animating
            },
            write_back,
            &DEFAULT_SPRING_CLOCK_CONFIG,
        );
        frames.push(json!({
            "evaluations": evaluations,
            "clock": [opt_hex(clock.animated_until_ms), f64_hex(clock.check_origin_ms), opt_hex(clock.last_evaluation_ms)],
        }));
    }
    Value::Array(frames)
}

#[test]
fn runtime_helpers_match() {
    let expected = &fixtures()["runtime"];
    for (count, prefixes) in expected["prefixes"].as_array().unwrap().iter().enumerate() {
        let items: Vec<String> = (0..count).map(|index| index.to_string()).collect();
        let combine = |earlier: &String, later: &String| format!("({earlier},{later})");
        assert_eq!(json!(sklansky_prefix(&items, combine)), prefixes["sklansky"], "Sklansky prefixes of {count}");
        assert_eq!(json!(brent_kung_prefix(&items, combine)), prefixes["brentKung"], "Brent-Kung prefixes of {count}");
    }
    let config = DEFAULT_SPRING_CLOCK_CONFIG;
    let recorded: Vec<f64> = expected["config"].as_array().unwrap().iter().map(|value| value.as_f64().unwrap()).collect();
    assert_eq!(
        recorded,
        [config.time_grid, config.step_ms, config.max_steps_per_frame as f64, config.settle_check_steps as f64]
    );
    let constants = [
        SpringConstants { stiffness: 650.0, damping: 45.0 },
        SpringConstants { stiffness: 170.0, damping: 26.0 },
        SpringConstants { stiffness: 0.0, damping: 0.0 },
        SpringConstants { stiffness: 4000.0, damping: 10.0 },
    ];
    for (spring, expected) in constants.iter().zip(expected["springs"].as_array().unwrap()) {
        let step = spring_step_matrix(*spring, config.step_ms);
        assert_eq!(hexes(&step), expected["step"], "{spring:?}: step matrix");
        let powers: Vec<f64> = (0..=300).flat_map(|steps| spring_matrix_power(&step, steps)).collect();
        assert_eq!(hash_f64(&powers), as_u32(&expected["powersHash"]), "{spring:?}: matrix powers");
        assert_eq!(hexes(&spring_matrix_power(&step, 17)), expected["power17"], "{spring:?}: 17 steps");
        assert_eq!(json!(f64_hex(spring_matrix_bound(*spring, &config))), expected["bound"], "{spring:?}: bound");
    }
    let timings: [(f64, Option<f64>, Vec<f64>); 6] = [
        (16.6667, None, vec![10.2, 16.6, 20.0]),
        (1000.123, Some(983.5), vec![980.0, 990.01, 1000.123, 1005.0]),
        (5000.0, Some(1000.0), vec![1500.0, 4999.99]),
        (0.0, None, vec![]),
        (33.3, Some(16.6), vec![-5.0, 16.6, 33.3]),
        (-0.001, None, vec![-0.001]),
    ];
    for ((frame, previous, events), expected) in timings.iter().zip(expected["frameTimings"].as_array().unwrap()) {
        let timing = quantize_frame_timing(*frame, *previous, events, &config);
        assert_eq!(
            json!([f64_hex(timing.frame_time_ms), f64_hex(timing.duration_ms), hexes(&timing.event_times_ms)]),
            *expected,
            "frame timing at {frame}"
        );
    }
    let counts = [(0.0, 300), (17.5, 300), (1300.0, 300), (1200.0, 300), (4.0, 0), (3.984375, 1), (8.0, 1)];
    for ((pending, budget), expected) in counts.iter().zip(expected["stepCounts"].as_array().unwrap()) {
        let count = count_spring_steps(*pending, *budget, &config);
        assert_eq!(json!([count.steps, f64_hex(count.leftover_ms), count.snap]), *expected, "steps for {pending} ms");
    }
    assert_eq!(advance_trace(WriteBack::EveryEvaluation), expected["advance"]["every"], "advanceFrame, every evaluation");
    assert_eq!(advance_trace(WriteBack::EventsAndSettleChecks), expected["advance"]["events"], "advanceFrame, events and settle checks");
    check_cells(&expected["cells"]);
}

fn check_cells(expected: &Value) {
    let mut table = SourceTable::new();
    let width = table.cell("width", input((0.0, 100.0)).initial(40.0).grid(1.0));
    let margin = table.cell("margin", constant(4.0, (0.0, 10.0)));
    let press_active = table.cell("press.active", event().grid(1.0));
    let press_x = table.cell("press.x", input((-50.0, 50.0)).grid(0.25));
    let count = table.cell("count", state((0.0, 1000.0)).grid(1.0));
    let on = table.cell("on", state_bit(1.0));
    let tabs = table.one_hot("tabs", &[state_bit(1.0), state_bit(0.0), state_bit(0.0)], false, true);
    let order = table.permutation("order", &[1, 0], state_bit(0.0));
    let spare = table.cell("spare", input_bit(0.0));
    table.at_most_one("press and spare", &[press_active, spare]);
    let mut circuit = table.create_circuit();
    let next_count = circuit.round_affine(&[(count, 1.0), (press_active, 1.0)], 0.0);
    table.next(&circuit, &[count], &[next_count]);
    let next_on = circuit.affine(&[(on, -1.0)], 1.0);
    table.next(&circuit, &[on], &[next_on]);
    table.next(&circuit, &tabs, &[tabs[2], tabs[0], tabs[1]]);
    table.next(&circuit, &[order[0][0], order[0][1], order[1][0], order[1][1]], &[order[1][0], order[1][1], order[0][0], order[0][1]]);
    let finished = table.finish(&mut circuit);
    let sum = circuit.affine(&[(width, 1.0), (margin, 2.0), (press_x, 1.0)], 0.0);
    let half = circuit.affine(&[(count, 0.5)], 0.0);
    let names: Vec<String> = (0..table.size()).map(|index| table.name_of(Wire::from_position(index))).collect();
    assert_eq!(json!(names), expected["names"], "cell names");
    let policies: Vec<&str> =
        (0..table.size()).map(|index| table.policy_of(Wire::from_position(index)).unwrap().name()).collect();
    assert_eq!(json!(policies), expected["policies"], "cell policies");
    assert_eq!(json!(f32_words(&finished.initial_source)), expected["initialSource"], "initial table");
    assert_eq!(
        json!(finished.next_source.iter().map(|wire| wire.index()).collect::<Vec<_>>()),
        expected["nextSource"],
        "next-state wires"
    );
    check_circuit("cells", &circuit, &expected["circuit"]);
    let broken: [(usize, f32); 12] =
        [(0, 101.0), (0, 2.5), (2, 0.5), (3, 0.1), (5, 2.0), (6, 1.0), (8, 1.0), (10, 0.0), (11, 1.0), (4, f32::NAN), (4, -1.0), (12, 1.0)];
    for ((cell, value), expected) in broken.iter().zip(expected["verdicts"].as_array().unwrap()) {
        let mut source = finished.initial_source.clone();
        source[*cell] = *value;
        assert_eq!(outcome(finished.validator.check(&source).map(|()| json!("ok"))), *expected, "cell {cell} at {value}");
    }
    let initial = finished.initial_source.clone();
    let mut evaluator = TableEvaluator::new(circuit, finished, vec![("sum".into(), sum), ("half".into(), half)]);
    let mut source = initial;
    for (frame, expected) in expected["evaluations"].as_array().unwrap().iter().enumerate() {
        if frame % 2 == 0 {
            source[press_active.index()] = 1.0;
            source[press_x.index()] = (frame as f64 * 3.25 - 7.0) as f32;
        }
        let result = evaluator.evaluate(&source).unwrap();
        assert_eq!(json!(f32_words(&result.next_values)), expected["next"], "cells frame {frame}");
        assert_eq!(json!(f64_hex(result.outputs[0] as f64)), expected["sum"], "cells frame {frame} sum");
        assert_eq!(json!(f64_hex(result.outputs[1] as f64)), expected["half"], "cells frame {frame} half");
        source = result.next_values;
    }
}

// ---------------------------------------------------------------------------
// HandwiredOS.

fn evaluation_json(evaluation: &handwired::handwiredos::photo_gallery::MachineEvaluation) -> Value {
    json!([evaluation.kind.name(), f64_hex(evaluation.time_ms), evaluation.steps, evaluation.writes_back, evaluation.events])
}

fn opt_time(value: &Value) -> Option<f64> {
    if value.is_null() { None } else { Some(hex(value)) }
}

#[test]
fn window_manager_matches() {
    let expected = &fixtures()["windowManager"];
    let machine = WindowManagerMachine::new(WriteBack::EventsAndSettleChecks);
    check_circuit("window manager", &machine.circuit, &expected["circuit"]);
    check_program("window manager", &machine.circuit, &machine.tensor_program, &expected["program"]);
    let cells: Vec<Value> = machine.source_cells.iter().map(|(name, policy)| json!([name, policy.name()])).collect();
    assert_eq!(Value::Array(cells), expected["sourceCells"], "source cells");
    let state_names: Vec<String> = machine.source_cells.iter().map(|(name, _)| state_row_name(name)).collect();
    assert_eq!(json!(state_names), expected["stateNames"], "Monitor's State rows");
    let outputs: Vec<Value> = machine.named_outputs.iter().map(|(wire, name)| json!([wire.index(), name])).collect();
    assert_eq!(Value::Array(outputs), expected["namedOutputs"], "named outputs");
    assert_eq!(json!(f32_words(&machine.source)), expected["initialSource"], "initial table");
    assert_eq!(hash_f32(&machine.circuit.evaluate_reference(&machine.source)), as_u32(&expected["referenceHash"]), "reference evaluation");
    assert_eq!(
        hash_f32(&machine.circuit.evaluate_tensor(&machine.source, Some(&machine.tensor_program))),
        as_u32(&expected["tensorHash"]),
        "tensor evaluation"
    );
    let rows = describe_operations(&machine.tensor_program);
    assert_eq!(
        hash_words(rows.iter().flat_map(|row| [row.wire, row.kind as u32, row.layer, row.weights])),
        as_u32(&expected["operationRowsHash"]),
        "Monitor's Operations rows"
    );
    let kinds: Vec<(String, u64)> = expected["operationKinds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|kind| (kind["name"].as_str().unwrap().to_string(), kind["count"].as_u64().unwrap()))
        .collect();
    let counted: Vec<(String, u64)> = OPERATION_KIND_NAMES
        .iter()
        .enumerate()
        .map(|(index, name)| (name.to_string(), rows.iter().filter(|row| row.kind as usize == index).count() as u64))
        .filter(|(_, count)| *count > 0)
        .collect();
    assert_eq!(counted, kinds, "operation kinds");
    let (weights, bytes) = measure_program(&machine.tensor_program);
    assert_eq!(json!({"weights": weights, "bytes": bytes}), expected["measure"], "program size");
    check_footer(&machine.tensor_program, &expected["footer"]);
    check_wiring(&machine.tensor_program, &expected["wiring"]);
    check_window_manager_trace(&expected["trace"]);
}

fn check_footer(program: &TensorProgram, expected: &Value) {
    let counts: Vec<String> =
        footer_counts(program).iter().map(|(count, noun)| format!("<b>{count}</b> {noun}")).collect();
    assert_eq!(json!(counts), expected["counts"], "footer counts");
    let mut footer = MonitorFooter::default();
    let mut seed: u32 = 7;
    for (frame, expected) in expected["frames"].as_array().unwrap().iter().enumerate() {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let ms = if frame == 40 { 3.7 } else { (seed % 400) as f64 / 1000.0 + 0.05 };
        let drawn = footer.update(ms);
        assert_eq!(json!([f64_hex(ms), drawn.frame_time_text, drawn.line, drawn.area, drawn.end]), *expected, "footer frame {frame}");
    }
    let times: Vec<f64> = (0..75).map(|index| ((index * 37) % 113) as f64 + 0.25).collect();
    assert_eq!(hexes(&average_frame_times(&times, 60, 10)), expected["averages"], "frame averages");
    let scales: Vec<f64> =
        [0.0, 99.0, 100.0, 101.0, 320.0, 999.0, 1000.0, 4999.0, 5001.0, 123456.0].iter().map(|&value| frame_time_scale(value)).collect();
    assert_eq!(scales, numbers(&expected["scales"]), "frame time scales");
    assert_eq!(json!(frame_times_path(&[0.5, 1.0, 0.0625, 0.3, 2.0], 60, 1.0)), expected["path"], "frame times path");
}

fn check_wiring(program: &TensorProgram, expected: &Value) {
    let pictures = [(1232.0, 746.0, 2.0), (300.0, 100.0, 1.5)];
    for ((width, height, ratio), expected) in pictures.iter().zip(expected["pictures"].as_array().unwrap()) {
        let (canvas, arcs) = wiring_arcs(program, *width, *height, *ratio);
        let words: Vec<f64> = arcs
            .iter()
            .flat_map(|arc| [if arc.subtracts { 1.0 } else { 0.0 }, arc.center_x, arc.center_y, arc.radius_x, arc.radius_y])
            .collect();
        let subtracts = arcs.iter().filter(|arc| arc.subtracts).count();
        assert_eq!(vec![canvas.0, canvas.1], numbers(&expected["canvas"]), "wallpaper canvas at {width} × {height}");
        assert_eq!(
            json!({"arcCount": arcs.len(), "subtractsCount": subtracts, "arcsHash": hash_f64(&words)}),
            json!({"arcCount": expected["arcCount"], "subtractsCount": expected["subtractsCount"], "arcsHash": expected["arcsHash"]}),
            "wallpaper at {width} × {height}"
        );
    }
    let weights: usize = program.layers.iter().map(|layer| layer.weights.len()).sum();
    assert_eq!(json!(f64_hex(arc_opacity(weights))), expected["opacity"], "arc opacity");
}

fn check_window_manager_trace(expected: &Value) {
    let mut machine = WindowManagerMachine::new(WriteBack::EventsAndSettleChecks);
    let hashes = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let recorded = hashes.clone();
    machine.after_evaluate = Some(Box::new(move |values: &[f32], table: &[f32], _: &PlannedEvaluation<_>| {
        recorded.borrow_mut().push(json!([hash_f32(values), hash_f32(table)]));
    }));
    let cells = machine.cells.clone();
    let rectangles = arrange_initial_windows(&mut machine.source, &cells, 1280.0, 800.0);
    let recorded: Vec<Value> = rectangles.iter().map(|r| hexes(&[r.left, r.top, r.width, r.height])).collect();
    assert_eq!(Value::Array(recorded), expected["rectangles"], "the windows' first rectangles");
    assert_eq!(hash_f32(&machine.source), as_u32(&expected["initialAfterArrangement"]), "the arranged table");
    // The arrangement is written before the first step, so it commits with the mount.
    for (index, step) in expected["steps"].as_array().unwrap().iter().enumerate() {
        for event in step["events"].as_array().unwrap() {
            let time = opt_time(&event["t"]);
            match event["kind"].as_str().unwrap() {
                "down" => machine.pointer_down(hex(&event["x"]), hex(&event["y"]), time),
                "move" => machine.pointer_move(hex(&event["x"]), hex(&event["y"]), time),
                "up" => machine.pointer_up(hex(&event["x"]), hex(&event["y"]), time),
                "cancel" => machine.pointer_cancel(hex(&event["x"]), hex(&event["y"]), time),
                "previous" => machine.photo_previous(time),
                "next" => machine.photo_next(time),
                "close" => machine.photo_close(time),
                "resize" => machine.resize(hex(&event["width"]), hex(&event["height"]), time),
                "key" => {
                    use handwired::handwiredos::value_list::ValueListKey::*;
                    let key = match event["key"].as_str().unwrap() {
                        "up" => Up,
                        "down" => Down,
                        "pageUp" => PageUp,
                        "pageDown" => PageDown,
                        "home" => Home,
                        _ => End,
                    };
                    machine.value_list_key(key, time);
                }
                other => panic!("unknown event {other}"),
            }
        }
        hashes.borrow_mut().clear();
        let scroll_top = opt_time(&step["scrollTop"]);
        let values = machine.step(hex(&step["width"]), hex(&step["height"]), hex(&step["frameTime"]), scroll_top).to_vec();
        let context = format!("window manager step {index}");
        let evaluations: Vec<Value> = machine.evaluations.iter().map(evaluation_json).collect();
        assert_eq!(Value::Array(evaluations), step["evaluations"], "{context}: evaluations");
        assert_eq!(Value::Array(hashes.borrow().clone()), step["evaluationHashes"], "{context}: values per evaluation");
        let windows: Vec<f32> = machine.output.windows.iter().map(|wire| values[wire.index()]).collect();
        assert_eq!(json!(f32_words(&windows)), step["windows"], "{context}: window outputs");
        assert_eq!(hash_f32(&values), as_u32(&step["valuesHash"]), "{context}: values");
        assert_eq!(hash_f32(&machine.source), as_u32(&step["sourceHash"]), "{context}: stored table");
        assert_eq!(hash_f32(machine.packed_values()), as_u32(&step["packedHash"]), "{context}: packed values");
        let clock = machine.clock;
        assert_eq!(
            json!([opt_hex(clock.animated_until_ms), f64_hex(clock.check_origin_ms), opt_hex(clock.last_evaluation_ms)]),
            step["clock"],
            "{context}: clock"
        );
        assert_eq!(opt_hex(machine.value_list_scroll_to), step["scrollTo"], "{context}: scroll request");
    }
    assert_eq!(machine.output.windows.len(), 3 * WINDOW_OUTPUT_FIELD_SIZE);
}

#[test]
fn photo_gallery_matches() {
    let expected = &fixtures()["photoGallery"];
    let mut machine = PhotoGalleryMachine::new(WriteBack::EventsAndSettleChecks);
    check_circuit("photo gallery", &machine.circuit, &expected["circuit"]);
    check_program("photo gallery", &machine.circuit, &machine.tensor_program, &expected["program"]);
    assert_eq!(json!(f32_words(&machine.source)), expected["initialSource"], "initial table");
    let powers: Vec<f64> = spring_matrix_powers().into_iter().flatten().collect();
    assert_eq!(hash_f64(&powers), as_u32(&expected["powersHash"]), "spring matrix powers");
    assert_eq!(machine.output.photos.len(), 12 * PHOTO_FIELD_SIZE);
    assert_eq!(machine.output.thumbnails.len(), 12 * THUMBNAIL_FIELD_SIZE);
    let hashes = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let recorded = hashes.clone();
    machine.after_evaluate = Some(Box::new(move |values: &[f32], table: &[f32], _: &PlannedEvaluation<_>| {
        recorded.borrow_mut().push(json!([hash_f32(values), hash_f32(table)]));
    }));
    for (index, step) in expected["trace"].as_array().unwrap().iter().enumerate() {
        for action in step["actions"].as_array().unwrap() {
            let time = opt_time(&action["t"]);
            match action["kind"].as_str().unwrap() {
                "click" => machine.click(hex(&action["x"]), hex(&action["y"]), time),
                "previous" => machine.previous(time),
                "next" => machine.next(time),
                "close" => machine.close(time),
                "resize" => machine.resize(hex(&action["width"]), hex(&action["height"]), time),
                other => panic!("unknown action {other}"),
            }
        }
        hashes.borrow_mut().clear();
        let viewport = step["viewport"].as_array().unwrap();
        let values = machine
            .step(hex(&viewport[0]), hex(&viewport[1]), hex(&step["present"]), hex(&step["interactive"]), Some(hex(&step["frameTime"])))
            .to_vec();
        let context = format!("photo gallery step {index}");
        let evaluations: Vec<Value> = machine.evaluations.iter().map(evaluation_json).collect();
        assert_eq!(Value::Array(evaluations), step["evaluations"], "{context}: evaluations");
        assert_eq!(Value::Array(hashes.borrow().clone()), step["evaluationHashes"], "{context}: values per evaluation");
        let photos: Vec<f32> = machine.output.photos.iter().map(|wire| values[wire.index()]).collect();
        assert_eq!(json!(f32_words(&photos)), step["photos"], "{context}: photo outputs");
        assert_eq!(hash_f32(&values), as_u32(&step["valuesHash"]), "{context}: values");
        assert_eq!(hash_f32(&machine.source), as_u32(&step["sourceHash"]), "{context}: stored table");
    }
}

#[test]
fn value_list_matches() {
    let expected = &fixtures()["valueList"];
    let (circuit, _, cells, output, finished) = create_value_list_circuit();
    check_circuit("value list", &circuit, &expected["circuit"]);
    let mut roots = finished.next_source.clone();
    roots.extend([output.total_height, output.scroll, output.scroll_target, output.scroll_request, output.slot_overflow, output.selected]);
    roots.extend(&output.locations);
    roots.extend(&output.slots);
    let program = circuit.compile_tensor_program(Some(&roots), true, CompileOptions::default());
    check_program("value list", &circuit, &program, &expected["program"]);
    assert_eq!(json!(f32_words(&finished.initial_source)), expected["initialSource"], "initial table");
    let mut evaluator = circuit.create_tensor_program_evaluator(program);
    let mut source = finished.initial_source.clone();
    let list = &cells.value_list;
    source[list.row_counts[0].index()] = 234.0;
    source[list.row_counts[1].index()] = 4646.0;
    source[list.row_counts[2].index()] = 777.0;
    struct Input {
        viewport_height: Option<f32>,
        focused: Option<f32>,
        click: Option<f32>,
        location: Option<usize>,
        key: Option<usize>,
        scroll_top: Option<f32>,
    }
    let blank = Input { viewport_height: None, focused: None, click: None, location: None, key: None, scroll_top: None };
    let script = [
        Input { ..blank },
        Input { key: Some(1), ..blank },
        Input { key: Some(1), ..blank },
        Input { key: Some(3), ..blank },
        Input { key: Some(5), ..blank },
        Input { scroll_top: Some(1000.5), ..blank },
        Input { key: Some(4), ..blank },
        Input { click: Some(100.25), ..blank },
        Input { click: Some(3000.0), ..blank },
        Input { location: Some(0), ..blank },
        Input { scroll_top: Some(24.0 * 200.0), ..blank },
        Input { click: Some(-20.0), ..blank },
        Input { focused: Some(0.0), key: Some(1), ..blank },
        Input { viewport_height: Some(1600.5), ..blank },
        Input { key: Some(3), ..blank },
        Input { location: Some(2), key: Some(0), ..blank },
        Input { viewport_height: Some(30.0), ..blank },
        Input { key: Some(2), ..blank },
        Input { location: Some(1), ..blank },
        Input { scroll_top: Some(2_399_000.0), ..blank },
        Input { key: Some(5), ..blank },
        Input { click: Some(700.0), ..blank },
        Input { viewport_height: Some(0.0), key: Some(1), ..blank },
    ];
    let count = finished.next_source.len();
    for (frame, (input, expected)) in script.iter().zip(expected["frames"].as_array().unwrap()).enumerate() {
        if let Some(height) = input.viewport_height {
            source[cells.viewport_height.index()] = height;
        }
        source[cells.focused.index()] = input.focused.unwrap_or(1.0);
        source[cells.list_click_active.index()] = if input.click.is_some() { 1.0 } else { 0.0 };
        source[cells.list_click_y.index()] = input.click.unwrap_or(0.0);
        for (index, wire) in cells.location_clicks.iter().enumerate() {
            source[wire.index()] = if input.location == Some(index) { 1.0 } else { 0.0 };
        }
        for (index, wire) in list.keys.iter().enumerate() {
            source[wire.index()] = if input.key == Some(index) { 1.0 } else { 0.0 };
        }
        if let Some(scroll_top) = input.scroll_top {
            source[list.scroll_top.index()] = scroll_top;
        }
        finished.validator.check(&source).unwrap();
        let root_values = evaluator.evaluate(&source).to_vec();
        let context = format!("value list frame {frame}");
        assert_eq!(hash_f32(&source), as_u32(&expected["source"]), "{context}: table");
        assert_eq!(json!(f32_words(&root_values[count..count + 9])), expected["outputs"], "{context}: outputs");
        assert_eq!(hash_f32(&root_values), as_u32(&expected["roots"]), "{context}: roots");
        assert_eq!(hash_f32(&circuit.evaluate_reference(&source)), as_u32(&expected["reference"]), "{context}: reference");
        source.copy_from_slice(&root_values[..count]);
    }
}

#[test]
fn arrangement_matches() {
    let expected = fixtures()["arrangement"].as_array().unwrap();
    let minimum = Size { width: 280.0, height: 190.0 };
    let natural = [Size { width: 540.0, height: 340.0 }, Size { width: 500.0, height: 366.0 }, Size { width: 560.0, height: 430.0 }];
    let metrics = ArrangementMetrics { title_height: 44.0, photos_gap: 18.0, maximized_margin: 12.0 };
    let mut results = Vec::new();
    let mut width = 120.0;
    while width <= 2400.0 {
        let mut height = 150.0;
        while height <= 1300.0 {
            let workspace = Rectangle { left: 0.0, top: 36.0, width, height: height - 112.0 };
            for r in arrange_windows(workspace, minimum, &natural, &metrics) {
                results.push(hexes(&[r.left, r.top, r.width, r.height]));
            }
            height += 71.25;
        }
        width += 97.5;
    }
    assert_eq!(results.len(), expected.len());
    for (index, (actual, expected)) in results.iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "rectangle {index}");
    }
}

#[test]
fn note_focus_and_value_formats_match() {
    let expected = fixtures()["noteFocus"].as_array().unwrap();
    for (bits, expected) in expected.iter().enumerate() {
        let change = note_focus_change(&NoteFocusFrame {
            notes_is_focused_window: bits & 1 != 0,
            note_has_keyboard_focus: bits & 2 != 0,
            note_has_had_keyboard_focus: bits & 4 != 0,
            pressed_or_released: bits & 8 != 0,
            last_press_by_mouse: bits & 16 != 0,
            last_press_outside_every_window: bits & 32 != 0,
        });
        let name = match change {
            None => Value::Null,
            Some(NoteFocusChange::Focus) => json!("focus"),
            Some(NoteFocusChange::FocusWithCaretAtEnd) => json!("focus with the caret at the end"),
            Some(NoteFocusChange::Blur) => json!("blur"),
        };
        assert_eq!(name, *expected, "note focus for {bits:06b}");
    }
    for expected in fixtures()["formatValues"].as_array().unwrap() {
        let value = hex(&expected[0]);
        assert_eq!(json!(format_value(value)), expected[1], "formatValue({value})");
    }
}

// ---------------------------------------------------------------------------
// JSON search.

fn texts() -> Vec<Vec<u16>> {
    fixtures()["jsonSearch"]["texts"].as_array().unwrap().iter().map(units_string).collect()
}

fn binding_json(binding: Binding) -> Value {
    match binding {
        Binding::Item { buffer } => json!({"kind": "item", "buffer": buffer}),
        Binding::Cell { buffer, column } => json!({"kind": "cell", "buffer": buffer, "column": column}),
        Binding::Register { register } => json!({"kind": "register", "register": register}),
        Binding::Field { field, component } => json!({"kind": "field", "field": field, "component": component}),
    }
}

fn write_json(write: Write) -> Value {
    match write {
        Write::Register { register } => json!({"kind": "register", "register": register}),
        Write::Field { field, component } => json!({"kind": "field", "field": field, "component": component}),
        Write::Position { group } => json!({"kind": "position", "group": group}),
        Write::Document { group, column } => json!({"kind": "document", "group": group, "column": column}),
    }
}

fn wgsl_hash(code: &str) -> u32 {
    hash_words(code.encode_utf16().map(u32::from))
}

fn check_loop(name: &str, program: &LoopProgram, expected: &Value) {
    for (index, (stage, expected)) in program.stages.iter().zip(expected["stages"].as_array().unwrap()).enumerate() {
        let context = format!("{name} stage {index}");
        assert_eq!(json!(stage.name), expected["name"], "{context}: name");
        assert_eq!(json!(stage.repeat.name()), expected["repeat"], "{context}: repeat");
        assert_eq!(json!(stage.shared_offset), expected["sharedOffset"], "{context}: shared offset");
        let bindings: Vec<Value> = stage.bindings.iter().map(|&(wire, binding)| json!([wire.index(), binding_json(binding)])).collect();
        assert_eq!(Value::Array(bindings), expected["bindings"], "{context}: bindings");
        let outputs: Vec<Value> =
            stage.outputs.iter().map(|&(wire, write)| json!({"wire": wire.index(), "write": write_json(write)})).collect();
        assert_eq!(Value::Array(outputs), expected["outputs"], "{context}: outputs");
        check_program(&context, &program.circuits[stage.circuit], &stage.program, &expected["program"]);
    }
    let described: Vec<Value> = describe_loop(program)
        .iter()
        .map(|stage| {
            json!({"name": stage.name, "repeat": stage.repeat.name(), "operations": stage.operations, "layers": stage.layers, "indexReads": stage.index_reads})
        })
        .collect();
    assert_eq!(Value::Array(described), expected["describe"], "{name}: describeLoop");
    assert_eq!(json!(operations_per_document(program)), expected["operationsPerDocument"], "{name}: operations per document");
    assert_eq!(hash_f32(&program.shared_values), as_u32(&expected["sharedValues"]), "{name}: shared values");
    assert_eq!(json!(program.shared_values.len()), expected["sharedLength"], "{name}: shared length");
    let wgsl = emit_loop_wgsl(program, 64, RowOrder::Use);
    let counts = wgsl.row_counts;
    assert_eq!(
        json!({
            "codeHash": wgsl_hash(&wgsl.code),
            "statementCount": wgsl.statement_count,
            "bytes": wgsl.bytes,
            "rowCounts": {"affine": counts.affine, "reglu": counts.reglu, "boolean": counts.boolean, "index": counts.index},
            "outputCounts": wgsl.output_counts,
            "windowedGroups": wgsl.windowed_groups,
            "inputCells": wgsl.input_cells,
        }),
        expected["wgsl"],
        "{name}: WGSL"
    );
    let layered = emit_loop_wgsl(program, 32, RowOrder::Layers);
    assert_eq!(
        json!({"codeHash": wgsl_hash(&layered.code), "statementCount": layered.statement_count, "bytes": layered.bytes}),
        expected["layeredWgsl"],
        "{name}: WGSL in layer order"
    );
}

fn document_cells(texts: &[Vec<u16>]) -> Vec<f32> {
    let mut cells = vec![0.0; texts.len() * CELLS_PER_DOCUMENT];
    for (index, text) in texts.iter().enumerate() {
        write_document_cells(&mut cells, index * CELLS_PER_DOCUMENT, text);
    }
    cells
}

// The highlighter's display kinds of every text, for the masked searches.
fn highlight(texts: &[Vec<u16>], check: Option<&Value>) -> Vec<f32> {
    let json_loop = build_json_loop(CELLS_PER_DOCUMENT);
    let program = &json_loop.program;
    if let Some(expected) = check {
        for (name, values) in all_tables() {
            let recorded: Vec<f64> =
                expected["tables"][name].as_array().unwrap().iter().map(|value| value.as_f64().unwrap()).collect();
            assert_eq!(values, recorded, "highlighter table {name}");
        }
        check_circuit("highlighter", &program.circuits[0], &expected["circuit"]);
        check_loop("highlighter", program, &expected["loop"]);
    }
    let cells = document_cells(texts);
    let mut runner = LoopRunner::new(program);
    let mut outputs = vec![vec![0.0; texts.len() * 4], vec![0.0; texts.len() * 512]];
    for index in 0..texts.len() {
        runner.run(&[&cells], index, &mut outputs, index, Some(index));
    }
    if let Some(expected) = check {
        assert_eq!(hash_f32(&cells), as_u32(&expected["cellsHash"]), "document cells");
        let table_loop = TableLoop::new();
        let mut kinds = vec![0u8; 512];
        for (index, (document, table_expected)) in
            expected["documents"].as_array().unwrap().iter().zip(expected["tableLoop"].as_array().unwrap()).enumerate()
        {
            let text = String::from_utf16_lossy(&texts[index]);
            let summary = &outputs[0][index * 4..index * 4 + 4];
            assert_eq!(json!(f32_words(summary)), document["summary"], "highlighter summary of {text:?}");
            let display = &outputs[1][index * 512..index * 512 + 512];
            assert_eq!(hash_f32(display), as_u32(&document["kindsHash"]), "display kinds of {text:?}");
            let result = table_loop.highlight(&texts[index], &mut kinds);
            assert_eq!(
                json!([
                    result.first_error_position,
                    result.first_error_kind,
                    result.truncated,
                    hash_words(kinds.iter().map(|&kind| kind as u32))
                ]),
                *table_expected,
                "table loop on {text:?}"
            );
        }
    }
    outputs.pop().unwrap()
}

#[test]
fn json_highlighter_matches() {
    let expected = &fixtures()["jsonSearch"]["highlighter"];
    highlight(&texts(), Some(expected));
}

#[test]
fn patterns_match() {
    for entry in fixtures()["jsonSearch"]["patterns"].as_array().unwrap() {
        let pattern = String::from_utf16(&units_string(&entry["pattern"])).unwrap();
        let expected = &entry["compiled"];
        match compile_pattern(&pattern) {
            Err(error) => {
                assert_eq!(json!({"error": error.message, "position": error.position}), *expected, "pattern {pattern:?}");
            }
            Ok(compiled) => {
                let automaton = &compiled.automaton;
                let sets: Vec<Vec<[u32; 2]>> =
                    automaton.state_sets.iter().map(|set| set.iter().map(|&(low, high)| [low, high]).collect()).collect();
                assert_eq!(
                    json!({"stateCount": automaton.state_count, "targets": automaton.targets, "stateSets": sets, "accepting": automaton.accepting}),
                    expected["automaton"],
                    "automaton of {pattern:?}"
                );
                let partition = &compiled.partition;
                assert_eq!(
                    json!({
                        "classCount": partition.class_count,
                        "boundaries": partition.boundaries,
                        "regionClasses": partition.region_classes,
                        "classReads": partition.class_reads,
                    }),
                    expected["partition"],
                    "classes of {pattern:?}"
                );
                let errors: Vec<Vec<String>> = [8, 16, 32]
                    .iter()
                    .map(|&states| {
                        capacity_errors(&compiled, SearchCapacity { states, classes: 16, boundaries: 32 })
                            .into_iter()
                            .map(|error| error.message)
                            .collect()
                    })
                    .collect();
                assert_eq!(json!(errors), expected["capacityErrors"], "capacity errors of {pattern:?}");
            }
        }
    }
}

fn scope_of(value: &Value) -> Option<SearchScope> {
    match value {
        Value::Null => None,
        Value::String(name) => Some(match name.as_str() {
            "everything" => SearchScope::Everything,
            "strings" => SearchScope::Strings,
            "keys" => SearchScope::Keys,
            _ => SearchScope::StringsAndKeys,
        }),
        Value::Array(kinds) => Some(SearchScope::Kinds(kinds.iter().map(|kind| kind.as_u64().unwrap() as usize).collect())),
        other => panic!("unknown scope {other}"),
    }
}

#[test]
fn json_search_matches() {
    let texts = texts();
    let kinds = highlight(&texts, None);
    let cells = document_cells(&texts);
    for expected in fixtures()["jsonSearch"]["search"].as_array().unwrap() {
        let configuration = &expected["configuration"];
        let states = configuration["states"].as_u64().unwrap() as usize;
        let masked = configuration["masked"].as_bool().unwrap();
        let documents = configuration["documents"].as_u64().unwrap() as usize;
        let name = format!("search at {states} states{}", if masked { ", masked" } else { "" });
        let capacity = SearchCapacity { states, classes: 16, boundaries: 32 };
        let built = build_search_loop(
            capacity,
            SearchLoopOptions {
                positions: 512,
                cells_per_document: CELLS_PER_DOCUMENT,
                masked,
                stop_at_length: configuration["stopAtLength"].as_bool().unwrap(),
                bit_fields: configuration["bitFields"].as_bool().unwrap(),
            },
        );
        check_circuit(&name, &built.program.circuits[0], &expected["circuit"]);
        check_loop(&name, &built.program, &expected["loop"]);
        assert_eq!(json!(built.pattern_cells), expected["patternCells"], "{name}: pattern cells");
        let mut runner = LoopRunner::new(&built.program);
        for run in expected["runs"].as_array().unwrap() {
            let pattern = run["pattern"].as_str().unwrap();
            let compiled = compile_pattern(pattern).unwrap();
            let mut shared = built.program.shared_values.clone();
            if let Some(error) = run.get("error") {
                assert_eq!(json!(built.write_pattern(&mut shared, &compiled).unwrap_err()), *error, "{name}: {pattern:?}");
                continue;
            }
            built.write_pattern(&mut shared, &compiled).unwrap();
            if let Some(scope) = scope_of(&run["scope"]) {
                built.write_scope(&mut shared, &scope);
            }
            let context = format!("{name}: {pattern:?} in {}", run["scope"]);
            assert_eq!(hash_f32(&shared), as_u32(&run["shared"]), "{context}: pattern cells");
            runner.set_shared(&shared);
            let mut outputs = vec![vec![0.0; documents], vec![0.0; documents * 512]];
            for document in 0..documents {
                let buffers: Vec<&[f32]> = if masked { vec![&cells, &kinds] } else { vec![&cells] };
                runner.run(&buffers, document, &mut outputs, document, Some(document));
            }
            assert_eq!(json!(f32_words(&outputs[0])), run["counts"], "{context}: match counts");
            for (document, hash) in run["marks"].as_array().unwrap().iter().enumerate() {
                assert_eq!(
                    hash_f32(&outputs[1][document * 512..document * 512 + 512]),
                    as_u32(hash),
                    "{context}: marks of document {document}"
                );
            }
        }
    }
}
