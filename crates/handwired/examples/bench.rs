// The Rust side of the TypeScript-versus-Rust benchmark. Runs the workloads of
// bench/bench.ts on the port and writes their timings and output hashes to
// bench/results-rust.json. bench/compare.ts prints both.
//
// cargo run --release -p handwired --example bench

use std::hint::black_box;
use std::time::{Duration, Instant};

use handwired::handwiredos::arrangement::arrange_initial_windows;
use handwired::handwiredos::value_list::ValueListKey;
use handwired::handwiredos::window_manager::{name_window_manager_outputs, WindowManagerMachine};
use handwired::host_timing::WriteBack;
use handwired::json_search::json_step::build_json_loop;
use handwired::json_search::loop_program::LoopRunner;
use handwired::json_search::program::{write_document_cells, CELLS_PER_DOCUMENT};
use handwired::json_search::regex::{compile_pattern, SearchCapacity};
use handwired::json_search::search_step::{build_search_loop, SearchLoopOptions};
use handwired::CompileOptions;
use serde_json::{json, Value};

// FNV-1a over the FP32 bits, every NaN as one NaN, as in conformance/demos.ts.
fn hash_f32(values: &[f32], mut hash: u32) -> u32 {
    for value in values {
        let mut word = value.to_bits();
        if (word & 0x7f800000) == 0x7f800000 && (word & 0x7fffff) != 0 {
            word = 0x7fc00000;
        }
        for shift in (0..32).step_by(8) {
            hash ^= (word >> shift) & 0xff;
            hash = hash.wrapping_mul(0x01000193);
        }
    }
    hash
}

const FNV: u32 = 0x811c9dc5;

struct Bench {
    target: Duration,
    /// `--only <prefix>` runs only the workloads whose names start with it.
    only: Option<String>,
    results: Vec<Value>,
}

impl Bench {
    // Runs `run` in batches sized to about `target / samples` each, after a
    // warmup, and records the median and fastest time per call.
    fn measure<T>(&mut self, name: &str, unit: &str, hash: Option<u32>, mut run: impl FnMut() -> T) {
        if self.only.as_ref().is_some_and(|only| !name.starts_with(only.as_str())) {
            return;
        }
        let warmup_start = Instant::now();
        let mut calls = 0u32;
        while warmup_start.elapsed() < self.target / 4 || calls < 1 {
            black_box(run());
            calls += 1;
        }
        let per_call = warmup_start.elapsed().as_secs_f64() / calls as f64;
        let target = self.target.as_secs_f64();
        let samples = if per_call > target / 15.0 { 5 } else { 15 };
        let batch = ((target / samples as f64 / per_call).round() as u32).max(1);
        let mut times: Vec<f64> = (0..samples)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..batch {
                    black_box(run());
                }
                start.elapsed().as_secs_f64() * 1e9 / batch as f64
            })
            .collect();
        times.sort_by(f64::total_cmp);
        let median = times[samples / 2];
        println!("{name:<30} {:>12.2} µs/{unit}  (fastest {:.2})", median / 1e3, times[0] / 1e3);
        self.results.push(json!({
            "name": name, "unit": unit, "medianNs": median, "minNs": times[0],
            "iterations": batch * samples as u32, "hash": hash,
        }));
    }
}

const GRAB_OFFSET: (f64, f64) = (60.3, 20.0);

// The 60-frame cycle of bench.ts: drag window 0 by its title bar out and back,
// press a value list key, and let the springs settle.
fn session_frame(machine: &mut WindowManagerMachine, frame: u32, frame_time: f64, grab: (f64, f64)) -> &[f32] {
    let phase = frame % 60;
    let t = Some(frame_time + 8.0);
    let (x, y) = grab;
    if phase == 0 {
        machine.pointer_down(x, y, t);
    } else if phase <= 20 {
        machine.pointer_move(x + phase as f64 * 5.5, y + phase as f64 * 2.25, t);
    } else if phase <= 39 {
        machine.pointer_move(x + (40 - phase) as f64 * 5.5, y + (40 - phase) as f64 * 2.25, t);
    } else if phase == 40 {
        machine.pointer_up(x, y, t);
    } else if phase == 50 {
        machine.value_list_key(if frame % 120 == 50 { ValueListKey::Down } else { ValueListKey::Up }, t);
    }
    machine.step(1280.0, 800.0, frame_time + 16.0, None)
}

fn start_session() -> (WindowManagerMachine, (f64, f64)) {
    let mut machine = WindowManagerMachine::new(WriteBack::EventsAndSettleChecks);
    let cells = machine.cells.clone();
    let rectangles = arrange_initial_windows(&mut machine.source, &cells, 1280.0, 800.0);
    let grab = (rectangles[0].left + GRAB_OFFSET.0, rectangles[0].top + GRAB_OFFSET.1);
    (machine, grab)
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let quick = arguments.iter().any(|argument| argument == "--quick");
    let only = arguments.iter().position(|argument| argument == "--only").map(|index| arguments[index + 1].clone());
    let target = Duration::from_millis(if quick { 300 } else { 2000 });
    let mut bench = Bench { target, only: only.clone(), results: Vec::new() };
    let root = env!("CARGO_MANIFEST_DIR");

    // HandwiredOS: the window manager, which includes Photos and Monitor.
    bench.measure("wm.build", "machine", None, || WindowManagerMachine::new(WriteBack::EventsAndSettleChecks));

    let machine = WindowManagerMachine::new(WriteBack::EventsAndSettleChecks);
    let mut roots = machine.output.next_source.clone();
    roots.extend(name_window_manager_outputs(&machine.output).iter().map(|&(wire, _)| wire));
    let source = machine.source.clone();
    bench.measure("wm.compile", "program", None, || {
        machine.circuit.compile_tensor_program(Some(&roots), true, CompileOptions::default())
    });

    let mut evaluator = machine.circuit.create_tensor_program_evaluator(machine.tensor_program.clone());
    let hash = hash_f32(evaluator.evaluate(&source), FNV);
    bench.measure("wm.evaluate", "frame", Some(hash), || evaluator.evaluate(&source).len());
    let hash = hash_f32(&machine.circuit.evaluate_reference(&source), FNV);
    bench.measure("wm.reference", "frame", Some(hash), || machine.circuit.evaluate_reference(&source));

    let (mut verify, grab) = start_session();
    let mut session_hash = FNV;
    for frame in 0..600 {
        session_hash = hash_f32(session_frame(&mut verify, frame, frame as f64 * 16.0, grab), session_hash);
    }
    let (mut session, grab) = start_session();
    let mut frame = 0;
    bench.measure("wm.session", "frame", Some(session_hash), || {
        let length = session_frame(&mut session, frame, frame as f64 * 16.0, grab).len();
        frame += 1;
        length
    });

    // JSON search over the 84 texts the demo conformance records.
    let demos: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/conformance/demos.json")).unwrap()).unwrap();
    let texts: Vec<Vec<u16>> = demos["jsonSearch"]["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|units| units.as_array().unwrap().iter().map(|unit| unit.as_u64().unwrap() as u16).collect())
        .collect();
    let mut cells = vec![0.0; texts.len() * CELLS_PER_DOCUMENT];
    for (index, text) in texts.iter().enumerate() {
        write_document_cells(&mut cells, index * CELLS_PER_DOCUMENT, text);
    }
    let documents = format!("{} docs", texts.len());

    bench.measure("json.highlight.build", "loop", None, || build_json_loop(CELLS_PER_DOCUMENT));
    let json_loop = build_json_loop(CELLS_PER_DOCUMENT);
    let mut runner = LoopRunner::new(&json_loop.program);
    let mut outputs = vec![vec![0.0; texts.len() * 4], vec![0.0; texts.len() * 512]];
    let mut highlight = |outputs: &mut Vec<Vec<f32>>| {
        for index in 0..texts.len() {
            runner.run(&[&cells], index, outputs, index, Some(index));
        }
    };
    highlight(&mut outputs);
    let hash = hash_f32(&outputs[1], hash_f32(&outputs[0], FNV));
    bench.measure("json.highlight.executor", &documents, Some(hash), || highlight(&mut outputs));

    let capacity = SearchCapacity { states: 16, classes: 16, boundaries: 32 };
    let options = SearchLoopOptions {
        positions: 512,
        cells_per_document: CELLS_PER_DOCUMENT,
        masked: false,
        stop_at_length: false,
        bit_fields: true,
    };
    bench.measure("json.search.build", "loop", None, || build_search_loop(capacity, options));
    let built = build_search_loop(capacity, options);
    let shared: Vec<Vec<f32>> = ["error", "\\d\\d\\d\\d-\\d\\d-\\d\\d", "\"[a-z]+\""]
        .iter()
        .map(|pattern| {
            let mut values = built.program.shared_values.clone();
            built.write_pattern(&mut values, &compile_pattern(pattern).unwrap()).unwrap();
            values
        })
        .collect();
    let mut runner = LoopRunner::new(&built.program);
    let mut outputs = vec![vec![0.0; texts.len()], vec![0.0; texts.len() * 512]];
    let mut search = |hashed: bool| {
        let mut hash = FNV;
        for values in &shared {
            runner.set_shared(values);
            for document in 0..texts.len() {
                runner.run(&[&cells], document, &mut outputs, document, Some(document));
            }
            if hashed {
                hash = hash_f32(&outputs[1], hash_f32(&outputs[0], hash));
            }
        }
        hash
    };
    let hash = search(true);
    bench.measure("json.search.executor", &format!("3 patterns × {documents}"), Some(hash), || search(false));

    if only.is_some() {
        return;
    }
    let report = json!({"runtime": "rust (release)", "results": bench.results});
    std::fs::write(format!("{root}/bench/results-rust.json"), serde_json::to_string_pretty(&report).unwrap() + "\n").unwrap();
}
