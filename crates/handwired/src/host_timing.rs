// Host timing for springs whose step matrix the host writes into source cells,
// a port of src/runtime/host-timing.ts. Everything here is f64 host code that
// runs between evaluations: times on a 1/64 ms grid, how many fixed 4 ms spring
// steps an evaluation runs, the spring matrix for that step count, and the
// evaluations of one frame, see `advance_frame`.

use crate::fp::{fround, js_round};

#[derive(Copy, Clone, Debug)]
pub struct SpringClockConfig {
    /// Times are rounded to multiples of 1 / time_grid ms.
    pub time_grid: f64,
    /// One spring step simulates this many milliseconds.
    pub step_ms: f64,
    /// The most steps that the evaluations of one frame run together.
    pub max_steps_per_frame: u32,
    /// Settling is checked every this many steps of simulated time.
    pub settle_check_steps: u32,
}

pub const DEFAULT_SPRING_CLOCK_CONFIG: SpringClockConfig =
    SpringClockConfig { time_grid: 64.0, step_ms: 4.0, max_steps_per_frame: 300, settle_check_steps: 16 };

impl Default for SpringClockConfig {
    fn default() -> SpringClockConfig {
        DEFAULT_SPRING_CLOCK_CONFIG
    }
}

pub fn validate_spring_clock_config(config: &SpringClockConfig) {
    let SpringClockConfig { time_grid, step_ms, max_steps_per_frame, settle_check_steps } = *config;
    assert!(
        time_grid >= 1.0 && time_grid.fract() == 0.0 && time_grid.log2().fract() == 0.0,
        "the time grid must be a whole power of two, got {time_grid}"
    );
    assert!(
        step_ms > 0.0 && (step_ms * time_grid).fract() == 0.0,
        "a spring step must be a positive multiple of the time grid"
    );
    assert!(max_steps_per_frame >= 1, "maxStepsPerFrame must be a positive integer");
    assert!(settle_check_steps >= 1, "settleCheckSteps must be a positive integer");
    assert!(
        max_steps_per_frame as f64 * step_ms * time_grid <= 16_777_216.0,
        "maxStepsPerFrame × stepMs must stay within 2^24 grid steps"
    );
}

/// A timestamp rounded to the time grid, with -0 turned into +0.
pub fn quantize_time_ms(time_stamp: f64, config: &SpringClockConfig) -> f64 {
    assert!(time_stamp.is_finite(), "a timestamp must be finite, got {time_stamp}");
    js_round(time_stamp * config.time_grid) / config.time_grid + 0.0
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrameTiming {
    pub frame_time_ms: f64,
    pub duration_ms: f64,
    pub event_times_ms: Vec<f64>,
}

/// What the host writes into a program's time cells for one frame.
pub fn quantize_frame_timing(
    frame_time_stamp: f64,
    previous_frame_time_ms: Option<f64>,
    event_time_stamps: &[f64],
    config: &SpringClockConfig,
) -> FrameTiming {
    validate_spring_clock_config(config);
    let cap_ms = config.max_steps_per_frame as f64 * config.step_ms;
    let mut frame_time_ms = quantize_time_ms(frame_time_stamp, config);
    if let Some(previous) = previous_frame_time_ms {
        frame_time_ms = frame_time_ms.max(previous);
    }
    let duration_ms = previous_frame_time_ms.map_or(0.0, |previous| cap_ms.min(frame_time_ms - previous));
    let earliest_event_ms = previous_frame_time_ms.map_or(-cap_ms, |previous| (-cap_ms).max(previous - frame_time_ms));
    let event_times_ms = event_time_stamps
        .iter()
        .map(|&time_stamp| 0f64.min(earliest_event_ms.max(quantize_time_ms(time_stamp, config) - frame_time_ms)) + 0.0)
        .collect();
    FrameTiming { frame_time_ms, duration_ms, event_times_ms }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct StepCount {
    pub steps: u32,
    pub leftover_ms: f64,
    pub snap: bool,
}

/// How many steps an evaluation runs, or a snap when more than the budget is pending.
pub fn count_spring_steps(pending_ms: f64, step_budget: u32, config: &SpringClockConfig) -> StepCount {
    assert!(
        (pending_ms * config.time_grid).fract() == 0.0 && pending_ms >= 0.0,
        "pending time must lie on the time grid and be nonnegative, got {pending_ms}"
    );
    let steps = (pending_ms / config.step_ms).floor();
    if steps > step_budget as f64 {
        return StepCount { steps: 0, leftover_ms: 0.0, snap: true };
    }
    StepCount { steps: steps as u32, leftover_ms: pending_ms - steps * config.step_ms, snap: false }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpringConstants {
    pub stiffness: f64,
    pub damping: f64,
}

/// (displacement, velocity) after the steps, as
/// displacement' = m[0] × displacement + m[1] × velocity and
/// velocity' = m[2] × displacement + m[3] × velocity.
pub type SpringMatrix = [f64; 4];

pub const IDENTITY_SPRING_MATRIX: SpringMatrix = [1.0, 0.0, 0.0, 1.0];

/// One semi-implicit Euler step as a matrix.
pub fn spring_step_matrix(constants: SpringConstants, step_ms: f64) -> SpringMatrix {
    assert!(
        constants.stiffness.is_finite()
            && constants.stiffness >= 0.0
            && constants.damping.is_finite()
            && constants.damping >= 0.0,
        "stiffness and damping must be finite and nonnegative"
    );
    let dt = step_ms / 1000.0;
    let velocity_keep = 1.0 - constants.damping * dt;
    [1.0 - constants.stiffness * dt * dt, dt * velocity_keep, -constants.stiffness * dt, velocity_keep]
}

pub fn multiply_spring_matrices(left: &SpringMatrix, right: &SpringMatrix) -> SpringMatrix {
    [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
    ]
}

/// matrix^count by repeated squaring.
pub fn spring_matrix_power(matrix: &SpringMatrix, count: u32) -> SpringMatrix {
    let mut result = IDENTITY_SPRING_MATRIX;
    let mut square = *matrix;
    let mut remaining = count;
    while remaining > 0 {
        if remaining % 2 == 1 {
            result = multiply_spring_matrices(&result, &square);
        }
        remaining /= 2;
        if remaining > 0 {
            square = multiply_spring_matrices(&square, &square);
        }
    }
    result
}

/// The magnitude to declare on the 4 matrix cells: the largest |entry| of M^n
/// for n = 0..maxStepsPerFrame, with a little headroom.
pub fn spring_matrix_bound(constants: SpringConstants, config: &SpringClockConfig) -> f64 {
    let step = spring_step_matrix(constants, config.step_ms);
    let mut power = IDENTITY_SPRING_MATRIX;
    let mut bound = 1f64;
    for _ in 1..=config.max_steps_per_frame {
        power = multiply_spring_matrices(&power, &step);
        for entry in power {
            bound = bound.max(entry.abs());
        }
    }
    let declared = fround(bound * (1.0 + 1e-6));
    assert!(declared.is_finite(), "the spring matrix overflows FP32 within maxStepsPerFrame steps");
    declared
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpringClock {
    /// How far springs have stepped, or None while the clock is stopped.
    pub animated_until_ms: Option<f64>,
    pub check_origin_ms: f64,
    /// The time of the last evaluation that wrote state back, or None after idle.
    pub last_evaluation_ms: Option<f64>,
}

pub const STOPPED_SPRING_CLOCK: SpringClock =
    SpringClock { animated_until_ms: None, check_origin_ms: 0.0, last_evaluation_ms: None };

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvaluationKind {
    SettleCheck,
    Events,
    Frame,
    Snap,
}

impl EvaluationKind {
    pub fn name(self) -> &'static str {
        match self {
            EvaluationKind::SettleCheck => "settle check",
            EvaluationKind::Events => "events",
            EvaluationKind::Frame => "frame",
            EvaluationKind::Snap => "snap",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WriteBack {
    EveryEvaluation,
    EventsAndSettleChecks,
}

#[derive(Clone, Debug)]
pub struct PlannedEvaluation<Event> {
    pub kind: EvaluationKind,
    pub time_ms: f64,
    pub simulated_time_ms: f64,
    pub duration_ms: f64,
    pub steps: u32,
    pub settle_check: bool,
    pub snap: bool,
    pub events: Vec<Event>,
    pub writes_back: bool,
}

/// A timestamped event for `advance_frame`.
pub trait TimedEvent: Clone {
    fn time_stamp(&self) -> f64;
}

/// Plans and runs one frame's evaluations, see advanceFrame in host-timing.ts.
/// `evaluate` returns whether anything still animates.
pub fn advance_frame<Event: TimedEvent>(
    clock: SpringClock,
    frame_time_stamp: f64,
    events: &[Event],
    evaluate: &mut dyn FnMut(&PlannedEvaluation<Event>) -> bool,
    write_back: WriteBack,
    config: &SpringClockConfig,
) -> SpringClock {
    validate_spring_clock_config(config);
    let check_interval_ms = config.step_ms * config.settle_check_steps as f64;
    let mut state = FrameState { clock, step_budget: config.max_steps_per_frame, write_back, config: *config };
    let earliest_ms = clock.last_evaluation_ms.unwrap_or(f64::NEG_INFINITY);
    let frame_time_ms = quantize_time_ms(frame_time_stamp, config).max(earliest_ms);

    let mut timed: Vec<(usize, f64)> = events
        .iter()
        .enumerate()
        .map(|(index, event)| {
            (index, frame_time_ms.min(earliest_ms.max(quantize_time_ms(event.time_stamp(), config))))
        })
        .collect();
    timed.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap().then(left.0.cmp(&right.0)));
    let mut start = 0;
    while start < timed.len() {
        let time_ms = timed[start].1;
        let mut end = start;
        while end < timed.len() && timed[end].1 == time_ms {
            end += 1;
        }
        let here: Vec<Event> = timed[start..end].iter().map(|&(index, _)| events[index].clone()).collect();
        state.step_to(time_ms, EvaluationKind::Events, here, check_interval_ms, evaluate);
        start = end;
    }
    state.step_to(frame_time_ms, EvaluationKind::Frame, Vec::new(), check_interval_ms, evaluate);
    state.clock
}

struct FrameState {
    clock: SpringClock,
    step_budget: u32,
    write_back: WriteBack,
    config: SpringClockConfig,
}

impl FrameState {
    fn run<Event: TimedEvent>(
        &mut self,
        kind: EvaluationKind,
        time_ms: f64,
        steps: u32,
        settle_check: bool,
        events: Vec<Event>,
        evaluate: &mut dyn FnMut(&PlannedEvaluation<Event>) -> bool,
    ) {
        let writes_back = kind != EvaluationKind::Frame || self.write_back == WriteBack::EveryEvaluation;
        let simulated_time_ms = match self.clock.animated_until_ms {
            None => time_ms,
            Some(until) => until + steps as f64 * self.config.step_ms,
        };
        let animating = evaluate(&PlannedEvaluation {
            kind,
            time_ms,
            simulated_time_ms,
            duration_ms: self.clock.last_evaluation_ms.map_or(0.0, |last| time_ms - last),
            steps,
            settle_check,
            snap: kind == EvaluationKind::Snap,
            events,
            writes_back,
        });
        if !writes_back {
            return;
        }
        self.step_budget -= steps;
        if self.clock.animated_until_ms.is_none() && animating {
            self.clock.check_origin_ms = time_ms;
        }
        self.clock.animated_until_ms = animating.then_some(simulated_time_ms);
        self.clock.last_evaluation_ms = animating.then_some(time_ms);
    }

    fn step_to<Event: TimedEvent>(
        &mut self,
        target_ms: f64,
        kind: EvaluationKind,
        events: Vec<Event>,
        check_interval_ms: f64,
        evaluate: &mut dyn FnMut(&PlannedEvaluation<Event>) -> bool,
    ) {
        if let Some(until) = self.clock.animated_until_ms
            && count_spring_steps(target_ms - until, self.step_budget, &self.config).snap
        {
            self.clock.animated_until_ms = None;
            self.run(EvaluationKind::Snap, target_ms, 0, true, Vec::new(), evaluate);
        }
        if self.clock.animated_until_ms.is_some() {
            while let Some(until_ms) = self.clock.animated_until_ms {
                let reachable_ms = until_ms
                    + count_spring_steps(target_ms - until_ms, self.step_budget, &self.config).steps as f64
                        * self.config.step_ms;
                let origin = self.clock.check_origin_ms;
                let next_check_ms = origin + (((until_ms - origin) / check_interval_ms).floor() + 1.0) * check_interval_ms;
                if next_check_ms > reachable_ms {
                    break;
                }
                let time_ms = next_check_ms.max(self.clock.last_evaluation_ms.unwrap_or(next_check_ms));
                let steps = ((next_check_ms - until_ms) / self.config.step_ms) as u32;
                self.run(EvaluationKind::SettleCheck, time_ms, steps, true, Vec::new(), evaluate);
            }
        }
        match self.clock.animated_until_ms {
            None => self.run(kind, target_ms, 0, true, events, evaluate),
            Some(until) => {
                let steps = count_spring_steps(target_ms - until, self.step_budget, &self.config).steps;
                self.run(kind, target_ms, steps, false, events, evaluate);
            }
        }
    }
}
