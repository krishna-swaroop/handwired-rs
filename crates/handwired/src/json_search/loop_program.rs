// A program written as loops over one document's positions, a port of
// src/json-search/loop-program.ts: the work of one position is ordinary
// builder code, compiled once as its own small program, and it runs once per
// position. The runner here is the 'executor' backend; the TypeScript
// version's 'generated' backend writes JavaScript and is left out.

use crate::circuit::{Circuit, Wire};
use crate::compile::{CompileOptions, TensorProgram};

/// Where the value of one source cell of a stage's circuit comes from. A
/// source cell without a binding is a shared cell.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    Item { buffer: usize },
    Cell { buffer: usize, column: usize },
    Register { register: usize },
    Field { field: usize, component: usize },
}

/// Where one output of a stage goes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Write {
    Register { register: usize },
    Field { field: usize, component: usize },
    /// One value per position in an output group of kind 'position'.
    Position { group: usize },
    /// One value per document in an output group of kind 'document'.
    Document { group: usize, column: usize },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Repeat {
    Once,
    Forward,
    Backward,
}

impl Repeat {
    pub fn name(self) -> &'static str {
        match self {
            Repeat::Once => "once",
            Repeat::Forward => "forward",
            Repeat::Backward => "backward",
        }
    }
}

pub struct Stage {
    pub name: String,
    pub repeat: Repeat,
    /// Index into LoopProgram::circuits.
    pub circuit: usize,
    /// Compiled for exactly the wires of `outputs`.
    pub program: TensorProgram,
    pub shared_offset: usize,
    /// In the order the TypeScript Map holds them.
    pub bindings: Vec<(Wire, Binding)>,
    pub outputs: Vec<(Wire, Write)>,
}

impl Stage {
    pub fn binding(&self, wire: Wire) -> Option<Binding> {
        self.bindings.iter().find(|(bound, _)| *bound == wire).map(|&(_, binding)| binding)
    }
}

#[derive(Clone, Debug)]
pub struct Buffer {
    pub name: String,
    pub cells_per_document: usize,
}

#[derive(Clone, Debug)]
pub struct Register {
    pub name: String,
    pub initial: f64,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub width: usize,
    pub bits: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GroupKind {
    Document,
    Position,
}

#[derive(Clone, Debug)]
pub struct OutputGroup {
    pub name: String,
    pub kind: GroupKind,
    pub values_per_document: usize,
    pub fill: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct LoopResult {
    pub group: usize,
    pub column: usize,
    pub register: usize,
}

pub struct LoopProgram {
    pub positions: usize,
    pub buffers: Vec<Buffer>,
    pub registers: Vec<Register>,
    pub fields: Vec<Field>,
    pub output_groups: Vec<OutputGroup>,
    /// The circuits the stages compile from; stages of one loop usually share one.
    pub circuits: Vec<Circuit>,
    pub stages: Vec<Stage>,
    pub results: Vec<LoopResult>,
    pub shared_values: Vec<f32>,
    pub length_cell: Option<(usize, usize)>,
}

/// Compiles one stage. `outputs` may name a wire more than once.
pub fn create_stage(
    circuit_index: usize,
    circuit: &Circuit,
    name: &str,
    repeat: Repeat,
    shared_offset: usize,
    bindings: Vec<(Wire, Binding)>,
    outputs: Vec<(Wire, Write)>,
) -> Stage {
    let mut roots: Vec<Wire> = Vec::new();
    for &(wire, _) in &outputs {
        if !roots.contains(&wire) {
            roots.push(wire);
        }
    }
    let program = circuit.compile_tensor_program(Some(&roots), true, CompileOptions::default());
    for &(wire, _) in &bindings {
        assert!(wire.index() < program.source_size, "a binding names a source cell");
    }
    for &(wire, write) in &outputs {
        assert!(wire.index() >= program.source_size, "{name}: an output is a computed value, not a source cell");
        assert!(!matches!(write, Write::Document { .. }) || repeat == Repeat::Once, "only a stage that runs once writes a document's outputs");
        assert!(!matches!(write, Write::Position { .. }) || repeat != Repeat::Once, "only a looped stage writes per position");
    }
    for &(_, binding) in &bindings {
        assert!(
            repeat != Repeat::Once || !matches!(binding, Binding::Item { .. } | Binding::Field { .. }),
            "a stage that runs once reads no item and no field"
        );
    }
    Stage { name: name.to_string(), repeat, circuit: circuit_index, program, shared_offset, bindings, outputs }
}

/// What describeLoop reports per stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageDescription {
    pub name: String,
    pub repeat: Repeat,
    pub operations: usize,
    pub layers: usize,
    pub index_reads: usize,
}

pub fn describe_loop(program: &LoopProgram) -> Vec<StageDescription> {
    program
        .stages
        .iter()
        .map(|stage| StageDescription {
            name: stage.name.clone(),
            repeat: stage.repeat,
            operations: stage.program.value_count - stage.program.source_size,
            layers: stage.program.layers.len(),
            index_reads: stage.program.layers.iter().map(|layer| layer.index_reads.len()).sum(),
        })
        .collect()
}

/// How many operations one document takes when every looped stage runs every position.
pub fn operations_per_document(program: &LoopProgram) -> usize {
    describe_loop(program)
        .iter()
        .map(|stage| stage.operations * if stage.repeat == Repeat::Once { 1 } else { program.positions })
        .sum()
}

struct StageRunner {
    evaluator: crate::compile::TensorEvaluator,
    /// The source cells no binding writes, which hold shared values and are
    /// checked once after `set_shared` instead of on every step.
    unbound: Vec<usize>,
    unbound_checked: bool,
    items: Vec<(usize, usize)>,
    cells: Vec<(usize, usize, usize)>,
    registers: Vec<(usize, usize)>,
    fields: Vec<(usize, usize, usize)>,
    writes: Vec<(usize, Write)>,
}

/// Runs a loop program on the CPU executor, one document per call.
pub struct LoopRunner<'a> {
    program: &'a LoopProgram,
    registers: Vec<f32>,
    fields: Vec<Vec<f32>>,
    stages: Vec<StageRunner>,
}

impl<'a> LoopRunner<'a> {
    pub fn new(program: &'a LoopProgram) -> LoopRunner<'a> {
        let stages = program
            .stages
            .iter()
            .map(|stage| {
                let circuit = &program.circuits[stage.circuit];
                let evaluator = circuit.create_tensor_program_evaluator(stage.program.clone());
                let roots = &stage.program.roots;
                let writes = stage
                    .outputs
                    .iter()
                    .map(|&(wire, write)| (roots.iter().position(|&root| root as usize == wire.index()).unwrap(), write))
                    .collect();
                let mut runner = StageRunner {
                    evaluator,
                    unbound: Vec::new(),
                    unbound_checked: false,
                    items: Vec::new(),
                    cells: Vec::new(),
                    registers: Vec::new(),
                    fields: Vec::new(),
                    writes,
                };
                for &(wire, binding) in &stage.bindings {
                    let position = wire.index();
                    match binding {
                        Binding::Item { buffer } => runner.items.push((position, buffer)),
                        Binding::Cell { buffer, column } => runner.cells.push((position, buffer, column)),
                        Binding::Register { register } => runner.registers.push((position, register)),
                        Binding::Field { field, component } => runner.fields.push((position, field, component)),
                    }
                }
                let mut bound = vec![false; stage.program.source_size];
                for &(wire, _) in &stage.bindings {
                    bound[wire.index()] = true;
                }
                runner.unbound = (0..bound.len()).filter(|&wire| !bound[wire]).collect();
                runner
            })
            .collect();
        let mut runner = LoopRunner {
            program,
            registers: vec![0.0; program.registers.len()],
            fields: program.fields.iter().map(|field| vec![0.0; program.positions * field.width]).collect(),
            stages,
        };
        runner.set_shared(&program.shared_values);
        runner
    }

    /// The shared values, e.g. after writing a new pattern's cells into them.
    pub fn set_shared(&mut self, shared: &[f32]) {
        assert!(shared.len() == self.program.shared_values.len(), "the shared values have the loop program's length");
        for (runner, stage) in self.stages.iter_mut().zip(&self.program.stages) {
            let size = stage.program.source_size;
            runner.evaluator.set_source_unchecked(0, &shared[stage.shared_offset..stage.shared_offset + size]);
            runner.unbound_checked = false;
        }
    }

    /// Runs document `row` of the buffers. The document's outputs go to row
    /// `document_row` of each 'document' group, and with `window_row`, the per
    /// position values go to that row of each 'position' group.
    pub fn run(&mut self, buffers: &[&[f32]], row: usize, outputs: &mut [Vec<f32>], document_row: usize, window_row: Option<usize>) {
        let program = self.program;
        let positions = program.positions;
        let bases: Vec<usize> = program.buffers.iter().map(|buffer| row * buffer.cells_per_document).collect();
        for (register, value) in program.registers.iter().zip(self.registers.iter_mut()) {
            *value = register.initial as f32;
        }
        let limit = match program.length_cell {
            None => positions,
            Some((buffer, column)) => (positions as f64).min(buffers[buffer][bases[buffer] + column] as f64) as usize,
        };
        for (runner, stage) in self.stages.iter_mut().zip(&program.stages) {
            if !runner.unbound_checked {
                runner.evaluator.check_source(runner.unbound.iter().copied());
                runner.unbound_checked = true;
            }
            for &(position, buffer, column) in &runner.cells {
                runner.evaluator.set_source(position, buffers[buffer][bases[buffer] + column]);
            }
            let count = if stage.repeat == Repeat::Once { 1 } else { limit };
            for step in 0..count {
                let position = if stage.repeat == Repeat::Backward { limit - 1 - step } else { step };
                for &(at, buffer) in &runner.items {
                    runner.evaluator.set_source(at, buffers[buffer][bases[buffer] + position]);
                }
                for &(at, register) in &runner.registers {
                    runner.evaluator.set_source(at, self.registers[register]);
                }
                for &(at, field, component) in &runner.fields {
                    runner.evaluator.set_source(at, self.fields[field][position * program.fields[field].width + component]);
                }
                let computed = runner.evaluator.evaluate_current();
                for &(at, write) in &runner.writes {
                    let value = computed[at];
                    match write {
                        Write::Register { register } => self.registers[register] = value,
                        Write::Field { field, component } => {
                            self.fields[field][position * program.fields[field].width + component] = value;
                        }
                        Write::Position { group } => {
                            if let Some(window_row) = window_row {
                                outputs[group][window_row * positions + position] = value;
                            }
                        }
                        Write::Document { group, column } => {
                            outputs[group][document_row * program.output_groups[group].values_per_document + column] = value;
                        }
                    }
                }
            }
        }
        if let Some(window_row) = window_row {
            for (index, group) in program.output_groups.iter().enumerate() {
                if group.kind == GroupKind::Position {
                    outputs[index][window_row * positions + limit..(window_row + 1) * positions].fill(group.fill as f32);
                }
            }
        }
        for result in &program.results {
            outputs[result.group][document_row * program.output_groups[result.group].values_per_document + result.column] =
                self.registers[result.register];
        }
    }
}
