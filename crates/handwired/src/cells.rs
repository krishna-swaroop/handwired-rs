// Cell declarations for a program's source table, the FP32 array that the host
// writes before an evaluation and that next-state values are written back into.
// A port of src/runtime/cells.ts. Each cell is declared once, with what happens
// to it around an evaluation, a range and an initial value. The table derives
// each cell's position (in declaration order), the circuit's range, grid, exact
// bit and one-hot declarations, the initial table, a validator, and the wire
// that holds each cell's next value.
//
// The TypeScript version declares nested objects of cells in one call and
// names each cell by its path, e.g. 'windows[2].left'. Here builder code
// declares cells one at a time in the same order and passes the same names, so
// positions, names and every error message match.

use crate::circuit::{Circuit, Wire};
use crate::compile::{CompileOptions, TensorEvaluator};
use crate::fp::js_number;
use crate::validate::describe_first_nan;

pub const ANY_NUMBER: (f64, f64) = (f64::NEG_INFINITY, f64::INFINITY);
pub const UNIT_INTERVAL: (f64, f64) = (0.0, 1.0);

/// What happens to a cell around an evaluation, see CellPolicy in cells.ts.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CellPolicy {
    Constant,
    Input,
    Event,
    State,
}

impl CellPolicy {
    pub fn name(self) -> &'static str {
        match self {
            CellPolicy::Constant => "constant",
            CellPolicy::Input => "input",
            CellPolicy::Event => "event",
            CellPolicy::State => "state",
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CellSpec {
    pub policy: CellPolicy,
    pub range: (f64, f64),
    pub initial: f64,
    /// The validator requires exactly 0 or 1.
    pub exact_bit: bool,
    /// Every value is a multiple of this power of two.
    pub grid: Option<f64>,
}

impl CellSpec {
    fn new(policy: CellPolicy, range: (f64, f64), exact_bit: bool) -> CellSpec {
        assert!(
            range.0 <= range.1,
            "a cell's range is inverted: {}..{}",
            js_number(range.0),
            js_number(range.1)
        );
        CellSpec { policy, range, initial: 0.0, exact_bit, grid: None }
    }

    pub fn initial(mut self, initial: f64) -> CellSpec {
        self.initial = initial;
        self
    }

    pub fn grid(mut self, grid: f64) -> CellSpec {
        self.grid = Some(grid);
        self
    }
}

pub fn constant(value: f64, range: (f64, f64)) -> CellSpec {
    CellSpec::new(CellPolicy::Constant, range, false).initial(value)
}

pub fn input(range: (f64, f64)) -> CellSpec {
    CellSpec::new(CellPolicy::Input, range, false)
}

/// An exact bit that the host writes, e.g. the settle check bit.
pub fn input_bit(initial: f64) -> CellSpec {
    CellSpec::new(CellPolicy::Input, UNIT_INTERVAL, true).initial(initial)
}

/// An exact bit that the host sets to 1 when an event arrives. The program resets it to 0.
pub fn event() -> CellSpec {
    CellSpec::new(CellPolicy::Event, UNIT_INTERVAL, true)
}

pub fn state(range: (f64, f64)) -> CellSpec {
    CellSpec::new(CellPolicy::State, range, false)
}

/// An exact bit that the program computes, e.g. whether a window is open.
pub fn state_bit(initial: f64) -> CellSpec {
    CellSpec::new(CellPolicy::State, UNIT_INTERVAL, true).initial(initial)
}

#[derive(Clone, Debug)]
struct DeclaredCell {
    name: String,
    spec: CellSpec,
    next: Option<Wire>,
}

#[derive(Clone, Debug)]
enum RuleKind {
    OneHot,
    AtMostOne,
}

#[derive(Clone, Debug)]
struct Rule {
    name: String,
    rule: RuleKind,
    members: Vec<Wire>,
    allow_all_zero: bool,
    declare_to_circuit: bool,
}

/// createSourceTable: declare every cell, create the circuit, build, pair every
/// state cell with its next value, then finish.
#[derive(Default)]
pub struct SourceTable {
    cells: Vec<DeclaredCell>,
    rules: Vec<Rule>,
    created: bool,
    finished: bool,
}

/// What `finish` returns besides the circuit.
pub struct FinishedTable {
    /// The wire that holds each cell's next value: a constant or input cell
    /// holds itself, and an event holds 0.
    pub next_source: Vec<Wire>,
    pub initial_source: Vec<f32>,
    pub validator: TableValidator,
}

impl SourceTable {
    pub fn new() -> SourceTable {
        SourceTable::default()
    }

    /// Declares one cell after the ones declared so far, e.g. `photos.focus[3]`.
    pub fn cell(&mut self, name: impl Into<String>, spec: CellSpec) -> Wire {
        assert!(!self.created, "declare every cell before createCircuit");
        self.cells.push(DeclaredCell { name: name.into(), spec, next: None });
        Wire::from_position(self.cells.len() - 1)
    }

    /// Exactly one member is 1, or none with `allow_all_zero`. Members are
    /// named `name[index]`.
    pub fn one_hot(&mut self, name: &str, members: &[CellSpec], allow_all_zero: bool, declare_to_circuit: bool) -> Vec<Wire> {
        assert!(members.iter().all(|member| member.exact_bit), "one-hot members must be exact bits, e.g. stateBit()");
        let wires: Vec<Wire> =
            members.iter().enumerate().map(|(index, &member)| self.cell(format!("{name}[{index}]"), member)).collect();
        self.rules.push(Rule {
            name: name.to_string(),
            rule: RuleKind::OneHot,
            members: wires.clone(),
            allow_all_zero,
            declare_to_circuit,
        });
        wires
    }

    /// A stacking order stored as a permutation matrix: cells[rank][item] is 1
    /// when the item is at that rank. `bit` is the spec of every cell before
    /// its initial value, e.g. a state bit on a grid of 1.
    pub fn permutation(&mut self, name: &str, order: &[usize], bit: CellSpec) -> Vec<Vec<Wire>> {
        let mut sorted = order.to_vec();
        sorted.sort_unstable();
        assert!(sorted.iter().enumerate().all(|(index, &item)| item == index), "not a permutation");
        let matrix: Vec<Vec<Wire>> = order
            .iter()
            .enumerate()
            .map(|(rank, &item)| {
                (0..order.len())
                    .map(|column| {
                        self.cell(format!("{name}[{rank}][{column}]"), bit.initial(if column == item { 1.0 } else { 0.0 }))
                    })
                    .collect()
            })
            .collect();
        for (rank, row) in matrix.iter().enumerate() {
            self.rules.push(Rule {
                name: format!("{name} rank {rank}"),
                rule: RuleKind::OneHot,
                members: row.clone(),
                allow_all_zero: false,
                declare_to_circuit: true,
            });
        }
        for item in 0..matrix.len() {
            self.rules.push(Rule {
                name: format!("{name} item {item}"),
                rule: RuleKind::OneHot,
                members: matrix.iter().map(|row| row[item]).collect(),
                allow_all_zero: false,
                declare_to_circuit: true,
            });
        }
        matrix
    }

    /// A rule that only the validator checks: the members add up to at most 1.
    pub fn at_most_one(&mut self, name: &str, members: &[Wire]) {
        assert!(!self.created, "add rules before createCircuit");
        assert!(
            members.iter().all(|member| self.cells.get(member.index()).is_some_and(|cell| cell.spec.exact_bit)),
            "{name}: members must be exact bits"
        );
        self.rules.push(Rule {
            name: name.to_string(),
            rule: RuleKind::AtMostOne,
            members: members.to_vec(),
            allow_all_zero: true,
            declare_to_circuit: false,
        });
    }

    pub fn size(&self) -> usize {
        self.cells.len()
    }

    pub fn name_of(&self, wire: Wire) -> String {
        self.cells.get(wire.index()).map_or_else(|| format!("wire {}", wire.index()), |cell| cell.name.clone())
    }

    pub fn policy_of(&self, wire: Wire) -> Option<CellPolicy> {
        self.cells.get(wire.index()).map(|cell| cell.spec.policy)
    }

    pub fn create_circuit(&mut self) -> Circuit {
        assert!(!self.created, "createCircuit may only run once");
        let mut circuit = Circuit::new(self.cells.len());
        for (index, cell) in self.cells.iter().enumerate() {
            let wire = circuit.source(index);
            circuit.assume_range(wire, cell.spec.range.0, cell.spec.range.1);
            if let Some(grid) = cell.spec.grid {
                circuit.assume_grid(wire, grid);
            }
            if cell.spec.exact_bit {
                circuit.assume_bit(wire);
            }
        }
        for rule in &self.rules {
            if rule.declare_to_circuit {
                circuit.assume_one_hot(&rule.members, rule.allow_all_zero, Default::default());
            }
        }
        self.created = true;
        circuit
    }

    /// Pairs every state cell in `stored` with the wire at the same index in
    /// `next`. Every pair is checked before any is recorded.
    pub fn next(&mut self, circuit: &Circuit, stored: &[Wire], next: &[Wire]) {
        assert!(self.created, "call createCircuit before table.next");
        assert!(!self.finished, "table.next after finish has no effect");
        assert!(stored.len() == next.len(), "next state must be a list of {}", stored.len());
        let width = circuit.width();
        let mut paired = std::collections::HashSet::new();
        for (&wire, &next_wire) in stored.iter().zip(next) {
            let declared = self.cells.get(wire.index()).unwrap_or_else(|| panic!("wire {} is not a source cell", wire.index()));
            assert!(declared.spec.policy == CellPolicy::State, "{} is not a state cell", declared.name);
            assert!(declared.next.is_none() && !paired.contains(&wire), "{} already has a next value", declared.name);
            assert!(next_wire.index() < width, "the next value of {} is not a wire of the circuit", declared.name);
            paired.insert(wire);
        }
        for (&wire, &next_wire) in stored.iter().zip(next) {
            self.cells[wire.index()].next = Some(next_wire);
        }
    }

    pub fn finish(&mut self, circuit: &mut Circuit) -> FinishedTable {
        assert!(self.created, "call createCircuit before finish");
        assert!(!self.finished, "finish may only run once");
        let missing: Vec<&str> = self
            .cells
            .iter()
            .filter(|cell| cell.spec.policy == CellPolicy::State && cell.next.is_none())
            .map(|cell| cell.name.as_str())
            .collect();
        assert!(missing.is_empty(), "{} never received a next value", missing.join(", "));
        self.finished = true;
        let zero = self.cells.iter().any(|cell| cell.spec.policy == CellPolicy::Event).then(|| circuit.constant(0.0));
        let next_source = self
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                if cell.spec.policy == CellPolicy::Event { zero.unwrap() } else { cell.next.unwrap_or(circuit.source(index)) }
            })
            .collect();
        let initial_source: Vec<f32> = self.cells.iter().map(|cell| cell.spec.initial as f32).collect();
        let validator = TableValidator {
            names: self.cells.iter().map(|cell| cell.name.clone()).collect(),
            specs: self.cells.iter().map(|cell| cell.spec).collect(),
            rules: self.rules.clone(),
        };
        if let Err(message) = validator.check(&initial_source) {
            panic!("{message}");
        }
        FinishedTable { next_source, initial_source, validator }
    }
}

/// Checks a table against every declaration: finite and within range, exact
/// bits, grids, then one-hot and at-most-one rules, with cells.ts's messages.
#[derive(Clone, Debug)]
pub struct TableValidator {
    names: Vec<String>,
    specs: Vec<CellSpec>,
    rules: Vec<Rule>,
}

/// An error about one cell's value, with the cell's position, or about a rule.
#[derive(Clone, Debug, PartialEq)]
pub struct TableError {
    pub message: String,
    pub position: Option<usize>,
}

impl TableValidator {
    pub fn check(&self, source: &[f32]) -> Result<(), String> {
        self.check_cells(source).map_err(|error| error.message)
    }

    pub fn check_cells(&self, source: &[f32]) -> Result<(), TableError> {
        let cell_error = |message: String, position: usize| Err(TableError { message, position: Some(position) });
        if source.len() != self.specs.len() {
            return Err(TableError {
                message: format!(
                    "the table holds {} numbers, but {} cells are declared",
                    source.len(),
                    self.specs.len()
                ),
                position: None,
            });
        }
        for (wire, spec) in self.specs.iter().enumerate() {
            let value = source[wire] as f64;
            if !(value >= spec.range.0 && value <= spec.range.1 && value.is_finite()) {
                return cell_error(
                    format!(
                        "{} must be finite and within {}..{}, got {}",
                        self.names[wire],
                        js_number(spec.range.0),
                        js_number(spec.range.1),
                        js_number(value)
                    ),
                    wire,
                );
            }
        }
        for (wire, spec) in self.specs.iter().enumerate() {
            if spec.exact_bit && source[wire] != 0.0 && source[wire] != 1.0 {
                return cell_error(
                    format!("{} must be exactly 0 or 1, got {}", self.names[wire], js_number(source[wire] as f64)),
                    wire,
                );
            }
        }
        for (wire, spec) in self.specs.iter().enumerate() {
            if let Some(grid) = spec.grid {
                let quotient = source[wire] as f64 / grid;
                if quotient.fract() != 0.0 || !quotient.is_finite() {
                    return cell_error(
                        format!(
                            "{} must be a multiple of {}, got {}",
                            self.names[wire],
                            js_number(grid),
                            js_number(source[wire] as f64)
                        ),
                        wire,
                    );
                }
            }
        }
        for rule in &self.rules {
            let mut mass = 0.0f64;
            for member in &rule.members {
                mass += source[member.index()] as f64;
            }
            let valid = match rule.rule {
                RuleKind::AtMostOne => mass <= 1.0,
                RuleKind::OneHot => mass == 1.0 || (rule.allow_all_zero && mass == 0.0),
            };
            if !valid {
                let rule_name = match rule.rule {
                    RuleKind::OneHot => "oneHot",
                    RuleKind::AtMostOne => "atMostOne",
                };
                return Err(TableError {
                    message: format!("{} breaks its {rule_name} rule with mass {}", rule.name, js_number(mass)),
                    position: None,
                });
            }
        }
        Ok(())
    }
}

/// createEvaluator: compiles the program for the next-state values and the
/// outputs, and checks the table, the next-state values and that every output
/// is finite around each evaluation. `output_names` names each output, e.g.
/// 'thumb.offset'.
pub struct TableEvaluator {
    pub circuit: Circuit,
    pub next_source: Vec<Wire>,
    pub validator: TableValidator,
    pub outputs: Vec<(String, Wire)>,
    roots: Vec<Wire>,
    evaluator: TensorEvaluator,
}

/// What one evaluation returns: the next-state values, then the outputs in order.
pub struct TableEvaluation {
    pub next_values: Vec<f32>,
    pub outputs: Vec<f32>,
}

impl TableEvaluator {
    pub fn new(circuit: Circuit, finished: FinishedTable, outputs: Vec<(String, Wire)>) -> TableEvaluator {
        let roots: Vec<Wire> =
            finished.next_source.iter().copied().chain(outputs.iter().map(|&(_, wire)| wire)).collect();
        let program = circuit.compile_tensor_program(Some(&roots), true, CompileOptions::default());
        let evaluator = circuit.create_tensor_program_evaluator(program);
        TableEvaluator {
            circuit,
            next_source: finished.next_source,
            validator: finished.validator,
            outputs,
            roots,
            evaluator,
        }
    }

    // Adds the operation that produced the first NaN to an error about roots[index].
    fn with_first_nan(&self, message: String, index: usize, source: &[f32], value: f32) -> String {
        if !value.is_nan() {
            return message;
        }
        let values = self.circuit.evaluate_tensor(source, Some(&self.evaluator.program));
        match describe_first_nan(&self.circuit, &values, Some(&[self.roots[index]])) {
            Some(first) => format!("{message}. {first}"),
            None => message,
        }
    }

    pub fn evaluate(&mut self, source: &[f32]) -> Result<TableEvaluation, String> {
        self.validator.check(source)?;
        let values = self.evaluator.evaluate(source).to_vec();
        let count = self.next_source.len();
        if let Err(error) = self.validator.check_cells(&values[..count]) {
            return Err(match error.position {
                Some(position) => self.with_first_nan(error.message, position, source, values[position]),
                None => error.message,
            });
        }
        for (index, (path, _)) in self.outputs.iter().enumerate() {
            let value = values[count + index];
            if !value.is_finite() {
                let message = format!("output {path} must be finite, got {}", js_number(value as f64));
                return Err(self.with_first_nan(message, count + index, source, value));
            }
        }
        Ok(TableEvaluation { next_values: values[..count].to_vec(), outputs: values[count..].to_vec() })
    }
}
