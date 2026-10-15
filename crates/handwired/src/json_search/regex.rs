// The pattern language and the pattern as data, a port of
// src/json-search/regex/pattern.ts, bits.ts and data-program.ts.
//
// pattern.ts: a subset of JavaScript's RegExp syntax with the u flag, compiled
// into a position automaton (Glushkov's construction) and character classes.
// bits.ts: bits in builder code that fold bits builder code already knows.
// data-program.ts: the automaton's cells, and the two operations that read them.

use std::collections::{BTreeSet, HashMap};

use crate::cells::{constant, input, input_bit, SourceTable};
use crate::circuit::{Circuit, OutOfRange, Term, Wire};
use crate::emit::{emit_compare, CompareOptions, Relation};

pub const MAX_CODE_POINT: u32 = 0x10ffff;
pub const CODE_POINT_LIMIT: u32 = 0x110000;

/// Inclusive ranges of code points, sorted, with a gap between neighbors.
pub type CodePointSet = Vec<(u32, u32)>;

fn dot_set() -> CodePointSet {
    vec![(0, 9), (11, 12), (14, 0x2027), (0x202a, MAX_CODE_POINT)]
}

fn digit_set() -> CodePointSet {
    vec![(48, 57)]
}

fn word_set() -> CodePointSet {
    vec![(48, 57), (65, 90), (95, 95), (97, 122)]
}

fn normalize_set(intervals: &[(u32, u32)]) -> CodePointSet {
    let mut sorted = intervals.to_vec();
    sorted.sort_by_key(|interval| interval.0);
    let mut merged: CodePointSet = Vec::new();
    for (low, high) in sorted {
        match merged.last_mut() {
            Some(previous) if low <= previous.1 + 1 => previous.1 = previous.1.max(high),
            _ => merged.push((low, high)),
        }
    }
    merged
}

fn complement_set(set: &[(u32, u32)]) -> CodePointSet {
    let mut gaps = Vec::new();
    let mut next = 0;
    for &(low, high) in set {
        if low > next {
            gaps.push((next, low - 1));
        }
        next = high + 1;
    }
    if next <= MAX_CODE_POINT {
        gaps.push((next, MAX_CODE_POINT));
    }
    gaps
}

pub fn set_contains(set: &[(u32, u32)], code_point: u32) -> bool {
    set.iter().any(|&(low, high)| code_point >= low && code_point <= high)
}

#[derive(Clone, Debug)]
pub enum PatternNode {
    Atom(CodePointSet),
    Sequence(Vec<PatternNode>),
    Alternation(Vec<PatternNode>),
    Repeat { body: Box<PatternNode>, optional: bool, repeats: bool },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatternSyntaxError {
    pub message: String,
    /// The index of the code point where the error was found.
    pub position: usize,
}

struct Parser {
    code_points: Vec<u32>,
    position: usize,
}

type Parsed<T> = Result<T, PatternSyntaxError>;

impl Parser {
    fn is(&self, character: char) -> bool {
        self.code_points.get(self.position) == Some(&(character as u32))
    }

    fn fail<T>(&self, message: impl Into<String>, at: usize) -> Parsed<T> {
        Err(PatternSyntaxError { message: message.into(), position: at })
    }

    fn alternation(&mut self) -> Parsed<PatternNode> {
        let mut options = vec![self.sequence()?];
        while self.is('|') {
            self.position += 1;
            options.push(self.sequence()?);
        }
        Ok(if options.len() == 1 { options.pop().unwrap() } else { PatternNode::Alternation(options) })
    }

    fn sequence(&mut self) -> Parsed<PatternNode> {
        let mut parts = Vec::new();
        while self.position < self.code_points.len() && !self.is('|') && !self.is(')') {
            parts.push(self.repeat()?);
        }
        Ok(if parts.len() == 1 { parts.pop().unwrap() } else { PatternNode::Sequence(parts) })
    }

    fn repeat(&mut self) -> Parsed<PatternNode> {
        let body = self.atom()?;
        if !(self.is('?') || self.is('*') || self.is('+')) {
            return Ok(body);
        }
        let quantifier = char::from_u32(self.code_points[self.position]).unwrap();
        self.position += 1;
        if self.is('?') {
            return self.fail(format!("{quantifier}? is a lazy quantifier, which this language doesn’t have"), self.position);
        }
        if self.is('*') || self.is('+') {
            return self.fail("nothing to repeat", self.position);
        }
        Ok(PatternNode::Repeat { body: Box::new(body), optional: quantifier != '+', repeats: quantifier != '?' })
    }

    fn atom(&mut self) -> Parsed<PatternNode> {
        let at = self.position;
        let code_point = self.code_points[self.position];
        let character = char::from_u32(code_point).unwrap();
        self.position += 1;
        match character {
            '(' => {
                if self.is('?') {
                    return self.fail("groups such as (?: aren’t supported", at);
                }
                let inner = self.alternation()?;
                if !self.is(')') {
                    return self.fail("this group is never closed", at);
                }
                self.position += 1;
                Ok(inner)
            }
            '[' => Ok(PatternNode::Atom(self.class(at)?)),
            '.' => Ok(PatternNode::Atom(dot_set())),
            '\\' => Ok(PatternNode::Atom(self.escape(at, false)?.0)),
            '?' | '*' | '+' => self.fail("nothing to repeat", at),
            '^' | '$' => self.fail("the anchors ^ and $ aren’t supported", at),
            '{' | '}' | ']' => self.fail(format!("{character} needs a backslash before it"), at),
            _ => Ok(PatternNode::Atom(vec![(code_point, code_point)])),
        }
    }

    fn escape(&mut self, at: usize, in_class: bool) -> Parsed<(CodePointSet, Option<u32>)> {
        let Some(&escaped) = self.code_points.get(self.position) else {
            return self.fail("a pattern can’t end with a backslash", at);
        };
        self.position += 1;
        if escaped == 'd' as u32 {
            return Ok((digit_set(), None));
        }
        if escaped == 'w' as u32 {
            return Ok((word_set(), None));
        }
        let syntax = "^$\\.*+?()[]{}|".chars().any(|character| character as u32 == escaped);
        if syntax || escaped == '/' as u32 || (in_class && escaped == '-' as u32) {
            return Ok((vec![(escaped, escaped)], Some(escaped)));
        }
        self.fail(format!("the escape \\{} isn’t supported", char::from_u32(escaped).unwrap()), at)
    }

    fn class_atom(&mut self) -> Parsed<(CodePointSet, Option<u32>)> {
        let at = self.position;
        let code_point = self.code_points[self.position];
        self.position += 1;
        if code_point == '\\' as u32 { self.escape(at, true) } else { Ok((vec![(code_point, code_point)], Some(code_point))) }
    }

    fn class(&mut self, at: usize) -> Parsed<CodePointSet> {
        let negated = self.is('^');
        if negated {
            self.position += 1;
        }
        let mut intervals = Vec::new();
        while !self.is(']') {
            if self.position >= self.code_points.len() {
                return self.fail("this set of characters is never closed", at);
            }
            let low = self.class_atom()?;
            let next = self.code_points.get(self.position + 1).copied();
            if self.is('-') && next.is_some() && next != Some(']' as u32) {
                let dash = self.position;
                self.position += 1;
                let high = self.class_atom()?;
                let (Some(low), Some(high)) = (low.1, high.1) else {
                    return self.fail("a range needs one character at each end", dash);
                };
                if low > high {
                    return self.fail("this range is out of order", dash);
                }
                intervals.push((low, high));
            } else {
                intervals.extend(low.0);
            }
        }
        self.position += 1;
        let set = normalize_set(&intervals);
        Ok(if negated { complement_set(&set) } else { set })
    }
}

pub fn parse_pattern(pattern: &str) -> Result<PatternNode, PatternSyntaxError> {
    let mut parser = Parser { code_points: pattern.chars().map(|character| character as u32).collect(), position: 0 };
    let node = parser.alternation()?;
    if parser.position < parser.code_points.len() {
        return parser.fail("this ) closes no group", parser.position);
    }
    Ok(node)
}

/// A position automaton: state 0 is the initial state, and state p + 1 is the
/// pattern's p-th atom.
#[derive(Clone, Debug, PartialEq)]
pub struct Automaton {
    pub state_count: usize,
    pub targets: Vec<Vec<usize>>,
    pub state_sets: Vec<CodePointSet>,
    pub accepting: Vec<bool>,
}

struct Summary {
    nullable: bool,
    first: BTreeSet<usize>,
    last: BTreeSet<usize>,
}

fn build_automaton(node: &PatternNode) -> Automaton {
    let mut sets: Vec<CodePointSet> = Vec::new();
    let mut follow: Vec<BTreeSet<usize>> = Vec::new();
    fn summarize(current: &PatternNode, sets: &mut Vec<CodePointSet>, follow: &mut Vec<BTreeSet<usize>>) -> Summary {
        match current {
            PatternNode::Atom(set) => {
                let atom = sets.len();
                sets.push(set.clone());
                follow.push(BTreeSet::new());
                Summary { nullable: false, first: BTreeSet::from([atom]), last: BTreeSet::from([atom]) }
            }
            PatternNode::Sequence(parts) => {
                let mut summary = Summary { nullable: true, first: BTreeSet::new(), last: BTreeSet::new() };
                for part in parts {
                    let next = summarize(part, sets, follow);
                    for &atom in &summary.last {
                        follow[atom].extend(next.first.iter().copied());
                    }
                    summary = Summary {
                        nullable: summary.nullable && next.nullable,
                        first: if summary.nullable { summary.first.union(&next.first).copied().collect() } else { summary.first },
                        last: if next.nullable { summary.last.union(&next.last).copied().collect() } else { next.last },
                    };
                }
                summary
            }
            PatternNode::Alternation(options) => {
                let options: Vec<Summary> = options.iter().map(|option| summarize(option, sets, follow)).collect();
                Summary {
                    nullable: options.iter().any(|option| option.nullable),
                    first: options.iter().flat_map(|option| option.first.iter().copied()).collect(),
                    last: options.iter().flat_map(|option| option.last.iter().copied()).collect(),
                }
            }
            PatternNode::Repeat { body, optional, repeats } => {
                let body = summarize(body, sets, follow);
                if *repeats {
                    for &atom in &body.last {
                        follow[atom].extend(body.first.iter().copied());
                    }
                }
                Summary { nullable: body.nullable || *optional, first: body.first, last: body.last }
            }
        }
    }
    let summary = summarize(node, &mut sets, &mut follow);
    let sorted = |atoms: &BTreeSet<usize>| atoms.iter().map(|atom| atom + 1).collect::<Vec<_>>();
    let mut targets = vec![sorted(&summary.first)];
    targets.extend(follow.iter().map(sorted));
    let mut state_sets = vec![Vec::new()];
    state_sets.extend(sets.iter().cloned());
    let mut accepting = vec![summary.nullable];
    accepting.extend((0..sets.len()).map(|atom| summary.last.contains(&atom)));
    Automaton { state_count: sets.len() + 1, targets, state_sets, accepting }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClassPartition {
    pub class_count: usize,
    pub boundaries: Vec<u32>,
    /// The class of each region, or -1 for a region that no step reads.
    pub region_classes: Vec<i64>,
    pub class_reads: Vec<Vec<bool>>,
}

fn partition_classes(automaton: &Automaton) -> ClassPartition {
    let mut cuts = BTreeSet::new();
    for set in &automaton.state_sets {
        for &(low, high) in set {
            cuts.insert(low);
            cuts.insert(high + 1);
        }
    }
    let mut region_starts = vec![0];
    region_starts.extend(cuts.into_iter().filter(|&cut| cut > 0 && cut < CODE_POINT_LIMIT));
    let mut class_by_signature: HashMap<Vec<bool>, usize> = HashMap::new();
    let mut class_reads: Vec<Vec<bool>> = Vec::new();
    let mut boundaries = Vec::new();
    let mut region_classes: Vec<i64> = Vec::new();
    for start in region_starts {
        let reads: Vec<bool> = automaton.state_sets.iter().map(|set| set_contains(set, start)).collect();
        let mut class_index = -1i64;
        if reads.iter().any(|&read| read) {
            let next = class_reads.len();
            let index = *class_by_signature.entry(reads.clone()).or_insert(next);
            if index == next {
                class_reads.push(reads);
            }
            class_index = index as i64;
        }
        assert!(
            start == 0 || region_classes.last() != Some(&class_index),
            "the regions on both sides of {start} have the same class"
        );
        if start > 0 {
            boundaries.push(start);
        }
        region_classes.push(class_index);
    }
    ClassPartition { class_count: class_reads.len(), boundaries, region_classes, class_reads }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledPattern {
    pub pattern: String,
    pub automaton: Automaton,
    pub partition: ClassPartition,
}

pub fn compile_pattern(pattern: &str) -> Result<CompiledPattern, PatternSyntaxError> {
    let automaton = build_automaton(&parse_pattern(pattern)?);
    let partition = partition_classes(&automaton);
    Ok(CompiledPattern { pattern: pattern.to_string(), automaton, partition })
}

/// What the data program holds: S states, C classes and K class boundaries.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SearchCapacity {
    pub states: usize,
    pub classes: usize,
    pub boundaries: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityError {
    pub reason: &'static str,
    pub needed: usize,
    pub capacity: usize,
    pub message: String,
}

pub fn capacity_errors(compiled: &CompiledPattern, capacity: SearchCapacity) -> Vec<CapacityError> {
    let needs = [
        ("states", compiled.automaton.state_count, capacity.states, "states"),
        ("classes", compiled.partition.class_count, capacity.classes, "character classes"),
        ("boundaries", compiled.partition.boundaries.len(), capacity.boundaries, "class boundaries"),
    ];
    needs
        .into_iter()
        .filter(|need| need.1 > need.2)
        .map(|(reason, needed, held, noun)| CapacityError {
            reason,
            needed,
            capacity: held,
            message: format!("the pattern needs {needed} {noun}, and the program holds {held}"),
        })
        .collect()
}

/// A bit in builder code: known while building, or a wire that holds 0 or 1.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Bit {
    Known(bool),
    Wire(Wire),
}

pub fn not(circuit: &mut Circuit, bit: Bit) -> Bit {
    match bit {
        Bit::Known(value) => Bit::Known(!value),
        Bit::Wire(wire) => Bit::Wire(circuit.affine(&[(wire, -1.0)], 1.0)),
    }
}

/// a && !b, as relu(a - b).
pub fn and_not(circuit: &mut Circuit, a: Bit, b: Bit) -> Bit {
    match (a, b) {
        (Bit::Known(false), _) | (_, Bit::Known(true)) => Bit::Known(false),
        (_, Bit::Known(false)) => a,
        (Bit::Known(true), _) => not(circuit, b),
        (Bit::Wire(a), Bit::Wire(b)) => Bit::Wire(circuit.reglu(&[], 1.0, &[(a, 1.0), (b, -1.0)], 0.0)),
    }
}

/// !bits.some(bit => bit), as relu(1 - sum of bits).
pub fn none_of(circuit: &mut Circuit, bits: &[Bit]) -> Bit {
    if bits.contains(&Bit::Known(true)) {
        return Bit::Known(false);
    }
    let wires: Vec<Wire> = bits.iter().filter_map(|bit| if let Bit::Wire(wire) = bit { Some(*wire) } else { None }).collect();
    if wires.is_empty() {
        return Bit::Known(true);
    }
    if wires.len() == 1 {
        return not(circuit, Bit::Wire(wires[0]));
    }
    let terms: Vec<Term> = wires.iter().map(|&wire| (wire, -1.0)).collect();
    Bit::Wire(circuit.reglu(&[], 1.0, &terms, 1.0))
}

/// a || b for two bits that are never both 1, as a + b.
pub fn or_exclusive(circuit: &mut Circuit, a: Bit, b: Bit) -> Bit {
    match (a, b) {
        (Bit::Known(false), _) => b,
        (_, Bit::Known(false)) | (Bit::Known(true), _) => a,
        (_, Bit::Known(true)) => b,
        (Bit::Wire(a), Bit::Wire(b)) => Bit::Wire(circuit.affine(&[(a, 1.0), (b, 1.0)], 0.0)),
    }
}

/// A wire that holds the bit, for an output.
pub fn bit_wire(circuit: &mut Circuit, bit: Bit) -> Wire {
    match bit {
        Bit::Known(value) => circuit.constant(if value { 1.0 } else { 0.0 }),
        Bit::Wire(wire) => wire,
    }
}

/// The cells of dataSearchCells.
#[derive(Clone, Debug)]
pub struct DataSearchCells {
    pub text: Vec<Wire>,
    pub boundaries: Vec<Wire>,
    pub region_classes: Vec<Wire>,
    /// transitions[class][from][to].
    pub transitions: Vec<Vec<Vec<Wire>>>,
    pub accepting: Vec<Wire>,
}

/// Declares dataSearchCells for a text of `text_length` code points under `prefix`.
pub fn declare_data_search_cells(table: &mut SourceTable, prefix: &str, capacity: SearchCapacity, text_length: usize) -> DataSearchCells {
    let SearchCapacity { states, classes, boundaries } = capacity;
    let limit = CODE_POINT_LIMIT as f64;
    let text = (0..text_length)
        .map(|index| table.cell(format!("{prefix}.text[{index}]"), input((-1.0, MAX_CODE_POINT as f64)).initial(-1.0).grid(1.0)))
        .collect();
    let boundary_cells = (0..boundaries)
        .map(|index| table.cell(format!("{prefix}.boundaries[{index}]"), input((1.0, limit)).initial(limit).grid(1.0)))
        .collect();
    let class_count = classes as f64;
    let region_classes = (0..boundaries + 2)
        .map(|index| {
            let spec = if index == 0 {
                constant(class_count, (class_count, class_count)).grid(1.0)
            } else {
                input((0.0, class_count)).initial(class_count).grid(1.0)
            };
            table.cell(format!("{prefix}.regionClasses[{index}]"), spec)
        })
        .collect();
    let transitions = (0..classes)
        .map(|class| {
            (0..states)
                .map(|from| {
                    (0..states)
                        .map(|to| table.cell(format!("{prefix}.transitions[{class}][{from}][{to}]"), input_bit(0.0)))
                        .collect()
                })
                .collect()
        })
        .collect();
    let accepting = (0..states).map(|state| table.cell(format!("{prefix}.accepting[{state}]"), input_bit(0.0))).collect();
    DataSearchCells { text, boundaries: boundary_cells, region_classes, transitions, accepting }
}

/// The class of each code point.
pub fn emit_classes(circuit: &mut Circuit, cells: &DataSearchCells) -> Vec<Wire> {
    let zero = circuit.constant(0.0);
    cells
        .text
        .iter()
        .map(|&code_point| {
            let mut terms = vec![(emit_compare(circuit, code_point, Relation::AtLeast, zero, CompareOptions::default()), 1.0)];
            for &boundary in &cells.boundaries {
                terms.push((emit_compare(circuit, code_point, Relation::AtLeast, boundary, CompareOptions::default()), 1.0));
            }
            let region = circuit.affine(&terms, 0.0);
            circuit.index_read(region, &cells.region_classes, OutOfRange::Clamp)
        })
        .collect()
}

/// transitions[codePointClass][from][to] && bit, as one index read.
pub struct StepAnd {
    step_tables: Vec<Vec<Vec<Wire>>>,
    classes: usize,
}

impl StepAnd {
    pub fn new(cells: &DataSearchCells) -> StepAnd {
        let states = cells.accepting.len();
        let step_tables = (0..states)
            .map(|from| (0..states).map(|to| cells.transitions.iter().map(|by_class| by_class[from][to]).collect()).collect())
            .collect();
        StepAnd { step_tables, classes: cells.transitions.len() }
    }

    pub fn apply(&self, circuit: &mut Circuit, code_point_class: Wire, from: usize, to: usize, bit: Bit) -> Bit {
        match bit {
            Bit::Known(false) => Bit::Known(false),
            Bit::Known(true) => Bit::Wire(circuit.index_read_affine(
                &[(code_point_class, 1.0)],
                0.0,
                &self.step_tables[from][to],
                OutOfRange::Clamp,
                OutOfRange::Zero,
                None,
            )),
            Bit::Wire(bit) => Bit::Wire(circuit.index_read_affine(
                &[(code_point_class, 1.0), (bit, -(self.classes as f64))],
                self.classes as f64,
                &self.step_tables[from][to],
                OutOfRange::Clamp,
                OutOfRange::Zero,
                None,
            )),
        }
    }
}

/// Writes a compiled pattern into the cells of dataSearchCells.
pub fn write_pattern(values: &mut [f32], cells: &DataSearchCells, capacity: SearchCapacity, compiled: &CompiledPattern) -> Result<(), String> {
    let errors = capacity_errors(compiled, capacity);
    if !errors.is_empty() {
        return Err(errors.iter().map(|error| error.message.clone()).collect::<Vec<_>>().join("; "));
    }
    let partition = &compiled.partition;
    for (index, wire) in cells.boundaries.iter().enumerate() {
        values[wire.index()] = partition.boundaries.get(index).copied().unwrap_or(CODE_POINT_LIMIT) as f32;
    }
    for (index, wire) in cells.region_classes.iter().enumerate().skip(1) {
        let class_index = partition.region_classes.get(index - 1).copied().unwrap_or(-1);
        values[wire.index()] = if class_index == -1 { capacity.classes as f32 } else { class_index as f32 };
    }
    for (class, by_class) in cells.transitions.iter().enumerate() {
        for (from, by_from) in by_class.iter().enumerate() {
            for (to, wire) in by_from.iter().enumerate() {
                let reads = partition.class_reads.get(class).and_then(|reads| reads.get(to)).copied() == Some(true);
                let targets = compiled.automaton.targets.get(from).is_some_and(|targets| targets.contains(&to));
                values[wire.index()] = if reads && targets { 1.0 } else { 0.0 };
            }
        }
    }
    for (state, wire) in cells.accepting.iter().enumerate() {
        values[wire.index()] = if compiled.automaton.accepting.get(state) == Some(&true) { 1.0 } else { 0.0 };
    }
    Ok(())
}
