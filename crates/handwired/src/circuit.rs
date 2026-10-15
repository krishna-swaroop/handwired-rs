// Three runtime atoms and one index read instruction, plus the macros that
// lower into them. A port of the Circuit class in
// src/runtime/numeric-ui-core.ts: the same operations, in the same order, with
// the same terms, so both runtimes evaluate a circuit to the same bits.

use std::collections::HashMap;

use crate::fp::{
    fround, is_power_of_two_grid, key_bits, lowest_power_of_two, relu_js, EXACT_GRID_STEPS, LARGEST_GRID,
    SMALLEST_GRID,
};

/// A position in the array an evaluation fills: source cells first, then one
/// position per operation. It has a type of its own, so a plain number doesn't
/// pass where a wire is expected.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Wire(u32);

impl Wire {
    /// A position counted by hand as a wire, e.g. a source cell's position
    /// before the circuit exists.
    pub fn from_position(position: usize) -> Wire {
        Wire(u32::try_from(position).expect("a wire position must fit in u32"))
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One `coefficient × wire` term of an affine form.
pub type Term = (Wire, f64);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AtomKind {
    Affine,
    Reglu,
    Boolean,
    /// The index read instruction, built only by `index_read_affine`.
    Index,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Macro {
    Multiply { left: Wire, right: Wire },
    Select { gate: Wire, off: Wire, on: Wire },
}

/// What an index read returns when floor(index) is outside the table: `Clamp`
/// reads the nearest end, `Zero` reads +0.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum OutOfRange {
    #[default]
    Clamp,
    Zero,
}

/// Why an operation's FP32 evaluation, term order or operand wires carry
/// meaning, so a backend that reorders or fuses sums knows what to keep. See
/// RoundingPointKind in the TypeScript runtime for each kind's contract.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RoundingPointKind {
    Boolean,
    IntegerBit,
    SignCascade,
    Marked,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Range {
    pub minimum: f64,
    pub maximum: f64,
}

impl Range {
    pub const UNKNOWN: Range = Range { minimum: f64::NEG_INFINITY, maximum: f64::INFINITY };

    pub fn new(minimum: f64, maximum: f64) -> Range {
        Range { minimum, maximum }
    }
}

/// One distinct candidates array, registered once per circuit. Reads of an
/// array with the same content share it.
#[derive(Clone, Debug)]
pub struct CandidateTable {
    pub id: usize,
    pub candidates: Vec<Wire>,
    pub maximum_depth: u32,
    range_by_reach: HashMap<(i64, i64, OutOfRange, OutOfRange), Range>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IndexRead {
    /// Position in `Circuit::candidate_tables`.
    pub table: usize,
    pub below: OutOfRange,
    pub above: OutOfRange,
    /// The index range the builder accepted, see `index_read_affine`.
    pub index_range: Range,
}

#[derive(Clone, Debug)]
pub struct Operation {
    pub kind: AtomKind,
    pub left_terms: Vec<Term>,
    pub left_bias: f64,
    /// For an index read, the gate side holds the index form and the left side
    /// is empty.
    pub gate_terms: Vec<Term>,
    pub gate_bias: f64,
    pub depth: u32,
    pub preserve_rounding: bool,
    pub macro_: Option<Macro>,
    pub index_read: Option<IndexRead>,
    pub rounding_point: Option<RoundingPointKind>,
    /// Set on the bit `emit_compare` returns.
    pub comparison_bit: bool,
}

/// A declared invariant: every member is exactly 0 or 1 and their sum is
/// exactly 1, or 0 too with `allow_all_zero`.
#[derive(Clone, Debug)]
pub struct OneHotGroup {
    pub members: Vec<Wire>,
    pub allow_all_zero: bool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct DeclarationOptions {
    /// Declare a range or grid even though operations already read the wire.
    /// Those operations keep what they were built with.
    pub allow_existing_readers: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum BooleanGradient {
    #[default]
    Reject,
    StraightThrough,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum IndexGradient {
    #[default]
    Reject,
    Zero,
    BackwardDifference,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BackpropagateOptions {
    pub boolean_gradient: BooleanGradient,
    pub index_gradient: IndexGradient,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackpropagateError {
    BooleanAtom,
    IndexOfIndexRead,
}

impl std::fmt::Display for BackpropagateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackpropagateError::BooleanAtom => write!(formatter, "cannot backpropagate through a boolean atom"),
            BackpropagateError::IndexOfIndexRead => {
                write!(formatter, "cannot backpropagate into the index of an index read; pass index_gradient")
            }
        }
    }
}

impl std::error::Error for BackpropagateError {}

// Every integer within ±2^24 is an FP32 value, so an index form whose range lies
// strictly inside these bounds can hold each position exactly.
const INDEX_MAGNITUDE_LIMIT: f64 = 16_777_216.0;

// Keep wide sums as explicit wires so every consumer sees the same FP32 rounding
// point. Small affine forms still compose into their consumers.
const AFFINE_SHARE_THRESHOLD: usize = 4;

// The deduplication key of an operation, the structured counterpart of the
// TypeScript runtime's expression strings.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExpressionKey {
    kind: AtomKind,
    left_bias: u64,
    left_terms: Vec<(Wire, u64)>,
    gate_bias: u64,
    gate_terms: Vec<(Wire, u64)>,
    preserve_rounding: bool,
    macro_: Option<Macro>,
    index: Option<(OutOfRange, OutOfRange, usize)>,
}

fn key_terms(terms: &[Term]) -> Vec<(Wire, u64)> {
    terms.iter().map(|&(wire, coefficient)| (wire, key_bits(coefficient))).collect()
}

pub struct Circuit {
    pub source_size: usize,
    pub operations: Vec<Operation>,
    pub depths: Vec<u32>,
    pub constants: Vec<Option<f64>>,
    pub ranges: Vec<Range>,
    /// The power of two every value of a wire is a multiple of, when known.
    pub grids: Vec<Option<f64>>,
    /// Declared with `assume_bit`.
    pub exact_bits: Vec<bool>,
    /// True once an operation reads the wire, so a later declaration can be
    /// refused.
    pub has_readers: Vec<bool>,
    pub one_hot_groups: Vec<OneHotGroup>,
    pub candidate_tables: Vec<CandidateTable>,
    expression_wires: HashMap<ExpressionKey, Wire>,
    tables_by_content: HashMap<Vec<Wire>, usize>,
}

impl Circuit {
    pub fn new(source_size: usize) -> Circuit {
        Circuit {
            source_size,
            operations: Vec::new(),
            depths: vec![0; source_size],
            constants: vec![None; source_size],
            ranges: vec![Range::UNKNOWN; source_size],
            grids: vec![None; source_size],
            exact_bits: vec![false; source_size],
            has_readers: vec![false; source_size],
            one_hot_groups: Vec::new(),
            candidate_tables: Vec::new(),
            expression_wires: HashMap::new(),
            tables_by_content: HashMap::new(),
        }
    }

    /// Every wire: the source cells, then one per operation.
    pub fn width(&self) -> usize {
        self.source_size + self.operations.len()
    }

    fn assert_wire(&self, wire: Wire) {
        assert!(wire.index() < self.width(), "wire is outside the compiled circuit");
    }

    fn operation_mut(&mut self, wire: Wire) -> &mut Operation {
        let index = wire.index() - self.source_size;
        &mut self.operations[index]
    }

    /// Declare that every value of this wire lies within minimum..maximum.
    /// Operations built before keep the range they were built with, so
    /// narrowing the range of a wire that is already read panics, unless the
    /// caller allows existing readers.
    pub fn assume_range_with(&mut self, wire: Wire, minimum: f64, maximum: f64, options: DeclarationOptions) {
        self.assert_wire(wire);
        assert!(minimum <= maximum, "numeric range is inverted");
        let current = self.ranges[wire.index()];
        let narrowed = Range::new(current.minimum.max(minimum), current.maximum.min(maximum));
        assert!(narrowed.minimum <= narrowed.maximum, "numeric range contradicts known bounds");
        let narrows = narrowed.minimum > current.minimum || narrowed.maximum < current.maximum;
        if narrows && self.has_readers[wire.index()] && !options.allow_existing_readers {
            panic!(
                "operations that already read wire {} were built with its range {}..{}, and declaring {minimum}..{maximum} \
                 now doesn't change them. Declare the range before the first operation that reads the wire, or allow \
                 existing readers",
                wire.index(),
                current.minimum,
                current.maximum,
            );
        }
        self.ranges[wire.index()] = narrowed;
    }

    pub fn assume_range(&mut self, wire: Wire, minimum: f64, maximum: f64) {
        self.assume_range_with(wire, minimum, maximum, DeclarationOptions::default());
    }

    /// Declare that these wires form an exact one-hot. Also narrows every
    /// member's range to 0..1.
    pub fn assume_one_hot(&mut self, members: &[Wire], allow_all_zero: bool, options: DeclarationOptions) {
        assert!(!members.is_empty(), "a one-hot group needs at least one member");
        let mut seen = std::collections::HashSet::new();
        for &member in members {
            self.assert_wire(member);
            assert!(seen.insert(member), "one-hot group members must be distinct wires");
            self.assume_range_with(member, 0.0, 1.0, options);
        }
        self.one_hot_groups.push(OneHotGroup { members: members.to_vec(), allow_all_zero });
    }

    /// Declare that every value of this wire is a multiple of `step`, a power
    /// of two, e.g. 1/64 for a pointer coordinate written in 1/64 px.
    pub fn assume_grid_with(&mut self, wire: Wire, step: f64, options: DeclarationOptions) {
        self.assert_wire(wire);
        assert!(is_power_of_two_grid(step), "a grid step must be a power of two from 2^-149 to 2^127, got {step}");
        let current = self.grids[wire.index()];
        let grid = current.unwrap_or(step).max(step);
        if Some(grid) != current && self.has_readers[wire.index()] && !options.allow_existing_readers {
            panic!(
                "operations that already read wire {} were built {}, and declaring the grid {step} now doesn't change \
                 them. Declare the grid before the first operation that reads the wire, or allow existing readers",
                wire.index(),
                match current {
                    None => "without a grid".to_string(),
                    Some(current) => format!("with its grid {current}"),
                },
            );
        }
        self.grids[wire.index()] = Some(grid);
    }

    pub fn assume_grid(&mut self, wire: Wire, step: f64) {
        self.assume_grid_with(wire, step, DeclarationOptions::default());
    }

    /// Declare that this wire holds exactly 0 or 1. Also narrows its range.
    pub fn assume_bit_with(&mut self, wire: Wire, options: DeclarationOptions) {
        self.assume_range_with(wire, 0.0, 1.0, options);
        self.exact_bits[wire.index()] = true;
    }

    pub fn assume_bit(&mut self, wire: Wire) {
        self.assume_bit_with(wire, DeclarationOptions::default());
    }

    /// Record an operation as a rounding point, e.g. a step of a round chain.
    pub fn mark_rounding_point(&mut self, wire: Wire) {
        self.record_rounding_point(wire, RoundingPointKind::Marked);
    }

    // Earlier records win, e.g. a boolean atom that emit_compare returns stays
    // Boolean.
    pub(crate) fn record_rounding_point(&mut self, wire: Wire, kind: RoundingPointKind) {
        self.assert_wire(wire);
        assert!(wire.index() >= self.source_size, "a rounding point must be an operation, not a source cell");
        let operation = self.operation_mut(wire);
        operation.rounding_point.get_or_insert(kind);
    }

    /// Every rounding point in wire order.
    pub fn rounding_points(&self) -> Vec<(Wire, RoundingPointKind)> {
        self.operations
            .iter()
            .enumerate()
            .filter_map(|(index, operation)| {
                operation.rounding_point.map(|kind| (Wire::from_position(self.source_size + index), kind))
            })
            .collect()
    }

    pub fn is_non_negative(&self, wire: Wire) -> bool {
        self.ranges[wire.index()].minimum >= 0.0
    }

    pub(crate) fn has_finite_range(&self, wire: Wire) -> bool {
        let range = self.ranges[wire.index()];
        fround(range.minimum).is_finite() && fround(range.maximum).is_finite()
    }

    /// The wire of source cell `index`.
    pub fn source(&self, index: usize) -> Wire {
        assert!(index < self.source_size, "source cell {index} is outside the source table");
        Wire::from_position(index)
    }

    /// A wire that always holds f(value). An ordinary affine that reads it adds
    /// the value to its bias instead.
    pub fn constant(&mut self, value: f64) -> Wire {
        self.affine(&[], value)
    }

    pub fn affine(&mut self, terms: &[Term], bias: f64) -> Wire {
        self.append(AtomKind::Affine, terms.to_vec(), bias, Vec::new(), 0.0, false, None)
    }

    /// Keep this evaluation, term order and input wires intact, including
    /// constants.
    pub fn round_affine(&mut self, terms: &[Term], bias: f64) -> Wire {
        self.append(AtomKind::Affine, terms.to_vec(), bias, Vec::new(), 0.0, true, None)
    }

    /// left × right as one exact FP32 product.
    pub fn multiply(&mut self, left: Wire, right: Wire) -> Wire {
        let macro_ = Some(Macro::Multiply { left, right });
        if self.is_non_negative(right) {
            return self.append(AtomKind::Reglu, vec![(left, 1.0)], 0.0, vec![(right, 1.0)], 0.0, true, macro_);
        }
        if self.is_non_negative(left) {
            return self.append(AtomKind::Reglu, vec![(right, 1.0)], 0.0, vec![(left, 1.0)], 0.0, true, macro_);
        }
        let positive = self.append(AtomKind::Reglu, vec![(left, 1.0)], 0.0, vec![(right, 1.0)], 0.0, true, None);
        let negative = self.append(AtomKind::Reglu, vec![(left, 1.0)], 0.0, vec![(right, -1.0)], 0.0, true, None);
        self.append(AtomKind::Affine, vec![(positive, 1.0), (negative, -1.0)], 0.0, Vec::new(), 0.0, true, macro_)
    }

    /// gate × on + (1 - gate) × off. Both sides always compute.
    pub fn select(&mut self, gate: Wire, off: Wire, on: Wire) -> Wire {
        let complement = self.round_affine(&[(gate, -1.0)], 1.0);
        let selected_on = self.multiply(gate, on);
        let selected_off = self.multiply(complement, off);
        self.append(
            AtomKind::Affine,
            vec![(selected_on, 1.0), (selected_off, 1.0)],
            0.0,
            Vec::new(),
            0.0,
            true,
            Some(Macro::Select { gate, off, on }),
        )
    }

    /// f(left × max(0, gate)) for two affine forms.
    pub fn reglu(&mut self, left_terms: &[Term], left_bias: f64, gate_terms: &[Term], gate_bias: f64) -> Wire {
        self.append(AtomKind::Reglu, left_terms.to_vec(), left_bias, gate_terms.to_vec(), gate_bias, false, None)
    }

    /// Round a score within 0..1 at 0.5: below becomes 0, the rest 1.
    pub fn boolean(&mut self, value: Wire) -> Wire {
        let range = self.ranges[value.index()];
        assert!(range.minimum >= 0.0 && range.maximum <= 1.0, "boolean input must be within zero and one");
        self.append(AtomKind::Boolean, vec![(value, 1.0)], 0.0, Vec::new(), 0.0, false, None)
    }

    /// Read candidates[floor(index)] as one instruction. The index needs a
    /// declared range strictly inside ±2^24.
    pub fn index_read(&mut self, index: Wire, candidates: &[Wire], out_of_range: OutOfRange) -> Wire {
        self.assert_wire(index);
        let index_range = self.ranges[index.index()];
        self.index_read_affine(&[(index, 1.0)], 0.0, candidates, out_of_range, out_of_range, Some(index_range))
    }

    /// The index read with the index written as an affine form and one rule per
    /// side. `index_range` is a caller promise about the form's value that
    /// narrows the range inferred from its operands.
    pub fn index_read_affine(
        &mut self,
        index_terms: &[Term],
        index_bias: f64,
        candidates: &[Wire],
        below: OutOfRange,
        above: OutOfRange,
        index_range: Option<Range>,
    ) -> Wire {
        let width = Wire::from_position(self.width());
        for &(wire, coefficient) in index_terms {
            self.assert_wire(wire);
            assert!(coefficient.is_finite(), "index coefficient must be finite");
        }
        assert!(index_bias.is_finite(), "index bias must be finite");
        let table = self.register_candidate_table(candidates);
        let (terms, bias) = fold_constant_terms(normalize_terms(index_terms), index_bias, &self.constants);
        // The wires that the read's index form reads, counted before copying
        // small affine producers in.
        let read_terms = terms.clone();
        let (terms, bias) = compose_affine_terms(&terms, bias, self.source_size, &self.operations);
        if terms.is_empty() {
            // A constant index selects its candidate while building.
            let constant_index = fround(bias);
            assert!(!constant_index.is_nan(), "a constant index must not be NaN");
            let candidates = &self.candidate_tables[table].candidates;
            let position = resolve_index_position(
                constant_index,
                candidates.len(),
                below == OutOfRange::Zero,
                above == OutOfRange::Zero,
            );
            return if position >= 0 { candidates[position as usize] } else { self.affine(&[], 0.0) };
        }
        let mut accepted = evaluate_affine_range(&terms, bias, &self.ranges);
        if let Some(declared) = index_range {
            accepted = Range::new(accepted.minimum.max(declared.minimum), accepted.maximum.min(declared.maximum));
            // A NaN endpoint passes here and fails the magnitude check below.
            assert!(
                !(accepted.minimum > accepted.maximum),
                "the declared index range contradicts the range inferred from the index form"
            );
        }
        if !(accepted.minimum > -INDEX_MAGNITUDE_LIMIT && accepted.maximum < INDEX_MAGNITUDE_LIMIT) {
            panic!(
                "index read needs an index range declared strictly inside ±2^24, got {}..{}",
                accepted.minimum, accepted.maximum
            );
        }
        let key = ExpressionKey {
            kind: AtomKind::Index,
            left_bias: key_bits(0.0),
            left_terms: Vec::new(),
            gate_bias: key_bits(bias),
            gate_terms: key_terms(&terms),
            preserve_rounding: false,
            macro_: None,
            index: Some((below, above, table)),
        };
        if let Some(&existing) = self.expression_wires.get(&key) {
            return existing;
        }
        let mut depth = self.candidate_tables[table].maximum_depth;
        for &(dependency, _) in &terms {
            depth = depth.max(self.depths[dependency.index()]);
        }
        let range = self.index_read_range(table, accepted, below, above);
        self.operations.push(Operation {
            kind: AtomKind::Index,
            left_terms: Vec::new(),
            left_bias: 0.0,
            gate_terms: terms,
            gate_bias: bias,
            depth: depth + 1,
            preserve_rounding: false,
            macro_: None,
            index_read: Some(IndexRead { table, below, above, index_range: accepted }),
            rounding_point: None,
            comparison_bit: false,
        });
        self.depths.push(depth + 1);
        self.constants.push(None);
        self.grids.push(None);
        self.exact_bits.push(false);
        self.has_readers.push(false);
        for &(wire, _) in &read_terms {
            self.has_readers[wire.index()] = true;
        }
        self.ranges.push(range);
        self.expression_wires.insert(key, width);
        width
    }

    // Register a candidates array once per circuit, by content.
    fn register_candidate_table(&mut self, candidates: &[Wire]) -> usize {
        if let Some(&table) = self.tables_by_content.get(candidates) {
            return table;
        }
        assert!(!candidates.is_empty(), "an index read needs at least one candidate");
        let mut maximum_depth = 0;
        for &candidate in candidates {
            self.assert_wire(candidate);
            maximum_depth = maximum_depth.max(self.depths[candidate.index()]);
        }
        let id = self.candidate_tables.len();
        self.candidate_tables.push(CandidateTable {
            id,
            candidates: candidates.to_vec(),
            maximum_depth,
            range_by_reach: HashMap::new(),
        });
        self.tables_by_content.insert(candidates.to_vec(), id);
        for &candidate in candidates {
            self.has_readers[candidate.index()] = true;
        }
        id
    }

    // The union of the ranges of every candidate an index within index_range
    // can reach, plus 0 where a side that reads zero is reachable.
    fn index_read_range(&mut self, table: usize, index_range: Range, below: OutOfRange, above: OutOfRange) -> Range {
        let count = self.candidate_tables[table].candidates.len() as i64;
        let lowest = (index_range.minimum.floor() as i64).clamp(-1, count);
        let highest = (index_range.maximum.floor() as i64).clamp(-1, count);
        let key = (lowest, highest, below, above);
        if let Some(&range) = self.candidate_tables[table].range_by_reach.get(&key) {
            return range;
        }
        let candidates = &self.candidate_tables[table].candidates;
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        let mut include = |range: Range| {
            minimum = minimum.min(range.minimum);
            maximum = maximum.max(range.maximum);
        };
        let last_direct = highest.min(count - 1);
        for position in lowest.max(0)..=last_direct {
            include(self.ranges[candidates[position as usize].index()]);
        }
        let zero = Range::new(0.0, 0.0);
        if lowest < 0 {
            include(if below == OutOfRange::Zero { zero } else { self.ranges[candidates[0].index()] });
        }
        if highest > count - 1 {
            include(if above == OutOfRange::Zero {
                zero
            } else {
                self.ranges[candidates[(count - 1) as usize].index()]
            });
        }
        let range = Range::new(minimum, maximum);
        self.candidate_tables[table].range_by_reach.insert(key, range);
        range
    }

    /// The wire an ordinary affine with exactly these normalized terms was
    /// registered under, e.g. the definition a consumer copied in.
    pub(crate) fn find_affine(&self, terms: &[Term], bias: f64) -> Option<Wire> {
        let key = ExpressionKey {
            kind: AtomKind::Affine,
            left_bias: key_bits(bias),
            left_terms: key_terms(terms),
            gate_bias: key_bits(0.0),
            gate_terms: Vec::new(),
            preserve_rounding: false,
            macro_: None,
            index: None,
        };
        self.expression_wires.get(&key).copied()
    }

    /// Append one atom. Without `preserve_rounding`, the terms are normalized,
    /// constant wires fold into the bias, small affine producers compose in,
    /// and an operation equal to an existing one returns that wire.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn append(
        &mut self,
        mut kind: AtomKind,
        mut left_terms: Vec<Term>,
        mut left_bias: f64,
        mut gate_terms: Vec<Term>,
        mut gate_bias: f64,
        preserve_rounding: bool,
        macro_: Option<Macro>,
    ) -> Wire {
        debug_assert!(kind != AtomKind::Index);
        // The wires that the operation reads, before small affine producers are
        // copied in, so a producer counts as read.
        let mut read_left_terms = left_terms.clone();
        let mut read_gate_terms = gate_terms.clone();
        if !preserve_rounding {
            (left_terms, left_bias) = fold_constant_terms(normalize_terms(&left_terms), left_bias, &self.constants);
            (gate_terms, gate_bias) = fold_constant_terms(normalize_terms(&gate_terms), gate_bias, &self.constants);
            read_left_terms = left_terms.clone();
            read_gate_terms = gate_terms.clone();
            if kind == AtomKind::Affine
                && let Some(alias) = identity_wire(&left_terms, left_bias)
            {
                return alias;
            }
            (left_terms, left_bias) = compose_affine_terms(&left_terms, left_bias, self.source_size, &self.operations);
            (gate_terms, gate_bias) = compose_affine_terms(&gate_terms, gate_bias, self.source_size, &self.operations);
            if kind == AtomKind::Reglu && gate_terms.is_empty() {
                let gate = relu_js(gate_bias);
                kind = AtomKind::Affine;
                left_terms = if gate == 0.0 {
                    Vec::new()
                } else {
                    left_terms.iter().map(|&(wire, coefficient)| (wire, coefficient * gate)).collect()
                };
                left_bias *= gate;
                gate_bias = 0.0;
            } else if kind == AtomKind::Reglu && left_terms.is_empty() && left_bias == 0.0 {
                kind = AtomKind::Affine;
                gate_terms = Vec::new();
                gate_bias = 0.0;
            }
            if kind == AtomKind::Affine
                && let Some(alias) = identity_wire(&left_terms, left_bias)
            {
                return alias;
            }
        }
        let key = ExpressionKey {
            kind,
            left_bias: key_bits(left_bias),
            left_terms: key_terms(&left_terms),
            gate_bias: key_bits(gate_bias),
            gate_terms: key_terms(&gate_terms),
            preserve_rounding,
            macro_,
            index: None,
        };
        if let Some(&existing) = self.expression_wires.get(&key) {
            return existing;
        }
        let wire = Wire::from_position(self.width());
        let mut depth = 0;
        for &(dependency, _) in left_terms.iter().chain(if kind == AtomKind::Reglu { &gate_terms[..] } else { &[] }) {
            depth = depth.max(self.depths[dependency.index()]);
        }
        let grid = infer_operation_grid(kind, &left_terms, left_bias, &gate_terms, gate_bias, &self.grids, &self.ranges);
        let left_range = evaluate_affine_range(&left_terms, left_bias, &self.ranges);
        let range = match kind {
            AtomKind::Affine => left_range,
            AtomKind::Reglu => {
                multiply_by_relu_range(left_range, evaluate_affine_range(&gate_terms, gate_bias, &self.ranges))
            }
            _ => Range::new(0.0, 1.0),
        };
        let constant = (!preserve_rounding && kind == AtomKind::Affine && left_terms.is_empty()).then_some(left_bias);
        if !left_terms.is_empty() {
            for &(dependency, _) in &read_left_terms {
                self.has_readers[dependency.index()] = true;
            }
        }
        if !gate_terms.is_empty() {
            for &(dependency, _) in &read_gate_terms {
                self.has_readers[dependency.index()] = true;
            }
        }
        self.operations.push(Operation {
            kind,
            left_terms,
            left_bias,
            gate_terms,
            gate_bias,
            depth: depth + 1,
            preserve_rounding,
            macro_,
            index_read: None,
            rounding_point: (kind == AtomKind::Boolean).then_some(RoundingPointKind::Boolean),
            comparison_bit: false,
        });
        self.depths.push(depth + 1);
        self.constants.push(constant);
        self.grids.push(grid);
        self.exact_bits.push(false);
        self.has_readers.push(false);
        self.ranges.push(range);
        self.expression_wires.insert(key, wire);
        wire
    }

    /// Evaluate every operation in order, in FP32. The result holds every
    /// wire's value: the source cells, then one per operation.
    pub fn evaluate_reference(&self, source: &[f32]) -> Vec<f32> {
        self.validate_source_ranges(source);
        let mut values = Vec::with_capacity(self.width());
        values.extend_from_slice(source);
        for operation in &self.operations {
            let mut value = evaluate_terms(&values, &operation.left_terms, operation.left_bias);
            match operation.kind {
                AtomKind::Affine => {}
                AtomKind::Reglu => {
                    let gate = evaluate_terms(&values, &operation.gate_terms, operation.gate_bias);
                    value = fround(value * relu_js(gate));
                }
                AtomKind::Boolean => value = evaluate_boolean(value),
                AtomKind::Index => {
                    let index = evaluate_terms(&values, &operation.gate_terms, operation.gate_bias);
                    value = self.read_indexed_value(&values, operation.index_read.as_ref().unwrap(), index);
                }
            }
            values.push(value as f32);
        }
        values
    }

    /// Backpropagate `output_gradients`, one per wire, through FP32 arithmetic
    /// taken as real-valued. ReLU has slope 0 at its corner.
    pub fn backpropagate_reference(
        &self,
        values: &[f32],
        output_gradients: &[f32],
        options: BackpropagateOptions,
    ) -> Result<Vec<f32>, BackpropagateError> {
        let width = self.width();
        assert!(values.len() == width, "forward values have the wrong length");
        assert!(output_gradients.len() == width, "output gradients have the wrong length");
        let mut gradients = output_gradients.to_vec();
        let value = |wire: Wire| values[wire.index()] as f64;
        for (operation_index, operation) in self.operations.iter().enumerate().rev() {
            let output_gradient = gradients[self.source_size + operation_index] as f64;
            if output_gradient == 0.0 {
                continue;
            }
            match operation.macro_ {
                Some(Macro::Multiply { left, right }) => {
                    accumulate_term_gradients(&mut gradients, &[(left, value(right))], output_gradient);
                    accumulate_term_gradients(&mut gradients, &[(right, value(left))], output_gradient);
                    continue;
                }
                Some(Macro::Select { gate, off, on }) => {
                    let gate_value = value(gate);
                    accumulate_term_gradients(
                        &mut gradients,
                        &[(off, 1.0 - gate_value), (on, gate_value)],
                        output_gradient,
                    );
                    accumulate_term_gradients(&mut gradients, &[(gate, value(on) - value(off))], output_gradient);
                    continue;
                }
                None => {}
            }
            match operation.kind {
                AtomKind::Index => {
                    // The selected candidate receives the output gradient, like a
                    // gather. The read is a step function of the index, so the
                    // caller chooses what the index form receives.
                    let read = operation.index_read.as_ref().unwrap();
                    if options.index_gradient == IndexGradient::Reject {
                        return Err(BackpropagateError::IndexOfIndexRead);
                    }
                    let index_value = evaluate_terms(values, &operation.gate_terms, operation.gate_bias);
                    let table = &self.candidate_tables[read.table];
                    let position = resolve_index_position(
                        index_value,
                        table.candidates.len(),
                        read.below == OutOfRange::Zero,
                        read.above == OutOfRange::Zero,
                    );
                    if position >= 0 {
                        accumulate_term_gradients(
                            &mut gradients,
                            &[(table.candidates[position as usize], 1.0)],
                            output_gradient,
                        );
                    }
                    if options.index_gradient == IndexGradient::BackwardDifference && index_value.is_finite() {
                        // What a ReLU one-hot masked sum gives at an integer index i:
                        // read(i) - read(i - 1), with the same rule outside the table.
                        let floor_index = index_value.floor();
                        let difference = self.read_indexed_value(values, read, floor_index)
                            - self.read_indexed_value(values, read, floor_index - 1.0);
                        accumulate_term_gradients(
                            &mut gradients,
                            &operation.gate_terms,
                            fround(output_gradient * difference),
                        );
                    }
                }
                AtomKind::Boolean => {
                    if options.boolean_gradient != BooleanGradient::StraightThrough {
                        return Err(BackpropagateError::BooleanAtom);
                    }
                    accumulate_term_gradients(&mut gradients, &operation.left_terms, output_gradient);
                }
                AtomKind::Affine => {
                    accumulate_term_gradients(&mut gradients, &operation.left_terms, output_gradient);
                }
                AtomKind::Reglu => {
                    let left = evaluate_terms(values, &operation.left_terms, operation.left_bias);
                    let gate = evaluate_terms(values, &operation.gate_terms, operation.gate_bias);
                    let active_gate = relu_js(gate);
                    accumulate_term_gradients(
                        &mut gradients,
                        &operation.left_terms,
                        fround(output_gradient * active_gate),
                    );
                    if gate > 0.0 {
                        accumulate_term_gradients(&mut gradients, &operation.gate_terms, fround(output_gradient * left));
                    }
                }
            }
        }
        Ok(gradients)
    }

    /// Panics when a source cell lies outside its declared range.
    pub fn validate_source_ranges(&self, source: &[f32]) {
        assert!(source.len() == self.source_size, "source table has the wrong length");
        for (wire, &value) in source.iter().enumerate() {
            let range = self.ranges[wire];
            let value = value as f64;
            if !(value >= range.minimum && value <= range.maximum) {
                panic!("source wire {wire} is outside its assumed range");
            }
        }
    }

    // The index read instruction on one index value: the selected candidate's
    // value, +0 where that side reads zero, and NaN for a NaN index.
    fn read_indexed_value(&self, values: &[f32], read: &IndexRead, index: f64) -> f64 {
        let candidates = &self.candidate_tables[read.table].candidates;
        let position = resolve_index_position(
            index,
            candidates.len(),
            read.below == OutOfRange::Zero,
            read.above == OutOfRange::Zero,
        );
        if position >= 0 {
            values[candidates[position as usize].index()] as f64
        } else if position == -1 {
            0.0
        } else {
            f64::NAN
        }
    }
}

// floor(index) as a table position: 0..count-1 inside the table; outside it the
// nearest end, or -1 where that side reads +0; -2 for a NaN index.
pub(crate) fn resolve_index_position(index: f64, count: usize, below_zero: bool, above_zero: bool) -> i64 {
    if index.is_nan() {
        return -2;
    }
    let position = index.floor();
    let last = count as f64 - 1.0;
    if position < 0.0 {
        if below_zero { -1 } else { 0 }
    } else if position > last {
        if above_zero { -1 } else { count as i64 - 1 }
    } else {
        position as i64
    }
}

pub(crate) fn evaluate_terms(values: &[f32], terms: &[Term], bias: f64) -> f64 {
    let mut value = fround(bias);
    for &(wire, coefficient) in terms {
        value = fround(value + fround(fround(coefficient) * values[wire.index()] as f64));
    }
    value
}

fn accumulate_term_gradients(gradients: &mut [f32], terms: &[Term], output_gradient: f64) {
    for &(wire, coefficient) in terms {
        let gradient = &mut gradients[wire.index()];
        *gradient = fround(*gradient as f64 + fround(output_gradient * fround(coefficient))) as f32;
    }
}

pub(crate) fn evaluate_boolean(value: f64) -> f64 {
    // FP32 rounds to whole numbers at this size; the odd base makes 0.5 round up.
    const BASE: f64 = 8_388_609.0;
    let score = fround(value);
    let shifted = fround(score + BASE);
    fround(shifted - BASE)
}

fn identity_wire(terms: &[Term], bias: f64) -> Option<Wire> {
    if bias != 0.0 || terms.len() != 1 || terms[0].1 != 1.0 {
        return None;
    }
    Some(terms[0].0)
}

fn compose_affine_terms(terms: &[Term], mut bias: f64, source_size: usize, operations: &[Operation]) -> (Vec<Term>, f64) {
    let mut composed = Vec::with_capacity(terms.len());
    for &(wire, coefficient) in terms {
        let operation = wire.index().checked_sub(source_size).map(|index| &operations[index]);
        match operation {
            Some(operation)
                if operation.kind == AtomKind::Affine
                    && !operation.preserve_rounding
                    && operation.left_terms.len() < AFFINE_SHARE_THRESHOLD =>
            {
                bias += coefficient * operation.left_bias;
                for &(dependency, dependency_coefficient) in &operation.left_terms {
                    composed.push((dependency, coefficient * dependency_coefficient));
                }
            }
            _ => composed.push((wire, coefficient)),
        }
    }
    (normalize_terms(&composed), bias)
}

fn fold_constant_terms(terms: Vec<Term>, mut bias: f64, constants: &[Option<f64>]) -> (Vec<Term>, f64) {
    let mut dynamic_terms = Vec::with_capacity(terms.len());
    for (wire, coefficient) in terms {
        match constants[wire.index()] {
            Some(constant) => bias += coefficient * constant,
            None => dynamic_terms.push((wire, coefficient)),
        }
    }
    (dynamic_terms, bias)
}

// Sort by wire (stably), merge repeated wires, and drop zero coefficients.
fn normalize_terms(terms: &[Term]) -> Vec<Term> {
    let mut sorted = terms.to_vec();
    sorted.sort_by_key(|&(wire, _)| wire);
    let mut normalized: Vec<Term> = Vec::with_capacity(sorted.len());
    for (wire, coefficient) in sorted {
        match normalized.last_mut() {
            Some(previous) if previous.0 == wire => previous.1 += coefficient,
            _ => normalized.push((wire, coefficient)),
        }
    }
    normalized.retain(|&(_, coefficient)| coefficient != 0.0);
    normalized
}

pub(crate) fn evaluate_affine_range(terms: &[Term], bias: f64, ranges: &[Range]) -> Range {
    let mut minimum = fround(bias);
    let mut maximum = fround(bias);
    for &(wire, coefficient) in terms {
        let range = ranges[wire.index()];
        let coefficient = fround(coefficient);
        if coefficient > 0.0 {
            minimum = fround(minimum + fround(coefficient * range.minimum));
            maximum = fround(maximum + fround(coefficient * range.maximum));
        } else {
            minimum = fround(minimum + fround(coefficient * range.maximum));
            maximum = fround(maximum + fround(coefficient * range.minimum));
        }
        if minimum.is_nan() || maximum.is_nan() {
            return Range::UNKNOWN;
        }
    }
    if minimum.is_nan() || maximum.is_nan() {
        return Range::UNKNOWN;
    }
    Range::new(minimum, maximum)
}

fn multiply_by_relu_range(left: Range, gate: Range) -> Range {
    let gate_minimum = relu_js(gate.minimum);
    let gate_maximum = relu_js(gate.maximum);
    let products = [
        fround(left.minimum * gate_minimum),
        fround(left.minimum * gate_maximum),
        fround(left.maximum * gate_minimum),
        fround(left.maximum * gate_maximum),
    ];
    // Zero times an infinite endpoint can produce NaN. Don't turn it into a
    // finite bound.
    if products.iter().any(|product| product.is_nan()) {
        return Range::UNKNOWN;
    }
    Range::new(
        products.iter().copied().fold(f64::INFINITY, js_min),
        products.iter().copied().fold(f64::NEG_INFINITY, js_max),
    )
}

// Math.min and Math.max on non-NaN values, where -0 is below +0.
fn js_min(a: f64, b: f64) -> f64 {
    if a < b || (a == b && a.is_sign_negative()) { a } else { b }
}

fn js_max(a: f64, b: f64) -> f64 {
    if a > b || (a == b && a.is_sign_positive()) { a } else { b }
}

struct GridForm {
    grid: f64,
    magnitude: f64,
}

// The grid of an affine form evaluated in FP32, when every term reads a wire
// with a grid and the declared ranges keep every product and partial sum within
// 2^24 grid steps, so nothing rounds.
fn infer_form_grid(terms: &[Term], bias: f64, grids: &[Option<f64>], ranges: &[Range]) -> Option<GridForm> {
    let rounded_bias = fround(bias);
    let mut grid = if rounded_bias == 0.0 { f64::INFINITY } else { lowest_power_of_two(rounded_bias) };
    let mut magnitude = rounded_bias.abs();
    for &(wire, coefficient) in terms {
        let wire_grid = grids[wire.index()]?;
        let range = ranges[wire.index()];
        let largest_operand = range.minimum.abs().max(range.maximum.abs());
        // An open range can hold Infinity, and f(0 × Infinity) is NaN.
        if !largest_operand.is_finite() {
            return None;
        }
        let rounded_coefficient = fround(coefficient);
        if rounded_coefficient == 0.0 {
            continue;
        }
        grid = grid.min(wire_grid * lowest_power_of_two(rounded_coefficient));
        magnitude += rounded_coefficient.abs() * largest_operand;
    }
    if !(grid.is_finite() && grid >= SMALLEST_GRID && magnitude <= EXACT_GRID_STEPS * grid) {
        return None;
    }
    Some(GridForm { grid: grid.min(LARGEST_GRID), magnitude })
}

fn infer_operation_grid(
    kind: AtomKind,
    left_terms: &[Term],
    left_bias: f64,
    gate_terms: &[Term],
    gate_bias: f64,
    grids: &[Option<f64>],
    ranges: &[Range],
) -> Option<f64> {
    if kind == AtomKind::Boolean {
        return Some(1.0);
    }
    let left = infer_form_grid(left_terms, left_bias, grids, ranges)?;
    if kind == AtomKind::Affine {
        return Some(left.grid);
    }
    let gate = infer_form_grid(gate_terms, gate_bias, grids, ranges)?;
    let grid = left.grid * gate.grid;
    if !(grid >= SMALLEST_GRID && left.magnitude * gate.magnitude <= EXACT_GRID_STEPS * grid) {
        return None;
    }
    Some(grid.min(LARGEST_GRID))
}
