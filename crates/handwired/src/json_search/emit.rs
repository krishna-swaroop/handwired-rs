// A compiled tensor program as a list of rows, one per operation, and the
// WGSL `let` statement for each row, a port of src/json-search/emit.ts.

use crate::compile::{TensorIndexRead, TensorProgram};
use crate::fp::fround;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RowTerm {
    pub weight: f64,
    pub position: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Row {
    Affine { position: usize, bias: f64, terms: Vec<RowTerm> },
    Boolean { position: usize, bias: f64, terms: Vec<RowTerm> },
    Reglu { position: usize, bias: f64, terms: Vec<RowTerm>, gate_bias: f64, gate_terms: Vec<RowTerm> },
    Index {
        position: usize,
        index_bias: f64,
        index_terms: Vec<RowTerm>,
        below_zero: bool,
        above_zero: bool,
        candidate_count: usize,
        candidate_base: i64,
        candidate_stride: i64,
    },
}

impl Row {
    pub fn position(&self) -> usize {
        match self {
            Row::Affine { position, .. }
            | Row::Boolean { position, .. }
            | Row::Reglu { position, .. }
            | Row::Index { position, .. } => *position,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Row::Affine { .. } => "affine",
            Row::Boolean { .. } => "boolean",
            Row::Reglu { .. } => "reglu",
            Row::Index { .. } => "index",
        }
    }

    /// The terms that the row reads, gate terms first for a ReGLU.
    pub fn read_terms(&self) -> Vec<RowTerm> {
        match self {
            Row::Affine { terms, .. } | Row::Boolean { terms, .. } => terms.clone(),
            Row::Reglu { terms, gate_terms, .. } => gate_terms.iter().chain(terms).copied().collect(),
            Row::Index { index_terms, .. } => index_terms.clone(),
        }
    }
}

/// The layers' CSR matrices as one list of rows, in execution order.
pub fn flatten_program(program: &TensorProgram) -> Vec<Row> {
    let mut rows = Vec::new();
    let mut available = program.source_size;
    for layer in &program.layers {
        assert!(
            layer.gather_reads.is_empty() && layer.gather_computes.is_empty(),
            "this emitter doesn't write gathers, and codegen's wgsl.ts does"
        );
        let terms_of = |projection_row: usize| -> Vec<RowTerm> {
            (layer.row_offsets[projection_row] as usize..layer.row_offsets[projection_row + 1] as usize)
                .map(|index| RowTerm {
                    weight: layer.weights[index] as f64,
                    position: layer.input_indices[layer.columns[index] as usize] as usize,
                })
                .collect()
        };
        for row in 0..layer.output_size {
            let position = available + row;
            let read: Option<&TensorIndexRead> = layer.index_reads.iter().find(|read| read.row == row);
            let gate_projection =
                layer.gate_rows.iter().position(|&gate_row| gate_row as usize == row).map(|gate| layer.output_size + gate);
            let is_boolean = layer.boolean_rows.iter().any(|&boolean_row| boolean_row as usize == row);
            if let Some(read) = read {
                assert!(gate_projection.is_none() && !is_boolean, "an index read row is neither a gate row nor a boolean row");
                let last = read.candidate_base + (read.candidate_count as i64 - 1) * read.candidate_stride;
                let source = program.source_size as i64;
                let computed = if !read.candidate_positions.is_empty() {
                    read.candidate_positions.iter().any(|&candidate| candidate as i64 >= source)
                } else {
                    read.candidate_base >= source || last >= source
                };
                assert!(!computed, "an index read whose candidates are computed values isn't emitted, only one whose candidates are source cells");
                assert!(read.candidate_positions.is_empty(), "an index read whose source cells aren't evenly spaced isn't emitted");
                rows.push(Row::Index {
                    position,
                    index_bias: layer.bias[read.index_row] as f64,
                    index_terms: terms_of(read.index_row),
                    below_zero: read.below_zero,
                    above_zero: read.above_zero,
                    candidate_count: read.candidate_count,
                    candidate_base: read.candidate_base,
                    candidate_stride: read.candidate_stride,
                });
            } else if let Some(gate_projection) = gate_projection {
                assert!(!is_boolean, "a gate row isn't a boolean row");
                rows.push(Row::Reglu {
                    position,
                    bias: layer.bias[row] as f64,
                    terms: terms_of(row),
                    gate_bias: layer.bias[gate_projection] as f64,
                    gate_terms: terms_of(gate_projection),
                });
            } else if is_boolean {
                rows.push(Row::Boolean { position, bias: layer.bias[row] as f64, terms: terms_of(row) });
            } else {
                rows.push(Row::Affine { position, bias: layer.bias[row] as f64, terms: terms_of(row) });
            }
        }
        available += layer.output_size;
    }
    assert!(available == program.value_count, "the layers hold every live value");
    rows
}

/// The packed position of each wire. A source wire's position is the wire itself.
pub fn positions_of_wires(program: &TensorProgram, wires: &[crate::circuit::Wire]) -> Vec<usize> {
    let mut position_of_wire = vec![-1i64; program.width];
    for (position, &wire) in program.wire_order.iter().enumerate() {
        position_of_wire[wire as usize] = position as i64;
    }
    wires
        .iter()
        .map(|wire| {
            let position = position_of_wire[wire.index()];
            assert!(position >= 0, "wire {} isn't a live value of this program", wire.index());
            position as usize
        })
        .collect()
}

/// A float32 value as one WGSL literal with the exact bits.
pub fn wgsl_float(value: f64) -> String {
    assert!(value.is_finite() && fround(value) == value, "a WGSL literal must be a finite float32 value");
    if value == 0.0 && value.is_sign_negative() {
        return "(-0.0f)".to_string();
    }
    if value.fract() == 0.0 && value.abs() < 16_777_216.0 {
        let whole = value as i64;
        return if value < 0.0 { format!("({whole}.0f)") } else { format!("{whole}.0f") };
    }
    let bits = (value.abs() as f32).to_bits();
    let exponent_bits = (bits >> 23) & 0xff;
    let fraction = format!("{:06x}", (bits & 0x7fffff) << 1);
    let magnitude = if exponent_bits == 0 {
        format!("0x0.{fraction}p-126f")
    } else {
        format!("0x1.{fraction}p{}f", exponent_bits as i32 - 127)
    };
    if value < 0.0 { format!("(-{magnitude})") } else { magnitude }
}

pub fn wgsl_int(value: i64) -> String {
    if value < 0 { format!("({value}i)") } else { format!("{value}i") }
}

const TERMS_PER_STATEMENT: usize = 32;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RowCounts {
    pub affine: usize,
    pub reglu: usize,
    pub boolean: usize,
    pub index: usize,
}

/// The text that reads an index read's candidate at the clamped index.
type CandidateReader<'a> = Box<dyn FnMut(&Row, &str) -> String + 'a>;

/// Writes rows as WGSL `let` statements, one row at a time, into `body`.
pub struct RowWriter<'a> {
    source_size: usize,
    prefix: String,
    source: Box<dyn FnMut(usize) -> String + 'a>,
    candidate: CandidateReader<'a>,
    pub body: Vec<String>,
    pub row_counts: RowCounts,
}

impl<'a> RowWriter<'a> {
    pub fn new(
        source_size: usize,
        prefix: &str,
        source: impl FnMut(usize) -> String + 'a,
        candidate: impl FnMut(&Row, &str) -> String + 'a,
    ) -> RowWriter<'a> {
        RowWriter {
            source_size,
            prefix: prefix.to_string(),
            source: Box::new(source),
            candidate: Box::new(candidate),
            body: Vec::new(),
            row_counts: RowCounts::default(),
        }
    }

    pub fn name(&mut self, position: usize) -> String {
        if position >= self.source_size { format!("{}v{position}", self.prefix) } else { (self.source)(position) }
    }

    fn product(&mut self, term: RowTerm) -> String {
        let magnitude = term.weight.abs();
        let name = self.name(term.position);
        if magnitude == 1.0 { name } else { format!("{} * {name}", wgsl_float(magnitude)) }
    }

    fn sum(&mut self, partial: &str, bias: f64, terms: &[RowTerm]) -> String {
        if terms.is_empty() {
            return wgsl_float(bias);
        }
        let mut text = String::new();
        for (index, &term) in terms.iter().enumerate() {
            let operand = self.product(term);
            let sign = if term.weight < 0.0 { '-' } else { '+' };
            if index == 0 && bias == 0.0 {
                text = if term.weight < 0.0 { format!("-{operand}") } else { operand };
            } else {
                let left = if index == 0 { wgsl_float(bias) } else { text.clone() };
                text = format!("{left} {sign} {operand}");
            }
            if (index + 1) % TERMS_PER_STATEMENT == 0 && index + 1 < terms.len() {
                let partial_name = format!("{partial}_{}", (index + 1) / TERMS_PER_STATEMENT);
                self.body.push(format!("let {partial_name}: f32 = {text};"));
                text = partial_name;
            }
        }
        text
    }

    pub fn write(&mut self, row: &Row) {
        let prefix = self.prefix.clone();
        let target = format!("{prefix}v{}", row.position());
        match row {
            Row::Affine { bias, terms, .. } => {
                self.row_counts.affine += 1;
                let sum = self.sum(&target, *bias, terms);
                self.body.push(format!("let {target}: f32 = {sum};"));
            }
            Row::Boolean { bias, terms, .. } => {
                self.row_counts.boolean += 1;
                let sum = self.sum(&target, *bias, terms);
                self.body.push(format!("let {target}: f32 = select(0.0f, 1.0f, {sum} >= 0.5f);"));
            }
            Row::Reglu { position, bias, terms, gate_bias, gate_terms } => {
                self.row_counts.reglu += 1;
                let gate_sum = self.sum(&format!("{prefix}g{position}"), *gate_bias, gate_terms);
                let gate = format!("max(0.0f, {gate_sum})");
                if terms.is_empty() && *bias == 1.0 {
                    self.body.push(format!("let {target}: f32 = {gate};"));
                } else {
                    let sum = self.sum(&target, *bias, terms);
                    self.body.push(format!("let {target}: f32 = ({sum}) * {gate};"));
                }
            }
            Row::Index { position, index_bias, index_terms, below_zero, above_zero, candidate_count, .. } => {
                self.row_counts.index += 1;
                let index = format!("{prefix}p{position}");
                let sum = self.sum(&index, *index_bias, index_terms);
                self.body.push(format!("let {index}: i32 = i32(floor({sum}));"));
                let last = *candidate_count as i64 - 1;
                let mut read = (self.candidate)(row, &format!("clamp({index}, 0i, {last}i)"));
                if *above_zero {
                    read = format!("select({read}, 0.0f, {index} > {last}i)");
                }
                if *below_zero {
                    read = format!("select({read}, 0.0f, {index} < 0i)");
                }
                self.body.push(format!("let {target}: f32 = {read};"));
            }
        }
    }
}
