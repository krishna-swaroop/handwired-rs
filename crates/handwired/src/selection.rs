// Tournaments that pick one item: argmax over scores, and the first active
// candidate. A port of emitArgmax and emitFirstActive in
// src/runtime/numeric-ui-core.ts.

use std::collections::HashSet;

use crate::circuit::{AtomKind, Circuit, DeclarationOptions, IndexRead, OutOfRange, Wire};
use crate::emit::{emit_maximum, emit_multiply, emit_select};

enum ArgmaxNode {
    Leaf { score: Wire, item: usize },
    Match { score: Wire, left_wins: Wire, left: Box<ArgmaxNode>, right: Box<ArgmaxNode> },
}

impl ArgmaxNode {
    fn score(&self) -> Wire {
        match self {
            ArgmaxNode::Leaf { score, .. } | ArgmaxNode::Match { score, .. } => *score,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Argmax {
    /// An exact one-hot over the items, declared as a group.
    pub selected: Vec<Wire>,
    pub score: Wire,
}

/// The item with the largest score, ties to the earlier item. Scores must be
/// within 0..1, and `unit` must be declared as exactly 1.
pub fn emit_argmax(circuit: &mut Circuit, scores: &[Wire], unit: Wire) -> Argmax {
    assert!(!scores.is_empty(), "argmax needs at least one score");
    let unit_range = circuit.ranges[unit.index()];
    assert!(unit_range.minimum == 1.0 && unit_range.maximum == 1.0, "argmax unit must equal one");
    let mut level: Vec<ArgmaxNode> = scores
        .iter()
        .enumerate()
        .map(|(item, &score)| {
            let range = circuit.ranges[score.index()];
            assert!(range.minimum >= 0.0 && range.maximum <= 1.0, "argmax scores must be within zero and one");
            ArgmaxNode::Leaf { score, item }
        })
        .collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut nodes = level.into_iter();
        while let Some(left) = nodes.next() {
            let Some(right) = nodes.next() else {
                next.push(left);
                break;
            };
            // One FP32 step above 0.5 makes exact ties choose left without exceeding 1.
            let comparison_bias = 0.5000000596046448;
            let comparison_scale = 0.4999999403953552;
            let comparison = circuit.affine(
                &[(left.score(), comparison_scale), (right.score(), -comparison_scale)],
                comparison_bias,
            );
            circuit.assume_range(comparison, 0.0, 1.0);
            let left_wins = circuit.boolean(comparison);
            let score = emit_select(circuit, left_wins, right.score(), left.score());
            next.push(ArgmaxNode::Match { score, left_wins, left: Box::new(left), right: Box::new(right) });
        }
        level = next;
    }
    let root = level.pop().unwrap();
    let mut selected = vec![None; scores.len()];
    route_argmax(circuit, &root, unit, &mut selected);
    let selected: Vec<Wire> = selected.into_iter().map(Option::unwrap).collect();
    // The tournament routes the unit down exactly one path.
    circuit.assume_one_hot(&selected, false, DeclarationOptions::default());
    Argmax { selected, score: root.score() }
}

fn route_argmax(circuit: &mut Circuit, node: &ArgmaxNode, route: Wire, selected: &mut [Option<Wire>]) {
    match node {
        ArgmaxNode::Leaf { item, .. } => selected[*item] = Some(route),
        ArgmaxNode::Match { left_wins, left, right, .. } => {
            let left_route = emit_multiply(circuit, route, *left_wins);
            let right_route = circuit.affine(&[(route, 1.0), (left_route, -1.0)], 0.0);
            circuit.assume_range(right_route, 0.0, 1.0);
            route_argmax(circuit, left, left_route, selected);
            route_argmax(circuit, right, right_route, selected);
        }
    }
}

enum WinnerNode {
    Leaf { mass: Wire, item: usize },
    Match { mass: Wire, left: Box<WinnerNode>, right: Box<WinnerNode> },
}

impl WinnerNode {
    fn mass(&self) -> Wire {
        match self {
            WinnerNode::Leaf { mass, .. } | WinnerNode::Match { mass, .. } => *mass,
        }
    }
}

/// Route `active` to the first candidate with mass, e.g. the topmost hit
/// window. Candidates and the gate are within 0..1. The selection is declared
/// one-hot-or-all-zero only when the gate and every candidate are proven bits.
pub fn emit_first_active(circuit: &mut Circuit, active: Wire, candidates: &[Wire]) -> Vec<Wire> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let active_range = circuit.ranges[active.index()];
    assert!(
        active_range.minimum >= 0.0 && active_range.maximum <= 1.0,
        "winner selection must be gated within zero and one"
    );
    let mut level: Vec<WinnerNode> = candidates
        .iter()
        .enumerate()
        .map(|(item, &mass)| {
            let range = circuit.ranges[mass.index()];
            assert!(range.minimum >= 0.0 && range.maximum <= 1.0, "winner candidates must be within zero and one");
            WinnerNode::Leaf { mass, item }
        })
        .collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut nodes = level.into_iter();
        while let Some(left) = nodes.next() {
            let Some(right) = nodes.next() else {
                next.push(left);
                break;
            };
            let mass = emit_maximum(circuit, left.mass(), right.mass());
            next.push(WinnerNode::Match { mass, left: Box::new(left), right: Box::new(right) });
        }
        level = next;
    }
    let root = level.pop().unwrap();
    let mut selected = vec![None; candidates.len()];
    let route = emit_multiply(circuit, active, root.mass());
    route_winner(circuit, &root, route, &mut selected);
    let selected: Vec<Wire> = selected.into_iter().map(Option::unwrap).collect();
    if is_proven_bit(circuit, active) && candidates.iter().all(|&candidate| is_proven_bit(circuit, candidate)) {
        let distinct: HashSet<Wire> = selected.iter().copied().collect();
        if distinct.len() == selected.len() {
            circuit.assume_one_hot(&selected, true, DeclarationOptions::default());
        }
    }
    selected
}

fn route_winner(circuit: &mut Circuit, node: &WinnerNode, route: Wire, selected: &mut [Option<Wire>]) {
    match node {
        WinnerNode::Leaf { item, .. } => selected[*item] = Some(route),
        WinnerNode::Match { left, right, .. } => {
            let left_route = emit_multiply(circuit, route, left.mass());
            let right_route = circuit.affine(&[(route, 1.0), (left_route, -1.0)], 0.0);
            circuit.assume_range(right_route, 0.0, 1.0);
            route_winner(circuit, left, left_route, selected);
            route_winner(circuit, right, right_route, selected);
        }
    }
}

/// A wire that provably holds exactly 0 or 1: a range pinned to 0 or 1, a
/// boolean atom, a bit `emit_compare` returned, a one-hot group member, or an
/// index read whose reachable candidates are all proven bits. A 0..1 range
/// alone is not enough.
pub fn is_proven_bit(circuit: &Circuit, wire: Wire) -> bool {
    let mut pending = vec![wire];
    let mut visited = HashSet::new();
    while let Some(current) = pending.pop() {
        if !visited.insert(current) {
            continue;
        }
        if is_bit_by_construction(circuit, current) {
            continue;
        }
        let read = (current.index() >= circuit.source_size)
            .then(|| circuit.operations[current.index() - circuit.source_size].index_read)
            .flatten();
        let Some(read) = read else { return false };
        pending.extend(reachable_candidates(circuit, &read));
    }
    true
}

fn is_bit_by_construction(circuit: &Circuit, wire: Wire) -> bool {
    let range = circuit.ranges[wire.index()];
    if range.minimum == range.maximum && (range.minimum == 0.0 || range.minimum == 1.0) {
        return true;
    }
    if wire.index() >= circuit.source_size {
        let operation = &circuit.operations[wire.index() - circuit.source_size];
        if operation.kind == AtomKind::Boolean || operation.comparison_bit {
            return true;
        }
    }
    circuit.one_hot_groups.iter().any(|group| group.members.contains(&wire))
}

// The candidates an index read can return for an index within its accepted
// range. A side that reads zero adds +0 instead of a candidate.
fn reachable_candidates(circuit: &Circuit, read: &IndexRead) -> Vec<Wire> {
    let candidates = &circuit.candidate_tables[read.table].candidates;
    let count = candidates.len() as i64;
    let lowest = (read.index_range.minimum.floor() as i64).clamp(-1, count);
    let highest = (read.index_range.maximum.floor() as i64).clamp(-1, count);
    let mut reachable = Vec::new();
    for position in lowest.max(0)..=highest.min(count - 1) {
        reachable.push(candidates[position as usize]);
    }
    if lowest < 0 && read.below == OutOfRange::Clamp {
        reachable.push(candidates[0]);
    }
    if highest > count - 1 && read.above == OutOfRange::Clamp {
        reachable.push(candidates[(count - 1) as usize]);
    }
    reachable
}
