// Builds each conformance scenario with the TypeScript runtime and writes what
// the Rust port has to reproduce bit for bit: every operation, range and grid,
// the reference evaluator's values and gradients on seeded inputs, the tensor
// programs compiled from the circuit and their evaluations, and what every
// validator says about each evaluation.
// crates/handwired/tests/conformance.rs builds the same scenarios in Rust.
//
//   node crates/handwired/conformance/generate.ts
//
// It also writes demos.json, the demo circuits of demos.ts, which
// crates/handwired/tests/demos.rs checks.
//
// Plain Node 23.6+ runs this file directly; it needs no install.

import {writeFileSync} from 'node:fs'
import {demoFixtures} from './demos.ts'
import {
  Circuit,
  describeFirstNaN,
  emitAnimatedObject,
  emitArgmax,
  emitClampExact,
  emitCompare,
  emitFirstActive,
  emitMaximum,
  emitMaximumExact,
  emitMinimum,
  emitMinimumExact,
  emitMultiply,
  emitRelu,
  emitSelect,
  emitSpring,
  inlineCircuit,
  type Relation,
  validateCircuit,
  validateGrids,
  validateIndexReads,
  validateOneHotGroups,
  validateTensorEvaluation,
  validateTensorProgram,
  type Wire,
} from '../../../src/runtime/numeric-ui-core.ts'

type Rng = () => number

function mulberry32(seed: number): Rng {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const scratch = new DataView(new ArrayBuffer(8))

function f64hex(value: number): string {
  scratch.setFloat64(0, value)
  return scratch.getBigUint64(0).toString(16).padStart(16, '0')
}

function f32bits(values: ArrayLike<number>): number[] {
  return Array.from(new Uint32Array(Float32Array.from(values).buffer))
}

function f32FromBits(bits: number): number {
  scratch.setUint32(0, bits)
  return scratch.getFloat32(0)
}

function f32ToBits(value: number): number {
  scratch.setFloat32(0, value)
  return scratch.getUint32(0)
}

// The neighboring FP32 values, for pairs that differ in the last bit.
function nextUp(value: number): number {
  const bits = f32ToBits(value)
  if (value === 0) return f32FromBits(1)
  return f32FromBits(value > 0 ? bits + 1 : bits - 1)
}

function nextDown(value: number): number {
  return -nextUp(-value)
}

const uniform = (rng: Rng, minimum: number, maximum: number) => Math.fround(minimum + rng() * (maximum - minimum))
const onGrid = (rng: Rng, minimum: number, maximum: number, step: number) =>
  Math.ceil(minimum / step) * step + Math.floor(rng() * ((maximum - minimum) / step + 1)) * step
const clampTo = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value))

type Scenario = {
  // Returns the roots of the scenario's roots-only program, the last operation by default.
  build: (circuit: Circuit) => Wire[] | void
  sourceSize: number
  samples: (rng: Rng) => number[][]
  backpropagate?: {booleanGradient?: 'reject' | 'straight-through', indexGradient?: 'reject' | 'zero' | 'backward-difference'}[]
}

const relations: Relation[] = ['>', '>=', '<', '<=', '===']

// Every relation, with and without a closing boolean atom, between the two
// sources, against a constant on both sides, and of a wire with itself.
function compareAll(circuit: Circuit, threshold: number) {
  const a = circuit.source(0)
  const b = circuit.source(1)
  const constant = circuit.constant(threshold)
  for (const endWithBoolean of [false, true]) {
    for (const relation of relations) {
      emitCompare(circuit, a, relation, b, {endWithBoolean})
      emitCompare(circuit, a, relation, constant, {endWithBoolean})
      emitCompare(circuit, constant, relation, b, {endWithBoolean})
      emitCompare(circuit, a, relation, a, {endWithBoolean})
    }
  }
}

// Random pairs plus the pairs that decide a comparison: equal, one FP32 step
// apart, zeros and the threshold itself.
function comparePairs(rng: Rng, minimum: number, maximum: number, threshold: number, draw: (rng: Rng) => number, steps = true) {
  const pairs: number[][] = []
  for (let index = 0; index < 40; index++) pairs.push([draw(rng), draw(rng)])
  for (let index = 0; index < 10; index++) {
    const value = draw(rng)
    pairs.push([value, value])
    if (steps) {
      pairs.push([clampTo(nextUp(value), minimum, maximum), value])
      pairs.push([clampTo(nextDown(value), minimum, maximum), value])
    }
  }
  pairs.push([0, 0], [-0, 0], [threshold, threshold], [minimum, maximum], [maximum, minimum])
  if (steps) pairs.push([clampTo(nextUp(threshold), minimum, maximum), clampTo(nextDown(threshold), minimum, maximum)])
  return pairs
}

function compareScenario(
  minimum: number,
  maximum: number,
  grids: [number | undefined, number | undefined],
  threshold: number,
  draw: (rng: Rng) => number,
  steps: boolean,
): Scenario {
  return {
    sourceSize: 2,
    build(circuit) {
      for (const index of [0, 1]) {
        const wire = circuit.source(index)
        circuit.assumeRange(wire, minimum, maximum)
        const grid = grids[index]
        if (grid !== undefined) circuit.assumeGrid(wire, grid)
      }
      compareAll(circuit, threshold)
    },
    samples: rng => comparePairs(rng, minimum, maximum, threshold, draw, steps),
  }
}

const scenarios: Record<string, Scenario> = {
  // The README's example: a > 4 ? 10 : 12.
  readme: {
    sourceSize: 1,
    build(circuit) {
      const a = circuit.source(0)
      circuit.assumeRange(a, -100, 100)
      circuit.assumeGrid(a, 1)
      const above4 = emitCompare(circuit, a, '>', circuit.constant(4))
      emitSelect(circuit, above4, circuit.constant(12), circuit.constant(10))
    },
    samples: () => Array.from({length: 201}, (_, index) => [index - 100]),
  },
  compare_grid_integer: compareScenario(-1000, 1000, [1, 1], 3, rng => onGrid(rng, -1000, 1000, 1), false),
  compare_grid_fine: compareScenario(0, 4096, [1 / 64, 1 / 64], 3.25, rng => onGrid(rng, 0, 4096, 1 / 64), false),
  compare_grid_mixed: compareScenario(-64, 64, [1 / 4, 1], 3.25, rng => onGrid(rng, -64, 64, 1), false),
  compare_grid_too_wide: compareScenario(-(2 ** 25), 2 ** 25, [1, 1], 3, rng => onGrid(rng, -(2 ** 25), 2 ** 25, 1), false),
  compare_cascade: compareScenario(-16384, 16384, [undefined, undefined], 0.1, rng => uniform(rng, -16384, 16384), true),
  compare_cascade_wide: compareScenario(-(2 ** 100), 2 ** 100, [undefined, undefined], 0.1, rng => uniform(rng, -(2 ** 100), 2 ** 100), true),
  compare_cascade_tiny: compareScenario(
    -(2 ** -100),
    2 ** -100,
    [undefined, undefined],
    2 ** -120,
    rng => (rng() < 0.3 ? f32FromBits(Math.floor(rng() * 1000)) * (rng() < 0.5 ? -1 : 1) : uniform(rng, -(2 ** -100), 2 ** -100)),
    true,
  ),
  minmax: {
    sourceSize: 6,
    build(circuit) {
      const [a, b, c, d, u, v] = [0, 1, 2, 3, 4, 5].map(index => circuit.source(index)) as Wire[]
      circuit.assumeRange(a, 0, 1000)
      circuit.assumeRange(b, 0, 1000)
      circuit.assumeRange(c, -50, 50)
      circuit.assumeRange(d, -50, 50)
      circuit.assumeRange(u, 0, 1)
      circuit.assumeRange(v, 0, 1)
      emitMinimum(circuit, a, b)
      emitMaximum(circuit, a, b)
      emitMinimum(circuit, c, d)
      emitMaximum(circuit, c, d)
      emitMaximum(circuit, u, v)
      emitMinimum(circuit, u, v)
      emitMinimumExact(circuit, a, b)
      emitMaximumExact(circuit, c, d)
      emitClampExact(circuit, a, b, circuit.constant(500))
      emitClampExact(circuit, c, d, circuit.constant(10))
    },
    samples: rng => Array.from({length: 60}, () => [
      uniform(rng, 0, 1000), uniform(rng, 0, 1000), uniform(rng, -50, 50), uniform(rng, -50, 50), uniform(rng, 0, 1), uniform(rng, 0, 1),
    ]).concat([[999.37, 0.3, -0, 0, 1, 1], [0, 0, 50, -50, 0, 0], [500, 500, 10, 10, 0.5, 0.5]].map(row => row.map(Math.fround))),
    backpropagate: [{}],
  },
  multiply_select: {
    sourceSize: 4,
    build(circuit) {
      const [x, y, g, z] = [0, 1, 2, 3].map(index => circuit.source(index)) as Wire[]
      circuit.assumeRange(x, -10, 10)
      circuit.assumeRange(y, 0, 5)
      circuit.assumeBit(g)
      emitMultiply(circuit, x, y)
      emitMultiply(circuit, y, x)
      emitMultiply(circuit, x, x)
      emitMultiply(circuit, x, z)
      emitSelect(circuit, g, x, y)
      emitSelect(circuit, g, x, z)
      emitRelu(circuit, x)
    },
    samples: rng => Array.from({length: 60}, () => [
      uniform(rng, -10, 10), uniform(rng, 0, 5), rng() < 0.5 ? 0 : 1, uniform(rng, -1e6, 1e6),
    ]).concat([[0, 0, 0, Infinity], [-0, 5, 1, -Infinity], [3, 2, 1, -1e30], [3, 2, 0, Infinity]]),
    backpropagate: [{}],
  },
  index_read: {
    sourceSize: 6,
    build(circuit) {
      const index = circuit.source(0)
      circuit.assumeRange(index, -3, 12)
      const table = [1, 2, 3, 4, 5].map(cell => circuit.source(cell))
      circuit.indexRead(index, table)
      circuit.indexRead(index, table, {outOfRange: 'zero'})
      circuit.indexRead(index, [...table], {outOfRange: 'zero'})
      circuit.indexReadAffine([[index, 1]], 2, table, 'clamp', 'zero')
      circuit.indexRead(circuit.affine([], 2), table)
      circuit.indexRead(circuit.affine([], -7), table, {outOfRange: 'zero'})
      const shifted = circuit.affine([[index, 1]], -1)
      circuit.indexRead(shifted, [table[4]!, table[0]!, table[2]!])
    },
    samples: rng => Array.from({length: 60}, () => [
      rng() < 0.5 ? Math.floor(uniform(rng, -3, 12)) : uniform(rng, -3, 12),
      ...Array.from({length: 5}, () => uniform(rng, -100, 100)),
    ]),
    backpropagate: [{indexGradient: 'zero'}, {indexGradient: 'backward-difference'}],
  },
  compose: {
    sourceSize: 5,
    build(circuit) {
      const [a, b, c, d, e] = [0, 1, 2, 3, 4].map(index => circuit.source(index)) as Wire[]
      for (const wire of [a, b, c, d]) circuit.assumeRange(wire, -100, 100)
      circuit.assumeRange(e, 0, 1)
      const t1 = circuit.affine([[a, 2], [b, 3]], 1)
      const t2 = circuit.affine([[t1, 0.5], [a, -1]])
      const k = circuit.constant(3)
      circuit.affine([[k, 2], [a, 1]])
      circuit.affine([[a, 1]])
      circuit.affine([[b, 3], [a, 2]], 1)
      const wide = circuit.affine([[a, 1], [b, 1], [c, 1], [d, 1]])
      circuit.affine([[wide, 2], [a, 1]])
      circuit.reglu([[a, 1]], 0, [], 2)
      circuit.reglu([], 0, [[a, 1]], 0)
      circuit.reglu([[t1, 1]], 0.5, [[t2, 1], [b, -1]], 0.25)
      circuit.boolean(e)
      circuit.boolean(circuit.affine([[e, 0.5]], 0.25))
      const rounded = circuit.roundAffine([[a, 1], [a, 1], [k, 1]], 0.5)
      circuit.markRoundingPoint(rounded)
      circuit.affine([[rounded, 0.1], [c, 1 / 3], [d, -1e-7]], 1e10)
    },
    samples: rng => Array.from({length: 60}, () => [
      uniform(rng, -100, 100), uniform(rng, -100, 100), uniform(rng, -100, 100), uniform(rng, -100, 100), uniform(rng, 0, 1),
    ]).concat([[0, 0, 0, 0, 0.5], [100, -100, 100, -100, Math.fround(0.49999997)]]),
    backpropagate: [{}, {booleanGradient: 'straight-through'}],
  },
}

// One-hot bits, then values, then extra cells. Most samples select one slot;
// some select none or two, which breaks the one-hot contract so the validators
// and the executor's clamp see it too.
function oneHotSamples(
  slotCount: number,
  valueRange: [number, number],
  extraCount = 0,
  extraRange: [number, number] = [0, 100],
): (rng: Rng) => number[][] {
  const finite = (range: [number, number]) => [Math.max(range[0], -1e6), Math.min(range[1], 1e6)] as const
  return rng => Array.from({length: 40}, (_, sample) => {
    const bits = Array<number>(slotCount).fill(0)
    const shape = sample % 10
    if (shape !== 8) bits[Math.floor(rng() * slotCount)] = 1
    if (shape === 9 && slotCount > 1) bits[(bits.indexOf(1) + 1) % slotCount] = 1
    const [low, high] = finite(valueRange)
    const values = Array.from({length: slotCount}, () => uniform(rng, low, high))
    // An infinite value at an unselected slot, where the declared range allows it.
    if (valueRange[1] === Infinity && sample % 4 === 0) {
      const unselected = bits.indexOf(0)
      if (unselected >= 0) values[unselected] = Infinity
    }
    const [extraLow, extraHigh] = finite(extraRange)
    const extras = Array.from({length: extraCount}, () => (extraRange[1] === Infinity && sample % 4 === 1 ? Infinity : uniform(rng, extraLow, extraHigh)))
    return [...bits, ...values, ...extras]
  })
}

// slotCount one-hot bits and slotCount values, declared before or after the
// products, and the reads that build() makes from them.
function gatherScenario(
  slotCount: number,
  valueRange: [number, number],
  declareBitsFirst: boolean,
  reads: (circuit: Circuit, products: [Wire, number][], extras: Wire[]) => Wire[],
  extraCount = 0,
  extraRange: [number, number] = [0, 100],
): Scenario {
  return {
    sourceSize: slotCount * 2 + extraCount,
    build(circuit) {
      const bits = Array.from({length: slotCount}, (_, slot) => circuit.source(slot))
      const values = Array.from({length: slotCount}, (_, slot) => circuit.source(slotCount + slot))
      const extras = Array.from({length: extraCount}, (_, index) => circuit.source(slotCount * 2 + index))
      if (declareBitsFirst) circuit.assumeOneHot(bits)
      for (const value of values) circuit.assumeRange(value, valueRange[0], valueRange[1])
      for (const extra of extras) circuit.assumeRange(extra, extraRange[0], extraRange[1])
      const products: [Wire, number][] = bits.map((bit, slot) => [emitMultiply(circuit, bit, values[slot]!), 1])
      const roots = reads(circuit, products, extras)
      if (!declareBitsFirst) circuit.assumeOneHot(bits, false, {allowExistingReaders: true})
      return roots
    },
    samples: oneHotSamples(slotCount, valueRange, extraCount, extraRange),
    backpropagate: [{}],
  }
}

const unbounded: [number, number] = [-Infinity, Infinity]

Object.assign(scenarios, {
  gather_basic: gatherScenario(4, [0, 4096], true, (circuit, products) => [
    circuit.affine(products, 0.1),
    circuit.affine(products),
  ]),
  gather_signed_pair: gatherScenario(6, [-4096, 4096], false, (circuit, products) => [circuit.affine(products)]),
  gather_bit_in_gate: gatherScenario(5, [-1e6, 1e6], true, (circuit, products) => [circuit.affine(products)]),
  gather_not_pure: gatherScenario(4, [0, 4096], true, (circuit, products, [extra]) => [
    circuit.affine(products.map(([wire], slot) => [wire, slot === 1 ? 2 : 1])),
    circuit.affine(products.slice(0, -1)),
    circuit.affine([...products, [circuit.affine([[extra!, 1]], 5), 1]]),
  ], 1, [0, 4096]),
  gather_unbounded_left: gatherScenario(2, [0, Infinity], true, (circuit, products) => [circuit.affine(products)]),
  gather_unbounded_gate: gatherScenario(2, unbounded, true, (circuit, products) => [circuit.affine(products)]),
  gather_unbounded_pair: gatherScenario(2, unbounded, false, (circuit, products) => [circuit.affine(products)]),
  gather_filler_finite: gatherScenario(2, [0, 100], true, (circuit, products, [filler]) => [
    circuit.affine([...products, [circuit.reglu([[filler!, 1]], 0, [[circuit.source(0), -1]], 0), 1]]),
  ], 1, [0, 100]),
  gather_filler_unbounded: gatherScenario(2, [0, 100], true, (circuit, products, [filler]) => [
    circuit.affine([...products, [circuit.reglu([[filler!, 1]], 0, [[circuit.source(0), -1]], 0), 1]]),
  ], 1, [0, Infinity]),
  // A negative arm with no positive partner keeps the sum dense: the exact
  // multiplies already read every slot.
  gather_unused_negative_arm: gatherScenario(3, [-100, 100], false, (circuit, products) => [
    circuit.affine([...products, [(products[0]![0] - 1) as Wire, -1]]),
  ]),
  gather_bit_in_left_signed: {
    sourceSize: 4,
    build(circuit) {
      const bits = [circuit.source(0), circuit.source(1)]
      const values = [circuit.source(2), circuit.source(3)]
      circuit.assumeOneHot(bits)
      for (const value of values) circuit.assumeRange(value, -100, 100)
      return [circuit.affine(bits.map((bit, slot) => [circuit.reglu([[bit, 1]], 0, [[values[slot]!, 1]], 0), 1]))]
    },
    samples: oneHotSamples(2, [-100, 100]),
  },
  gather_two_groups: {
    sourceSize: 8,
    build(circuit) {
      circuit.assumeOneHot([circuit.source(0), circuit.source(1)])
      circuit.assumeOneHot([circuit.source(2), circuit.source(3)])
      for (let value = 4; value < 8; value++) circuit.assumeRange(circuit.source(value), 0, 100)
      return [circuit.affine([0, 1, 2, 3].map(bit => [emitMultiply(circuit, circuit.source(bit), circuit.source(4 + bit)), 1]))]
    },
    samples: rng => Array.from({length: 30}, () => {
      const first = rng() < 0.5 ? [1, 0] : [0, 1]
      const second = rng() < 0.5 ? [1, 0] : [0, 1]
      return [...first, ...second, ...Array.from({length: 4}, () => uniform(rng, 0, 100))]
    }),
  },
  // A gather's output is a candidate of a second gather.
  gather_chain: {
    sourceSize: 10,
    build(circuit) {
      const first = [0, 1, 2].map(cell => circuit.source(cell))
      const second = [3, 4, 5].map(cell => circuit.source(cell))
      const values = [6, 7, 8, 9].map(cell => circuit.source(cell))
      circuit.assumeOneHot(first)
      circuit.assumeOneHot(second)
      for (const value of values) circuit.assumeRange(value, -50, 50)
      const inner = circuit.affine(first.map((bit, slot) => [emitMultiply(circuit, bit, values[slot]!), 1]))
      const outerValues = [inner, values[3]!, values[0]!]
      const outer = circuit.affine(second.map((bit, slot) => [emitMultiply(circuit, bit, outerValues[slot]!), 1]))
      return [outer, emitMinimum(circuit, outer, values[1]!)]
    },
    samples: rng => Array.from({length: 30}, () => {
      const pick = () => {
        const bits = [0, 0, 0]
        bits[Math.floor(rng() * 3)] = 1
        return bits
      }
      return [...pick(), ...pick(), ...Array.from({length: 4}, () => uniform(rng, -50, 50))]
    }),
    backpropagate: [{}],
  },
  argmax: {
    sourceSize: 9,
    build(circuit) {
      const scores = [0, 1, 2, 3].map(cell => circuit.source(cell))
      const values = [4, 5, 6, 7].map(cell => circuit.source(cell))
      const unit = circuit.source(8)
      for (const score of scores) circuit.assumeRange(score, 0, 1)
      for (const value of values) circuit.assumeRange(value, -50, 50)
      circuit.assumeRange(unit, 1, 1)
      const {selected, score} = emitArgmax(circuit, scores, unit)
      const read = circuit.affine(selected.map((bit, slot) => [emitMultiply(circuit, bit, values[slot]!), 1]))
      return [read, score, ...selected]
    },
    samples: rng => Array.from({length: 40}, (_, sample) => {
      const scores = Array.from({length: 4}, () => (sample % 5 === 0 ? 0.5 : uniform(rng, 0, 1)))
      return [...scores, ...Array.from({length: 4}, () => uniform(rng, -50, 50)), 1]
    }),
    backpropagate: [{booleanGradient: 'straight-through'}],
  },
  first_active_bits: {
    sourceSize: 9,
    build(circuit) {
      const xs = [0, 1, 2, 3].map(cell => circuit.source(cell))
      const values = [4, 5, 6, 7].map(cell => circuit.source(cell))
      const gate = circuit.source(8)
      for (const x of [...xs, gate]) {
        circuit.assumeRange(x, 0, 10)
        circuit.assumeGrid(x, 1)
      }
      for (const value of values) circuit.assumeRange(value, 0, 100)
      const five = circuit.constant(5)
      const candidates = xs.map(x => emitCompare(circuit, x, '>=', five))
      const active = emitCompare(circuit, gate, '>', circuit.constant(0))
      const selected = emitFirstActive(circuit, active, candidates)
      const read = circuit.affine(selected.map((bit, slot) => [emitMultiply(circuit, bit, values[slot]!), 1]))
      return [read, ...selected]
    },
    samples: rng => Array.from({length: 40}, () => [
      ...Array.from({length: 4}, () => onGrid(rng, 0, 10, 1)),
      ...Array.from({length: 4}, () => uniform(rng, 0, 100)),
      onGrid(rng, 0, 2, 1),
    ]),
  },
  first_active_fractional: {
    sourceSize: 9,
    build(circuit) {
      const candidates = [0, 1, 2, 3].map(cell => circuit.source(cell))
      const values = [4, 5, 6, 7].map(cell => circuit.source(cell))
      const active = circuit.source(8)
      for (const wire of [...candidates, active]) circuit.assumeRange(wire, 0, 1)
      for (const value of values) circuit.assumeRange(value, 0, 100)
      const selected = emitFirstActive(circuit, active, candidates)
      return [circuit.affine(selected.map((bit, slot) => [emitMultiply(circuit, bit, values[slot]!), 1]))]
    },
    samples: rng => Array.from({length: 30}, (_, sample) => [
      ...Array.from({length: 4}, () => (sample % 2 === 0 ? (rng() < 0.4 ? 1 : 0) : uniform(rng, 0, 1))),
      ...Array.from({length: 4}, () => uniform(rng, 0, 100)),
      sample % 3 === 0 ? 0 : 1,
    ]),
    backpropagate: [{}],
  },
  // The candidates share one constant, but the routing products stay distinct.
  first_active_shared_constant: {
    sourceSize: 1,
    build(circuit) {
      const unit = circuit.source(0)
      circuit.assumeRange(unit, 1, 1)
      const zero = circuit.constant(0)
      return emitFirstActive(circuit, unit, [zero, zero])
    },
    samples: () => [[1]],
  },
  spring: {
    sourceSize: 11,
    build(circuit) {
      const cells = Array.from({length: 11}, (_, cell) => circuit.source(cell))
      const ranges: [number, number][] = [
        [-1000, 1000], [-1000, 1000], [-5000, 5000], [0, 1000], [0, 100], [0, 0.05], [0, 1], [0, 1], [0, 10000], [0, 10000], [1, 1],
      ]
      cells.forEach((cell, index) => circuit.assumeRange(cell, ranges[index]![0], ranges[index]![1]))
      const [position, destination, velocity, stiffness, damping, timestep, settleDistance, settleSpeed, distanceScale, speedScale, unit] = cells as [Wire, Wire, Wire, Wire, Wire, Wire, Wire, Wire, Wire, Wire, Wire]
      const spring = emitSpring(circuit, position, destination, velocity, stiffness, damping, timestep, settleDistance, settleSpeed, distanceScale, speedScale, unit)
      return [spring.position, spring.velocity, spring.active]
    },
    samples: rng => Array.from({length: 40}, (_, sample) => {
      const destination = uniform(rng, -1000, 1000)
      const nearRest = sample % 4 === 0
      return [
        nearRest ? destination : uniform(rng, -1000, 1000),
        destination,
        nearRest ? 0 : uniform(rng, -5000, 5000),
        uniform(rng, 0, 1000), uniform(rng, 0, 100), Math.fround(0.004), Math.fround(0.01), Math.fround(0.01),
        uniform(rng, 0, 10000), uniform(rng, 0, 10000), 1,
      ]
    }),
    backpropagate: [{}],
  },
  animated_object: {
    sourceSize: 17,
    build(circuit) {
      const cells = Array.from({length: 17}, (_, cell) => circuit.source(cell))
      for (const cell of cells.slice(0, 8)) circuit.assumeRange(cell, -2, 2)
      for (const cell of cells.slice(8, 15)) circuit.assumeRange(cell, -1000, 1000)
      circuit.assumeBit(cells[15]!)
      circuit.assumeBit(cells[16]!)
      const object = emitAnimatedObject(circuit, [
        {matrix: [cells[0]!, cells[1]!, cells[2]!, cells[3]!], displacement: cells[8]!, velocity: cells[9]!, destination: cells[12]!, destinationDuringSteps: cells[14]!},
        {matrix: [cells[4]!, cells[5]!, cells[6]!, cells[7]!], displacement: cells[10]!, velocity: cells[11]!, destination: cells[13]!},
      ], {settleCheck: cells[15]!, snap: cells[16]!, settleDistance: 0.01, settleSpeed: 0.02})
      return [...object.positions, object.settled, object.active]
    },
    samples: rng => Array.from({length: 40}, (_, sample) => {
      const matrix = Array.from({length: 8}, () => uniform(rng, -2, 2))
      const atRest = sample % 4 === 0
      const nearRest = sample % 4 === 1
      const state = Array.from({length: 4}, () => (atRest ? (rng() < 0.5 ? 0 : -0) : nearRest ? uniform(rng, -0.004, 0.004) : uniform(rng, -1000, 1000)))
      const destinations = [uniform(rng, -1000, 1000), uniform(rng, -1000, 1000)]
      const during = sample % 3 === 0 ? destinations[0]! : uniform(rng, -1000, 1000)
      const identity = atRest || nearRest ? [1, 0, 0, 1, 1, 0, 0, 1] : matrix
      return [...identity, ...state, ...destinations, during, sample % 2, sample % 7 === 0 ? 1 : 0]
    }),
    backpropagate: [{booleanGradient: 'straight-through'}],
  },
  inline: {
    sourceSize: 4,
    build(target) {
      const part = new Circuit(3)
      const [a, b, index] = [0, 1, 2].map(cell => part.source(cell)) as [Wire, Wire, Wire]
      part.assumeRange(a, 0, 100)
      part.assumeRange(b, 0, 100)
      part.assumeGrid(a, 1)
      part.assumeGrid(b, 1)
      part.assumeRange(index, 0, 2)
      const greater = emitCompare(part, a, '>', b)
      const picked = emitSelect(part, greater, b, a)
      const smaller = emitMinimumExact(part, a, b)
      const read = part.indexRead(index, [a, b, picked], {outOfRange: 'zero'})
      const rounded = part.roundAffine([[read, 1], [smaller, 1]], 0.5)
      part.markRoundingPoint(rounded)
      const sources = [0, 1, 2, 3].map(cell => target.source(cell))
      const first = inlineCircuit(target, part, [sources[0]!, sources[1]!, sources[2]!])
      const second = inlineCircuit(target, part, [sources[1]!, sources[0]!, sources[3]!])
      return [first.at(-1)!, second.at(-1)!]
    },
    samples: rng => Array.from({length: 40}, () => [onGrid(rng, 0, 100, 1), onGrid(rng, 0, 100, 1), onGrid(rng, 0, 2, 1), uniform(rng, 0, 2)]),
    backpropagate: [{indexGradient: 'backward-difference'}],
  },
} satisfies Record<string, Scenario>)

// FNV-1a over the FP32 bits, every NaN as one NaN, so a program's values fit
// in one number.
function hashValues(values: Float32Array): number {
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.length)
  let hash = 0x811c9dc5
  for (const word of bits) {
    const canonical = (word & 0x7f800000) === 0x7f800000 && (word & 0x7fffff) !== 0 ? 0x7fc00000 : word
    for (let shift = 0; shift < 32; shift += 8) {
      hash ^= (canonical >>> shift) & 0xff
      hash = Math.imul(hash, 0x01000193) >>> 0
    }
  }
  return hash
}

function attempt(check: () => unknown): unknown {
  try {
    const result = check()
    return result === undefined ? 'ok' : result
  } catch (error) {
    return {error: (error as Error).message}
  }
}

type Program = ReturnType<Circuit['compileTensorProgram']>

function serializeProgram(program: Program) {
  return {
    layers: program.layers.map(layer => ({
      inputIndices: Array.from(layer.inputIndices),
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      bias: f32bits(layer.bias),
      gateRows: Array.from(layer.gateRows),
      booleanRows: Array.from(layer.booleanRows),
      rowOffsets: Array.from(layer.rowOffsets),
      columns: Array.from(layer.columns),
      weights: f32bits(layer.weights),
      gatherComputes: layer.gatherComputes.map(compute => ({slot: compute.slot, bitPositions: Array.from(compute.bitPositions)})),
      gatherReads: layer.gatherReads.map(read => ({
        row: read.row, slot: read.slot, masked: read.masked, candidatePositions: Array.from(read.candidatePositions),
      })),
      indexReads: layer.indexReads.map(read => ({
        row: read.row,
        indexRow: read.indexRow,
        belowZero: read.belowZero,
        aboveZero: read.aboveZero,
        candidateCount: read.candidateCount,
        candidateBase: read.candidateBase,
        candidateStride: read.candidateStride,
        candidatePositions: Array.from(read.candidatePositions),
      })),
    })),
    roots: Array.from(program.roots),
    valueCount: program.valueCount,
    wireOrder: Array.from(program.wireOrder),
    width: program.width,
    gatherSlotCount: program.gatherSlotCount,
  }
}

// How many samples per scenario also run the tensor programs and validators.
const programSampleCount = 16

const output: Record<string, unknown> = {}
let seed = 1
for (const [name, scenario] of Object.entries(scenarios)) {
  const circuit = new Circuit(scenario.sourceSize)
  const builtRoots = scenario.build(circuit)
  const width = circuit.sourceSize + circuit.operations.length
  const roots = builtRoots ?? [(width - 1) as Wire]
  const rng = mulberry32(seed++)
  const evaluations = scenario.samples(rng).map(row => {
    const source = Float32Array.from(row)
    return {source: f32bits(source), values: f32bits(circuit.evaluateReference(source))}
  })
  const gradients = (scenario.backpropagate ?? []).flatMap(options => evaluations.slice(0, 20).map(evaluation => {
    const values = new Float32Array(new Uint32Array(evaluation.values).buffer)
    const outputGradients = Float32Array.from({length: width}, () => uniform(rng, -1, 1))
    let result: number[] | string
    try {
      result = f32bits(circuit.backpropagateReference(values, outputGradients, options))
    } catch (error) {
      result = String((error as Error).message)
    }
    return {
      source: evaluation.source,
      outputGradients: f32bits(outputGradients),
      booleanGradient: options.booleanGradient ?? 'reject',
      indexGradient: options.indexGradient ?? 'reject',
      gradients: result,
    }
  }))
  // The first samples and the last few, where scenarios put their edge cases.
  const sampleIndices = [...new Set([
    ...Array.from({length: Math.min(programSampleCount - 4, evaluations.length)}, (_, index) => index),
    ...Array.from({length: Math.min(4, evaluations.length)}, (_, index) => evaluations.length - 1 - index),
  ])]
  const programSources = sampleIndices.map(index => new Float32Array(new Uint32Array(evaluations[index]!.source).buffer))
  const compiled: Record<string, Program> = {
    lowered: circuit.compileTensorProgram(undefined, true),
    plain: circuit.compileTensorProgram(undefined, false),
    recomputed: circuit.compileTensorProgram(undefined, true, {recomputeDepths: true}),
    roots: circuit.compileTensorProgram(roots, true, {recomputeDepths: true}),
  }
  const programs = Object.fromEntries(Object.entries(compiled).map(([label, program]) => [label, {
    program: serializeProgram(program),
    valid: attempt(() => validateTensorProgram(circuit, program)),
    valueHashes: programSources.map(source => hashValues(circuit.evaluateTensor(source, program))),
  }]))
  const evaluator = circuit.createTensorEvaluator(roots, {recomputeDepths: true})
  const validations = programSources.map((source, index) => {
    const values = new Float32Array(new Uint32Array(evaluations[sampleIndices[index]!]!.values).buffer)
    circuit.tensorProgram = null
    return {
      sample: sampleIndices[index],
      rootValues: f32bits(evaluator(source)),
      oneHotGroups: attempt(() => validateOneHotGroups(circuit, values)),
      indexReads: attempt(() => validateIndexReads(circuit, values)),
      grids: attempt(() => validateGrids(circuit, values)),
      firstNaN: describeFirstNaN(circuit, values),
      firstNaNOfRoots: describeFirstNaN(circuit, values, roots),
      tensorEvaluation: attempt(() => validateTensorEvaluation(source, circuit, circuit.evaluateTensor(source))),
    }
  })
  output[name] = {
    sourceSize: circuit.sourceSize,
    roots,
    operations: circuit.operations.map(operation => ({
      kind: operation.kind,
      leftTerms: operation.leftTerms.map(([wire, coefficient]) => [wire, f64hex(coefficient)]),
      leftBias: f64hex(operation.leftBias),
      gateTerms: operation.gateTerms.map(([wire, coefficient]) => [wire, f64hex(coefficient)]),
      gateBias: f64hex(operation.gateBias),
      depth: operation.depth,
      preserveRounding: operation.preserveRounding === true,
      macro: operation.macro === undefined
        ? null
        : operation.macro.kind === 'multiply'
          ? ['multiply', operation.macro.left, operation.macro.right]
          : ['select', operation.macro.gate, operation.macro.off, operation.macro.on],
      indexRead: operation.indexRead === undefined
        ? null
        : {
            table: operation.indexRead.table.id,
            below: operation.indexRead.below,
            above: operation.indexRead.above,
            indexRange: [f64hex(operation.indexRead.indexRange.minimum), f64hex(operation.indexRead.indexRange.maximum)],
          },
      roundingPoint: operation.roundingPoint ?? null,
      comparisonBit: operation.comparisonBit === true,
    })),
    ranges: circuit.ranges.map(range => [f64hex(range.minimum), f64hex(range.maximum)]),
    grids: circuit.grids.map(grid => (grid === undefined ? null : f64hex(grid))),
    exactBits: circuit.exactBits,
    oneHotGroups: circuit.oneHotGroups.map(group => ({members: group.members, allowAllZero: group.allowAllZero})),
    validCircuit: attempt(() => validateCircuit(circuit)),
    evaluations,
    gradients,
    programs,
    validations,
  }
}

const path = new URL('./fixtures.json', import.meta.url)
writeFileSync(path, JSON.stringify(output))
console.log(`wrote ${Object.keys(output).length} scenarios to ${path.pathname}`)

// The demo circuits of HandwiredOS and JSON search, see demos.ts.
const demosPath = new URL('./demos.json', import.meta.url)
writeFileSync(demosPath, JSON.stringify(demoFixtures()))
console.log(`wrote the demo circuits to ${demosPath.pathname}`)
