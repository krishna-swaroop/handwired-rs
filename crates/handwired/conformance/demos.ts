// The demo circuits of HandwiredOS and JSON search, built and driven with the
// TypeScript originals, recorded for crates/handwired/tests/demos.rs: each
// circuit's operations and compiled program, and every value along scripted
// traces of the machines and loop programs. generate.ts writes this next to
// fixtures.json as demos.json.
//
// Large arrays are recorded as FNV-1a hashes of their exact bits (wordHash),
// with the hash of each operation and each compiled layer kept apart, so a
// mismatch names the first operation or layer that differs.

import {Circuit, type Wire, validateCircuit, validateTensorProgram} from '../../../src/runtime/numeric-ui-core.ts'

type TensorProgram = ReturnType<Circuit['compileTensorProgram']>
import {brentKungPrefix, sklanskyPrefix} from '../../../src/runtime/prefix-networks.ts'
import {
  type PlannedEvaluation,
  advanceFrame,
  countSpringSteps,
  defaultSpringClockConfig,
  quantizeFrameTiming,
  springMatrixBound,
  springMatrixPower,
  springStepMatrix,
  stoppedSpringClock,
} from '../../../src/runtime/host-timing.ts'
import {constant, createEvaluator, createSourceTable, event, input, inputBit, list, oneHot, permutation, state, stateBit} from '../../../src/runtime/cells.ts'
import {createWindowManagerMachine, noteFocusChange, windowOutputField} from '../../../src/handwiredos/window-manager.ts'
import {createPhotoGalleryMachine, photoField, springConstants, springMatrixPowers} from '../../../src/handwiredos/photo-gallery.ts'
import {createValueListCircuit, describeOperations, describeState, formatValue, monitorLayout} from '../../../src/handwiredos/value-list.ts'
import {arrangeInitialWindows, arrangeWindows} from '../../../src/handwiredos/arrangement.ts'
import {averageFrameTimes, createMonitorFooter, frameTimeScale, frameTimesPath, measureProgram} from '../../../src/handwiredos/monitor-footer.ts'
import {arcOpacity, drawWiring} from '../../../src/handwiredos/wiring.ts'
import {buildJsonLoop} from '../../../src/json-search/json-step.ts'
import {buildSearchLoop, type SearchScope} from '../../../src/json-search/search-step.ts'
import {createLoopRunner, describeLoop, type LoopProgram, operationsPerDocument} from '../../../src/json-search/loop-program.ts'
import {emitLoopWgsl} from '../../../src/json-search/loop-emit.ts'
import {capacityErrors, compilePattern, PatternSyntaxError} from '../../../src/json-search/regex/pattern.ts'
import {highlightWithTables, tableLoopResult} from '../../../src/json-search/table-loop.ts'
import {generateDocuments} from '../../../src/json-search/documents.ts'
import {cellsPerDocument, writeDocumentCells} from '../../../src/json-search/program.ts'
import {
  classTable, depthChangeTable, depthClassTable, displayTable, grammarErrorTable, grammarNextTable, grammarTokenTable,
  levelWeightTable, lexerErrorKindTable, lexerEventTable, lexerNextTable, objectWeightTable,
} from '../../../src/json-search/highlighter/tables.ts'

const scratch = new DataView(new ArrayBuffer(8))

export function f64hex(value: number): string {
  scratch.setFloat64(0, value)
  return scratch.getBigUint64(0).toString(16).padStart(16, '0')
}

const f64s = (values: readonly number[]) => values.map(f64hex)

// FNV-1a over 32-bit words, byte by byte, low byte first.
function hashWords(words: readonly number[]): number {
  let hash = 0x811c9dc5
  for (const word of words) {
    for (let shift = 0; shift < 32; shift += 8) {
      hash ^= (word >>> shift) & 0xff
      hash = Math.imul(hash, 0x01000193) >>> 0
    }
  }
  return hash
}

// The words of an f64, every NaN as one NaN: low half, then high half.
function f64Words(value: number): [number, number] {
  if (Number.isNaN(value)) return [0, 0x7ff80000]
  scratch.setFloat64(0, value)
  return [scratch.getUint32(4), scratch.getUint32(0)]
}

// The words of FP32 values, every NaN as one NaN.
function f32Words(values: ArrayLike<number>): number[] {
  const bits = new Uint32Array(Float32Array.from(values).buffer)
  return Array.from(bits, word => (word & 0x7f800000) === 0x7f800000 && (word & 0x7fffff) !== 0 ? 0x7fc00000 : word)
}

const hashF32 = (values: ArrayLike<number>) => hashWords(f32Words(values))
const hashF64 = (values: readonly number[]) => hashWords(values.flatMap(f64Words))
const f32bits = (values: ArrayLike<number>) => f32Words(values)

const kindCodes = {affine: 0, reglu: 1, boolean: 2, index: 3}
const roundingCodes = {'boolean': 1, 'integer bit': 2, 'sign cascade': 3, 'marked': 4}
const outOfRangeCodes = {clamp: 0, zero: 1}

function termWords(terms: readonly (readonly [number, number])[]) {
  return [terms.length, ...terms.flatMap(([wire, coefficient]) => [wire, ...f64Words(coefficient)])]
}

function operationHash(circuit: Circuit, index: number) {
  const operation = circuit.operations[index]!
  const macro = operation.macro === undefined
    ? [0]
    : operation.macro.kind === 'multiply'
      ? [1, operation.macro.left, operation.macro.right]
      : [2, operation.macro.gate, operation.macro.off, operation.macro.on]
  const read = operation.indexRead === undefined
    ? [0]
    : [
      1, operation.indexRead.table.id, outOfRangeCodes[operation.indexRead.below], outOfRangeCodes[operation.indexRead.above],
      ...f64Words(operation.indexRead.indexRange.minimum), ...f64Words(operation.indexRead.indexRange.maximum),
    ]
  return hashWords([
    kindCodes[operation.kind], ...termWords(operation.leftTerms), ...f64Words(operation.leftBias),
    ...termWords(operation.gateTerms), ...f64Words(operation.gateBias), operation.depth, operation.preserveRounding === true ? 1 : 0,
    ...macro, ...read, operation.roundingPoint === undefined ? 0 : roundingCodes[operation.roundingPoint], operation.comparisonBit === true ? 1 : 0,
  ])
}

function layerHash(layer: TensorProgram['layers'][number]) {
  const list = (values: ArrayLike<number>) => [values.length, ...Array.from(values)]
  return hashWords([
    ...list(layer.inputIndices), layer.inputSize, layer.outputSize, ...list(f32Words(layer.bias)), ...list(layer.gateRows),
    ...list(layer.booleanRows), ...list(layer.rowOffsets), ...list(layer.columns), ...list(f32Words(layer.weights)),
    layer.gatherComputes.length, ...layer.gatherComputes.flatMap(compute => [compute.slot, ...list(compute.bitPositions)]),
    layer.gatherReads.length, ...layer.gatherReads.flatMap(read => [read.row, read.slot, read.masked ? 1 : 0, ...list(read.candidatePositions)]),
    layer.indexReads.length, ...layer.indexReads.flatMap(read => [
      read.row, read.indexRow, read.belowZero ? 1 : 0, read.aboveZero ? 1 : 0, read.candidateCount, read.candidateBase >>> 0,
      read.candidateStride >>> 0, ...list(read.candidatePositions),
    ]),
  ])
}

function attempt(check: () => unknown): unknown {
  try {
    const result = check()
    return result === undefined ? 'ok' : result
  } catch (error) {
    return {error: (error as Error).message}
  }
}

// What a circuit is: its operations, declarations and validity.
export function describeCircuit(circuit: Circuit) {
  return {
    sourceSize: circuit.sourceSize,
    operationCount: circuit.operations.length,
    operationHashes: circuit.operations.map((_, index) => operationHash(circuit, index)),
    rangesHash: hashF64(circuit.ranges.flatMap(range => [range.minimum, range.maximum])),
    gridsHash: hashF64(circuit.grids.map(grid => grid ?? -1)),
    exactBitsHash: hashWords(circuit.exactBits.map(Number)),
    oneHotGroups: circuit.oneHotGroups.map(group => ({members: group.members, allowAllZero: group.allowAllZero})),
    roundingPoints: circuit.roundingPoints().map(({wire, kind}) => [wire, roundingCodes[kind]]),
    validCircuit: attempt(() => validateCircuit(circuit)),
  }
}

// What a compiled program is: its layers, roots and packed order.
export function describeProgram(circuit: Circuit, program: TensorProgram) {
  return {
    layerCount: program.layers.length,
    layerHashes: program.layers.map(layerHash),
    roots: Array.from(program.roots),
    sourceSize: program.sourceSize,
    valueCount: program.valueCount,
    wireOrder: Array.from(program.wireOrder),
    width: program.width,
    gatherSlotCount: program.gatherSlotCount,
    valid: attempt(() => validateTensorProgram(circuit, program)),
  }
}

// ---------------------------------------------------------------------------
// Runtime helpers.

function runtimeHelpers() {
  const prefixes = Array.from({length: 20}, (_, count) => {
    const items = Array.from({length: count}, (_, index) => String(index))
    const combine = (earlier: string, later: string) => `(${earlier},${later})`
    return {sklansky: sklanskyPrefix(items, combine), brentKung: brentKungPrefix(items, combine)}
  })

  const config = defaultSpringClockConfig
  const constants = [springConstants, {stiffness: 170, damping: 26}, {stiffness: 0, damping: 0}, {stiffness: 4000, damping: 10}]
  const springs = constants.map(spring => {
    const step = springStepMatrix(spring)
    return {
      step: f64s(step),
      powersHash: hashF64(Array.from({length: 301}, (_, steps) => springMatrixPower(step, steps)).flat()),
      power17: f64s(springMatrixPower(step, 17)),
      bound: f64hex(springMatrixBound(spring)),
    }
  })
  const frameTimings = [
    [16.6667, null, [10.2, 16.6, 20]], [1000.123, 983.5, [980, 990.01, 1000.123, 1005]], [5000, 1000, [1500, 4999.99]],
    [0, null, []], [33.3, 16.6, [-5, 16.6, 33.3]], [-0.001, null, [-0.001]],
  ].map(([frame, previous, events]) => {
    const timing = quantizeFrameTiming(frame as number, previous as number | null, events as number[])
    return [f64hex(timing.frameTimeMs), f64hex(timing.durationMs), f64s(timing.eventTimesMs)]
  })
  const stepCounts = [[0, 300], [17.5, 300], [1300, 300], [1200, 300], [4, 0], [3.984375, 1], [8, 1]].map(([pending, budget]) => {
    const count = countSpringSteps(pending!, budget!)
    return [count.steps, f64hex(count.leftoverMs), count.snap]
  })

  // advanceFrame with an evaluator that animates for a scripted time after each event.
  type Event = {timeStamp: number, label: string}
  const advance = (writeBack: 'every evaluation' | 'events and settle checks') => {
    let clock = stoppedSpringClock
    let animateUntil = -Infinity
    const frames: unknown[] = []
    const script: [number, Event[]][] = [
      [16, [{timeStamp: 10, label: 'a'}]], [33, []], [50, [{timeStamp: 40, label: 'b'}, {timeStamp: 38, label: 'c'}, {timeStamp: 40, label: 'd'}]],
      [66, []], [74, []], [83, []], [400, [{timeStamp: 120, label: 'e'}]], [2500, []], [2516, [{timeStamp: 2600, label: 'f'}]], [2533, []],
      [2600, []], [2617, [{timeStamp: 2610, label: 'g'}]], [4000, []], [4016, []], [4100.5, [{timeStamp: 4100.25, label: 'h'}]], [4200, []],
    ]
    for (const [frameTime, events] of script) {
      const evaluations: unknown[] = []
      clock = advanceFrame(clock, frameTime, events, (evaluation: PlannedEvaluation<Event>) => {
        for (const event of evaluation.events) animateUntil = Math.max(animateUntil, evaluation.timeMs + 150)
        const animating = evaluation.simulatedTimeMs < animateUntil && !evaluation.snap
        evaluations.push([
          evaluation.kind, f64hex(evaluation.timeMs), f64hex(evaluation.simulatedTimeMs), f64hex(evaluation.durationMs),
          evaluation.steps, evaluation.settleCheck, evaluation.snap, evaluation.events.map(event => event.label), evaluation.writesBack, animating,
        ])
        return animating
      }, writeBack)
      frames.push({
        evaluations,
        clock: [clock.animatedUntilMs === null ? null : f64hex(clock.animatedUntilMs), f64hex(clock.checkOriginMs), clock.lastEvaluationMs === null ? null : f64hex(clock.lastEvaluationMs)],
      })
    }
    return frames
  }

  // A table with every kind of cell, its validator's verdicts, and an evaluator.
  const table = createSourceTable()
  const cells = table.declare({
    width: input([0, 100], {initial: 40, grid: 1}),
    margin: constant(4, [0, 10]),
    press: {active: event({grid: 1}), x: input([-50, 50], {grid: 0.25})},
    count: state([0, 1000], {grid: 1}),
    on: stateBit(1),
    tabs: oneHot(list(3, index => stateBit(Number(index === 0))), {allowAllZero: false}),
    order: permutation([1, 0]),
    spare: inputBit(0),
  })
  table.atMostOne('press and spare', [cells.press.active, cells.spare])
  const circuit = table.createCircuit()
  const nextCount = circuit.roundAffine([[cells.count, 1], [cells.press.active, 1]])
  table.next(cells.count, nextCount)
  table.next(cells.on, circuit.affine([[cells.on, -1]], 1))
  table.next(cells.tabs, [cells.tabs[2]!, cells.tabs[0]!, cells.tabs[1]!])
  table.next(cells.order, [[cells.order[1]![0]!, cells.order[1]![1]!], [cells.order[0]![0]!, cells.order[0]![1]!]])
  const {nextSource, initialSource, validate} = table.finish()
  const output = {sum: circuit.affine([[cells.width, 1], [cells.margin, 2], [cells.press.x, 1]]), half: circuit.affine([[cells.count, 0.5]])}
  const evaluate = createEvaluator({circuit, nextSource, validate, output})
  const broken: [number, number][] = [[0, 101], [0, 2.5], [2, 0.5], [3, 0.1], [5, 2], [6, 1], [8, 1], [10, 0], [11, 1], [4, Number.NaN], [4, -1], [12, 1]]
  const verdicts = broken.map(([cell, value]) => {
    const source = Float32Array.from(initialSource)
    source[cell] = value
    return attempt(() => validate(source))
  })
  const evaluations: unknown[] = []
  let source = Float32Array.from(initialSource)
  for (let frame = 0; frame < 4; frame++) {
    if (frame % 2 === 0) {
      source[cells.press.active] = 1
      source[cells.press.x] = frame * 3.25 - 7
    }
    const result = evaluate(source)
    evaluations.push({next: f32bits(result.nextValues), sum: f64hex(result.output.sum), half: f64hex(result.output.half)})
    source = Float32Array.from(result.nextValues)
  }
  return {
    prefixes,
    springs,
    frameTimings,
    stepCounts,
    advance: {every: advance('every evaluation'), events: advance('events and settle checks')},
    cells: {
      names: Array.from({length: table.size}, (_, index) => table.nameOf(index as Wire)),
      policies: Array.from({length: table.size}, (_, index) => table.policyOf(index as Wire)),
      initialSource: f32bits(initialSource),
      nextSource,
      circuit: describeCircuit(circuit),
      verdicts,
      evaluations,
    },
    config: [config.timeGrid, config.stepMs, config.maxStepsPerFrame, config.settleCheckSteps],
  }
}

// ---------------------------------------------------------------------------
// HandwiredOS.

type WmMachine = ReturnType<typeof createWindowManagerMachine>
type WmEvent =
  | {kind: 'down' | 'move' | 'up' | 'cancel', x: number, y: number, t: number | null}
  | {kind: 'previous' | 'next' | 'close', t: number | null}
  | {kind: 'key', key: 'up' | 'down' | 'pageUp' | 'pageDown' | 'home' | 'end', t: number | null}
  | {kind: 'resize', width: number, height: number, t: number | null}

function queueWm(machine: WmMachine, event: WmEvent) {
  const t = event.t ?? undefined
  switch (event.kind) {
    case 'down': return machine.pointerDown(event.x, event.y, t)
    case 'move': return machine.pointerMove(event.x, event.y, t)
    case 'up': return machine.pointerUp(event.x, event.y, t)
    case 'cancel': return machine.pointerCancel(event.x, event.y, t)
    case 'previous': return machine.photoPrevious(t)
    case 'next': return machine.photoNext(t)
    case 'close': return machine.photoClose(t)
    case 'key': return machine.valueListKey(event.key, t)
    case 'resize': return machine.resize(event.width, event.height, t)
  }
}

function windowManagerTrace() {
  const machine = createWindowManagerMachine()
  const evaluations: unknown[] = []
  machine.observe({
    afterEvaluate(values, table) {
      evaluations.push([hashF32(values), hashF32(table)])
    },
  })
  let width = 1280
  let height = 800
  const rectangles = arrangeInitialWindows(machine.source, machine.cells, width, height)
  const initialAfterArrangement = hashF32(machine.source)
  let frameTime = 0
  const steps: unknown[] = []
  let values = new Float32Array(0)
  const out = (windowIndex: number, field: number) => values[machine.output.windows[windowIndex * windowOutputField.size + field]!]!
  // The window's drawn rectangle from its outputs.
  const rect = (windowIndex: number) => ({
    left: out(windowIndex, windowOutputField.translateX), top: out(windowIndex, windowOutputField.translateY),
    width: out(windowIndex, windowOutputField.width), height: out(windowIndex, windowOutputField.height),
  })
  const step = (events: WmEvent[], options: {elapsed?: number, scrollTop?: number} = {}) => {
    frameTime += options.elapsed ?? 16
    for (const event of events) queueWm(machine, event)
    evaluations.length = 0
    values = machine.step(width, height, frameTime, options.scrollTop)
    steps.push({
      frameTime: f64hex(frameTime), width: f64hex(width), height: f64hex(height),
      scrollTop: options.scrollTop === undefined ? null : f64hex(options.scrollTop),
      events: events.map(event => ({...event, ...('x' in event ? {x: f64hex(event.x), y: f64hex(event.y)} : {}),
        ...('width' in event ? {width: f64hex(event.width), height: f64hex(event.height)} : {}), t: event.t === null ? null : f64hex(event.t)})),
      evaluations: machine.evaluations.map(evaluation => [evaluation.kind, f64hex(evaluation.timeMs), evaluation.steps, evaluation.writesBack, evaluation.events]),
      evaluationHashes: [...evaluations],
      valuesHash: hashF32(values),
      sourceHash: hashF32(machine.source),
      packedHash: hashF32(machine.packedValues),
      windows: f32bits(machine.output.windows.map(wire => values[wire]!)),
      clock: [machine.clock.animatedUntilMs === null ? null : f64hex(machine.clock.animatedUntilMs), f64hex(machine.clock.checkOriginMs),
        machine.clock.lastEvaluationMs === null ? null : f64hex(machine.clock.lastEvaluationMs)],
      scrollTo: machine.valueListScrollTo === null ? null : f64hex(machine.valueListScrollTo),
    })
  }
  const settle = (limit = 120) => {
    for (let frame = 0; frame < limit; frame++) {
      step([])
      if (values[machine.output.animationActive] === 0) return
    }
  }
  const at = (offset: number) => frameTime + 16 - offset
  const click = (x: number, y: number) => step([{kind: 'down', x, y, t: at(8)}, {kind: 'up', x, y, t: at(6)}])
  const titleBar = (windowIndex: number) => {
    const r = rect(windowIndex)
    return {x: r.left + 60.3, y: r.top + 20}
  }
  // A window button's center: 0 close, 1 maximize, 2 minimize.
  const button = (windowIndex: number, slot: number) => {
    const r = rect(windowIndex)
    return {x: r.left + r.width - 10 - 18 - slot * 28 + 9, y: r.top + 22}
  }

  step([])
  settle()
  // Hover moves while idle, which coalesce, then a drag of Monitor's title bar in samples.
  step([0, 1, 2, 3].map(index => ({kind: 'move', x: 300 + index * 7.3, y: 400 + index, t: at(12 - index * 2)})))
  let point = titleBar(0)
  step([{kind: 'down', x: point.x, y: point.y, t: at(10)}, {kind: 'move', x: point.x + 12, y: point.y + 5, t: at(4)}])
  for (let sample = 1; sample <= 5; sample++) {
    step([{kind: 'move', x: point.x + 12 + sample * 31.7, y: point.y + 5 + sample * 9.1, t: at(9)}, {kind: 'move', x: point.x + 20 + sample * 31.7, y: point.y + 8 + sample * 9.1, t: at(3)}])
  }
  step([{kind: 'up', x: point.x + 180, y: point.y + 52, t: at(5)}])
  settle()
  // Resize Photos by its grip.
  let r = rect(2)
  point = {x: r.left + r.width - 10, y: r.top + r.height - 10}
  step([{kind: 'down', x: point.x, y: point.y, t: at(8)}])
  step([{kind: 'move', x: point.x - 40, y: point.y - 20, t: at(8)}])
  step([{kind: 'move', x: point.x - 90.5, y: point.y - 61.25, t: at(8)}, {kind: 'up', x: point.x - 90.5, y: point.y - 61.25, t: at(2)}])
  settle()
  // Maximize Notes, in the middle of its animation restore it, then settle.
  point = button(1, 1)
  click(point.x, point.y)
  step([])
  step([])
  point = button(1, 1)
  click(point.x, point.y)
  settle()
  // Minimize Photos to the dock, then reopen it from the dock.
  point = button(2, 2)
  click(point.x, point.y)
  settle()
  const dock = (windowIndex: number) => {
    const start = windowIndex * 7
    const read = (field: number) => values[machine.output.dockItems[start + field]!]!
    return {x: read(0) + read(2) / 2, y: read(1) + read(3) / 2}
  }
  point = dock(2)
  click(point.x, point.y)
  settle()
  // Open a photo in Photos' grid, step through photos, and close it.
  r = rect(2)
  const photoStart = 5 * photoField.size
  const photo = (field: number) => values[machine.output.photos.photos[photoStart + field]!]!
  point = {x: r.left + 1 + photo(photoField.left) + photo(photoField.width) / 2, y: r.top + 45 + photo(photoField.top) + photo(photoField.height) / 2}
  click(point.x, point.y)
  step([])
  step([{kind: 'next', t: at(8)}])
  step([{kind: 'next', t: at(10)}, {kind: 'previous', t: at(5)}])
  settle()
  step([{kind: 'close', t: at(8)}])
  step([])
  // A click on another photo while the first returns.
  const other = 2 * photoField.size
  point = {x: r.left + 1 + values[machine.output.photos.photos[other]!]! + 10, y: r.top + 45 + values[machine.output.photos.photos[other + 1]!]! + 10}
  click(point.x, point.y)
  settle()
  step([{kind: 'close', t: at(8)}])
  settle()
  // Monitor: focus it, move the selection with keys, scroll, change location, click a row.
  point = titleBar(0)
  click(point.x, point.y)
  settle()
  step([{kind: 'key', key: 'down', t: at(8)}])
  step([{kind: 'key', key: 'down', t: at(8)}, {kind: 'key', key: 'down', t: at(4)}])
  step([{kind: 'key', key: 'pageDown', t: at(8)}])
  step([{kind: 'key', key: 'end', t: at(8)}])
  step([], {scrollTop: 1000.6})
  step([{kind: 'key', key: 'home', t: at(8)}])
  step([], {scrollTop: 500.3})
  r = rect(0)
  const contentTop = r.top + 45
  click(r.left + 1 + monitorLayout.locationInset + 20, contentTop + monitorLayout.locationTop + 0.5 * monitorLayout.locationHeight)
  step([], {scrollTop: 0})
  click(r.left + 1 + monitorLayout.sidebarWidth + 40, contentTop + monitorLayout.headerHeight + 3.5 * monitorLayout.rowHeight)
  step([{kind: 'key', key: 'up', t: at(8)}])
  click(r.left + 1 + monitorLayout.locationInset + 20, contentTop + monitorLayout.locationTop + 2.5 * monitorLayout.locationHeight)
  step([{kind: 'key', key: 'pageUp', t: at(8)}])
  settle()
  // The viewport shrinks to a tablet, a phone, a sideways phone, and back.
  for (const [nextWidth, nextHeight] of [[900, 700], [390, 844], [844, 390], [1280, 800]] as const) {
    width = nextWidth
    height = nextHeight
    step([])
    step([])
    settle()
  }
  // Close Notes and reopen it from the dock, with a stall that snaps.
  point = button(1, 0)
  click(point.x, point.y)
  step([])
  step([], {elapsed: 2000})
  settle()
  point = dock(1)
  click(point.x, point.y)
  step([])
  // A drag that a pointercancel ends, and a press on a button released elsewhere.
  point = titleBar(1)
  step([{kind: 'down', x: point.x, y: point.y, t: at(8)}])
  step([{kind: 'move', x: point.x - 50, y: point.y + 30, t: at(8)}])
  step([{kind: 'cancel', x: point.x - 55, y: point.y + 33, t: at(8)}])
  settle()
  point = button(0, 0)
  step([{kind: 'down', x: point.x, y: point.y, t: at(8)}])
  step([{kind: 'up', x: point.x + 200, y: point.y + 200, t: at(8)}])
  // Events without timestamps, and a resize event between frames.
  step([{kind: 'down', x: 5, y: 5, t: null}, {kind: 'up', x: 5, y: 5, t: null}])
  step([{kind: 'resize', width: 1100.3, height: 760.9, t: at(10)}])
  settle()

  return {
    rectangles: rectangles.map(rectangle => f64s([rectangle.left, rectangle.top, rectangle.width, rectangle.height])),
    initialAfterArrangement,
    steps,
  }
}

function windowManager() {
  const machine = createWindowManagerMachine()
  const {circuit, tensorProgram: program} = machine
  const operationRows = describeOperations(program, machine.packedValues)
  const rows = Array.from({length: operationRows.rowCount}, (_, row) => operationRows.text(row))
  const state = describeState(machine.sourceCells, machine.packedValues)
  const referenceSource = Float32Array.from(machine.source)
  const footer = footerTrace(program)
  return {
    circuit: describeCircuit(circuit),
    program: describeProgram(circuit, program),
    sourceCells: machine.sourceCells.map(cell => [cell.name, cell.policy]),
    stateNames: Array.from({length: state.rowCount}, (_, row) => state.text(row)[0]),
    namedOutputs: machine.namedOutputs.map(output => [output.wire, output.name]),
    initialSource: f32bits(machine.source),
    referenceHash: hashF32(circuit.evaluateReference(referenceSource)),
    tensorHash: hashF32(circuit.evaluateTensor(referenceSource, program)),
    operationRowsHash: hashWords(rows.flatMap(([wire, kind, layer, weights]) => [Number(wire), ['affine', 'ReGLU', 'boolean', 'index read'].indexOf(kind), Number(layer), Number(weights)])),
    operationKinds: operationRows.kinds,
    measure: measureProgram(program),
    footer,
    wiring: wiringTrace(program),
    trace: windowManagerTrace(),
  }
}

// Monitor's footer with a stub element that records what the footer writes.
function footerTrace(program: TensorProgram) {
  const written: Record<string, string> = {}
  const element = (selector: string) => ({
    set textContent(value: string) { written[`${selector} text`] = value },
    setAttribute(name: string, value: string) { written[`${selector} ${name}`] = value },
  })
  const footerElement = {innerHTML: '', querySelector: (selector: string) => element(selector)}
  const footer = createMonitorFooter(footerElement as unknown as HTMLElement, program)
  const counts = footerElement.innerHTML.match(/<b>([^<]*)<\/b> (\w+)/g)
  const frames: unknown[] = []
  let seed = 7
  for (let frame = 0; frame < 90; frame++) {
    seed = (Math.imul(seed, 1103515245) + 12345) >>> 0
    const ms = frame === 40 ? 3.7 : (seed % 400) / 1000 + 0.05
    footer.update(ms)
    frames.push([f64hex(ms), written['.monitor-frame-time text'] ?? null, written['.monitor-chart .line d'], written['.monitor-chart .area d'], written['.monitor-chart .end d']])
  }
  const times = Array.from({length: 75}, (_, index) => (index * 37) % 113 + 0.25)
  return {
    counts,
    frames,
    averages: f64s(averageFrameTimes(times, 60, 10)),
    scales: [0, 99, 100, 101, 320, 999, 1000, 4999, 5001, 123456].map(value => frameTimeScale(value)),
    path: frameTimesPath([0.5, 1, 0.0625, 0.3, 2], 60, 1),
  }
}

function wiringTrace(program: TensorProgram) {
  const arcs: number[] = []
  let subtractsCount = 0
  const context = {
    lineWidth: 0, globalAlpha: 0, strokeStyle: '',
    setTransform() {},
    beginPath() {},
    ellipse(centerX: number, centerY: number, radiusX: number, radiusY: number) {
      arcs.push(context.strokeStyle === 'indigo' ? 1 : 0, centerX, centerY, radiusX, radiusY)
      if (context.strokeStyle === 'indigo') subtractsCount++
    },
    stroke() {},
  }
  const canvas = {width: 0, height: 0, getContext: () => context}
  const pictures = [[1232, 746, 2], [300, 100, 1.5]].map(([width, height, pixelRatio]) => {
    arcs.length = 0
    subtractsCount = 0
    drawWiring(canvas as unknown as HTMLCanvasElement, program, {width: width!, height: height!, pixelRatio: pixelRatio!, adds: 'copper', subtracts: 'indigo', alpha: 0.5, lineWidth: 0.5})
    return {canvas: [canvas.width, canvas.height], arcCount: arcs.length / 5, subtractsCount, arcsHash: hashF64(arcs)}
  })
  let weightCount = 0
  for (const layer of program.layers) weightCount += layer.weights.length
  return {pictures, opacity: f64hex(arcOpacity(weightCount))}
}

function photoGallery() {
  const machine = createPhotoGalleryMachine()
  const {circuit, tensorProgram: program} = machine
  const initialSource = f32bits(machine.source)
  const evaluations: unknown[] = []
  machine.observe({afterEvaluate(values, table) {
    evaluations.push([hashF32(values), hashF32(table)])
  }})
  const steps: unknown[] = []
  let frameTime = 0
  let values = new Float32Array(0)
  type Action = {kind: 'click', x: number, y: number, t: number | null} | {kind: 'previous' | 'next' | 'close', t: number | null} | {kind: 'resize', width: number, height: number, t: number | null}
  const step = (viewport: [number, number], actions: Action[], present = 1, interactive = 1, elapsed = 16) => {
    frameTime += elapsed
    for (const action of actions) {
      const t = action.t ?? undefined
      if (action.kind === 'click') machine.click(action.x, action.y, t)
      else if (action.kind === 'resize') machine.resize(action.width, action.height, t)
      else machine[action.kind](t)
    }
    evaluations.length = 0
    values = machine.step(viewport[0], viewport[1], present, interactive, frameTime)
    steps.push({
      viewport: f64s(viewport), present: f64hex(present), interactive: f64hex(interactive), frameTime: f64hex(frameTime),
      actions: actions.map(action => ({...action, ...('x' in action ? {x: f64hex(action.x), y: f64hex(action.y)} : {}),
        ...('width' in action ? {width: f64hex(action.width), height: f64hex(action.height)} : {}), t: action.t === null ? null : f64hex(action.t)})),
      evaluations: machine.evaluations.map(evaluation => [evaluation.kind, f64hex(evaluation.timeMs), evaluation.steps, evaluation.writesBack, evaluation.events]),
      evaluationHashes: [...evaluations],
      valuesHash: hashF32(values),
      sourceHash: hashF32(machine.source),
      photos: f32bits(machine.output.photos.map(wire => values[wire]!)),
    })
  }
  const viewport: [number, number] = [558, 384]
  const center = (item: number) => {
    const start = item * photoField.size
    return {x: values[machine.output.photos[start]!]! + values[machine.output.photos[start + 2]!]! / 2, y: values[machine.output.photos[start + 1]!]! + values[machine.output.photos[start + 3]!]! / 2}
  }
  step(viewport, [])
  let point = center(6)
  step(viewport, [{kind: 'click', x: point.x, y: point.y, t: frameTime + 8}])
  for (let frame = 0; frame < 6; frame++) step(viewport, [])
  step(viewport, [{kind: 'next', t: frameTime + 4}])
  step(viewport, [{kind: 'previous', t: frameTime + 4}, {kind: 'previous', t: frameTime + 9}])
  for (let frame = 0; frame < 30; frame++) step(viewport, [])
  // A click on a thumbnail, and a resize during the motion.
  const thumbnail = (item: number) => {
    const start = item * 6
    return {x: values[machine.output.thumbnails[start]!]! + values[machine.output.thumbnails[start + 2]!]! / 2, y: values[machine.output.thumbnails[start + 1]!]! + values[machine.output.thumbnails[start + 3]!]! / 2}
  }
  point = thumbnail(7)
  step(viewport, [{kind: 'click', x: point.x, y: point.y, t: frameTime + 6}])
  step([700, 500], [])
  step([700, 500], [{kind: 'close', t: frameTime + 3}])
  point = center(1)
  step([700, 500], [{kind: 'click', x: point.x, y: point.y, t: frameTime + 10}], 0.5)
  for (let frame = 0; frame < 10; frame++) step([700, 500], [], frame < 5 ? 0.25 : 1, frame % 3 === 0 ? 0 : 1)
  step([700, 500], [{kind: 'close', t: null}])
  step([700, 500], [{kind: 'click', x: 2, y: 2, t: frameTime + 2}], 1, 1, 900)
  for (let frame = 0; frame < 60; frame++) step([300, 200], [])
  step([300, 200], [{kind: 'resize', width: 33, height: 339, t: frameTime + 5}])
  for (let frame = 0; frame < 20; frame++) step([33, 339], [])
  return {
    circuit: describeCircuit(circuit),
    program: describeProgram(circuit, program),
    initialSource,
    powersHash: hashF64(springMatrixPowers.flat()),
    trace: steps,
  }
}

function valueList() {
  const {circuit, cells, output, nextSource, initialSource, validate} = createValueListCircuit()
  const roots = [...nextSource, output.totalHeight, output.scroll, output.scrollTarget, output.scrollRequest, output.slotOverflow, output.selected, ...output.locations, ...output.slots]
  const program = circuit.compileTensorProgram(roots)
  const evaluate = circuit.createTensorProgramEvaluator(program)
  const source = Float32Array.from(initialSource)
  source[cells.valueList.rowCounts[0]!] = 234
  source[cells.valueList.rowCounts[1]!] = 4646
  source[cells.valueList.rowCounts[2]!] = 777
  const frames: unknown[] = []
  type Input = {viewportHeight?: number, focused?: number, click?: number, location?: number, key?: number, scrollTop?: number}
  const script: Input[] = [
    {}, {key: 1}, {key: 1}, {key: 3}, {key: 5}, {scrollTop: 1000.5}, {key: 4}, {click: 100.25}, {click: 3000}, {location: 0},
    {scrollTop: 24 * 200}, {click: -20}, {focused: 0, key: 1}, {viewportHeight: 1600.5}, {key: 3}, {location: 2, key: 0},
    {viewportHeight: 30}, {key: 2}, {location: 1}, {scrollTop: 2_399_000}, {key: 5}, {click: 700}, {viewportHeight: 0, key: 1},
  ]
  for (const input of script) {
    source[cells.viewportHeight] = input.viewportHeight ?? source[cells.viewportHeight]!
    source[cells.focused] = input.focused ?? 1
    source[cells.listClick.active] = input.click === undefined ? 0 : 1
    source[cells.listClick.y] = input.click ?? 0
    cells.locationClicks.forEach((wire, index) => {
      source[wire] = input.location === index ? 1 : 0
    })
    Object.values(cells.valueList.keys).forEach((wire, index) => {
      source[wire] = input.key === index ? 1 : 0
    })
    if (input.scrollTop !== undefined) source[cells.valueList.scrollTop] = input.scrollTop
    validate(source)
    const rootValues = evaluate(source)
    const next = rootValues.subarray(0, nextSource.length)
    frames.push({source: hashF32(source), roots: hashF32(rootValues), outputs: f32bits(rootValues.subarray(nextSource.length, nextSource.length + 9)), reference: hashF32(circuit.evaluateReference(source))})
    source.set(next)
  }
  return {circuit: describeCircuit(circuit), program: describeProgram(circuit, program), initialSource: f32bits(initialSource), frames}
}

function arrangement() {
  const minimum = {width: 280, height: 190}
  const natural = [{width: 540, height: 340}, {width: 500, height: 366}, {width: 560, height: 430}]
  const metrics = {titleHeight: 44, photosGap: 18, maximizedMargin: 12}
  const results: string[][] = []
  for (let width = 120; width <= 2400; width += 97.5) {
    for (let height = 150; height <= 1300; height += 71.25) {
      const workspace = {left: 0, top: 36, width, height: height - 112}
      for (const rectangle of arrangeWindows(workspace, minimum, natural, metrics)) results.push(f64s([rectangle.left, rectangle.top, rectangle.width, rectangle.height]))
    }
  }
  return results
}

function noteFocus() {
  return Array.from({length: 64}, (_, bits) => noteFocusChange({
    notesIsFocusedWindow: (bits & 1) !== 0, noteHasKeyboardFocus: (bits & 2) !== 0, noteHasHadKeyboardFocus: (bits & 4) !== 0,
    pressedOrReleased: (bits & 8) !== 0, lastPress: {byMouse: (bits & 16) !== 0, outsideEveryWindow: (bits & 32) !== 0},
  }))
}

// ---------------------------------------------------------------------------
// JSON search.

const units = (text: string) => Array.from({length: text.length}, (_, index) => text.charCodeAt(index))

function jsonTexts() {
  const documents = generateDocuments(48, 20261013, 0.35).map(document => document.text)
  const deep = (depth: number) => '['.repeat(depth) + '1' + ']'.repeat(depth)
  const deepObject = (depth: number) => '{"a":'.repeat(depth) + 'null' + '}'.repeat(depth)
  const edge = [
    '', ' ', '[1 2]', '[1', '{"a" 2}', '{1: 2}', '[1}', '01', '-x', '1.', '1e', 'tru', 'nul!', '"\\q"', '"\\u12x4"', '"a\tb"',
    '[x]\n[1,2]', '{"id": 1, "name": "ada"}', '"café 😀"', '"\ud800"', '\ud83d', deep(16), deep(17), deepObject(16),
    deepObject(17), '"' + 'x'.repeat(509) + '"', '"' + 'x'.repeat(510) + '"', '"' + 'x'.repeat(511) + '"', '[' + '1,'.repeat(300) + '1]',
    '1 1', '1]', '[1,]', '[:]', '{"a":1,}', '{"a":[1,{"b":"c"}],"d":true}', '  \r\n{" ": "\\n"}',
  ]
  return [...edge, ...documents]
}

const allTables = {
  classes: classTable, lexerNext: lexerNextTable, lexerEvent: lexerEventTable, grammarToken: grammarTokenTable,
  lexerErrorKind: lexerErrorKindTable, depthChange: depthChangeTable, objectWeight: objectWeightTable, levelWeight: levelWeightTable,
  depthClass: depthClassTable, grammarNext: grammarNextTable, grammarError: grammarErrorTable, display: displayTable,
}

function describeLoopProgram(loop: LoopProgram) {
  const wgsl = emitLoopWgsl(loop)
  const layered = emitLoopWgsl(loop, {order: 'layers', workgroupSize: 32})
  return {
    stages: loop.stages.map(stage => ({
      name: stage.name, repeat: stage.repeat, sharedOffset: stage.sharedOffset, bindings: [...stage.bindings].map(([wire, binding]) => [wire, binding]),
      outputs: stage.outputs, program: describeProgram(stage.circuit, stage.program),
    })),
    describe: describeLoop(loop),
    operationsPerDocument: operationsPerDocument(loop),
    sharedValues: hashF32(loop.sharedValues),
    sharedLength: loop.sharedValues.length,
    wgsl: {codeHash: hashWords(units(wgsl.code)), statementCount: wgsl.statementCount, bytes: wgsl.bytes, rowCounts: wgsl.rowCounts, outputCounts: wgsl.outputCounts, windowedGroups: wgsl.windowedGroups, inputCells: wgsl.inputCells},
    layeredWgsl: {codeHash: hashWords(units(layered.code)), statementCount: layered.statementCount, bytes: layered.bytes},
  }
}

function jsonHighlighter(texts: string[]) {
  const {loop, circuit} = buildJsonLoop()
  const runner = createLoopRunner(loop, 'executor')
  const cells = new Float32Array(texts.length * cellsPerDocument)
  texts.forEach((text, index) => writeDocumentCells(cells, index * cellsPerDocument, text))
  const summary = new Float32Array(texts.length * 4)
  const kinds = new Float32Array(texts.length * 512)
  texts.forEach((_, index) => runner.run([cells], index, [summary, kinds], index, index))
  const tableKinds = new Uint8Array(512)
  const tableLoop = texts.map(text => {
    highlightWithTables(text, tableKinds, 0)
    return [tableLoopResult.firstErrorPosition, tableLoopResult.firstErrorKind, tableLoopResult.truncated, hashWords(Array.from(tableKinds))]
  })
  return {
    tables: Object.fromEntries(Object.entries(allTables).map(([name, values]) => [name, values])),
    circuit: describeCircuit(circuit),
    loop: describeLoopProgram(loop),
    cellsHash: hashF32(cells),
    documents: texts.map((_, index) => ({
      summary: f32bits(summary.subarray(index * 4, index * 4 + 4)),
      kindsHash: hashF32(kinds.subarray(index * 512, index * 512 + 512)),
    })),
    tableLoop,
    kinds,
  }
}

const patterns = [
  'error', '\\d\\d\\d\\d-\\d\\d-\\d\\d', '(the|an?) [a-z]+', 'a|ab', 'a*', '[^0-9]+', '.', 'connection reset', '"[a-z]+"',
  '[a-z_]+\\.[a-z]+', '(([a-z]+)+)+x', '[]', '[^]', 'é|😀', '[-a]|[a-]', '\\w+@example\\.com', 'true|false|null',
  '(ab)*c', '[\\d.]+e?', 'x?y?z?', '\\[|\\]|\\{|\\}',
  // Adjacent intervals that the class parser merges.
  '[a-cd]+', '[0-45-9]x', '[^a-bc-d]', '[\\w:-@]',
]
const badPatterns = ['a**', '(?:a)', '^a', 'a{2}', '[z-a]', '(', ')', 'a\\', '[\\d-z]', 'a*?', '[abc', '\\q', '*', ']', 'a{', '$']

function compiledDescription(pattern: string) {
  try {
    const compiled = compilePattern(pattern)
    return {
      automaton: {stateCount: compiled.automaton.stateCount, targets: compiled.automaton.targets, stateSets: compiled.automaton.stateSets, accepting: compiled.automaton.accepting},
      partition: compiled.partition,
      capacityErrors: [8, 16, 32].map(states => capacityErrors(compiled, {states, classes: 16, boundaries: 32}).map(error => error.message)),
    }
  } catch (error) {
    if (!(error instanceof PatternSyntaxError)) throw error
    return {error: error.message, position: error.position}
  }
}

const scopes: SearchScope[] = ['everything', 'strings', 'keys', 'stringsAndKeys', [4, 5]]

function search(texts: string[], kinds: Float32Array) {
  const configurations = [
    {states: 8, masked: false, stopAtLength: false, bitFields: true, patterns: 21, documents: 24},
    {states: 8, masked: true, stopAtLength: true, bitFields: true, patterns: 21, documents: 24},
    {states: 16, masked: true, stopAtLength: false, bitFields: false, patterns: 8, documents: 12},
    {states: 32, masked: false, stopAtLength: true, bitFields: true, patterns: 4, documents: 8},
  ]
  const cells = new Float32Array(texts.length * cellsPerDocument)
  texts.forEach((text, index) => writeDocumentCells(cells, index * cellsPerDocument, text))
  return configurations.map(configuration => {
    const capacity = {states: configuration.states, classes: 16, boundaries: 32}
    const built = buildSearchLoop(capacity, {positions: 512, cellsPerDocument, masked: configuration.masked, stopAtLength: configuration.stopAtLength, bitFields: configuration.bitFields})
    const runner = createLoopRunner(built.loop, 'executor')
    const runs: unknown[] = []
    for (const pattern of patterns.slice(0, configuration.patterns)) {
      const compiled = compilePattern(pattern)
      if (capacityErrors(compiled, capacity).length > 0) {
        const shared = Float32Array.from(built.loop.sharedValues)
        let message = ''
        try {
          built.writePattern(shared, compiled)
        } catch (error) {
          message = (error as Error).message
        }
        runs.push({pattern, error: message})
        continue
      }
      for (const scope of configuration.masked ? scopes.slice(0, pattern.length % 2 === 0 ? 5 : 2) : [null]) {
        const shared = Float32Array.from(built.loop.sharedValues)
        built.writePattern(shared, compiled)
        if (scope !== null) built.writeScope(shared, scope)
        runner.setShared(shared)
        const count = new Float32Array(configuration.documents)
        const marks = new Float32Array(configuration.documents * 512)
        for (let document = 0; document < configuration.documents; document++) {
          const buffers = configuration.masked ? [cells, kinds] : [cells]
          runner.run(buffers, document, [count, marks], document, document)
        }
        runs.push({pattern, scope, shared: hashF32(shared), counts: f32bits(count), marks: Array.from({length: configuration.documents}, (_, document) => hashF32(marks.subarray(document * 512, document * 512 + 512)))})
      }
    }
    return {configuration, circuit: describeCircuit(built.loop.stages[0]!.circuit), loop: describeLoopProgram(built.loop), patternCells: built.patternCells, runs}
  })
}

function jsonSearch() {
  const texts = jsonTexts()
  const highlighter = jsonHighlighter(texts)
  const {kinds, ...recorded} = highlighter
  return {
    texts: texts.map(units),
    highlighter: recorded,
    patterns: [...patterns, ...badPatterns].map(pattern => ({pattern: units(pattern), compiled: compiledDescription(pattern)})),
    search: search(texts, kinds),
  }
}

export function demoFixtures() {
  return {
    runtime: runtimeHelpers(),
    windowManager: windowManager(),
    photoGallery: photoGallery(),
    valueList: valueList(),
    arrangement: arrangement(),
    noteFocus: noteFocus(),
    formatValues: [0.1, 1 / 3, 2399976, -0, Math.fround(0.7), 1e-7, 123456.789, Number.NaN, 2 ** 70, -5.5].map(value => [f64hex(Math.fround(value)), formatValue(Math.fround(value))]),
    jsonSearch: jsonSearch(),
  }
}
