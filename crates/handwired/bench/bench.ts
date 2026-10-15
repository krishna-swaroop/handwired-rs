// The TypeScript side of the TypeScript-versus-Rust benchmark. Runs the same
// workloads as examples/bench.rs on the TypeScript originals and writes their
// timings and output hashes to bench/results-ts.json. compare.ts prints both.
//
// node crates/handwired/bench/bench.ts

import {writeFileSync, readFileSync} from 'node:fs'
import {createWindowManagerMachine} from '../../../src/handwiredos/window-manager.ts'
import {arrangeInitialWindows} from '../../../src/handwiredos/arrangement.ts'
import {buildJsonLoop} from '../../../src/json-search/json-step.ts'
import {buildSearchLoop} from '../../../src/json-search/search-step.ts'
import {createLoopRunner} from '../../../src/json-search/loop-program.ts'
import {compilePattern} from '../../../src/json-search/regex/pattern.ts'
import {cellsPerDocument, writeDocumentCells} from '../../../src/json-search/program.ts'

const here = new URL('.', import.meta.url)
const quick = process.argv.includes('--quick')

// FNV-1a over the FP32 bits, every NaN as one NaN, as in conformance/demos.ts.
function hashF32(values: ArrayLike<number>, hash = 0x811c9dc5): number {
  const bits = new Uint32Array(Float32Array.from(values).buffer)
  for (let word of bits) {
    if ((word & 0x7f800000) === 0x7f800000 && (word & 0x7fffff) !== 0) word = 0x7fc00000
    for (let shift = 0; shift < 32; shift += 8) {
      hash ^= (word >>> shift) & 0xff
      hash = Math.imul(hash, 0x01000193) >>> 0
    }
  }
  return hash
}

type Result = {name: string, unit: string, medianNs: number, minNs: number, iterations: number, hash: number | null}
const results: Result[] = []
let sink = 0

// Runs `run` in batches sized to about `targetMs / samples` each, after a
// warmup, and records the median and fastest time per call.
function measure(name: string, unit: string, run: () => unknown, hash: number | null = null, targetMs = quick ? 300 : 2000) {
  const warmupEnd = performance.now() + targetMs / 4
  let calls = 0
  while (performance.now() < warmupEnd || calls < 1) {
    sink += Number(Boolean(run()))
    calls++
  }
  const perCall = (performance.now() - warmupEnd + targetMs / 4) / calls
  // Calls slower than a sample's share run 5 samples of one call.
  const samples = perCall > targetMs / 15 ? 5 : 15
  const batch = Math.max(1, Math.round(targetMs / samples / perCall))
  const times: number[] = []
  for (let sample = 0; sample < samples; sample++) {
    const start = performance.now()
    for (let call = 0; call < batch; call++) sink += Number(Boolean(run()))
    times.push((performance.now() - start) * 1e6 / batch)
  }
  times.sort((a, b) => a - b)
  const result = {name, unit, medianNs: times[samples >> 1]!, minNs: times[0]!, iterations: batch * samples, hash}
  results.push(result)
  console.log(`${name.padEnd(30)} ${(result.medianNs / 1e3).toFixed(2).padStart(12)} µs/${unit}`)
}

// ---------------------------------------------------------------------------
// HandwiredOS: the window manager, which includes Photos and Monitor.

type Machine = ReturnType<typeof createWindowManagerMachine>

measure('wm.build', 'machine', () => createWindowManagerMachine())

const machine = createWindowManagerMachine()
const roots = [...machine.output.nextSource, ...machine.namedOutputs.map(({wire}) => wire)]
const source = Float32Array.from(machine.source)
measure('wm.compile', 'program', () => machine.circuit.compileTensorProgram(roots))

const evaluate = machine.circuit.createTensorProgramEvaluator(machine.tensorProgram)
measure('wm.evaluate', 'frame', () => evaluate(source), hashF32(evaluate(source)))
measure('wm.reference', 'frame', () => machine.circuit.evaluateReference(source), hashF32(machine.circuit.evaluateReference(source)))

// A 60-frame cycle: drag window 0 by its title bar out and back, press a value
// list key, and let the springs settle. Every cycle ends where it started.
function sessionFrame(target: Machine, frame: number, frameTime: number, grab: {x: number, y: number}) {
  const phase = frame % 60
  const t = frameTime + 8
  if (phase === 0) target.pointerDown(grab.x, grab.y, t)
  else if (phase <= 20) target.pointerMove(grab.x + phase * 5.5, grab.y + phase * 2.25, t)
  else if (phase <= 39) target.pointerMove(grab.x + (40 - phase) * 5.5, grab.y + (40 - phase) * 2.25, t)
  else if (phase === 40) target.pointerUp(grab.x, grab.y, t)
  else if (phase === 50) target.valueListKey(frame % 120 === 50 ? 'down' : 'up', t)
  return target.step(1280, 800, frameTime + 16)
}

function startSession() {
  const target = createWindowManagerMachine()
  const [first] = arrangeInitialWindows(target.source, target.cells, 1280, 800)
  return {target, grab: {x: first!.left + 60.3, y: first!.top + 20}, frame: 0, frameTime: 0}
}

const verify = startSession()
let sessionHash = 0x811c9dc5
for (let frame = 0; frame < 600; frame++) {
  sessionHash = hashF32(sessionFrame(verify.target, frame, frame * 16, verify.grab), sessionHash)
}
const session = startSession()
measure('wm.session', 'frame', () => {
  const values = sessionFrame(session.target, session.frame, session.frameTime, session.grab)
  session.frame++
  session.frameTime += 16
  return values
}, sessionHash)

// ---------------------------------------------------------------------------
// JSON search over the 84 texts the demo conformance records.

const demos = JSON.parse(readFileSync(new URL('../conformance/demos.json', here), 'utf8'))
const texts: string[] = demos.jsonSearch.texts.map((units: number[]) => String.fromCharCode(...units))
const cells = new Float32Array(texts.length * cellsPerDocument)
texts.forEach((text, index) => writeDocumentCells(cells, index * cellsPerDocument, text))

measure('json.highlight.build', 'loop', () => buildJsonLoop())
const {loop: jsonLoop} = buildJsonLoop()
const summary = new Float32Array(texts.length * 4)
const kinds = new Float32Array(texts.length * 512)
for (const backend of ['executor', 'generated'] as const) {
  const runner = createLoopRunner(jsonLoop, backend)
  const highlight = () => {
    for (let index = 0; index < texts.length; index++) runner.run([cells], index, [summary, kinds], index, index)
    return kinds
  }
  highlight()
  measure(`json.highlight.${backend}`, `${texts.length} docs`, highlight, hashF32(kinds, hashF32(summary)))
}

const capacity = {states: 16, classes: 16, boundaries: 32}
const searchOptions = {positions: 512, cellsPerDocument, masked: false, stopAtLength: false, bitFields: true}
measure('json.search.build', 'loop', () => buildSearchLoop(capacity, searchOptions))
const built = buildSearchLoop(capacity, searchOptions)
const shared = ['error', '\\d\\d\\d\\d-\\d\\d-\\d\\d', '"[a-z]+"'].map(pattern => {
  const values = Float32Array.from(built.loop.sharedValues)
  built.writePattern(values, compilePattern(pattern))
  return values
})
const counts = new Float32Array(texts.length)
const marks = new Float32Array(texts.length * 512)
for (const backend of ['executor', 'generated'] as const) {
  const runner = createLoopRunner(built.loop, backend)
  const search = (hashed: boolean) => {
    let hash = 0x811c9dc5
    for (const values of shared) {
      runner.setShared(values)
      for (let document = 0; document < texts.length; document++) runner.run([cells], document, [counts, marks], document, document)
      if (hashed) hash = hashF32(marks, hashF32(counts, hash))
    }
    return hash
  }
  measure(`json.search.${backend}`, `3 patterns × ${texts.length} docs`, () => search(false), search(true))
}

writeFileSync(new URL('results-ts.json', here), JSON.stringify({runtime: `node ${process.version}`, results}, null, 2) + '\n')
if (sink < 0) console.log(sink)
