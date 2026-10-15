// Prints bench/results-ts.json and bench/results-rust.json side by side, and
// fails when a workload's outputs differ between the two.
//
// node crates/handwired/bench/compare.ts

import {readFileSync} from 'node:fs'

type Result = {name: string, unit: string, medianNs: number, hash: number | null}
type Report = {runtime: string, results: Result[]}

const here = new URL('.', import.meta.url)
const read = (file: string): Report => JSON.parse(readFileSync(new URL(file, here), 'utf8'))
const ts = read('results-ts.json')
const rust = read('results-rust.json')
const rustByName = new Map(rust.results.map(result => [result.name, result]))

const format = (ns: number) => ns >= 1e6 ? `${(ns / 1e6).toFixed(1)} ms` : `${(ns / 1e3).toFixed(1)} µs`
const rows = [['workload', 'per', `TS (${ts.runtime})`, `Rust`, 'Rust speedup', 'outputs']]
let mismatches = 0
for (const result of ts.results) {
  const backend = result.name.split('.').at(-1)
  const counterpart = rustByName.get(result.name) ?? (backend === 'generated' ? rustByName.get(result.name.replace(/generated$/, 'executor')) : undefined)
  if (counterpart === undefined) continue
  let outputs = '—'
  if (result.hash !== null && counterpart.hash !== null) {
    outputs = result.hash === counterpart.hash ? 'identical' : 'DIFFER'
    if (outputs === 'DIFFER') mismatches++
  }
  rows.push([result.name, result.unit, format(result.medianNs), format(counterpart.medianNs), `${(result.medianNs / counterpart.medianNs).toFixed(1)}×`, outputs])
}
const widths = rows[0]!.map((_, column) => Math.max(...rows.map(row => row[column]!.length)))
for (const row of rows) console.log(row.map((cell, column) => column >= 2 && column <= 4 ? cell.padStart(widths[column]!) : cell.padEnd(widths[column]!)).join('  '))
if (mismatches > 0) {
  console.error(`${mismatches} workload(s) computed different outputs`)
  process.exit(1)
}
