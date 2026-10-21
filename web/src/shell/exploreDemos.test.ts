import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"

import type { CatalogEntry } from "../api.ts"
import { recordingConfiguration } from "./demoCatalog.ts"
import {
  CURATED_DEMO_IDS, curatedDemos, demoOptions, demoSelection, initialDemo, INITIAL_DEMO_SELECTION,
  LAST_DEMO_STORAGE_KEY, normalizeDemoSelection, rememberDemo, resolveDemo,
} from "./exploreDemos.ts"

const catalog: CatalogEntry[] = JSON.parse(readFileSync(new URL("../../public/recordings/index.json", import.meta.url), "utf8")).recordings
  .map((entry: CatalogEntry) => ({ ...entry, configuration: recordingConfiguration(JSON.parse(readFileSync(new URL(`../../public/recordings/${entry.id}/manifest.json`, import.meta.url), "utf8"))) }))

test("only the six current, observed demos appear; archived and unobserved entries stay outside the chooser", () => {
  assert.deepEqual(curatedDemos(catalog).map((entry) => entry.id), CURATED_DEMO_IDS)
  const [baseline] = curatedDemos(catalog)
  assert.deepEqual(curatedDemos([{ ...baseline, configuration: undefined }]), [])
  assert.deepEqual(curatedDemos([{ ...baseline, evidence: "synthetic" }]), [])
  assert.deepEqual(curatedDemos([{ ...baseline, configuration: { ...baseline.configuration!, weightFormat: "unknown" } }]), [])
  for (const entry of curatedDemos(catalog)) assert.equal(resolveDemo(catalog, demoSelection(entry))?.id, entry.id)
  assert.equal(resolveDemo(catalog, { ...INITIAL_DEMO_SELECTION, weightFormat: "any" }), undefined)
  assert.equal(resolveDemo(catalog, { ...INITIAL_DEMO_SELECTION, kvFormat: "fp8_e4m3", loraMode: "mixed" }), undefined)
})

test("every recorded demo is reachable through cascading selectors without exposing uncaptured combinations", () => {
  for (const entry of curatedDemos(catalog)) {
    const target = demoSelection(entry)
    let selection = { ...INITIAL_DEMO_SELECTION }
    for (const key of ["modelId", "weightFormat", "kvFormat", "loraMode"] as const) {
      assert.ok(demoOptions(catalog, selection, key).includes(target[key]), `${entry.id}: ${key} must be reachable`)
      selection = normalizeDemoSelection(catalog, { ...selection, [key]: target[key] }, key)
      assert.ok(resolveDemo(catalog, selection), `${entry.id}: ${key} must keep a recorded combination`)
    }
    assert.equal(resolveDemo(catalog, selection)?.id, entry.id)
  }
})

test("changing models remains possible from FP8 and LoRA configurations and normalizes incompatible later facets", () => {
  const modelId = "Qwen/Qwen3.8-27B"
  for (const id of ["gpu-h100-realistic-fp8-weights", "gpu-h100-realistic-fp8-kv", "gpu-h100-realistic-lora-mixed"]) {
    const previous = demoSelection(catalog.find((entry) => entry.id === id)!)
    assert.ok(demoOptions(catalog, previous, "modelId").includes(modelId))
    const next = normalizeDemoSelection(catalog, { ...previous, modelId }, "modelId")
    assert.equal(resolveDemo(catalog, next)?.id, "gpu-h100-realistic-qwen38-27b")
    assert.deepEqual(previous, demoSelection(catalog.find((entry) => entry.id === id)!))
  }
  assert.deepEqual(demoOptions(catalog, { ...INITIAL_DEMO_SELECTION, weightFormat: "fp8" }, "kvFormat"), ["bf16"])
  assert.deepEqual(demoOptions(catalog, { ...INITIAL_DEMO_SELECTION, kvFormat: "fp8_e4m3" }, "loraMode"), ["none"])
  assert.deepEqual(normalizeDemoSelection([], INITIAL_DEMO_SELECTION, "modelId"), INITIAL_DEMO_SELECTION)
})

test("startup restores only an available current demo and tolerates unavailable browser storage", () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "localStorage")
  const saved = new Map<string, string>()
  try {
    Object.defineProperty(globalThis, "localStorage", { configurable: true, value: { getItem: (key: string) => saved.get(key) ?? null, setItem: (key: string, value: string) => saved.set(key, value) } })
    assert.equal(initialDemo(catalog)?.id, CURATED_DEMO_IDS[0])
    rememberDemo("gpu-h100-realistic-lora-mixed")
    assert.equal(initialDemo(catalog)?.id, "gpu-h100-realistic-lora-mixed")
    rememberDemo("gpu-kv-layer-diagnostics")
    assert.equal(initialDemo(catalog)?.id, "gpu-h100-realistic-lora-mixed")
    saved.set(LAST_DEMO_STORAGE_KEY, "gpu-h100-lora-mixed")
    assert.equal(initialDemo(catalog)?.id, CURATED_DEMO_IDS[0])
    saved.set(LAST_DEMO_STORAGE_KEY, "gpu-h100-realistic-lora-mixed")
    assert.equal(initialDemo(catalog.filter((entry) => entry.id !== "gpu-h100-realistic-lora-mixed"))?.id, CURATED_DEMO_IDS[0])
    Object.defineProperty(globalThis, "localStorage", { configurable: true, get() { throw new Error("Storage unavailable") } })
    assert.equal(initialDemo(catalog)?.id, CURATED_DEMO_IDS[0])
    assert.doesNotThrow(() => rememberDemo("gpu-h100-realistic-lora-mixed"))
    assert.equal(initialDemo([]), undefined)
  } finally {
    if (descriptor) Object.defineProperty(globalThis, "localStorage", descriptor)
    else Reflect.deleteProperty(globalThis, "localStorage")
  }
})
