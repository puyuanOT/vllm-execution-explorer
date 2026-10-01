import assert from "node:assert/strict"
import { test } from "node:test"

import { defaultSettings, staggeredExample } from "./example.ts"
import {
  loadScenario,
  loadSettings,
  saveScenario,
  saveSettings,
  SCENARIO_STORAGE_KEY,
  SETTINGS_STORAGE_KEY,
} from "./storage.ts"

function withStorage(run: (values: Map<string, string>) => void) {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage")
  const values = new Map<string, string>()
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
      removeItem: (key: string) => values.delete(key),
    },
  })
  try {
    run(values)
  } finally {
    if (original) Object.defineProperty(globalThis, "localStorage", original)
    else Reflect.deleteProperty(globalThis, "localStorage")
  }
}

test("malformed stored batches and samples recover without a startup exception", () => {
  withStorage((values) => {
    const example = staggeredExample()
    const malformed = [
      { ...example, batches: [null] },
      { ...example, batches: [{}] },
      { ...example, batches: [{ ...example.batches[0], samples: [null] }] },
      { ...example, batches: [{ ...example.batches[0], samples: [{}] }] },
      { ...example, batches: [{ ...example.batches[0], label: null }] },
      { ...example, nextOrder: null },
    ]
    for (const scenario of malformed) {
      values.set(SCENARIO_STORAGE_KEY, JSON.stringify({ version: 1, scenario }))
      const loaded = loadScenario()
      assert.deepEqual(loaded.value, example)
      assert.notEqual(loaded.error, "")
    }
  })
})

test("valid incomplete drafts remain editable after reload", () => {
  withStorage(() => {
    const scenario = staggeredExample()
    scenario.batches[0].label = ""
    scenario.batches[0].samples[0].prompt = ""
    scenario.batches[0].samples[0].maxOutputTokens = 0
    assert.equal(saveScenario(scenario).error, "")
    assert.deepEqual(loadScenario().value, scenario)
    assert.equal(loadScenario().error, "")
    const settings = { ...defaultSettings(), model: "", tokenBudget: "" }
    saveSettings(settings)
    assert.deepEqual(loadSettings(), settings)
  })
})

test("older stored drafts still gain input-mode and settings defaults", () => {
  withStorage((values) => {
    const scenario = staggeredExample()
    const legacy = JSON.parse(JSON.stringify(scenario))
    delete legacy.batches[0].defaultAdapterId
    for (const sample of legacy.batches[0].samples) {
      delete sample.inputMode
      delete sample.syntheticInputTokens
      delete sample.resolvedInputTokens
      delete sample.adapterId
    }
    values.set(SCENARIO_STORAGE_KEY, JSON.stringify({ version: 1, scenario: legacy }))
    assert.deepEqual(loadScenario().value, scenario)
    assert.equal(loadScenario().error, "")
    values.set(SETTINGS_STORAGE_KEY, JSON.stringify({ version: 1, settings: { engine: "fixture" } }))
    assert.deepEqual(loadSettings(), { ...defaultSettings(), engine: "fixture" })
  })
})

test("malformed persisted settings cannot reach string or adapter controls", () => {
  withStorage((values) => {
    for (const patch of [
      { runnerUrl: null },
      { model: {} },
      { seed: 0 },
      { automaticSetup: "false" },
      { hostKind: "unsupported" },
      { stopStrings: [null] },
      { loraAdapters: [null] },
      { loraAdapters: [{ id: "a", name: "A", source: "host", location: "/adapter", revision: null }] },
    ]) {
      const settings = { ...defaultSettings(), engine: "fixture", ...patch }
      values.set(SETTINGS_STORAGE_KEY, JSON.stringify({ version: 1, settings }))
      assert.deepEqual(loadSettings(), defaultSettings())
    }
  })
})
