import { defaultSettings, staggeredExample } from "./example.ts"
import type { EngineSettings, SampleDraft, ScenarioDraft } from "./types.ts"

export const LEARNING_EXAMPLES = [
  { id: "staggered", label: "Staggered arrivals", description: "Six requests arrive at 0, 100 and 120 ms. Observe changing batch membership." },
  { id: "prefix", label: "Partial prefix reuse", description: "Two prompts share a long prefix and differ at the end. Prefix caching is enabled; inspect the recorded hits and remaining prefill." },
  { id: "chunked", label: "Chunked prefill", description: "A 512-token input shares a 64-token scheduling budget with a short request. Inspect which positions each step schedules." },
  { id: "pressure", label: "Cache pressure", description: "Four 192-token inputs compete for a 72 MiB KV budget. Actual pressure depends on model geometry; inspect capacity and any observed preemption." },
] as const
export type LearningExampleId = typeof LEARNING_EXAMPLES[number]["id"]

export function learningExample(id: LearningExampleId, settings: EngineSettings): { scenario: ScenarioDraft; settings: EngineSettings } {
  if (id === "staggered") {
    const defaults = defaultSettings()
    return { scenario: staggeredExample(), settings: { ...settings, prefixCaching: false, chunkedPrefill: true, maxModelLen: defaults.maxModelLen, tokenBudget: defaults.tokenBudget, maxNumSeqs: defaults.maxNumSeqs, kvCacheMemoryBytes: "", schedulingPolicy: "fcfs", longPrefillTokenThreshold: "0" } }
  }
  const synthetic = (name: string, length: number): SampleDraft => ({
    id: `sample-${id}-${name}`, label: name, prompt: "", inputMode: "synthetic",
    syntheticInputTokens: length, resolvedInputTokens: null, maxOutputTokens: id === "pressure" ? 32 : 8,
    adapterId: null, priority: 0,
  })
  let samples: SampleDraft[]
  let overrides: Partial<EngineSettings>
  if (id === "prefix") {
    const prefix = "Read this shared reference before answering. " + "The explorer records requests, scheduler steps and cache blocks. ".repeat(24)
    samples = ["Name the three things recorded.", "Explain why cache blocks are useful."].map((suffix, index) => ({
      ...synthetic(`prefix-${index + 1}`, 128), inputMode: "prompt", prompt: prefix + suffix,
    }))
    overrides = { prefixCaching: true, chunkedPrefill: true, maxModelLen: "4096", tokenBudget: "2048", maxNumSeqs: "4", kvCacheMemoryBytes: "", schedulingPolicy: "fcfs", longPrefillTokenThreshold: "0" }
  } else if (id === "chunked") {
    samples = [synthetic("long-input", 512), synthetic("short-input", 16)]
    overrides = { chunkedPrefill: true, prefixCaching: false, maxModelLen: "1024", tokenBudget: "64", maxNumSeqs: "4", kvCacheMemoryBytes: "", schedulingPolicy: "fcfs", longPrefillTokenThreshold: "64" }
  } else {
    samples = Array.from({ length: 4 }, (_, index) => synthetic(`pressure-${index + 1}`, 192))
    overrides = { chunkedPrefill: true, prefixCaching: false, maxModelLen: "256", tokenBudget: "256", maxNumSeqs: "4", kvCacheMemoryBytes: String(72 * 1024 * 1024), schedulingPolicy: "fcfs", longPrefillTokenThreshold: "0" }
  }
  const batches = id === "prefix" ? samples.map((sample, index) => ({
    id: `batch-${id}-${index}`, label: index === 0 ? "Prime prefix" : "Reuse prefix", arrivalOffsetMs: index * 1000,
    createdOrder: index, defaultAdapterId: null, samples: [sample],
  })) : [{ id: `batch-${id}`, label: LEARNING_EXAMPLES.find((example) => example.id === id)!.label, arrivalOffsetMs: 0, createdOrder: 0, defaultAdapterId: null, samples }]
  return { scenario: { version: 1, id: `scenario-${id}`, name: LEARNING_EXAMPLES.find((example) => example.id === id)!.label, nextOrder: batches.length, batches }, settings: { ...settings, ...overrides } }
}
