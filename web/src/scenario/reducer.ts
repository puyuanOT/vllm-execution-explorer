import { createId } from "./ids.ts"
import type { ArrivalBatchDraft, SampleDraft, ScenarioDraft } from "./types.ts"

export type ScenarioAction =
  | { type: "replace"; scenario: ScenarioDraft }
  | { type: "rename"; name: string }
  | { type: "add-batch" }
  | { type: "duplicate-batch"; batchId: string }
  | { type: "delete-batch"; batchId: string }
  | { type: "set-batch-label"; batchId: string; label: string }
  | { type: "commit-batch-time"; batchId: string; arrivalOffsetMs: number }
  | { type: "set-batch-adapter"; batchId: string; adapterId: string | null }
  | { type: "add-sample"; batchId: string }
  | { type: "duplicate-sample"; batchId: string; sampleId: string }
  | { type: "delete-sample"; batchId: string; sampleId: string }
  | { type: "move-sample"; sampleId: string; toBatchId: string }
  | { type: "set-sample-label"; sampleId: string; label: string }
  | { type: "set-sample-prompt"; sampleId: string; prompt: string }
  | { type: "set-sample-mode"; sampleId: string; inputMode: SampleDraft["inputMode"] }
  | { type: "set-sample-synthetic-length"; sampleId: string; syntheticInputTokens: number }
  | { type: "set-sample-resolved-length"; sampleId: string; resolvedInputTokens: number | null }
  | { type: "set-sample-output"; sampleId: string; maxOutputTokens: number }
  | { type: "set-sample-adapter"; sampleId: string; adapterId: string | null }

export function reduceScenario(scenario: ScenarioDraft, action: ScenarioAction): ScenarioDraft {
  switch (action.type) {
    case "replace":
      return action.scenario
    case "rename":
      return { ...scenario, name: action.name }
    case "add-batch":
      return addBatch(scenario)
    case "duplicate-batch":
      return duplicateBatch(scenario, action.batchId)
    case "delete-batch":
      return { ...scenario, batches: scenario.batches.filter((batch) => batch.id !== action.batchId) }
    case "set-batch-label":
      return mapBatch(scenario, action.batchId, (batch) => ({ ...batch, label: action.label }))
    case "commit-batch-time":
      return sortBatches(mapBatch(scenario, action.batchId, (batch) => ({ ...batch, arrivalOffsetMs: action.arrivalOffsetMs })))
    case "set-batch-adapter":
      return mapBatch(scenario, action.batchId, (batch) => ({ ...batch, defaultAdapterId: action.adapterId }))
    case "add-sample":
      return mapBatch(scenario, action.batchId, (batch) => ({
        ...batch,
        samples: [...batch.samples, newSample(batch.samples.length + 1)],
      }))
    case "duplicate-sample":
      return mapBatch(scenario, action.batchId, (batch) => {
        const source = batch.samples.find((sample) => sample.id === action.sampleId)
        if (!source) return batch
        const copy: SampleDraft = {
          ...source,
          id: createId("sample"),
          label: `${source.label} copy`,
        }
        const index = batch.samples.findIndex((sample) => sample.id === action.sampleId)
        const samples = batch.samples.slice()
        samples.splice(index + 1, 0, copy)
        return { ...batch, samples }
      })
    case "delete-sample":
      return mapBatch(scenario, action.batchId, (batch) => ({
        ...batch,
        samples: batch.samples.filter((sample) => sample.id !== action.sampleId),
      }))
    case "move-sample":
      return moveSample(scenario, action.sampleId, action.toBatchId)
    case "set-sample-label":
      return mapSample(scenario, action.sampleId, (sample) => ({ ...sample, label: action.label }))
    case "set-sample-prompt":
      return mapSample(scenario, action.sampleId, (sample) => ({ ...sample, prompt: action.prompt }))
    case "set-sample-mode":
      return mapSample(scenario, action.sampleId, (sample) => ({ ...sample, inputMode: action.inputMode }))
    case "set-sample-synthetic-length":
      return mapSample(scenario, action.sampleId, (sample) => ({ ...sample, syntheticInputTokens: action.syntheticInputTokens }))
    case "set-sample-resolved-length":
      return mapSample(scenario, action.sampleId, (sample) => ({ ...sample, resolvedInputTokens: action.resolvedInputTokens }))
    case "set-sample-output":
      return mapSample(scenario, action.sampleId, (sample) => ({ ...sample, maxOutputTokens: action.maxOutputTokens }))
    case "set-sample-adapter":
      return mapSample(scenario, action.sampleId, (sample) => ({ ...sample, adapterId: action.adapterId }))
    default:
      return scenario
  }
}

export function sampleCount(scenario: ScenarioDraft): number {
  return scenario.batches.reduce((sum, batch) => sum + batch.samples.length, 0)
}

function addBatch(scenario: ScenarioDraft): ScenarioDraft {
  const greatest = scenario.batches.reduce((max, batch) => Math.max(max, batch.arrivalOffsetMs), -100)
  const createdOrder = scenario.nextOrder
  const batch: ArrivalBatchDraft = {
    id: createId("batch"),
    label: nextBatchLabel(scenario.batches.length),
    arrivalOffsetMs: scenario.batches.length === 0 ? 0 : greatest + 100,
    createdOrder,
    defaultAdapterId: null,
    samples: [],
  }
  return sortBatches({
    ...scenario,
    nextOrder: createdOrder + 1,
    batches: [...scenario.batches, batch],
  })
}

function duplicateBatch(scenario: ScenarioDraft, batchId: string): ScenarioDraft {
  const source = scenario.batches.find((batch) => batch.id === batchId)
  if (!source) return scenario
  const createdOrder = scenario.nextOrder
  const copy: ArrivalBatchDraft = {
    ...source,
    id: createId("batch"),
    label: `${source.label} copy`,
    createdOrder,
    samples: source.samples.map((sample) => ({
      ...sample,
      id: createId("sample"),
      label: `${sample.label} copy`,
    })),
  }
  const index = scenario.batches.findIndex((batch) => batch.id === batchId)
  const batches = scenario.batches.slice()
  batches.splice(index + 1, 0, copy)
  return sortBatches({ ...scenario, nextOrder: createdOrder + 1, batches })
}

function moveSample(scenario: ScenarioDraft, sampleId: string, toBatchId: string): ScenarioDraft {
  let moving: SampleDraft | null = null
  const stripped = scenario.batches.map((batch) => {
    const sample = batch.samples.find((item) => item.id === sampleId)
    if (!sample) return batch
    moving = sample
    return { ...batch, samples: batch.samples.filter((item) => item.id !== sampleId) }
  })
  if (!moving) return scenario
  const taken: SampleDraft = moving
  return {
    ...scenario,
    batches: stripped.map((batch) => (
      batch.id === toBatchId ? { ...batch, samples: [...batch.samples, taken] } : batch
    )),
  }
}

function mapBatch(
  scenario: ScenarioDraft,
  batchId: string,
  update: (batch: ArrivalBatchDraft) => ArrivalBatchDraft,
): ScenarioDraft {
  return {
    ...scenario,
    batches: scenario.batches.map((batch) => (batch.id === batchId ? update(batch) : batch)),
  }
}

function mapSample(
  scenario: ScenarioDraft,
  sampleId: string,
  update: (sample: SampleDraft) => SampleDraft,
): ScenarioDraft {
  return {
    ...scenario,
    batches: scenario.batches.map((batch) => ({
      ...batch,
      samples: batch.samples.map((sample) => (sample.id === sampleId ? update(sample) : sample)),
    })),
  }
}

function sortBatches(scenario: ScenarioDraft): ScenarioDraft {
  const batches = scenario.batches.slice().sort((left, right) => (
    left.arrivalOffsetMs - right.arrivalOffsetMs || left.createdOrder - right.createdOrder
  ))
  return { ...scenario, batches }
}

function newSample(index: number): SampleDraft {
  return {
    id: createId("sample"),
    label: `input_${index}`,
    prompt: "",
    inputMode: "prompt",
    syntheticInputTokens: 128,
    resolvedInputTokens: null,
    maxOutputTokens: 8,
    adapterId: null,
  }
}

function nextBatchLabel(count: number): string {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  return alphabet[count] ?? `Batch ${count + 1}`
}
