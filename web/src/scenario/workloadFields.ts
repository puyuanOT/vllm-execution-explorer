import { formatTime, parseMilliseconds, parseTime } from "./time.ts"
import type { ScenarioDraft, TimeUnit } from "./types.ts"
import type { FieldIssue } from "./validate.ts"

type ParsedField = { value: number } | { issue: FieldIssue }

export function resolveBatchTime(fieldId: string, raw: string, unit: TimeUnit): ParsedField {
  const value = parseTime(raw, unit)
  return value == null
    ? { issue: { fieldId, message: unit === "ms"
      ? "Enter a nonnegative whole number of milliseconds."
      : "Enter seconds precise to a whole millisecond, such as 0.12." } }
    : { value }
}

export function resolvePositiveInteger(fieldId: string, raw: string): ParsedField {
  const value = parseMilliseconds(raw)
  return value == null || value < 1
    ? { issue: { fieldId, message: "Enter a positive whole number." } }
    : { value }
}

export function resolvePriority(fieldId: string, raw: string): ParsedField {
  const value = Number(raw)
  return !/^-?\d+$/.test(raw.trim()) || !Number.isSafeInteger(value) || Math.abs(value) > 2_147_483_647
    ? { issue: { fieldId, message: "Enter a whole priority number between −2147483647 and 2147483647." } }
    : { value }
}

export function resolveWorkloadFields(
  scenario: ScenarioDraft,
  unit: TimeUnit,
  readValue: (fieldId: string) => string | undefined,
): { scenario: ScenarioDraft } | { issue: FieldIssue } {
  const batches = []
  for (const batch of scenario.batches) {
    const fieldId = `batch-time-${batch.id}`
    const time = resolveBatchTime(fieldId, readValue(fieldId) ?? formatTime(batch.arrivalOffsetMs, unit), unit)
    if ("issue" in time) return time
    const samples = []
    for (const sample of batch.samples) {
      const priorityId = `sample-priority-${sample.id}`
      const priority = resolvePriority(priorityId, readValue(priorityId) ?? String(sample.priority ?? 0))
      if ("issue" in priority) return priority
      const outputId = `sample-output-${sample.id}`
      const output = resolvePositiveInteger(outputId, readValue(outputId) ?? String(sample.maxOutputTokens))
      if ("issue" in output) return output
      let syntheticInputTokens = sample.syntheticInputTokens
      if (sample.inputMode === "synthetic") {
        const inputId = `sample-synthetic-${sample.id}`
        const input = resolvePositiveInteger(inputId, readValue(inputId) ?? String(syntheticInputTokens))
        if ("issue" in input) return input
        syntheticInputTokens = input.value
      }
      samples.push({ ...sample, maxOutputTokens: output.value, syntheticInputTokens, priority: priority.value })
    }
    batches.push({ ...batch, arrivalOffsetMs: time.value, samples })
  }
  return { scenario: { ...scenario, batches } }
}
