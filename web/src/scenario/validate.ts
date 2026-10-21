import type { FeatureCapability, Limits } from "../api.ts"
import { sampleCount } from "./reducer.ts"
import { captureIndices } from "./diagnostics.ts"
import type { EngineSettings, ScenarioDraft } from "./types.ts"

export interface FieldIssue {
  fieldId: string
  message: string
}

export function validateScenario(
  scenario: ScenarioDraft,
  limits: Limits | null,
  settings: EngineSettings,
  quantizationCapabilities?: Record<string, FeatureCapability>,
): FieldIssue[] {
  const issues: FieldIssue[] = []
  const maxRequests = limits?.max_requests ?? null
  const maxPrompt = limits?.max_prompt_characters ?? null
  const maxOutput = limits?.max_output_tokens ?? null
  const maxOffset = limits ? limits.max_run_seconds * 1000 : null
  if (scenario.batches.length === 0) {
    issues.push({ fieldId: "add-batch", message: "Add an arrival batch before running." })
  }
  const ids = new Set<string>()
  if (maxRequests != null && sampleCount(scenario) > maxRequests) {
    issues.push({
      fieldId: "sample-count",
      message: `This runner accepts at most ${maxRequests} samples.`,
    })
  }
  for (const batch of scenario.batches) {
    if (!batch.label.trim()) {
      issues.push({ fieldId: `batch-label-${batch.id}`, message: "Give this arrival batch a label." })
    }
    if (!Number.isSafeInteger(batch.arrivalOffsetMs) || batch.arrivalOffsetMs < 0) {
      issues.push({ fieldId: `batch-time-${batch.id}`, message: "Arrival time must be a nonnegative whole number of milliseconds." })
    } else if (maxOffset != null && batch.arrivalOffsetMs > maxOffset) {
      issues.push({
        fieldId: `batch-time-${batch.id}`,
        message: `Arrival time must be within the ${limits?.max_run_seconds ?? ""}s run limit.`,
      })
    }
    if (batch.samples.length === 0) {
      issues.push({ fieldId: `add-sample-${batch.id}`, message: "This arrival batch has no samples. Add one or delete the batch." })
    }
    for (const sample of batch.samples) {
      if (ids.has(sample.id)) {
        issues.push({ fieldId: `sample-label-${sample.id}`, message: "Sample identities must be unique." })
      }
      ids.add(sample.id)
      if (!Number.isSafeInteger(sample.priority ?? 0) || Math.abs(sample.priority ?? 0) > 2_147_483_647) {
        issues.push({ fieldId: `sample-priority-${sample.id}`, message: "Priority must be a whole number between −2147483647 and 2147483647." })
      }
      if (settings.schedulingPolicy === "fcfs" && (sample.priority ?? 0) !== 0) {
        issues.push({ fieldId: `sample-priority-${sample.id}`, message: "Choose priority scheduling to use nonzero request priorities, or set this priority to 0." })
      }
      if (!sample.label.trim()) {
        issues.push({ fieldId: `sample-label-${sample.id}`, message: "Give this sample a display label." })
      }
      if (sample.inputMode === "prompt" && !sample.prompt.trim()) {
        issues.push({ fieldId: `sample-prompt-${sample.id}`, message: "Enter a prompt." })
      } else if (sample.inputMode === "prompt" && maxPrompt != null && sample.prompt.length > maxPrompt) {
        issues.push({
          fieldId: `sample-prompt-${sample.id}`,
          message: `Prompts are limited to ${maxPrompt} characters on this runner.`,
        })
      }
      if (sample.inputMode === "synthetic" && (!Number.isSafeInteger(sample.syntheticInputTokens) || sample.syntheticInputTokens < 1)) {
        issues.push({ fieldId: `sample-synthetic-${sample.id}`, message: "Synthetic input length must be a positive token count." })
      }
      if (!Number.isSafeInteger(sample.maxOutputTokens) || sample.maxOutputTokens < 1) {
        issues.push({ fieldId: `sample-output-${sample.id}`, message: "Maximum output tokens must be a positive integer." })
      } else if (maxOutput != null && sample.maxOutputTokens > maxOutput) {
        issues.push({
          fieldId: `sample-output-${sample.id}`,
          message: `Output tokens are limited to ${maxOutput} on this runner.`,
        })
      }
    }
  }
  issues.push(...validateSettings(settings))
  if (settings.profilingMode !== "off") {
    for (const [fieldId, selected] of [["setting-profiling-trigger", settings.profilingTriggerRequestId], ["setting-profiling-tensor-request", settings.tensorCapture ? settings.tensorRequestId : ""]]) {
      if (selected && !ids.has(selected)) issues.push({ fieldId, message: "Choose a request that still exists in this workload." })
    }
  }
  const quantizationCapability = quantizationCapabilities?.[settings.quantization]
  if (settings.engine === "vllm" && quantizationCapability?.selectable === false) {
    issues.push({ fieldId: "model-quantization", message: quantizationCapability.message })
  }
  return issues
}

export function validateSettings(settings: EngineSettings): FieldIssue[] {
  const issues: FieldIssue[] = []
  if (!isPositiveInteger(settings.maxNumSeqs, 256)) {
    issues.push({ fieldId: "setting-max-seqs", message: "Max sequences must be a positive integer." })
  }
  if (!isPositiveInteger(settings.tokenBudget, 1_000_000)) {
    issues.push({ fieldId: "setting-token-budget", message: "Token budget must be a positive integer." })
  }
  if (!isPositiveInteger(settings.maxModelLen, 2_000_000)) {
    issues.push({ fieldId: "setting-max-model-len", message: "Maximum context length must be a positive integer." })
  }
  if (!isDecimalInRange(settings.gpuMemoryUtilization, 0, 1, false, true)) {
    issues.push({ fieldId: "setting-gpu-memory", message: "GPU memory fraction must be greater than 0 and no more than 1." })
  }
  if (!isDecimalInRange(settings.temperature, 0, 10, true, true)) {
    issues.push({ fieldId: "setting-temperature", message: "Temperature must be a number from 0 to 10." })
  }
  if (!isDecimalInRange(settings.topP, 0, 1, false, true)) {
    issues.push({ fieldId: "setting-top-p", message: "Top-p must be greater than 0 and no more than 1." })
  }
  if (settings.seed.trim() && (!/^\d+$/.test(settings.seed.trim()) || Number(settings.seed) > Number.MAX_SAFE_INTEGER)) {
    issues.push({ fieldId: "setting-seed", message: "Seed must be a nonnegative whole number or left empty." })
  }
  const stop = settings.stopStrings.map((value) => value.trim()).filter(Boolean)
  if (stop.length > 8 || stop.some((value) => value.length > 256)) {
    issues.push({ fieldId: "setting-stop-strings", message: "Use at most 8 stop strings, each no longer than 256 characters." })
  }
  if (settings.engine === "vllm") {
    if (!settings.model.trim()) {
      issues.push({ fieldId: "setting-model", message: "Choose a model or enter its location." })
    }
    if (settings.modelSource === "operator" && settings.hostKind === "ssh" && !settings.modelDestination.trim().startsWith("/")) {
      issues.push({ fieldId: "model-destination", message: "Use an absolute destination path on the execution host." })
    }
    if (settings.hostKind === "ssh" && (!settings.sshUser.trim() || !settings.sshHost.trim())) {
      issues.push({ fieldId: "ssh-host", message: "Enter the SSH user and host address." })
    }
    if (settings.hostKind === "ssh" && !isPort(settings.sshPort)) {
      issues.push({ fieldId: "ssh-port", message: "SSH port must be between 1 and 65535." })
    }
    for (const adapter of settings.loraAdapters) {
      if (!adapter.name.trim()) {
        issues.push({ fieldId: `adapter-name-${adapter.id}`, message: "Give this adapter a name." })
      }
      if (!adapter.location.trim()) {
        issues.push({ fieldId: `adapter-location-${adapter.id}`, message: "Enter the adapter repository or path." })
      }
    }
  }
  if (settings.engine === "vllm" && isPositiveInteger(settings.tokenBudget, 1_000_000)) {
    if (Number(settings.tokenBudget) < Number(settings.maxNumSeqs)) {
      issues.push({ fieldId: "setting-token-budget", message: "Token budget must be at least the maximum concurrent sequences." })
    }
    if (!settings.chunkedPrefill && Number(settings.tokenBudget) < Number(settings.maxModelLen)) {
      issues.push({ fieldId: "setting-token-budget", message: "Without chunked prefill, token budget must cover the maximum context length." })
    }
  }
  if (!["fcfs", "priority"].includes(settings.schedulingPolicy)) {
    issues.push({ fieldId: "setting-scheduling-policy", message: "Choose FCFS or priority scheduling." })
  }
  if (!/^\d+$/.test(settings.longPrefillTokenThreshold.trim()) || !Number.isSafeInteger(Number(settings.longPrefillTokenThreshold))) {
    issues.push({ fieldId: "setting-long-prefill", message: "Long-prefill threshold must be a nonnegative whole number." })
  }
  if (settings.kvCacheMemoryBytes.trim() && !isPositiveInteger(settings.kvCacheMemoryBytes, Number.MAX_SAFE_INTEGER)) {
    issues.push({ fieldId: "setting-kv-memory", message: "KV memory budget must be a positive whole number of bytes, or left empty." })
  }
  if (!["off", "eager_diagnostic"].includes(settings.profilingMode)) {
    issues.push({ fieldId: "setting-profiling-mode", message: "Choose a supported profiling mode." })
  }
  if (settings.profilingMode !== "off") {
    if (settings.engine === "fixture") issues.push({ fieldId: "setting-profiling-mode", message: "Measured profiling requires live vLLM on a GPU host." })
    if (!isPositiveInteger(settings.profilingMaxSteps, 16)) issues.push({ fieldId: "setting-profiling-steps", message: "Profile between 1 and 16 scheduler steps." })
    if (!isPositiveInteger(settings.profilingStartStep, 1_000_000)) issues.push({ fieldId: "setting-profiling-start", message: "Starting scheduler step must be between 1 and 1000000." })
    if (settings.tensorCapture) {
      if (!["small", "custom"].includes(settings.tensorPreset)) issues.push({ fieldId: "setting-profiling-tensor-preset", message: "Choose a tensor capture preset." })
      if (!isPositiveInteger(settings.tensorMaxBytes, 16 * 1024 * 1024) || Number(settings.tensorMaxBytes) < 64) issues.push({ fieldId: "setting-profiling-tensor-bytes", message: "Tensor byte limit must be between 64 and 16777216." })
      if (settings.tensorPreset === "custom") {
        for (const [fieldId, value, limit] of [["setting-profiling-tensor-positions", settings.tensorTokenPositions, 64], ["setting-profiling-tensor-layers", settings.tensorLayerIndices, 64], ["setting-profiling-tensor-heads", settings.tensorHeadIndices, 32]] as const) {
          if (captureIndices(value, limit) == null || fieldId.endsWith("heads") && !captureIndices(value, limit)?.length) issues.push({ fieldId, message: `Enter up to ${limit} zero-based indices, separated by commas or spaces.${fieldId.endsWith("heads") ? " Select at least one head." : " Leave empty for the preset selection."}` })
        }
        if (!/^\d+$/.test(settings.tensorDimensionStart.trim()) || Number(settings.tensorDimensionStart) > 65535) issues.push({ fieldId: "setting-profiling-tensor-dimension-start", message: "Starting dimension must be between 0 and 65535." })
        if (!isPositiveInteger(settings.tensorDimensionCount, 256)) issues.push({ fieldId: "setting-profiling-tensor-dimension-count", message: "Capture between 1 and 256 dimensions." })
      }
    }
  }
  return issues
}

function isDecimalInRange(
  value: string,
  minimum: number,
  maximum: number,
  includeMinimum: boolean,
  includeMaximum: boolean,
): boolean {
  if (!value.trim()) return false
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return false
  const above = includeMinimum ? parsed >= minimum : parsed > minimum
  const below = includeMaximum ? parsed <= maximum : parsed < maximum
  return above && below
}

function isPort(value: string): boolean {
  if (!/^\d+$/.test(value.trim())) return false
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed >= 1 && parsed <= 65535
}

function isPositiveInteger(value: string, maximum: number): boolean {
  if (!/^\d+$/.test(value.trim())) return false
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) && parsed >= 1 && parsed <= maximum
}
