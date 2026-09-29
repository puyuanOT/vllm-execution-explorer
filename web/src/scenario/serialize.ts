import { createId } from "./ids.ts"
import type {
  ArtifactSourcePayload,
  EngineSettings,
  LoraAdapterDraft,
  RunRequestBody,
  ScenarioDraft,
  WorkloadPlan,
} from "./types.ts"

export function flattenRequests(scenario: ScenarioDraft): RunRequestBody["requests"] {
  return scenario.batches.flatMap((batch) => batch.samples.map((sample) => ({
    request_id: sample.id,
    prompt: sample.inputMode === "synthetic" ? "" : sample.prompt,
    input_mode: sample.inputMode,
    synthetic_input_tokens: sample.inputMode === "synthetic" ? sample.syntheticInputTokens : null,
    target_input_tokens: sample.inputMode === "synthetic" ? sample.syntheticInputTokens : null,
    resolved_input_tokens: sample.resolvedInputTokens,
    arrival_offset_ms: batch.arrivalOffsetMs,
    max_output_tokens: sample.maxOutputTokens,
    adapter_id: sample.adapterId === "__base__" ? null : sample.adapterId ?? batch.defaultAdapterId,
  })))
}

export function workloadPlan(scenario: ScenarioDraft): WorkloadPlan {
  return {
    version: 1,
    batches: scenario.batches.map((batch) => ({
      id: batch.id,
      label: batch.label.trim(),
      arrival_offset_ms: batch.arrivalOffsetMs,
      default_adapter_id: batch.defaultAdapterId,
      adapter_id: batch.defaultAdapterId ?? "base",
      samples: batch.samples.map((sample) => ({
        id: sample.id,
        label: sample.label.trim(),
        input_mode: sample.inputMode,
        synthetic_input_tokens: sample.inputMode === "synthetic" ? sample.syntheticInputTokens : null,
        target_input_tokens: sample.inputMode === "synthetic" ? sample.syntheticInputTokens : null,
        resolved_input_tokens: sample.resolvedInputTokens,
        max_output_tokens: sample.maxOutputTokens,
        adapter_id: sample.adapterId === "__base__" ? "base" : sample.adapterId,
      })),
    })),
  }
}

export function toRunBody(scenario: ScenarioDraft, settings: EngineSettings): RunRequestBody {
  const stop = settings.stopStrings.map((value) => value.trim()).filter(Boolean)
  const seed = settings.seed.trim() ? Number(settings.seed) : 0
  const source = artifactSource(settings.modelSource, settings.model, settings.modelRevision, settings.modelDestination)
  return {
    engine: settings.engine,
    requests: flattenRequests(scenario),
    workload_plan: workloadPlan(scenario),
    max_num_seqs: Number(settings.maxNumSeqs),
    prefix_caching: settings.engine === "fixture" ? false : settings.prefixCaching,
    token_budget: Number(settings.tokenBudget),
    model: settings.model.trim() || "Qwen/Qwen3-4B-Instruct-2507",
    model_revision: settings.modelRevision.trim() || null,
    tokenizer_revision: settings.tokenizerRevision.trim() || null,
    model_source: settings.modelSource,
    model_destination: settings.modelDestination.trim() || null,
    dtype: settings.dtype,
    quantization: settings.quantization === "none" ? null : settings.quantization,
    lora_adapters: settings.loraAdapters.map((adapter) => ({
      id: adapter.id,
      name: adapter.name.trim(),
      source: adapter.source,
      location: adapter.location.trim(),
      revision: adapter.revision.trim() || null,
    })),
    max_model_len: Number(settings.maxModelLen),
    chunked_prefill: settings.chunkedPrefill,
    gpu_memory_utilization: Number(settings.gpuMemoryUtilization),
    temperature: Number(settings.temperature),
    top_p: Number(settings.topP),
    seed,
    stop,
    model_configuration: {
      source,
      tokenizer: {
        locator: settings.model.trim(),
        revision: settings.tokenizerRevision.trim() || null,
      },
      dtype: settings.dtype,
      quantization: { kind: settings.quantization },
      lora_adapters: settings.loraAdapters.map((adapter) => ({
        id: adapter.id,
        source: adapterSource(adapter, settings.modelDestination),
      })),
    },
    vllm_configuration: {
      max_model_len: Number(settings.maxModelLen),
      max_num_seqs: Number(settings.maxNumSeqs),
      max_num_scheduled_tokens: Number(settings.tokenBudget),
      enable_chunked_prefill: settings.chunkedPrefill,
      enable_prefix_caching: settings.engine === "fixture" ? false : settings.prefixCaching,
      gpu_memory_utilization: Number(settings.gpuMemoryUtilization),
      fresh_prefix_cache: true,
    },
    sampling: {
      temperature: Number(settings.temperature),
      top_p: Number(settings.topP),
      seed,
      stop,
    },
  }
}

function artifactSource(
  source: EngineSettings["modelSource"],
  locator: string,
  revision: string,
  destination: string,
): ArtifactSourcePayload {
  if (source === "huggingface") {
    return { kind: "huggingface", repository: locator.trim(), revision: revision.trim() || null }
  }
  if (source === "host") {
    return { kind: "execution_host", path: locator.trim(), revision: revision.trim() || null }
  }
  return {
    kind: "operator_local",
    path: locator.trim(),
    revision: revision.trim() || null,
    transfer: { destination: destination.trim() || null },
  }
}

function adapterSource(adapter: LoraAdapterDraft, destination: string): ArtifactSourcePayload {
  return artifactSource(adapter.source, adapter.location, adapter.revision, destination)
}

export function scenarioFromRecordedPlan(
  plan: unknown,
  prompts: Record<string, string>,
): ScenarioDraft | null {
  if (!plan || typeof plan !== "object" || !Array.isArray((plan as { batches?: unknown }).batches)) return null
  const batches = (plan as { batches: unknown[] }).batches
  const draftBatches = batches.flatMap((batch, index) => {
    if (!batch || typeof batch !== "object") return []
    const row = batch as Record<string, unknown>
    if (typeof row.id !== "string" || !Array.isArray(row.samples)) return []
    const samples = row.samples.flatMap((sample) => {
      if (!sample || typeof sample !== "object") return []
      const item = sample as Record<string, unknown>
      if (typeof item.id !== "string") return []
      return [{
        id: item.id,
        label: typeof item.label === "string" ? item.label : item.id,
        prompt: prompts[item.id] ?? "",
        inputMode: item.input_mode === "synthetic" ? "synthetic" as const : "prompt" as const,
        syntheticInputTokens: typeof item.target_input_tokens === "number"
          ? item.target_input_tokens
          : typeof item.synthetic_input_tokens === "number" ? item.synthetic_input_tokens : 128,
        resolvedInputTokens: typeof item.resolved_input_tokens === "number" ? item.resolved_input_tokens : null,
        maxOutputTokens: typeof item.max_output_tokens === "number" ? item.max_output_tokens : 8,
        adapterId: item.adapter_id === "base"
          ? "__base__"
          : typeof item.adapter_id === "string" ? item.adapter_id : null,
      }]
    })
    if (samples.length === 0) return []
    return [{
      id: row.id,
      label: typeof row.label === "string" ? row.label : row.id,
      arrivalOffsetMs: typeof row.arrival_offset_ms === "number" ? row.arrival_offset_ms : 0,
      createdOrder: index,
      defaultAdapterId: typeof row.default_adapter_id === "string" && row.default_adapter_id !== "base"
        ? row.default_adapter_id
        : null,
      samples,
    }]
  })
  if (draftBatches.length === 0) return null
  return {
    version: 1,
    id: createId("scenario"),
    name: "Recorded sequence",
    nextOrder: draftBatches.length,
    batches: draftBatches,
  }
}

export function cloneScenario(scenario: ScenarioDraft, id: string): ScenarioDraft {
  return {
    ...scenario,
    id,
    name: scenario.name,
    batches: scenario.batches.map((batch) => ({
      ...batch,
      samples: batch.samples.map((sample) => ({ ...sample })),
    })),
  }
}
