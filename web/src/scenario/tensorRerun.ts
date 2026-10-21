import type { RecordingManifest } from "../../../replay/src/reduce.ts"
import type { EngineSettings } from "./types.ts"

export interface TensorRerunTarget { requestId: string; tokenPosition: number; layerIndex: number; kvHead: number; startStep?: number }

/** Retain the operator's connection while restoring recorded inference settings. */
export function tensorRerunSettings(current: EngineSettings, manifest: RecordingManifest, target: TensorRerunTarget): EngineSettings {
  const configuration = record(manifest.run_configuration)
  const model = record(configuration.model)
  const source = record(model.source)
  const tokenizer = record(model.tokenizer)
  const sampling = record(configuration.sampling ?? manifest.sampling)
  const engine = record(configuration.vllm ?? manifest.engine_settings)
  const next: EngineSettings = {
    ...current, demoLoraMode: null, engine: "vllm", profilingMode: "eager_diagnostic", profilingMaxSteps: "8",
    profilingStartStep: String(target.startStep ?? 1), profilingTriggerRequestId: target.requestId,
    tensorCapture: true, tensorPreset: "custom", tensorRequestId: target.requestId,
    tensorTokenPositions: String(target.tokenPosition), tensorLayerIndices: String(target.layerIndex),
    tensorHeadIndices: String(target.kvHead), tensorDimensionStart: "0", tensorDimensionCount: "16", tensorMaxBytes: "1048576",
  }
  const sourceKind = source.kind === "huggingface" ? "huggingface" : source.kind === "execution_host" ? "host" : source.kind === "operator_local" ? "operator" : null
  if (sourceKind && typeof source.locator === "string") {
    next.modelSource = sourceKind; next.model = source.locator
    next.modelRevision = typeof source.revision === "string" ? source.revision : ""
  }
  if (typeof tokenizer.locator === "string") {
    next.tokenizer = tokenizer.locator
    next.tokenizerRevision = typeof tokenizer.revision === "string" ? tokenizer.revision : ""
  }
  if (["auto", "float16", "bfloat16", "float32"].includes(String(model.dtype))) next.dtype = model.dtype as EngineSettings["dtype"]
  const quantization = record(model.quantization).kind
  if (["none", "awq", "gptq", "bitsandbytes", "fp8"].includes(String(quantization))) next.quantization = quantization as EngineSettings["quantization"]
  for (const [key, field] of [["maxModelLen", "max_model_len"], ["maxNumSeqs", "max_num_seqs"], ["tokenBudget", "max_num_scheduled_tokens"],
    ["gpuMemoryUtilization", "gpu_memory_utilization"], ["longPrefillTokenThreshold", "long_prefill_token_threshold"]] as const) {
    if (typeof engine[field] === "number" && Number.isFinite(engine[field])) next[key] = String(engine[field])
  }
  if (typeof engine.kv_cache_memory_bytes === "number") next.kvCacheMemoryBytes = String(engine.kv_cache_memory_bytes)
  else if (engine.kv_cache_memory_bytes === null) next.kvCacheMemoryBytes = ""
  if (["auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"].includes(String(engine.kv_cache_dtype))) next.kvCacheDtype = engine.kv_cache_dtype as EngineSettings["kvCacheDtype"]
  else next.kvCacheDtype = "auto"
  if (typeof engine.enable_prefix_caching === "boolean") next.prefixCaching = engine.enable_prefix_caching
  if (typeof engine.enable_chunked_prefill === "boolean") next.chunkedPrefill = engine.enable_chunked_prefill
  if (engine.scheduling_policy === "fcfs" || engine.scheduling_policy === "priority") next.schedulingPolicy = engine.scheduling_policy
  for (const [key, field] of [["temperature", "temperature"], ["topP", "top_p"], ["seed", "seed"]] as const) {
    if (typeof sampling[field] === "number" && Number.isFinite(sampling[field])) next[key] = String(sampling[field])
  }
  if (Array.isArray(sampling.stop) && sampling.stop.every((item) => typeof item === "string")) next.stopStrings = [...sampling.stop]
  if (Array.isArray(model.lora_adapters)) next.loraAdapters = model.lora_adapters.flatMap((raw) => {
    const adapter = record(raw), location = record(adapter.source)
    const kind = location.kind === "huggingface" ? "huggingface" as const : location.kind === "execution_host" ? "host" as const : location.kind === "operator_local" ? "operator" as const : null
    return typeof adapter.id === "string" && typeof location.locator === "string" && kind
      ? [{ id: adapter.id, name: adapter.id, source: kind, location: location.locator, revision: typeof location.revision === "string" ? location.revision : "" }] : []
  })
  return next
}

function record(value: unknown): Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {}
}
