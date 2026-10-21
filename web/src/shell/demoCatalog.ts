import type { RecordingManifest } from "../../../replay/src/reduce.ts"
import type { CatalogEntry } from "../api.ts"
import type { EngineSettings, ScenarioDraft } from "../scenario/types.ts"

export type WeightFormat = "bf16" | "fp16" | "fp32" | "fp8" | "nvfp4" | "mxfp4" | "unknown"
export type KvFormat = "bf16" | "fp16" | "fp8_e4m3" | "fp8_e5m2" | "unknown"
export type LoraMode = "none" | "all" | "mixed" | "unknown"

export interface RecordingConfiguration {
  modelId: string
  modelRevision: string | null
  weightFormat: WeightFormat
  kvFormat: KvFormat
  loraMode: LoraMode
  requestedComputeDtype: string | null
  observedComputeDtype: string | null
  requestedQuantization: string | null
  runtimeVersion: string | null
}

export interface DemoSelection {
  modelId: string
  weightFormat: WeightFormat | "any"
  kvFormat: KvFormat | "any"
  loraMode: LoraMode | "any"
}

export const DEMO_MODELS = [
  { id: "Qwen/Qwen3-4B-Instruct-2507", label: "Qwen3-4B-Instruct-2507" },
  { id: "Qwen/Qwen3.8-27B", label: "Qwen3.8-27B" },
  { id: "Qwen/Qwen3.8-Flash-Next", label: "Qwen3.8-Flash-Next" },
] as const

export const DEFAULT_DEMO_SELECTION: DemoSelection = {
  modelId: DEMO_MODELS[0].id, weightFormat: "any", kvFormat: "any", loraMode: "none",
}

export const WEIGHT_FORMAT_LABELS: Record<WeightFormat | "any", string> = {
  any: "Any recorded format", bf16: "BF16", fp16: "FP16", fp32: "FP32", fp8: "FP8",
  nvfp4: "NVFP4", mxfp4: "MXFP4", unknown: "Not captured",
}
export const KV_FORMAT_LABELS: Record<KvFormat | "any", string> = {
  any: "Any recorded dtype", bf16: "BF16", fp16: "FP16", fp8_e4m3: "FP8 E4M3",
  fp8_e5m2: "FP8 E5M2", unknown: "Not captured",
}
export const LORA_MODE_LABELS: Record<LoraMode | "any", string> = {
  any: "Any recorded mode", none: "None", all: "All requests: Adapter A", mixed: "Mixed: Base + Adapter A", unknown: "Not captured",
}

/** Requested settings are retained separately and cannot qualify an observed storage format. */
export function recordingConfiguration(manifest: RecordingManifest): RecordingConfiguration {
  const model = record(manifest.model)
  const engine = record(manifest.engine)
  const settings = record(manifest.engine_settings)
  const weights = record(model.weight_storage)
  const cache = record(manifest.kv_cache)
  const layers = cache.status === "observed" && Array.isArray(cache.layers) ? cache.layers.map(record) : []
  const formats = new Set(layers.map((layer) => kvFormat(layer)))
  const arrivals = Array.isArray(manifest.arrivals) ? manifest.arrivals.map(record) : []
  const observedRoutes = arrivals.length > 0 && arrivals.every((row) => row.engine_admission_observed === true && typeof row.adapter_id === "string")
  const adapterIds = new Set(observedRoutes ? arrivals.map((row) => row.adapter_id) : [])
  const adapters = [...adapterIds].filter((id) => id !== "base")
  const loraMode: LoraMode = settings.enable_lora === false ? "none"
    : !observedRoutes || adapters.length > 1 ? "unknown"
      : adapters.length === 0 ? "none" : adapterIds.has("base") ? "mixed" : "all"
  return {
    modelId: string(model.id) ?? "Unknown model",
    modelRevision: string(model.revision),
    weightFormat: weights.status === "observed" ? weightFormat(weights.format) : "unknown",
    kvFormat: formats.size === 1 ? [...formats][0] : "unknown",
    loraMode,
    requestedComputeDtype: string(model.dtype),
    observedComputeDtype: weights.status === "observed" ? string(weights.compute_dtype) : null,
    requestedQuantization: string(model.quantization),
    runtimeVersion: string(engine.version),
  }
}

export function matchesDemo(entry: CatalogEntry, selection: DemoSelection): boolean {
  const config = entry.configuration
  return entry.evidence === "gpu" && Boolean(config
    && modelFamily(config.modelId) === modelFamily(selection.modelId)
    && (selection.weightFormat === "any" || selection.weightFormat === config.weightFormat)
    && (selection.kvFormat === "any" || selection.kvFormat === config.kvFormat)
    && (selection.loraMode === "any" || selection.loraMode === config.loraMode))
}

export function demoAvailability(catalog: CatalogEntry[], selection: DemoSelection): {
  status: "recorded" | "capture_required" | "unsupported"
  label: string
  reason: string
  recordings: CatalogEntry[]
} {
  const recordings = catalog.filter((entry) => matchesDemo(entry, selection))
  if (recordings.length > 0) return { status: "recorded", label: "Recorded", reason: `${recordings.length} matching real GPU recording${recordings.length === 1 ? "" : "s"}.`, recordings }
  if (modelFamily(selection.modelId) === DEMO_MODELS[2].id) {
    return { status: "unsupported", label: "Unsupported", reason: "The current qualification host has one H100 80GB with PyPI vLLM 0.30. The official Flash-Next recipe uses a dedicated runtime image and validates 4× H100 with CPU embedding offload; its FP8 checkpoint is 172.78 GiB. This configuration needs a different qualified deployment before capture.", recordings }
  }
  if (selection.weightFormat === "nvfp4" || selection.weightFormat === "mxfp4") {
    return { status: "unsupported", label: "Unsupported", reason: `This explorer has no qualified ${WEIGHT_FORMAT_LABELS[selection.weightFormat]} checkpoint and capture path for the selected model. A compatible checkpoint and runtime must be qualified first.`, recordings }
  }
  if (selection.weightFormat === "unknown" || selection.kvFormat === "unknown" || selection.loraMode === "unknown") {
    return { status: "capture_required", label: "Capture required", reason: "No saved recording matches the missing-metadata filters. Choose a concrete format and adapter mode to prepare a new capture.", recordings }
  }
  const hybrid = modelFamily(selection.modelId) !== DEMO_MODELS[0].id
  return {
    status: "capture_required", label: "Capture required", recordings,
    reason: hybrid
      ? "No matching real recording is bundled. The selected model needs a compatible runtime, sufficient GPU memory and qualified hybrid-layer diagnostics before its demo can be captured."
      : "No matching real recording is bundled. Prepare this configuration on a compatible GPU; model, cache dtype and adapter support are checked during preparation.",
  }
}

/** Maps a requested capture to Configure without changing any saved recording. Adapter artifacts remain explicit. */
export function demoCaptureSettings(settings: EngineSettings, selection: DemoSelection): EngineSettings {
  const dtype = selection.weightFormat === "bf16" ? "bfloat16"
    : selection.weightFormat === "fp16" ? "float16" : selection.weightFormat === "fp32" ? "float32" : "auto"
  const quantization = selection.weightFormat === "fp8" ? "fp8" : "none"
  const kvCacheDtype = selection.kvFormat === "bf16" ? "bfloat16" : selection.kvFormat === "fp16" ? "float16"
    : selection.kvFormat === "fp8_e4m3" || selection.kvFormat === "fp8_e5m2" ? selection.kvFormat : "auto"
  const model = selection.modelId === DEMO_MODELS[2].id && selection.weightFormat === "fp8"
    ? `${selection.modelId}-FP8` : selection.modelId
  const demoLoraMode = selection.loraMode === "any" || selection.loraMode === "unknown" ? null : selection.loraMode
  return { ...settings, model, modelSource: "huggingface", modelRevision: "", tokenizerRevision: "", tokenizer: "", dtype, quantization, kvCacheDtype, demoLoraMode, loraAdapters: selection.loraMode === "none" ? [] : settings.loraAdapters }
}

/** Assigns a real configured adapter; missing artifacts cannot become a fabricated Adapter A. */
export function demoAdapterScenario(scenario: ScenarioDraft, settings: EngineSettings, mode: DemoSelection["loraMode"]): ScenarioDraft {
  if (mode === "any" || mode === "unknown") return scenario
  if (mode === "mixed" && scenario.batches.reduce((count, batch) => count + batch.samples.length, 0) < 2) throw new Error("Mixed LoRA assignment needs at least two requests.")
  const adapter = settings.loraAdapters.find((item) => item.location.trim())
  if (mode !== "none" && !adapter) throw new Error("Choose a compatible LoRA adapter repository or path in Model configuration before applying this assignment.")
  let requestIndex = 0
  return {
    ...scenario,
    batches: scenario.batches.map((batch) => ({
      ...batch, defaultAdapterId: null,
      samples: batch.samples.map((sample) => ({
        ...sample,
        adapterId: mode === "none" || mode === "mixed" && requestIndex++ % 2 === 0 ? "__base__" : adapter!.id,
      })),
    })),
  }
}

function modelFamily(id: string): string {
  return id.replace(/-FP8$/, "")
}

function weightFormat(value: unknown): WeightFormat {
  if (value === "bfloat16" || value === "bf16") return "bf16"
  if (value === "float16" || value === "fp16") return "fp16"
  if (value === "float32" || value === "fp32") return "fp32"
  if (value === "fp8" || value === "fp8_e4m3" || value === "fp8_e5m2") return "fp8"
  return value === "nvfp4" || value === "mxfp4" ? value : "unknown"
}

function kvFormat(layer: Record<string, unknown>): KvFormat {
  const quantization = record(layer.quantization)
  if (layer.dtype === "bfloat16") return "bf16"
  if (layer.dtype === "float16") return "fp16"
  if (layer.dtype === "uint8" && layer.layout_status === "qualified"
    && (layer.numerical_encoding === "fp8_e4m3" || layer.numerical_encoding === "fp8_e5m2")) return layer.numerical_encoding
  if (quantization.cache_dtype === "fp8_e4m3" || quantization.cache_dtype === "fp8_e5m2") return quantization.cache_dtype
  return "unknown"
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function string(value: unknown): string | null {
  return typeof value === "string" && value ? value : null
}
