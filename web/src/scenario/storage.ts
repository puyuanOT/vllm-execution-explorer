import { defaultSettings, staggeredExample } from "./example.ts"
import type { EngineSettings, ScenarioDraft } from "./types.ts"

export const SCENARIO_STORAGE_KEY = "vllm-explorer.scenario.v1"
export const SETTINGS_STORAGE_KEY = "vllm-explorer.settings.v1"

export interface StorageResult<T> {
  value: T
  note: string
  error: string
}

export function loadScenario(): StorageResult<ScenarioDraft> {
  const loaded = readStored(SCENARIO_STORAGE_KEY, isScenario, staggeredExample(), "Saved on this device")
  return { ...loaded, value: normalizeScenario(loaded.value) }
}

export function saveScenario(scenario: ScenarioDraft): { note: string; error: string } {
  return writeStored(SCENARIO_STORAGE_KEY, { version: 1, scenario })
}

export function clearScenarioStorage(): { note: string; error: string } {
  try {
    localStorage.removeItem(SCENARIO_STORAGE_KEY)
    return { note: "Draft cleared on this device", error: "" }
  } catch (error) {
    return { note: "", error: storageMessage(error) }
  }
}

export function loadSettings(): EngineSettings {
  const loaded = readStored(SETTINGS_STORAGE_KEY, isSettings, defaultSettings(), "")
  return {
    ...defaultSettings(),
    ...loaded.value,
    loraAdapters: Array.isArray(loaded.value.loraAdapters) ? loaded.value.loraAdapters : [],
    stopStrings: Array.isArray(loaded.value.stopStrings) ? loaded.value.stopStrings : [],
  }
}

export function saveSettings(settings: EngineSettings): void {
  writeStored(SETTINGS_STORAGE_KEY, { version: 1, settings })
}

function readStored<T>(key: string, check: (value: unknown) => value is T, fallback: T, note: string): StorageResult<T> {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return { value: fallback, note: "", error: "" }
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== "object") return { value: fallback, note: "", error: "Stored draft was unreadable. Started from the example." }
    const record = parsed as Record<string, unknown>
    if (record.version !== 1 || !check(record.scenario ?? record.settings)) {
      return { value: fallback, note: "", error: "Stored draft was from an unsupported version. Started from the example." }
    }
    return { value: (record.scenario ?? record.settings) as T, note, error: "" }
  } catch (error) {
    return { value: fallback, note: "", error: storageMessage(error) }
  }
}

function writeStored(key: string, value: unknown): { note: string; error: string } {
  try {
    localStorage.setItem(key, JSON.stringify(value))
    return { note: "Saved on this device", error: "" }
  } catch (error) {
    return { note: "", error: storageMessage(error) }
  }
}

function storageMessage(error: unknown): string {
  const detail = error instanceof Error ? error.message : "storage unavailable"
  return `Could not use browser storage (${detail}). Editing still works in this tab.`
}

function isScenario(value: unknown): value is ScenarioDraft {
  if (!isRecord(value)) return false
  return value.version === 1
    && typeof value.id === "string"
    && typeof value.name === "string"
    && Number.isSafeInteger(value.nextOrder)
    && Array.isArray(value.batches)
    && value.batches.every((batch: unknown) => isRecord(batch)
      && typeof batch.id === "string"
      && typeof batch.label === "string"
      && Number.isFinite(batch.arrivalOffsetMs)
      && Number.isSafeInteger(batch.createdOrder)
      && Array.isArray(batch.samples)
      && batch.samples.every((sample: unknown) => isRecord(sample)
        && typeof sample.id === "string"
        && typeof sample.label === "string"
        && typeof sample.prompt === "string"
        && Number.isFinite(sample.maxOutputTokens)))
}

function isSettings(value: unknown): value is EngineSettings {
  if (!isRecord(value) || (value.engine !== "fixture" && value.engine !== "vllm")) return false
  // Missing fields in older drafts inherit defaults; malformed present fields
  // must not reach components that call string or array methods on them.
  for (const [key, fallback] of Object.entries(defaultSettings())) {
    if (key in value && !Array.isArray(fallback) && typeof value[key] !== typeof fallback) return false
  }
  const choices: Record<string, readonly string[]> = {
    hostKind: ["local", "ssh"],
    modelSource: ["huggingface", "host", "operator"],
    dtype: ["auto", "float16", "bfloat16", "float32"],
    quantization: ["none", "awq", "gptq", "bitsandbytes", "fp8"],
  }
  for (const [key, options] of Object.entries(choices)) {
    if (key in value && !options.includes(value[key] as string)) return false
  }
  if ("stopStrings" in value && (!Array.isArray(value.stopStrings)
    || !value.stopStrings.every((item: unknown) => typeof item === "string"))) return false
  if ("loraAdapters" in value && (!Array.isArray(value.loraAdapters)
    || !value.loraAdapters.every((item: unknown) => isRecord(item)
      && ["id", "name", "location", "revision"].every((key) => typeof item[key] === "string")
      && choices.modelSource.includes(item.source as string)))) return false
  return true
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

export function normalizeScenario(scenario: ScenarioDraft): ScenarioDraft {
  return {
    ...scenario,
    batches: scenario.batches.map((batch) => ({
      ...batch,
      defaultAdapterId: typeof batch.defaultAdapterId === "string" ? batch.defaultAdapterId : null,
      samples: batch.samples.map((sample) => ({
        ...sample,
        inputMode: sample.inputMode === "synthetic" ? "synthetic" : "prompt",
        syntheticInputTokens: Number.isSafeInteger(sample.syntheticInputTokens) && sample.syntheticInputTokens > 0
          ? sample.syntheticInputTokens
          : 128,
        resolvedInputTokens: Number.isSafeInteger(sample.resolvedInputTokens) ? sample.resolvedInputTokens : null,
        adapterId: typeof sample.adapterId === "string" ? sample.adapterId : null,
      })),
    })),
  }
}
