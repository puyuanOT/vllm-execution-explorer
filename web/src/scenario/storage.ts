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
  if (!value || typeof value !== "object") return false
  const scenario = value as ScenarioDraft
  return scenario.version === 1 && typeof scenario.id === "string" && Array.isArray(scenario.batches)
}

function isSettings(value: unknown): value is EngineSettings {
  if (!value || typeof value !== "object") return false
  const settings = value as EngineSettings
  return settings.engine === "fixture" || settings.engine === "vllm"
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
