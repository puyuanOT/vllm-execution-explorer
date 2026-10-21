import type { CatalogEntry } from "../api.ts"
import type { DemoSelection } from "./demoCatalog.ts"

export const CURATED_DEMO_IDS = [
  "gpu-h100-realistic-bf16",
  "gpu-h100-realistic-fp8-weights",
  "gpu-h100-realistic-fp8-kv",
  "gpu-h100-realistic-lora-all",
  "gpu-h100-realistic-lora-mixed",
  "gpu-h100-realistic-qwen38-27b",
] as const

export const LAST_DEMO_STORAGE_KEY = "vllm-explorer.last-demo.v1"
export const INITIAL_DEMO_SELECTION: DemoSelection = {
  modelId: "Qwen/Qwen3-4B-Instruct-2507",
  weightFormat: "bf16",
  kvFormat: "bf16",
  loraMode: "none",
}

const FACETS = ["modelId", "weightFormat", "kvFormat", "loraMode"] as const
export type DemoFacet = typeof FACETS[number]

/** Only current recordings whose observed configuration can be selected completely. */
export function curatedDemos(catalog: CatalogEntry[]): CatalogEntry[] {
  return CURATED_DEMO_IDS.flatMap((id) => {
    const entry = catalog.find((item) => item.id === id)
    const config = entry?.configuration
    return entry && entry.evidence === "gpu" && config
      && config.weightFormat !== "unknown" && config.kvFormat !== "unknown" && config.loraMode !== "unknown"
      ? [entry] : []
  })
}

export function demoSelection(entry: CatalogEntry): DemoSelection {
  const config = entry.configuration
  return {
    modelId: config?.modelId.replace(/-FP8$/, "") ?? "Unknown model",
    weightFormat: config?.weightFormat ?? "unknown",
    kvFormat: config?.kvFormat ?? "unknown",
    loraMode: config?.loraMode ?? "unknown",
  }
}

export function resolveDemo(catalog: CatalogEntry[], selection: DemoSelection): CatalogEntry | undefined {
  return curatedDemos(catalog).find((entry) => {
    const candidate = demoSelection(entry)
    return FACETS.every((key) => candidate[key] === selection[key])
  })
}

/** Later selectors are constrained by earlier choices; model changes always remain reachable. */
export function demoOptions<K extends DemoFacet>(catalog: CatalogEntry[], selection: DemoSelection, key: K): DemoSelection[K][] {
  const earlier = FACETS.slice(0, FACETS.indexOf(key))
  return [...new Set(curatedDemos(catalog).map(demoSelection)
    .filter((candidate) => earlier.every((facet) => candidate[facet] === selection[facet]))
    .map((candidate) => candidate[key]))]
}

export function normalizeDemoSelection(catalog: CatalogEntry[], selection: DemoSelection, changedKey: DemoFacet): DemoSelection {
  let candidates = curatedDemos(catalog).map(demoSelection)
  if (candidates.length === 0) return { ...INITIAL_DEMO_SELECTION }
  const result = { ...selection }
  const changedIndex = FACETS.indexOf(changedKey)
  for (const [index, key] of FACETS.entries()) {
    const matching = candidates.filter((candidate) => candidate[key] === result[key])
    if (matching.length > 0) {
      candidates = matching
    } else {
      // Prefer the baseline only for facets after the user's change. Earlier invalid
      // values recover to the first available recording instead of creating a wildcard.
      const fallback = index > changedIndex
        ? candidates.find((candidate) => candidate[key] === INITIAL_DEMO_SELECTION[key]) ?? candidates[0]
        : candidates[0]
      Object.assign(result, { [key]: fallback[key] })
      candidates = candidates.filter((candidate) => candidate[key] === fallback[key])
    }
  }
  return result
}

export function initialDemo(catalog: CatalogEntry[]): CatalogEntry | undefined {
  const demos = curatedDemos(catalog)
  try {
    const saved = globalThis.localStorage?.getItem(LAST_DEMO_STORAGE_KEY)
    const previous = demos.find((entry) => entry.id === saved)
    if (previous) return previous
  } catch {
    // Private browsing and storage restrictions must not prevent exploring a demo.
  }
  return demos.find((entry) => entry.id === CURATED_DEMO_IDS[0]) ?? demos[0]
}

/** Call only after loading has succeeded; pending changes do not replace this preference. */
export function rememberDemo(id: string): void {
  if (!CURATED_DEMO_IDS.some((candidate) => candidate === id)) return
  try {
    globalThis.localStorage?.setItem(LAST_DEMO_STORAGE_KEY, id)
  } catch {
    // Demo playback remains available when browser storage is disabled or full.
  }
}
