import { Archive, ArrowLeft, Play } from "lucide-react"
import { useId } from "react"

import type { CatalogEntry, LoadedRecording } from "../api.ts"
import { cn } from "../lib/cn.ts"
import {
  DEMO_MODELS, KV_FORMAT_LABELS, LORA_MODE_LABELS, recordingConfiguration, WEIGHT_FORMAT_LABELS,
  type DemoSelection,
} from "./demoCatalog.ts"
import { demoOptions, normalizeDemoSelection, resolveDemo, type DemoFacet } from "./exploreDemos.ts"

export function ExploreToolbar({
  catalog, selection, onSelection, onLoad, loaded, isDemo, loading, error, onBackToDemos, onMoreRecordings,
}: {
  catalog: CatalogEntry[]
  selection: DemoSelection
  onSelection: (selection: DemoSelection) => void
  onLoad: () => void
  loaded: LoadedRecording | null
  isDemo: boolean
  loading: string
  error: string
  onBackToDemos: () => void
  onMoreRecordings: () => void
}) {
  const id = useId()
  const selectedDemo = resolveDemo(catalog, selection)
  const observed = loaded ? recordingConfiguration(loaded.manifest) : null
  const pending = isDemo && observed !== null && (
    observed.modelId.replace(/-FP8$/, "") !== selection.modelId || observed.weightFormat !== selection.weightFormat
    || observed.kvFormat !== selection.kvFormat || observed.loraMode !== selection.loraMode
  )
  const summary = observed
    ? `${modelLabel(observed.modelId)} · Weights ${WEIGHT_FORMAT_LABELS[observed.weightFormat]} · KV ${KV_FORMAT_LABELS[observed.kvFormat]} · LoRA ${LORA_MODE_LABELS[observed.loraMode]}`
    : "No recording loaded"
  const buttonClass = "inline-flex h-9 shrink-0 items-center justify-center gap-2 border border-line-strong bg-raised px-3 text-xs font-medium text-ink hover:border-accent/60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50"
  const fields: { key: DemoFacet; label: string }[] = [
    { key: "modelId", label: "Model" }, { key: "weightFormat", label: "Weight precision" },
    { key: "kvFormat", label: "KV-cache dtype" }, { key: "loraMode", label: "LoRA" },
  ]
  return (
    <section aria-label={isDemo ? "Demo configuration" : "Your recording"} aria-busy={Boolean(loading)} className="shrink-0 border-b border-line bg-panel px-4 py-2 xl:h-28">
      <div className="flex min-h-14 flex-wrap items-end gap-3 xl:flex-nowrap">
        {isDemo ? (
          <fieldset disabled={Boolean(loading)} className="contents">
            {fields.map(({ key, label }) => {
              const options = demoOptions(catalog, selection, key)
              return (
                <label key={key} htmlFor={`${id}-${key}`} className={cn("grid min-w-0 flex-1 gap-1 text-xs text-muted", key === "modelId" ? "basis-64 xl:max-w-80" : "basis-32 xl:max-w-56")}>
                  {label}
                  <select
                    id={`${id}-${key}`}
                    className="h-9 w-full min-w-0 border border-line-strong bg-raised px-2 text-xs text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50"
                    value={selection[key]}
                    aria-describedby={`${id}-status`}
                    disabled={Boolean(loading) || options.length === 0}
                    onChange={(event) => onSelection(normalizeDemoSelection(catalog, { ...selection, [key]: event.target.value }, key))}
                  >
                    {options.length === 0 ? <option value={selection[key]}>Unavailable</option> : options.map((value) => <option key={value} value={value}>{facetLabel(key, value)}</option>)}
                  </select>
                </label>
              )
            })}
            <button type="button" className={cn(buttonClass, "border-accent/60 text-accent")} onClick={onLoad} disabled={Boolean(loading) || !selectedDemo} aria-describedby={`${id}-status ${id}-pending`}>
              <Play aria-hidden="true" className="size-3.5" />Load demo
            </button>
            <span id={`${id}-pending`} role="status" className="inline-flex h-9 w-32 shrink-0 items-center text-xs text-amber-300">{pending && !loading ? "Changes not loaded" : ""}</span>
          </fieldset>
        ) : (
          <div className="flex min-w-0 flex-1 items-center gap-4 self-center">
            <span className="shrink-0 font-mono text-sm text-ink">Your recording</span>
            <button type="button" className={buttonClass} onClick={onBackToDemos} disabled={Boolean(loading)}><ArrowLeft aria-hidden="true" className="size-3.5" />Back to demos</button>
          </div>
        )}
        <button type="button" className={cn(buttonClass, "ml-auto")} onClick={onMoreRecordings} disabled={Boolean(loading)}><Archive aria-hidden="true" className="size-3.5" />More recordings</button>
      </div>
      <div className="mt-2 flex min-h-6 min-w-0 items-center gap-4 text-xs">
        <p className="m-0 min-w-0 flex-1 truncate text-muted" title={`Viewing: ${summary}`}><span className="text-ink">Viewing:</span> {summary}</p>
        <p id={`${id}-status`} role={error ? "alert" : "status"} aria-live={error ? "assertive" : "polite"} className={cn("m-0 min-w-0 max-w-xl truncate text-pretty", error ? "text-rose-300" : "text-muted")} title={error || loading || undefined}>
          {error || loading || (isDemo && !selectedDemo ? "No demo available for this configuration" : "")}
        </p>
      </div>
    </section>
  )
}

function modelLabel(modelId: string): string {
  const family = modelId.replace(/-FP8$/, "")
  return DEMO_MODELS.find((model) => model.id === family)?.label ?? modelId.split("/").at(-1) ?? modelId
}

function facetLabel(key: DemoFacet, value: string): string {
  if (key === "modelId") return modelLabel(value)
  if (key === "weightFormat") return WEIGHT_FORMAT_LABELS[value as DemoSelection["weightFormat"]]
  if (key === "kvFormat") return KV_FORMAT_LABELS[value as DemoSelection["kvFormat"]]
  return LORA_MODE_LABELS[value as DemoSelection["loraMode"]]
}
