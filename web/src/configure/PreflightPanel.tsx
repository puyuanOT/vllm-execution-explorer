import { RefreshCw, Server } from "lucide-react"

import { cn } from "../lib/cn.ts"
import { Led, SectionTitle, type Tone } from "../ui/controls.tsx"
import type { ReadinessState, SetupStage } from "./types.ts"
import { buttonClass, StatusPill } from "./ui.tsx"

const STAGE_TONE: Record<SetupStage["status"], Tone> = { pending: "neutral", active: "warning", complete: "ready", error: "error" }
const STAGE_LABEL: Record<SetupStage["status"], string> = { pending: "Pending", active: "In progress", complete: "Complete", error: "Failed" }

export function PreflightPanel({ readiness, stages, checking, disabled, effectiveSettings, onCheck, onUseSsh }: {
  readiness?: ReadinessState
  stages?: SetupStage[]
  checking: boolean
  disabled: boolean
  effectiveSettings?: Record<string, string | number | boolean | null>
  onCheck?: () => void
  onUseSsh?: () => void
}) {
  return (
    <aside className="grid grid-cols-1 content-start gap-4 xl:sticky xl:top-4 xl:max-h-[calc(100dvh-140px)] xl:self-start xl:overflow-y-auto" aria-label="Preflight">
      <section className="grid gap-3 border border-slate-800 bg-surface p-4" aria-labelledby="preflight-heading">
        <SectionTitle><span id="preflight-heading">Preflight · execution host</span></SectionTitle>
        {readiness?.message ? (
          <div
            className={cn(
              "flex items-start gap-2.5 border p-3",
              readiness.tone === "warning" ? "border-amber-600/40 bg-amber-400/[0.05]" : readiness.tone === "error" ? "border-red-600/45 bg-red-400/[0.05]" : "border-slate-800 bg-slate-950",
            )}
            role={readiness.tone === "error" ? "alert" : "status"}
          >
            <span className="pt-1"><Led tone={readiness.tone} /></span>
            <span className="grid min-w-0 gap-1.5">
              <span><StatusPill tone={readiness.tone}>{readiness.label}</StatusPill></span>
              <span className={cn("text-pretty text-xs leading-5", readiness.tone === "warning" ? "text-amber-200" : readiness.tone === "error" ? "text-red-200" : "text-slate-300")}>{readiness.message}</span>
            </span>
          </div>
        ) : null}
        {stages && stages.length > 0 ? (
          <ol className="m-0 grid list-none border border-slate-800 p-0" aria-label="Server setup progress">
            {stages.map((stage) => (
              <li key={stage.id} className={cn("grid grid-cols-[14px_minmax(0,1fr)_auto] gap-x-2.5 gap-y-0.5 border-t border-slate-800 px-3 py-2.5 first:border-t-0", stage.status === "error" && "bg-red-400/[0.04]")}>
                <span className="pt-1"><Led tone={STAGE_TONE[stage.status]} /></span>
                <span className="text-[13px] text-slate-200">{stage.label}</span>
                <span className="font-mono text-[9.5px] uppercase tracking-[0.08em] text-slate-500">{STAGE_LABEL[stage.status]}</span>
                {stage.detail ? <span className={cn("col-start-2 col-end-4 text-pretty text-xs", stage.status === "error" ? "text-red-200" : "text-slate-400")}>{stage.detail}</span> : null}
              </li>
            ))}
          </ol>
        ) : null}
        <div className="flex flex-wrap items-center gap-2">
          <button className={buttonClass} type="button" disabled={checking || disabled} onClick={onCheck}>
            <RefreshCw className="size-3.5" aria-hidden="true" />
            {checking ? "Checking host…" : readiness?.tone === "error" ? "Retry host check" : "Check host"}
          </button>
          {onUseSsh ? (
            <button className={buttonClass} type="button" disabled={checking || disabled} onClick={onUseSsh}>
              <Server className="size-3.5" aria-hidden="true" />Use SSH GPU host
            </button>
          ) : null}
        </div>
        <p className="m-0 text-pretty text-xs text-slate-400">No execution begins until you start capture.</p>
      </section>

      {effectiveSettings && Object.keys(effectiveSettings).length > 0 ? (
        <section className="border border-slate-800 bg-surface p-4" aria-labelledby="effective-settings-heading">
          <SectionTitle><span id="effective-settings-heading">Planned launch settings</span></SectionTitle>
          <dl className="m-0 grid grid-cols-1">
            {Object.entries(effectiveSettings).map(([label, value]) => (
              <div key={label} className="flex items-baseline justify-between gap-4 border-b border-slate-800 py-1.5 last:border-b-0">
                <dt className="shrink-0 text-xs text-slate-400">{label}</dt>
                <dd className={cn("m-0 min-w-0 truncate text-right font-mono text-xs tabular-nums", value === true ? "text-accent" : "text-slate-200")} title={formatSetting(value)}>{formatSetting(value)}</dd>
              </div>
            ))}
          </dl>
        </section>
      ) : null}
    </aside>
  )
}

function formatSetting(value: string | number | boolean | null): string {
  if (value == null) return "Unknown"
  if (value === true) return "On"
  if (value === false) return "Off"
  return String(value)
}
