import { useState } from "react"

import type { FeatureCapability, Limits } from "../api.ts"
import type { ScenarioAction } from "../scenario/reducer.ts"
import type { EngineSettings, ScenarioDraft, TimeUnit } from "../scenario/types.ts"
import type { FieldIssue } from "../scenario/validate.ts"
import { resolveWorkloadFields } from "../scenario/workloadFields.ts"
import { focusValidationField } from "./focusValidation.ts"
import { ModelSection } from "./ModelSection.tsx"
import { ServerSection } from "./ServerSection.tsx"
import type { ConfigureRuntimeInputs, ReadinessState, SetupStage } from "./types.ts"
import { buttonClass, primaryButtonClass, StatusPill } from "./ui.tsx"
import { VllmSection } from "./VllmSection.tsx"
import { WorkloadSection } from "./WorkloadSection.tsx"

export interface ConfigurePageProps {
  scenario: ScenarioDraft
  settings: EngineSettings
  limits: Limits | null
  issues: FieldIssue[]
  unit: TimeUnit
  storageNote?: string
  storageError?: string
  hostReadiness?: ReadinessState
  modelReadiness?: ReadinessState
  quantizationCapabilities?: Record<string, FeatureCapability>
  vllmReadiness?: ReadinessState
  setupStages?: SetupStage[]
  effectiveSettings?: Record<string, string | number | boolean | null>
  checkingHost?: boolean
  running: boolean
  runBlockedReason?: string
  runStatus?: string
  runError?: string
  onUnit: (unit: TimeUnit) => void
  onAction: (action: ScenarioAction) => void
  onSettings: (settings: EngineSettings) => void
  onInvalid: (issue: FieldIssue) => void
  onCheckHost?: (runtime: ConfigureRuntimeInputs) => void
  onSshIdentityChange?: () => void
  onRun: (scenario: ScenarioDraft, runtime: ConfigureRuntimeInputs) => void
  onCancel?: () => void
  canUndo?: boolean
  onUndo?: () => void
  onStartEmpty?: () => void
  onClearDraft?: () => void
}

export function ConfigurePage({
  scenario,
  settings,
  limits,
  issues,
  unit,
  storageNote,
  storageError,
  hostReadiness,
  modelReadiness,
  quantizationCapabilities,
  vllmReadiness,
  setupStages,
  effectiveSettings,
  checkingHost = false,
  running,
  runBlockedReason,
  runStatus,
  runError,
  onUnit,
  onAction,
  onSettings,
  onInvalid,
  onCheckHost,
  onSshIdentityChange,
  onRun,
  onCancel,
  canUndo = false,
  onUndo,
  onStartEmpty,
  onClearDraft,
}: ConfigurePageProps) {
  const [sshIdentity, setSshIdentity] = useState<File | null>(null)
  const [huggingFaceToken, setHuggingFaceToken] = useState("")
  const runtime: ConfigureRuntimeInputs = {
    sshIdentity,
    huggingFaceToken,
  }
  const requestCount = scenario.batches.reduce((sum, batch) => sum + batch.samples.length, 0)

  function removeAdapter(adapterId: string) {
    onSettings({
      ...settings,
      loraAdapters: settings.loraAdapters.filter((adapter) => adapter.id !== adapterId),
    })
    onAction({
      type: "replace",
      scenario: {
        ...scenario,
        batches: scenario.batches.map((batch) => ({
          ...batch,
          defaultAdapterId: batch.defaultAdapterId === adapterId ? null : batch.defaultAdapterId,
          samples: batch.samples.map((sample) => ({
            ...sample,
            adapterId: sample.adapterId === adapterId ? null : sample.adapterId,
          })),
        })),
      },
    })
  }

  return (
    <main className="min-h-[calc(100dvh-4rem)] bg-slate-950 text-slate-100">
      <form
        className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-8 sm:px-6 lg:px-8"
        aria-busy={running || checkingHost}
        noValidate
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.nativeEvent.isComposing && event.target instanceof HTMLInputElement
            && ["text", "search", "tel", "url", "email", "password", "date", "month", "week", "time", "datetime-local", "number"].includes(event.target.type)) {
            event.preventDefault()
          }
        }}
        onSubmit={(event) => {
          event.preventDefault()
          if (running || checkingHost) return
          const form = event.currentTarget
          const result = resolveWorkloadFields(scenario, unit, (fieldId) => {
            const input = form.elements.namedItem(fieldId)
            return input instanceof HTMLInputElement ? input.value : undefined
          })
          if ("issue" in result) {
            onInvalid(result.issue)
            focusValidationField(result.issue.fieldId)
            return
          }
          onRun(result.scenario, runtime)
        }}
      >
        <header className="mb-2 flex flex-col gap-5 border-b border-slate-800 pb-7 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <p className="text-xs font-semibold uppercase text-blue-400">Configure · Capture · Explore</p>
            <h1 className="mt-2 text-balance text-3xl font-semibold text-slate-50 sm:text-4xl">Build an observed vLLM capture</h1>
            <p className="mt-3 text-pretty text-base leading-7 text-slate-400">
              Define timed requests, prepare the serving host, and capture instrumented vLLM boundaries. Opening this page never starts a run.
            </p>
          </div>
          <div className="flex shrink-0 flex-col items-start gap-3 lg:items-end">
            <div className="flex flex-wrap gap-2" aria-label="Draft summary">
              <StatusPill tone={issues.length > 0 ? "warning" : "ready"}>{issues.length > 0 ? `${issues.length} to review` : "Draft ready"}</StatusPill>
              <StatusPill tone="neutral">{scenario.batches.length} batches</StatusPill>
              <StatusPill tone="neutral">{requestCount} requests</StatusPill>
            </div>
            {(onUndo || onStartEmpty || onClearDraft) ? (
              <div className="flex flex-wrap gap-2" aria-label="Draft actions">
                {onUndo ? <button className={buttonClass} type="button" disabled={!canUndo} onClick={onUndo}>Undo deletion</button> : null}
                {onStartEmpty ? <button className={buttonClass} type="button" onClick={onStartEmpty}>Start empty</button> : null}
                {onClearDraft ? <button className={buttonClass} type="button" onClick={onClearDraft}>Reset example</button> : null}
              </div>
            ) : null}
            {storageNote || storageError ? (
              <p className={`max-w-sm text-pretty text-right text-xs ${storageError ? "text-red-300" : "text-slate-400"}`} role={storageError ? "alert" : "status"}>
                {storageError || storageNote}
              </p>
            ) : null}
          </div>
        </header>

        <WorkloadSection
          scenario={scenario}
          settings={settings}
          limits={limits}
          issues={issues}
          unit={unit}
          onUnit={onUnit}
          onAction={onAction}
          onInvalid={onInvalid}
        />
        <fieldset disabled={running || checkingHost} className="min-w-0">
          <ServerSection
            settings={settings}
            issues={issues}
            readiness={hostReadiness}
            stages={setupStages}
            checking={checkingHost}
            sshIdentity={sshIdentity}
            onSshIdentity={(file) => {
              setSshIdentity(file)
              onSshIdentityChange?.()
            }}
            onSettings={onSettings}
            onCheck={() => onCheckHost?.(runtime)}
          />
        </fieldset>
        <ModelSection
          settings={settings}
          issues={issues}
          readiness={modelReadiness}
          quantizationCapabilities={quantizationCapabilities}
          huggingFaceToken={huggingFaceToken}
          onSettings={onSettings}
          onHuggingFaceToken={setHuggingFaceToken}
          onRemoveAdapter={removeAdapter}
        />
        <VllmSection
          settings={settings}
          issues={issues}
          readiness={vllmReadiness}
          effectiveSettings={effectiveSettings}
          onSettings={onSettings}
        />

        <footer className="safe-sticky-footer sticky bottom-0 z-10 mt-2 rounded-xl border border-slate-700 bg-slate-900 p-4 shadow-xl">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <p id={runBlockedReason ? "capture-blocked-reason" : undefined} className="text-sm font-medium text-slate-200">
                {running ? runStatus || "Preparing capture" : runBlockedReason || "Ready to prepare and capture"}
              </p>
              <p className="mt-1 text-pretty text-xs text-slate-400" role="status">
                {storageError || storageNote || "Your non-secret draft is saved on this device."}
              </p>
              {runError ? <p className="mt-1 text-pretty text-xs text-red-300" role="alert">{runError}</p> : null}
            </div>
            <div className="flex shrink-0 flex-wrap gap-2">
              {running && onCancel ? (
                <button className="inline-flex min-h-10 items-center justify-center rounded-md border border-slate-700 bg-slate-950 px-4 py-2 text-sm font-medium text-slate-200 outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:opacity-50" type="button" onClick={onCancel}>
                  Cancel capture
                </button>
              ) : null}
              <button className={primaryButtonClass} type="submit" disabled={running || checkingHost || Boolean(runBlockedReason)} aria-describedby={runBlockedReason ? "capture-blocked-reason" : undefined}>
                {running ? "Capture in progress…" : "Prepare and capture"}
              </button>
            </div>
          </div>
        </footer>
      </form>
    </main>
  )
}
