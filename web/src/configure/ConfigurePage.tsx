import { Play, Radio } from "lucide-react"
import { useState } from "react"

import type { FeatureCapability, Limits } from "../api.ts"
import { learningExample, LEARNING_EXAMPLES, matchingLearningExample, type LearningExampleId } from "../scenario/learningExamples.ts"
import type { ScenarioAction } from "../scenario/reducer.ts"
import { formatTime } from "../scenario/time.ts"
import type { EngineSettings, ScenarioDraft, TimeUnit } from "../scenario/types.ts"
import type { FieldIssue } from "../scenario/validate.ts"
import { resolveWorkloadFields } from "../scenario/workloadFields.ts"
import { ConfirmDialog } from "../shell/ConfirmDialog.tsx"
import { Chip, Led } from "../ui/controls.tsx"
import { ConsoleRail, openConfigureSection, type ConsoleModule } from "./ConsoleRail.tsx"
import { focusValidationField } from "./focusValidation.ts"
import { ModelSection } from "./ModelSection.tsx"
import { PreflightPanel } from "./PreflightPanel.tsx"
import { ServerSection } from "./ServerSection.tsx"
import type { ConfigureRuntimeInputs, ReadinessState, SetupStage } from "./types.ts"
import { buttonClass, primaryButtonClass } from "./ui.tsx"
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
  openingDemo?: boolean
  demoError?: string
  onOpenDemo: () => void
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

const WORKLOAD_FIELDS = ["sample-count", "add-batch", "batch-", "add-sample-", "sample-"]

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
  openingDemo = false,
  demoError,
  onOpenDemo,
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
  const [pendingExample, setPendingExample] = useState<LearningExampleId | null>(null)
  const runtime: ConfigureRuntimeInputs = {
    sshIdentity,
    huggingFaceToken,
  }
  const busy = running || checkingHost || openingDemo
  const requestCount = scenario.batches.reduce((sum, batch) => sum + batch.samples.length, 0)
  const latest = scenario.batches.reduce((value, batch) => Math.max(value, batch.arrivalOffsetMs), 0)
  const workloadIssue = issues.some((issue) => WORKLOAD_FIELDS.some((prefix) => issue.fieldId === prefix || issue.fieldId.startsWith(prefix)))
  const modelName = settings.model.split("/").at(-1) || "No model"
  const loadedExample = matchingLearningExample(scenario, settings)
  const modules: ConsoleModule[] = [
    { number: 1, title: "Workload", summary: `${scenario.batches.length} batch${scenario.batches.length === 1 ? "" : "es"} · ${requestCount} req · ${formatTime(latest, unit)} ${unit}`, tone: workloadIssue || requestCount === 0 ? "warning" : "ready" },
    { number: 2, title: "Execution host", summary: settings.hostKind === "local" ? `Local machine${hostReadiness?.tone === "warning" ? " · needs GPU" : ""}` : `${settings.sshUser || "user"}@${settings.sshHost || "remote host"}`, tone: hostReadiness?.tone ?? "neutral" },
    { number: 3, title: "Model", summary: modelName, tone: modelReadiness?.tone ?? "neutral" },
    { number: 4, title: "vLLM engine", summary: `${settings.maxModelLen} ctx · ${settings.maxNumSeqs} seq · ${settings.tokenBudget} tok`, tone: vllmReadiness?.tone ?? "neutral" },
  ]
  const launchTone = running || openingDemo ? "accent" : runError ? "error" : runBlockedReason ? "warning" : "ready"
  const launchTitle = openingDemo ? "Opening the recorded GPU demo"
    : running ? runStatus || "Preparing capture"
    : runBlockedReason ? "Live vLLM capture is blocked on this host"
      : "Ready to prepare and capture"
  const launchDetail = runBlockedReason
    ? `${runBlockedReason} You can open the recorded GPU demo below.`
    : storageError || storageNote || "Your non-secret draft is saved on this device."

  function applyExample(id: LearningExampleId) {
    const example = learningExample(id, settings)
    onAction({ type: "replace", scenario: example.scenario })
    onSettings(example.settings)
  }

  function requestExample(id: LearningExampleId) {
    if (loadedExample || requestCount === 0) applyExample(id)
    else setPendingExample(id)
  }

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
    <main className="min-h-[calc(100dvh-52px)] bg-[#05080e] text-slate-100">
      <form
        className="mx-auto grid w-full max-w-[1520px] gap-x-6 gap-y-5 px-4 pt-6 sm:px-6 xl:grid-cols-[216px_minmax(0,1fr)_340px]"
        aria-busy={busy}
        noValidate
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.nativeEvent.isComposing && event.target instanceof HTMLInputElement
            && !["file", "submit", "button", "reset"].includes(event.target.type)) {
            event.preventDefault()
          }
        }}
        onSubmit={(event) => {
          event.preventDefault()
          if (busy) return
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
        <ConsoleRail
          modules={modules}
          loadedExample={loadedExample}
          disabled={busy}
          canUndo={canUndo}
          storageNote={storageNote}
          storageError={storageError}
          onLoadExample={requestExample}
          onUndo={onUndo}
          onStartEmpty={onStartEmpty}
          onReset={onClearDraft}
        />

        <div className="flex min-w-0 flex-col gap-4">
          <header className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-800 pb-5">
            <div className="min-w-0 max-w-3xl flex-1 basis-[27rem]">
              <p className="cp-label m-0"><b className="font-medium text-muted">Configure</b> · capture · explore</p>
              <h1 className="mt-2 text-balance text-[28px] font-semibold leading-tight tracking-[-0.015em] text-slate-50">Build an observed vLLM capture</h1>
              <p className="mt-2 text-pretty text-sm leading-6 text-slate-400">
                Define timed requests, prepare the serving host, and capture instrumented vLLM boundaries. Opening this page never starts a run.
              </p>
            </div>
            <div className="flex min-w-0 max-w-full flex-col items-start gap-2.5">
              <div className="flex flex-wrap gap-1.5" aria-label="Draft summary">
                <Chip tone={issues.length > 0 ? "warning" : undefined}><Led tone={issues.length > 0 ? "warning" : "ready"} /><span>{issues.length > 0 ? `${issues.length} to review` : "Draft ready"}</span></Chip>
                <Chip>{scenario.batches.length} {scenario.batches.length === 1 ? "batch" : "batches"}</Chip>
                <Chip>{requestCount} {requestCount === 1 ? "request" : "requests"}</Chip>
              </div>
              {(onUndo || onStartEmpty || onClearDraft) ? (
                <div className="flex flex-wrap gap-1.5 xl:hidden" aria-label="Draft actions">
                  {onUndo ? <button className={buttonClass} type="button" disabled={!canUndo || busy} onClick={onUndo}>Undo deletion</button> : null}
                  {onStartEmpty ? <button className={buttonClass} type="button" disabled={busy} onClick={onStartEmpty}>Start empty</button> : null}
                  {onClearDraft ? <button className={buttonClass} type="button" disabled={busy} onClick={onClearDraft}>Reset example</button> : null}
                </div>
              ) : null}
              {storageError ? (
                <p className="m-0 max-w-sm text-pretty text-xs text-red-300" role="alert">{storageError}</p>
              ) : storageNote ? (
                <p className="m-0 max-w-sm text-pretty text-xs text-slate-400 xl:hidden" role="status">{storageNote}</p>
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
            onSettings={onSettings}
            onLoadExample={requestExample}
          />
          <fieldset disabled={busy} className="min-w-0 border-0 p-0">
            <ServerSection
              settings={settings}
              issues={issues}
              readiness={hostReadiness}
              checking={checkingHost}
              sshIdentity={sshIdentity}
              onSshIdentity={(file) => {
                setSshIdentity(file)
                onSshIdentityChange?.()
              }}
              onSettings={onSettings}
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
            requests={scenario.batches.flatMap((batch) => batch.samples.map((sample) => ({ id: sample.id, label: sample.label })))}
            issues={issues}
            readiness={vllmReadiness}
            onSettings={onSettings}
          />
        </div>

        <PreflightPanel
          readiness={hostReadiness}
          stages={setupStages}
          checking={checkingHost}
          disabled={running || openingDemo}
          effectiveSettings={effectiveSettings}
          onCheck={() => onCheckHost?.(runtime)}
          onUseSsh={settings.hostKind === "local" && hostReadiness?.tone !== "ready" ? () => {
            onSettings({ ...settings, hostKind: "ssh" })
            window.requestAnimationFrame(() => openConfigureSection(2))
          } : undefined}
        />

        <footer className="safe-sticky-footer sticky bottom-0 z-10 col-span-full -mx-4 mt-1 border-t border-slate-800 bg-[#060a11]/95 px-4 pt-3 backdrop-blur sm:-mx-6 sm:px-6">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
            <Led tone={launchTone} />
            <div className="min-w-0 flex-1 basis-64">
              <p className="m-0 text-sm font-medium text-slate-100">{launchTitle}</p>
              <p id={runBlockedReason ? "capture-blocked-reason" : undefined} className="m-0 mt-0.5 text-pretty text-xs text-slate-400" role="status">
                {launchDetail}
              </p>
              {runError ? <p className="m-0 mt-1 text-pretty text-xs text-red-300" role="alert">{runError}</p> : null}
            </div>
            <div className="hidden items-center gap-1.5 2xl:flex" aria-hidden="true">
              <Chip>{requestCount} req</Chip>
              <Chip>{scenario.batches.length} batches</Chip>
              <Chip className="max-w-40"><span>{modelName}</span></Chip>
            </div>
            <button className={buttonClass} type="button" disabled={busy} onClick={onOpenDemo} aria-describedby="recorded-demo-description">
              <Play className="size-3.5" aria-hidden="true" />
              {openingDemo ? "Opening demo…" : "Open real GPU demo"}
            </button>
            {running && onCancel ? (
              <button className={buttonClass} type="button" onClick={onCancel}>
                Cancel capture
              </button>
            ) : null}
            <button className={primaryButtonClass} type="submit" disabled={busy || Boolean(runBlockedReason)} aria-describedby={runBlockedReason ? "capture-blocked-reason" : undefined}>
              <Radio className="size-3.5" aria-hidden="true" />
              {running ? "Capture in progress…" : "Prepare and capture"}
            </button>
          </div>
          <p id="recorded-demo-description" className="m-0 mt-2 text-pretty text-xs text-slate-400">The GPU demo replays a saved run with its recorded workload and settings. Your editable draft is preserved.</p>
          {demoError ? <p className="m-0 mt-1 text-pretty text-xs text-red-300" role="alert">{demoError}</p> : null}
        </footer>
      </form>

      <ConfirmDialog
        open={pendingExample != null}
        onOpenChange={(open) => { if (!open) setPendingExample(null) }}
        title={`Load “${LEARNING_EXAMPLES.find((example) => example.id === pendingExample)?.label ?? "example"}”?`}
        description="This replaces the current workload and applies the example’s scheduling and cache settings. Host and model choices are kept."
        confirmLabel="Load example"
        onConfirm={() => { if (pendingExample) applyExample(pendingExample) }}
      />
    </main>
  )
}
