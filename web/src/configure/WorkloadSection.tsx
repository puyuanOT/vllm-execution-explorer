import { ChevronDown, Copy, Plus, SlidersHorizontal, Trash2 } from "lucide-react"
import { useState, type KeyboardEvent, type ReactNode } from "react"

import type { Limits } from "../api.ts"
import { cn } from "../lib/cn.ts"
import { learningExample, LEARNING_EXAMPLES, type LearningExampleId } from "../scenario/learningExamples.ts"
import { sampleCount, type ScenarioAction } from "../scenario/reducer.ts"
import { formatTime } from "../scenario/time.ts"
import type { EngineSettings, ScenarioDraft, TimeUnit } from "../scenario/types.ts"
import type { FieldIssue } from "../scenario/validate.ts"
import { resolveBatchTime, resolvePositiveInteger, resolvePriority } from "../scenario/workloadFields.ts"
import { requestColor } from "../theme.ts"
import { Segmented } from "../ui/controls.tsx"
import { ArrivalSequencer } from "./ArrivalSequencer.tsx"
import {
  buttonClass,
  compactInputClass,
  ConfigureSection,
  dangerButtonClass,
  Details,
  FieldError,
  iconButtonClass,
  inputClass,
} from "./ui.tsx"

const BASE_OVERRIDE = "__base__"
const ROW_GRID = "grid grid-cols-[3px_minmax(0,1fr)_minmax(0,1fr)] items-start gap-x-2.5 gap-y-2 [@container(min-width:700px)]:grid-cols-[3px_minmax(96px,132px)_minmax(0,1fr)_104px_84px_auto]"

type Sample = ScenarioDraft["batches"][number]["samples"][number]

export function WorkloadSection({
  scenario,
  settings,
  limits,
  issues,
  unit,
  onUnit,
  onAction,
  onInvalid,
  onSettings,
  onLoadExample,
}: {
  scenario: ScenarioDraft
  settings: EngineSettings
  limits: Limits | null
  issues: FieldIssue[]
  unit: TimeUnit
  onUnit: (unit: TimeUnit) => void
  onAction: (action: ScenarioAction) => void
  onInvalid: (issue: FieldIssue) => void
  onSettings: (settings: EngineSettings) => void
  onLoadExample?: (id: LearningExampleId) => void
}) {
  const [exampleId, setExampleId] = useState<LearningExampleId>("staggered")
  const count = sampleCount(scenario)
  const latest = scenario.batches.reduce((value, batch) => Math.max(value, batch.arrivalOffsetMs), 0)
  const summary = `${scenario.batches.length} ${plural(scenario.batches.length, "batch", "batches")} · ${count} ${plural(count, "request", "requests")} · ${formatTime(latest, unit)} ${unit}`
  const adapters = settings.loraAdapters
  const order = scenario.batches.flatMap((batch) => batch.samples.map((sample) => sample.id))

  function issueFor(id: string) {
    return issues.find((issue) => issue.fieldId === id)?.message ?? ""
  }

  function commitPositiveInteger(fieldId: string, raw: string, onCommit: (value: number) => void) {
    const result = resolvePositiveInteger(fieldId, raw)
    if ("issue" in result) {
      onInvalid(result.issue)
      return
    }
    onCommit(result.value)
  }

  function loadExample(id: LearningExampleId) {
    if (onLoadExample) {
      onLoadExample(id)
      return
    }
    const example = learningExample(id, settings)
    onAction({ type: "replace", scenario: example.scenario })
    onSettings(example.settings)
  }

  return (
    <ConfigureSection
      number={1}
      title="Query batch sequence"
      summary={summary}
      status={count > 0 ? "Ready" : "Needs requests"}
      statusTone={count > 0 ? "ready" : "warning"}
      defaultOpen
      forceOpen={issues.some((issue) => (
        issue.fieldId === "sample-count"
        || issue.fieldId === "add-batch"
        || issue.fieldId.startsWith("batch-")
        || issue.fieldId.startsWith("add-sample-")
        || issue.fieldId.startsWith("sample-")
      ))}
    >
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <p className="m-0 min-w-0 max-w-2xl flex-1 basis-72 text-pretty text-xs leading-5 text-slate-400">
            Requests in one batch are dispatched concurrently at the planned offset. Recorded scheduler events show how vLLM groups the work.
          </p>
          <div className="flex items-center gap-2">
            <span className="cp-label" aria-hidden="true">Display</span>
            <Segmented<TimeUnit> label="Display times in" value={unit} onChange={onUnit} options={[{ value: "ms", label: "ms" }, { value: "s", label: "s" }]} />
          </div>
        </div>

        <ArrivalSequencer scenario={scenario} unit={unit} />

        <Details className="border border-slate-800 p-3 xl:hidden">
          <summary className="cursor-pointer text-sm font-medium text-slate-200">Editable learning examples</summary>
          <div className="mt-3 flex flex-wrap items-end gap-3">
            <label className="min-w-48 flex-1 text-xs font-medium text-slate-400" htmlFor="learning-example">Example
              <select id="learning-example" className={inputClass} value={exampleId} onChange={(event) => setExampleId(event.target.value as LearningExampleId)} aria-describedby="learning-example-description">
                {LEARNING_EXAMPLES.map((example) => <option key={example.id} value={example.id}>{example.label}</option>)}
              </select>
            </label>
            <button type="button" className={buttonClass} onClick={() => loadExample(exampleId)}>Use example</button>
          </div>
          <p id="learning-example-description" className="mt-2 text-pretty text-xs text-slate-400">{LEARNING_EXAMPLES.find((example) => example.id === exampleId)?.description} Replaces the draft and applies the example’s scheduling/cache settings. All fields remain editable for live vLLM capture. The recorded GPU demo uses its own saved workload.</p>
        </Details>

        {issueFor("sample-count") ? <FieldError id="sample-count-error" message={issueFor("sample-count")} /> : null}

        {scenario.batches.length === 0 ? (
          <div className="border border-dashed border-slate-700 bg-slate-950 p-6 text-center">
            <p className="m-0 text-pretty text-sm text-slate-400">This sequence has no arrivals yet.</p>
            <button className={cn(buttonClass, "mt-4")} type="button" id="add-batch" onClick={() => onAction({ type: "add-batch" })}>
              <Plus className="size-3.5" aria-hidden="true" />Add the first batch
            </button>
            <FieldError id="add-batch-error" message={issueFor("add-batch")} />
          </div>
        ) : (
          <ol className="m-0 grid min-w-0 grid-cols-1 list-none gap-2 p-0" aria-label="Arrival batches">
            {scenario.batches.map((batch, batchIndex) => {
              const timeError = issueFor(`batch-time-${batch.id}`)
              const labelError = issueFor(`batch-label-${batch.id}`)
              const batchHasIssue = issues.some((issue) => (
                issue.fieldId.includes(batch.id)
                || batch.samples.some((sample) => issue.fieldId.includes(sample.id))
              ))
              return (
                <li key={batch.id}>
                  <Details className="group/batch border border-slate-800 bg-[#0a101c] open:border-slate-700" defaultOpen={batchIndex === 0} forceOpen={batchHasIssue}>
                    <summary className="flex cursor-pointer list-none items-center gap-3 px-3 py-2 outline-none hover:bg-slate-900/60 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-400 [&::-webkit-details-marker]:hidden">
                      <span className="grid size-[26px] shrink-0 place-items-center border border-slate-600 font-mono text-xs font-semibold tabular-nums text-muted group-open/batch:border-accent-400/45 group-open/batch:text-accent" aria-hidden="true">
                        {batchIndex + 1}
                      </span>
                      <span className="min-w-0 truncate font-mono text-xs text-ink">
                        {batch.label || "Untitled batch"}
                        <span className="text-faint"> · t+{formatTime(batch.arrivalOffsetMs, unit)} {unit} · {batch.samples.length} {plural(batch.samples.length, "request", "requests")}</span>
                      </span>
                      <span className="hidden shrink-0 gap-1 sm:flex group-open/batch:hidden" aria-hidden="true">
                        {batch.samples.slice(0, 8).map((sample) => <i key={sample.id} className="size-[7px]" style={{ background: requestColor(order, sample.id) }} />)}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-right font-mono text-[11px] text-faint group-open/batch:invisible">
                        <span className="hidden sm:inline">{batch.samples.map(samplePreview).join(" · ")}</span>
                      </span>
                      <ChevronDown className="size-4 shrink-0 text-faint group-open/batch:rotate-180" aria-hidden="true" />
                    </summary>
                    <div className="border-t border-slate-800 [container-type:inline-size]">
                      <div className={cn("grid items-start gap-3 px-3 py-3 sm:grid-cols-2", adapters.length > 0 ? "md:grid-cols-[minmax(0,240px)_160px_minmax(0,240px)]" : "md:grid-cols-[minmax(0,240px)_160px]")}>
                        <label className="block text-xs font-medium text-slate-400" htmlFor={`batch-label-${batch.id}`}>
                          Batch label
                          <input
                            id={`batch-label-${batch.id}`}
                            className={inputClass}
                            value={batch.label}
                            aria-required="true"
                            aria-invalid={Boolean(labelError)}
                            aria-describedby={labelError ? `batch-label-${batch.id}-error` : undefined}
                            onChange={(event) => onAction({ type: "set-batch-label", batchId: batch.id, label: event.target.value })}
                          />
                          <FieldError id={`batch-label-${batch.id}-error`} message={labelError} />
                        </label>
                        <label className="block text-xs font-medium text-slate-400" htmlFor={`batch-time-${batch.id}`}>
                          Planned send offset ({unit})
                          <input
                            key={`${batch.id}-${batch.arrivalOffsetMs}-${unit}`}
                            id={`batch-time-${batch.id}`}
                            className={cn(inputClass, "font-mono tabular-nums")}
                            inputMode="decimal"
                            defaultValue={formatTime(batch.arrivalOffsetMs, unit)}
                            aria-required="true"
                            aria-invalid={Boolean(timeError)}
                            aria-describedby={timeError ? `batch-time-${batch.id}-error` : undefined}
                            onBlur={(event) => {
                              const result = resolveBatchTime(`batch-time-${batch.id}`, event.target.value, unit)
                              if ("issue" in result) {
                                onInvalid(result.issue)
                                return
                              }
                              onAction({ type: "commit-batch-time", batchId: batch.id, arrivalOffsetMs: result.value })
                            }}
                            onKeyDown={blurOnEnter}
                          />
                          <FieldError id={`batch-time-${batch.id}-error`} message={timeError} />
                        </label>
                        {adapters.length > 0 ? (
                          <label className="block text-xs font-medium text-slate-400" htmlFor={`batch-adapter-${batch.id}`}>
                            Default adapter
                            <select
                              id={`batch-adapter-${batch.id}`}
                              className={inputClass}
                              value={batch.defaultAdapterId ?? ""}
                              onChange={(event) => onAction({
                                type: "set-batch-adapter",
                                batchId: batch.id,
                                adapterId: event.target.value || null,
                              })}
                            >
                              <option value="">Base model</option>
                              {adapters.map((adapter) => <option key={adapter.id} value={adapter.id}>{adapter.name || "Untitled adapter"}</option>)}
                            </select>
                          </label>
                        ) : null}
                      </div>

                      {batch.samples.length === 0 ? (
                        <p className="m-0 border-t border-slate-800 px-3 py-4 text-center text-pretty text-sm text-slate-400">No requests arrive in this batch. Add a request below.</p>
                      ) : (
                        <>
                          <div className={cn(ROW_GRID, "hidden border-t border-slate-800 px-3 pb-1 pt-2 [@container(min-width:700px)]:grid")} aria-hidden="true">
                            <span />
                            <span className="cp-label">Label</span>
                            <span className="cp-label">Prompt or input length</span>
                            <span className="cp-label">Output</span>
                            <span className="cp-label">Priority</span>
                            <span className="w-[88px]" />
                          </div>
                          <ul className="m-0 list-none p-0" aria-label={`Requests in ${batch.label || "untitled batch"}`}>
                            {batch.samples.map((sample) => (
                              <RequestRow
                                key={sample.id}
                                scenario={scenario}
                                batchId={batch.id}
                                sample={sample}
                                color={requestColor(order, sample.id)}
                                adapters={adapters}
                                issues={issues}
                                onAction={onAction}
                                commitPositiveInteger={commitPositiveInteger}
                                priorityPolicy={settings.schedulingPolicy === "priority"}
                                onInvalid={onInvalid}
                              />
                            ))}
                          </ul>
                        </>
                      )}

                      <div className="flex flex-wrap items-center gap-2 border-t border-slate-800 px-3 py-2">
                        <button className={buttonClass} type="button" id={`add-sample-${batch.id}`} onClick={() => onAction({ type: "add-sample", batchId: batch.id })}>
                          <Plus className="size-3.5" aria-hidden="true" />Add request
                        </button>
                        <button className={buttonClass} type="button" onClick={() => onAction({ type: "duplicate-batch", batchId: batch.id })}>
                          Duplicate batch
                        </button>
                        <span className="flex-1" />
                        <button className={dangerButtonClass} type="button" onClick={() => onAction({ type: "delete-batch", batchId: batch.id })}>
                          <Trash2 className="size-3.5" aria-hidden="true" />Delete batch
                        </button>
                      </div>
                      {issueFor(`add-sample-${batch.id}`) ? <div className="px-3 pb-2"><FieldError id={`add-sample-${batch.id}-error`} message={issueFor(`add-sample-${batch.id}`)} /></div> : null}
                    </div>
                  </Details>
                </li>
              )
            })}
          </ol>
        )}

        <div className="flex flex-wrap items-center gap-3">
          {scenario.batches.length > 0 ? (
            <button className={buttonClass} type="button" id="add-batch" onClick={() => onAction({ type: "add-batch" })}>
              <Plus className="size-3.5" aria-hidden="true" />Add batch
            </button>
          ) : null}
          <span className="flex items-center gap-2.5 font-mono text-[10.5px] tabular-nums text-faint">
            {limits ? (
              <span className="relative h-1.5 w-32 border border-line-strong bg-[#0b1527]" aria-hidden="true">
                <i className="absolute inset-y-0 left-0 bg-[#4b6283]" style={{ width: `${Math.min(100, (count / Math.max(1, limits.max_requests)) * 100)}%` }} />
              </span>
            ) : null}
            {limits ? `${count} / ${limits.max_requests} requests` : `${count} ${plural(count, "request", "requests")}`}
          </span>
        </div>
        {scenario.batches.length > 0 && issueFor("add-batch") ? <FieldError id="add-batch-error" message={issueFor("add-batch")} /> : null}
      </div>
    </ConfigureSection>
  )
}

function RequestRow({
  scenario,
  batchId,
  sample,
  color,
  adapters,
  issues,
  onAction,
  commitPositiveInteger,
  priorityPolicy,
  onInvalid,
}: {
  scenario: ScenarioDraft
  batchId: string
  sample: Sample
  color: string
  adapters: EngineSettings["loraAdapters"]
  issues: FieldIssue[]
  onAction: (action: ScenarioAction) => void
  commitPositiveInteger: (fieldId: string, raw: string, onCommit: (value: number) => void) => void
  priorityPolicy: boolean
  onInvalid: (issue: FieldIssue) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const issueFor = (id: string) => issues.find((issue) => issue.fieldId === id)?.message ?? ""
  const labelError = issueFor(`sample-label-${sample.id}`)
  const promptError = issueFor(`sample-prompt-${sample.id}`)
  const syntheticError = issueFor(`sample-synthetic-${sample.id}`)
  const outputError = issueFor(`sample-output-${sample.id}`)
  const priorityError = issueFor(`sample-priority-${sample.id}`)
  const name = sample.label || "untitled request"
  const optionsId = `sample-options-${sample.id}`

  return (
    <li className="border-t border-slate-800 px-3 py-2">
      <div className={ROW_GRID}>
        <i className="row-span-3 w-[3px] self-stretch [@container(min-width:700px)]:row-span-1 [@container(min-width:700px)]:h-7 [@container(min-width:700px)]:self-center" style={{ background: color }} aria-hidden="true" />
        <Cell className="col-start-2 [@container(min-width:700px)]:col-start-auto" label="Request label" htmlFor={`sample-label-${sample.id}`}>
          <input
            id={`sample-label-${sample.id}`}
            className={cn(compactInputClass, "font-mono")}
            value={sample.label}
            aria-required="true"
            aria-invalid={Boolean(labelError)}
            aria-describedby={labelError ? `sample-label-${sample.id}-error` : undefined}
            onChange={(event) => onAction({ type: "set-sample-label", sampleId: sample.id, label: event.target.value })}
          />
        </Cell>
        {sample.inputMode === "synthetic" ? (
          <Cell className="col-span-2 col-start-2 [@container(min-width:700px)]:col-span-1 [@container(min-width:700px)]:col-start-auto" label="Target input tokens" htmlFor={`sample-synthetic-${sample.id}`}>
            <Suffixed suffix="tokens">
              <input
                key={`${sample.id}-synthetic-${sample.syntheticInputTokens}`}
                id={`sample-synthetic-${sample.id}`}
                className={cn(compactInputClass, "[@container(min-width:700px)]:pr-16 font-mono tabular-nums")}
                inputMode="numeric"
                defaultValue={sample.syntheticInputTokens}
                aria-required="true"
                aria-invalid={Boolean(syntheticError)}
                aria-describedby={syntheticError ? `sample-synthetic-${sample.id}-error` : undefined}
                onBlur={(event) => commitPositiveInteger(
                  `sample-synthetic-${sample.id}`,
                  event.target.value,
                  (value) => onAction({ type: "set-sample-synthetic-length", sampleId: sample.id, syntheticInputTokens: value }),
                )}
                onKeyDown={blurOnEnter}
              />
            </Suffixed>
          </Cell>
        ) : (
          <Cell className="col-span-2 col-start-2 [@container(min-width:700px)]:col-span-1 [@container(min-width:700px)]:col-start-auto" label="Prompt" htmlFor={`sample-prompt-${sample.id}`}>
            <textarea
              id={`sample-prompt-${sample.id}`}
              className={cn(compactInputClass, "max-h-40 min-h-[34px] resize-y [field-sizing:content]")}
              rows={1}
              value={sample.prompt}
              aria-required="true"
              aria-invalid={Boolean(promptError)}
              aria-describedby={promptError ? `sample-prompt-${sample.id}-error` : undefined}
              onChange={(event) => onAction({ type: "set-sample-prompt", sampleId: sample.id, prompt: event.target.value })}
            />
          </Cell>
        )}
        <Cell className="col-start-2 [@container(min-width:700px)]:col-start-auto" label="Maximum output tokens" htmlFor={`sample-output-${sample.id}`}>
          <Suffixed suffix="max out">
            <input
              key={`${sample.id}-output-${sample.maxOutputTokens}`}
              id={`sample-output-${sample.id}`}
              className={cn(compactInputClass, "[@container(min-width:700px)]:pr-14 font-mono tabular-nums")}
              inputMode="numeric"
              defaultValue={sample.maxOutputTokens}
              aria-required="true"
              aria-invalid={Boolean(outputError)}
              aria-describedby={outputError ? `sample-output-${sample.id}-error` : undefined}
              onBlur={(event) => commitPositiveInteger(
                `sample-output-${sample.id}`,
                event.target.value,
                (value) => onAction({ type: "set-sample-output", sampleId: sample.id, maxOutputTokens: value }),
              )}
              onKeyDown={blurOnEnter}
            />
          </Suffixed>
        </Cell>
        <Cell label="Request priority" htmlFor={`sample-priority-${sample.id}`}>
          <Suffixed suffix="pri">
            <input
              id={`sample-priority-${sample.id}`}
              key={`${sample.id}-priority-${sample.priority ?? 0}`}
              className={cn(compactInputClass, "[@container(min-width:700px)]:pr-9 font-mono tabular-nums")}
              inputMode="numeric"
              defaultValue={sample.priority ?? 0}
              aria-invalid={Boolean(priorityError)}
              aria-describedby={priorityError ? `sample-priority-${sample.id}-error` : undefined}
              onKeyDown={blurOnEnter}
              onBlur={(event) => {
                const result = resolvePriority(`sample-priority-${sample.id}`, event.target.value)
                if ("issue" in result) onInvalid(result.issue)
                else onAction({ type: "set-sample-priority", sampleId: sample.id, priority: result.value })
              }}
            />
          </Suffixed>
        </Cell>
        <div className="col-start-3 row-start-1 flex items-center justify-end gap-0.5 [@container(min-width:700px)]:col-start-auto [@container(min-width:700px)]:row-start-auto [@container(min-width:700px)]:h-[34px]">
          <button
            type="button"
            className={cn(iconButtonClass, expanded && "text-accent")}
            aria-label={`More options for ${name}`}
            title="Input mode, arrival batch and adapter"
            aria-expanded={expanded}
            aria-controls={optionsId}
            onClick={() => setExpanded((value) => !value)}
          >
            <SlidersHorizontal className="size-3.5" aria-hidden="true" />
          </button>
          <button type="button" className={iconButtonClass} aria-label={`Duplicate ${name}`} title="Duplicate request" onClick={() => onAction({ type: "duplicate-sample", batchId, sampleId: sample.id })}>
            <Copy className="size-3.5" aria-hidden="true" />
          </button>
          <button type="button" className={cn(iconButtonClass, "hover:text-red-300")} aria-label={`Delete ${name}`} title="Delete request" onClick={() => onAction({ type: "delete-sample", batchId, sampleId: sample.id })}>
            <Trash2 className="size-3.5" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="pl-[13px]">
        <FieldError id={`sample-label-${sample.id}-error`} message={labelError} />
        <FieldError id={`sample-prompt-${sample.id}-error`} message={promptError} />
        <FieldError id={`sample-synthetic-${sample.id}-error`} message={syntheticError} />
        <FieldError id={`sample-output-${sample.id}-error`} message={outputError} />
        <FieldError id={`sample-priority-${sample.id}-error`} message={priorityError} />
      </div>

      {expanded ? (
        <div id={optionsId} className="ml-[13px] mt-2 grid items-start gap-3 border border-slate-800 bg-[#060a11] p-3 sm:grid-cols-2 lg:grid-cols-3">
          <label className="block text-xs font-medium text-slate-400" htmlFor={`sample-mode-${sample.id}`}>
            Input mode
            <select
              id={`sample-mode-${sample.id}`}
              className={inputClass}
              value={sample.inputMode}
              onChange={(event) => onAction({
                type: "set-sample-mode",
                sampleId: sample.id,
                inputMode: event.target.value === "synthetic" ? "synthetic" : "prompt",
              })}
            >
              <option value="prompt">Prompt text</option>
              <option value="synthetic">Synthetic token length</option>
            </select>
          </label>
          <label className="block text-xs font-medium text-slate-400" htmlFor={`sample-move-${sample.id}`}>
            Arrival batch
            <select
              id={`sample-move-${sample.id}`}
              className={inputClass}
              value={batchId}
              onChange={(event) => onAction({ type: "move-sample", sampleId: sample.id, toBatchId: event.target.value })}
            >
              {scenario.batches.map((batch) => (
                <option key={batch.id} value={batch.id}>{batch.label || "Untitled batch"} · {batch.arrivalOffsetMs} ms</option>
              ))}
            </select>
          </label>
          {adapters.length > 0 ? (
            <label className="block text-xs font-medium text-slate-400" htmlFor={`sample-adapter-${sample.id}`}>
              Adapter override
              <select
                id={`sample-adapter-${sample.id}`}
                className={inputClass}
                value={sample.adapterId ?? ""}
                onChange={(event) => onAction({
                  type: "set-sample-adapter",
                  sampleId: sample.id,
                  adapterId: event.target.value || null,
                })}
              >
                <option value="">Use batch default</option>
                <option value={BASE_OVERRIDE}>Base model</option>
                {adapters.map((adapter) => <option key={adapter.id} value={adapter.id}>{adapter.name || "Untitled adapter"}</option>)}
              </select>
            </label>
          ) : null}
          <ul className="m-0 grid list-none gap-1 p-0 text-pretty text-xs leading-5 text-slate-400 sm:col-span-full">
            <li>
              {sample.inputMode === "synthetic"
                ? "Before t = 0, the workload driver must generate and verify this many tokens with the selected tokenizer."
                : sample.resolvedInputTokens == null
                  ? "Exact input tokens will be measured with the selected tokenizer during preparation."
                  : `${sample.resolvedInputTokens} input tokens with the selected tokenizer.`}
            </li>
            <li>{priorityPolicy ? "Lower priority numbers are favored. Equal priorities use arrival order." : "Choose request priority scheduling in vLLM configuration to use nonzero priorities."}</li>
          </ul>
        </div>
      ) : null}
    </li>
  )
}

function Cell({ label, htmlFor, className, children }: { label: string; htmlFor: string; className?: string; children: ReactNode }) {
  return (
    <div className={cn("min-w-0", className)}>
      <label className="mb-1 block text-[11px] font-medium text-slate-400 [@container(min-width:700px)]:sr-only" htmlFor={htmlFor}>{label}</label>
      {children}
    </div>
  )
}

function Suffixed({ suffix, children }: { suffix: string; children: ReactNode }) {
  return (
    <div className="relative">
      {children}
      <span className="pointer-events-none absolute right-2 top-[17px] hidden -translate-y-1/2 font-mono text-[9px] [@container(min-width:700px)]:block uppercase tracking-[0.08em] text-faint" aria-hidden="true">{suffix}</span>
    </div>
  )
}

function samplePreview(sample: Sample): string {
  if (sample.inputMode === "synthetic") return `${sample.syntheticInputTokens} synthetic tokens`
  return sample.prompt.trim() || sample.label || "Empty prompt"
}

function blurOnEnter(event: KeyboardEvent<HTMLInputElement>) {
  if (event.key === "Enter" && !event.nativeEvent.isComposing) {
    event.preventDefault()
    event.currentTarget.blur()
  }
}

function plural(count: number, one: string, many: string) {
  return count === 1 ? one : many
}
