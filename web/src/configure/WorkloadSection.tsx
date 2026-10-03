import type { KeyboardEvent } from "react"

import type { Limits } from "../api.ts"
import { cn } from "../lib/cn.ts"
import { sampleCount, type ScenarioAction } from "../scenario/reducer.ts"
import { formatTime } from "../scenario/time.ts"
import type { EngineSettings, ScenarioDraft, TimeUnit } from "../scenario/types.ts"
import type { FieldIssue } from "../scenario/validate.ts"
import { resolveBatchTime, resolvePositiveInteger } from "../scenario/workloadFields.ts"
import {
  buttonClass,
  ConfigureSection,
  dangerButtonClass,
  Details,
  FieldError,
  FieldHint,
  inputClass,
} from "./ui.tsx"

const BASE_OVERRIDE = "__base__"

export function WorkloadSection({
  scenario,
  settings,
  limits,
  issues,
  unit,
  onUnit,
  onAction,
  onInvalid,
}: {
  scenario: ScenarioDraft
  settings: EngineSettings
  limits: Limits | null
  issues: FieldIssue[]
  unit: TimeUnit
  onUnit: (unit: TimeUnit) => void
  onAction: (action: ScenarioAction) => void
  onInvalid: (issue: FieldIssue) => void
}) {
  const count = sampleCount(scenario)
  const latest = scenario.batches.reduce((value, batch) => Math.max(value, batch.arrivalOffsetMs), 0)
  const summary = `${scenario.batches.length} ${plural(scenario.batches.length, "batch", "batches")} · ${count} ${plural(count, "request", "requests")} · ${formatTime(latest, unit)} ${unit}`
  const adapters = settings.loraAdapters

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

  return (
    <ConfigureSection
      number={1}
      title="Query batch sequence"
      summary={summary}
      status={count > 0 ? "Draft ready" : "Needs requests"}
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
      <div className="flex flex-col gap-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h3 className="text-balance text-sm font-semibold text-slate-200">Timed arrivals</h3>
            <p className="mt-1 max-w-2xl text-pretty text-sm text-slate-400">
              Requests in one batch are dispatched concurrently at the planned offset. Recorded scheduler events show how vLLM groups the work.
            </p>
          </div>
          <fieldset>
            <legend className="mb-1 text-xs font-medium text-slate-400">Display times in</legend>
            <div className="flex rounded-md border border-slate-700 bg-slate-950 p-1">
              {(["ms", "s"] as const).map((option) => (
                <label key={option} className={cn(
                  "cursor-pointer rounded px-3 py-1 text-xs font-medium focus-within:ring-2 focus-within:ring-blue-500",
                  unit === option ? "bg-slate-700 text-slate-100" : "text-slate-400",
                )}>
                  <input
                    className="sr-only"
                    type="radio"
                    name="configure-time-unit"
                    checked={unit === option}
                    onChange={() => onUnit(option)}
                  />
                  {option}
                </label>
              ))}
            </div>
          </fieldset>
        </div>

        {issueFor("sample-count") ? <FieldError id="sample-count-error" message={issueFor("sample-count")} /> : null}

        {scenario.batches.length === 0 ? (
          <div className="rounded-lg border border-dashed border-slate-700 bg-slate-950 p-6 text-center">
            <p className="text-pretty text-sm text-slate-400">This sequence has no arrivals yet.</p>
            <button className={cn(buttonClass, "mt-4")} type="button" id="add-batch" onClick={() => onAction({ type: "add-batch" })}>
              Add the first batch
            </button>
            <FieldError id="add-batch-error" message={issueFor("add-batch")} />
          </div>
        ) : (
          <ol className="space-y-3" aria-label="Arrival batches">
            {scenario.batches.map((batch, batchIndex) => {
              const timeError = issueFor(`batch-time-${batch.id}`)
              const labelError = issueFor(`batch-label-${batch.id}`)
              const batchHasIssue = issues.some((issue) => (
                issue.fieldId.includes(batch.id)
                || batch.samples.some((sample) => issue.fieldId.includes(sample.id))
              ))
              return (
                <li key={batch.id}>
                  <Details className="group/batch rounded-lg border border-slate-800 bg-slate-950" forceOpen={batchHasIssue}>
                    <summary className="cursor-pointer list-none rounded-lg px-4 py-3 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500 [&::-webkit-details-marker]:hidden">
                      <div className="flex flex-wrap items-center gap-3">
                        <span className="flex size-7 shrink-0 items-center justify-center rounded-md bg-slate-800 text-xs font-semibold tabular-nums text-slate-300" aria-hidden="true">
                          {batchIndex + 1}
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-sm font-medium text-slate-200">{batch.label || "Untitled batch"}</span>
                          <span className="mt-0.5 block text-xs tabular-nums text-slate-400">
                            t = {formatTime(batch.arrivalOffsetMs, unit)} {unit} · {batch.samples.length} {plural(batch.samples.length, "request", "requests")}
                          </span>
                        </span>
                        <span className="text-xs text-slate-400 group-open/batch:hidden">Edit</span>
                        <span className="hidden text-xs text-slate-400 group-open/batch:inline">Done</span>
                      </div>
                    </summary>
                    <div className="space-y-5 border-t border-slate-800 p-4">
                      <div className="grid items-start gap-4 md:grid-cols-2">
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
                            className={cn(inputClass, "tabular-nums")}
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
                      </div>

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

                      {batch.samples.length === 0 ? (
                        <div className="rounded-md border border-dashed border-slate-700 p-4 text-center">
                          <p className="text-pretty text-sm text-slate-400">No requests arrive in this batch.</p>
                          <button className={cn(buttonClass, "mt-3")} type="button" id={`add-sample-${batch.id}`} onClick={() => onAction({ type: "add-sample", batchId: batch.id })}>
                            Add a request
                          </button>
                          <FieldError id={`add-sample-${batch.id}-error`} message={issueFor(`add-sample-${batch.id}`)} />
                        </div>
                      ) : (
                        <ul className="space-y-2" aria-label={`Requests in ${batch.label || batch.id}`}>
                          {batch.samples.map((sample, requestIndex) => (
                            <li key={sample.id}>
                              <Details
                                className="group/request rounded-md border border-slate-800 bg-slate-900"
                                forceOpen={issues.some((issue) => issue.fieldId.includes(sample.id))}
                              >
                                <summary className="cursor-pointer list-none rounded-md px-3 py-3 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500 [&::-webkit-details-marker]:hidden">
                                  <div className="flex items-center gap-3">
                                    <span className="flex size-6 shrink-0 items-center justify-center rounded-full border border-slate-700 text-xs tabular-nums text-slate-400" aria-hidden="true">{requestIndex + 1}</span>
                                    <span className="min-w-0 flex-1 truncate text-sm text-slate-200">{sample.label || "Untitled request"}</span>
                                    <span className="text-xs tabular-nums text-slate-400">
                                      {sample.inputMode === "synthetic" ? `${sample.syntheticInputTokens} input` : tokenLabel(sample.resolvedInputTokens)} · {sample.maxOutputTokens} max output
                                    </span>
                                  </div>
                                </summary>
                                <div className="space-y-4 border-t border-slate-800 p-3">
                                  <RequestEditor
                                    scenario={scenario}
                                    batchId={batch.id}
                                    sample={sample}
                                    adapters={adapters}
                                    issues={issues}
                                    onAction={onAction}
                                    commitPositiveInteger={commitPositiveInteger}
                                  />
                                </div>
                              </Details>
                            </li>
                          ))}
                        </ul>
                      )}

                      <div className="flex flex-wrap gap-2 border-t border-slate-800 pt-4">
                        <button className={buttonClass} type="button" id={`add-sample-${batch.id}`} onClick={() => onAction({ type: "add-sample", batchId: batch.id })}>
                          Add request
                        </button>
                        <button className={buttonClass} type="button" onClick={() => onAction({ type: "duplicate-batch", batchId: batch.id })}>
                          Duplicate batch
                        </button>
                        <button className={dangerButtonClass} type="button" onClick={() => onAction({ type: "delete-batch", batchId: batch.id })}>
                          Delete batch
                        </button>
                      </div>
                    </div>
                  </Details>
                </li>
              )
            })}
          </ol>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <button className={buttonClass} type="button" id="add-batch" onClick={() => onAction({ type: "add-batch" })}>
            Add batch
          </button>
          <span className="text-xs tabular-nums text-slate-400">
            {limits ? `${count} of ${limits.max_requests} requests` : `${count} requests`}
          </span>
        </div>
      </div>
    </ConfigureSection>
  )
}

function RequestEditor({
  scenario,
  batchId,
  sample,
  adapters,
  issues,
  onAction,
  commitPositiveInteger,
}: {
  scenario: ScenarioDraft
  batchId: string
  sample: ScenarioDraft["batches"][number]["samples"][number]
  adapters: EngineSettings["loraAdapters"]
  issues: FieldIssue[]
  onAction: (action: ScenarioAction) => void
  commitPositiveInteger: (fieldId: string, raw: string, onCommit: (value: number) => void) => void
}) {
  const issueFor = (id: string) => issues.find((issue) => issue.fieldId === id)?.message ?? ""
  const labelError = issueFor(`sample-label-${sample.id}`)
  const promptError = issueFor(`sample-prompt-${sample.id}`)
  const syntheticError = issueFor(`sample-synthetic-${sample.id}`)
  const outputError = issueFor(`sample-output-${sample.id}`)

  return (
    <>
      <div className="grid items-start gap-4 md:grid-cols-2">
        <label className="block text-xs font-medium text-slate-400" htmlFor={`sample-label-${sample.id}`}>
          Request label
          <input
            id={`sample-label-${sample.id}`}
            className={inputClass}
            value={sample.label}
            aria-required="true"
            aria-invalid={Boolean(labelError)}
            aria-describedby={labelError ? `sample-label-${sample.id}-error` : undefined}
            onChange={(event) => onAction({ type: "set-sample-label", sampleId: sample.id, label: event.target.value })}
          />
          <FieldError id={`sample-label-${sample.id}-error`} message={labelError} />
        </label>
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
      </div>

      {sample.inputMode === "synthetic" ? (
        <label className="block text-xs font-medium text-slate-400" htmlFor={`sample-synthetic-${sample.id}`}>
          Target input tokens
          <input
            key={`${sample.id}-synthetic-${sample.syntheticInputTokens}`}
            id={`sample-synthetic-${sample.id}`}
            className={cn(inputClass, "tabular-nums")}
            type="number"
            min={1}
            step={1}
            defaultValue={sample.syntheticInputTokens}
            aria-required="true"
            aria-invalid={Boolean(syntheticError)}
            aria-describedby={`${syntheticError ? `sample-synthetic-${sample.id}-error ` : ""}sample-synthetic-${sample.id}-hint`.trim()}
            onBlur={(event) => commitPositiveInteger(
              `sample-synthetic-${sample.id}`,
              event.target.value,
              (value) => onAction({ type: "set-sample-synthetic-length", sampleId: sample.id, syntheticInputTokens: value }),
            )}
            onKeyDown={blurOnEnter}
          />
          <FieldHint id={`sample-synthetic-${sample.id}-hint`}>
            Before t = 0, the workload driver must generate and verify this many tokens with the selected tokenizer.
          </FieldHint>
          <FieldError id={`sample-synthetic-${sample.id}-error`} message={syntheticError} />
        </label>
      ) : (
        <label className="block text-xs font-medium text-slate-400" htmlFor={`sample-prompt-${sample.id}`}>
          Prompt
          <textarea
            id={`sample-prompt-${sample.id}`}
            className={cn(inputClass, "min-h-24 resize-y")}
            value={sample.prompt}
            aria-required="true"
            aria-invalid={Boolean(promptError)}
            aria-describedby={`${promptError ? `sample-prompt-${sample.id}-error ` : ""}sample-prompt-${sample.id}-hint`.trim()}
            onChange={(event) => onAction({ type: "set-sample-prompt", sampleId: sample.id, prompt: event.target.value })}
          />
          <FieldHint id={`sample-prompt-${sample.id}-hint`}>
            {sample.resolvedInputTokens == null
              ? "Exact input tokens will be measured with the selected tokenizer during preparation."
              : `${sample.resolvedInputTokens} input tokens with the selected tokenizer.`}
          </FieldHint>
          <FieldError id={`sample-prompt-${sample.id}-error`} message={promptError} />
        </label>
      )}

      <div className="grid items-start gap-4 md:grid-cols-2">
        <label className="block text-xs font-medium text-slate-400" htmlFor={`sample-output-${sample.id}`}>
          Maximum output tokens
          <input
            key={`${sample.id}-output-${sample.maxOutputTokens}`}
            id={`sample-output-${sample.id}`}
            className={cn(inputClass, "tabular-nums")}
            type="number"
            min={1}
            step={1}
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
          <FieldError id={`sample-output-${sample.id}-error`} message={outputError} />
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
      </div>

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

      <div className="flex flex-wrap gap-2 border-t border-slate-800 pt-4">
        <button className={buttonClass} type="button" onClick={() => onAction({ type: "duplicate-sample", batchId, sampleId: sample.id })}>
          Duplicate request
        </button>
        <button className={dangerButtonClass} type="button" onClick={() => onAction({ type: "delete-sample", batchId, sampleId: sample.id })}>
          Delete request
        </button>
      </div>
    </>
  )
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

function tokenLabel(value: number | null) {
  return value == null ? "tokens pending" : `${value} input`
}
