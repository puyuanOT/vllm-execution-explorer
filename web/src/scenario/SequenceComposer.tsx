import { useEffect, useRef, useState } from "react"

import type { Limits } from "../api.ts"
import { reduceScenario, sampleCount, type ScenarioAction } from "./reducer.ts"
import type { ArrivalBatchDraft, ScenarioDraft, TimeUnit } from "./types.ts"
import type { FieldIssue } from "./validate.ts"
import { axisSpanMs, formatTime, millisecondsFromAxis, parseTime } from "./time.ts"

export function SequenceComposer({
  scenario,
  limits,
  issues,
  unit,
  onUnit,
  storageNote,
  storageError,
  canUndo,
  onUndo,
  onAction,
  onInvalid,
  onRun,
  onStartEmpty,
  onClearDraft,
  runError,
  running,
  colors,
  selectedSampleId,
  onSelectSample,
}: {
  scenario: ScenarioDraft
  limits: Limits | null
  issues: FieldIssue[]
  unit: TimeUnit
  onUnit: (unit: TimeUnit) => void
  storageNote: string
  storageError: string
  canUndo: boolean
  onUndo: () => void
  onAction: (action: ScenarioAction) => void
  onInvalid: (issue: FieldIssue) => void
  onRun: (scenario: ScenarioDraft) => void
  onStartEmpty: () => void
  onClearDraft: () => void
  runError: string
  running: boolean
  colors: Record<string, string>
  selectedSampleId: string | null
  onSelectSample: (id: string) => void
}) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const [timeText, setTimeText] = useState<Record<string, string>>({})
  const [outputText, setOutputText] = useState<Record<string, string>>({})
  const [focusId, setFocusId] = useState("")
  const count = sampleCount(scenario)
  const cap = limits?.max_requests

  useEffect(() => {
    if (!focusId) return
    document.getElementById(focusId)?.focus()
    setFocusId("")
  }, [focusId, scenario])

  function issueFor(fieldId: string): string {
    return issues.find((issue) => issue.fieldId === fieldId)?.message ?? ""
  }

  function flush(current: ScenarioDraft): { scenario: ScenarioDraft; issue: FieldIssue | null } {
    let next = current
    for (const batch of current.batches) {
      const text = timeText[batch.id]
      if (text != null) {
        const parsed = parseTime(text, unit)
        if (parsed == null) {
          return {
            scenario: current,
            issue: {
              fieldId: `batch-time-${batch.id}`,
              message: unit === "ms"
                ? "Enter a nonnegative whole number of milliseconds."
                : "Enter seconds that land on a whole millisecond, such as 0.1 or 0.12.",
            },
          }
        }
        next = {
          ...next,
          batches: next.batches.map((item) => (
            item.id === batch.id ? { ...item, arrivalOffsetMs: parsed } : item
          )),
        }
      }
      for (const sample of batch.samples) {
        const raw = outputText[sample.id]
        if (raw == null) continue
        if (!/^\d+$/.test(raw.trim())) {
          return {
            scenario: current,
            issue: { fieldId: `sample-output-${sample.id}`, message: "Maximum output tokens must be a positive integer." },
          }
        }
        const parsed = Number(raw)
        next = {
          ...next,
          batches: next.batches.map((item) => ({
            ...item,
            samples: item.samples.map((entry) => (
              entry.id === sample.id ? { ...entry, maxOutputTokens: parsed } : entry
            )),
          })),
        }
      }
    }
    if (next !== current) {
      const ordered = {
        ...next,
        batches: next.batches.slice().sort((left, right) => (
          left.arrivalOffsetMs - right.arrivalOffsetMs || left.createdOrder - right.createdOrder
        )),
      }
      setTimeText({})
      setOutputText({})
      return { scenario: ordered, issue: null }
    }
    return { scenario: current, issue: null }
  }

  function commitTime(batch: ArrivalBatchDraft) {
    const text = timeText[batch.id]
    if (text == null) return
    const parsed = parseTime(text, unit)
    if (parsed == null) return
    onAction({ type: "commit-batch-time", batchId: batch.id, arrivalOffsetMs: parsed })
    setTimeText((current) => {
      const next = { ...current }
      delete next[batch.id]
      return next
    })
  }

  return (
    <section className="composer" aria-label="Arrival sequence">
      <p className="lede">Create arrivals, run the sequence, then follow each sample.</p>
      <div className="composer-tools">
        <p className="meta counts">
          <span>{scenario.batches.length} batches</span>
          <span id="sample-count">{cap == null ? `${count} samples` : `${count} / ${cap} samples`}</span>
        </p>
        <fieldset className="unit-toggle">
          <legend>Time unit</legend>
          <label>
            <input
              type="radio"
              name="time-unit"
              checked={unit === "ms"}
              onChange={() => {
                setTimeText({})
                onUnit("ms")
              }}
            />
            ms
          </label>
          <label>
            <input
              type="radio"
              name="time-unit"
              checked={unit === "s"}
              onChange={() => {
                setTimeText({})
                onUnit("s")
              }}
            />
            s
          </label>
        </fieldset>
      </div>
      <ArrivalOverview
        batches={scenario.batches}
        capMs={(limits?.max_run_seconds ?? 120) * 1000}
        onSelect={(id) => document.getElementById(`batch-${id}`)?.scrollIntoView({ block: "nearest" })}
        onRetime={(batchId, arrivalOffsetMs) => {
          onAction({ type: "commit-batch-time", batchId, arrivalOffsetMs })
          setTimeText((current) => {
            const next = { ...current }
            delete next[batchId]
            return next
          })
        }}
      />
      <p className="meta">Drag a marker to change its arrival time. The arrival field stays editable.</p>
      {issueFor("sample-count") ? <p className="field-error" id="sample-count-error">{issueFor("sample-count")}</p> : null}
      <div className="batch-list">
        {scenario.batches.map((batch) => (
          <article className="batch-card" id={`batch-${batch.id}`} key={batch.id}>
            <header className="batch-head">
              <label>
                Batch label
                <input
                  id={`batch-label-${batch.id}`}
                  value={batch.label}
                  aria-invalid={Boolean(issueFor(`batch-label-${batch.id}`))}
                  aria-describedby={issueFor(`batch-label-${batch.id}`) ? `batch-label-${batch.id}-error` : undefined}
                  onChange={(event) => onAction({ type: "set-batch-label", batchId: batch.id, label: event.target.value })}
                />
              </label>
              <label>
                Arrival
                <input
                  id={`batch-time-${batch.id}`}
                  className="mono"
                  inputMode="decimal"
                  value={timeText[batch.id] ?? formatTime(batch.arrivalOffsetMs, unit)}
                  aria-invalid={Boolean(issueFor(`batch-time-${batch.id}`))}
                  aria-describedby={issueFor(`batch-time-${batch.id}`) ? `batch-time-${batch.id}-error` : undefined}
                  onChange={(event) => setTimeText((current) => ({ ...current, [batch.id]: event.target.value }))}
                  onBlur={() => commitTime(batch)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault()
                      commitTime(batch)
                    }
                  }}
                />
              </label>
              <p className="meta">{batch.samples.length} samples</p>
            </header>
            {issueFor(`batch-label-${batch.id}`) ? <p className="field-error" id={`batch-label-${batch.id}-error`}>{issueFor(`batch-label-${batch.id}`)}</p> : null}
            {issueFor(`batch-time-${batch.id}`) ? <p className="field-error" id={`batch-time-${batch.id}-error`}>{issueFor(`batch-time-${batch.id}`)}</p> : null}
            <ul className="sample-list">
              {batch.samples.map((sample) => {
                const open = expanded === sample.id
                const promptError = issueFor(`sample-prompt-${sample.id}`)
                const outputError = issueFor(`sample-output-${sample.id}`)
                const labelError = issueFor(`sample-label-${sample.id}`)
                return (
                  <li key={sample.id} className={selectedSampleId === sample.id ? "sample-row selected" : "sample-row"}>
                    <div className="sample-main">
                      <button
                        type="button"
                        className="swatch-button"
                        aria-pressed={selectedSampleId === sample.id}
                        onClick={() => onSelectSample(sample.id)}
                      >
                        <span className="swatch" style={{ background: colors[sample.id] ?? "#8b97a8" }} aria-hidden="true" />
                        <span className="sr-only">Select {sample.label || sample.id}</span>
                      </button>
                      <label className="grow">
                        <span className="sr-only">Display label for {sample.label || sample.id}</span>
                        <input
                          id={`sample-label-${sample.id}`}
                          value={sample.label}
                          aria-invalid={Boolean(labelError)}
                          aria-describedby={labelError ? `sample-label-${sample.id}-error` : undefined}
                          onFocus={() => onSelectSample(sample.id)}
                          onChange={(event) => onAction({ type: "set-sample-label", sampleId: sample.id, label: event.target.value })}
                        />
                      </label>
                      <button
                        type="button"
                        aria-expanded={open}
                        aria-controls={`sample-editor-${sample.id}`}
                        aria-label={`${open ? "Close" : "Edit"} ${sample.label || sample.id}`}
                        onClick={() => setExpanded(open ? null : sample.id)}
                      >
                        {open ? "Close" : "Edit"}
                      </button>
                    </div>
                    {labelError ? <p className="field-error" id={`sample-label-${sample.id}-error`}>{labelError}</p> : null}
                    {open ? (
                      <div className="sample-editor" id={`sample-editor-${sample.id}`}>
                        <label>
                          Prompt
                          <textarea
                            id={`sample-prompt-${sample.id}`}
                            rows={3}
                            value={sample.prompt}
                            aria-invalid={Boolean(promptError)}
                            aria-describedby={promptError ? `sample-prompt-${sample.id}-error` : undefined}
                            onChange={(event) => onAction({ type: "set-sample-prompt", sampleId: sample.id, prompt: event.target.value })}
                          />
                        </label>
                        {promptError ? <p className="field-error" id={`sample-prompt-${sample.id}-error`}>{promptError}</p> : null}
                        <label>
                          Maximum output tokens
                          <input
                            id={`sample-output-${sample.id}`}
                            className="mono"
                            inputMode="numeric"
                            value={outputText[sample.id] ?? String(sample.maxOutputTokens)}
                            aria-invalid={Boolean(outputError)}
                            aria-describedby={outputError ? `sample-output-${sample.id}-error` : undefined}
                            onChange={(event) => setOutputText((current) => ({ ...current, [sample.id]: event.target.value }))}
                            onBlur={() => {
                              const raw = outputText[sample.id]
                              if (raw == null || !/^\d+$/.test(raw.trim())) return
                              const parsed = Number(raw)
                              if (parsed < 1) return
                              onAction({ type: "set-sample-output", sampleId: sample.id, maxOutputTokens: parsed })
                              setOutputText((current) => {
                                const next = { ...current }
                                delete next[sample.id]
                                return next
                              })
                            }}
                          />
                        </label>
                        {outputError ? <p className="field-error" id={`sample-output-${sample.id}-error`}>{outputError}</p> : null}
                        <label>
                          Move to batch
                          <select
                            value={batch.id}
                            aria-label={`Move ${sample.label || sample.id} to another batch`}
                            onChange={(event) => onAction({ type: "move-sample", sampleId: sample.id, toBatchId: event.target.value })}
                          >
                            {scenario.batches.map((option) => (
                              <option key={option.id} value={option.id}>
                                {option.label || option.id} · {formatTime(option.arrivalOffsetMs, unit)} {unit}
                              </option>
                            ))}
                          </select>
                        </label>
                        <div className="row-actions">
                          <button
                            type="button"
                            aria-label={`Duplicate ${sample.label || sample.id}`}
                            onClick={() => onAction({ type: "duplicate-sample", batchId: batch.id, sampleId: sample.id })}
                          >
                            Duplicate sample
                          </button>
                          <button
                            type="button"
                            aria-label={`Delete ${sample.label || sample.id}`}
                            onClick={() => {
                              const index = batch.samples.findIndex((item) => item.id === sample.id)
                              const neighbor = batch.samples[index + 1] ?? batch.samples[index - 1]
                              onAction({ type: "delete-sample", batchId: batch.id, sampleId: sample.id })
                              setFocusId(neighbor ? `sample-label-${neighbor.id}` : `add-sample-${batch.id}`)
                            }}
                          >
                            Delete sample
                          </button>
                        </div>
                      </div>
                    ) : null}
                  </li>
                )
              })}
            </ul>
            {issueFor(`add-sample-${batch.id}`) ? (
              <p className="field-error" id={`add-sample-${batch.id}-error`}>{issueFor(`add-sample-${batch.id}`)}</p>
            ) : null}
            <div className="row-actions">
              <button
                type="button"
                id={`add-sample-${batch.id}`}
                aria-describedby={issueFor(`add-sample-${batch.id}`) ? `add-sample-${batch.id}-error` : undefined}
                onClick={() => {
                  onAction({ type: "add-sample", batchId: batch.id })
                }}
              >
                Add sample
              </button>
              <button type="button" aria-label={`Duplicate batch ${batch.label || batch.id}`} onClick={() => onAction({ type: "duplicate-batch", batchId: batch.id })}>
                Duplicate batch
              </button>
              <button
                type="button"
                aria-label={`Delete batch ${batch.label || batch.id}`}
                onClick={() => {
                  const index = scenario.batches.findIndex((item) => item.id === batch.id)
                  const neighbor = scenario.batches[index + 1] ?? scenario.batches[index - 1]
                  onAction({ type: "delete-batch", batchId: batch.id })
                  setFocusId(neighbor ? `batch-label-${neighbor.id}` : "add-batch")
                }}
              >
                Delete batch
              </button>
            </div>
          </article>
        ))}
      </div>
      <div className="composer-actions">
        <button
          type="button"
          id="add-batch"
          onClick={() => {
            const next = reduceScenario(scenario, { type: "add-batch" })
            const added = next.batches.find((batch) => !scenario.batches.some((item) => item.id === batch.id))
            onAction({ type: "replace", scenario: next })
            if (added) setFocusId(`batch-time-${added.id}`)
          }}
        >
          Add batch
        </button>
        {issueFor("add-batch") ? <p className="field-error">{issueFor("add-batch")}</p> : null}
        <button
          type="button"
          className="primary"
          disabled={running}
          onClick={() => {
            const flushed = flush(scenario)
            if (flushed.issue) {
              onInvalid(flushed.issue)
              document.getElementById(flushed.issue.fieldId)?.focus()
              return
            }
            onRun(flushed.scenario)
          }}
        >
          Run sequence
        </button>
        {runError ? <p className="field-error" role="alert">{runError}</p> : null}
        <div className="row-actions">
          <button type="button" onClick={onStartEmpty}>Start empty</button>
          <button type="button" onClick={onUndo} disabled={!canUndo}>Undo delete</button>
          <button type="button" onClick={onClearDraft}>Clear draft</button>
        </div>
        <p className="meta" role="status">{storageError || storageNote}</p>
        {limits ? (
          <p className="meta">
            Runner limits: {limits.max_requests} samples, {limits.max_prompt_characters} prompt characters, {limits.max_output_tokens} output tokens, {limits.max_run_seconds}s.
          </p>
        ) : (
          <p className="meta">Runner limits are not loaded. Local checks still apply, and the server remains authoritative.</p>
        )}
      </div>
    </section>
  )
}

function ArrivalOverview({
  batches,
  capMs,
  onSelect,
  onRetime,
}: {
  batches: ArrivalBatchDraft[]
  capMs: number
  onSelect: (batchId: string) => void
  onRetime: (batchId: string, arrivalOffsetMs: number) => void
}) {
  const axisRef = useRef<HTMLDivElement>(null)
  const gesture = useRef<{ id: string; span: number; originX: number; moved: boolean } | null>(null)
  const [preview, setPreview] = useState<{ id: string; ms: number } | null>(null)
  const [gestureSpan, setGestureSpan] = useState<number | null>(null)
  const displayed = batches.map((batch) => (
    preview?.id === batch.id ? { ...batch, arrivalOffsetMs: preview.ms } : batch
  ))
  const span = gestureSpan ?? axisSpanMs(batches.map((batch) => batch.arrivalOffsetMs), capMs)
  const clusters = clusterBatches(displayed, span)

  function timeAt(clientX: number, axisSpan: number): number {
    const rect = axisRef.current?.getBoundingClientRect()
    if (!rect) return 0
    return millisecondsFromAxis(clientX, rect.left, rect.width, axisSpan, capMs)
  }

  return (
    <div className="overview">
      <div ref={axisRef} className="overview-axis" aria-hidden="true">
        {clusters.map((cluster) => (
          <div
            key={cluster.key}
            className="overview-cluster"
            style={{
              left: `${cluster.percent}%`,
              transform: cluster.percent > 72 ? "translateX(-100%)" : cluster.percent < 12 ? "none" : "translateX(-50%)",
            }}
          >
            {cluster.batches.map((batch) => (
              <button
                key={batch.id}
                type="button"
                tabIndex={-1}
                onPointerDown={(event) => {
                  if (event.button !== 0) return
                  event.currentTarget.setPointerCapture(event.pointerId)
                  gesture.current = { id: batch.id, span, originX: event.clientX, moved: false }
                  setGestureSpan(span)
                }}
                onPointerMove={(event) => {
                  const current = gesture.current
                  if (!current || current.id !== batch.id) return
                  if (!current.moved && Math.abs(event.clientX - current.originX) < 4) return
                  current.moved = true
                  setPreview({ id: batch.id, ms: timeAt(event.clientX, current.span) })
                }}
                onPointerUp={(event) => {
                  const current = gesture.current
                  gesture.current = null
                  setGestureSpan(null)
                  if (!current || current.id !== batch.id) return
                  if (!current.moved) {
                    setPreview(null)
                    onSelect(batch.id)
                    return
                  }
                  const ms = timeAt(event.clientX, current.span)
                  setPreview(null)
                  onRetime(batch.id, ms)
                }}
                onPointerCancel={() => {
                  gesture.current = null
                  setGestureSpan(null)
                  setPreview(null)
                }}
              >
                {batch.label} {batch.arrivalOffsetMs}
              </button>
            ))}
          </div>
        ))}
      </div>
      <ol className="overview-list">
        {batches.map((batch) => (
          <li key={batch.id}>
            <button type="button" onClick={() => onSelect(batch.id)}>
              <span className="mono">{batch.arrivalOffsetMs} ms</span>
              <span>{batch.label}</span>
              <span>{batch.samples.length}</span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  )
}

function clusterBatches(batches: ArrivalBatchDraft[], max: number): Array<{ key: string; percent: number; batches: ArrivalBatchDraft[] }> {
  const sorted = batches.slice().sort((left, right) => left.arrivalOffsetMs - right.arrivalOffsetMs || left.createdOrder - right.createdOrder)
  const clusters: Array<{ key: string; percent: number; batches: ArrivalBatchDraft[] }> = []
  for (const batch of sorted) {
    const percent = (batch.arrivalOffsetMs / max) * 100
    const previous = clusters[clusters.length - 1]
    if (previous && Math.abs(previous.percent - percent) < 18) {
      previous.batches.push(batch)
      continue
    }
    clusters.push({ key: batch.id, percent, batches: [batch] })
  }
  return clusters
}
