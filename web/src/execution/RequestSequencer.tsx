import { useLayoutEffect, useMemo, useRef, useState } from "react"

import type { ReplayState } from "../../../replay/src/reduce.ts"
import type { IndexedStep, SampleProjection } from "../../../replay/src/timeline.ts"
import { arrivalGroup } from "../explore/model.ts"
import { cn } from "../lib/cn.ts"
import { EvidenceTag } from "../ui/controls.tsx"
import type { Selection } from "./types.ts"

const LABEL_WIDTH = 220
const ROW_HEIGHT = 44

interface Column {
  id: string
  ordinal: number
  sequence: number
  reached: boolean
  total: number | null
  budget: number | null
}

export function RequestSequencer({ samples, steps, allSteps, colors, selection, highlightedIds, sequence, onSelect }: {
  samples: SampleProjection[]
  steps: ReplayState["steps"]
  allSteps: IndexedStep[]
  colors: Record<string, string>
  selection: Selection | null
  highlightedIds: Set<string>
  sequence: number
  onSelect: (selection: Selection) => void
}) {
  const scroller = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(900)
  useLayoutEffect(() => {
    const element = scroller.current
    if (!element) return
    const update = () => setWidth(element.clientWidth)
    update()
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const columns = useMemo<Column[]>(() => {
    const reached = new Map(steps.map((step) => [step.id, step]))
    const source = allSteps.length ? allSteps : steps.map((step) => ({ id: step.id, sequence: step.sequence, tokenBudget: step.tokenBudget, totalScheduledTokens: step.totalScheduledTokens, participants: [] }))
    return source.map((step, index) => {
      const observed = reached.get(step.id)
      const isReached = Boolean(observed) && step.sequence <= sequence
      return {
        id: step.id,
        ordinal: index + 1,
        sequence: step.sequence,
        reached: isReached,
        total: isReached ? observed!.totalScheduledTokens : null,
        budget: isReached ? observed!.tokenBudget : null,
      }
    })
  }, [allSteps, sequence, steps])
  const columnIndex = useMemo(() => new Map(columns.map((column, index) => [column.id, index])), [columns])
  const reachedCount = columns.filter((column) => column.reached).length
  const currentIndex = reachedCount - 1
  const columnWidth = Math.max(34, Math.min(64, Math.floor((width - LABEL_WIDTH - 2) / Math.max(1, columns.length))))
  const trackWidth = columnWidth * Math.max(1, columns.length)
  const waitingByStep = useMemo(() => new Map(steps.map((step) => [step.id, new Set((step.queueAfter?.waiting ?? []).map((entry) => entry.requestId))])), [steps])
  const groups = useMemo(() => groupByBatch(samples), [samples])
  const current = currentIndex >= 0 ? steps.find((step) => step.id === columns[currentIndex].id) ?? null : null
  const selectedStep = selection?.kind === "step" ? steps.find((step) => step.id === selection.id) ?? current : current

  useLayoutEffect(() => {
    const element = scroller.current
    if (!element || currentIndex < 0) return
    const right = LABEL_WIDTH + (currentIndex + 1) * columnWidth
    if (right > element.scrollLeft + element.clientWidth - 24 || right < element.scrollLeft + LABEL_WIDTH) {
      element.scrollLeft = Math.max(0, right - element.clientWidth + columnWidth * 3)
    }
  }, [currentIndex, columnWidth])

  if (samples.length === 0) {
    return <p className="m-0 border border-dashed border-line-strong p-6 text-center text-sm text-muted">Add requests on the Configure page to see a lane for each one.</p>
  }

  const labels = new Map(samples.map((item) => [item.sample.id, item.sample.label]))
  const step = (index: number) => columns[index]
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2.5">
      <div ref={scroller} className="cp-roll flex-1" role="region" aria-label="Request lanes by scheduler step" tabIndex={0}>
        <div className="cp-roll-grid" style={{ gridTemplateColumns: `${LABEL_WIDTH}px ${trackWidth}px` }}>
          <div className="cp-roll-head col-span-2">
            <div className="cp-roll-corner flex shrink-0 flex-col justify-center px-2.5" style={{ width: LABEL_WIDTH, height: 46 }}>
              <span className="cp-label">Request</span>
              <span className="cp-label text-[#3d4f6c]">{columns.length} steps{current ? ` · budget ${current.tokenBudget}` : ""}</span>
            </div>
            <div className="relative flex" style={{ width: trackWidth }}>
              {columns.map((column, index) => (
                <button
                  key={column.id}
                  type="button"
                  className={cn(
                    "grid h-[46px] min-h-0 shrink-0 content-between rounded-none border-0 border-l border-line px-1 py-1 text-left font-mono",
                    column.reached ? "bg-transparent" : "cp-hatch cursor-default",
                    index === currentIndex && "bg-accent-400/[0.05] shadow-[inset_0_-2px_0_theme(colors.accent.400)]",
                  )}
                  style={{ width: columnWidth }}
                  disabled={!column.reached}
                  aria-label={column.reached ? `${column.id}: ${column.total} of ${column.budget} tokens scheduled` : `${column.id} has not been played yet`}
                  onClick={() => onSelect({ kind: "step", id: column.id })}
                >
                  <span className={cn("text-[10px] font-semibold", column.reached ? index === currentIndex ? "text-accent" : "text-muted" : "text-[#33445f]")}>S{column.ordinal}</span>
                  {column.reached && columnWidth >= 44 ? <span className="truncate text-[8.5px] text-faint">{column.total} tok</span> : <span />}
                  <span className="relative h-1 bg-[#0b1527]">
                    {column.reached && column.budget ? <i className={cn("absolute inset-y-0 left-0 min-w-0.5", index === currentIndex ? "bg-accent" : "bg-[#4b6283]")} style={{ width: `${Math.min(100, Math.max(4, (column.total! / Math.max(1, ...columns.filter((item) => item.reached).map((item) => item.total ?? 0))) * 100))}%` }} /> : null}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {groups.map((group) => (
            <div key={group.batchId} className="contents">
              <div className="cp-roll-label col-span-1 flex items-center gap-2 border-t border-line-strong bg-[#0a1220] px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-[0.1em] text-faint">
                <b className="font-semibold tracking-[0.06em] text-ink">{group.batchLabel}</b>
                <span className="truncate">{group.recorded ? `${group.arrivalOffsetMs == null ? "arrival time not planned" : `t+${group.arrivalOffsetMs} ms`} · ` : ""}{group.samples.length}</span>
              </div>
              <div className="border-t border-line-strong bg-[#0a1220]" />
              {group.samples.map((item) => {
                const color = colors[item.sample.id] ?? "#7a8799"
                const selected = (selection?.kind === "request" && selection.id === item.sample.id) || (selection?.kind === "token" && selection.requestId === item.sample.id)
                const highlighted = highlightedIds.has(item.sample.id)
                return (
                  <div key={item.sample.id} className="contents">
                    <button
                      type="button"
                      className={cn(
                        "cp-roll-label grid min-h-0 grid-cols-[3px_minmax(0,1fr)_auto] grid-rows-2 items-center gap-x-2.5 rounded-none border-0 border-t border-line px-2.5 py-1 text-left",
                        selected && "bg-accent-400/[0.06] shadow-[inset_2px_0_0_theme(colors.accent.400)]",
                      )}
                      style={{ height: ROW_HEIGHT }}
                      aria-pressed={selected}
                      data-sample-id={item.sample.id}
                      onClick={() => onSelect({ kind: "request", id: item.sample.id })}
                    >
                      <i className="row-span-2 h-7 w-[3px]" style={{ background: color }} aria-hidden="true" />
                      <span className="truncate font-mono text-xs font-semibold text-ink">{item.sample.label}</span>
                      <span className="row-span-1 justify-self-end"><PhaseTag item={item} /></span>
                      <span className="truncate font-mono text-[9.5px] text-faint">{item.sample.id}{item.sample.maxOutputTokens != null ? ` · max ${item.sample.maxOutputTokens}` : ""}</span>
                      <span className="justify-self-end font-mono text-[10px] text-muted">{item.acceptedTokenCount ?? "—"}{item.sample.maxOutputTokens != null ? `/${item.sample.maxOutputTokens}` : ""} out</span>
                    </button>
                    <div
                      className={cn("cp-roll-track border-t border-line", (selected || highlighted) && "bg-accent-400/[0.025]")}
                      style={{ height: ROW_HEIGHT, backgroundSize: `${columnWidth}px 100%` }}
                    >
                      {reachedCount < columns.length ? <div className="cp-hatch absolute inset-y-0 opacity-60" style={{ left: reachedCount * columnWidth, right: 0 }} aria-hidden="true" /> : null}
                      {columns.slice(0, reachedCount).map((column, index) => {
                        if (!waitingByStep.get(column.id)?.has(item.sample.id)) return null
                        if (item.scheduled.some((segment) => segment.stepId === column.id)) return null
                        return (
                          <span key={`wait-${column.id}`} className="cp-note-wait absolute top-1/2 grid h-[22px] -translate-y-1/2 place-items-center font-mono text-[8.5px] tracking-[0.06em]" style={{ left: index * columnWidth + 3, width: columnWidth - 6, ["--note" as string]: color }} aria-hidden="true">
                            {columnWidth >= 44 ? "WAIT" : ""}
                          </span>
                        )
                      })}
                      {item.scheduled.map((segment) => {
                        const index = columnIndex.get(segment.stepId)
                        if (index == null || !step(index)?.reached) return null
                        const prefill = segment.kind !== "decode"
                        const text = prefill ? (columnWidth >= 44 ? `${segment.rangeStart}–${segment.rangeEnd}` : "") : (columnWidth >= 30 ? String(segment.rangeStart) : "")
                        return (
                          <button
                            key={`${segment.stepId}-${segment.rangeStart}`}
                            type="button"
                            className={cn(
                              "absolute top-1/2 grid min-h-0 -translate-y-1/2 place-items-center rounded-none p-0 font-mono text-[9.5px] font-semibold",
                              prefill ? "cp-note-prefill h-6 border-0" : "cp-note-decode h-[22px]",
                            )}
                            style={{
                              left: index * columnWidth + (prefill ? 3 : Math.max(3, (columnWidth - 22) / 2)),
                              width: prefill ? columnWidth - 6 : Math.min(22, columnWidth - 6),
                              ["--note" as string]: color,
                            }}
                            title={`${item.sample.label} · ${segment.stepId} · ${segment.kind} positions ${segment.rangeStart}–${segment.rangeEnd} (${segment.promptTokens} prompt, ${segment.generationTokens} generation)`}
                            aria-label={`${item.sample.label} in ${segment.stepId}: ${segment.kind} positions ${segment.rangeStart} to ${segment.rangeEnd}`}
                            onClick={() => onSelect({ kind: "step", id: segment.stepId })}
                          >
                            {text}
                          </button>
                        )
                      })}
                      {currentIndex >= 0 ? <span className="pointer-events-none absolute inset-y-0 w-[1.5px] bg-accent" style={{ left: (currentIndex + 1) * columnWidth - 1 }} aria-hidden="true" /> : null}
                    </div>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      <StepDetail step={selectedStep} current={current} columns={columns} labels={labels} colors={colors} steps={steps} onSelect={onSelect} />
    </div>
  )
}

function PhaseTag({ item }: { item: SampleProjection }) {
  const kind = item.phase === "completed" ? "neutral"
    : item.phase === "unknown" ? "unknown"
      : item.phase === "cancelled" || item.phase === "preempted" ? "warning"
        : item.phase === "decode" || item.phase === "prefill" || item.phase === "mixed" ? "derived"
          : "neutral"
  return <EvidenceTag kind={kind}>{item.phaseLabel}</EvidenceTag>
}

function StepDetail({ step, current, columns, labels, colors, steps, onSelect }: {
  step: ReplayState["steps"][number] | null
  current: ReplayState["steps"][number] | null
  columns: Column[]
  labels: Map<string, string>
  colors: Record<string, string>
  steps: ReplayState["steps"]
  onSelect: (selection: Selection) => void
}) {
  const reached = columns.filter((column) => column.reached)
  const recent = reached.slice(-32)
  const peak = Math.max(1, ...recent.map((column) => column.total ?? 0))
  if (!step) {
    return <p className="m-0 border border-line bg-surface px-3 py-2.5 text-xs text-muted">No scheduler step has reached this playhead yet. Play or step forward to see batch membership.</p>
  }
  const ordinal = columns.find((column) => column.id === step.id)?.ordinal ?? null
  const chip = (requestId: string, waiting: boolean) => (
    <button key={`${waiting ? "w" : "r"}-${requestId}`} type="button" className="inline-flex min-h-0 items-center gap-1.5 border-line-strong bg-surface px-2 py-0.5 font-mono text-[10px] text-muted" onClick={() => onSelect({ kind: "request", id: requestId })}>
      <i className={cn("size-[7px]", waiting && "border bg-transparent")} style={waiting ? { borderColor: colors[requestId] } : { background: colors[requestId] ?? "#7a8799" }} aria-hidden="true" />
      {labels.get(requestId) ?? requestId}
    </button>
  )
  const queues = step.queueBefore && step.queueAfter ? step : null
  return (
    <div className="grid gap-2.5 md:grid-cols-[minmax(0,1fr)_minmax(0,1.35fr)_minmax(0,1.4fr)]">
      <section className="grid content-start gap-2 border border-line bg-surface px-3 py-2.5 shadow-[inset_0_2px_0_theme(colors.accent.400)]" aria-label="Scheduler step detail">
        <div className="flex items-center gap-2"><span className="cp-label text-muted">{step.id === current?.id ? "Current step" : "Selected step"}</span><span className="cp-rule" /><EvidenceTag kind="observed">Observed</EvidenceTag></div>
        <p className="m-0 flex items-baseline gap-2.5"><span className="font-mono text-2xl font-medium text-ink">S{ordinal ?? "?"}</span><span className="font-mono text-xs text-muted">{step.id}</span></p>
        <dl className="facts m-0">
          <div><dt>Scheduled tokens</dt><dd>{step.totalScheduledTokens} / {step.tokenBudget}</dd></div>
          <div><dt>Participants</dt><dd>{step.requests.length} · {step.requests.filter((request) => request.generationTokens > 0).length} decode</dd></div>
          <div><dt>Preempted</dt><dd>{step.preemptedRequestIds.length ? step.preemptedRequestIds.map((id) => labels.get(id) ?? id).join(", ") : "None recorded"}</dd></div>
        </dl>
      </section>
      <section className="grid content-start gap-2 border border-line bg-surface px-3 py-2.5" aria-label="Queue snapshot">
        <div className="flex items-center gap-2"><span className="cp-label text-muted">Queue snapshot</span><span className="cp-rule" /><span className="cp-label">{step.policy ? `policy ${step.policy}` : "policy not captured"}</span></div>
        {queues ? (["queueBefore", "queueAfter"] as const).map((key) => (
          <div key={key} className="grid grid-cols-[52px_minmax(0,1fr)] items-start gap-2">
            <span className="cp-label pt-1">{key === "queueBefore" ? "Before" : "After"}</span>
            <span className="flex flex-wrap gap-1">
              {step[key]!.running.map((entry) => chip(entry.requestId, false))}
              {step[key]!.waiting.map((entry) => chip(entry.requestId, true))}
              {!step[key]!.running.length && !step[key]!.waiting.length ? <span className="text-xs text-faint">Empty</span> : null}
            </span>
          </div>
        )) : <p className="m-0 text-xs text-muted">Ordered scheduler queues were not captured at this step.</p>}
        <p className="m-0 text-[11.5px] text-faint">Filled = running · hollow = waiting. Skip reasons stay unknown unless recorded.</p>
      </section>
      <section className="grid content-start gap-2 border border-line bg-surface px-3 py-2.5" aria-label="Tokens scheduled per step">
        <div className="flex items-center gap-2"><span className="cp-label text-muted">Tokens / step</span><span className="cp-rule" /><span className="cp-label">{steps.at(-1) ? `budget ${steps.at(-1)!.tokenBudget}` : ""}</span></div>
        <div className="flex h-16 items-end gap-[3px] border-b border-line-strong" aria-hidden="true">
          {recent.map((column) => (
            <button
              key={column.id}
              type="button"
              className={cn("min-h-0 flex-1 rounded-none border-0 p-0", column.id === step.id ? "bg-accent" : "bg-[#3a5070] hover:bg-[#4b6283]")}
              style={{ height: `${Math.max(4, ((column.total ?? 0) / peak) * 100)}%` }}
              tabIndex={-1}
              title={`${column.id}: ${column.total} tokens`}
              onClick={() => onSelect({ kind: "step", id: column.id })}
            />
          ))}
        </div>
        <p className="m-0 font-mono text-[10px] text-faint">{recent.length ? `S${recent[0].ordinal}–S${recent.at(-1)!.ordinal} · peak ${peak} tokens` : "No step reached"}</p>
      </section>
    </div>
  )
}

function groupByBatch(samples: SampleProjection[]): Array<{ batchId: string; batchLabel: string; recorded: boolean; arrivalOffsetMs: number | null; samples: SampleProjection[] }> {
  const groups: Array<{ batchId: string; batchLabel: string; recorded: boolean; arrivalOffsetMs: number | null; samples: SampleProjection[] }> = []
  for (const item of samples) {
    const group = arrivalGroup(item.sample)
    const previous = groups[groups.length - 1]
    if (previous && previous.batchId === group.key) {
      previous.samples.push(item)
      continue
    }
    groups.push({
      batchId: group.key,
      batchLabel: group.label,
      recorded: group.recorded,
      arrivalOffsetMs: item.sample.arrivalOffsetMs,
      samples: [item],
    })
  }
  return groups
}

export function SequencerLegend() {
  const item = "flex items-center gap-1.5"
  return (
    <ul className="m-0 flex list-none flex-wrap items-center gap-x-4 gap-y-1 p-0 font-mono text-[10.5px] text-muted" aria-label="Lane legend">
      <li className={item}><i className="h-2.5 w-4 bg-[#6f86a6]" aria-hidden="true" />Prefill range</li>
      <li className={item}><i className="size-2.5 border border-[#8fa5c4] bg-[#8fa5c4]/15" aria-hidden="true" />Decode position</li>
      <li className={item}><i className="h-2.5 w-4 border border-dashed border-[#6b7a92]" aria-hidden="true" />Waiting</li>
      <li className={item}><i className="cp-hatch h-2.5 w-4 border border-[#2a3c58]" aria-hidden="true" />Unplayed</li>
      <li className={item}><i className="h-3 w-[1.5px] bg-accent" aria-hidden="true" />Playhead</li>
    </ul>
  )
}
