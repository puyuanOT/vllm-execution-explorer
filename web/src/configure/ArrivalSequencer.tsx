import { useLayoutEffect, useRef, useState } from "react"

import { formatTime } from "../scenario/time.ts"
import type { ScenarioDraft, TimeUnit } from "../scenario/types.ts"
import { requestColor } from "../theme.ts"

const LABELS = 112
const PREP = 64
const HEIGHT = 136
const TOP = 26

/** Planned dispatch offsets only; vLLM still decides how arrivals join scheduler steps. */
export function ArrivalSequencer({ scenario, unit }: { scenario: ScenarioDraft; unit: TimeUnit }) {
  const container = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(640)
  const rows = scenario.batches.flatMap((batch) => batch.samples.map((sample) => ({ batch, sample })))
  const hasRows = rows.length > 0
  useLayoutEffect(() => {
    const element = container.current
    if (!element) return
    const update = () => setWidth(Math.max(280, element.clientWidth))
    update()
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [hasRows])
  if (!hasRows) return null
  const order = rows.map((row) => row.sample.id)
  const latest = Math.max(0, ...scenario.batches.map((batch) => batch.arrivalOffsetMs))
  const span = niceSpan(latest)
  const left = PREP + 18 + LABELS
  const right = width - 12
  const x = (ms: number) => left + (ms / span) * Math.max(40, right - left - 60)
  const available = HEIGHT - TOP - 22
  const rowHeight = Math.max(2, Math.min(14, Math.floor(available / rows.length)))
  const rowY = (index: number) => TOP + index * rowHeight
  const showLabels = rowHeight >= 10
  const step = tickStep(span)
  const ticks = Array.from({ length: Math.floor(span / step) + 1 }, (_, index) => index * step)
  const bottom = rowY(rows.length) + 2
  const prepBottom = Math.max(bottom, TOP + 48)
  const height = Math.max(HEIGHT, bottom + 20, prepBottom + 8)
  const description = scenario.batches
    .map((batch) => `batch ${batch.label || "untitled"} at ${formatTime(batch.arrivalOffsetMs, unit)} ${unit} with ${batch.samples.length} request${batch.samples.length === 1 ? "" : "s"}`)
    .join("; ")
  let rowIndex = 0
  return (
    <div className="flex flex-col border border-slate-800 bg-[#060a11] md:flex-row">
      <div ref={container} className="relative min-w-0 flex-1 overflow-hidden">
        <svg width={width} height={height} role="img" aria-label={`Arrival sequencer: ${description}`}>
          <defs>
            <pattern id="arrival-prep" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" stroke="#152640" strokeWidth="2" />
            </pattern>
          </defs>
          <rect x={8} y={TOP - 4} width={PREP - 8} height={prepBottom - TOP + 4} fill="url(#arrival-prep)" stroke="#152640" />
          {["LOAD", "WARM", "RESET"].map((text, index) => <text key={text} x={8 + (PREP - 8) / 2} y={TOP + 16 + index * 13} textAnchor="middle" className="fill-[#6b84a8] font-mono text-[8.5px] tracking-[0.08em]">{text}</text>)}
          {ticks.map((tick) => (
            <g key={tick}>
              <line x1={x(tick)} x2={x(tick)} y1={TOP - 4} y2={bottom} stroke={tick === 0 ? "transparent" : "#131e30"} />
              <text x={x(tick)} y={bottom + 12} textAnchor="middle" className="fill-slate-500 font-mono text-[9px]">{formatTime(tick, unit)} {unit}</text>
            </g>
          ))}
          {rows.map(({ sample }, index) => (
            <g key={sample.id}>
              <rect x={left - 6} y={rowY(index)} width={right - left + 6} height={Math.max(1, rowHeight - 2)} fill={index % 2 ? "#060c17" : "#08101d"} />
              {showLabels ? <text x={left - 14} y={rowY(index) + rowHeight - 4} textAnchor="end" className="fill-slate-400 font-mono text-[9px]">{sample.label || "untitled"}</text> : null}
              {showLabels ? <rect x={left - 10} y={rowY(index) + 1} width={3} height={rowHeight - 4} fill={requestColor(order, sample.id)} /> : null}
            </g>
          ))}
          <line x1={x(0)} x2={x(0)} y1={TOP - 12} y2={bottom} stroke="var(--accent)" strokeWidth={1.2} />
          <text x={x(0) + 5} y={TOP - 8} className="fill-accent font-mono text-[9px] tracking-[0.04em]">t=0 · cache verified fresh</text>
          {scenario.batches.map((batch) => {
            const first = rowIndex
            const notes = batch.samples.map((sample) => {
              const y = rowY(rowIndex++)
              return <rect key={sample.id} x={x(batch.arrivalOffsetMs)} y={y} width={44} height={Math.max(1, rowHeight - 4)} fill={requestColor(order, sample.id)} opacity={0.85} />
            })
            const bx = x(batch.arrivalOffsetMs)
            return (
              <g key={batch.id}>
                {batch.arrivalOffsetMs > 0 && batch.samples.length ? <line x1={bx} x2={bx} y1={rowY(first) - 3} y2={rowY(rowIndex) - 1} stroke="#e2eaf4" opacity={0.5} /> : null}
                {notes}
                {batch.samples.length && showLabels ? <text x={bx + 49} y={rowY(first) + rowHeight - 4} className="fill-slate-100 font-mono text-[9px] font-semibold">{batch.label || "?"}<tspan className="fill-slate-500 font-medium"> +{formatTime(batch.arrivalOffsetMs, unit)}{unit}</tspan></text> : null}
              </g>
            )
          })}
        </svg>
      </div>
      <div className="grid shrink-0 content-start gap-1.5 px-3.5 pb-3 md:w-52 md:py-3.5 md:pl-0">
        <span className="cp-label">Arrival sequencer</span>
        <span className="text-pretty text-[11.5px] leading-[1.45] text-faint">Each batch dispatches at its offset after a verified fresh prefix cache. vLLM still decides how requests join scheduler steps.</span>
      </div>
    </div>
  )
}

function niceSpan(latest: number): number {
  const target = Math.max(10, latest * 1.25)
  const magnitude = 10 ** Math.floor(Math.log10(target))
  return [1, 1.5, 2, 2.5, 3, 5, 7.5, 10].map((factor) => factor * magnitude).find((value) => value >= target) ?? 10 * magnitude
}

function tickStep(span: number): number {
  const raw = span / 4
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  return [1, 2, 2.5, 5, 10].map((factor) => factor * magnitude).find((value) => value >= raw) ?? magnitude * 10
}
