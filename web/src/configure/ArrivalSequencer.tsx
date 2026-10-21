import { useId, useLayoutEffect, useRef, useState } from "react"

import { arrivalAxis } from "./arrivalAxis.ts"
import { cn } from "../lib/cn.ts"
import { formatTime } from "../scenario/time.ts"
import type { ScenarioDraft, TimeUnit } from "../scenario/types.ts"
import { requestColor } from "../theme.ts"

const TOP = 48
const DIAMOND = 5

/** Planned dispatch offsets only; vLLM still decides how arrivals join scheduler steps. */
export function ArrivalSequencer({ scenario, unit }: { scenario: ScenarioDraft; unit: TimeUnit }) {
  const container = useRef<HTMLDivElement>(null)
  const descriptionId = useId()
  const [width, setWidth] = useState(640)
  const rows = scenario.batches.flatMap((batch) => batch.samples.map((sample) => ({ batch, sample })))
  const hasRows = rows.length > 0
  useLayoutEffect(() => {
    const element = container.current
    if (!element) return
    const update = () => setWidth(element.clientWidth)
    update()
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [hasRows])
  if (!hasRows) return null

  const compact = width < 440
  const left = compact ? 90 : 110
  const right = Math.max(left + 1, width - 24)
  const rowHeight = compact ? 40 : 32
  const rowY = (index: number) => TOP + index * rowHeight
  const bottom = rowY(rows.length - 1) + 22
  const order = rows.map((row) => row.sample.id)
  const latest = Math.max(0, ...rows.map((row) => row.batch.arrivalOffsetMs))
  const { span, ticks } = arrivalAxis(latest)
  const x = (ms: number) => left + (ms / span) * (right - left)
  const tickSpace = Math.max(...ticks.map((tick) => formatTime(tick, unit).length)) * 7.2 + 16
  const tickStride = Math.max(1, Math.ceil(tickSpace / (x(ticks[1] ?? span) - left)))
  const visibleTicks = ticks.filter((_, index) => index % tickStride === 0)
  const description = scenario.batches.filter((batch) => batch.samples.length > 0)
    .map((batch) => `Arrival batch ${batch.label || "untitled"} dispatches ${batch.samples.map((sample) => sample.label || "untitled request").join(", ")} at ${formatTime(batch.arrivalOffsetMs, unit)} ${unit}.`)
    .join(" ")
  let rowIndex = 0

  return (
    <div className="min-w-0 border border-slate-800 bg-[#060a11] p-3 sm:p-4">
      <div className="border-b border-slate-800 pb-3">
        <span className="cp-label">Planned request arrivals</span>
        <p className="m-0 mt-1 text-pretty text-xs leading-5 text-slate-400">One diamond per request · shared arrival time per arrival batch</p>
      </div>
      <p id={descriptionId} className="sr-only">Diamonds mark planned dispatch instants, not execution durations. Vertical brackets group requests in an arrival batch. {description}</p>
      <div ref={container} className="mt-3 max-h-96 min-w-0 overflow-y-auto" role="region" aria-label="Planned request arrivals" aria-describedby={descriptionId} tabIndex={0}>
        <svg width={width} height={bottom} role="img" aria-label="Request dispatch diamonds" aria-describedby={descriptionId}>
          <title>Planned dispatch instants for each request</title>
          {visibleTicks.map((tick) => <line key={tick} x1={x(tick)} x2={x(tick)} y1={TOP - 18} y2={bottom} stroke="#172334" />)}
          <text x={left} y={TOP - 36} className="fill-accent font-mono text-xs">t=0</text>
          {rows.map(({ batch, sample }, index) => {
            const px = x(batch.arrivalOffsetMs)
            const py = rowY(index)
            return (
              <g key={sample.id}>
                <line x1={left} x2={right} y1={py} y2={py} stroke="#172334" />
                <foreignObject x={0} y={py - 10} width={left - 14} height={22}>
                  <div className="truncate text-right font-mono text-xs leading-5 text-slate-400" title={sample.label || "Untitled request"}>{sample.label || "Untitled request"}</div>
                </foreignObject>
                <path d={`M${px},${py - DIAMOND} L${px + DIAMOND},${py} L${px},${py + DIAMOND} L${px - DIAMOND},${py} Z`} fill={requestColor(order, sample.id)}>
                  <title>{sample.label || "Untitled request"}: planned dispatch at {formatTime(batch.arrivalOffsetMs, unit)} {unit}, arrival batch {batch.label || "untitled"}</title>
                </path>
              </g>
            )
          })}
          {scenario.batches.map((batch) => {
            if (!batch.samples.length) return null
            const first = rowIndex
            rowIndex += batch.samples.length
            const bx = x(batch.arrivalOffsetMs)
            const inline = right - bx - 22 >= Math.min(150, Math.max(80, formatTime(batch.arrivalOffsetMs, unit).length * 7.2 + 56))
            const labelWidth = inline ? Math.min(200, right - bx - 22) : Math.min(200, right - left)
            return (
              <g key={batch.id}>
                {batch.samples.length > 1 ? <path d={`M${bx + 9},${rowY(first)} H${bx + 13} V${rowY(rowIndex - 1)} H${bx + 9}`} fill="none" stroke="#9aaac0" /> : null}
                <foreignObject x={inline ? bx + 22 : right - labelWidth} y={rowY(first) - (inline ? 10 : 27)} width={labelWidth} height={22}>
                  <div className={cn("flex items-center gap-1 font-mono text-xs leading-5 text-slate-100", !inline && "justify-end")} title={`Arrival batch ${batch.label || "untitled"} · ${formatTime(batch.arrivalOffsetMs, unit)} ${unit}`}>
                    <span className="min-w-0 truncate">{batch.label || "Untitled"}</span>
                    <span className="shrink-0 tabular-nums">· {formatTime(batch.arrivalOffsetMs, unit)} {unit}</span>
                  </div>
                </foreignObject>
              </g>
            )
          })}
        </svg>
      </div>
      <svg width={width} height={49} role="img" aria-label={`Planned offset axis in ${unit}`}>
        <line x1={left} x2={right} y1={1} y2={1} stroke="#253246" />
        {visibleTicks.map((tick, index) => <text key={tick} x={x(tick)} y={19} textAnchor={index === 0 ? "start" : index === visibleTicks.length - 1 ? "end" : "middle"} className="fill-slate-400 font-mono text-xs tabular-nums">{formatTime(tick, unit)}</text>)}
        <text x={(left + right) / 2} y={42} textAnchor="middle" className="fill-slate-400 font-mono text-xs">Planned offset ({unit})</text>
      </svg>
      <div className="flex items-center gap-2 text-xs text-slate-400"><span className="size-2 rotate-45 bg-accent" aria-hidden="true" />Planned dispatch</div>
      <div className="mt-3 flex flex-wrap justify-between gap-x-4 gap-y-1 border-t border-slate-800 pt-3 font-mono text-xs text-slate-400">
        <span>Clock origin <span className="text-slate-100">t = 0</span></span>
        <span className="text-pretty">Fresh cache required before dispatch</span>
      </div>
    </div>
  )
}
