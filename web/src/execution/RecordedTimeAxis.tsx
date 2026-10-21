import { useState } from "react"
import type { TimePoint } from "../../../replay/src/timeline.ts"
import { recordedTimeRange } from "../iterationPlayback.ts"
import { cn } from "../lib/cn.ts"

export function RecordedTimeAxis({ points, sequence, labels, onSeek }: {
  points: TimePoint[]
  sequence: number
  labels: ReadonlyMap<number, string>
  onSeek: (sequence: number) => void
}) {
  const [zoom, setZoom] = useState(1)
  const range = recordedTimeRange(points)
  if (!range) return null
  const timed = points.filter(point => point.elapsedMs != null && Number.isFinite(point.elapsedMs))
  const span = Math.max(0.001, range.endMs - range.startMs)
  const current = timed.filter(point => point.sequence <= sequence).at(-1)
  const currentTime = current?.elapsedMs ?? range.startMs
  const position = (time: number) => (time - range.startMs) / span * 100
  return <section className="min-w-0 flex-1" aria-label="Recorded host time axis">
    <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] tabular-nums text-muted">
      <span className="font-mono">{currentTime.toFixed(2)} ms · EVT {sequence}</span>
      <label className="flex items-center gap-2">Time zoom<select aria-label="Recorded time zoom" className="min-h-6 rounded-none bg-surface px-1 py-0 font-mono text-[11px]" value={zoom} onChange={event => setZoom(Number(event.target.value))}>{[1, 4, 16, 64, 256, 1024, 4096].map(value => <option key={value} value={value}>{value}×</option>)}</select></label>
    </div>
    <div className="mt-1 overflow-x-auto border border-line bg-surface" tabIndex={0} aria-label="Timestamped observations; scroll horizontally when zoomed">
      <div className="relative h-16" style={{width:`${zoom * 100}%`}}>
        <div className="absolute inset-y-0 left-2 right-2">
          <div className="absolute left-0 right-0 top-7 border-t border-line-strong" />
          {Array.from({length:6}, (_, index) => <div key={index} className="absolute top-8 font-mono text-[10px] tabular-nums text-muted" style={{left:`${index * 20}%`,transform:index === 0 ? undefined : index === 5 ? 'translateX(-100%)' : 'translateX(-50%)'}}>{(range.startMs + span * index / 5).toFixed(1)} ms</div>)}
          {timed.map(point => <button key={point.sequence} type="button" className={cn("absolute top-2 grid h-6 min-h-0 w-2 -translate-x-1/2 place-items-center rounded-none border-0 bg-transparent p-0", point.sequence === sequence ? "text-accent" : "text-muted")} style={{left:`${position(point.elapsedMs!)}%`}} title={`EVT ${point.sequence} · ${labels.get(point.sequence) ?? 'Observation'} · ${point.elapsedMs!.toFixed(6)} ms`} aria-label={`Observation ${point.sequence}: ${labels.get(point.sequence) ?? 'recorded event'}, ${point.elapsedMs!.toFixed(6)} ms`} aria-current={point.sequence === sequence ? "step" : undefined} onClick={() => onSeek(point.sequence)}><span className={cn("block w-px", point.sequence === sequence ? "h-6 bg-accent" : "h-2 bg-current")} /></button>)}
          <span className="pointer-events-none absolute top-0 h-8 border-l border-accent" style={{left:`${position(currentTime)}%`}} aria-hidden="true" />
        </div>
      </div>
    </div>
    <p className="mb-0 mt-1 text-pretty text-[10.5px] text-muted">Point marks are observations at their recorded host timestamps. Spacing represents elapsed time; marks have no execution duration. {points.length - timed.length > 0 ? `${points.length - timed.length} observations have no usable timestamp.` : "Zoom to separate close observations."}</p>
  </section>
}
