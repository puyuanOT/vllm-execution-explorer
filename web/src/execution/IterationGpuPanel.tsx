import { useEffect, useId, useMemo, useRef, useState, type MouseEvent } from "react"
import type { ProfileKvObservations } from "../../../replay/src/profiling.ts"
import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import { cn } from "../lib/cn.ts"
import { requestLabel } from "../explore/model.ts"
import { LayerTokenMatrix } from "./LayerTokenMatrix.tsx"
import { formatPositions, iterationProfile, nativeWindowAt, participantPickerValue, participantSelection, selectedLayerIndex } from "./iterationProfile.ts"
import type { Selection } from "./types.ts"

const WIDTH = 960, LEFT = 55, RIGHT = 18, TOP = 31, ROW = 19
const format = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 3 })
const time = (value: number) => Math.abs(value) >= 1000 ? `${format(value / 1000)} ms` : `${format(value)} µs`

export function IterationGpuPanel({ state, manifest, observations, stepId, timeUs, layerCount, labels, selection, requestId, tokenPosition, error, onTime, onSelect, onClearTracking, onClose }: {
  state: ReplayState; manifest: RecordingManifest | null; observations: ProfileKvObservations | null; stepId: string
  timeUs: number; layerCount: number | null; labels: ReadonlyMap<string, string>; selection: Selection | null
  requestId: string | null; tokenPosition: number | null; error?: string; onTime: (value: number) => void
  onSelect: (selection: Selection) => void; onClearTracking?: () => void; onClose: () => void
}) {
  const id = useId()
  const scope = useMemo(() => iterationProfile(observations, stepId), [observations, stepId])
  const [storedWindow, setWindow] = useState<{ startUs: number; endUs: number } | null>(null)
  const [nativeLayer, setNativeLayer] = useState(() => selectedLayerIndex(selection) ?? scope.writes[0]?.layerIndex ?? 0)
  const viewport = useRef<HTMLDivElement>(null)
  const svg = useRef<SVGSVGElement>(null)
  const explicitLayer = selectedLayerIndex(selection)
  const layer = explicitLayer ?? nativeLayer
  const layers = Array.from({ length: Math.max(layerCount ?? 0, ...scope.writes.map((write) => write.layerIndex + 1), 1) }, (_, index) => index)
  const extent = scope.extent
  const visible = storedWindow ?? extent
  const span = visible ? Math.max(0.001, visible.endUs - visible.startUs) : 1
  const fullSpan = extent ? Math.max(0.001, extent.endUs - extent.startUs) : 1
  const selectedIntervals = scope.intervals.filter((interval) => interval.write.layerIndex === layer)
  const writes = scope.writes.filter((write) => write.layerIndex === layer)
  const minimumWindow = Math.max(0.001, Math.min(...scope.intervals.map(({ activity }) => activity.durationUs).filter((duration) => duration > 0), 1) / 2)
  const onTimeRef = useRef(onTime)
  onTimeRef.current = onTime

  function moveCursor(cursor: number, width = span) {
    if (!extent) return
    const next = nativeWindowAt(extent, cursor, width)
    setWindow(next)
    onTime(next.cursor)
  }

  function chooseLayer(nextLayer: number) {
    setNativeLayer(nextLayer)
    onSelect({ kind: "layer", layerIndex: nextLayer })
    const first = scope.intervals.find(({ write }) => write.layerIndex === nextLayer)
    if (first) moveCursor(first.activity.startUs)
  }

  // Layer navigation in the existing architecture controls uses this local native clock.
  useEffect(() => {
    setNativeLayer(layer)
    const first = scope.intervals.find(({ write }) => write.layerIndex === layer)
    if (first) {
      if (extent) setWindow(current => {
        if (!current || first.activity.startUs >= current.startUs && first.activity.startUs <= current.endUs) return current
        return nativeWindowAt(extent, first.activity.startUs, current.endUs - current.startUs)
      })
      onTimeRef.current(first.activity.startUs)
    }
    if (viewport.current && svg.current) viewport.current.scrollTop = Math.max(0, (TOP + layer * ROW) * svg.current.getBoundingClientRect().width / WIDTH - viewport.current.clientHeight / 2)
  }, [layer, scope])

  function zoomLayer() {
    if (!selectedIntervals.length) return
    const start = selectedIntervals[0].activity.startUs
    const end = Math.max(...selectedIntervals.map(({ activity }) => activity.endUs))
    moveCursor((start + end) / 2, Math.max(minimumWindow, end - start) * 4)
  }

  function pickLayer(event: MouseEvent<SVGSVGElement>) {
    const bounds = event.currentTarget.getBoundingClientRect()
    const next = Math.floor(((event.clientY - bounds.top) / bounds.height * (TOP + layers.length * ROW + 15) - TOP) / ROW)
    if (next >= 0 && next < layers.length) chooseLayer(next)
  }

  function pickParticipant(nextLayer: number, id: string, position: number) {
    const write = scope.writes.find((write) => write.layerIndex === nextLayer && write.participants.some((row) => row.requestId === id && row.tokenPosition === position))
    const participant = write?.participants.find((row) => row.requestId === id && row.tokenPosition === position)
    if (write && participant) onSelect(participantSelection(state, write, participant).selection)
    else chooseLayer(nextLayer)
    const first = scope.intervals.find((interval) => interval.write.layerIndex === nextLayer)
    if (first) moveCursor(first.activity.startUs)
  }

  return <section className="shrink-0 border-t border-accent/30 bg-[#060a11]" aria-labelledby={`${id}-title`}>
    <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-3">
      <div><h2 id={`${id}-title`} className="m-0 text-sm font-semibold text-ink">Inside {stepId} · native CUDA cache writes</h2><p className="mb-0 mt-1 text-xs text-muted">Linked by recorded step and operation IDs. Host replay stays at event {state.lastSequence}.</p></div>
      <button type="button" className="min-h-7 text-xs" onClick={onClose}>Close iteration inspection</button>
    </div>
    {!scope.intervals.length || !extent || !visible ? <p className="m-0 px-3.5 pb-3 text-sm text-muted" role={error ? "status" : undefined}>{error || scope.writes.find((write) => write.reason)?.reason || "No correlated native CUDA cache-write intervals were recorded for this host iteration. Model compute and token timings are not inferred."}</p> : <>
      <div className="grid gap-2 px-3.5 pb-3">
        <p id={`${id}-clock`} className="m-0 text-xs leading-5 text-muted">Native activity extent: <b className="font-mono font-medium text-ink">{time(fullSpan)}</b> including gaps, not continuous execution time. This is the independent profiler clock; no alignment with controller timestamps is implied. Only measured cache-write activities are drawn.</p>
        <div className="flex flex-wrap items-end gap-2">
          <label className="grid gap-1 text-xs text-muted">Layer<select aria-label="Iteration inspection layer" value={layer} onChange={event => chooseLayer(Number(event.target.value))}>{layers.map(index => <option key={index} value={index}>Layer {index + 1}{scope.intervals.some(({ write }) => write.layerIndex === index) ? " · measured" : " · not captured"}</option>)}</select></label>
          <button type="button" className="min-h-8 text-xs" disabled={span <= minimumWindow} onClick={() => moveCursor(timeUs, Math.max(minimumWindow, span / 2))}>Zoom in</button>
          <button type="button" className="min-h-8 text-xs" disabled={span >= fullSpan} onClick={() => moveCursor(timeUs, span * 2)}>Zoom out</button>
          <button type="button" className="min-h-8 text-xs text-accent" disabled={!selectedIntervals.length} onClick={zoomLayer}>Zoom to layer {layer + 1}</button>
          <button type="button" className="min-h-8 text-xs" onClick={() => setWindow(extent)}>Full native extent</button>
          <span className="font-mono text-xs tabular-nums text-muted">{format(fullSpan / span)}×</span>
        </div>
        <div ref={viewport} className="max-h-64 overflow-auto border border-line">
          <svg ref={svg} viewBox={`0 0 ${WIDTH} ${TOP + layers.length * ROW + 15}`} className="block h-auto w-full min-w-[600px]" role="img" aria-labelledby={`${id}-plot-title ${id}-plot-desc`} onClick={pickLayer}>
            <title id={`${id}-plot-title`}>Recorded native cache-write intervals for {stepId}</title><desc id={`${id}-plot-desc`}>Rows are model layers. Position and width use native profiler microseconds. Tiny intervals are start markers until zoomed. Equivalent layer and cursor controls are outside the chart.</desc>
            <defs><clipPath id={`${id}-clip`}><rect x={LEFT} y={TOP} width={WIDTH - LEFT - RIGHT} height={layers.length * ROW} /></clipPath></defs>
            {layers.map(index => <g key={index}><rect x="0" y={TOP + index * ROW} width={WIDTH} height={ROW} fill={layer === index ? "#102e3b" : index % 2 ? "#09111d" : "#060a11"} /><text x={LEFT - 8} y={TOP + index * ROW + 13} textAnchor="end" fontFamily="monospace" fontSize="11" fill={index === layer ? "#40d9ee" : "#8294a9"}>L{index + 1}</text></g>)}
            {Array.from({ length: 6 }, (_, index) => <g key={index}><line x1={LEFT + (WIDTH - LEFT - RIGHT) * index / 5} x2={LEFT + (WIDTH - LEFT - RIGHT) * index / 5} y1={TOP} y2={TOP + layers.length * ROW} stroke="#1c2c3e" /><text x={LEFT + (WIDTH - LEFT - RIGHT) * index / 5} y="18" textAnchor={index === 0 ? "start" : index === 5 ? "end" : "middle"} fill="#92a4b8" fontFamily="monospace" fontSize="11">+{time(visible.startUs + span * index / 5 - extent.startUs)}</text></g>)}
            <g clipPath={`url(#${id}-clip)`}>{scope.intervals.map(({ write, activity }) => {
              const x = LEFT + (activity.startUs - visible.startUs) / span * (WIDTH - LEFT - RIGHT)
              const width = (activity.endUs - activity.startUs) / span * (WIDTH - LEFT - RIGHT)
              const y = TOP + write.layerIndex * ROW + 4
              const color = layer === write.layerIndex ? "#40d9ee" : "#629cdd"
              return width < 1 ? <line key={`${write.operationId}:${activity.id}`} x1={x} x2={x} y1={y} y2={y + 11} stroke={color} strokeWidth="1.5"><title>Start marker · {activity.durationUs} µs · {activity.name}</title></line> : <rect key={`${write.operationId}:${activity.id}`} x={x} y={y} width={width} height="11" fill={color}><title>{activity.durationUs} µs · {activity.name}</title></rect>
            })}<line x1={LEFT + (timeUs - visible.startUs) / span * (WIDTH - LEFT - RIGHT)} x2={LEFT + (timeUs - visible.startUs) / span * (WIDTH - LEFT - RIGHT)} y1={TOP} y2={TOP + layers.length * ROW} stroke="#e6eff9" strokeDasharray="3 3" /></g>
          </svg>
        </div>
        <p className="m-0 text-[11px] text-muted">Thin marks indicate interval starts when the measured width is smaller than a pixel. Blank space has no displayed cache-write interval.</p>
        <div className="grid gap-2 sm:grid-cols-2">
          <label className="grid gap-1 text-xs text-muted">Native-time cursor · +{format(timeUs - extent.startUs)} µs<input aria-label="Iteration native-time cursor" aria-describedby={`${id}-clock ${id}-scope`} aria-valuetext={`${format(timeUs - extent.startUs)} microseconds from this iteration's native extent start`} type="range" min="0" max={span} step={Math.max(0.0001, span / 10000)} value={Math.max(0, Math.min(span, timeUs - visible.startUs))} onChange={event => onTime(visible.startUs + Number(event.target.value))} /></label>
          <label className="grid gap-1 text-xs text-muted">Native window position · +{time(visible.startUs - extent.startUs)}<input aria-label="Iteration native window position" type="range" min="0" max={Math.max(0, fullSpan - span)} step={Math.max(0.0001, (fullSpan - span) / 10000)} disabled={span >= fullSpan} value={Math.max(0, Math.min(fullSpan - span, visible.startUs - extent.startUs))} onChange={event => { const startUs = extent.startUs + Number(event.target.value); setWindow({ startUs, endUs: startUs + span }); onTime(Math.max(startUs, Math.min(startUs + span, timeUs))) }} /></label>
        </div>
        <p id={`${id}-scope`} className="m-0 border border-line px-2 py-2 text-xs text-muted">Intervals belong to a shared layer cache-write operation. Participant membership does not establish individual request, token, head, or complete model-layer execution duration. Physical coordinates below describe that recorded write; host pool state remains at its own replay boundary.</p>
        <div className="border border-line p-2">
          <h3 className="mb-2 mt-0 text-xs font-semibold text-ink">Layer {layer + 1} · measured write scope</h3>
          {writes.length ? writes.map(write => <div key={write.operationId} className="mb-2 last:mb-0">
            <p className="mb-1 mt-0 text-[11px] text-muted">{write.device ?? "Device not captured"} · {write.participants.length} recorded participant positions · {write.observationBoundary}</p>
            {scope.intervals.filter(interval => interval.write.operationId === write.operationId).map(({ activity }) => <details key={activity.id} className="text-xs text-muted"><summary><button type="button" className="min-h-6 border-0 bg-transparent px-1 py-0 text-accent" onClick={event => { event.preventDefault(); moveCursor((activity.startUs + activity.endUs) / 2, Math.max(minimumWindow, activity.durationUs) * 6) }}>{format(activity.durationUs)} µs · zoom to activity</button> · stream {activity.stream ?? "not captured"}</summary><p className="m-0 break-all font-mono text-[11px]">{activity.name}</p></details>)}
            {write.reason ? <p className="m-0 text-xs text-muted">{write.reason}</p> : null}
            <label className="mt-2 grid gap-1 text-xs text-muted">Inspect a recorded participant token<select key={write.operationId} aria-label={`Inspect participant token for layer ${layer + 1}`} value={participantPickerValue(write, selection)} onChange={event => { const participant = write.participants[Number(event.target.value)]; if (event.target.value !== "" && participant) onSelect(participantSelection(state, write, participant).selection) }}><option value="">Choose request / position / physical slot</option>{write.participants.map((participant, index) => <option key={participant.row} value={index}>{requestLabel(labels, participant.requestId)} · P{participant.tokenPosition} · {participant.pool ?? "pool unknown"} / block {participant.blockId ?? "?"} / slot {participant.slot ?? "?"} / generation {participant.allocationGeneration ?? "?"}{participantSelection(state, write, participant).physical ? "" : " · historical mapping"}</option>)}</select></label>
            <div className="mt-2 flex flex-wrap gap-2">{[...new Set(write.participants.map(row => row.requestId))].map(id => <button key={id} type="button" className={cn("min-h-6 px-1 py-0 text-[11px]", requestId === id && "text-accent")} onClick={() => onSelect({ kind: "request", id })}>{requestLabel(labels, id)} · {formatPositions(write.participants.filter(row => row.requestId === id).map(row => row.tokenPosition))}</button>)}</div>
          </div>) : <p className="m-0 text-xs text-muted">No cache-write operation was captured for this layer and iteration.</p>}
        </div>
      </div>
      <LayerTokenMatrix state={state} manifest={manifest} profile={observations} timeUs={timeUs} stepId={stepId} requestId={requestId} tokenPosition={tokenPosition} labels={labels} onSelect={onSelect} onClearTracking={onClearTracking} onLayerTimeSelect={chooseLayer} onParticipantSelect={pickParticipant} />
    </>}
  </section>
}
