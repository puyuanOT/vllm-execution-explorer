import { useState } from "react"
import type { MeasuredProfileProjection, ProfileActivity } from "../../../replay/src/profiling.ts"
import type { Selection } from "./types.ts"

export function MeasuredProfileDetails({ profile, activities, error, active, onTime, onSelect }: {
  profile: MeasuredProfileProjection
  activities: ProfileActivity[]
  error: string
  active: boolean
  onTime: (value: number) => void
  onSelect: (selection: Selection) => void
}) {
  const [rowPage, setRowPage] = useState(0)
  const stepId = activities.find((activity) => activity.scope.stepId)?.scope.stepId ?? null
  const step = profile.steps.find((item) => item.stepId === stepId)
  const rows = step?.packedRows ?? []
  const page = Math.min(rowPage, Math.max(0, Math.ceil(rows.length / 64) - 1))
  const layers = profile.layers.filter((layer) => layer.stepId === stepId)
  return <details className="border-b border-line pb-3" open={active || undefined}>
    <summary className="cursor-pointer font-mono text-[10.5px] uppercase tracking-[0.1em] text-muted">Measured diagnostic profile</summary>
    {error || profile.status !== "observed" ? <p className="text-sm text-muted" role={error ? "status" : undefined}>{error || profile.reason}</p> : <div className="space-y-3">
      <p className="text-sm text-muted">Eager execution with profiler overhead. Native profiler time is independent of the host replay playhead. Measurements apply to packed batch activities; individual token and head durations remain unknown.</p>
      <p className="text-xs text-muted">Capture: {profile.captureStatus ?? "unknown"} · coverage: {profile.completeness ?? "unknown"} · qualification: {profile.qualification ?? "unverified"}</p>
      {profile.reason ? <p className="text-sm text-amber-300">{profile.reason}</p> : null}
      <label className="block text-xs text-muted">Profiled batch
        <select aria-label="Profiled batch" value={stepId ?? ""} onChange={(event) => {
          const activity = profile.activities.find((item) => item.scope.stepId === event.target.value)
          if (activity) { setRowPage(0); onTime(activity.startUs) }
        }}>
          <option value="">Outside mapped batch intervals</option>
          {profile.steps.map((item) => <option value={item.stepId} key={item.stepId}>{item.stepId}</option>)}
        </select>
      </label>
      <p className="text-xs text-muted">{activities.length} active recorded activities · {profile.unmappedActivityCount} activities have no batch/layer correlation.</p>
      <h3 className="text-sm font-medium text-ink">Layer measurements for {stepId ?? "no selected batch"}</h3>
      <ul className="max-h-56 list-none space-y-2 overflow-auto p-0">
        {layers.map((layer) => <li key={`${layer.layerIndex}:${layer.component}`}>
          <button className="w-full text-left text-sm" type="button" onClick={() => onSelect(layer.component ? { kind: "module", layerIndex: layer.layerIndex, module: layer.component } : { kind: "layer", layerIndex: layer.layerIndex })}>
            Layer {layer.layerIndex + 1}{layer.component ? ` · ${layer.component === "ffn" ? "feed-forward" : layer.component}` : ""}
          </button>
          <p className="m-0 text-xs tabular-nums text-muted">CPU scope span: {layer.cpuScopeSpanUs == null ? "unavailable" : `${layer.cpuScopeSpanUs.toFixed(3)} µs`}</p>
          {layer.devices.map((device) => <p className="m-0 text-xs tabular-nums text-muted" key={String(device.device)}>Device {device.device ?? "unknown"}: activity span {device.spanUs.toFixed(3)} µs · busy interval union {device.busyUs == null ? "unavailable" : `${device.busyUs.toFixed(3)} µs`} · {device.activityCount} activities on {device.streams.length} streams</p>)}
          {layer.devices.length === 0 ? <p className="m-0 text-xs text-muted">CUDA activity correlation unavailable.</p> : null}
        </li>)}
      </ul>
      <h3 className="text-sm font-medium text-ink">KV cache-write operations</h3>
      <p className="text-xs text-muted">These are shared batch operations for one layer. Participant membership and correlated GPU activity establish which positions are involved; they do not assign individual token durations.</p>
      {profile.cacheWrites.filter((write) => write.stepId === stepId).length ? <ul className="max-h-56 list-none space-y-2 overflow-auto p-0" aria-label="Measured layer cache writes">{profile.cacheWrites.filter((write) => write.stepId === stepId).map((write) => <li key={write.operationId} className="border border-line px-2 py-1.5">
        <button type="button" className="min-h-6 border-0 bg-transparent p-0 text-left text-xs text-ink" onClick={() => onSelect({ kind: "layer", layerIndex: write.layerIndex })}>Layer {write.layerIndex + 1} · {write.device ?? "device not captured"}</button>
        <p className="m-0 text-xs text-muted">{write.participants.length} participating positions · {write.gpuActivityIds.length} correlated GPU activities · {write.observationBoundary}</p>
        {write.reason ? <p className="m-0 text-xs text-muted">{write.reason}</p> : null}
        <div className="mt-1 flex flex-wrap gap-1">{write.participants.slice(0, 8).map((participant) => <button key={participant.row} type="button" className="min-h-6 px-1 py-0 text-[10px]" onClick={() => onSelect({ kind: "token", requestId: participant.requestId, position: participant.tokenPosition })}>{participant.requestId} · position {participant.tokenPosition}</button>)}</div>
        {write.participants.length > 8 ? <p className="m-0 text-[10px] text-muted">Use the layer matrix for the remaining {write.participants.length - 8} positions.</p> : null}
      </li>)}</ul> : <p className="text-xs text-muted">Layer-specific cache-write operations were not captured for this batch. General layer activity does not establish cache-write chronology.</p>}
      <h3 className="text-sm font-medium text-ink">Packed input token rows</h3>
      <p className="text-xs text-muted">{step?.mappingStatus === "observed" ? `${step.unpaddedTokens} actual rows; ${step.paddedTokens == null ? `padding unavailable (${step.paddingReason ?? "not observed"})` : `${step.paddedTokens} including padding`}. Observed at ${step.mappingObservationBoundary}. Row order can differ from scheduler order.` : step?.mappingReason ?? "Select a measured batch interval to inspect its input rows."}</p>
      {rows.length > 0 ? <>
        <div className="flex items-center gap-2 text-xs text-muted"><button type="button" disabled={page === 0} onClick={() => setRowPage(page - 1)}>Previous rows</button><span>{page * 64}–{Math.min(rows.length, (page + 1) * 64) - 1} of {rows.length}</span><button type="button" disabled={(page + 1) * 64 >= rows.length} onClick={() => setRowPage(page + 1)}>Next rows</button></div>
        <ol className="max-h-40 list-none space-y-1 overflow-auto p-0" aria-label="Measured packed input rows">
          {rows.slice(page * 64, (page + 1) * 64).map((row) => <li key={row.row}>{row.requestId != null && row.tokenPosition != null ? <button className="w-full text-left text-xs" type="button" onClick={() => onSelect({ kind: "token", requestId: row.requestId!, position: row.tokenPosition! })}>Row {row.row} · {row.requestId} · position {row.tokenPosition}</button> : <span className="text-xs text-muted">Row {row.row} · {row.isPadding ? "padding" : "mapping unavailable"}</span>}</li>)}
        </ol>
      </> : null}
    </div>}
  </details>
}
