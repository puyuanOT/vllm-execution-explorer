import { useId, useMemo, useState } from "react"
import type { RecordingManifest, TraceEvent } from "../../../replay/src/reduce.ts"
import { projectMemory, storageBreakdown, type DeviceMemory, type MemoryProjection } from "../../../replay/src/memory.ts"
import { Modal } from "../shell/Modal.tsx"
import { summarizeDeviceMemory, type DeviceMemorySummary } from "./memorySummary.ts"

const CATEGORIES: Record<string, string> = { weights: "Model weights / registered buffers", attention_kv: "Attention KV allocated storage", recurrent_state: "Gated DeltaNet / recurrent state", lora: "LoRA resident tensors", runtime_buffers: "Named runtime tensor buffers", shared: "Shared backing allocations" }
const bytes = (value: number | null) => value == null ? "Not captured" : value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(2)} GiB` : value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(2)} MiB` : value >= 1024 ? `${(value / 1024).toFixed(2)} KiB` : `${value} B`
const age = (value: number | null) => value == null ? "Unknown" : value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(1)} ms`

export function GpuMemoryPanel({ events, manifest, sequence, live = false }: { events: TraceEvent[]; manifest: RecordingManifest | null; sequence: number; live?: boolean }) {
  const id = useId()
  const memory = useMemo(() => projectMemory(events, manifest, sequence), [events, manifest, sequence])
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [deviceId, setDeviceId] = useState("")
  const devices = memory.current?.devices ?? []
  const provenanceNote = manifest?.execution_kind === "fixture" ? "This fixture contains authored memory values, not GPU measurements." : manifest?.execution_kind !== "gpu" && memory.current ? "Recording provenance does not establish a GPU capture." : null
  const device = devices.find((row) => row.uuid === deviceId) ?? devices[0] ?? null
  const summary = device ? summarizeDeviceMemory(memory, device.uuid, sequence, live) : null
  const waitingForSample = live && !memory.samples.length && !manifest?.gpu_memory
  return <section className="shrink-0 border-t border-line bg-[#060a11]" aria-labelledby={`${id}-title`}>
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-3.5 py-2">
      <h2 id={`${id}-title`} className="m-0 text-xs font-medium text-muted">GPU memory</h2>
      {device ? <>
        <span className="font-mono text-xs tabular-nums text-ink">{bytes(device.used)} / {bytes(device.total)}</span>
        {memory.gap ? <span className="text-xs text-warn">Sample gap</span> : null}
      </> : <p className="m-0 flex-1 text-xs text-muted" role="status">{waitingForSample ? "Waiting for the first GPU memory sample…" : memory.reason}</p>}
      <button type="button" className="ml-auto min-h-8 text-xs" aria-haspopup="dialog" onClick={() => setDetailsOpen(true)}>Memory breakdown</button>
    </div>
    <Modal open={detailsOpen} onOpenChange={setDetailsOpen} title="Memory breakdown" description={`${live ? "Live" : "Recorded"} GPU measurements at the host playhead. Memory counters and storage attribution use separately captured observations.`}>
      <div className="grid gap-3">
        {memory.captureReason ? <p className="m-0 text-pretty text-xs text-warn" role="status">Memory capture incomplete: {memory.captureReason}</p> : null}
        {provenanceNote ? <p className="m-0 text-pretty text-xs text-warn" role="status">{provenanceNote}</p> : null}
        {!device ? <div className="grid gap-2">
          <p className="m-0 text-pretty text-sm text-muted" role="status">{waitingForSample ? "Waiting for the first GPU memory sample…" : memory.reason}</p>
          {memory.latestAttempt?.reason ? <p className="m-0 text-pretty text-xs text-warn">Latest sample: {memory.latestAttempt.reason}</p> : null}
          <p className="m-0 text-pretty text-xs text-muted">{memory.samples.some((sample) => sample.status === "observed" && sample.devices.length) ? "Seek to a successful recorded GPU memory sample to inspect its counters and storage attribution." : waitingForSample ? "Memory measurements will appear when the capture returns its first sample." : "Whole-device, engine-process and framework allocation counters require a new measured capture. Existing KV inventory describes cache allocations only."}</p>
        </div> : <>
          <div className="flex flex-wrap items-center gap-3"><label className="flex min-w-0 flex-1 flex-col gap-2 text-xs text-muted">GPU<select className="min-w-0 w-full" aria-label="Memory device" value={device.uuid} onChange={(event) => setDeviceId(event.target.value)}>{devices.map((row) => <option key={row.uuid} value={row.uuid}>{row.name} · {row.uuid}</option>)}</select></label></div>
          <dl className="m-0 grid grid-cols-[1fr_auto] gap-x-3 gap-y-2 text-xs">
            <dt className="text-muted">Whole device used / capacity</dt><dd className="m-0 font-mono tabular-nums text-ink">{bytes(device.used)} / {bytes(device.total)}</dd>
            <dt className="text-muted">{live ? "Sampled maximum so far" : "Run sampled maximum"}</dt><dd className="m-0 font-mono tabular-nums text-ink">{bytes(summary?.maximum ?? null)}</dd>
            <dt className="text-muted">Host playhead event</dt><dd className="m-0 font-mono tabular-nums text-ink">{sequence}</dd>
            <dt className="text-muted">Sample age</dt><dd className="m-0 font-mono tabular-nums text-ink">{age(memory.ageMs)}</dd>
            <dt className="text-muted">Phase</dt><dd className="m-0 text-ink">{memory.phase?.replaceAll("_", " ") ?? "Unknown"}</dd>
          </dl>
          {memory.gap ? <p className="m-0 text-pretty text-xs text-warn" role="status">Sample gap. Showing the last successful measurement; it may no longer describe usage at this host playhead.{memory.latestAttempt?.status !== "observed" && memory.latestAttempt?.reason ? ` Latest sample: ${memory.latestAttempt.reason}` : ""}</p> : null}
          <p className="m-0 break-words text-pretty text-xs leading-5 text-muted">Source: {memory.current?.source}; requested interval {memory.intervalMs} ms. Engine PID {memory.current?.pid ?? "unknown"}. Highest sampled usage can miss short spikes. Sample age uses the host playhead; delivery latency is not measured.</p>
          <Accounting memory={memory} device={device} />
          <Attribution memory={memory} device={device} />
          {summary ? <PhaseSamples summary={summary} live={live} /> : null}
          <HostStorages memory={memory} />
        </>}
      </div>
    </Modal>
  </section>
}

function Accounting({ memory, device }: { memory: MemoryProjection; device: DeviceMemory }) {
  const framework = memory.current?.framework?.uuid === device.uuid ? memory.current.framework : null
  const rows: [string, number | null][] = [["Whole device used", device.used], ["Whole device free", device.free], ["Engine process used", device.processUsed], ["Framework tensor allocation", framework?.allocated ?? null], ["Framework allocator reservation", framework?.reserved ?? null], ["Allocator peak tensor allocation", framework?.peakAllocated ?? null], ["Allocator peak reservation", framework?.peakReserved ?? null]]
  return <div className="border border-line p-3"><h3 className="m-0 text-balance text-sm font-medium text-ink">Accounting views</h3><dl className="mb-0 mt-2 grid grid-cols-[1fr_auto] gap-x-3 gap-y-2 text-xs">{rows.map(([label, value]) => <div className="contents" key={label}><dt className="text-muted">{label}</dt><dd className="m-0 font-mono tabular-nums text-ink">{bytes(value)}</dd></div>)}</dl><p className="mb-0 mt-3 text-[11px] leading-5 text-muted">These counters overlap and must not be added together. Framework peaks are counters since CUDA allocator initialization or the last external reset; device maxima are periodic samples. {framework ? `Worker rank ${framework.rank ?? "unknown"}, local ${framework.device}.` : memory.phase === "cleanup" ? "Worker cleanup ended framework sampling; current allocator counters are unavailable." : "A worker allocator sample matching this GPU UUID was not captured."}</p></div>
}

function Attribution({ memory, device }: { memory: MemoryProjection; device: DeviceMemory }) {
  const framework = memory.current?.framework?.uuid === device.uuid ? memory.current.framework : null
  const parts = framework ? storageBreakdown(memory.inventory, framework.device) : []
  const known = parts.reduce((sum, row) => sum + row.bytes, 0)
  const remainder = device.processUsed == null || !parts.length || device.processUsed < known ? null : device.processUsed - known
  return <div className="border border-line p-3"><h3 className="m-0 text-balance text-sm font-medium text-ink">Loaded storage attribution</h3><p className="mb-0 mt-1 text-[11px] text-muted">Inventory age {age(memory.inventoryAgeMs)} · unique backing allocations</p>{parts.length ? <dl className="mb-0 mt-2 grid grid-cols-[1fr_auto] gap-x-3 gap-y-2 text-xs">{parts.map((row) => <div className="contents" key={row.category}><dt className="text-muted">{CATEGORIES[row.category] ?? row.category} <span className="text-faint">({row.storageCount})</span></dt><dd className="m-0 font-mono tabular-nums text-ink">{bytes(row.bytes)}</dd></div>)}<dt className="border-t border-line pt-2 text-muted">Unattributed process remainder</dt><dd className="m-0 border-t border-line pt-2 font-mono tabular-nums text-ink">{bytes(remainder)}</dd></dl> : <p className="mb-0 mt-3 text-xs text-muted">{memory.phase === "cleanup" ? "Cleanup invalidated the loaded storage inventory. Earlier weights and caches are no longer attributed to the current process sample." : "A loaded storage inventory matching this worker was not captured."}</p>}<p className="mb-0 mt-3 text-[11px] leading-5 text-muted">{memory.inventory?.coverage ?? "Storage coverage unavailable."} Remainder uses separately timed process and inventory samples, so it is not a simultaneous allocation census. Attention KV bytes describe the allocated pool; free logical blocks remain in that allocation.</p></div>
}

function HostStorages({ memory }: { memory: MemoryProjection }) {
  const rows = storageBreakdown(memory.inventory, "cpu")
  return rows.length ? <div className="border border-line p-3"><h3 className="m-0 text-balance text-sm font-medium text-ink">Host tensor storage</h3><p className="mb-0 mt-2 text-xs text-muted">{rows.map((row) => `${CATEGORIES[row.category] ?? row.category}: ${bytes(row.bytes)}`).join(" · ")}. Captured CPU tensor backing storage; excluded from GPU totals. This is not total host RAM or process RSS.</p></div> : null
}

function PhaseSamples({ summary, live }: { summary: DeviceMemorySummary; live: boolean }) {
  return <div className="overflow-x-auto border border-line p-3"><h3 className="m-0 text-balance text-sm font-medium text-ink">{live ? "Observed phases so far" : "Recorded run phases"}</h3><table className="mt-2 w-full text-left text-xs"><thead><tr className="text-muted"><th scope="col" className="pb-2 font-medium">Phase</th><th scope="col" className="pb-2 font-medium">Successful samples</th><th scope="col" className="pb-2 font-medium">Highest sampled device usage</th></tr></thead><tbody>{summary.phases.map((row) => <tr key={row.phase} className="border-t border-line"><th scope="row" className="py-2 pr-3 font-medium text-muted">{row.phase.replaceAll("_", " ")}</th><td className="py-2 pr-3 font-mono tabular-nums text-ink">{row.samples}</td><td className="py-2 font-mono tabular-nums text-ink">{bytes(row.maximum)}</td></tr>)}</tbody></table></div>
}
