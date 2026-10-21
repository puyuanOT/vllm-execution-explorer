import { projectKvConfiguration } from "../../../replay/src/kv.ts"
import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import { EvidenceTag, Readouts } from "../ui/controls.tsx"

export function formatStorageBytes(value: number | null): string {
  if (value == null) return "Not captured"
  if (value < 1024) return `${value.toLocaleString()} B`
  const units = ["KiB", "MiB", "GiB", "TiB"]
  const unit = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)) - 1)
  return `${(value / 1024 ** (unit + 1)).toLocaleString(undefined, { maximumFractionDigits: 2 })} ${units[unit]}`
}

export function KvConfigurationDetails({ state, manifest }: { state: ReplayState; manifest: RecordingManifest | null }) {
  const configuration = projectKvConfiguration(state, manifest)
  const kind = configuration.evidenceKind === "fixture" ? "synthetic" : configuration.status === "observed" ? "observed" : "unknown"
  return <section className="grid gap-3" aria-label="KV cache configuration and occupancy">
    <div className="flex items-center justify-between gap-2"><h3 className="m-0 text-sm text-ink">Configuration</h3><EvidenceTag kind={kind}>{kind === "synthetic" ? "Synthetic" : kind === "observed" ? "Observed" : "Not captured"}</EvidenceTag></div>
    <dl className="facts text-sm">
      <div><dt>Unique physical KV storage</dt><dd>{formatStorageBytes(configuration.uniqueStorageBytes)}</dd></div>
      <div><dt>Actual KV datatype</dt><dd>{[...new Set(configuration.layers.map((layer) => layer.dtype))].join(", ") || "Not captured"}</dd></div>
      <div><dt>Actual KV quantization</dt><dd>{[...new Set(configuration.layers.flatMap((layer) => layer.quantization?.mode ? [layer.quantization.mode] : []))].join(", ") || "Not captured"}</dd></div>
      <div><dt>Requested cache datatype</dt><dd>{configuration.requestedDtype ?? "Not captured"}</dd></div>
      <div><dt>Prefix caching</dt><dd>{configuration.prefixCaching == null ? "Not captured" : configuration.prefixCaching ? "Enabled" : "Disabled"}</dd></div>
      <div><dt>Context length limit per request</dt><dd className="tabular-nums">{configuration.maxModelLength == null ? "Not captured" : `${configuration.maxModelLength.toLocaleString()} tokens`}</dd></div>
    </dl>
    {configuration.reason ? <p className="m-0 text-pretty text-xs text-muted">{configuration.reason}</p> : null}
    {configuration.pools.map((pool) => <section key={pool.pool} className="grid gap-2 border-t border-line pt-3" aria-label={`${pool.pool} capacity and current state`}>
      <h3 className="m-0 text-sm text-ink">{pool.devices.length ? pool.devices.join(", ") : "Device not captured"} / {pool.pool}</h3>
      <dl className="facts text-sm">
        <div><dt>Physical / usable blocks</dt><dd className="tabular-nums">{pool.capacity.toLocaleString()} / {pool.usableCapacity.toLocaleString()}</dd></div>
        <div><dt>Token slots per block</dt><dd>{pool.blockSize ?? "Group-specific; see below"}</dd></div>
        <div><dt>Usable token capacity</dt><dd className="tabular-nums">{pool.tokenCapacity == null ? "Group-specific" : `${pool.tokenCapacity.toLocaleString()} slots`}</dd></div>
        <div><dt>Physical storage bytes</dt><dd>{formatStorageBytes(pool.storageBytes)}</dd></div>
        <div><dt>Layers using this pool</dt><dd>{pool.layerIndices.length ? pool.layerIndices.map((layer) => layer + 1).join(", ") : "Not captured"}</dd></div>
        <div><dt>Cache groups</dt><dd>{pool.groups.length ? pool.groups.map((group) => `${group.cacheGroup} (${group.blockSize ?? "?"} slots/block)`).join(", ") : "Not captured"}</dd></div>
      </dl>
      <h4 className="m-0 text-xs font-medium text-muted">Current allocation state</h4>
      <Readouts items={[
        { label: "Referenced", value: pool.referencedBlocks?.toLocaleString() ?? "Unknown" },
        { label: "Reusable prefix", value: pool.cachedBlocks?.toLocaleString() ?? "Unknown" },
        { label: "Uncached available", value: pool.freeBlocks?.toLocaleString() ?? "Unknown" },
      ]} />
      <p className="m-0 text-pretty text-xs text-muted">{pool.cachedBlocks == null || pool.freeBlocks == null ? "Current availability is unknown after the trace gap." : `${(pool.cachedBlocks + pool.freeBlocks).toLocaleString()} blocks can be reassigned, including ${pool.cachedBlocks.toLocaleString()} reusable prefix blocks.`}</p>
    </section>)}
    {!configuration.pools.length ? <p className="m-0 text-xs text-muted">Pool inventory has not reached this playhead. Step forward to the engine-start observation.</p> : null}
    <details className="border-t border-line pt-2"><summary className="cursor-pointer text-xs text-muted">Explanation</summary>
      <div className="mt-2 grid gap-2 text-pretty text-xs text-muted">
        <p className="m-0">Pool capacity is shared by requests. The context length limit constrains one request and does not describe the total cache size.</p>
        <p className="m-0">Referenced blocks have request holders. Released prefix blocks can be reused or reassigned. Available blocks can retain recorded old contents until overwritten; availability does not mean erased tensor memory.</p>
        <p className="m-0">Layer/head views are slices of the same physical allocations. Storage bytes are counted once even when layer tensors share a backing allocation.</p>
        <p className="m-0">The requested dtype can be “auto.” Actual dtype, device, tensor layout and quantization details require runtime observations. Quantization scales are not inferred from dtype.</p>
        {configuration.observationBoundary ? <p className="m-0">Configuration observed at {configuration.observationBoundary}.</p> : null}
      </div>
    </details>
  </section>
}
