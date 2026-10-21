import type { KvCacheProjection, RequestTokenProjection } from "../../../replay/src/learning.ts"

/** Current holders and original writers describe different relationships to a slot. */
export function tokenSlotOwnership(token: RequestTokenProjection, cache: KvCacheProjection | null) {
  let known = cache?.status === "observed" && token.cacheGroups.length > 0 && token.cacheGroups.every((group) => group.mappingStatus === "observed")
  const holders = new Set<string>()
  const writers = new Set<string>()
  for (const location of token.locations) {
    if (location.sourceRequestId) writers.add(location.sourceRequestId)
    const pool = cache?.pools.find((candidate) => candidate.pool === location.pool)
    const block = pool?.blocks.find((candidate) => candidate.blockId === location.blockId)
    const blockSize = pool?.groups.find((group) => group.cacheGroup === location.cacheGroup)?.blockSize ?? pool?.blockSize
    if (!block || !blockSize || block.trust !== "known" || block.generation !== location.generation) {
      known = false
      continue
    }
    if (block.isNull) continue
    if (block.refCount > 0 && !block.bindings.length) known = false
    for (const binding of block.bindings) {
      const firstSlot = binding.tokenStart % blockSize
      if (binding.cacheGroup === location.cacheGroup && location.slot >= firstSlot
        && location.slot < Math.min(blockSize, firstSlot + binding.tokenEnd - binding.tokenStart)) holders.add(binding.requestId)
    }
  }
  return { known, holders: [...holders], writers: [...writers] }
}

export function tokenKvLabel(token: Pick<RequestTokenProjection, "kvState" | "computedEver">, compact = false): string {
  if (token.kvState === "computed") return compact ? "Computed" : "Computed contents retained"
  if (token.kvState === "reserved") return compact ? "Reserved" : "Reserved; write unobserved"
  if (token.kvState === "unknown") return "Unknown"
  if (token.computedEver) return compact ? "No mapping" : "No retained KV mapping"
  return token.computedEver === false ? compact ? "Not computed" : "Not yet computed" : compact ? "No mapping" : "No observed KV mapping"
}
