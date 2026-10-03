import { useLayoutEffect, useMemo, useRef } from "react"
import * as THREE from "three"

import type { KvBlockProjection, KvCacheProjection, KvPoolProjection } from "../../../replay/src/learning.ts"
import type { Selection } from "../execution/types.ts"
import { requestColor } from "../theme.ts"
import { mappedBlockSlot, poolCellPosition, poolGridLayout } from "./model.ts"
import { createPoolRaycast } from "./poolPicking.ts"
import { SceneEmpty, SCENE_COLORS, sceneClick, TextSprite } from "./primitives.tsx"

export function CacheStage({ cache, requestOrder, selection, trackedRequestId, trackedTokenPosition, onSelect }: {
  cache: KvCacheProjection; requestOrder: string[]; selection: Selection | null;
  trackedRequestId?: string | null; trackedTokenPosition?: number | null; onSelect: (selection: Selection) => void
}) {
  const lastBlock = useRef<string | null>(null)
  useLayoutEffect(() => { if (selection?.kind === "block") lastBlock.current = selection.key }, [selection])
  const blocks = useMemo(() => cache.pools.flatMap((pool) => pool.blocks), [cache.pools])
  const tokenBlocks = useMemo(() => selection?.kind === "token" ? blocks.filter((block) => [...block.contents, ...block.bindings].some((span) =>
    span.requestId === selection.requestId && selection.position >= span.tokenStart && selection.position < span.tokenEnd,
  )) : [], [blocks, selection])
  if (!cache.pools.length) return <SceneEmpty label="Full KV pool inventory unavailable" detail={cache.reason ?? "Capture a new run with pool inventory enabled."} />
  const selected = selection?.kind === "block" ? blocks.find((block) => blockMatches(block, selection.key))
    : tokenBlocks.find((block) => lastBlock.current != null && blockMatches(block, lastBlock.current)) ?? tokenBlocks[0]
  const selectedPool = selected ? cache.pools.find((pool) => pool.pool === selected.pool) : undefined
  return <group>
    <TextSprite text="Physical KV pool" position={[0, 4.8, 0]} scale={[6.5, 0.48, 1]} />
    <TextSprite text="One cell per physical block; drill into a cell for token slots" position={[0, 4.25, 0]} scale={[9, 0.34, 1]} muted />
    <group position={selected ? [-3.8, 0.2, 0] : [0, 0.2, 0]} scale={selected ? 0.66 : 1}>
      {cache.pools.map((pool, index) => <group key={pool.pool} position={[(index - (cache.pools.length - 1) / 2) * 8.5, 0, 0]}>
        <PoolInstances pool={pool} requestOrder={requestOrder} selection={selection} highlightedBlockKey={selected?.key ?? null} onSelect={onSelect} />
        <TextSprite text={`${pool.pool}: ${pool.capacity.toLocaleString()} blocks / ${pool.tokenOccupancyStatus !== "observed" ? "token-slot layout unavailable" : pool.blockSize == null ? "group-specific" : `${pool.blockSize} slots per block`}`} position={[0, -4, 0]} scale={[7.4, 0.42, 1]} />
      </group>)}
    </group>
    {selected && selectedPool ? <BlockSlots block={selected} pool={selectedPool} requestOrder={requestOrder} selection={selection}
      trackedRequestId={trackedRequestId} trackedTokenPosition={trackedTokenPosition} onSelect={onSelect} /> : null}
    <CacheLegend unknown={cache.status === "unknown"} />
  </group>
}

function PoolInstances({ pool, requestOrder, selection, highlightedBlockKey, onSelect }: {
  pool: KvPoolProjection; requestOrder: string[]; selection: Selection | null; highlightedBlockKey: string | null; onSelect: (selection: Selection) => void
}) {
  const ref = useRef<THREE.InstancedMesh>(null)
  const layout = useMemo(() => poolGridLayout(pool.capacity), [pool.capacity])
  const raycast = useMemo(() => createPoolRaycast(layout), [layout])
  useLayoutEffect(() => {
    const mesh = ref.current
    if (!mesh) return
    const transform = new THREE.Object3D()
    for (let index = 0; index < layout.capacity; index += 1) {
      transform.position.set(...poolCellPosition(index, layout))
      transform.scale.setScalar(layout.spacing * 0.74)
      transform.updateMatrix()
      mesh.setMatrixAt(index, transform.matrix)
    }
    mesh.instanceMatrix.needsUpdate = true
    mesh.computeBoundingBox(); mesh.computeBoundingSphere()
  }, [layout])
  useLayoutEffect(() => {
    const mesh = ref.current
    if (!mesh) return
    const color = new THREE.Color()
    const byId = new Map(pool.blocks.map((block) => [block.blockId, block]))
    for (let index = 0; index < layout.capacity; index += 1) {
      const block = byId.get(index)
      const selected = block && (highlightedBlockKey === block.key || selection?.kind === "block" && blockMatches(block, selection.key))
      color.set(selected ? SCENE_COLORS.ink : block ? blockColor(block, requestOrder, selection) : SCENE_COLORS.unknown)
      mesh.setColorAt(index, color)
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [pool.blocks, layout.capacity, selection, highlightedBlockKey, requestOrder])
  return <instancedMesh ref={ref} args={[undefined, undefined, layout.capacity]} raycast={raycast}
    onClick={(event) => {
      if (!sceneClick(event) || event.instanceId == null) return
      event.stopPropagation()
      const block = pool.blocks.find((item) => item.blockId === event.instanceId)
      if (block) onSelect({ kind: "block", key: block.key })
    }}>
    <boxGeometry args={[1, 1, 1]} /><meshStandardMaterial roughness={0.9} />
  </instancedMesh>
}

function BlockSlots({ block, pool, requestOrder, selection, trackedRequestId, trackedTokenPosition, onSelect }: {
  block: KvBlockProjection; pool: KvPoolProjection; requestOrder: string[]; selection: Selection | null;
  trackedRequestId?: string | null; trackedTokenPosition?: number | null; onSelect: (selection: Selection) => void
}) {
  const groups = new Set([...block.contents, ...block.bindings].map((span) => span.cacheGroup))
  if (block.tokenOccupancyStatus !== "observed") {
    return <group position={[3.2, 0, 0]}><SceneEmpty label={`Block ${block.blockId}: token contents ${block.tokenOccupancyStatus === "unknown" ? "unknown" : "unavailable"}`} detail={block.tokenOccupancyReason ?? "This cache layout has no qualified token-slot mapping."} /></group>
  }
  if (groups.size > 1 || pool.groups.filter((group) => group.pool === pool.pool).length > 1) {
    return <group position={[3.2, 0, 0]}><SceneEmpty label="Cache-group slot view unavailable" detail="Multiple groups have separate slot contents; see their captured bindings in the inspector." /></group>
  }
  const sizes = [...new Set(pool.groups.filter((group) => group.pool === pool.pool && group.blockSize != null).map((group) => group.blockSize))]
  const blockSize = pool.blockSize ?? (sizes.length === 1 ? sizes[0] : null)
  if (!blockSize) return <group position={[3.2, 0, 0]}><SceneEmpty label="Select a cache group" detail="This pool has no single captured token-slot geometry." /></group>
  const columns = Math.max(1, Math.ceil(Math.sqrt(blockSize)))
  const spacing = Math.min(1.05, 5.8 / columns)
  const rows = Math.ceil(blockSize / columns)
  const activeRequestId = selection?.kind === "token" ? selection.requestId : selection?.kind === "request" ? selection.id : trackedRequestId
  const activePosition = selection?.kind === "token" ? selection.position : trackedTokenPosition
  return <group position={[3.25, 0.7, 0.3]}>
    <TextSprite text={`${block.pool} / block ${block.blockId} / generation ${block.generation}`} position={[0, 2.8, 0]} scale={[5.8, 0.4, 1]} />
    <TextSprite text={block.isNull ? "Reserved null placeholder" : `${block.refCount} references; ${block.validTokenCount}/${blockSize} observed computed slots`}
      position={[0, 2.3, 0]} scale={[5.8, 0.32, 1]} muted />
    {Array.from({ length: blockSize }, (_, slot) => {
      const { span, binding, position, requestId, tokenId: id, tokenPiece: piece } = mappedBlockSlot(block, blockSize, slot, activeRequestId)
      const selected = activeRequestId === requestId && activePosition === position
      const unknown = block.trust === "unknown"
      const status = block.isNull ? "reserved" : unknown ? "unknown" : span ? "computed" : binding ? "scheduled" : "unwritten"
      const color = selected ? "#435f7d" : block.isNull ? SCENE_COLORS.reserved : unknown ? SCENE_COLORS.unknown
        : span && requestId ? requestColor(requestOrder, requestId) : binding ? SCENE_COLORS.accent : SCENE_COLORS.surface
      const x = ((slot % columns) - (columns - 1) / 2) * spacing
      const y = (rows - 1) * spacing / 2 - Math.floor(slot / columns) * spacing
      return <group key={slot} position={[x, y, 0]}>
        <mesh onClick={requestId != null && position != null ? (event) => {
          if (!sceneClick(event)) return
          event.stopPropagation(); onSelect({ kind: "token", requestId, position })
        } : undefined}>
          <boxGeometry args={[spacing * 0.88, spacing * 0.7, 0.6]} /><meshStandardMaterial color={color} roughness={0.85} />
        </mesh>
        <TextSprite text={piece ? piece.replace(/\n/g, "↵").replace(/^\s+/, "␠") : id == null ? `Slot ${slot}` : `ID ${id}`} position={[0, 0.06, 0.4]} scale={[spacing * 0.83, spacing * 0.25, 1]} />
        <TextSprite text={position == null ? status : `p${position} ${status}`} position={[0, -0.2, 0.4]} scale={[spacing * 0.83, spacing * 0.2, 1]} muted />
      </group>
    })}
    <TextSprite text="These slots label K/V occupancy, not numeric tensor values" position={[0, -2.6, 0]} scale={[6.3, 0.3, 1]} muted />
    <TextSprite text={block.contents.some((span) => span.tokenIds != null) ? `Stored content: ${[...new Set(block.contents.map((span) => span.requestId))].join(", ")} / selection follows current bindings` : "No token text/IDs are available for this block at this playhead"}
      position={[0, -3.02, 0]} scale={[6.2, 0.3, 1]} muted />
  </group>
}

function CacheLegend({ unknown }: { unknown: boolean }) {
  const states = [
    ["Referenced", SCENE_COLORS.accent], ["Cached / available", SCENE_COLORS.cached],
    ["Available", SCENE_COLORS.free], ["Reserved", SCENE_COLORS.reserved], ["Unknown", SCENE_COLORS.unknown],
  ]
  return <group position={[0, -4.3, 1.5]}>
    {states.map(([label, color], index) => <group key={label} position={[(index - 2) * 2.25, 0, 0]}>
      <mesh position={[-0.84, 0, 0]}><boxGeometry args={[0.18, 0.18, 0.18]} /><meshBasicMaterial color={color} /></mesh>
      <TextSprite text={label} position={[0.16, 0, 0]} scale={[1.86, 0.3, 1]} muted />
    </group>)}
    {unknown ? <TextSprite text="Trace gap: inventory is frozen; current slot state is unknown" position={[0, -0.48, 0]} scale={[8.5, 0.3, 1]} muted /> : null}
  </group>
}

function blockMatches(block: KvBlockProjection, key: string): boolean {
  return key === block.key || key === `${block.pool}:${block.blockId}` || key === `${block.pool}:${block.blockId}:${block.generation}`
}

function blockColor(block: KvBlockProjection, requestOrder: string[], selection: Selection | null): string {
  if (block.isNull) return SCENE_COLORS.reserved
  if (block.trust === "unknown") return SCENE_COLORS.unknown
  if (block.refCount > 0) {
    const selectedRequest = selection?.kind === "request" ? selection.id : selection?.kind === "token" ? selection.requestId : null
    const holders = [...new Set(block.bindings.map((binding) => binding.requestId))]
    if (selectedRequest && holders.includes(selectedRequest)) return SCENE_COLORS.ink
    return block.refCount === 1 && holders.length === 1 ? requestColor(requestOrder, holders[0]) : SCENE_COLORS.accent
  }
  return block.hasHash ? SCENE_COLORS.cached : SCENE_COLORS.free
}
