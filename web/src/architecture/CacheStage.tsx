import { useThree } from "@react-three/fiber"
import { X } from "lucide-react"
import { useLayoutEffect, useMemo, useRef, type ReactNode } from "react"
import * as THREE from "three"

import type { KvBlockProjection, KvCacheProjection, KvPoolProjection } from "../../../replay/src/learning.ts"
import type { Selection } from "../execution/types.ts"
import { requestLabel } from "../explore/model.ts"
import { cn } from "../lib/cn.ts"
import { requestColor } from "../theme.ts"
import { EvidenceTag } from "../ui/controls.tsx"
import { dieCellPosition, dieLayout, mappedBlockSlot } from "./model.ts"
import { createDieRaycast } from "./poolPicking.ts"
import { BoxEdges, FloorGrid, SceneEmpty, SceneLabel, SceneText, SCENE_COLORS, sceneClick, useHoverCursor, type Position3 } from "./primitives.tsx"

const DIE_Y = -1.0
const RIM_LIMIT = 400

export function CacheStage({ cache, requestOrder, requestLabels, selection, trackedRequestId, trackedTokenPosition, onSelect }: {
  cache: KvCacheProjection; requestOrder: string[]; requestLabels?: ReadonlyMap<string, string>; selection: Selection | null;
  trackedRequestId?: string | null; trackedTokenPosition?: number | null; onSelect: (selection: Selection) => void
}) {
  const lastBlock = useRef<string | null>(null)
  useLayoutEffect(() => { if (selection?.kind === "block") lastBlock.current = selection.key }, [selection])
  const blocks = useMemo(() => cache.pools.flatMap((pool) => pool.blocks), [cache.pools])
  const tokenBlocks = useMemo(() => selection?.kind === "token" ? blocks.filter((block) => [...block.contents, ...block.bindings].some((span) =>
    span.requestId === selection.requestId && selection.position >= span.tokenStart && selection.position < span.tokenEnd,
  )) : [], [blocks, selection])
  if (!cache.pools.length) return <group><FloorGrid y={DIE_Y - 0.25} /><SceneEmpty label="Full KV pool inventory not observed" detail={cache.reason ?? "Capture a new run with pool inventory enabled."} /></group>
  const selected = selection?.kind === "block" ? blocks.find((block) => blockMatches(block, selection.key))
    : tokenBlocks.find((block) => lastBlock.current != null && blockMatches(block, lastBlock.current)) ?? tokenBlocks[0]
  const span = cache.pools.length > 1 ? 9 : 11
  const offset = (index: number) => (index - (cache.pools.length - 1) / 2) * (span + 2.5)
  return <group>
    <FloorGrid y={DIE_Y - 0.25} />
    {cache.pools.map((pool, index) => (
      <PoolDie key={pool.pool} pool={pool} span={span} x={offset(index)} requestOrder={requestOrder} requestLabels={requestLabels} selection={selection}
        selected={selected?.pool === pool.pool ? selected : null}
        trackedRequestId={trackedRequestId} trackedTokenPosition={trackedTokenPosition} onSelect={onSelect} />
    ))}
    {cache.status === "unknown" ? <SceneText text="Trace gap: inventory is frozen; current slot state is unknown" position={[0, DIE_Y, 7.4]} /> : null}
  </group>
}

function PoolDie({ pool, span, x, requestOrder, requestLabels, selection, selected, trackedRequestId, trackedTokenPosition, onSelect }: {
  pool: KvPoolProjection; span: number; x: number; requestOrder: string[]; requestLabels?: ReadonlyMap<string, string>; selection: Selection | null; selected: KvBlockProjection | null;
  trackedRequestId?: string | null; trackedTokenPosition?: number | null; onSelect: (selection: Selection) => void
}) {
  const layout = useMemo(() => dieLayout(pool.capacity, span), [pool.capacity, span])
  const raycast = useMemo(() => createDieRaycast(layout, 0.04), [layout])
  const mesh = useRef<THREE.InstancedMesh>(null)
  const hover = useHoverCursor()
  const byId = useMemo(() => new Map(pool.blocks.map((block) => [block.blockId, block])), [pool.blocks])
  const selectedRequest = selection?.kind === "request" ? selection.id : selection?.kind === "token" ? selection.requestId : null
  const heightFor = (block: KvBlockProjection | undefined) => {
    if (!block) return layout.pitch * 0.05
    if (selected && block.key === selected.key) return Math.max(0.6, Math.min(1.15, layout.pitch * 6))
    if (!block.isNull && block.refCount > 0) return Math.min(0.24, layout.pitch * 0.32)
    if (block.hasHash) return Math.min(0.12, layout.pitch * 0.18)
    return Math.min(0.035, layout.pitch * 0.06)
  }
  useLayoutEffect(() => {
    const instance = mesh.current
    if (!instance) return
    const transform = new THREE.Object3D()
    const tint = new THREE.Color()
    for (let index = 0; index < layout.capacity; index += 1) {
      const block = byId.get(index)
      const [cx, cz] = dieCellPosition(index, layout)
      const height = heightFor(block)
      transform.position.set(cx, height / 2, cz)
      transform.scale.set(layout.pitch * 0.78, height, layout.pitch * 0.78)
      transform.updateMatrix()
      instance.setMatrixAt(index, transform.matrix)
      instance.setColorAt(index, tint.set(cellColor(block, requestOrder, selectedRequest, selected?.key ?? null)))
    }
    instance.instanceMatrix.needsUpdate = true
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true
    instance.computeBoundingSphere()
  }, [layout, byId, requestOrder, selectedRequest, selected?.key])
  const viewportWidth = useThree((state) => state.size.width)
  const special = useMemo(() => pool.blocks.filter((block) => block.isNull || block.refCount > 0 || block.hasHash || block.key === selected?.key).slice(0, RIM_LIMIT), [pool.blocks, selected?.key])
  const rulerColumns = rulerStep(layout.columns)
  const rulerRows = rulerStep(layout.rows)
  const front = layout.depth / 2
  const left = -layout.width / 2
  const selectedPosition = selected ? dieCellPosition(selected.blockId, layout) : null
  // A small die leaves room beside it, so the cartridge sits off the die instead of covering its other blocks.
  const besideDie = layout.width < span * 0.75 && viewportWidth >= 900
  const cartridgeSide = selectedPosition && selectedPosition[0] < 0 ? "left" : "right"
  const cartridgeX = selectedPosition ? besideDie ? (cartridgeSide === "right" ? 1 : -1) * (layout.width / 2 + 0.45) : selectedPosition[0] : 0
  const cartridgeAnchor = besideDie ? cartridgeSide : selectedPosition && selectedPosition[0] > layout.width * 0.15 ? "left" : "right"
  // Keep clear of the canvas overlays: camera controls (a right column, or a top strip on phones) and the legend and evidence watermark below.
  const cartridgeInsets = viewportWidth < 640 ? { top: 84, bottom: 104 } : { top: 72, right: 84, bottom: viewportWidth < 1000 ? 80 : 52 }
  return <group position={[x, DIE_Y, 0]}>
    <mesh position={[0, -0.09, 0]} raycast={() => null}>
      <boxGeometry args={[layout.width + 0.5, 0.16, layout.depth + 0.5]} />
      <meshStandardMaterial color="#0b131f" roughness={0.85} metalness={0.1} />
    </mesh>
    <BoxEdges size={[layout.width + 0.5, 0.16, layout.depth + 0.5]} position={[0, -0.09, 0]} color={SCENE_COLORS.edge} />
    <instancedMesh ref={mesh} args={[undefined, undefined, Math.max(1, layout.capacity)]} raycast={raycast} {...hover.bind}
      onClick={(event) => {
        if (!sceneClick(event) || event.instanceId == null) return
        event.stopPropagation()
        const block = byId.get(event.instanceId)
        if (block) onSelect({ kind: "block", key: block.key })
      }}>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
    {special.map((block) => {
      const [cx, cz] = dieCellPosition(block.blockId, layout)
      const height = heightFor(block)
      const isSelected = block.key === selected?.key
      const origin = block.contents.at(-1)?.requestId ?? block.bindings.at(-1)?.requestId ?? null
      const rim = isSelected ? SCENE_COLORS.accentSoft : block.isNull ? "#8494ad" : block.refCount === 0 ? SCENE_COLORS.cached : origin ? requestColor(requestOrder, origin) : SCENE_COLORS.edge
      return <BoxEdges key={block.key} size={[layout.pitch * 0.8, height, layout.pitch * 0.8]} position={[cx, height / 2, cz]} color={rim} />
    })}
    {Array.from({ length: Math.ceil(layout.columns / rulerColumns) }, (_, step) => {
      const column = step * rulerColumns
      const [cx] = dieCellPosition(column, layout)
      return <SceneText key={`c-${column}`} text={`+${column}`} position={[cx, -0.15, front + 0.55]} muted />
    })}
    {Array.from({ length: Math.ceil(layout.rows / rulerRows) }, (_, step) => {
      const row = step * rulerRows
      const [, cz] = dieCellPosition(row * layout.columns, layout)
      return <SceneText key={`r-${row}`} text={String(row * layout.columns).padStart(4, "0")} position={[left - 0.75, -0.15, cz]} muted />
    })}
    <SceneText text={`${pool.pool} · ${pool.capacity.toLocaleString()} physical blocks · ${pool.tokenOccupancyStatus !== "observed" ? "token-slot layout unavailable" : pool.blockSize == null ? "group-specific slots" : `${pool.blockSize} slots each`}`} position={[0, -0.15, -front - 0.75]} muted />
    {selected && selectedPosition ? (
      <SceneLabel position={[cartridgeX, heightFor(selected) + 0.05, selectedPosition[1]] as Position3} anchor={cartridgeAnchor} interactive clamp={cartridgeInsets}>
        <BlockCartridge block={selected} pool={pool} requestOrder={requestOrder} requestLabels={requestLabels} selection={selection} trackedRequestId={trackedRequestId} trackedTokenPosition={trackedTokenPosition} narrow={besideDie} short={viewportWidth < 640} onSelect={onSelect} />
      </SceneLabel>
    ) : null}
  </group>
}

function BlockCartridge({ block, pool, requestOrder, requestLabels, selection, trackedRequestId, trackedTokenPosition, narrow = false, short = false, onSelect }: {
  block: KvBlockProjection; pool: KvPoolProjection; requestOrder: string[]; requestLabels?: ReadonlyMap<string, string>; selection: Selection | null;
  trackedRequestId?: string | null; trackedTokenPosition?: number | null; narrow?: boolean; short?: boolean; onSelect: (selection: Selection) => void
}) {
  const nameOf = (requestId: string) => requestLabel(requestLabels, requestId)
  const groups = new Set([...block.contents, ...block.bindings].map((span) => span.cacheGroup))
  const sizes = [...new Set(pool.groups.filter((group) => group.pool === pool.pool && group.blockSize != null).map((group) => group.blockSize))]
  const blockSize = pool.blockSize ?? (sizes.length === 1 ? sizes[0] : null)
  const activeRequestId = selection?.kind === "token" ? selection.requestId : selection?.kind === "request" ? selection.id : trackedRequestId
  const activePosition = selection?.kind === "token" ? selection.position : trackedTokenPosition
  const header = (
    <div className="mb-2 flex items-center gap-2">
      <span className="grid h-[18px] shrink-0 place-items-center whitespace-nowrap border border-accent-400/45 px-1.5 font-mono text-[9.5px] font-semibold text-accent">BLK {block.blockId}</span>
      <span className="shrink-0 whitespace-nowrap font-mono text-[11px] font-semibold tracking-[0.04em] text-ink">Token slots</span>
      <span className="min-w-0 truncate font-mono text-[10px] text-faint">{block.pool} · gen {block.generation} · {block.trust === "unknown" ? "refs unknown" : `${block.refCount} ref`}{blockSize && block.validTokenCount != null ? ` · ${block.validTokenCount}/${blockSize} computed` : ""}</span>
      <span className="flex-1" />
      <EvidenceTag kind={block.trust === "unknown" ? "unknown" : "observed"}>{block.trust === "unknown" ? "Unknown" : "Observed"}</EvidenceTag>
      <button
        type="button"
        className="-mr-1 grid size-6 min-h-0 shrink-0 place-items-center rounded-none border-0 bg-transparent p-0 text-faint hover:text-ink"
        aria-label={`Close token slots for block ${block.blockId}`}
        title="Close token slots"
        onClick={() => onSelect({ kind: "component", id: "kv-cache" })}
      >
        <X className="size-3.5" aria-hidden="true" />
      </button>
    </div>
  )
  const shell = (body: ReactNode) => (
    <div className={cn("cp-hud w-[440px] bg-[rgba(7,11,19,0.96)] p-3 shadow-[0_18px_50px_-24px_rgba(0,0,0,0.9)]", narrow ? "max-w-[max(300px,34cqw)]" : "max-w-[min(440px,70vw)]")} data-active="true">
      {header}{body}
    </div>
  )
  if (block.isNull) return shell(<p className="m-0 text-xs text-muted">Reserved null placeholder. This cell cannot hold request KV contents.</p>)
  if (block.tokenOccupancyStatus !== "observed") return shell(<p className="m-0 text-pretty text-xs text-muted">{block.tokenOccupancyReason ?? `Token contents ${block.tokenOccupancyStatus === "unknown" ? "unknown" : "unavailable"} for this cache layout.`}</p>)
  if (groups.size > 1 || pool.groups.filter((group) => group.pool === pool.pool).length > 1) return shell(<p className="m-0 text-xs text-muted">Multiple cache groups have separate slot contents; see their captured bindings in the inspector.</p>)
  if (!blockSize) return shell(<p className="m-0 text-xs text-muted">This pool has no single captured token-slot geometry.</p>)
  const columns = Math.max(1, Math.min(8, Math.ceil(Math.sqrt(blockSize))))
  const holder = block.bindings.at(-1)?.requestId ?? null
  const origin = block.contents.at(-1)?.requestId ?? null
  return shell(<>
    <ol className="m-0 grid list-none gap-[5px] p-0" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }} aria-label={`Token slots in block ${block.blockId}`}>
      {Array.from({ length: blockSize }, (_, slot) => {
        const mapped = mappedBlockSlot(block, blockSize, slot, activeRequestId)
        const known = block.trust === "known"
        const isSelected = known && activeRequestId != null && mapped.requestId === activeRequestId && activePosition != null && mapped.position === activePosition
        const holderColor = mapped.requestId ? requestColor(requestOrder, mapped.requestId) : "#2a3c58"
        const originColor = mapped.sourceRequestId ? requestColor(requestOrder, mapped.sourceRequestId) : holderColor
        const status = !known ? "unknown" : mapped.span ? "computed" : mapped.binding ? "reserved" : "unwritten"
        const piece = mapped.tokenPiece != null ? mapped.tokenPiece.replace(/\n/g, "↵") : null
        return <li key={slot}>
          <button
            type="button"
            className={cn("relative block w-full min-h-0 overflow-hidden rounded-none pl-2 pr-1.5 text-left disabled:cursor-default disabled:opacity-100", short ? "h-11 py-0.5" : "h-[52px] py-1")}
            style={{
              borderColor: status === "computed" || status === "reserved" ? `color-mix(in srgb, ${holderColor} 35%, transparent)` : "#1c2a41",
              background: status === "computed" ? `color-mix(in srgb, ${holderColor} 9%, #080d16)` : status === "unknown" ? undefined : "#080d16",
              boxShadow: status === "computed" ? `inset 3px 0 0 ${originColor}` : undefined,
              outline: isSelected ? "1.5px solid var(--accent)" : undefined,
              outlineOffset: isSelected ? 1 : undefined,
            }}
            disabled={!known || mapped.position == null || mapped.requestId == null}
            aria-label={`Slot ${slot}: ${status}${mapped.requestId ? `, ${nameOf(mapped.requestId)} position ${mapped.position}` : ""}${mapped.sourceRequestId && mapped.sourceRequestId !== mapped.requestId ? `; content source ${nameOf(mapped.sourceRequestId)}` : ""}`}
            onClick={() => { if (mapped.requestId && mapped.position != null) onSelect({ kind: "token", requestId: mapped.requestId, position: mapped.position }) }}
          >
            {status === "unknown" ? <span className="cp-hatch absolute inset-0 opacity-40" aria-hidden="true" /> : null}
            <span className="relative flex justify-between font-mono text-[9px] text-faint"><span>{mapped.position == null ? "—" : `P${mapped.position}`}</span><span>S{slot}</span></span>
            <span className="relative mt-0.5 block truncate whitespace-pre font-mono text-xs text-ink [font-variant-ligatures:none]">
              {piece == null ? <span className="text-faint">{status}</span> : <>{piece.startsWith(" ") ? <span className="text-faint">·</span> : null}{piece.startsWith(" ") ? piece.slice(1) : piece}</>}
            </span>
            <span className="relative block truncate font-mono text-[9px] text-faint">{mapped.tokenId ?? ""}</span>
          </button>
        </li>
      })}
    </ol>
    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[9.5px] text-faint">
      {holder ? <span className="flex items-center gap-1.5"><i className="size-2" style={{ background: requestColor(requestOrder, holder) }} />Holder {nameOf(holder)}</span> : null}
      {origin ? <span className="flex items-center gap-1.5"><i className="size-2" style={{ background: requestColor(requestOrder, origin) }} />Written by {nameOf(origin)}</span> : null}
      <span className="flex-1" />
      <span>Rim = origin · fill = holder · no tensor values</span>
    </div>
  </>)
}

function rulerStep(count: number): number {
  if (count <= 12) return 1
  const raw = count / 7
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  return [1, 2, 5, 10].map((factor) => factor * magnitude).find((value) => value >= raw) ?? magnitude * 10
}

function blockMatches(block: KvBlockProjection, key: string): boolean {
  return key === block.key || key === `${block.pool}:${block.blockId}` || key === `${block.pool}:${block.blockId}:${block.generation}`
}

function cellColor(block: KvBlockProjection | undefined, requestOrder: string[], selectedRequest: string | null, selectedKey: string | null): string {
  if (!block) return SCENE_COLORS.unknown
  if (block.key === selectedKey) return "#2f6f80"
  if (block.isNull) return SCENE_COLORS.reserved
  if (block.trust === "unknown") return SCENE_COLORS.unknown
  if (block.refCount > 0) {
    const holders = [...new Set(block.bindings.map((binding) => binding.requestId))]
    if (selectedRequest && holders.includes(selectedRequest)) return SCENE_COLORS.accentSoft
    const color = new THREE.Color(holders.length === 1 ? requestColor(requestOrder, holders[0]) : SCENE_COLORS.accent).multiplyScalar(0.5)
    return `#${color.getHexString()}`
  }
  return block.hasHash ? "#3a2e18" : SCENE_COLORS.free
}
