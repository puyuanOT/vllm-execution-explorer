import type { KvBlockProjection } from "../../../replay/src/learning.ts"
import type { KvLayerConfiguration } from "../../../replay/src/kv.ts"
import type { Selection } from "../execution/types.ts"
import { requestLabel } from "../explore/model.ts"
import { Segmented } from "../ui/controls.tsx"
import { BoxEdges, FloorGrid, SceneLabel, SceneText, SCENE_COLORS, type Position3 } from "./primitives.tsx"

type TokenSelection = Extract<Selection, { kind: "token" }>
interface SliceProps {
  block: KvBlockProjection; layers: KvLayerConfiguration[]; selection: TokenSelection
  labels?: ReadonlyMap<string, string>; onSelect: (selection: Selection) => void
}

function selectedSlice({ block, layers, selection }: SliceProps) {
  const layer = layers.find((layer) => layer.layerIndex === selection.kvLayerIndex) ?? layers[0]
  const head = Math.min(selection.kvHead ?? 0, Math.max(0, (layer?.numKvHeads ?? 1) - 1))
  const binding = block.bindings.find((binding) => binding.requestId === selection.requestId && selection.position >= binding.tokenStart && selection.position < binding.tokenEnd)
  const slot = binding && layer?.managerBlockSize ? selection.position % layer.managerBlockSize : null
  return { layer, head, slot, kind: selection.kvKind ?? "K" }
}

/** Controls live below the canvas; scene and inspector share the token slice selection. */
export function CacheTensorControls(props: SliceProps & { onInspect?: () => void }) {
  const { block, layers, selection, onSelect } = props
  const { layer, head, slot, kind } = selectedSlice(props)
  function update(patch: Partial<TokenSelection>) {
    onSelect({ ...selection, kvLayerIndex: layer?.layerIndex ?? 0, kvHead: head, kvKind: kind, kvBlockKey: block.key, ...patch })
  }
  return <section className="grid gap-2 border-t border-line bg-[#060a11] px-3.5 py-2.5" aria-label="3D KV tensor slice controls">
    <nav className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted" aria-label="3D KV slice breadcrumb">
      <button type="button" className="min-h-6 border-0 bg-transparent p-0 text-accent" onClick={() => onSelect({ kind: "token", requestId: selection.requestId, position: selection.position })}>{layer?.device ?? "Device unknown"} / {block.pool}{layer?.cacheGroup != null ? ` / group ${layer.cacheGroup}` : ""} / Block {block.blockId}</button>
      <span aria-hidden="true">/</span><span>Slot {slot ?? "?"}</span><span aria-hidden="true">/</span><span>Token position {selection.position}</span><span aria-hidden="true">/</span><span aria-current="page">Layer {(layer?.layerIndex ?? 0) + 1} / KV head {head} / {kind}</span>
    </nav>
    <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
      <label className="grid min-w-32 gap-1 text-[11px] text-muted">Layer<select className="h-8 py-0 text-xs" aria-label="3D cache layer" value={layer?.layerIndex ?? 0} onChange={(event) => update({ kvLayerIndex: Number(event.target.value) })}>{layers.map((layer) => <option key={layer.layerIndex} value={layer.layerIndex}>Layer {layer.layerIndex + 1}</option>)}</select></label>
      <label className="grid min-w-32 gap-1 text-[11px] text-muted">KV head index<select className="h-8 py-0 text-xs" aria-label="3D cache head" value={head} disabled={!layer?.numKvHeads} onChange={(event) => update({ kvHead: Number(event.target.value) })}>{Array.from({ length: layer?.numKvHeads ?? 1 }, (_, index) => <option key={index} value={index}>{index}</option>)}</select></label>
      <Segmented<"K" | "V"> label="3D cache keys or values" value={kind} onChange={(value) => update({ kvKind: value })} options={[{ value: "K", label: "Keys (K)" }, { value: "V", label: "Values (V)" }]} />
      <button type="button" className="min-h-8 text-xs" onClick={() => { update({}); props.onInspect?.() }}>Inspect token and captured values</button>
    </div>
    <p className="m-0 text-pretty text-[11px] text-muted">One token’s structural K/V slice. Layer, head and K/V choices are shared with the inspector; numerical data appears only for captured snapshots.</p>
  </section>
}

/** One physical allocation, one selected layer/head. Geometry never multiplies capacity. */
export function CacheTensorStage(props: SliceProps) {
  const { block, selection, labels, onSelect } = props
  const { layer, head, slot, kind } = selectedSlice(props)
  const headDimension = layer?.headDim ?? null
  return <group>
    <FloorGrid y={-2.8} />
    <SceneText text={`${requestLabel(labels, selection.requestId)} · token position ${selection.position} · block ${block.blockId} / slot ${slot ?? "unmapped"}`} position={[0, 1.9, 0]} strong />
    <SceneText text={`Layer ${(layer?.layerIndex ?? 0) + 1} · KV head ${head} · ${layer?.device ?? "device not captured"} · ${layer?.dtype ?? "datatype not captured"}`} position={[0, 1.5, 0]} muted />
    {(["K", "V"] as const).map((tensorKind, index) => {
      const x = (index - 0.5) * 5.1
      const selected = kind === tensorKind
      const dimensions = headDimension == null ? 1 : Math.min(16, headDimension)
      const width = 3.8
      const segmentWidth = width / dimensions
      return <group key={tensorKind} position={[x, 0, 0]}>
        <mesh raycast={() => null}><boxGeometry args={[width + 0.3, 0.2, 1.9]} /><meshStandardMaterial color={selected ? SCENE_COLORS.surfaceSelected : SCENE_COLORS.surface} /></mesh>
        <BoxEdges size={[width + 0.3, 0.2, 1.9]} color={selected ? SCENE_COLORS.accent : SCENE_COLORS.edge} />
        {Array.from({ length: dimensions }, (_, dimension) => <mesh key={dimension} position={[(dimension - (dimensions - 1) / 2) * segmentWidth, 0.28, 0]} raycast={() => null}>
          <boxGeometry args={[segmentWidth * 0.88, 0.3, 1.5]} /><meshBasicMaterial color={selected ? "#408ccc" : "#34445f"} toneMapped={false} />
        </mesh>)}
        <SceneLabel position={[0, 0.8, 0] as Position3} interactive><button type="button" className="cp-hud min-h-7 px-2 py-0 font-mono text-xs text-ink" aria-pressed={selected} onClick={() => onSelect({ ...selection, kvLayerIndex: layer?.layerIndex ?? 0, kvHead: head, kvKind: tensorKind, kvBlockKey: block.key })}>{tensorKind === "K" ? "Keys" : "Values"} · shape [{headDimension ?? "?"}]</button></SceneLabel>
        <SceneText text={headDimension && headDimension > dimensions ? `${dimensions} visual segments / ${headDimension} dimensions` : "Structural dimensions"} position={[0, -0.55, 0]} muted />
      </group>
    })}
  </group>
}
