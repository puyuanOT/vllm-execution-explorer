import { useLayoutEffect, useMemo, useRef } from "react"
import * as THREE from "three"

import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import type { Selection } from "../execution/types.ts"
import { requestColor } from "../theme.ts"
import { activeTeachingStep, architectureFocusLabel, architectureLabel, kvHeadForQuery, observedProfileTokenPositions, poolCellPosition, poolGridLayout, projectArchitecture, scheduledTokenPositions } from "./model.ts"
import type { ArchitectureFocus, MeasuredSceneProfile, ModelArchitecture, ScheduledTokenPosition } from "./model.ts"
import { BoxEdges, Connector, FloorGrid, Polyline, SceneEmpty, SceneText, SceneTile, SCENE_COLORS, sceneClick, useHoverCursor, type Position3 } from "./primitives.tsx"
import { createPoolRaycast } from "./poolPicking.ts"

export interface ModelStageProps {
  state: ReplayState
  manifest?: RecordingManifest | null
  requestOrder: string[]
  requestLabels?: ReadonlyMap<string, string>
  selection: Selection | null
  architectureFocus?: ArchitectureFocus | null
  activeStepId?: string | null
  trackedRequestId?: string | null
  trackedTokenPosition?: number | null
  measuredProfile?: MeasuredSceneProfile
  profileAvailable?: boolean
  onSelect: (selection: Selection) => void
  onArchitectureExplore?: (focus: ArchitectureFocus) => void
}

export function ModelStage(props: ModelStageProps) {
  const architecture = projectArchitecture(props.manifest ?? undefined)
  const focus = props.architectureFocus ?? { kind: "model" }
  if (!architecture.supported || !architecture.layerCount) {
    return <SceneEmpty label="Detailed architecture unavailable" detail={architecture.reason || "This recording has no captured supported architecture metadata."} />
  }
  const step = activeTeachingStep(props.state, props.activeStepId)
  const overview = focus.kind === "model"
  const linearFocus = focus.kind !== "model" && focus.kind !== "ffn" && architecture.layerTypes[focus.layerIndex] === "linear_attention"
  const profile = props.measuredProfile
  const evidence = profile ? profile.evidenceKind === "fixture" ? "Synthetic batch/layer activity · fixture profiler clock" : profile.evidenceKind === "gpu" ? "Measured batch/layer activity · profiler clock" : "Profiler batch/layer activity · source unverified"
    : props.state.trust === "known" ? props.profileAvailable ? "Logical model structure · layer activity available in GPU activity" : "Logical model structure · GPU layer timing not captured" : "Trace gap · structure is known; current token participation is unknown"
  const bottomY = overview ? -4.75 : -4.15
  return <group>
    <FloorGrid y={overview ? -6.2 : -5.5} />
    {overview ? <SceneText text={architectureLabel(props.manifest)} position={[0, 4.3, 0]} strong /> : null}
    {focus.kind === "ffn" ? <SceneText text={architectureFocusLabel(focus)} position={[0, 2.9, 0]} strong /> : null}
    {!overview ? <SceneText text={evidence} position={[0, 3.9, 0]} muted /> : null}
    {focus.kind === "model" ? <LayerStack {...props} architecture={architecture} />
      : focus.kind === "layer" ? <LayerStage {...props} architecture={architecture} layerIndex={focus.layerIndex} />
        : focus.kind === "ffn" ? <FfnStage architecture={architecture} measuredActive={profile?.activeModules?.some((module) => module.layerIndex === focus.layerIndex && module.module === "ffn")} />
          : linearFocus ? <GatedDeltaNetStage architecture={architecture} layerIndex={focus.layerIndex} measuredActive={profile?.activeModules?.some((module) => module.layerIndex === focus.layerIndex && module.module === "recurrent")} />
          : focus.kind === "head" ? <HeadStage {...props} architecture={architecture} focus={focus} />
            : <AttentionStage {...props} architecture={architecture} layerIndex={focus.layerIndex} />}
    {profile ? <MeasuredParticipantInputs {...props} profile={profile} y={bottomY} />
      : step && props.state.trust === "known" ? <ParticipantInputs {...props} y={bottomY} />
        : <SceneText text={step ? "Trace gap · current token participation is unknown" : "No scheduler step has reached this playhead"} position={[0, bottomY, 0.7]} muted />}
  </group>
}

export function LayerStack({ architecture, compact = false, baseY, ...props }: ModelStageProps & { architecture: ModelArchitecture; compact?: boolean; baseY?: number }) {
  const count = architecture.layerCount ?? 0
  const mesh = useRef<THREE.InstancedMesh>(null)
  const hover = useHoverCursor()
  const measuredLayers = useMemo(() => new Set(props.measuredProfile?.activeLayers.filter((layer) => Number.isInteger(layer) && layer >= 0 && layer < count) ?? []), [props.measuredProfile?.activeLayers, count])
  const selectedLayer = props.selection?.kind === "layer" || props.selection?.kind === "module" || props.selection?.kind === "head" ? props.selection.layerIndex : null
  const height = compact ? 3.74 : 5.15
  const width = compact ? 2.9 : 3.2
  const depth = compact ? 2.0 : 1.8
  const spacing = height / Math.max(1, count)
  const heightForLayer = (layer: number) => Math.max(spacing * (architecture.recurrent ? architecture.layerTypes[layer] === "linear_attention" ? 0.38 : 0.72 : 0.5), 0.012)
  const top = compact ? (baseY ?? 0) + height : height / 2
  const layerY = (layer: number) => top - (layer + 0.5) * spacing
  useLayoutEffect(() => {
    const instance = mesh.current
    if (!instance) return
    const transform = new THREE.Object3D()
    const color = new THREE.Color()
    for (let layer = 0; layer < count; layer += 1) {
      transform.position.set(0, layerY(layer), 0)
      transform.scale.set(width, heightForLayer(layer), depth)
      transform.updateMatrix()
      instance.setMatrixAt(layer, transform.matrix)
      color.set(measuredLayers.has(layer) ? "#9e7231"
        : layer === selectedLayer ? SCENE_COLORS.surfaceSelected
          : architecture.layerTypes[layer] === "linear_attention" ? "#2b3444" : "#1b2a42")
      instance.setColorAt(layer, color)
    }
    instance.instanceMatrix.needsUpdate = true
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true
    instance.computeBoundingSphere()
  }, [count, height, spacing, width, depth, top, measuredLayers, selectedLayer, architecture.layerTypes])
  const markers = [...new Set([0, Math.floor(count / 4), Math.floor(count / 2), Math.floor(count * 3 / 4), count - 1])]
  const plate: Position3 = [width, Math.max(spacing * 0.5, 0.012), depth]
  return <group>
    <instancedMesh ref={mesh} args={[undefined, undefined, count]} {...hover.bind}
      onClick={(event) => {
        if (!sceneClick(event) || event.instanceId == null) return
        event.stopPropagation(); props.onSelect({ kind: "layer", layerIndex: event.instanceId })
      }}
      onDoubleClick={(event) => {
        if (!sceneClick(event) || event.instanceId == null) return
        event.stopPropagation()
        props.onSelect({ kind: "layer", layerIndex: event.instanceId })
        props.onArchitectureExplore?.({ kind: "layer", layerIndex: event.instanceId })
      }}>
      <boxGeometry args={[1, 1, 1]} /><meshStandardMaterial roughness={0.7} metalness={0.1} />
    </instancedMesh>
    <StackRims count={count} plate={plate} layerY={layerY} layerKinds={architecture.recurrent ? architecture.layerTypes : undefined} />
    {markers.map((layer) => <BoxEdges key={layer} size={[width, heightForLayer(layer), depth]} position={[0, layerY(layer), 0]} color={SCENE_COLORS.edgeMarker} width={1} />)}
    {selectedLayer != null && selectedLayer >= 0 && selectedLayer < count ? <BoxEdges size={[width + 0.18, heightForLayer(selectedLayer) + 0.07, depth + 0.18]} position={[0, layerY(selectedLayer), 0]} color={SCENE_COLORS.accent} width={2.25} /> : null}
    {[...measuredLayers].map((layer) => <group key={`m-${layer}`} position={[0, layerY(layer), 0]}>
      <mesh position={[0, 0, depth / 2 + 0.025]} raycast={() => null}>
        <planeGeometry args={[width, heightForLayer(layer)]} />
        <meshBasicMaterial color={SCENE_COLORS.measured} toneMapped={false} />
      </mesh>
      <BoxEdges size={[width, heightForLayer(layer), depth]} color={SCENE_COLORS.measured} width={3} />
    </group>)}
    <BoxEdges size={[width + 0.35, height + 0.4, depth + 0.35]} position={[0, top - height / 2, 0]} color={SCENE_COLORS.edge} opacity={hover.hovered ? 1 : 0.95} width={2.25} />
    {markers.map((layer) => <SceneText key={`label-${layer}`} text={`L${layer + 1}${architecture.recurrent ? architecture.layerTypes[layer] === "linear_attention" ? " · GDN" : " · Attention" : ""}`} position={[width / 2 + (architecture.recurrent ? 0.8 : 0.5), layerY(layer), depth / 2]} muted={compact} strong={!compact} />)}
    {!compact ? <>
      <SceneText text="Token embeddings" position={[0, top + 0.45, 0]} />
      <SceneText text="Norm → LM head → sampling" position={[0, top - height - 0.6, 0]} muted />
    </> : null}
    {props.measuredProfile && !compact ? <SceneText text={measuredLayers.size ? `Active: ${[...measuredLayers].map((layer) => `Layer ${layer + 1}`).join(", ")}` : "No mapped layer activity"}
      position={[-3.9, 0, 1]} /> : null}
  </group>
}

function StackRims({ count, plate, layerY, layerKinds }: { count: number; plate: Position3; layerY: (layer: number) => number; layerKinds?: string[] }) {
  const kindKey = layerKinds?.join(",") ?? ""
  const lines = useMemo(() => {
    const unit = new THREE.EdgesGeometry(new THREE.BoxGeometry(...plate))
    const source = unit.getAttribute("position").array as ArrayLike<number>
    const positions = new Float32Array(source.length * count)
    for (let layer = 0; layer < count; layer += 1) {
      const offset = layerY(layer)
      for (let index = 0; index < source.length; index += 3) {
        const target = layer * source.length + index
        positions[target] = source[index]
        positions[target + 1] = source[index + 1] * (layerKinds ? layerKinds[layer] === "linear_attention" ? 0.76 : 1.44 : 1) + offset
        positions[target + 2] = source[index + 2]
      }
    }
    unit.dispose()
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    const object = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: SCENE_COLORS.edgeDim }))
    object.raycast = () => undefined
    return object
  }, [count, plate[0], plate[1], plate[2], layerY(0), kindKey])
  useLayoutEffect(() => () => { lines.geometry.dispose(); (lines.material as THREE.Material).dispose() }, [lines])
  return <primitive object={lines} />
}

function LayerStage({ architecture, layerIndex, ...props }: ModelStageProps & { architecture: ModelArchitecture; layerIndex: number }) {
  const moduleActive = (module: "attention" | "ffn" | "recurrent") => props.measuredProfile?.activeModules?.some((entry) => entry.layerIndex === layerIndex && entry.module === module)
  if (architecture.layerTypes[layerIndex] === "linear_attention") return <GatedDeltaNetStage architecture={architecture} layerIndex={layerIndex} measuredActive={moduleActive("recurrent")} />
  return <group>
    <SceneText text={`Layer ${layerIndex + 1}`} position={[0, 2.85, 0]} strong />
    <Connector from={[-5, 0, 0]} to={[-3.25, 0, 0]} />
    <Connector from={[-0.35, 0, 0]} to={[0.85, 0, 0]} />
    <Connector from={[4.05, 0, 0]} to={[5.4, 0, 0]} />
    <SceneTile position={[-1.8, 0, 0]} label="Self-attention"
      detail={`${architecture.queryHeads ?? "?"} Q / ${architecture.kvHeads ?? "?"} KV heads${architecture.attentionOutputGate ? " · gated output" : ""}`} width={2.9}
      color={moduleActive("attention") ? SCENE_COLORS.measured : SCENE_COLORS.accent}
      selected={props.selection?.kind === "module" && props.selection.layerIndex === layerIndex && props.selection.module === "attention"}
      onSelect={() => props.onSelect({ kind: "module", layerIndex, module: "attention" })}
      onExplore={() => props.onArchitectureExplore?.({ kind: "attention", layerIndex })} />
    <SceneTile position={[2.45, 0, 0]} label="Feed-forward"
      detail={`${architecture.hiddenSize ?? "?"} → ${architecture.intermediateSize ?? "?"} → ${architecture.hiddenSize ?? "?"}`} width={3.2}
      color={moduleActive("ffn") ? SCENE_COLORS.measured : SCENE_COLORS.accent}
      selected={props.selection?.kind === "module" && props.selection.layerIndex === layerIndex && props.selection.module === "ffn"}
      onSelect={() => props.onSelect({ kind: "module", layerIndex, module: "ffn" })}
      onExplore={() => props.onArchitectureExplore?.({ kind: "ffn", layerIndex })} />
    <SceneText text="RMSNorm" position={[-4.3, 0.55, 0]} muted />
    <SceneText text="+ residual → RMSNorm" position={[0.25, 0.75, 0]} muted />
    <Polyline points={[[-4.5, 0.2, -0.25], [-4.5, 1.65, -0.25], [0.25, 1.65, -0.25], [0.25, 0.25, -0.25]]} color={SCENE_COLORS.lineDim} />
    <SceneText text="Residual path" position={[-2.3, 1.9, -0.25]} muted />
    <Polyline points={[[0.25, -0.2, -0.25], [0.25, -1.45, -0.25], [4.85, -1.45, -0.25], [4.85, -0.25, -0.25]]} color={SCENE_COLORS.lineDim} />
    <SceneText text="Feed-forward residual" position={[2.6, -1.75, -0.25]} muted />
    <SceneText text={props.measuredProfile ? "Amber: mapped profiler scope active · activation values not captured" : "Operator structure; no activation values were captured"} position={[0, -2.3, 0]} muted />
  </group>
}

function GatedDeltaNetStage({ architecture, layerIndex, measuredActive }: { architecture: ModelArchitecture; layerIndex: number; measuredActive?: boolean }) {
  const recurrent = architecture.recurrent
  return <group>
    <SceneText text={`Layer ${layerIndex + 1} · Gated DeltaNet`} position={[0, 2.85, 0]} strong />
    <SceneTile position={[-3.8, 0.25, 0]} label="Input projections + convolution" detail={`Convolution kernel ${recurrent?.convKernelSize ?? "not captured"}`} width={3.5} />
    <SceneTile position={[0, 0.25, 0]} label="Gated recurrent update" detail={recurrent ? `${recurrent.keyHeads} key / ${recurrent.valueHeads} value heads` : "Recurrent geometry unavailable"} width={3.4} color={measuredActive ? SCENE_COLORS.measured : SCENE_COLORS.accent} />
    <SceneTile position={[3.8, 0.25, 0]} label="Output projection + dense FFN" detail={`${architecture.hiddenSize ?? "?"} hidden dimensions`} width={3.4} />
    <Connector from={[-2, 0.25, 0]} to={[-1.75, 0.25, 0]} /><Connector from={[1.75, 0.25, 0]} to={[2.05, 0.25, 0]} />
    <SceneTile position={[0, -1.45, 1]} label="Recurrent state" detail={recurrent ? `${recurrent.valueHeads} value heads · ${recurrent.keyHeadDim} × ${recurrent.valueHeadDim} state per head` : "State dimensions unavailable"} width={4.4} height={0.95} />
    <Polyline points={[[0, -0.2, 0], [0, -0.55, 1], [0, -0.95, 1]]} dashed />
    <SceneText text="Recurrent state is separate from the full-attention KV pool" position={[0, -2.3, 1]} muted />
    <SceneText text={measuredActive ? "Captured GDN scope active · state values and update timing not captured" : "Captured model structure · recurrent state values not captured"} position={[0, -2.75, 1]} muted />
  </group>
}

function AttentionStage({ architecture, layerIndex, ...props }: ModelStageProps & { architecture: ModelArchitecture; layerIndex: number }) {
  const qHeads = architecture.queryHeads ?? 0
  const kvHeads = architecture.kvHeads ?? 0
  if (!qHeads || !kvHeads) return <SceneEmpty label="Head geometry was not captured" />
  const grouped = qHeads % kvHeads === 0
  const headsPerKv = grouped ? qHeads / kvHeads : 0
  const columns = Math.min(8, grouped ? kvHeads : qHeads)
  const colWidth = Math.min(1.34, 10.4 / columns)
  const rows = grouped ? headsPerKv * Math.ceil(kvHeads / columns) : Math.ceil(qHeads / columns)
  const rowSpacing = Math.min(0.56, 2.6 / rows)
  const tileHeight = Math.min(0.4, rowSpacing * 0.78)
  const selectedHead = props.selection?.kind === "head" && props.selection.layerIndex === layerIndex ? props.selection : null
  const selectedKv = selectedHead?.headKind === "kv" ? selectedHead.headIndex
    : selectedHead ? kvHeadForQuery(selectedHead.headIndex, qHeads, kvHeads) : null
  const qPosition = (index: number): Position3 => {
    const kv = grouped ? Math.floor(index / headsPerKv) : index % columns
    const row = grouped ? (index % headsPerKv) + Math.floor(kv / columns) * headsPerKv : Math.floor(index / columns)
    return [((kv % columns) - (columns - 1) / 2) * colWidth, 2.55 - row * rowSpacing, -1.2]
  }
  const kvColumns = Math.min(8, kvHeads)
  const kvPosition = (index: number): Position3 => [
    ((index % kvColumns) - (kvColumns - 1) / 2) * colWidth,
    -0.75 - Math.floor(index / kvColumns) * 0.85, 1.2,
  ]
  const wallTop = 2.55 + rowSpacing * 0.5 + 0.25
  const wallBottom = 2.55 - (rows - 1) * rowSpacing - rowSpacing * 0.5 - 0.25
  const wallHalf = (columns * colWidth) / 2 + 0.3
  return <group>
    <SceneText text={`${qHeads} query heads share ${kvHeads} KV heads`} position={[0, wallTop + 0.35, -1.2]} strong />
    <Polyline points={[[-wallHalf, wallBottom, -1.55], [wallHalf, wallBottom, -1.55], [wallHalf, wallTop, -1.55], [-wallHalf, wallTop, -1.55], [-wallHalf, wallBottom, -1.55]]}
      color={SCENE_COLORS.explain} opacity={0.7} dashed dash={0.3} gap={0.18} />
    {grouped ? Array.from({ length: kvHeads }, (_, kv) => {
      const to = kvPosition(kv)
      return <group key={`sharing-${kv}`}>
        {Array.from({ length: headsPerKv }, (_, offset) => {
          const index = kv * headsPerKv + offset
          const from = qPosition(index)
          const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(from[0], from[1] - tileHeight / 2, from[2] + 0.14),
            new THREE.Vector3(from[0] + (to[0] - from[0]) * 0.3, (from[1] + to[1]) / 2, (from[2] + to[2]) / 2 - 0.2),
            new THREE.Vector3(to[0], to[1] + 0.3, to[2] - 0.22),
          ])
          const points = curve.getPoints(18).map((point) => [point.x, point.y, point.z] as Position3)
          const selected = selectedHead?.headKind === "query" && selectedHead.headIndex === index
          const related = selectedKv === kv
          return <Polyline key={offset} points={points} color={selected ? SCENE_COLORS.accentSoft : related ? SCENE_COLORS.accent : "#1b2a40"} />
        })}
      </group>
    }) : null}
    {Array.from({ length: qHeads }, (_, index) => {
      const selected = selectedHead?.headKind === "query" && selectedHead.headIndex === index
      const related = selectedKv != null && kvHeadForQuery(index, qHeads, kvHeads) === selectedKv
      return <HeadCell key={`q-${index}`} label={`Q${index + 1}`} position={qPosition(index)} selected={selected}
        related={related} size={[Math.min(1.0, colWidth * 0.76), tileHeight, 0.26]}
        onSelect={() => props.onSelect({ kind: "head", layerIndex, headIndex: index, headKind: "query" })}
        onExplore={() => props.onArchitectureExplore?.({ kind: "head", layerIndex, headIndex: index, headKind: "query" })} />
    })}
    <mesh position={[0, -1.12, 1.2]} raycast={() => null}>
      <boxGeometry args={[columns * colWidth + 0.6, 0.1, 1.25]} /><meshStandardMaterial color={SCENE_COLORS.surface} roughness={0.8} />
    </mesh>
    <BoxEdges size={[columns * colWidth + 0.6, 0.1, 1.25]} position={[0, -1.12, 1.2]} color={SCENE_COLORS.edgeDim} />
    {Array.from({ length: kvHeads }, (_, index) => {
      const [x, y, z] = kvPosition(index)
      const selected = selectedHead?.headKind === "kv" && selectedHead.headIndex === index
      const related = selectedKv === index
      const cell = Math.min(0.44, colWidth * 0.32)
      return <group key={`kv-${index}`}>
        <HeadCell label="K" position={[x - cell * 0.6, y, z]} selected={selected} related={related} size={[cell, 0.56, cell]}
          onSelect={() => props.onSelect({ kind: "head", layerIndex, headIndex: index, headKind: "kv" })}
          onExplore={() => props.onArchitectureExplore?.({ kind: "head", layerIndex, headIndex: index, headKind: "kv" })} />
        <HeadCell label="V" position={[x + cell * 0.6, y, z]} selected={selected} related={related} size={[cell, 0.56, cell]}
          onSelect={() => props.onSelect({ kind: "head", layerIndex, headIndex: index, headKind: "kv" })}
          onExplore={() => props.onArchitectureExplore?.({ kind: "head", layerIndex, headIndex: index, headKind: "kv" })} />
        <SceneText text={`KV${index + 1}`} position={[x, y - 0.55, z + 0.5]} muted />
      </group>
    })}
    <SceneText text={`Layer ${layerIndex + 1}`} position={[-wallHalf + 1.4, wallBottom - 0.22, -1.55]} muted />
    <SceneText text={`Head dimension ${architecture.headDim ?? "not recorded"}; edges show model KV sharing, not tensor values`} position={[0, -2.15, 1.2]} muted />
  </group>
}

function HeadCell({ label, position, selected, related = false, size, onSelect, onExplore }: {
  label: string; position: Position3; selected: boolean; related?: boolean; size: Position3; onSelect: () => void; onExplore: () => void
}) {
  const hover = useHoverCursor()
  return <group position={position}>
    <mesh {...hover.bind}
      onClick={(event) => { if (sceneClick(event)) { event.stopPropagation(); onSelect() } }}
      onDoubleClick={(event) => { if (sceneClick(event)) { event.stopPropagation(); onSelect(); onExplore() } }}>
      <boxGeometry args={size} />
      <meshStandardMaterial color={selected ? SCENE_COLORS.surfaceSelected : related ? SCENE_COLORS.surfaceRelated : "#142034"} roughness={0.7} metalness={0.1} />
    </mesh>
    <BoxEdges size={size} color={selected ? SCENE_COLORS.accentSoft : related ? SCENE_COLORS.accent : hover.hovered ? SCENE_COLORS.edgeMarker : SCENE_COLORS.lineDim} />
    {size[0] >= 0.5 || label.length === 1 ? <SceneText text={label} position={[0, 0, size[2] / 2 + 0.02]} /> : null}
  </group>
}

function HeadStage({ architecture, focus }: ModelStageProps & { architecture: ModelArchitecture; focus: Extract<ArchitectureFocus, { kind: "head" }> }) {
  const kv = focus.headKind === "kv" ? focus.headIndex : kvHeadForQuery(focus.headIndex, architecture.queryHeads ?? 0, architecture.kvHeads ?? 0)
  const queriesPerKv = architecture.queryHeads && architecture.kvHeads && architecture.queryHeads % architecture.kvHeads === 0 ? architecture.queryHeads / architecture.kvHeads : null
  const queryLabel = focus.headKind === "query" ? `Q${focus.headIndex + 1}` : queriesPerKv
    ? `Q${focus.headIndex * queriesPerKv + 1}–${(focus.headIndex + 1) * queriesPerKv}` : "Shared query group"
  return <group>
    <SceneText text={`Layer ${focus.layerIndex + 1} / ${focus.headKind === "query" ? "query" : "KV"} head ${focus.headIndex + 1}`} position={[0, 2.85, 0]} strong />
    <SceneText text="Q/K RMSNorm → RoPE → grouped-query attention" position={[0, 2.3, 0]} muted />
    <SceneTile position={[-3.4, 0.8, 0.2]} label={queryLabel} detail={`Dimension ${architecture.headDim ?? "unknown"}`} width={2.6} />
    <SceneTile position={[-0.1, 0.8, -0.7]} label={`K${kv == null ? "?" : kv + 1}`} detail="Computed prefix keys" width={2.6} />
    <SceneTile position={[-0.1, -0.85, -0.7]} label={`V${kv == null ? "?" : kv + 1}`} detail="Computed prefix values" width={2.6} />
    <SceneTile position={[3.6, 0, 0]} label="Attention output" detail={architecture.attentionOutputGate ? "Attention → sigmoid gate → output projection" : "Causal attention → output projection"} width={3} />
    <Connector from={[-2.1, 0.8, 0.2]} to={[2.05, 0.1, 0]} />
    <Connector from={[1.2, 0.8, -0.7]} to={[2.05, 0.1, 0]} />
    <Connector from={[1.2, -0.85, -0.7]} to={[2.05, -0.1, 0]} />
    <SceneText text="Conceptual Q · K → causal mask → softmax → V; individual head timing unavailable" position={[0, -1.85, 0]} muted />
    <SceneText text="Numeric Q/K/V and attention weights were not captured" position={[0, -2.25, 0]} muted />
  </group>
}

function FfnStage({ architecture, measuredActive }: { architecture: ModelArchitecture; measuredActive?: boolean }) {
  const gated = architecture.modelType === "qwen3" || architecture.recurrent != null
  return <group>
    <SceneTile position={[-3.6, 0, 0]} label={gated ? "Gate + up projections" : "First projection"} detail={`${architecture.hiddenSize ?? "?"} → ${architecture.intermediateSize ?? "?"}`} width={3.2} color={measuredActive ? SCENE_COLORS.measured : SCENE_COLORS.accent} />
    <SceneTile position={[0.1, 0, 0]} label={gated ? `${architecture.activation ?? "Activation"} × up` : architecture.activation ?? "Activation"} detail={gated ? "Elementwise gating" : "Configured activation"} width={2.6} />
    <SceneTile position={[3.8, 0, 0]} label="Down projection" detail={`${architecture.intermediateSize ?? "?"} → ${architecture.hiddenSize ?? "?"}`} width={3.1} />
    <Connector from={[-1.95, 0, 0]} to={[-1.25, 0, 0]} /><Connector from={[1.45, 0, 0]} to={[2.2, 0, 0]} />
    <SceneText text={measuredActive ? "Measured FFN scope active · individual projection/vector values not captured" : "Each input token has a feed-forward vector; values are not captured"} position={[0, -2.2, 0]} muted />
  </group>
}

/** The stable iteration band carries request counts/ranges; only pickable inputs live in the scene. */
function ParticipantInputs({ state, requestOrder, selection, activeStepId, trackedRequestId, trackedTokenPosition, onSelect, y }: ModelStageProps & { y: number }) {
  const step = activeTeachingStep(state, activeStepId)
  const tokens = useMemo(() => step ? scheduledTokenPositions(step) : [], [step])
  if (!step || state.trust !== "known") return null
  const requestId = selection?.kind === "token" ? selection.requestId : selection?.kind === "request" ? selection.id : trackedRequestId
  const position = selection?.kind === "token" ? selection.position : trackedTokenPosition
  return <group position={[-3.9, y + 0.16, 0.7]}>
    <TokenBatch tokens={tokens} requestOrder={requestOrder} selectedRequestId={requestId ?? null} selectedPosition={position ?? null} onSelect={onSelect} />
    <SceneText text="Scheduled input positions" position={[0, 0.9, 0]} muted />
  </group>
}

function MeasuredParticipantInputs({ profile, requestOrder, selection, trackedRequestId, trackedTokenPosition, onSelect, y }: ModelStageProps & { profile: MeasuredSceneProfile; y: number }) {
  const tokens = useMemo(() => observedProfileTokenPositions(profile.packedRows), [profile.packedRows])
  if (!tokens.length) return null
  const requestId = selection?.kind === "token" ? selection.requestId : selection?.kind === "request" ? selection.id : trackedRequestId
  const position = selection?.kind === "token" ? selection.position : trackedTokenPosition
  return <group position={[-3.9, y + 0.16, 0.7]}>
    <TokenBatch tokens={tokens} requestOrder={requestOrder} selectedRequestId={requestId ?? null} selectedPosition={position ?? null} onSelect={onSelect} />
    <SceneText text={profile.evidenceKind === "fixture" ? "Fixture packed rows" : "Observed packed rows"} position={[0, 0.9, 0]} muted />
  </group>
}

function TokenBatch({ tokens, requestOrder, selectedRequestId, selectedPosition, onSelect }: {
  tokens: ScheduledTokenPosition[]; requestOrder: string[];
  selectedRequestId: string | null; selectedPosition: number | null; onSelect: (selection: Selection) => void
}) {
  const layout = useMemo(() => poolGridLayout(tokens.length, 1.4), [tokens.length])
  const raycast = useMemo(() => createPoolRaycast(layout), [layout])
  const mesh = useRef<THREE.InstancedMesh>(null)
  const hover = useHoverCursor()
  useLayoutEffect(() => {
    const instance = mesh.current
    if (!instance) return
    const transform = new THREE.Object3D()
    const color = new THREE.Color()
    const decodeTint = new THREE.Color(SCENE_COLORS.ink)
    tokens.forEach((token, index) => {
      transform.position.set(...poolCellPosition(token.row ?? index, layout)); transform.scale.setScalar(layout.spacing * 0.74); transform.updateMatrix()
      instance.setMatrixAt(index, transform.matrix)
      color.set(requestColor(requestOrder, token.requestId)).multiplyScalar(0.75)
      if (token.kind === "decode") color.lerp(decodeTint, 0.35)
      if (token.requestId === selectedRequestId && token.position === selectedPosition) color.set(SCENE_COLORS.accentSoft)
      instance.setColorAt(index, color)
    })
    instance.instanceMatrix.needsUpdate = true
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true
    instance.computeBoundingSphere()
  }, [tokens, layout, requestOrder, selectedRequestId, selectedPosition])
  return <instancedMesh ref={mesh} args={[undefined, undefined, tokens.length]} raycast={raycast} {...hover.bind}
    onClick={(event) => {
      if (!sceneClick(event) || event.instanceId == null) return
      const token = tokens[event.instanceId]
      if (!token) return
      event.stopPropagation(); onSelect({ kind: "token", requestId: token.requestId, position: token.position })
    }}>
    <boxGeometry args={[1, 1, 1]} /><meshBasicMaterial toneMapped={false} />
  </instancedMesh>
}
