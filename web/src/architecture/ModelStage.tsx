import { useLayoutEffect, useMemo, useRef } from "react"
import * as THREE from "three"

import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import type { Selection } from "../execution/types.ts"
import { requestLabel } from "../explore/model.ts"
import { requestColor } from "../theme.ts"
import { activeTeachingStep, architectureFocusLabel, architectureLabel, kvHeadForQuery, observedProfileTokenPositions, poolCellPosition, poolGridLayout, projectArchitecture, scheduledTokenPositions, teachingLayer } from "./model.ts"
import type { ArchitectureFocus, MeasuredSceneProfile, ModelArchitecture, ScheduledTokenPosition } from "./model.ts"
import { BoxEdges, Connector, FloorGrid, Polyline, SceneEmpty, SceneLabel, SceneText, SceneTile, SCENE_COLORS, sceneClick, useHoverCursor, type Position3 } from "./primitives.tsx"
import { createPoolRaycast } from "./poolPicking.ts"

export interface ModelStageProps {
  state: ReplayState
  manifest?: RecordingManifest | null
  requestOrder: string[]
  requestLabels?: ReadonlyMap<string, string>
  selection: Selection | null
  architectureFocus?: ArchitectureFocus | null
  activeStepId?: string | null
  teachingProgress?: number
  trackedRequestId?: string | null
  trackedTokenPosition?: number | null
  measuredProfile?: MeasuredSceneProfile
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
  const profile = props.measuredProfile
  const profileTokenCount = profile?.packedRows.filter((row) => !row.isPadding && row.requestId != null && row.tokenPosition != null).length ?? 0
  const evidence = profile ? profile.evidenceKind === "fixture" ? "Synthetic batch/layer activity · fixture profiler clock" : profile.evidenceKind === "gpu" ? "Measured batch/layer activity · profiler clock" : "Profiler batch/layer activity · source unverified"
    : props.state.trust === "known" ? "Logical model structure · GPU layer timing not captured" : "Trace gap · structure is known; current token participation is unknown"
  const tracked = props.trackedRequestId && overview && !profile
    ? `${props.trackedRequestId}${props.trackedTokenPosition == null ? "" : ` / position ${props.trackedTokenPosition}`} / ${step?.requests.some((request) => request.requestId === props.trackedRequestId && (props.trackedTokenPosition == null || props.trackedTokenPosition >= request.rangeStart && props.trackedTokenPosition < request.rangeStart + request.rangeLength)) ? "scheduled in this step" : "not scheduled in this step"}`
    : null
  const bottomY = overview ? -4.75 : -4.15
  return <group>
    <FloorGrid y={overview ? -6.2 : -5.5} />
    {overview ? <SceneText text={architectureLabel(props.manifest)} position={[0, 4.3, 0]} strong /> : null}
    {focus.kind === "ffn" ? <SceneText text={architectureFocusLabel(focus)} position={[0, 2.9, 0]} strong /> : null}
    <SceneText text={evidence} position={[0, overview ? 3.85 : 3.9, 0]} muted />
    {tracked ? <SceneText text={tracked} position={[0, 3.45, 0]} muted /> : null}
    {focus.kind === "model" ? <LayerStack {...props} architecture={architecture} />
      : focus.kind === "layer" ? <LayerStage {...props} architecture={architecture} layerIndex={focus.layerIndex} />
        : focus.kind === "ffn" ? <FfnStage architecture={architecture} measuredActive={profile?.activeModules?.some((module) => module.layerIndex === focus.layerIndex && module.module === "ffn")} />
          : focus.kind === "head" ? <HeadStage {...props} architecture={architecture} focus={focus} />
            : <AttentionStage {...props} architecture={architecture} layerIndex={focus.layerIndex} />}
    {profile ? <MeasuredParticipantInputs {...props} profile={profile} y={bottomY}
      summary={profileTokenCount ? `${profile.stepId ?? "Measured batch"}: ${profileTokenCount} observed packed inputs / padding excluded` : "No observed packed-row mapping for this profiler position"} />
      : step && props.state.trust === "known" ? <ParticipantInputs {...props} y={bottomY} summary={`${step.id}: ${step.totalScheduledTokens} scheduled input positions / ${step.requests.length} request${step.requests.length === 1 ? "" : "s"}`} />
        : <SceneText text={step ? "Trace gap · current token participation is unknown" : "No scheduler step has reached this playhead"} position={[0, bottomY, 0.7]} muted />}
  </group>
}

export function LayerStack({ architecture, compact = false, baseY, ...props }: ModelStageProps & { architecture: ModelArchitecture; compact?: boolean; baseY?: number }) {
  const count = architecture.layerCount ?? 0
  const mesh = useRef<THREE.InstancedMesh>(null)
  const hover = useHoverCursor()
  const activeLayer = !props.measuredProfile && props.state.trust === "known" && activeTeachingStep(props.state, props.activeStepId)
    ? teachingLayer(props.teachingProgress, count) : null
  const measuredLayers = useMemo(() => new Set(props.measuredProfile?.activeLayers.filter((layer) => Number.isInteger(layer) && layer >= 0 && layer < count) ?? []), [props.measuredProfile?.activeLayers, count])
  const selectedLayer = props.selection?.kind === "layer" || props.selection?.kind === "module" || props.selection?.kind === "head" ? props.selection.layerIndex : null
  const height = compact ? 3.74 : 5.15
  const width = compact ? 2.9 : 3.2
  const depth = compact ? 2.0 : 1.8
  const spacing = height / Math.max(1, count)
  const top = compact ? (baseY ?? 0) + height : height / 2
  const layerY = (layer: number) => top - (layer + 0.5) * spacing
  useLayoutEffect(() => {
    const instance = mesh.current
    if (!instance) return
    const transform = new THREE.Object3D()
    const color = new THREE.Color()
    for (let layer = 0; layer < count; layer += 1) {
      transform.position.set(0, layerY(layer), 0)
      transform.scale.set(width, Math.max(spacing * 0.5, 0.012), depth)
      transform.updateMatrix()
      instance.setMatrixAt(layer, transform.matrix)
      color.set(layer === selectedLayer ? SCENE_COLORS.surfaceSelected
        : measuredLayers.has(layer) ? "#6b5126"
          : layer === activeLayer ? "#3d3477"
            : "#1b2a42")
      instance.setColorAt(layer, color)
    }
    instance.instanceMatrix.needsUpdate = true
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true
    instance.computeBoundingSphere()
  }, [count, height, spacing, width, depth, top, activeLayer, measuredLayers, selectedLayer])
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
    <StackRims count={count} plate={plate} layerY={layerY} />
    {markers.map((layer) => <BoxEdges key={layer} size={plate} position={[0, layerY(layer), 0]} color={SCENE_COLORS.edgeMarker} />)}
    {selectedLayer != null && selectedLayer < count ? <BoxEdges size={plate} position={[0, layerY(selectedLayer), 0]} color={SCENE_COLORS.accent} /> : null}
    {activeLayer != null ? <BoxEdges size={plate} position={[0, layerY(activeLayer), 0]} color={SCENE_COLORS.explain} /> : null}
    {[...measuredLayers].map((layer) => <BoxEdges key={`m-${layer}`} size={plate} position={[0, layerY(layer), 0]} color={SCENE_COLORS.measured} />)}
    <BoxEdges size={[width + 0.35, height + 0.4, depth + 0.35]} position={[0, top - height / 2, 0]} color={SCENE_COLORS.edge} opacity={hover.hovered ? 1 : 0.85} />
    {markers.map((layer) => <SceneText key={`label-${layer}`} text={`L${layer + 1}`} position={[width / 2 + 0.5, layerY(layer), depth / 2]} muted />)}
    {!compact ? <>
      <SceneText text="Token embeddings" position={[0, top + 0.45, 0]} />
      <SceneText text="Norm → LM head → sampling" position={[0, top - height - 0.6, 0]} muted />
    </> : null}
    {activeLayer != null && !compact ? <>
      <Polyline points={[[-3.6, layerY(activeLayer), 0.2], [-width / 2 - 0.12, layerY(activeLayer), 0.2]]} color={SCENE_COLORS.explain} dashed />
      <SceneText text={`Walkthrough position · layer ${activeLayer + 1}`} position={[-3.6, layerY(activeLayer) + 0.32, 0.2]} />
    </> : null}
    {props.measuredProfile && !compact ? <SceneText text={measuredLayers.size ? `Profiler activity: ${[...measuredLayers].map((layer) => `L${layer + 1}`).join(", ")}` : "No mapped layer active at this profiler instant"}
      position={[-3.9, 0, 1]} /> : null}
  </group>
}

function StackRims({ count, plate, layerY }: { count: number; plate: Position3; layerY: (layer: number) => number }) {
  const lines = useMemo(() => {
    const unit = new THREE.EdgesGeometry(new THREE.BoxGeometry(...plate))
    const source = unit.getAttribute("position").array as ArrayLike<number>
    const positions = new Float32Array(source.length * count)
    for (let layer = 0; layer < count; layer += 1) {
      const offset = layerY(layer)
      for (let index = 0; index < source.length; index += 3) {
        const target = layer * source.length + index
        positions[target] = source[index]
        positions[target + 1] = source[index + 1] + offset
        positions[target + 2] = source[index + 2]
      }
    }
    unit.dispose()
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    const object = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: SCENE_COLORS.edgeDim }))
    object.raycast = () => undefined
    return object
  }, [count, plate[0], plate[1], plate[2], layerY(0)])
  useLayoutEffect(() => () => { lines.geometry.dispose(); (lines.material as THREE.Material).dispose() }, [lines])
  return <primitive object={lines} />
}

function LayerStage({ architecture, layerIndex, ...props }: ModelStageProps & { architecture: ModelArchitecture; layerIndex: number }) {
  const moduleActive = (module: "attention" | "ffn") => props.measuredProfile?.activeModules?.some((entry) => entry.layerIndex === layerIndex && entry.module === module)
  return <group>
    <SceneText text={`Layer ${layerIndex + 1}`} position={[0, 2.85, 0]} strong />
    <Connector from={[-5, 0, 0]} to={[-3.25, 0, 0]} />
    <Connector from={[-0.35, 0, 0]} to={[0.85, 0, 0]} />
    <Connector from={[4.05, 0, 0]} to={[5.4, 0, 0]} />
    <SceneTile position={[-1.8, 0, 0]} label="Self-attention"
      detail={`${architecture.queryHeads ?? "?"} Q / ${architecture.kvHeads ?? "?"} KV heads`} width={2.9}
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
    <SceneText text={architecture.modelType === "qwen3" ? "RMSNorm" : "LayerNorm"} position={[-4.3, 0.55, 0]} muted />
    <SceneText text={architecture.modelType === "qwen3" ? "+ residual → RMSNorm" : "+ residual → LayerNorm"} position={[0.25, 0.75, 0]} muted />
    <Polyline points={[[-4.5, 0.2, -0.25], [-4.5, 1.65, -0.25], [0.25, 1.65, -0.25], [0.25, 0.25, -0.25]]} color={SCENE_COLORS.lineDim} />
    <SceneText text="Residual path" position={[-2.3, 1.9, -0.25]} muted />
    <Polyline points={[[0.25, -0.2, -0.25], [0.25, -1.45, -0.25], [4.85, -1.45, -0.25], [4.85, -0.25, -0.25]]} color={SCENE_COLORS.lineDim} />
    <SceneText text="Feed-forward residual" position={[2.6, -1.75, -0.25]} muted />
    <SceneText text={props.measuredProfile ? "Amber: mapped profiler scope active · activation values not captured" : "Operator structure; no activation values were captured"} position={[0, -2.3, 0]} muted />
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
    <SceneText text={`Walkthrough position · layer ${layerIndex + 1}`} position={[-wallHalf + 1.4, wallBottom - 0.22, -1.55]} muted />
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
    {architecture.modelType === "qwen3" ? <SceneText text="Q/K RMSNorm → RoPE → grouped-query attention" position={[0, 2.3, 0]} muted /> : null}
    <SceneTile position={[-3.4, 0.8, 0.2]} label={queryLabel} detail={`Dimension ${architecture.headDim ?? "unknown"}`} width={2.6} />
    <SceneTile position={[-0.1, 0.8, -0.7]} label={`K${kv == null ? "?" : kv + 1}`} detail="Computed prefix keys" width={2.6} />
    <SceneTile position={[-0.1, -0.85, -0.7]} label={`V${kv == null ? "?" : kv + 1}`} detail="Computed prefix values" width={2.6} />
    <SceneTile position={[3.6, 0, 0]} label="Attention output" detail="Causal attention → output projection" width={3} />
    <Connector from={[-2.1, 0.8, 0.2]} to={[2.05, 0.1, 0]} />
    <Connector from={[1.2, 0.8, -0.7]} to={[2.05, 0.1, 0]} />
    <Connector from={[1.2, -0.85, -0.7]} to={[2.05, -0.1, 0]} />
    <SceneText text="Conceptual Q · K → causal mask → softmax → V; individual head timing unavailable" position={[0, -1.85, 0]} muted />
    <SceneText text="Numeric Q/K/V and attention weights were not captured" position={[0, -2.25, 0]} muted />
  </group>
}

function FfnStage({ architecture, measuredActive }: { architecture: ModelArchitecture; measuredActive?: boolean }) {
  const gated = architecture.modelType === "qwen3"
  return <group>
    <SceneTile position={[-3.6, 0, 0]} label={gated ? "Gate + up projections" : "First projection"} detail={`${architecture.hiddenSize ?? "?"} → ${architecture.intermediateSize ?? "?"}`} width={3.2} color={measuredActive ? SCENE_COLORS.measured : SCENE_COLORS.accent} />
    <SceneTile position={[0.1, 0, 0]} label={gated ? `${architecture.activation ?? "Activation"} × up` : architecture.activation ?? "Activation"} detail={gated ? "Elementwise gating" : "Configured activation"} width={2.6} />
    <SceneTile position={[3.8, 0, 0]} label="Down projection" detail={`${architecture.intermediateSize ?? "?"} → ${architecture.hiddenSize ?? "?"}`} width={3.1} />
    <Connector from={[-1.95, 0, 0]} to={[-1.25, 0, 0]} /><Connector from={[1.45, 0, 0]} to={[2.2, 0, 0]} />
    <SceneText text={measuredActive ? "Measured FFN scope active · individual projection/vector values not captured" : "Each input token has a feed-forward vector; values are not captured"} position={[0, -2.2, 0]} muted />
  </group>
}

function InputCaption({ position, lines }: { position: Position3; lines: Array<{ text: string; tone?: "strong" | "muted" }> }) {
  return <SceneLabel position={position} anchor="right">
    <span className="grid gap-0.5 leading-snug">
      {lines.map((line) => <span key={line.text} className={line.tone === "muted" ? "scene-tag" : line.tone === "strong" ? "scene-text text-ink" : "scene-text"}>{line.text}</span>)}
    </span>
  </SceneLabel>
}

function ParticipantInputs({ state, requestOrder, requestLabels, selection, activeStepId, trackedRequestId, trackedTokenPosition, onSelect, y, summary }: ModelStageProps & { y: number; summary: string }) {
  const step = activeTeachingStep(state, activeStepId)
  const tokens = useMemo(() => step ? scheduledTokenPositions(step) : [], [step])
  if (!step || state.trust !== "known") return null
  const requestId = selection?.kind === "token" ? selection.requestId : selection?.kind === "request" ? selection.id : trackedRequestId
  const request = step.requests.find((item) => item.requestId === requestId) ?? step.requests[0]
  const position = selection?.kind === "token" ? selection.position : trackedTokenPosition
  if (!request) return null
  return <group>
    <group position={[-3.9, y + 0.16, 0.7]}>
      <TokenBatch tokens={tokens} requestOrder={requestOrder} selectedRequestId={requestId ?? null} selectedPosition={position ?? null} onSelect={onSelect} />
      <SceneText text="Scheduled input positions" position={[0, 0.9, 0]} muted />
    </group>
    <SceneTile position={[-0.65, y, 0.7]} width={2.7} label={requestLabel(requestLabels, request.requestId)}
      color={requestColor(requestOrder, request.requestId)} selected={request.requestId === requestId}
      detail={`${request.promptTokens} prefill / ${request.generationTokens} decode`}
      onSelect={() => onSelect({ kind: "request", id: request.requestId })} />
    <InputCaption position={[0.75, y, 0.8]} lines={[
      { text: summary, tone: "strong" },
      { text: requestId && position != null ? `${requestLabel(requestLabels, requestId)} / p${position}${request.requestId === requestId && position >= request.rangeStart && position < request.rangeStart + request.rangeLength ? ` / ${position < request.numPromptTokens ? "prefill" : "decode"}` : " / outside this batch"}` : `${step.requests.length} requests · ${step.totalScheduledTokens} positions` },
      { text: "Decode lighter · explanatory layout", tone: "muted" },
    ]} />
  </group>
}

function MeasuredParticipantInputs({ profile, requestOrder, requestLabels, selection, trackedRequestId, trackedTokenPosition, onSelect, y, summary }: ModelStageProps & { profile: MeasuredSceneProfile; y: number; summary: string }) {
  const tokens = useMemo(() => observedProfileTokenPositions(profile.packedRows), [profile.packedRows])
  if (!tokens.length) return <SceneText text={summary} position={[0, y, 0.7]} muted />
  const requestId = selection?.kind === "token" ? selection.requestId : selection?.kind === "request" ? selection.id : trackedRequestId
  const position = selection?.kind === "token" ? selection.position : trackedTokenPosition
  const mapped = tokens.filter((token) => token.requestId === requestId)
  const shownRequest = mapped[0]?.requestId ?? tokens[0].requestId
  const shownRows = tokens.filter((token) => token.requestId === shownRequest)
  return <group>
    <group position={[-3.9, y + 0.16, 0.7]}>
      <TokenBatch tokens={tokens} requestOrder={requestOrder} selectedRequestId={requestId ?? null} selectedPosition={position ?? null} onSelect={onSelect} />
      <SceneText text={profile.evidenceKind === "fixture" ? "Fixture packed rows" : "Observed packed rows"} position={[0, 0.9, 0]} muted />
    </group>
    <SceneTile position={[-0.65, y, 0.7]} width={2.7} label={requestLabel(requestLabels, shownRequest)} detail={`${shownRows.length} mapped input positions`}
      color={requestColor(requestOrder, shownRequest)} selected={shownRequest === requestId}
      onSelect={() => onSelect({ kind: "request", id: shownRequest })} />
    <InputCaption position={[0.75, y, 0.8]} lines={[
      { text: summary, tone: "strong" },
      { text: requestId && position != null ? `${requestLabel(requestLabels, requestId)} / p${position}${mapped.some((token) => token.position === position) ? " / mapped" : " / outside this batch"}` : `${tokens.length} real input rows · padding excluded` },
      { text: "Batch/layer timing · no token or head duration", tone: "muted" },
    ]} />
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
