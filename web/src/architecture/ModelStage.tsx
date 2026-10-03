import { useLayoutEffect, useMemo, useRef } from "react"
import * as THREE from "three"

import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import type { Selection } from "../execution/types.ts"
import { requestColor } from "../theme.ts"
import { activeTeachingStep, architectureFocusLabel, architectureLabel, kvHeadForQuery, observedProfileTokenPositions, poolCellPosition, poolGridLayout, projectArchitecture, scheduledTokenPositions, teachingLayer } from "./model.ts"
import type { ArchitectureFocus, MeasuredSceneProfile, ModelArchitecture, ScheduledTokenPosition } from "./model.ts"
import { Connector, SceneEmpty, SceneTile, SCENE_COLORS, sceneClick, TextSprite } from "./primitives.tsx"
import { createPoolRaycast } from "./poolPicking.ts"

export interface ModelStageProps {
  state: ReplayState
  manifest?: RecordingManifest | null
  requestOrder: string[]
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
  return <group>
    {overview ? <TextSprite text={architectureLabel(props.manifest)} position={[0, 4.25, 0]} scale={[6.6, 0.5, 1]} /> : null}
    {focus.kind === "ffn" ? <TextSprite text={architectureFocusLabel(focus)} position={[0, 2.85, 0]} scale={[6.8, 0.46, 1]} /> : null}
    <TextSprite text={profile ? profile.evidenceKind === "fixture" ? "Synthetic batch/layer activity · fixture profiler clock" : profile.evidenceKind === "gpu" ? "Measured batch/layer activity · profiler clock" : "Profiler batch/layer activity · source unverified"
      : props.state.trust === "known" ? "Logical model structure · GPU layer timing not captured" : "Trace gap · structure is known; current token participation is unknown"} position={[0, overview ? 3.7 : 3.42, 0]} scale={[8.5, 0.36, 1]} muted />
    {props.trackedRequestId && overview && !profile ? <TextSprite
      text={`${props.trackedRequestId}${props.trackedTokenPosition == null ? "" : ` / position ${props.trackedTokenPosition}`} / ${step?.requests.some((request) => request.requestId === props.trackedRequestId && (props.trackedTokenPosition == null || props.trackedTokenPosition >= request.rangeStart && props.trackedTokenPosition < request.rangeStart + request.rangeLength)) ? "scheduled in this step" : "not scheduled in this step"}`}
      position={[0, 3.25, 0]} scale={[8.2, 0.3, 1]} muted /> : null}
    {focus.kind === "model" ? <LayerStack {...props} architecture={architecture} />
      : focus.kind === "layer" ? <LayerStage {...props} architecture={architecture} layerIndex={focus.layerIndex} />
        : focus.kind === "ffn" ? <FfnStage architecture={architecture} measuredActive={profile?.activeModules?.some((module) => module.layerIndex === focus.layerIndex && module.module === "ffn")} />
          : focus.kind === "head" ? <HeadStage {...props} architecture={architecture} focus={focus} />
            : <AttentionStage {...props} architecture={architecture} layerIndex={focus.layerIndex} />}
    <TextSprite
      text={profile ? profileTokenCount ? `${profile.stepId ?? "Measured batch"}: ${profileTokenCount} observed packed inputs / padding excluded` : "No observed packed-row mapping for this profiler position"
        : step ? `${step.id}: ${step.totalScheduledTokens} scheduled input positions / ${step.requests.length} request${step.requests.length === 1 ? "" : "s"}` : "No scheduler step has reached this playhead"}
      position={[0, overview ? -5.15 : -4.55, 0]} scale={[8, 0.36, 1]} muted
    />
    {profile ? <MeasuredParticipantInputs {...props} profile={profile} y={overview ? -4.15 : -3.55} />
      : step ? <ParticipantInputs {...props} y={overview ? -4.15 : -3.55} /> : null}
  </group>
}

export function LayerStack({ architecture, compact = false, ...props }: ModelStageProps & { architecture: ModelArchitecture; compact?: boolean }) {
  const count = architecture.layerCount ?? 0
  const mesh = useRef<THREE.InstancedMesh>(null)
  const activeLayer = !props.measuredProfile && props.state.trust === "known" && activeTeachingStep(props.state, props.activeStepId)
    ? teachingLayer(props.teachingProgress, count) : null
  const measuredLayers = useMemo(() => new Set(props.measuredProfile?.activeLayers.filter((layer) => Number.isInteger(layer) && layer >= 0 && layer < count) ?? []), [props.measuredProfile?.activeLayers, count])
  const height = compact ? 2.45 : 5.15
  const width = compact ? 1.52 : 3
  const spacing = height / Math.max(1, count)
  useLayoutEffect(() => {
    const instance = mesh.current
    if (!instance) return
    const transform = new THREE.Object3D()
    const color = new THREE.Color()
    for (let layer = 0; layer < count; layer += 1) {
      transform.position.set(0, height / 2 - (layer + 0.5) * spacing, 0)
      transform.scale.set(width, Math.max(spacing * 0.68, 0.012), compact ? 0.7 : 1.65)
      transform.updateMatrix()
      instance.setMatrixAt(layer, transform.matrix)
      const selected = props.selection?.kind === "layer" && props.selection.layerIndex === layer
      color.set(selected ? SCENE_COLORS.ink : measuredLayers.has(layer) ? "#f0c478" : layer === activeLayer ? SCENE_COLORS.accent : SCENE_COLORS.surface)
      instance.setColorAt(layer, color)
    }
    instance.instanceMatrix.needsUpdate = true
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true
    instance.computeBoundingSphere()
  }, [count, height, spacing, width, compact, activeLayer, measuredLayers, props.selection])
  const labels = [...new Set([0, Math.floor(count / 4), Math.floor(count / 2), Math.floor(count * 3 / 4), count - 1])]
  return <group>
    <instancedMesh ref={mesh} args={[undefined, undefined, count]}
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
      <boxGeometry args={[1, 1, 1]} /><meshStandardMaterial roughness={0.85} />
    </instancedMesh>
    {labels.map((layer) => <TextSprite key={layer} text={`L${layer + 1}`}
      position={[width / 2 + (compact ? 0.35 : 0.55), height / 2 - (layer + 0.5) * spacing, 0.82]}
      scale={[compact ? 0.5 : 0.9, compact ? 0.25 : 0.34, 1]} muted />)}
    <TextSprite text={`${count} layers`} position={[0, -height / 2 - 0.3, 0.7]} scale={[width + 0.5, 0.35, 1]} />
    {!compact ? <>
      <TextSprite text="Token embeddings" position={[0, height / 2 + 0.3, 0]} scale={[3.4, 0.34, 1]} />
      <TextSprite text="Norm → LM head → sampling" position={[0, -height / 2 - 0.8, 0]} scale={[4.4, 0.34, 1]} muted />
    </> : null}
    {activeLayer != null && !compact ? <>
      <Connector from={[-3.4, height / 2 - (activeLayer + 0.5) * spacing, 0.2]} to={[-width / 2 - 0.12, height / 2 - (activeLayer + 0.5) * spacing, 0.2]} />
      <TextSprite text={`Teaching position: layer ${activeLayer + 1}`} position={[-3.05, height / 2 - (activeLayer + 0.5) * spacing + 0.32, 0.2]} scale={[2.7, 0.3, 1]} />
    </> : null}
    {props.measuredProfile && !compact ? <TextSprite text={measuredLayers.size ? `Profiler activity: ${[...measuredLayers].map((layer) => `L${layer + 1}`).join(", ")}` : "No mapped layer active at this profiler instant"}
      position={[-3.4, 0, 1]} scale={[3.2, 0.34, 1]} /> : null}
  </group>
}

function LayerStage({ architecture, layerIndex, ...props }: ModelStageProps & { architecture: ModelArchitecture; layerIndex: number }) {
  const moduleActive = (module: "attention" | "ffn") => props.measuredProfile?.activeModules?.some((entry) => entry.layerIndex === layerIndex && entry.module === module)
  return <group>
    <TextSprite text={`Layer ${layerIndex + 1}`} position={[0, 2.75, 0]} scale={[3.5, 0.46, 1]} />
    <Connector from={[-5, 0, 0]} to={[-2.7, 0, 0]} />
    <Connector from={[-0.85, 0, 0]} to={[1.35, 0, 0]} />
    <Connector from={[3.5, 0, 0]} to={[5.3, 0, 0]} />
    <SceneTile position={[-1.8, 0, 0]} label="Self-attention"
      detail={`${architecture.queryHeads ?? "?"} Q / ${architecture.kvHeads ?? "?"} KV heads`} width={2.8}
      color={moduleActive("attention") ? "#f0c478" : SCENE_COLORS.accent}
      selected={props.selection?.kind === "module" && props.selection.layerIndex === layerIndex && props.selection.module === "attention"}
      onSelect={() => props.onSelect({ kind: "module", layerIndex, module: "attention" })}
      onExplore={() => props.onArchitectureExplore?.({ kind: "attention", layerIndex })} />
    <SceneTile position={[2.45, 0, 0]} label="Feed-forward"
      detail={`${architecture.hiddenSize ?? "?"} → ${architecture.intermediateSize ?? "?"} → ${architecture.hiddenSize ?? "?"}`} width={3.2}
      color={moduleActive("ffn") ? "#f0c478" : SCENE_COLORS.accent}
      selected={props.selection?.kind === "module" && props.selection.layerIndex === layerIndex && props.selection.module === "ffn"}
      onSelect={() => props.onSelect({ kind: "module", layerIndex, module: "ffn" })}
      onExplore={() => props.onArchitectureExplore?.({ kind: "ffn", layerIndex })} />
    <TextSprite text={architecture.modelType === "qwen3" ? "RMSNorm" : "LayerNorm"} position={[-4.3, 0.55, 0]} scale={[1.6, 0.3, 1]} muted />
    <TextSprite text={architecture.modelType === "qwen3" ? "+ residual → RMSNorm" : "+ residual → LayerNorm"} position={[0.35, 0.75, 0]} scale={[2.8, 0.3, 1]} muted />
    <Connector from={[-4.5, 0.2, -0.25]} to={[-4.5, 1.65, -0.25]} muted arrow={false} />
    <Connector from={[-4.5, 1.65, -0.25]} to={[0.2, 1.65, -0.25]} muted arrow={false} />
    <Connector from={[0.2, 1.65, -0.25]} to={[0.2, 0.25, -0.25]} muted />
    <TextSprite text="Residual path" position={[-2.3, 1.9, -0.25]} scale={[2.5, 0.3, 1]} muted />
    <Connector from={[0.2, -0.2, -0.25]} to={[0.2, -1.45, -0.25]} muted arrow={false} />
    <Connector from={[0.2, -1.45, -0.25]} to={[4.85, -1.45, -0.25]} muted arrow={false} />
    <Connector from={[4.85, -1.45, -0.25]} to={[4.85, -0.25, -0.25]} muted />
    <TextSprite text="Feed-forward residual" position={[2.7, -1.75, -0.25]} scale={[3.4, 0.3, 1]} muted />
    <TextSprite text={props.measuredProfile ? "Amber: mapped profiler scope active · activation values not captured" : "Operator structure; no activation values were captured"} position={[0, -2.25, 0]} scale={[8.5, 0.34, 1]} muted />
  </group>
}

function AttentionStage({ architecture, layerIndex, ...props }: ModelStageProps & { architecture: ModelArchitecture; layerIndex: number }) {
  const qHeads = architecture.queryHeads ?? 0
  const kvHeads = architecture.kvHeads ?? 0
  if (!qHeads || !kvHeads) return <SceneEmpty label="Head geometry was not captured" />
  const grouped = qHeads % kvHeads === 0
  const headsPerKv = grouped ? qHeads / kvHeads : 0
  const columns = Math.min(8, grouped ? kvHeads : qHeads)
  const width = Math.min(8.8, columns * 1.1)
  const rows = grouped ? headsPerKv * Math.ceil(kvHeads / columns) : Math.ceil(qHeads / columns)
  const rowSpacing = Math.min(0.65, 2.8 / rows)
  const kvColumns = Math.min(8, kvHeads)
  const kvWidth = Math.min(8.8, kvColumns * 1.1)
  const selectedHead = props.selection?.kind === "head" && props.selection.layerIndex === layerIndex ? props.selection : null
  const selectedKv = selectedHead?.headKind === "kv" ? selectedHead.headIndex
    : selectedHead ? kvHeadForQuery(selectedHead.headIndex, qHeads, kvHeads) : null
  const qPosition = (index: number): [number, number, number] => {
    const kv = grouped ? Math.floor(index / headsPerKv) : index % columns
    const row = grouped ? (index % headsPerKv) + Math.floor(kv / columns) * headsPerKv : Math.floor(index / columns)
    return [((kv % columns) + 0.5) * width / columns - width / 2, 2.1 - row * rowSpacing, 0.6]
  }
  const kvPosition = (index: number): [number, number, number] => [
    ((index % kvColumns) + 0.5) * kvWidth / kvColumns - kvWidth / 2,
    1.45 - rows * rowSpacing - Math.floor(index / kvColumns) * 0.8, -0.25,
  ]
  return <group>
    <TextSprite text={`${qHeads} query heads share ${kvHeads} KV heads`} position={[0, 2.95, 0]} scale={[7, 0.4, 1]} />
    {grouped ? Array.from({ length: kvHeads }, (_, kv) => {
      const to = kvPosition(kv)
      return <group key={`sharing-${kv}`}>
        {Array.from({ length: headsPerKv }, (_, offset) => {
          const from = qPosition(kv * headsPerKv + offset)
          return <Connector key={offset} from={[from[0] + 0.3, from[1], -0.8]} to={[to[0] + 0.3, to[1] + 0.25, -0.8]} muted={selectedKv !== kv} arrow={false} />
        })}
        <Connector from={[to[0] + 0.3, to[1] + 0.25, -0.8]} to={[to[0], to[1], -0.25]} muted={selectedKv !== kv} />
      </group>
    }) : null}
    {Array.from({ length: qHeads }, (_, index) => {
      const selected = props.selection?.kind === "head" && props.selection.headKind === "query" && props.selection.headIndex === index && props.selection.layerIndex === layerIndex
      return <HeadCell key={`q-${index}`} label={`Q${index + 1}`} position={qPosition(index)} selected={selected}
        related={selectedKv != null && kvHeadForQuery(index, qHeads, kvHeads) === selectedKv} height={Math.min(0.46, rowSpacing * 0.8)}
        onSelect={() => props.onSelect({ kind: "head", layerIndex, headIndex: index, headKind: "query" })}
        onExplore={() => props.onArchitectureExplore?.({ kind: "head", layerIndex, headIndex: index, headKind: "query" })} />
    })}
    {Array.from({ length: kvHeads }, (_, index) => {
      const [x, y] = kvPosition(index)
      const selected = props.selection?.kind === "head" && props.selection.headKind === "kv" && props.selection.headIndex === index && props.selection.layerIndex === layerIndex
      return <group key={`kv-${index}`}>
        <HeadCell label={`K${index + 1}`} position={[x - 0.2, y, -0.25]} selected={selected} related={selectedKv === index} small
          onSelect={() => props.onSelect({ kind: "head", layerIndex, headIndex: index, headKind: "kv" })}
          onExplore={() => props.onArchitectureExplore?.({ kind: "head", layerIndex, headIndex: index, headKind: "kv" })} />
        <HeadCell label={`V${index + 1}`} position={[x + 0.2, y, -0.25]} selected={selected} related={selectedKv === index} small
          onSelect={() => props.onSelect({ kind: "head", layerIndex, headIndex: index, headKind: "kv" })}
          onExplore={() => props.onArchitectureExplore?.({ kind: "head", layerIndex, headIndex: index, headKind: "kv" })} />
      </group>
    })}
    <TextSprite text={`Head dimension ${architecture.headDim ?? "not recorded"}; edges show model KV sharing, not tensor values`} position={[0, -2.15, 0]} scale={[8.9, 0.34, 1]} muted />
  </group>
}

function HeadCell({ label, position, selected, related = false, height = 0.46, small = false, onSelect, onExplore }: {
  label: string; position: [number, number, number]; selected: boolean; related?: boolean; height?: number; small?: boolean; onSelect: () => void; onExplore: () => void
}) {
  return <group position={position}>
    <mesh onClick={(event) => { if (sceneClick(event)) { event.stopPropagation(); onSelect() } }}
      onDoubleClick={(event) => { if (sceneClick(event)) { event.stopPropagation(); onSelect(); onExplore() } }}>
      <boxGeometry args={[small ? 0.36 : 0.85, height, 0.65]} />
      <meshStandardMaterial color={selected ? "#435f7d" : related ? "#334358" : SCENE_COLORS.surface} roughness={0.8} />
    </mesh>
    <TextSprite text={label} position={[0, 0, 0.4]} scale={[small ? 0.35 : 0.74, Math.min(0.28, height * 0.65), 1]} />
  </group>
}

function HeadStage({ architecture, focus }: ModelStageProps & { architecture: ModelArchitecture; focus: Extract<ArchitectureFocus, { kind: "head" }> }) {
  const kv = focus.headKind === "kv" ? focus.headIndex : kvHeadForQuery(focus.headIndex, architecture.queryHeads ?? 0, architecture.kvHeads ?? 0)
  const queriesPerKv = architecture.queryHeads && architecture.kvHeads && architecture.queryHeads % architecture.kvHeads === 0 ? architecture.queryHeads / architecture.kvHeads : null
  const queryLabel = focus.headKind === "query" ? `Q${focus.headIndex + 1}` : queriesPerKv
    ? `Q${focus.headIndex * queriesPerKv + 1}–${(focus.headIndex + 1) * queriesPerKv}` : "Shared query group"
  return <group>
    <TextSprite text={`Layer ${focus.layerIndex + 1} / ${focus.headKind === "query" ? "query" : "KV"} head ${focus.headIndex + 1}`} position={[0, 2.8, 0]} scale={[6, 0.44, 1]} />
    {architecture.modelType === "qwen3" ? <TextSprite text="Q/K RMSNorm → RoPE → grouped-query attention" position={[0, 2.18, 0]} scale={[7.2, 0.32, 1]} muted /> : null}
    <SceneTile position={[-3.3, 0.8, 0.2]} label={queryLabel} detail={`Dimension ${architecture.headDim ?? "unknown"}`} width={2.6} />
    <SceneTile position={[-0.1, 0.8, -0.7]} label={`K${kv == null ? "?" : kv + 1}`} detail="Computed prefix keys" width={2.6} />
    <SceneTile position={[-0.1, -0.85, -0.7]} label={`V${kv == null ? "?" : kv + 1}`} detail="Computed prefix values" width={2.6} />
    <SceneTile position={[3.5, 0, 0]} label="Attention output" detail="Causal attention → output projection" width={3} />
    <Connector from={[-2, 0.8, 0.2]} to={[1.8, 0.1, 0]} />
    <Connector from={[1.1, 0.8, -0.7]} to={[1.8, 0.1, 0]} />
    <Connector from={[1.1, -0.85, -0.7]} to={[1.8, -0.1, 0]} />
    <TextSprite text="Conceptual Q · K → causal mask → softmax → V; individual head timing unavailable" position={[0, -1.8, 0]} scale={[9, 0.36, 1]} muted />
    <TextSprite text="Numeric Q/K/V and attention weights were not captured" position={[0, -2.2, 0]} scale={[8, 0.34, 1]} muted />
  </group>
}

function FfnStage({ architecture, measuredActive }: { architecture: ModelArchitecture; measuredActive?: boolean }) {
  const gated = architecture.modelType === "qwen3"
  return <group>
    <SceneTile position={[-3.45, 0, 0]} label={gated ? "Gate + up projections" : "First projection"} detail={`${architecture.hiddenSize ?? "?"} → ${architecture.intermediateSize ?? "?"}`} width={3.2} color={measuredActive ? "#f0c478" : SCENE_COLORS.accent} />
    <SceneTile position={[0.1, 0, 0]} label={gated ? `${architecture.activation ?? "Activation"} × up` : architecture.activation ?? "Activation"} detail={gated ? "Elementwise gating" : "Configured activation"} width={2.5} />
    <SceneTile position={[3.6, 0, 0]} label="Down projection" detail={`${architecture.intermediateSize ?? "?"} → ${architecture.hiddenSize ?? "?"}`} width={3.1} />
    <Connector from={[-1.75, 0, 0]} to={[-1.2, 0, 0]} /><Connector from={[1.35, 0, 0]} to={[2, 0, 0]} />
    <TextSprite text={measuredActive ? "Measured FFN scope active · individual projection/vector values not captured" : "Each input token has a feed-forward vector; values are not captured"} position={[0, -2.2, 0]} scale={[9, 0.38, 1]} muted />
  </group>
}

function ParticipantInputs({ state, requestOrder, selection, activeStepId, trackedRequestId, trackedTokenPosition, onSelect, y }: ModelStageProps & { y: number }) {
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
      <TextSprite text="Scheduled input positions" position={[0, 0.83, 0]} scale={[2.8, 0.28, 1]} />
    </group>
    <SceneTile position={[-0.65, y, 0.7]} width={2.7} label={request.requestId}
      color={requestColor(requestOrder, request.requestId)} selected={request.requestId === requestId}
      detail={`${request.promptTokens} prefill / ${request.generationTokens} decode`}
      onSelect={() => onSelect({ kind: "request", id: request.requestId })} />
    <TextSprite text={requestId && position != null ? `${requestId} / p${position}${request.requestId === requestId && position >= request.rangeStart && position < request.rangeStart + request.rangeLength ? ` / ${position < request.numPromptTokens ? "prefill" : "decode"}` : " / outside this batch"}` : `${step.requests.length} requests · ${step.totalScheduledTokens} positions`}
      position={[3.15, y + 0.18, 0.8]} scale={[3.8, 0.3, 1]} />
    <TextSprite text="Decode brighter · explanatory layout" position={[3.15, y - 0.24, 0.8]} scale={[4.1, 0.28, 1]} muted />
  </group>
}

function MeasuredParticipantInputs({ profile, requestOrder, selection, trackedRequestId, trackedTokenPosition, onSelect, y }: ModelStageProps & { profile: MeasuredSceneProfile; y: number }) {
  const tokens = useMemo(() => observedProfileTokenPositions(profile.packedRows), [profile.packedRows])
  if (!tokens.length) return null
  const requestId = selection?.kind === "token" ? selection.requestId : selection?.kind === "request" ? selection.id : trackedRequestId
  const position = selection?.kind === "token" ? selection.position : trackedTokenPosition
  const mapped = tokens.filter((token) => token.requestId === requestId)
  const shownRequest = mapped[0]?.requestId ?? tokens[0].requestId
  const shownRows = tokens.filter((token) => token.requestId === shownRequest)
  return <group>
    <group position={[-3.9, y + 0.16, 0.7]}>
      <TokenBatch tokens={tokens} requestOrder={requestOrder} selectedRequestId={requestId ?? null} selectedPosition={position ?? null} onSelect={onSelect} />
      <TextSprite text={profile.evidenceKind === "fixture" ? "Fixture packed rows" : "Observed packed rows"} position={[0, 0.83, 0]} scale={[2.8, 0.28, 1]} />
    </group>
    <SceneTile position={[-0.65, y, 0.7]} width={2.7} label={shownRequest} detail={`${shownRows.length} mapped input positions`}
      color={requestColor(requestOrder, shownRequest)} selected={shownRequest === requestId}
      onSelect={() => onSelect({ kind: "request", id: shownRequest })} />
    <TextSprite text={requestId && position != null ? `${requestId} / p${position}${mapped.some((token) => token.position === position) ? " / mapped" : " / outside this batch"}` : `${tokens.length} real input rows · padding excluded`}
      position={[3.15, y + 0.18, 0.8]} scale={[3.8, 0.3, 1]} />
    <TextSprite text="Batch/layer timing · no token or head duration" position={[3.15, y - 0.24, 0.8]} scale={[4.1, 0.28, 1]} muted />
  </group>
}

function TokenBatch({ tokens, requestOrder, selectedRequestId, selectedPosition, onSelect }: {
  tokens: ScheduledTokenPosition[]; requestOrder: string[];
  selectedRequestId: string | null; selectedPosition: number | null; onSelect: (selection: Selection) => void
}) {
  const layout = useMemo(() => poolGridLayout(tokens.length, 1.4), [tokens.length])
  const raycast = useMemo(() => createPoolRaycast(layout), [layout])
  const mesh = useRef<THREE.InstancedMesh>(null)
  useLayoutEffect(() => {
    const instance = mesh.current
    if (!instance) return
    const transform = new THREE.Object3D()
    const color = new THREE.Color()
    const decodeTint = new THREE.Color(SCENE_COLORS.ink)
    tokens.forEach((token, index) => {
      transform.position.set(...poolCellPosition(token.row ?? index, layout)); transform.scale.setScalar(layout.spacing * 0.74); transform.updateMatrix()
      instance.setMatrixAt(index, transform.matrix)
      color.set(requestColor(requestOrder, token.requestId))
      if (token.kind === "decode") color.lerp(decodeTint, 0.46)
      if (token.requestId === selectedRequestId && token.position === selectedPosition) color.set(SCENE_COLORS.ink)
      instance.setColorAt(index, color)
    })
    instance.instanceMatrix.needsUpdate = true
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true
    instance.computeBoundingSphere()
  }, [tokens, layout, requestOrder, selectedRequestId, selectedPosition])
  return <instancedMesh ref={mesh} args={[undefined, undefined, tokens.length]} raycast={raycast}
    onClick={(event) => {
      if (!sceneClick(event) || event.instanceId == null) return
      const token = tokens[event.instanceId]
      if (!token) return
      event.stopPropagation(); onSelect({ kind: "token", requestId: token.requestId, position: token.position })
    }}>
    <boxGeometry args={[1, 1, 1]} /><meshStandardMaterial roughness={0.85} />
  </instancedMesh>
}
