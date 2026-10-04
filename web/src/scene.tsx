import { Canvas, useThree } from "@react-three/fiber"
import { useEffect, useLayoutEffect, useMemo, useRef } from "react"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import * as THREE from "three"

import { projectArchitecture, projectKvCache, projectScheduler, type KvCacheProjection } from "../../replay/src/learning.ts"
import type { QueueEntry, RecordingManifest, ReplayState } from "../../replay/src/reduce.ts"
import { CacheStage } from "./architecture/CacheStage.tsx"
import { activeTeachingStep, architectureLabel, type ArchitectureFocus, type MeasuredSceneProfile } from "./architecture/model.ts"
import { LayerStack, ModelStage } from "./architecture/ModelStage.tsx"
import { SceneProfileDiagnostics } from "./architecture/SceneProfileDiagnostics.tsx"
import { Connector, SceneEmpty, SceneTile, SCENE_COLORS, TextSprite } from "./architecture/primitives.tsx"
import { componentSnapshot, WORKFLOW_COMPONENTS, type WorkflowComponentId } from "./explore/model.ts"
import type { Selection } from "./execution/types.ts"
import { requestColor } from "./theme.ts"

export type CameraAction = "orbit-left" | "orbit-right" | "zoom-in" | "zoom-out" | "pan-left" | "pan-right" | "pan-up" | "pan-down" | "reset"
export interface CameraCommand { id: number; action: CameraAction }

export interface SceneProps {
  state: ReplayState
  manifest?: RecordingManifest | null
  kvCache?: KvCacheProjection
  requestOrder: string[]
  selection: Selection | null
  animate: boolean
  focus?: WorkflowComponentId | null
  architectureFocus?: ArchitectureFocus | null
  activeStepId?: string | null
  teachingProgress?: number
  trackedRequestId?: string | null
  trackedTokenPosition?: number | null
  measuredProfile?: MeasuredSceneProfile
  cameraCommand?: CameraCommand | null
  onSelect: (selection: Selection) => void
  onExplore?: (component: WorkflowComponentId) => void
  onArchitectureExplore?: (focus: ArchitectureFocus) => void
}

const OVERVIEW_POSITIONS: Record<WorkflowComponentId, [number, number, number]> = {
  "request-intake": [-5.8, 0.35, 0], scheduler: [-2.8, 0.35, 0],
  "kv-cache": [-2.7, -2.55, -0.7], "model-execution": [0.5, 0.55, 0], output: [4.05, 0.35, 0],
}

export function ExecutionScene(props: SceneProps) {
  const focusKey = `${props.focus ?? "overview"}/${JSON.stringify(props.architectureFocus ?? { kind: "model" })}`
  return <Canvas camera={{ position: [0, 2.5, 17], fov: 38 }} frameloop="demand" dpr={[1, 1.5]}
    aria-label={props.focus ? `${componentLabel(props.focus)} focused 3D view` : "vLLM serving workflow and model architecture 3D overview"}>
    <color attach="background" args={[SCENE_COLORS.background]} />
    <ambientLight intensity={0.85} /><directionalLight position={[3, 7, 8]} intensity={1.45} />
    <CameraRig focusKey={focusKey} focus={props.focus ?? null} architectureFocus={props.architectureFocus ?? null}
      selectedBlockKey={props.selection?.kind === "block" ? props.selection.key : null} poolCount={props.kvCache?.pools.length ?? 1} command={props.cameraCommand ?? null} />
    <Invalidate {...props} />
    {import.meta.env.VITE_EXPLORER_SCENE_PROFILE === "1" ? <SceneProfileDiagnostics sceneProps={props} focusKey={focusKey} /> : null}
    {props.focus ? <FocusedStage {...props} focus={props.focus} /> : <WorkflowOverview {...props} />}
  </Canvas>
}

function CameraRig({ focusKey, focus, architectureFocus, selectedBlockKey, poolCount, command }: {
  focusKey: string; focus: WorkflowComponentId | null; architectureFocus: ArchitectureFocus | null;
  selectedBlockKey: string | null; poolCount: number; command: CameraCommand | null
}) {
  const camera = useThree((item) => item.camera)
  const gl = useThree((item) => item.gl)
  const size = useThree((item) => item.size)
  const invalidate = useThree((item) => item.invalidate)
  const controls = useRef<OrbitControls | null>(null)
  const lastCommand = useRef<number | null>(null)
  const lastBlock = useRef<string | null>(null)
  useLayoutEffect(() => {
    const next = new OrbitControls(camera, gl.domElement)
    const onChange = () => invalidate()
    next.enableDamping = false
    next.enablePan = true
    next.screenSpacePanning = true
    next.minDistance = 0.5
    next.maxDistance = 80
    next.minPolarAngle = Math.PI * 0.08
    next.maxPolarAngle = Math.PI * 0.92
    next.addEventListener("change", onChange)
    controls.current = next
    return () => { next.removeEventListener("change", onChange); next.dispose(); controls.current = null }
  }, [camera, gl, invalidate])
  useLayoutEffect(() => {
    resetCamera(camera, controls.current, focus, architectureFocus, size.width / Math.max(1, size.height), poolCount)
    invalidate()
  }, [camera, focusKey, poolCount, size.width, size.height, invalidate])
  useLayoutEffect(() => {
    if (focus !== "kv-cache" || selectedBlockKey == null || selectedBlockKey === lastBlock.current) return
    lastBlock.current = selectedBlockKey
    resetCamera(camera, controls.current, focus, architectureFocus, size.width / Math.max(1, size.height), poolCount)
    invalidate()
  }, [camera, focus, selectedBlockKey, architectureFocus, poolCount, size.width, size.height, invalidate])
  useEffect(() => {
    const active = controls.current
    if (!active || !command || command.id === lastCommand.current) return
    lastCommand.current = command.id
    if (command.action === "reset") {
      resetCamera(camera, active, focus, architectureFocus, size.width / Math.max(1, size.height), poolCount)
    } else if (command.action.startsWith("pan-")) {
      camera.updateMatrixWorld()
      const horizontal = command.action === "pan-left" || command.action === "pan-right"
      const direction = command.action === "pan-left" || command.action === "pan-down" ? -1 : 1
      const shift = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, horizontal ? 0 : 1)
        .multiplyScalar(direction * camera.position.distanceTo(active.target) * 0.06)
      camera.position.add(shift); active.target.add(shift); active.update()
    } else if (command.action === "orbit-left" || command.action === "orbit-right") {
      const offset = camera.position.clone().sub(active.target)
      offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), (command.action === "orbit-left" ? 1 : -1) * Math.PI / 12)
      camera.position.copy(active.target).add(offset); active.update()
    } else {
      const offset = camera.position.clone().sub(active.target)
      const distance = THREE.MathUtils.clamp(offset.length() * (command.action === "zoom-in" ? 0.78 : 1.25), active.minDistance, active.maxDistance)
      camera.position.copy(active.target).add(offset.setLength(distance)); active.update()
    }
    invalidate()
  }, [camera, command, focus, architectureFocus, poolCount, size.width, size.height, invalidate])
  return null
}

function resetCamera(camera: THREE.Camera, controls: OrbitControls | null, focus: WorkflowComponentId | null,
  architectureFocus: ArchitectureFocus | null, aspect: number, poolCount: number) {
  const poolView = focus === "kv-cache"
  const detailedModel = focus === "model-execution" && architectureFocus != null && architectureFocus.kind !== "model"
  const height = detailedModel ? 8.7 : focus === "model-execution" ? 10.5 : poolView ? 10.4 : focus === "scheduler" ? 10.2 : 8.5
  const width = poolView ? Math.max(14.5, poolCount * 8.5 + 1.6) : focus === "scheduler" ? 13.5 : detailedModel ? 11.6 : focus === "model-execution" ? 11.8 : 15.1
  const centerY = poolView ? 0.15 : -0.45
  const verticalFov = camera instanceof THREE.PerspectiveCamera ? camera.fov : 38
  const distance = Math.max(height, width / Math.max(0.45, aspect)) / (2 * Math.tan(THREE.MathUtils.degToRad(verticalFov) / 2))
  camera.position.set(poolView ? distance * 0.18 : distance * 0.035, centerY + (poolView ? distance * 0.13 : distance * 0.025), distance)
  camera.lookAt(0, centerY, 0)
  controls?.target.set(0, centerY, 0); controls?.update()
}

function Invalidate(props: SceneProps) {
  const invalidate = useThree((item) => item.invalidate)
  useEffect(() => { invalidate() }, [props.state, props.selection, props.focus, props.architectureFocus,
    props.teachingProgress, props.manifest, props.kvCache, props.trackedRequestId, props.trackedTokenPosition, props.measuredProfile, invalidate])
  return null
}

function WorkflowOverview(props: SceneProps) {
  const architecture = projectArchitecture(props.manifest)
  const cache = props.kvCache ?? projectKvCache(props.state, props.manifest)
  const cacheSummary = useMemo(() => ({
    capacity: cache.pools.reduce((count, pool) => count + pool.capacity, 0),
    referenced: cache.pools.reduce((count, pool) => count + pool.blocks.filter((block) => !block.isNull && block.refCount > 0).length, 0),
  }), [cache])
  const snapshot = componentSnapshot(props.state, "model-execution")
  return <group>
    <Connector from={[-4.7, 0.35, 0]} to={[-4, 0.35, 0]} />
    <Connector from={[-1.65, 0.35, 0]} to={[-0.5, 0.35, 0]} />
    <Connector from={[1.55, 0.35, 0]} to={[2.9, 0.35, 0]} />
    <Connector from={[-2.8, -0.2, 0]} to={[-2.7, -2.02, -0.7]} />
    <Connector from={[-1.6, -2.55, -0.7]} to={[0.15, -0.7, -0.45]} />
    <Connector from={[0.5, 2.3, -1]} to={[-2.8, 2.3, -1]} muted />
    <Connector from={[-2.8, 2.3, -1]} to={[-2.8, 0.95, -0.45]} muted />
    <TextSprite text="Next scheduler cycle" position={[-1.1, 2.67, -0.8]} scale={[3.5, 0.3, 1]} muted />
    {WORKFLOW_COMPONENTS.filter((component) => component.id !== "model-execution").map((component) => {
      const current = componentSnapshot(props.state, component.id)
      const detail = component.id === "kv-cache" ? !cache.pools.length ? "Pool inventory unavailable" : cache.status === "unknown" ? `${cacheSummary.capacity.toLocaleString()} blocks / current state unknown`
        : `${cacheSummary.capacity.toLocaleString()} blocks / ${cacheSummary.referenced.toLocaleString()} referenced`
        : `${current.metric} ${current.metricLabel}`
      return <SceneTile key={component.id} position={OVERVIEW_POSITIONS[component.id]} label={component.shortLabel}
        detail={detail} width={2.25}
        selected={props.selection?.kind === "component" && props.selection.id === component.id}
        color={current.unavailable ? SCENE_COLORS.line : SCENE_COLORS.accent}
        onSelect={() => props.onSelect({ kind: "component", id: component.id })}
        onExplore={() => props.onExplore?.(component.id)} />
    })}
    {architecture.supported ? <group position={OVERVIEW_POSITIONS["model-execution"]}>
      <LayerStack {...props} architecture={architecture} compact />
      <TextSprite text={architectureLabel(props.manifest)} position={[0, 1.65, 0.35]} scale={[3.2, 0.35, 1]} />
      <TextSprite text={`${architecture.queryHeads} Q / ${architecture.kvHeads} KV heads`} position={[0, -1.9, 0.6]} scale={[3.2, 0.28, 1]} muted />
    </group> : <SceneTile position={OVERVIEW_POSITIONS["model-execution"]} label="Model execution"
      detail={`${snapshot.metric} ${snapshot.metricLabel}`} width={2.3}
      selected={props.selection?.kind === "component" && props.selection.id === "model-execution"}
      onSelect={() => props.onSelect({ kind: "component", id: "model-execution" })}
      onExplore={() => props.onExplore?.("model-execution")} />}
    <TextSprite text={props.measuredProfile ? "Profiler clock · serving and KV nodes follow host replay" : architecture.supported ? "Serving observations connected to captured model structure" : "Model architecture is unknown until a supported recording is captured"}
      position={[0, -4.05, 0]} scale={[10, 0.36, 1]} muted />
  </group>
}

function FocusedStage(props: SceneProps & { focus: WorkflowComponentId }) {
  if (props.focus === "model-execution") return <ModelStage {...props} />
  if (props.focus === "kv-cache") return <CacheStage cache={props.kvCache ?? projectKvCache(props.state, props.manifest)}
    requestOrder={props.requestOrder} selection={props.selection} trackedRequestId={props.trackedRequestId} trackedTokenPosition={props.trackedTokenPosition} onSelect={props.onSelect} />
  if (props.focus === "scheduler") return <SchedulerStage {...props} />
  const requests = orderedRequests(props.state, props.requestOrder)
  if (!requests.length) return <SceneEmpty label="No request observation at this playhead" />
  return <group>
    <TextSprite text={componentLabel(props.focus)} position={[0, 4, 0]} scale={[6, 0.5, 1]} />
    {requests.map((request, index) => <SceneTile key={request.id} position={gridPosition(index, 4)}
      label={request.id} width={2.3} color={requestColor(props.requestOrder, request.id)}
      selected={props.selection?.kind === "request" && props.selection.id === request.id || props.trackedRequestId === request.id}
      detail={props.focus === "output" ? `${request.acceptedTokenCount} accepted / ${request.finishReason ?? request.status}` : request.status}
      onSelect={() => props.onSelect({ kind: "request", id: request.id })} />)}
    <TextSprite text={props.focus === "output" ? "Only accepted outputs reached by this playhead are shown" : "Planned arrivals do not imply simultaneous scheduler admission"}
      position={[0, -4, 0]} scale={[8.5, 0.34, 1]} muted />
  </group>
}

function SchedulerStage(props: SceneProps) {
  const step = activeTeachingStep(props.state, props.activeStepId)
  const queues = projectScheduler(props.state, step?.id)
  if (!step) return <SceneEmpty label="No scheduler step at this playhead" />
  return <group>
    <TextSprite text={`${step.id} / ${step.totalScheduledTokens} of ${step.tokenBudget} tokens scheduled`} position={[0, 4.35, 0]} scale={[8, 0.46, 1]} />
    <TextSprite text={queues.policy ? `Recorded scheduling policy: ${queues.policy}` : "Scheduling policy was not captured"}
      position={[0, 3.8, 0]} scale={[7, 0.33, 1]} muted />
    <TextSprite text="Before schedule" position={[-3.7, 2.75, 0]} scale={[3.2, 0.34, 1]} />
    <TextSprite text="After schedule" position={[3.7, 2.75, 0]} scale={[3.2, 0.34, 1]} />
    {queues.status === "observed" ? <>
      <QueueColumns waiting={queues.before?.waiting ?? []} running={queues.before?.running ?? []} x={-3.7} {...props} />
      <QueueColumns waiting={queues.after?.waiting ?? []} running={queues.after?.running ?? []} x={3.7} {...props} />
    </> : <TextSprite text={queues.reason ?? "Queues unavailable"} position={[0, 2.2, 0]} scale={[8.5, 0.34, 1]} muted />}
    <Connector from={[-2.6, 0, 0]} to={[-1.3, 0, 0]} /><Connector from={[1.3, 0, 0]} to={[2.6, 0, 0]} />
    <SceneTile position={[0, 0, 0]} label="Scheduled batch" detail={`${step.requests.length} actual participants`} width={2.55}
      selected={props.selection?.kind === "step" && props.selection.id === step.id}
      onSelect={() => props.onSelect({ kind: "step", id: step.id })} />
    {step.requests.map((request, index) => <SceneTile key={request.requestId}
      position={[(index - (step.requests.length - 1) / 2) * 2.8, -2.4, 0.7]} width={2.55}
      label={request.requestId} color={requestColor(props.requestOrder, request.requestId)}
      detail={`${request.promptTokens} prefill + ${request.generationTokens} decode`}
      selected={props.trackedRequestId === request.requestId || props.selection?.kind === "request" && props.selection.id === request.requestId}
      onSelect={() => props.onSelect({ kind: "request", id: request.requestId })} />)}
    <TextSprite text="These are captured choices; unrecorded decision reasons remain unknown" position={[0, -3.65, 0]} scale={[9, 0.34, 1]} muted />
  </group>
}

function QueueColumns({ waiting, running, x, onSelect, requestOrder }: {
  waiting: QueueEntry[]; running: QueueEntry[]; x: number; onSelect: (selection: Selection) => void; requestOrder: string[]
}) {
  return <group>
    <TextSprite text={`Waiting ${waiting.length} / running ${running.length}`} position={[x, 2.2, 0]} scale={[3.2, 0.3, 1]} muted />
    {[...waiting.map((entry) => ({ entry, label: "waiting" })), ...running.map((entry) => ({ entry, label: "running" }))].map(({ entry, label }, index) => <SceneTile key={`${label}-${entry.requestId}`}
      position={[x, 1.45 - index * 1.12, -0.55]} label={entry.requestId} detail={`${label}; cursor ${entry.computedTokens ?? "?"}`} width={2.8}
      color={requestColor(requestOrder, entry.requestId)} onSelect={() => onSelect({ kind: "request", id: entry.requestId })} />)}
  </group>
}

function orderedRequests(state: ReplayState, requestOrder: string[]) {
  const order = [...new Set([...requestOrder, ...Object.keys(state.requests)])]
  return order.map((id) => state.requests[id]).filter((request) => request != null)
}
function gridPosition(index: number, columns: number): [number, number, number] {
  return [(index % columns - (columns - 1) / 2) * 2.75, 2.4 - Math.floor(index / columns) * 1.4, 0]
}
function componentLabel(id: WorkflowComponentId): string { return WORKFLOW_COMPONENTS.find((item) => item.id === id)?.label ?? id }
