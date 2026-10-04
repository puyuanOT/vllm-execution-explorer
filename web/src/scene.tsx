import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { useEffect, useLayoutEffect, useMemo, useRef, type ReactNode } from "react"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import * as THREE from "three"

import { projectArchitecture, projectKvCache, projectScheduler, type KvBlockProjection, type KvCacheProjection } from "../../replay/src/learning.ts"
import type { QueueEntry, RecordingManifest, ReplayState } from "../../replay/src/reduce.ts"
import { CacheStage } from "./architecture/CacheStage.tsx"
import { activeTeachingStep, architectureLabel, dieCellPosition, dieLayout, type ArchitectureFocus, type MeasuredSceneProfile } from "./architecture/model.ts"
import { LayerStack, ModelStage } from "./architecture/ModelStage.tsx"
import { SceneProfileDiagnostics } from "./architecture/SceneProfileDiagnostics.tsx"
import { BoxEdges, Connector, FloorGrid, Polyline, SceneCard, SceneEmpty, SceneText, SceneTile, SCENE_COLORS, sceneClick, useHoverCursor, type Position3 } from "./architecture/primitives.tsx"
import { componentSnapshot, requestLabel, WORKFLOW_COMPONENTS, type WorkflowComponentId } from "./explore/model.ts"
import type { Selection } from "./execution/types.ts"
import { requestColor } from "./theme.ts"

export type CameraAction = "orbit-left" | "orbit-right" | "zoom-in" | "zoom-out" | "pan-left" | "pan-right" | "pan-up" | "pan-down" | "reset"
export interface CameraCommand { id: number; action: CameraAction }

export interface SceneProps {
  state: ReplayState
  manifest?: RecordingManifest | null
  kvCache?: KvCacheProjection
  requestOrder: string[]
  requestLabels?: ReadonlyMap<string, string>
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

const FLOOR = -1.2
const X = { intake: -9.0, scheduler: -4.6, tower: 1.0, output: 6.2 }
const KV_BANK = { x: -1.95, z: 4.1, width: 4.1, depth: 2.2 }

export function ExecutionScene(props: SceneProps) {
  const focusKey = `${props.focus ?? "overview"}/${JSON.stringify(props.architectureFocus ?? { kind: "model" })}`
  return <Canvas camera={{ position: [0, 6, 18], fov: 34 }} frameloop="demand" dpr={[1, 1.75]}
    aria-label={props.focus ? `${componentLabel(props.focus)} focused 3D view` : "vLLM serving workflow and model architecture 3D overview"}>
    <color attach="background" args={[SCENE_COLORS.background]} />
    <fogExp2 attach="fog" args={[SCENE_COLORS.background, props.focus === "kv-cache" ? 0.03 : 0.022]} />
    <hemisphereLight args={["#c3d3e8", "#05080e", 0.75]} />
    <directionalLight position={[6, 12, 9]} intensity={0.75} color="#dfe9f5" />
    <CameraRig focusKey={focusKey} focus={props.focus ?? null} architectureFocus={props.architectureFocus ?? null}
      poolCount={props.kvCache?.pools.length ?? 1} command={props.cameraCommand ?? null} />
    <Invalidate {...props} />
    {import.meta.env.VITE_EXPLORER_SCENE_PROFILE === "1" ? <SceneProfileDiagnostics sceneProps={props} focusKey={focusKey} /> : null}
    {props.focus ? <FocusedStage {...props} focus={props.focus} /> : <WorkflowOverview {...props} />}
  </Canvas>
}

interface StageFrame { target: Position3; direction: Position3; width: number; height: number }

function stageFrame(focus: WorkflowComponentId | null, architectureFocus: ArchitectureFocus | null, poolCount: number): StageFrame {
  if (!focus) return { target: [-1.3, 0.35, 1.3], direction: [0.16, 0.44, 1], width: 19.5, height: 9.5 }
  if (focus === "model-execution") {
    return architectureFocus && architectureFocus.kind !== "model"
      ? { target: [0.4, -0.55, 0.3], direction: [0.08, 0.2, 1], width: 13.5, height: 11.8 }
      : { target: [0, -0.75, 0.4], direction: [0.12, 0.24, 1], width: 12.5, height: 12.6 }
  }
  if (focus === "kv-cache") return { target: [0, -0.6, 0.8], direction: [0, 0.85, 1], width: Math.max(14, poolCount * 12.8), height: 12.5 }
  if (focus === "scheduler") return { target: [0, 0.1, 0], direction: [0.04, 0.16, 1], width: 16, height: 11.4 }
  return { target: [0, 0.6, 0], direction: [0.04, 0.14, 1], width: 13, height: 8 }
}

function CameraRig({ focusKey, focus, architectureFocus, poolCount, command }: {
  focusKey: string; focus: WorkflowComponentId | null; architectureFocus: ArchitectureFocus | null;
  poolCount: number; command: CameraCommand | null
}) {
  const camera = useThree((item) => item.camera)
  const gl = useThree((item) => item.gl)
  const size = useThree((item) => item.size)
  const invalidate = useThree((item) => item.invalidate)
  const controls = useRef<OrbitControls | null>(null)
  const lastCommand = useRef<number | null>(null)
  useLayoutEffect(() => {
    const next = new OrbitControls(camera, gl.domElement)
    const onChange = () => invalidate()
    next.enableDamping = false
    next.enablePan = true
    next.screenSpacePanning = true
    next.minDistance = 0.5
    next.maxDistance = 90
    next.minPolarAngle = Math.PI * 0.04
    next.maxPolarAngle = Math.PI * 0.92
    next.addEventListener("change", onChange)
    controls.current = next
    return () => { next.removeEventListener("change", onChange); next.dispose(); controls.current = null }
  }, [camera, gl, invalidate])
  useLayoutEffect(() => {
    resetCamera(camera, controls.current, stageFrame(focus, architectureFocus, poolCount), size.width / Math.max(1, size.height))
    invalidate()
  }, [camera, focusKey, poolCount, size.width, size.height, invalidate])
  useEffect(() => {
    const active = controls.current
    if (!active || !command || command.id === lastCommand.current) return
    lastCommand.current = command.id
    if (command.action === "reset") {
      resetCamera(camera, active, stageFrame(focus, architectureFocus, poolCount), size.width / Math.max(1, size.height))
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

function resetCamera(camera: THREE.Camera, controls: OrbitControls | null, frame: StageFrame, aspect: number) {
  const verticalFov = camera instanceof THREE.PerspectiveCamera ? camera.fov : 34
  const distance = Math.max(frame.height, frame.width / Math.max(0.45, aspect)) / (2 * Math.tan(THREE.MathUtils.degToRad(verticalFov) / 2))
  const target = new THREE.Vector3(...frame.target)
  const offset = new THREE.Vector3(...frame.direction).normalize().multiplyScalar(distance)
  camera.position.copy(target).add(offset)
  camera.lookAt(target)
  controls?.target.copy(target); controls?.update()
}

function Invalidate(props: SceneProps) {
  const invalidate = useThree((item) => item.invalidate)
  useEffect(() => { invalidate() }, [props.state, props.selection, props.focus, props.architectureFocus,
    props.teachingProgress, props.manifest, props.kvCache, props.trackedRequestId, props.trackedTokenPosition, props.measuredProfile, props.animate, invalidate])
  return null
}

function Module({ position, size, selected, onSelect, onExplore, children }: {
  position: Position3; size: Position3; selected: boolean; onSelect: () => void; onExplore: () => void; children?: ReactNode
}) {
  const hover = useHoverCursor()
  return <group position={position}>
    <mesh {...hover.bind}
      onClick={(event) => { if (!sceneClick(event)) return; event.stopPropagation(); onSelect() }}
      onDoubleClick={(event) => { if (!sceneClick(event)) return; event.stopPropagation(); onSelect(); onExplore() }}>
      <boxGeometry args={size} />
      <meshStandardMaterial color={SCENE_COLORS.surface} roughness={0.75} metalness={0.1} />
    </mesh>
    <BoxEdges size={size} color={selected ? SCENE_COLORS.accent : hover.hovered ? SCENE_COLORS.edgeMarker : SCENE_COLORS.edge} />
    {children}
  </group>
}

function Cube({ position, color, size = 0.22, hollow = false }: { position: Position3; color: string; size?: number; hollow?: boolean }) {
  if (hollow) return <BoxEdges size={[size, size, size]} color={color} position={position} />
  return <mesh position={position} raycast={() => null}>
    <boxGeometry args={[size, size, size]} />
    <meshBasicMaterial color={new THREE.Color(color).multiplyScalar(0.75)} toneMapped={false} />
  </mesh>
}

function Packets({ from, to, colors, animate, size = 0.14 }: { from: Position3; to: Position3; colors: string[]; animate: boolean; size?: number }) {
  const group = useRef<THREE.Group>(null)
  const phase = useRef(0)
  const invalidate = useThree((state) => state.invalidate)
  const start = useMemo(() => new THREE.Vector3(...from), [from[0], from[1], from[2]])
  const end = useMemo(() => new THREE.Vector3(...to), [to[0], to[1], to[2]])
  const place = (offset: number) => {
    const children = group.current?.children ?? []
    children.forEach((child, index) => {
      const t = ((index + 0.5) / Math.max(1, children.length) * 0.82 + 0.09 + offset) % 1
      child.position.lerpVectors(start, end, t)
    })
  }
  useLayoutEffect(() => { place(animate ? phase.current : 0) })
  useEffect(() => { if (animate) invalidate() }, [animate, invalidate])
  useFrame((_, delta) => {
    if (!animate) return
    phase.current = (phase.current + delta * 0.12) % 1
    place(phase.current)
    invalidate()
  })
  return <group ref={group}>
    {colors.map((color, index) => <mesh key={`${color}-${index}`} raycast={() => null}>
      <boxGeometry args={[size, size, size]} />
      <meshBasicMaterial color={new THREE.Color(color).multiplyScalar(0.85)} toneMapped={false} />
    </mesh>)}
  </group>
}

function WorkflowOverview(props: SceneProps) {
  const architecture = projectArchitecture(props.manifest)
  const cache = props.kvCache ?? projectKvCache(props.state, props.manifest)
  const color = (id: string) => requestColor(props.requestOrder, id)
  const selectedComponent = props.selection?.kind === "component" ? props.selection.id : null
  const select = (id: WorkflowComponentId) => () => props.onSelect({ kind: "component", id })
  const explore = (id: WorkflowComponentId) => () => props.onExplore?.(id)
  const observed = props.requestOrder.filter((id) => props.state.requests[id]).concat(Object.keys(props.state.requests).filter((id) => !props.requestOrder.includes(id)))
  const latest = props.state.steps.at(-1) ?? null
  const running = latest?.queueAfter?.running ?? []
  const waiting = latest?.queueAfter?.waiting ?? []
  const resultStep = [...props.state.steps].reverse().find((step) => step.resultSequence != null) ?? null
  const emitted = resultStep ? observed.filter((id) => props.state.requests[id]?.acceptedSpans.some((span) => span.stepId === resultStep.id)) : []
  const accepted = observed.map((id) => props.state.requests[id]?.acceptedTokenCount ?? 0)
  const acceptedScale = Math.max(1, ...accepted)
  const snapshot = (id: WorkflowComponentId) => componentSnapshot(props.state, id)
  const capacity = cache.pools.reduce((count, pool) => count + pool.capacity, 0)
  const referenced = cache.pools.reduce((count, pool) => count + pool.blocks.filter((block) => !block.isNull && block.refCount > 0).length, 0)
  const towerTop = architecture.supported ? FLOOR + 0.35 + 36 * 0.104 + 0.28 : FLOOR + 1.3
  const flowY = FLOOR + 0.5
  const flows: Array<[Position3, Position3]> = [
    [[X.intake + 1.15, flowY, 0], [X.scheduler - 1.25, flowY, 0]],
    [[X.scheduler + 1.25, flowY, 0], [X.tower - 1.62, flowY, 0]],
    [[X.tower + 1.62, flowY, 0], [X.output - 1.15, flowY, 0]],
  ]
  const bankLeft = KV_BANK.x - KV_BANK.width / 2 - 0.25
  const bankRight = KV_BANK.x + KV_BANK.width / 2 + 0.25
  const caption = props.measuredProfile ? "Profiler clock · serving and KV nodes follow host replay"
    : props.state.trust === "unknown" ? "Trace gap · counts describe only the trusted prefix"
      : architecture.supported ? "Serving observations connected to captured model structure"
        : "Model architecture is unknown until a supported recording is captured"
  return <group>
    <FloorGrid y={FLOOR} />

    <Module position={[X.intake, FLOOR + 0.525, 0]} size={[2.3, 1.05, 1.5]} selected={selectedComponent === "request-intake"} onSelect={select("request-intake")} onExplore={explore("request-intake")}>
      {observed.slice(0, 7).map((id, index) => <Cube key={id} position={[-0.92 + index * 0.31, 0.64, 0.35]} color={color(id)} size={0.2} />)}
    </Module>
    <SceneCard position={[X.intake, FLOOR + 2.15, 0]} code="INT" name="Intake" value={String(observed.length).padStart(2, "0")} detail={observed.length > 7 ? `requests observed · first 7 shown` : "requests observed"} active={selectedComponent === "request-intake"} />

    <Module position={[X.scheduler, FLOOR + 0.65, 0]} size={[2.5, 1.3, 1.7]} selected={selectedComponent === "scheduler"} onSelect={select("scheduler")} onExplore={explore("scheduler")}>
      {running.slice(0, 4).map((entry, index) => <Cube key={`r-${entry.requestId}`} position={[-0.95 + index * 0.34, 0.79, 0.25]} color={color(entry.requestId)} size={0.24} />)}
      {waiting.slice(0, 3).map((entry, index) => <Cube key={`w-${entry.requestId}`} position={[0.4 + index * 0.34, 0.79, 0.25]} color={color(entry.requestId)} size={0.24} hollow />)}
    </Module>
    <SceneCard position={[X.scheduler, FLOOR + 1.7, 0]} code="SCH" name="Scheduler" value={latest?.id ?? "—"}
      detail={latest ? `${latest.totalScheduledTokens}/${latest.tokenBudget} tok${latest.queueAfter ? ` · ${running.length} run · ${waiting.length} wait` : ""}` : "no step yet"} active={selectedComponent === "scheduler"} />

    {architecture.supported ? <group position={[X.tower, 0, 0]}>
      <LayerStack {...props} architecture={architecture} compact baseY={FLOOR + 0.35} />
      <SceneCard position={[0, towerTop + 0.15, 0]} code="MDL" name={architectureLabel(props.manifest)} value={String(architecture.layerCount)}
        detail={`layers · ${architecture.queryHeads} Q / ${architecture.kvHeads} KV heads`} active={selectedComponent === "model-execution"} />
    </group> : <>
      <Module position={[X.tower, FLOOR + 0.65, 0]} size={[2.9, 1.3, 2]} selected={selectedComponent === "model-execution"} onSelect={select("model-execution")} onExplore={explore("model-execution")} />
      <SceneCard position={[X.tower, FLOOR + 1.55, 0]} code="MDL" name="Model execution" value={snapshot("model-execution").metric} detail={snapshot("model-execution").metricLabel} active={selectedComponent === "model-execution"} />
    </>}

    <Module position={[X.output, FLOOR + 0.525, 0]} size={[2.3, 1.05, 1.5]} selected={selectedComponent === "output"} onSelect={select("output")} onExplore={explore("output")}>
      {observed.slice(0, 7).map((id, index) => {
        const height = Math.max(0.04, ((props.state.requests[id]?.acceptedTokenCount ?? 0) / acceptedScale) * 1.1)
        return <mesh key={id} position={[-0.92 + index * 0.31, 0.53 + height / 2 + 0.02, 0.3]} raycast={() => null}>
          <boxGeometry args={[0.2, height, 0.2]} />
          <meshBasicMaterial color={new THREE.Color(color(id)).multiplyScalar(height > 0.05 ? 0.75 : 0.2)} toneMapped={false} />
        </mesh>
      })}
    </Module>
    <SceneCard position={[X.output, FLOOR + 2.3, 0.3]} code="OUT" name="Output" value={snapshot("output").metric} detail="tokens accepted" active={selectedComponent === "output"} />

    <KvBank cache={cache} requestOrder={props.requestOrder} selected={selectedComponent === "kv-cache"} onSelect={select("kv-cache")} onExplore={explore("kv-cache")} />
    <SceneCard position={[bankLeft - 0.05, FLOOR + 0.1, KV_BANK.z - 0.2]} anchor="left" code="KV" name="KV-cache manager"
      value={cache.pools.length ? `${referenced.toLocaleString()}/${capacity.toLocaleString()}` : "—"}
      detail={!cache.pools.length ? "pool inventory unavailable" : cache.status === "unknown" ? "blocks · state unknown after gap" : "blocks referenced"} active={selectedComponent === "kv-cache"} />

    {flows.map(([from, to], index) => <Polyline key={index} points={[from, to]} color={SCENE_COLORS.line} />)}
    <Packets from={flows[1][0]} to={flows[1][1]} colors={(latest?.requests ?? []).slice(0, 6).map((request) => color(request.requestId))} animate={props.animate} />
    <Packets from={flows[2][0]} to={flows[2][1]} colors={emitted.slice(0, 6).map(color)} animate={props.animate} size={0.12} />
    <Polyline points={[[X.scheduler + 0.5, FLOOR + 0.03, 0.86], [X.scheduler + 0.5, FLOOR + 0.03, KV_BANK.z], [bankLeft, FLOOR + 0.03, KV_BANK.z]]} color={SCENE_COLORS.lineDim} dashed />
    <Polyline points={[[bankRight, FLOOR + 0.03, KV_BANK.z], [X.tower, FLOOR + 0.03, KV_BANK.z], [X.tower, FLOOR + 0.03, 1.18]]} color={SCENE_COLORS.lineDim} dashed />
    <Polyline points={[[X.tower, towerTop, -0.4], [X.tower, towerTop + 0.5, -0.4], [X.scheduler, towerTop + 0.5, -0.4], [X.scheduler, FLOOR + 1.45, -0.4]]} color="#24344c" dashed dash={0.2} gap={0.14} />
    <SceneText text="Next scheduler cycle" position={[X.scheduler + 1.6, towerTop + 0.5, -0.4]} muted />
    <SceneText text="Block table" position={[X.scheduler + 0.5, FLOOR + 0.02, 2.4]} muted />
    <SceneText text="KV reads" position={[X.tower, FLOOR + 0.02, 2.6]} muted />
    <SceneText text={caption} position={[-1.3, FLOOR + 0.02, 6.1]} muted />
  </group>
}

function KvBank({ cache, requestOrder, selected, onSelect, onExplore }: {
  cache: KvCacheProjection; requestOrder: string[]; selected: boolean; onSelect: () => void; onExplore: () => void
}) {
  const blocks = useMemo(() => cache.pools.flatMap((pool) => pool.blocks), [cache.pools])
  const layout = useMemo(() => dieLayout(blocks.length, 3.9, KV_BANK.width / KV_BANK.depth), [blocks.length])
  const mesh = useRef<THREE.InstancedMesh>(null)
  const hover = useHoverCursor()
  useLayoutEffect(() => {
    const instance = mesh.current
    if (!instance) return
    const transform = new THREE.Object3D()
    const tint = new THREE.Color()
    blocks.forEach((block, index) => {
      const [x, z] = dieCellPosition(index, layout)
      const referenced = !block.isNull && block.refCount > 0
      const height = referenced ? Math.min(0.34, layout.pitch * 0.4) : Math.min(0.08, layout.pitch * 0.12)
      transform.position.set(x, height / 2, z)
      transform.scale.set(layout.pitch * 0.8, height, layout.pitch * 0.8)
      transform.updateMatrix()
      instance.setMatrixAt(index, transform.matrix)
      instance.setColorAt(index, tint.set(bankColor(block, requestOrder)))
    })
    instance.count = blocks.length
    instance.instanceMatrix.needsUpdate = true
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true
    instance.computeBoundingSphere()
  }, [blocks, layout, requestOrder])
  const ring: Position3[] = [
    [-KV_BANK.width / 2 - 0.25, 0.02, -KV_BANK.depth / 2 - 0.25], [KV_BANK.width / 2 + 0.25, 0.02, -KV_BANK.depth / 2 - 0.25],
    [KV_BANK.width / 2 + 0.25, 0.02, KV_BANK.depth / 2 + 0.25], [-KV_BANK.width / 2 - 0.25, 0.02, KV_BANK.depth / 2 + 0.25],
    [-KV_BANK.width / 2 - 0.25, 0.02, -KV_BANK.depth / 2 - 0.25],
  ]
  return <group position={[KV_BANK.x, FLOOR, KV_BANK.z]}>
    <mesh position={[0, 0.07, 0]} {...hover.bind}
      onClick={(event) => { if (!sceneClick(event)) return; event.stopPropagation(); onSelect() }}
      onDoubleClick={(event) => { if (!sceneClick(event)) return; event.stopPropagation(); onSelect(); onExplore() }}>
      <boxGeometry args={[KV_BANK.width, 0.14, KV_BANK.depth]} />
      <meshStandardMaterial color="#0f1828" roughness={0.8} metalness={0.1} />
    </mesh>
    <BoxEdges size={[KV_BANK.width, 0.14, KV_BANK.depth]} position={[0, 0.07, 0]} color={selected ? SCENE_COLORS.accent : hover.hovered ? SCENE_COLORS.edgeMarker : SCENE_COLORS.edge} />
    {selected ? <Polyline points={ring} color={SCENE_COLORS.accent} /> : null}
    {blocks.length ? <group position={[0, 0.14, 0]}>
      <instancedMesh ref={mesh} args={[undefined, undefined, Math.max(1, blocks.length)]} raycast={() => null}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
    </group> : <SceneText text="Pool inventory unavailable" position={[0, 0.3, 0]} muted />}
  </group>
}

function bankColor(block: KvBlockProjection, requestOrder: string[]): string {
  if (block.isNull) return SCENE_COLORS.reserved
  if (block.trust === "unknown") return SCENE_COLORS.unknown
  if (block.refCount > 0) {
    const holders = [...new Set(block.bindings.map((binding) => binding.requestId))]
    const tint = new THREE.Color(holders.length === 1 ? requestColor(requestOrder, holders[0]) : SCENE_COLORS.accent).multiplyScalar(0.5)
    return `#${tint.getHexString()}`
  }
  if (block.hasHash) return "#3a2e18"
  return SCENE_COLORS.free
}

function FocusedStage(props: SceneProps & { focus: WorkflowComponentId }) {
  if (props.focus === "model-execution") return <ModelStage {...props} />
  if (props.focus === "kv-cache") return <CacheStage cache={props.kvCache ?? projectKvCache(props.state, props.manifest)}
    requestOrder={props.requestOrder} requestLabels={props.requestLabels} selection={props.selection} trackedRequestId={props.trackedRequestId} trackedTokenPosition={props.trackedTokenPosition} onSelect={props.onSelect} />
  if (props.focus === "scheduler") return <SchedulerStage {...props} />
  const requests = orderedRequests(props.state, props.requestOrder)
  return <group>
    <FloorGrid y={-2.6} />
    {!requests.length ? <SceneEmpty label="No request observation at this playhead" detail="Play or step forward until a request arrives." /> : <>
      <SceneText text={componentLabel(props.focus)} position={[0, 3.7, 0]} strong />
      {requests.slice(0, 24).map((request, index) => <SceneTile key={request.id} position={gridPosition(index, 4)}
        label={requestLabel(props.requestLabels, request.id)} width={2.4} color={requestColor(props.requestOrder, request.id)}
        selected={props.selection?.kind === "request" && props.selection.id === request.id || props.trackedRequestId === request.id}
        detail={props.focus === "output" ? `${request.acceptedTokenCount} accepted · ${request.finishReason ?? request.status}` : request.status}
        onSelect={() => props.onSelect({ kind: "request", id: request.id })} />)}
      {requests.length > 24 ? <SceneText text={`+${requests.length - 24} more requests · use Find request`} position={[0, -3.5, 0]} muted /> : null}
      <SceneText text={props.focus === "output" ? "Only accepted outputs reached by this playhead are shown" : "Planned arrivals do not imply simultaneous scheduler admission"} position={[0, -3.95, 0]} muted />
    </>}
  </group>
}

const QUEUE_ROWS = 6

function SchedulerStage(props: SceneProps) {
  const step = activeTeachingStep(props.state, props.activeStepId)
  const queues = projectScheduler(props.state, step?.id)
  if (!step) return <group><FloorGrid y={-4.4} /><SceneEmpty label="No scheduler step at this playhead" detail="Play or step forward to the first scheduler step." /></group>
  const participants = step.requests.slice(0, 6)
  const spread = (index: number, count: number) => (index - (count - 1) / 2) * 2.7
  return <group>
    <FloorGrid y={-4.4} />
    <SceneText text={`${step.id} · ${step.totalScheduledTokens} of ${step.tokenBudget} tokens scheduled`} position={[0, 4.1, 0]} strong />
    <SceneText text={queues.policy ? `Recorded scheduling policy: ${queues.policy}` : "Scheduling policy was not captured"} position={[0, 3.6, 0]} muted />
    <SceneText text="Before schedule" position={[-5, 2.95, 0]} strong />
    <SceneText text="After schedule" position={[5, 2.95, 0]} strong />
    {queues.status === "observed" ? <>
      <QueueColumn waiting={queues.before?.waiting ?? []} running={queues.before?.running ?? []} x={-5} {...props} />
      <QueueColumn waiting={queues.after?.waiting ?? []} running={queues.after?.running ?? []} x={5} {...props} />
    </> : <SceneText text={queues.reason ?? "Queues unavailable"} position={[0, 2.35, 0]} muted />}
    <Connector from={[-3.45, 1.0, 0]} to={[-1.55, 1.0, 0]} /><Connector from={[1.55, 1.0, 0]} to={[3.45, 1.0, 0]} />
    <SceneTile position={[0, 1.0, 0]} label="Scheduled batch" detail={`${step.requests.length} actual participant${step.requests.length === 1 ? "" : "s"}`} width={2.9}
      selected={props.selection?.kind === "step" && props.selection.id === step.id}
      onSelect={() => props.onSelect({ kind: "step", id: step.id })} />
    {participants.map((request, index) => {
      const x = spread(index, participants.length)
      return <group key={request.requestId}>
        <Polyline points={[[0, 0.48, 0], [0, -1.6, 0], [x, -1.6, 0], [x, -2.38, 0]]} color={SCENE_COLORS.lineDim} />
        <SceneTile position={[x, -2.95, 0.4]} width={2.45} height={0.9}
          label={requestLabel(props.requestLabels, request.requestId)} color={requestColor(props.requestOrder, request.requestId)}
          detail={`${request.promptTokens} prefill + ${request.generationTokens} decode`}
          selected={props.trackedRequestId === request.requestId || props.selection?.kind === "request" && props.selection.id === request.requestId}
          onSelect={() => props.onSelect({ kind: "request", id: request.requestId })} />
      </group>
    })}
    {step.requests.length > participants.length ? <SceneText text={`+${step.requests.length - participants.length} more participants in the inspector`} position={[0, -3.75, 0.4]} muted /> : null}
    <SceneText text="These are captured choices; unrecorded decision reasons remain unknown" position={[0, -4.25, 0.4]} muted />
  </group>
}

function QueueColumn({ waiting, running, x, onSelect, requestOrder, requestLabels, selection }: {
  waiting: QueueEntry[]; running: QueueEntry[]; x: number; onSelect: (selection: Selection) => void; requestOrder: string[];
  requestLabels?: ReadonlyMap<string, string>; selection: Selection | null
}) {
  const entries = [...running.map((entry) => ({ entry, label: "running" })), ...waiting.map((entry) => ({ entry, label: "waiting" }))]
  const shown = entries.slice(0, QUEUE_ROWS)
  return <group>
    <SceneText text={`Running ${running.length} · waiting ${waiting.length}`} position={[x, 2.5, 0]} muted />
    {shown.map(({ entry, label }, index) => <SceneTile key={`${label}-${entry.requestId}`}
      position={[x, 1.85 - index * 0.78, -0.4]} width={2.7} height={0.62} depth={0.3}
      label={requestLabel(requestLabels, entry.requestId)} detail={`${label} · cursor ${entry.computedTokens ?? "?"}`}
      color={requestColor(requestOrder, entry.requestId)}
      selected={selection?.kind === "request" && selection.id === entry.requestId}
      onSelect={() => onSelect({ kind: "request", id: entry.requestId })} />)}
    {entries.length > shown.length ? <SceneText text={`+${entries.length - shown.length} more`} position={[x, 1.85 - QUEUE_ROWS * 0.78 + 0.2, -0.4]} muted /> : null}
  </group>
}

function orderedRequests(state: ReplayState, requestOrder: string[]) {
  const order = [...new Set([...requestOrder, ...Object.keys(state.requests)])]
  return order.map((id) => state.requests[id]).filter((request) => request != null)
}
function gridPosition(index: number, columns: number): [number, number, number] {
  return [(index % columns - (columns - 1) / 2) * 2.75, 2.5 - Math.floor(index / columns) * 1.3, 0]
}
function componentLabel(id: WorkflowComponentId): string { return WORKFLOW_COMPONENTS.find((item) => item.id === id)?.label ?? id }
