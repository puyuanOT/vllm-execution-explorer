import { Canvas, useThree } from "@react-three/fiber"
import { useEffect, useLayoutEffect, useMemo, useRef } from "react"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import * as THREE from "three"

import type { BlockState, ReplayState, StepState } from "../../replay/src/reduce.ts"
import { blockRequestIds } from "../../replay/src/timeline.ts"
import { componentSnapshot, WORKFLOW_COMPONENTS } from "./explore/model.ts"
import type { WorkflowComponentId } from "./explore/model.ts"
import type { Selection } from "./execution/types.ts"
import { requestColor } from "./theme.ts"

export type CameraAction = "orbit-left" | "orbit-right" | "zoom-in" | "zoom-out" | "reset"

export interface CameraCommand {
  id: number
  action: CameraAction
}

interface SceneProps {
  state: ReplayState
  requestOrder: string[]
  selection: Selection | null
  animate: boolean
  focus?: WorkflowComponentId | null
  cameraCommand?: CameraCommand | null
  onSelect: (selection: Selection) => void
  onExplore?: (component: WorkflowComponentId) => void
}

const OVERVIEW_POSITIONS: Record<WorkflowComponentId, [number, number, number]> = {
  "request-intake": [-5.2, 0.85, 0],
  scheduler: [-2.55, 0.85, 0],
  "kv-cache": [0, -1.65, 0],
  "model-execution": [0.2, 0.85, 0],
  output: [3.2, 0.85, 0],
}

export function ExecutionScene(props: SceneProps) {
  return (
    <Canvas
      camera={{ position: [0, 0.25, 9.5], fov: 38 }}
      frameloop="demand"
      aria-label={props.focus ? `${componentLabel(props.focus)} focused 3D view` : "vLLM execution workflow 3D overview"}
    >
      <color attach="background" args={["#06080c"]} />
      <ambientLight intensity={0.95} />
      <directionalLight position={[2, 5, 7]} intensity={1.25} />
      <CameraRig focus={props.focus ?? null} command={props.cameraCommand ?? null} />
      <Invalidate
        state={props.state}
        selection={props.selection}
        focus={props.focus ?? null}
        animate={props.animate}
      />
      {props.focus ? <FocusedStage {...props} focus={props.focus} /> : <WorkflowOverview {...props} />}
    </Canvas>
  )
}

function CameraRig({ focus, command }: { focus: WorkflowComponentId | null; command: CameraCommand | null }) {
  const camera = useThree((item) => item.camera)
  const gl = useThree((item) => item.gl)
  const invalidate = useThree((item) => item.invalidate)
  const controls = useRef<OrbitControls | null>(null)

  useEffect(() => {
    const next = new OrbitControls(camera, gl.domElement)
    const onChange = () => invalidate()
    next.enableDamping = false
    next.enablePan = false
    next.minDistance = 5
    next.maxDistance = 22
    next.minPolarAngle = Math.PI * 0.18
    next.maxPolarAngle = Math.PI * 0.82
    next.target.set(0, focus ? 0 : -0.1, 0)
    next.addEventListener("change", onChange)
    controls.current = next
    next.update()
    return () => {
      next.removeEventListener("change", onChange)
      next.dispose()
      controls.current = null
    }
  }, [camera, gl, invalidate, focus])

  useLayoutEffect(() => {
    resetCamera(camera, controls.current, focus)
    invalidate()
  }, [camera, focus, invalidate])

  useEffect(() => {
    const active = controls.current
    if (!active || !command) return
    if (command.action === "reset") {
      resetCamera(camera, active, focus)
    } else if (command.action === "orbit-left" || command.action === "orbit-right") {
      const direction = command.action === "orbit-left" ? 1 : -1
      const offset = camera.position.clone().sub(active.target)
      offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), direction * Math.PI / 12)
      camera.position.copy(active.target).add(offset)
      camera.lookAt(active.target)
      active.update()
    } else {
      const offset = camera.position.clone().sub(active.target)
      const multiplier = command.action === "zoom-in" ? 0.84 : 1.18
      const nextDistance = THREE.MathUtils.clamp(offset.length() * multiplier, active.minDistance, active.maxDistance)
      offset.setLength(nextDistance)
      camera.position.copy(active.target).add(offset)
      camera.lookAt(active.target)
      active.update()
    }
    invalidate()
  }, [camera, command, focus, invalidate])
  return null
}

function resetCamera(camera: THREE.Camera, controls: OrbitControls | null, focus: WorkflowComponentId | null) {
  camera.position.set(0, focus ? 0.25 : 0.35, focus ? 10.5 : 9.5)
  camera.lookAt(0, focus ? 0 : -0.1, 0)
  if (!controls) return
  controls.target.set(0, focus ? 0 : -0.1, 0)
  controls.update()
}

function Invalidate({
  state,
  selection,
  focus,
  animate,
}: {
  state: ReplayState
  selection: Selection | null
  focus: WorkflowComponentId | null
  animate: boolean
}) {
  const invalidate = useThree((item) => item.invalidate)
  useEffect(() => {
    invalidate()
  }, [state, selection, focus, animate, invalidate])
  return null
}

function WorkflowOverview({ state, selection, onSelect, onExplore }: SceneProps) {
  return (
    <group position={[0.7, 0.15, 0]}>
      <Connector from={OVERVIEW_POSITIONS["request-intake"]} to={OVERVIEW_POSITIONS.scheduler} />
      <Connector from={OVERVIEW_POSITIONS.scheduler} to={OVERVIEW_POSITIONS["model-execution"]} />
      <Connector from={OVERVIEW_POSITIONS["model-execution"]} to={OVERVIEW_POSITIONS.output} />
      <Connector from={OVERVIEW_POSITIONS.scheduler} to={OVERVIEW_POSITIONS["kv-cache"]} />
      <Connector from={OVERVIEW_POSITIONS["kv-cache"]} to={OVERVIEW_POSITIONS["model-execution"]} />
      <Connector from={[-0.05, 2.45, -0.15]} to={[-2.4, 2.45, -0.15]} muted />
      <Connector from={[0.2, 1.62, -0.15]} to={[-0.05, 2.45, -0.15]} muted />
      <Connector from={[-2.4, 2.45, -0.15]} to={[-2.55, 1.62, -0.15]} muted />
      <TextSprite text="next scheduler cycle" position={[-1.2, 2.72, 0]} scale={[2.5, 0.5, 1]} muted />
      {WORKFLOW_COMPONENTS.map((component) => {
        const snapshot = componentSnapshot(state, component.id)
        return (
          <WorkflowNode
            key={component.id}
            id={component.id}
            label={component.shortLabel}
            metric={`${snapshot.metric} ${snapshot.metricLabel}`}
            position={OVERVIEW_POSITIONS[component.id]}
            selected={selection?.kind === "component" && selection.id === component.id}
            unavailable={snapshot.unavailable}
            onSelect={() => onSelect({ kind: "component", id: component.id })}
            onExplore={() => onExplore?.(component.id)}
          />
        )
      })}
    </group>
  )
}

function WorkflowNode({
  id,
  label,
  metric,
  position,
  selected,
  unavailable,
  onSelect,
  onExplore,
}: {
  id: WorkflowComponentId
  label: string
  metric: string
  position: [number, number, number]
  selected: boolean
  unavailable: boolean
  onSelect: () => void
  onExplore: () => void
}) {
  const select = (event: { stopPropagation: () => void }) => {
    event.stopPropagation()
    onSelect()
  }
  const explore = (event: { stopPropagation: () => void }) => {
    event.stopPropagation()
    onSelect()
    onExplore()
  }
  return (
    <group position={position} name={id}>
      {selected ? (
        <mesh position={[0, 0, -0.12]}>
          <boxGeometry args={[2.28, 1.58, 0.16]} />
          <meshStandardMaterial color="#77b7ff" />
        </mesh>
      ) : null}
      <mesh onClick={select} onDoubleClick={explore}>
        <boxGeometry args={[2.12, 1.42, 0.52]} />
        <meshStandardMaterial color={unavailable ? "#171d27" : "#222b39"} />
      </mesh>
      <mesh position={[-0.86, 0, 0.31]} onClick={select} onDoubleClick={explore}>
        <boxGeometry args={[0.08, 1.08, 0.08]} />
        <meshStandardMaterial color={unavailable ? "#58677d" : "#77b7ff"} />
      </mesh>
      <TextSprite text={label} position={[0.1, 0.22, 0.34]} scale={[1.7, 0.46, 1]} />
      <TextSprite text={metric} position={[0.1, -0.28, 0.34]} scale={[1.72, 0.35, 1]} muted />
    </group>
  )
}

function Connector({
  from,
  to,
  muted = false,
}: {
  from: [number, number, number]
  to: [number, number, number]
  muted?: boolean
}) {
  const start = new THREE.Vector3(...from)
  const end = new THREE.Vector3(...to)
  const midpoint = start.clone().add(end).multiplyScalar(0.5)
  const difference = end.clone().sub(start)
  const length = difference.length()
  const angle = Math.atan2(difference.y, difference.x)
  return (
    <mesh position={midpoint} rotation={[0, 0, angle]}>
      <boxGeometry args={[length, 0.035, 0.035]} />
      <meshStandardMaterial color={muted ? "#3e4a5d" : "#58677d"} />
    </mesh>
  )
}

function FocusedStage({
  state,
  requestOrder,
  selection,
  focus,
  onSelect,
}: SceneProps & { focus: WorkflowComponentId }) {
  const requests = orderedRequests(state, requestOrder)
  const title = componentLabel(focus)
  return (
    <group position={[0, 0.2, 0]}>
      <TextSprite text={title} position={[0, 3.55, 0]} scale={[4.8, 0.7, 1]} />
      {focus === "request-intake" ? (
        <RequestGrid requests={requests} requestOrder={requestOrder} selection={selection} onSelect={onSelect} />
      ) : null}
      {focus === "scheduler" ? (
        <StepGrid steps={state.steps} selection={selection} onSelect={onSelect} title="recorded scheduler steps" />
      ) : null}
      {focus === "kv-cache" ? (
        <CacheGrid blocks={Object.values(state.blocks)} state={state} requestOrder={requestOrder} selection={selection} onSelect={onSelect} />
      ) : null}
      {focus === "model-execution" ? (
        <StepGrid steps={state.steps.filter((step) => step.submittedSequence != null)} selection={selection} onSelect={onSelect} title="host-observed submissions" />
      ) : null}
      {focus === "output" ? (
        <OutputGrid requests={requests} requestOrder={requestOrder} selection={selection} onSelect={onSelect} />
      ) : null}
    </group>
  )
}

function RequestGrid({
  requests,
  requestOrder,
  selection,
  onSelect,
}: {
  requests: ReplayState["requests"][string][]
  requestOrder: string[]
  selection: Selection | null
  onSelect: (selection: Selection) => void
}) {
  if (requests.length === 0) return <SceneEmpty label="No observed requests at this playhead" />
  return (
    <group>
      {requests.slice(0, 30).map((request, index) => (
        <DataTile
          key={request.id}
          position={gridPosition(index, 5)}
          label={request.id}
          detail={request.status}
          color={requestColor(requestOrder, request.id)}
          selected={selection?.kind === "request" && selection.id === request.id}
          onSelect={() => onSelect({ kind: "request", id: request.id })}
        />
      ))}
      {requests.length > 30 ? <TextSprite text={`+ ${requests.length - 30} more requests`} position={[0, -3.25, 0]} scale={[3, 0.5, 1]} muted /> : null}
    </group>
  )
}

function StepGrid({
  steps,
  selection,
  onSelect,
  title,
}: {
  steps: StepState[]
  selection: Selection | null
  onSelect: (selection: Selection) => void
  title: string
}) {
  if (steps.length === 0) return <SceneEmpty label={`No ${title} at this playhead`} />
  const visible = steps.slice(-24)
  return (
    <group>
      {visible.map((step, index) => (
        <DataTile
          key={`${step.id}-${step.sequence}`}
          position={gridPosition(index, 4)}
          label={step.id}
          detail={`${step.totalScheduledTokens} tokens · ${step.status.replaceAll("_", " ")}`}
          color="#77b7ff"
          selected={selection?.kind === "step" && selection.id === step.id}
          onSelect={() => onSelect({ kind: "step", id: step.id })}
        />
      ))}
      {steps.length > visible.length ? <TextSprite text={`Showing latest ${visible.length} of ${steps.length} steps`} position={[0, -3.3, 0]} scale={[3.8, 0.5, 1]} muted /> : null}
    </group>
  )
}

function CacheGrid({
  blocks,
  state,
  requestOrder,
  selection,
  onSelect,
}: {
  blocks: BlockState[]
  state: ReplayState
  requestOrder: string[]
  selection: Selection | null
  onSelect: (selection: Selection) => void
}) {
  if (blocks.length === 0) return <SceneEmpty label="No cache block transition at this playhead" />
  const sorted = [...blocks].sort((left, right) => left.pool.localeCompare(right.pool) || left.blockId - right.blockId)
  const visible = sorted.slice(0, 80)
  return (
    <group position={[-4.8, 2.35, 0]}>
      {visible.map((block, index) => {
        const column = index % 12
        const row = Math.floor(index / 12)
        const selected = selection?.kind === "block" && (
          selection.key === block.key || selection.key === `${block.pool}:${block.blockId}`
        )
        return (
          <mesh
            key={block.key}
            position={[column * 0.84, -row * 0.72, 0]}
            onClick={(event) => {
              event.stopPropagation()
              onSelect({ kind: "block", key: block.key })
            }}
          >
            <boxGeometry args={[selected ? 0.72 : 0.64, selected ? 0.6 : 0.52, 0.5]} />
            <meshStandardMaterial color={blockColor(block, state, requestOrder, Boolean(selected))} />
          </mesh>
        )
      })}
      <TextSprite
        text={`${blocks.length} recorded allocation generations${blocks.length > visible.length ? ` · showing ${visible.length}` : ""}`}
        position={[4.55, -Math.ceil(visible.length / 12) * 0.72 - 0.25, 0]}
        scale={[5, 0.5, 1]}
        muted
      />
    </group>
  )
}

function OutputGrid({
  requests,
  requestOrder,
  selection,
  onSelect,
}: {
  requests: ReplayState["requests"][string][]
  requestOrder: string[]
  selection: Selection | null
  onSelect: (selection: Selection) => void
}) {
  if (requests.length === 0) return <SceneEmpty label="No request output at this playhead" />
  return (
    <group>
      {requests.slice(0, 30).map((request, index) => (
        <DataTile
          key={request.id}
          position={gridPosition(index, 5)}
          label={request.id}
          detail={`${request.acceptedTokenCount} accepted · ${request.finishReason ?? request.status}`}
          color={requestColor(requestOrder, request.id)}
          selected={selection?.kind === "request" && selection.id === request.id}
          onSelect={() => onSelect({ kind: "request", id: request.id })}
        />
      ))}
    </group>
  )
}

function DataTile({
  position,
  label,
  detail,
  color,
  selected,
  onSelect,
}: {
  position: [number, number, number]
  label: string
  detail: string
  color: string
  selected: boolean
  onSelect: () => void
}) {
  return (
    <group position={position}>
      {selected ? (
        <mesh position={[0, 0, -0.1]}>
          <boxGeometry args={[1.9, 1.02, 0.14]} />
          <meshStandardMaterial color="#e8edf5" />
        </mesh>
      ) : null}
      <mesh
        onClick={(event) => {
          event.stopPropagation()
          onSelect()
        }}
      >
        <boxGeometry args={[1.76, 0.88, 0.45]} />
        <meshStandardMaterial color="#1a202b" />
      </mesh>
      <mesh position={[-0.69, 0, 0.26]}>
        <boxGeometry args={[0.08, 0.62, 0.06]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <TextSprite text={label} position={[0.1, 0.16, 0.29]} scale={[1.34, 0.34, 1]} />
      <TextSprite text={detail} position={[0.1, -0.2, 0.29]} scale={[1.36, 0.27, 1]} muted />
    </group>
  )
}

function SceneEmpty({ label }: { label: string }) {
  return (
    <group>
      <mesh>
        <boxGeometry args={[6.2, 1.5, 0.3]} />
        <meshStandardMaterial color="#171d27" />
      </mesh>
      <TextSprite text={label} position={[0, 0.05, 0.22]} scale={[5.4, 0.62, 1]} muted />
    </group>
  )
}

function TextSprite({
  text,
  position,
  scale,
  muted = false,
}: {
  text: string
  position: [number, number, number]
  scale: [number, number, number]
  muted?: boolean
}) {
  const texture = useMemo(() => {
    const canvas = document.createElement("canvas")
    canvas.width = 1024
    canvas.height = 256
    const context = canvas.getContext("2d")
    if (context) {
      context.clearRect(0, 0, canvas.width, canvas.height)
      context.fillStyle = muted ? "#a2afc2" : "#e8edf5"
      context.font = '600 104px "Segoe UI", system-ui, sans-serif'
      context.textAlign = "center"
      context.textBaseline = "middle"
      const fitted = text.length > 44 ? `${text.slice(0, 41)}…` : text
      context.fillText(fitted, canvas.width / 2, canvas.height / 2, canvas.width - 32)
    }
    const next = new THREE.CanvasTexture(canvas)
    next.colorSpace = THREE.SRGBColorSpace
    next.minFilter = THREE.LinearFilter
    return next
  }, [muted, text])
  useEffect(() => () => texture.dispose(), [texture])
  return (
    <sprite position={position} scale={scale}>
      <spriteMaterial map={texture} transparent depthWrite={false} />
    </sprite>
  )
}

function orderedRequests(state: ReplayState, requestOrder: string[]): ReplayState["requests"][string][] {
  const order = [...requestOrder]
  for (const id of Object.keys(state.requests)) {
    if (!order.includes(id)) order.push(id)
  }
  return order.map((id) => state.requests[id]).filter((request) => request != null)
}

function gridPosition(index: number, columns: number): [number, number, number] {
  const column = index % columns
  const row = Math.floor(index / columns)
  const width = (columns - 1) * 2.08
  return [column * 2.08 - width / 2, 2.35 - row * 1.22, 0]
}

function componentLabel(id: WorkflowComponentId): string {
  return WORKFLOW_COMPONENTS.find((item) => item.id === id)?.label ?? id
}

function blockColor(block: BlockState, state: ReplayState, order: string[], selected: boolean): string {
  if (selected) return "#e8edf5"
  if (block.refCount > 0) {
    const holders = blockRequestIds(state, block)
    return block.refCount === 1 && holders.length === 1 ? requestColor(order, holders[0]) : "#77b7ff"
  }
  if (block.cachedAtZeroRefs) return "#e0b15a"
  return "#3e4a5d"
}
