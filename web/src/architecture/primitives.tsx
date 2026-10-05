import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber"
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { createRoot, type Root } from "react-dom/client"
import * as THREE from "three"
import { sceneClick } from "./model.ts"

export { sceneClick }

export type Position3 = [number, number, number]

export const SCENE_COLORS = {
  background: "#04070c",
  ink: "#e2eaf4",
  muted: "#95a5bb",
  accent: "#3fcde6",
  accentSoft: "#9fe8f5",
  surface: "#152036",
  surfaceSelected: "#1d4b58",
  surfaceRelated: "#132f3b",
  edge: "#3d5779",
  edgeDim: "#26395a",
  edgeMarker: "#56739a",
  line: "#45618a",
  lineDim: "#2c4262",
  explain: "#9d90ec",
  measured: "#dfad57",
  cached: "#dfad57",
  free: "#122220",
  freeEdge: "#21443b",
  reserved: "#4b5a74",
  unknown: "#1a2232",
} as const

export type LabelAnchor = "center" | "above" | "below" | "left" | "right"

/** Top-left of the anchored box relative to its anchor point; mirrors the .scene-label transforms in styles.css. */
function anchorOffset(anchor: LabelAnchor, width: number, height: number): [number, number] {
  if (anchor === "above") return [-width / 2, -height - 12]
  if (anchor === "below") return [-width / 2, 12]
  if (anchor === "left") return [-width - 14, -height / 2]
  if (anchor === "right") return [14, -height / 2]
  return [-width / 2, -height / 2]
}

const CLAMP_MARGIN = 8

/** Pixel bands a clamped label keeps clear of, measured from each canvas edge. */
export interface ClampInsets { top?: number; right?: number; bottom?: number; left?: number }

/**
 * DOM text that follows a 3D anchor; it stays crisp at any zoom and never blocks scene picking unless interactive.
 * Clamped labels are nudged so the whole box stays inside the canvas, outside any given inset bands.
 */
export function SceneLabel({ position = [0, 0, 0], anchor = "center", interactive = false, clamp = false, children }: {
  position?: Position3
  anchor?: LabelAnchor
  interactive?: boolean
  clamp?: boolean | ClampInsets
  children: ReactNode
}) {
  const group = useRef<THREE.Group>(null)
  const gl = useThree((state) => state.gl)
  const camera = useThree((state) => state.camera)
  const size = useThree((state) => state.size)
  const invalidate = useThree((state) => state.invalidate)
  const [element] = useState(() => document.createElement("div"))
  const root = useRef<Root | null>(null)
  const vector = useMemo(() => new THREE.Vector3(), [])

  useLayoutEffect(() => {
    const parent = gl.domElement.parentElement
    if (!parent) return
    element.className = "scene-label"
    element.style.pointerEvents = "none"
    parent.appendChild(element)
    const next = createRoot(element)
    root.current = next
    invalidate()
    return () => {
      root.current = null
      window.setTimeout(() => next.unmount(), 0)
      element.remove()
    }
  }, [gl, element, invalidate])

  useLayoutEffect(() => {
    element.dataset.anchor = anchor
    element.style.zIndex = interactive ? "20" : ""
  }, [element, anchor, interactive])

  // Only the translated inner box takes pointer events; the outer box stays at the anchor and would block the scene.
  useLayoutEffect(() => {
    root.current?.render(<div className="scene-label-inner" style={interactive ? { pointerEvents: "auto" } : undefined}>{children}</div>)
  })

  useEffect(() => { invalidate() }, [position[0], position[1], position[2], invalidate])

  const clamped = Boolean(clamp)
  useEffect(() => {
    if (!clamped) return
    const observer = new ResizeObserver(() => invalidate())
    observer.observe(element)
    return () => observer.disconnect()
  }, [clamped, element, invalidate])

  useFrame(() => {
    const object = group.current
    if (!object) return
    object.getWorldPosition(vector).project(camera)
    const visible = vector.z > -1 && vector.z < 1
    element.style.display = visible ? "" : "none"
    if (!visible) return
    let x = (vector.x * 0.5 + 0.5) * size.width
    let y = (-vector.y * 0.5 + 0.5) * size.height
    const box = clamp ? element.firstElementChild as HTMLElement | null : null
    if (box) {
      const insets = clamp === true ? {} : clamp as ClampInsets
      const free = `${Math.max(96, Math.floor(size.height - (insets.top ?? CLAMP_MARGIN) - (insets.bottom ?? CLAMP_MARGIN)))}px`
      if (element.style.getPropertyValue("--label-max-height") !== free) element.style.setProperty("--label-max-height", free)
      const [dx, dy] = anchorOffset(anchor, box.offsetWidth, box.offsetHeight)
      const maxX = size.width - (insets.right ?? CLAMP_MARGIN) - box.offsetWidth
      const maxY = size.height - (insets.bottom ?? CLAMP_MARGIN) - box.offsetHeight
      // A box larger than the free band keeps clear of the right and bottom bands and overflows toward the top-left.
      x = Math.max(CLAMP_MARGIN, Math.min(Math.max(x + dx, insets.left ?? CLAMP_MARGIN), maxX)) - dx
      y = Math.max(CLAMP_MARGIN, Math.min(Math.max(y + dy, insets.top ?? CLAMP_MARGIN), maxY)) - dy
    }
    element.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`
  })

  return <group ref={group} position={position} />
}

export function SceneText({ text, position, muted = false, strong = false, anchor = "center" }: {
  text: string; position: Position3; muted?: boolean; strong?: boolean; anchor?: LabelAnchor
}) {
  return (
    <SceneLabel position={position} anchor={anchor}>
      <span className={muted ? "scene-tag" : "scene-text"} data-tone={strong ? "strong" : undefined}>{text}</span>
    </SceneLabel>
  )
}

export function SceneCard({ position, code, name, value, detail, active = false, anchor = "above" }: {
  position: Position3; code?: string; name: string; value?: string; detail?: string; active?: boolean; anchor?: LabelAnchor
}) {
  return (
    <SceneLabel position={position} anchor={anchor}>
      <div className="scene-card" data-active={active || undefined}>
        <div className="flex items-center gap-2">
          {code ? <span className={`grid h-[18px] min-w-7 place-items-center border px-1 font-mono text-[9.5px] font-semibold tracking-[0.06em] ${active ? "border-accent-400/45 text-accent" : "border-[#2a3c58] text-muted"}`}>{code}</span> : null}
          <span className="scene-card-name">{name}</span>
        </div>
        {value || detail ? <div className="scene-card-metric">{value ? <b>{value}</b> : null}{detail ? <span>{detail}</span> : null}</div> : null}
      </div>
    </SceneLabel>
  )
}

export function useHoverCursor() {
  const gl = useThree((state) => state.gl)
  const [hovered, setHovered] = useState(false)
  useEffect(() => () => { gl.domElement.style.cursor = "" }, [gl])
  return {
    hovered,
    bind: {
      onPointerOver: (event: ThreeEvent<PointerEvent>) => { event.stopPropagation(); setHovered(true); gl.domElement.style.cursor = "pointer" },
      onPointerOut: () => { setHovered(false); gl.domElement.style.cursor = "" },
    },
  }
}

/** One-pixel outline of a box; raycasting is disabled so outlines never steal clicks from their meshes. */
export function BoxEdges({ size, color = SCENE_COLORS.edge, opacity = 1, position }: {
  size: Position3; color?: string; opacity?: number; position?: Position3
}) {
  const geometry = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(...size)), [size[0], size[1], size[2]])
  useEffect(() => () => geometry.dispose(), [geometry])
  return (
    <lineSegments geometry={geometry} position={position} raycast={() => null}>
      <lineBasicMaterial color={color} transparent={opacity < 1} opacity={opacity} />
    </lineSegments>
  )
}

export function Polyline({ points, color = SCENE_COLORS.line, opacity = 1, dashed = false, dash = 0.16, gap = 0.12 }: {
  points: Position3[]; color?: string; opacity?: number; dashed?: boolean; dash?: number; gap?: number
}) {
  const key = points.map((point) => point.join(",")).join(";")
  const line = useMemo(() => {
    const geometry = new THREE.BufferGeometry().setFromPoints(points.map((point) => new THREE.Vector3(...point)))
    const material = dashed
      ? new THREE.LineDashedMaterial({ color, dashSize: dash, gapSize: gap, transparent: opacity < 1, opacity })
      : new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity })
    const object = new THREE.Line(geometry, material)
    if (dashed) object.computeLineDistances()
    object.raycast = () => undefined
    return object
  }, [key, color, opacity, dashed, dash, gap])
  useEffect(() => () => { line.geometry.dispose(); (line.material as THREE.Material).dispose() }, [line])
  return <primitive object={line} />
}

export function Connector({ from, to, muted = false, arrow = true, dashed = false }: {
  from: Position3; to: Position3; muted?: boolean; arrow?: boolean; dashed?: boolean
}) {
  const start = new THREE.Vector3(...from)
  const end = new THREE.Vector3(...to)
  const direction = end.clone().sub(start).normalize()
  const arrowRotation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction)
  const color = muted ? SCENE_COLORS.lineDim : SCENE_COLORS.line
  return <group>
    <Polyline points={[from, to]} color={color} dashed={dashed} />
    {arrow ? <mesh position={end.clone().addScaledVector(direction, -0.07)} quaternion={arrowRotation} raycast={() => null}>
      <coneGeometry args={[0.055, 0.15, 6]} /><meshBasicMaterial color={color} />
    </mesh> : null}
  </group>
}

export function SceneTile({ position, label, detail, selected = false, color = SCENE_COLORS.accent,
  width = 2, height = 1.02, depth = 0.46, onSelect, onExplore }: {
  position: Position3; label: string; detail: string; selected?: boolean; color?: string; width?: number; height?: number; depth?: number;
  onSelect?: () => void; onExplore?: () => void
}) {
  const hover = useHoverCursor()
  const interactive = Boolean(onSelect || onExplore)
  function select(event: ThreeEvent<MouseEvent>) {
    if (!sceneClick(event)) return
    event.stopPropagation(); onSelect?.()
  }
  function explore(event: ThreeEvent<MouseEvent>) {
    if (!sceneClick(event)) return
    event.stopPropagation(); onSelect?.(); onExplore?.()
  }
  const edge = selected ? SCENE_COLORS.accentSoft : hover.hovered && interactive ? SCENE_COLORS.edgeMarker : SCENE_COLORS.edge
  return <group position={position}>
    <mesh onClick={onSelect ? select : undefined} onDoubleClick={onExplore ? explore : undefined} {...(interactive ? hover.bind : {})}>
      <boxGeometry args={[width, height, depth]} />
      <meshStandardMaterial color={selected ? SCENE_COLORS.surfaceSelected : SCENE_COLORS.surface} roughness={0.75} metalness={0.1} />
    </mesh>
    <BoxEdges size={[width, height, depth]} color={edge} />
    {selected ? <BoxEdges size={[width + 0.14, height + 0.14, depth + 0.14]} color={SCENE_COLORS.accent} opacity={0.55} /> : null}
    <mesh position={[-width / 2 + 0.13, 0, depth / 2 + 0.006]} raycast={() => null}>
      <planeGeometry args={[0.05, height * 0.66]} /><meshBasicMaterial color={color} toneMapped={false} />
    </mesh>
    <SceneLabel position={[0.06, 0, depth / 2 + 0.02]}>
      <span className="block max-w-56 text-center leading-tight">
        <span className="block truncate font-mono text-[11px] font-medium text-ink">{label}</span>
        <span className="block truncate text-[10.5px] text-muted">{detail}</span>
      </span>
    </SceneLabel>
  </group>
}

export function SceneEmpty({ label, detail = "Open a compatible recording to inspect this evidence." }: { label: string; detail?: string }) {
  return <SceneLabel position={[0, 0, 0]}>
    <div className="max-w-md border border-line-strong bg-surface/95 px-4 py-3 text-center">
      <p className="m-0 text-sm font-semibold text-ink">{label}</p>
      <p className="m-0 mt-1 text-pretty text-xs text-muted">{detail}</p>
    </div>
  </SceneLabel>
}

export function FloorGrid({ y, size = 160 }: { y: number; size?: number }) {
  const grids = useMemo(() => {
    const minor = new THREE.GridHelper(size, size / 2, "#0d1624", "#0d1624")
    const major = new THREE.GridHelper(size, size / 10, "#16243a", "#16243a")
    minor.raycast = () => undefined
    major.raycast = () => undefined
    return { minor, major }
  }, [size])
  useEffect(() => () => {
    grids.minor.geometry.dispose(); (grids.minor.material as THREE.Material).dispose()
    grids.major.geometry.dispose(); (grids.major.material as THREE.Material).dispose()
  }, [grids])
  return <group position={[0, y, 0]}>
    <primitive object={grids.minor} />
    <primitive object={grids.major} position={[0, 0.002, 0]} />
  </group>
}
