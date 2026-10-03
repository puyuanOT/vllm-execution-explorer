import type { ThreeEvent } from "@react-three/fiber"
import { useLayoutEffect, useRef, useState } from "react"
import * as THREE from "three"
import { sceneClick } from "./model.ts"

export { sceneClick }

export type Position3 = [number, number, number]

export const SCENE_COLORS = {
  background: "#06080c", ink: "#e8edf5", muted: "#a2afc2", accent: "#77b7ff",
  surface: "#222b39", line: "#58677d", cached: "#e0b15a", free: "#294b40",
  reserved: "#677080", unknown: "#202733",
} as const

export function TextSprite({ text, position, scale, muted = false }: {
  text: string; position: Position3; scale: Position3; muted?: boolean
}) {
  const material = useRef<THREE.SpriteMaterial>(null)
  const [aspect, setAspect] = useState(1)
  useLayoutEffect(() => {
    const canvas = document.createElement("canvas")
    const context = canvas.getContext("2d")
    if (context) {
      context.font = '600 64px system-ui, sans-serif'
      canvas.width = Math.min(4096, Math.max(80, Math.ceil(context.measureText(text).width) + 20))
      canvas.height = 88
      setAspect(canvas.width / canvas.height)
      context.fillStyle = muted ? SCENE_COLORS.muted : SCENE_COLORS.ink
      context.font = '600 64px system-ui, sans-serif'
      context.textAlign = "center"
      context.textBaseline = "middle"
      context.fillText(text, canvas.width / 2, canvas.height / 2, canvas.width - 20)
    }
    const texture = new THREE.CanvasTexture(canvas)
    texture.colorSpace = THREE.SRGBColorSpace
    texture.minFilter = THREE.LinearFilter
    texture.generateMipmaps = false
    const active = material.current
    if (active) { active.map = texture; active.needsUpdate = true }
    return () => {
      if (active?.map === texture) { active.map = null; active.needsUpdate = true }
      texture.dispose()
    }
  }, [text, muted])
  const height = Math.min(scale[1], scale[0] / aspect)
  return <sprite position={position} scale={[height * aspect, height, 1]}><spriteMaterial ref={material} transparent depthWrite={false} /></sprite>
}

export function Connector({ from, to, muted = false, arrow = true }: {
  from: Position3; to: Position3; muted?: boolean; arrow?: boolean
}) {
  const start = new THREE.Vector3(...from)
  const end = new THREE.Vector3(...to)
  const direction = end.clone().sub(start)
  const length = direction.length()
  const midpoint = start.clone().add(end).multiplyScalar(0.5)
  const rotation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1, 0, 0), direction.clone().normalize())
  const arrowRotation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize())
  return <group>
    <mesh position={midpoint} quaternion={rotation}>
      <boxGeometry args={[length, 0.024, 0.024]} /><meshBasicMaterial color={muted ? "#3e4a5d" : SCENE_COLORS.line} />
    </mesh>
    {arrow ? <mesh position={end.clone().addScaledVector(direction.normalize(), -0.08)} quaternion={arrowRotation}>
      <coneGeometry args={[0.075, 0.18, 5]} /><meshBasicMaterial color={muted ? "#3e4a5d" : SCENE_COLORS.line} />
    </mesh> : null}
  </group>
}

export function SceneTile({ position, label, detail, selected = false, color = SCENE_COLORS.accent,
  width = 2, onSelect, onExplore }: {
  position: Position3; label: string; detail: string; selected?: boolean; color?: string; width?: number;
  onSelect?: () => void; onExplore?: () => void
}) {
  function select(event: ThreeEvent<MouseEvent>) {
    if (!sceneClick(event)) return
    event.stopPropagation(); onSelect?.()
  }
  function explore(event: ThreeEvent<MouseEvent>) {
    if (!sceneClick(event)) return
    event.stopPropagation(); onSelect?.(); onExplore?.()
  }
  return <group position={position}>
    {selected ? <mesh position={[0, 0, -0.12]}><boxGeometry args={[width + 0.12, 1.14, 0.15]} /><meshBasicMaterial color={SCENE_COLORS.ink} /></mesh> : null}
    <mesh onClick={onSelect ? select : undefined} onDoubleClick={onExplore ? explore : undefined}>
      <boxGeometry args={[width, 1.02, 0.46]} /><meshStandardMaterial color={SCENE_COLORS.surface} roughness={0.8} />
    </mesh>
    <mesh position={[-width / 2 + 0.15, 0, 0.27]}><boxGeometry args={[0.06, 0.72, 0.035]} /><meshBasicMaterial color={color} /></mesh>
    <TextSprite text={label} position={[0.08, 0.18, 0.3]} scale={[width - 0.3, 0.35, 1]} />
    <TextSprite text={detail} position={[0.08, -0.23, 0.3]} scale={[width - 0.3, 0.27, 1]} muted />
  </group>
}

export function SceneEmpty({ label, detail = "Open a compatible recording to inspect this evidence." }: { label: string; detail?: string }) {
  return <group>
    <TextSprite text={label} position={[0, 0.35, 0]} scale={[7.2, 0.55, 1]} />
    <TextSprite text={detail} position={[0, -0.35, 0]} scale={[7.6, 0.4, 1]} muted />
  </group>
}
