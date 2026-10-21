import { useLayoutEffect, useMemo, useRef } from "react"
import * as THREE from "three"
import type { KvBlockProjection } from "../../../replay/src/learning.ts"
import { requestColor } from "../theme.ts"
import { physicalBlockState } from "./model.ts"
import { SegmentLines, SCENE_COLORS } from "./primitives.tsx"

export interface BlockMarkCell {
  block: KvBlockProjection
  x: number
  z: number
  height: number
  size: number
  selected?: boolean
}

/** The same visual grammar is used at both zoom levels. Color identifies current holders only. */
export function BlockMarks({ cells, requestOrder }: { cells: BlockMarkCell[]; requestOrder: string[] }) {
  const tops = useRef<THREE.InstancedMesh>(null)
  const marks = useMemo(() => {
    const holders: Array<{ cell: BlockMarkCell; id: string; index: number; count: number }> = []
    const pattern: number[] = []
    const brackets: number[] = []
    const segment = (target: number[], ax: number, ay: number, az: number, bx: number, by: number, bz: number) => target.push(ax, ay, az, bx, by, bz)
    for (const cell of cells) {
      const state = physicalBlockState(cell.block)
      const ids = state === "referenced" ? [...new Set(cell.block.bindings.map((binding) => binding.requestId))] : []
      ids.sort((a, b) => requestOrder.indexOf(a) - requestOrder.indexOf(b) || a.localeCompare(b))
      ids.forEach((id, index) => holders.push({ cell, id, index, count: ids.length }))
      const { x, z, size, height } = cell
      const half = size / 2
      const y = height + 0.015
      if (state === "cached") {
        // A hatched top remains recognizable without request color or a brown status fill.
        for (const offset of [-0.22, 0, 0.22]) {
          segment(pattern, x - size * 0.25, y, z + size * (offset - 0.14), x + size * 0.25, y, z + size * (offset + 0.14))
        }
      } else if (state === "null") {
        segment(pattern, x - half * 0.7, y, z - half * 0.7, x + half * 0.7, y, z + half * 0.7)
        segment(pattern, x - half * 0.7, y, z + half * 0.7, x + half * 0.7, y, z - half * 0.7)
      } else if (state === "unknown") {
        segment(pattern, x - size * 0.2, y, z, x + size * 0.2, y, z)
      }
      if (cell.selected) {
        const outer = half + size * 0.12
        const inset = size * 0.23
        for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
          segment(brackets, x + sx * (outer - inset), y + 0.02, z + sz * outer, x + sx * outer, y + 0.02, z + sz * outer)
          segment(brackets, x + sx * outer, y + 0.02, z + sz * outer, x + sx * outer, y + 0.02, z + sz * (outer - inset))
        }
      }
    }
    return { holders, pattern, brackets }
  }, [cells, requestOrder])
  useLayoutEffect(() => {
    const mesh = tops.current
    if (!mesh) return
    const transform = new THREE.Object3D()
    transform.rotation.x = -Math.PI / 2
    const color = new THREE.Color()
    marks.holders.forEach(({ cell, id, index, count }, slot) => {
      // Equal bands identify shared references, not token occupancy or ownership fractions.
      const face = cell.size * 0.92
      const band = face / count
      const gap = count > 1 ? Math.min(cell.size * 0.02, band * 0.08) : 0
      transform.position.set(cell.x - face / 2 + band * (index + 0.5), cell.height + cell.size * 0.005, cell.z)
      transform.scale.set(band - gap, face, 1)
      transform.updateMatrix()
      mesh.setMatrixAt(slot, transform.matrix)
      mesh.setColorAt(slot, color.set(requestColor(requestOrder, id)))
    })
    mesh.count = marks.holders.length
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    mesh.computeBoundingSphere()
  }, [marks, requestOrder])
  return <group>
    {marks.holders.length ? <instancedMesh ref={tops} args={[undefined, undefined, marks.holders.length]} raycast={() => null}>
      <planeGeometry args={[1, 1]} /><meshBasicMaterial toneMapped={false} />
    </instancedMesh> : null}
    {marks.pattern.length ? <SegmentLines positions={marks.pattern} color="#bac7d8" width={1.25} /> : null}
    {marks.brackets.length ? <SegmentLines positions={marks.brackets} color={SCENE_COLORS.accentSoft} width={2} /> : null}
  </group>
}
