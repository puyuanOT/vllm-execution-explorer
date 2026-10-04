import * as THREE from "three"
import type { DieLayout, PoolGridLayout } from "./model.ts"

/** Picks a flat die by intersecting its top plane, so cost does not grow with pool capacity. */
export function createDieRaycast(layout: DieLayout, top = 0): THREE.InstancedMesh["raycast"] {
  const inverse = new THREE.Matrix4()
  const ray = new THREE.Ray()
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -top)
  const hit = new THREE.Vector3()
  const worldHit = new THREE.Vector3()
  return function (this: THREE.InstancedMesh, raycaster, intersections) {
    if (!layout.capacity) return
    inverse.copy(this.matrixWorld).invert()
    ray.copy(raycaster.ray).applyMatrix4(inverse)
    if (!ray.intersectPlane(plane, hit)) return
    const column = Math.floor((hit.x + layout.width / 2) / layout.pitch)
    const row = Math.floor((layout.depth / 2 - hit.z) / layout.pitch)
    if (column < 0 || column >= layout.columns || row < 0 || row >= layout.rows) return
    const index = row * layout.columns + column
    if (index >= layout.capacity) return
    worldHit.copy(hit).applyMatrix4(this.matrixWorld)
    const distance = raycaster.ray.origin.distanceTo(worldHit)
    if (distance < raycaster.near || distance > raycaster.far) return
    intersections.push({ distance, point: worldHit.clone(), object: this, instanceId: index })
  }
}

/** Grid traversal avoids the default O(capacity) instanced-mesh raycast. */
export function createPoolRaycast(layout: PoolGridLayout): THREE.InstancedMesh["raycast"] {
  const inverse = new THREE.Matrix4()
  const ray = new THREE.Ray()
  const entry = new THREE.Vector3()
  const hit = new THREE.Vector3()
  const worldHit = new THREE.Vector3()
  const cellBox = new THREE.Box3()
  const half = new THREE.Vector3(layout.columns, layout.rows, layout.depths).multiplyScalar(layout.spacing / 2)
  const bounds = new THREE.Box3(half.clone().negate(), half.clone())
  return function (this: THREE.InstancedMesh, raycaster, intersections) {
    if (!layout.capacity) return
    inverse.copy(this.matrixWorld).invert()
    ray.copy(raycaster.ray).applyMatrix4(inverse)
    if (!ray.intersectBox(bounds, entry)) return
    if (bounds.containsPoint(ray.origin)) entry.copy(ray.origin)
    const dims = [layout.columns, layout.rows, layout.depths]
    const origin = ray.origin.toArray()
    const direction = ray.direction.toArray()
    const min = bounds.min.toArray()
    const coordinates = entry.toArray().map((value, axis) => THREE.MathUtils.clamp(
      Math.floor((value + direction[axis] * 1e-7 - min[axis]) / layout.spacing), 0, dims[axis] - 1,
    ))
    const strides = direction.map((value) => value > 0 ? 1 : value < 0 ? -1 : 0)
    const next = direction.map((value, axis) => value === 0 ? Infinity : (
      min[axis] + (coordinates[axis] + (value > 0 ? 1 : 0)) * layout.spacing - origin[axis]
    ) / value)
    const delta = direction.map((value) => value === 0 ? Infinity : layout.spacing / Math.abs(value))
    for (let visited = 0; visited <= dims[0] + dims[1] + dims[2]; visited += 1) {
      const [x, y, z] = coordinates
      // The visible grid goes top-to-bottom and front-to-back in physical-ID order.
      const index = (layout.depths - 1 - z) * layout.columns * layout.rows
        + (layout.rows - 1 - y) * layout.columns + x
      if (index >= 0 && index < layout.capacity) {
        const center = min.map((value, axis) => value + (coordinates[axis] + 0.5) * layout.spacing)
        cellBox.min.set(...center.map((value) => value - layout.spacing * 0.37) as [number, number, number])
        cellBox.max.set(...center.map((value) => value + layout.spacing * 0.37) as [number, number, number])
        if (ray.intersectBox(cellBox, hit)) {
          worldHit.copy(hit).applyMatrix4(this.matrixWorld)
          const distance = raycaster.ray.origin.distanceTo(worldHit)
          if (distance >= raycaster.near && distance <= raycaster.far) {
            intersections.push({ distance, point: worldHit.clone(), object: this, instanceId: index })
            return
          }
          if (distance > raycaster.far) return
        }
      }
      const axis = next[0] <= next[1] && next[0] <= next[2] ? 0 : next[1] <= next[2] ? 1 : 2
      coordinates[axis] += strides[axis]
      if (coordinates[axis] < 0 || coordinates[axis] >= dims[axis]) return
      next[axis] += delta[axis]
    }
  }
}
