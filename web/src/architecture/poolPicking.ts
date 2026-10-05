import * as THREE from "three"
import type { DieLayout, PoolGridLayout } from "./model.ts"

/** Traverse floor cells, then intersect their actual variable-height instance bounds. */
export function createDieRaycast(layout: DieLayout): THREE.InstancedMesh["raycast"] {
  const inverse = new THREE.Matrix4()
  const instanceMatrix = new THREE.Matrix4()
  const ray = new THREE.Ray()
  const entry = new THREE.Vector3()
  const hit = new THREE.Vector3()
  const worldHit = new THREE.Vector3()
  const bounds = new THREE.Box3()
  const cellBox = new THREE.Box3()
  return function (this: THREE.InstancedMesh, raycaster, intersections) {
    if (!layout.capacity) return
    if (!this.boundingBox) this.computeBoundingBox()
    if (!this.geometry.boundingBox) this.geometry.computeBoundingBox()
    bounds.copy(this.boundingBox!)
    // Include gaps and padding in the traversal grid, but only actual columns can be hit.
    bounds.min.x = -layout.width / 2
    bounds.max.x = layout.width / 2
    bounds.min.z = -layout.depth / 2
    bounds.max.z = layout.depth / 2
    inverse.copy(this.matrixWorld).invert()
    ray.copy(raycaster.ray).applyMatrix4(inverse)
    if (!ray.intersectBox(bounds, entry)) return
    if (bounds.containsPoint(ray.origin)) entry.copy(ray.origin)
    const dims = [layout.columns, layout.rows]
    const origin = [ray.origin.x, ray.origin.z]
    const direction = [ray.direction.x, ray.direction.z]
    const min = [bounds.min.x, bounds.min.z]
    const coordinates = [entry.x, entry.z].map((value, axis) => THREE.MathUtils.clamp(
      Math.floor((value + direction[axis] * 1e-7 - min[axis]) / layout.pitch), 0, dims[axis] - 1,
    ))
    const strides = direction.map((value) => value > 0 ? 1 : value < 0 ? -1 : 0)
    const next = direction.map((value, axis) => value === 0 ? Infinity : (
      min[axis] + (coordinates[axis] + (value > 0 ? 1 : 0)) * layout.pitch - origin[axis]
    ) / value)
    const delta = direction.map((value) => value === 0 ? Infinity : layout.pitch / Math.abs(value))
    for (let visited = 0; visited <= layout.columns + layout.rows; visited += 1) {
      const index = (layout.rows - 1 - coordinates[1]) * layout.columns + coordinates[0]
      if (index < layout.capacity) {
        this.getMatrixAt(index, instanceMatrix)
        cellBox.copy(this.geometry.boundingBox!).applyMatrix4(instanceMatrix)
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
      const axis = next[0] <= next[1] ? 0 : 1
      if (!strides[axis]) return
      coordinates[axis] += strides[axis]
      if (coordinates[axis] < 0 || coordinates[axis] >= dims[axis]) return
      next[axis] += delta[axis]
    }
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
