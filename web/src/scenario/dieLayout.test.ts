import assert from "node:assert/strict"
import test from "node:test"
import * as THREE from "three"

import { dieCellPosition, dieLayout } from "../architecture/model.ts"
import { createDieRaycast } from "../architecture/poolPicking.ts"

test("a flat die gives every physical block its own floor cell", () => {
  const layout = dieLayout(35000)
  assert.equal(layout.capacity, 35000)
  assert.ok(layout.columns * layout.rows >= layout.capacity)
  const cells = new Set(Array.from({ length: layout.capacity }, (_, index) => dieCellPosition(index, layout).join(",")))
  assert.equal(cells.size, 35000)
  for (const index of [0, 1, layout.columns, 34999]) {
    const [x, z] = dieCellPosition(index, layout)
    assert.ok(Math.abs(x) <= layout.width / 2 && Math.abs(z) <= layout.depth / 2)
  }
  assert.equal(dieLayout(0).capacity, 0)
  assert.equal(dieLayout(Number.NaN).capacity, 0)
})

test("block zero sits at the front-left corner of the die", () => {
  const layout = dieLayout(8)
  const [x0, z0] = dieCellPosition(0, layout)
  const [x1] = dieCellPosition(1, layout)
  const [, zNextRow] = dieCellPosition(layout.columns, layout)
  assert.ok(x1 > x0)
  assert.ok(zNextRow < z0)
})

test("die picking returns the physical block under the pointer and skips padding", () => {
  const layout = dieLayout(4410)
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial(), layout.capacity)
  mesh.raycast = createDieRaycast(layout)
  const transform = new THREE.Object3D()
  for (let index = 0; index < layout.capacity; index++) {
    const [x, z] = dieCellPosition(index, layout)
    transform.position.set(x, 0.02, z)
    transform.scale.set(layout.pitch * 0.78, 0.04, layout.pitch * 0.78)
    transform.updateMatrix()
    mesh.setMatrixAt(index, transform.matrix)
  }
  mesh.computeBoundingBox()
  mesh.position.set(3, -1, 2)
  mesh.updateMatrixWorld(true)
  try {
    for (const index of [0, 1, 4409, layout.columns * 7 + 3]) {
      const [x, z] = dieCellPosition(index, layout)
      const raycaster = new THREE.Raycaster(new THREE.Vector3(x + 3, 10, z + 2), new THREE.Vector3(0, -1, 0))
      assert.equal(raycaster.intersectObject(mesh)[0]?.instanceId, index)
    }
    const paddingIndex = layout.columns * layout.rows - 1
    assert.ok(paddingIndex >= layout.capacity)
    const [px, pz] = dieCellPosition(paddingIndex, layout)
    const miss = new THREE.Raycaster(new THREE.Vector3(px + 3, 10, pz + 2), new THREE.Vector3(0, -1, 0))
    assert.equal(miss.intersectObject(mesh).length, 0)
  } finally {
    mesh.geometry.dispose()
    ;(mesh.material as THREE.Material).dispose()
  }
})

// Match the variable-height instances rendered by CacheStage, including a raised selection.
test("oblique die picking hits the rendered columns rather than a fixed floor plane", () => {
  for (const [capacity, selected, target] of [[8, 3, 3], [4410, 300, 300], [35000, 300, 21000]] as const) {
    const layout = dieLayout(capacity)
    const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial(), capacity)
    const transform = new THREE.Object3D()
    const freeHeight = Math.min(0.035, layout.pitch * 0.06)
    const selectedHeight = Math.max(0.6, Math.min(1.15, layout.pitch * 6))
    for (let index = 0; index < capacity; index++) {
      const [x, z] = dieCellPosition(index, layout)
      const height = index === selected ? selectedHeight : freeHeight
      transform.position.set(x, height / 2, z)
      transform.scale.set(layout.pitch * 0.78, height, layout.pitch * 0.78)
      transform.updateMatrix()
      mesh.setMatrixAt(index, transform.matrix)
    }
    mesh.computeBoundingBox()
    mesh.raycast = createDieRaycast(layout)
    try {
      const [x, z] = dieCellPosition(target, layout)
      const height = target === selected ? selectedHeight : freeHeight
      const top = new THREE.Vector3(x, height, z)
      for (const camera of [new THREE.Vector3(0, 0.85, 1), new THREE.Vector3(1, 0.85, 0), new THREE.Vector3(-1, 0.85, -1)]) {
        const origin = top.clone().add(camera.clone().multiplyScalar(20))
        const raycaster = new THREE.Raycaster(origin, camera.clone().negate().normalize())
        const hit = raycaster.intersectObject(mesh)[0]
        assert.equal(hit?.instanceId, target, `capacity ${capacity}, target ${target}, camera ${camera.toArray()}`)
        assert.ok(hit!.point.distanceTo(top) < 1e-5)
      }
      // World-space transforms and distance limits must retain physical identity.
      mesh.position.set(3, -1, 2)
      mesh.rotation.set(0.2, 0.5, -0.1)
      mesh.scale.set(1.5, 0.8, 1.2)
      mesh.updateMatrixWorld(true)
      const worldTop = top.clone().applyMatrix4(mesh.matrixWorld)
      const worldOrigin = top.clone().add(new THREE.Vector3(0, 20, 0)).applyMatrix4(mesh.matrixWorld)
      const direction = worldTop.clone().sub(worldOrigin).normalize()
      const distance = worldTop.distanceTo(worldOrigin)
      assert.equal(new THREE.Raycaster(worldOrigin, direction, 0, distance + 0.01).intersectObject(mesh)[0]?.instanceId, target)
      assert.equal(new THREE.Raycaster(worldOrigin, direction, 0, distance - 0.01).intersectObject(mesh).length, 0)
      assert.equal(new THREE.Raycaster(worldOrigin, direction, distance + 0.1).intersectObject(mesh).length, 0)
      const gap = new THREE.Vector3(x + layout.pitch / 2, 10, z).applyMatrix4(mesh.matrixWorld)
      const down = new THREE.Vector3(0, -1, 0).transformDirection(mesh.matrixWorld)
      assert.equal(new THREE.Raycaster(gap, down).intersectObject(mesh).length, 0)
    } finally {
      mesh.geometry.dispose()
      ;(mesh.material as THREE.Material).dispose()
    }
  }
})
