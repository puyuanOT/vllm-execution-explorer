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
