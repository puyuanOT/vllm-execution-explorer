import assert from 'node:assert/strict'
import { readFileSync, writeFileSync } from 'node:fs'
import { createPlayer, parseJsonl } from '../../../replay/src/reduce.ts'
import { buildTimeline } from '../../../replay/src/timeline.ts'
import { observedRequestIds } from '../../../web/src/explore/model.ts'
import { requestColor } from '../../../web/src/theme.ts'
const evidence = new URL('../../../output/playwright/oct13-intake-requests/', import.meta.url)
const source = new URL('../../../web/public/recordings/gpu-h100-bf16/', import.meta.url)
const json = url => JSON.parse(readFileSync(url,'utf8'))
const manifest = json(new URL('manifest.json', source))
const events = parseJsonl(readFileSync(new URL('events.jsonl', source),'utf8'))
const player = createPlayer(events, manifest)
const order = buildTimeline(events, manifest).roster.samples.map(sample => sample.id)
const cases = json(new URL('browser-cases.json', evidence))
const rgb = hex => `rgb(${[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)).join(', ')})`
const outcomes = cases.map(item => {
  const observed = observedRequestIds(player.at(item.sequence), order)
  assert.deepEqual(item.requests.map(r=>r.label), observed.map(id=>`Request ${id.split('-')[1]}`))
  assert.deepEqual(item.requests.map(r=>r.color), observed.map(id=>rgb(requestColor(order,id))))
  assert.deepEqual(item.viewport,{width:1920,height:1080})
  assert.equal(item.removedMessagePresent,false)
  assert.deepEqual(item.canvas,cases[0].canvas)
  assert.deepEqual(item.layer1,cases[0].layer1)
  assert.deepEqual(item.bar,cases[0].bar)
  assert.equal(item.bar.height,144)
  assert.ok(item.scene.includes(`${String(observed.length).padStart(2,'0')}requests observed`))
  return {sequence:item.sequence,requests:observed,outcome:'pass',canvas:item.canvas}
})
const native = json(new URL('gpu-clock.json', evidence))
for (const item of native) {
  assert.equal(item.intake,'Request 1')
  assert.ok(item.header.includes('Intake · host replay'))
  assert.ok(!item.participants.includes('Request 1'))
  for (const request of ['Request 2','Request 3','Request 4']) assert.ok(item.participants.includes(request))
  assert.deepEqual(item.canvas,cases[0].canvas)
}
const result = {recording:'gpu-h100-bf16',outcomes,nativeClock:'pass: host Intake remains Request 1 at EVT174; native iteration 20 has Requests 3, 2, 4',unchangedCanvas:cases[0].canvas,unchangedLayer1Label:cases[0].layer1}
writeFileSync(new URL('assessment.json',evidence),JSON.stringify(result,null,2)+'\n')
console.log(JSON.stringify(result,null,2))
