import assert from 'node:assert/strict'
import { readFileSync, writeFileSync } from 'node:fs'
import { createPlayer, parseJsonl } from '../../../replay/src/reduce.ts'
import { buildTimeline } from '../../../replay/src/timeline.ts'
import { modelIterationSummary } from '../../../web/src/architecture/model.ts'
import { observedRequestIds } from '../../../web/src/explore/model.ts'
const base = new URL('../../../output/playwright/oct13-legend-disclosure/', import.meta.url)
const source = new URL('../../../web/public/recordings/gpu-h100-bf16/', import.meta.url)
const json=url=>JSON.parse(readFileSync(url,'utf8'))
const manifest=json(new URL('manifest.json',source))
const events=parseJsonl(readFileSync(new URL('events.jsonl',source),'utf8'))
const player=createPlayer(events,manifest)
const order=buildTimeline(events,manifest).roster.samples.map(r=>r.id)
const cases=json(new URL('host-cases.json',base))
for(const item of cases.collapsed){
  assert.equal(item.open,'false')
  assert.equal(item.participantListPresent,false)
  assert.deepEqual(item.viewport,{width:1920,height:1080})
  assert.deepEqual(item.bar,cases.collapsed[0].bar)
  assert.deepEqual(item.canvas,cases.collapsed[0].canvas)
  assert.deepEqual(item.requests,observedRequestIds(player.at(item.sequence),order).map(id=>`Request ${id.split('-')[1]}`))
}
for(const item of cases.expanded){
  assert.equal(item.open,'true'); assert.equal(item.controlsPanel,true)
  assert.deepEqual(item.bar,cases.expanded[0].bar)
  assert.deepEqual(item.canvas,cases.expanded[0].canvas)
  const summary=modelIterationSummary(player.at(item.sequence))
  for(const r of summary.participants) assert.ok(item.participants.includes(`Prefill inputs: ${r.prefill} · Decode inputs: ${r.decode}`))
}
const native=json(new URL('native-clock.json',base))
assert.equal(native.intake,'Request 1')
assert.ok(native.header.includes('Intake · host replay'))
assert.ok(!native.participants.includes('Request 1'))
for(const id of [2,3,4])assert.ok(native.participants.includes(`Request ${id}`))
const limits=manifest.workload_plan.batches.flatMap(b=>b.samples)
assert.ok(limits.every(r=>r.max_output_tokens===6))
const final=player.at(player.maxSequence)
assert.ok(limits.every(r=>final.requests[r.id].acceptedTokenCount===6))
const result={hostCollapsed:cases.collapsed.length,hostExpanded:cases.expanded.length,nativeIsolation:'pass',outputLimit:6,eachRequestAcceptedOutput:6,collapsedCanvas:cases.collapsed[0].canvas,expandedCanvas:cases.expanded[0].canvas,outcome:'pass'}
writeFileSync(new URL('assessment.json',base),JSON.stringify(result,null,2)+'\n')
console.log(JSON.stringify(result,null,2))
