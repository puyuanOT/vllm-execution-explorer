# Independent scene correctness review

**Verdict: no significant unresolved issues found after the final repairs.**

Reviewed the Cursor redesign at `ca9a516` and the final working-tree repairs on `codex/control-plane-review` against `25432ac`. This reviewer did not author the redesign or these repairs. Review concentrated on architecture/scene contracts and followed their integration through replay, selection, navigation, playback, telemetry and evidence labeling. Source and capture data were read-only; this report is the only file written by the reviewer.

## Reassessment of the confirmed picking defect

The original constant-plane picker selected a neighboring physical ID when the pointer hit a variable-height column. The initial independent reproduction selected fixture block 3 but returned block 6; using the real 4,410-block selected height returned block 797 for visible block 300. The repaired `createDieRaycast` traverses the floor grid and intersects each candidate's actual instance bounds. `CacheStage` refreshes the aggregate bounding box after updating transforms. Padding, gaps, world transforms and distance limits remain respected.

The new oblique-ray regression exercises the actual rendered dimensions at capacities 8, 4,410 and 35,000. An additional independent, deterministic probe compared the repaired picker against native `THREE.InstancedMesh.prototype.raycast` for **640 rays** over capacities 8, 137, 4,410 and 35,000, varied heights, camera directions, translated/rotated/nonuniformly scaled meshes and near/far clipping. There were **zero identity or hit-distance mismatches**. The preserved direct browser-click receipt is consistent with this result. The original finding is closed.

All physical blocks remain instantiated; the bounded outline decoration does not cap addressability. Native physical-ID forms retain equivalent keyboard access. Decorative outlines do not raycast, selection handlers preserve drag/modifier guards, and selecting a reused token keeps current binding distinct from original content provenance. Demand rendering and camera-control cleanup contain no newly identified persistent render-loop or resource-ownership defect. These checks establish correctness of the reviewed paths, not frame-rate performance.

## Actual capture and evidence checks

- Independently verified the compressed archive's size/digest and every one of its **11 indexed files** against `artifact-index.json`; the archive has no unindexed file members.
- Recomputed the current 38-file runner/instrumentation identity: `c48b0fd1c8eccdc0b0d1d1d37e08b02f854a6e8bfd13531f81bf116bc524704d`. It matches both successful A100 captures. The frontend repair does not change distributed capture source.
- Replayed the actual prefix capture through current replay and slot helpers. At event **304**, block **1**, slot **0** selects the second request at position 0 while retaining the first request as content source. All **455** physical IDs, including **454**, remain projected.
- Parsed the actual native attachment through the production parser/projection. It retains four mapped batches, **64 real rows** in the first batch, the independent trace-local microsecond clock, and `qualification: unverified`. The model scene's batch/layer wording and tensor/token/head limitations agree with these contracts. Host replay is not converted to profiler time.
- Checked the four completed request streams in the two successful recordings: each has eight accepted output tokens, and its sampled final output remains `not_computed` KV in the current token projection.
- Reassessed the final shared evidence classifier against the actual manifests. Diagnostic and prefix captures classify as GPU evidence; rejected preparation `run-14001494e084` classifies as unverified. Provisional GPU intent remains unverified, and fixture provenance takes precedence. Header and workspace use the same result. The final failed-import screenshot shows the amber classification and disabled playback.

The rejected-budget report correctly remains a failed successful-GPU qualification with no events/admissions; it is not counted as a successful capture. The README separates diagnostic/synthetic count checks from semantic answer quality, retains bounded output limits and the profiler qualification caveat, and makes no numerical-tensor, individual-token/head timing or frame-rate claim. Both responsive draft actions now say `Undo deletion`.

## Validation and limits

Independently ran:

```text
node --experimental-strip-types --test web/src/scenario/architectureScene.test.ts web/src/scenario/dieLayout.test.ts web/src/scenario/redesignRegressions.test.ts replay/src/profiling.test.ts
```

**31 tests passed, zero failed.** The tests include the final GPU-provenance classification, trace-gap exclusion, native clock/row contracts, slot provenance and oblique picking. `git diff --check` passed. The retained integrated receipts show 214 Python, 51 replay and the final **68 web** tests passing, plus the TypeScript/Vite production build.

Inspected static mobile KV, trace-gap, actual A100 profile/prefix and final failed-provenance screenshots, and the completed qualification README/audits. This reviewer did not drive a browser, launch a GPU run, contact the remote host or invoke LangSmith. Responsive interaction verification and actual runtime execution are supported by the preserved integrated evaluation; this review independently checks their source/data contracts and receipt consistency. No broader performance or semantic-model qualification is inferred.
