# Final independent review: replay integration and evidence

**Verdict: no unresolved significant issues found in the reviewed scope.**

This is a final, post-evaluation review by a reviewer who did not author the redesign or repairs. Reviewed `codex/control-plane-review`, with `ca9a5160d69dcbea6b076d1dc445a450fbc9a84b` plus the completed working-tree repairs, against `25432ac`. Source remained read-only during this review; this report is the reviewer's only file change.

## Scope and closed findings

Reviewed the redesign and repair diffs with particular attention to App playback integration, ExecutionWorkspace, PlaybackTransport, RequestSequencer, scope/telemetry, SelectionInspector, LearningDetails, MeasuredProfileDetails, recording import/classification, responsive dialogs, configuration integration, and the small repair helpers. The visual direction is retained. No broad cosmetic rewrite or unmeasured optimization is needed for this scope.

The two significant issues from this reviewer's preliminary audit of `ca9a516` are closed:

1. **Trace-gap tail promoted into played activity.** Originally, App normalized all raw events and scheduler markers against the reducer's shorter `maxSequence`. Events after a gap collapsed onto the final column, including an erroneous current-channel highlight and scheduler marker. `replayableEvents` now supplies the same bounded prefix to the timeline, scope, event playback, keyboard navigation, and transport navigation; `seek` is also bounded. Directly replaying the retained nine-event gap-tail fixture with current production helpers stops at event 7, exposes only `step-1`, returns event 7 when asked to advance, draws no output tick for the tail completion, and has no current-channel highlight at the gap. The retained browser receipt shows the same event-7 endpoint and unknown-state warning.
2. **Mobile autoscroll concealed the current step.** Originally, reserving three trailing columns could put the current column entirely beneath the 220-pixel sticky label. The `sequencerScrollLeft` helper bounds trailing space by actual available track width. Forward/backward regressions cover narrow and wide regions with both column widths. The 320-pixel browser receipt visibly retains S2 and its decode positions beside the labels.

Also independently reassessed the final provenance repair discovered by another reviewer. `recordingEvidence` supplies both kind and text to the header/workspace, prioritizes fixture labeling, and requires explicit real-GPU provenance plus recorded hardware before claiming observed GPU evidence. Direct checks against all three archived manifests classify the rejected preparation as unverified and both successful captures as GPU evidence. Provisional live preparation is unverified. The final failed-import screenshot displays the unverified header and scene labels with disabled playback; the successful prefix import retains its observed label.

## Independent checks and retained evidence

- Independently ran **68 web tests and 51 replay tests: all passed**. Inspected the author's final **214 Python-test** and production-build receipts. `git diff --check` passed.
- Verified the archive's size/SHA-256 and all **11 indexed members** against `artifact-index.json`. Read raw manifests, events, content, normalized profiling data, and the original compressed native trace in memory. No archive extraction or evidence alteration was required.
- Recomputed the current distributed runner/instrumentation fingerprint: **38 files**, `c48b0fd1c8eccdc0b0d1d1d37e08b02f854a6e8bfd13531f81bf116bc524704d`. Both successful manifests match it. Their declared Git revision is null, accurately disclosed in the qualification README.
- Invoked the current reducer, content attachment, KV/token projection, and profile parser/projection directly on retained artifacts. The diagnostic run has 610 events, 113 physical blocks, verified 512/16-token inputs, eight accepted outputs for each request, and length termination. Its four measured steps each retain 64 actual packed rows at `GPUModelRunner._prepare_inputs.returned`; layer summaries cover all 36 layers. The profile preserves its separate clock and `qualification: unverified` status.
- The ordinary run has 515 events, 455 physical blocks, 277/278-token prompts, and eight accepted outputs per request, both length-terminated. The second request records a 256-token prefix match. At binding event 304, its token 0 projects to computed KV in block 1 while retaining the first request as content source. This matches the refreshed browser cartridge/inspector receipt.
- The rejected-budget attempt has zero events, admissions, and outputs and no GPU hardware in its manifest. Its strict GPU qualification failure is retained instead of counted as success. Inspected worker finalization records with exit code 0 and the final 4-MiB GPU receipt.
- Inspected the retained mobile lane, gap-tail, native-profile, prefix-cache, and failed-provenance screenshots alongside the documented direct browser journeys. They support visible state/control claims, including host event 610 remaining independent of the profiler cursor.

## Limits

This reviewer performed no new browser or GPU actions; the author performed the integrated journeys, and this review inspected their retained receipts and independently exercised the production replay code. Screenshots alone do not establish keyboard interactions, so those claims are limited to the documented author journeys. No frame-rate benchmark, numerical tensor validation, or individual-token/head duration qualification is claimed. The outputs establish bounded inference/count/position behavior, not answer quality. LangSmith was waived by the user. The existing dependency findings and bundle-size warning do not originate in these repairs and are disclosed in the main report.

No further significant correctness, evidence-truthfulness, integration, or maintainability finding was demonstrated in this final pass.
