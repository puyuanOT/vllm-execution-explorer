# Browser observations on actual GPU recordings

The production preview used an SSH-forwarded isolated controller on the provided
A10 host. The initial imported serving recording was `run-225baaef2a99` at
`7da8f31`; the initial imported native profile was `run-c78d2dc1dcf3` at `9fce070`.
These observations establish browser behavior on real retained data, independently
of the later repeated-capture lifetime qualification.

| Journey | Observed outcome |
| --- | --- |
| Full physical inventory | Opened the actual 4,477-block pool; block 4,476 is addressable, available, generation zero and has zero computed positions at the initial cursor. The reserved null block is included. |
| Camera and navigation | Pan/orbit buttons and dragging operated on the actual pool. Repeated Overview → Model → Overview → KV transitions remained usable. Model controls expose all 36 layers. |
| Native attachment import | Imported manifest, host events, token content and declared normalized profile together. Measured mode became available without generating host events. |
| Profiler cursor | Native Home and End moved the profiler to 0 and approximately 353.169 ms; host replay remained at event zero. Out-of-scope native time showed no invented mapped batch. |
| Layer and packed-row navigation | Batch step-1 exposes 37 post-reorder rows. Row 19 selects profile-a / position 0; the 3D attention view shows 32 query heads sharing eight KV heads and explicit dimension 128. At native 16.080 ms the inspector displays correlated device activity alongside separate CPU scope spans. |
| Slow playback | Recorded time and measured profile both offer and accept 0.001×. Host intervals, native activity intervals and explanatory layer time retain separate labels. |
| Preparation failure recovery | A second capture before the lifetime repair displayed the actual insufficient-free-memory error and restored the Prepare and capture control. [Screenshot](screenshots/repeat-run-before-cleanup.png). |

[Full pool screenshot](screenshots/actual-full-pool.png) and
[native attention screenshot](screenshots/actual-native-profile.png) retain the
visible evidence. Earlier synthetic/compact interaction checks remain in the
[local acceptance report](../../acceptance/3d-learning-local/README.md).

## CPU submission and resource observations

An opt-in build (`VITE_EXPLORER_SCENE_PROFILE=1`) observed existing demand frames;
it did not schedule frames or force rendering. The retained
[console observations](scene-cpu-resource-observations.json) identify the source
recording, focus, host sequence, viewport, drawing buffer and DPR for each burst.
Within the 4,477-block focused view, steady counters were 14 draw calls,
53,800 triangles, seven geometries, eight textures and three programs. Repeated
model/pool transitions disposed transient resources and returned to those pool
counters. CPU submission during steady pool observations was approximately
0.1–0.4 ms; transition frames were separately retained and could be slower.

These are synchronous `gl.render` CPU submission observations and Three.js
counters. They exclude React commits, GPU completion and browser presentation.
Demand-loop idle intervals and background browser throttling prevent deriving
FPS from these intervals. The resource counters are not GPU byte measurements
or proof of complete resource ownership. The final ordinary production build
contains no scene diagnostic logger. The existing bundle-size warning remains.

Native device activity in the model recording measures the instrumented eager
workload on the remote GPU; it does not measure browser rendering performance.
No individual token/head duration, numerical tensor readback or alignment of
host and profiler clocks is claimed.

## Final deployed source and configuration spacing

The final production preview imported `run-5347dd4404a4` from exact capture
source `75f2ccd3a13b585c576f2597449a23dcad670921`. Its native clock spans
337.937 ms. Home and End operate on that clock; host replay remains at event zero.
Step 1 exposes 55 actual post-reorder input rows, with no padding in this observed
batch: profile-a occupies rows 0–26, profile-b rows 27–54. Selecting row 0 maps
profile-a / position 0. Native 16.943 ms intersects an actually correlated layer-0
attention CUDA interval; the model displays the active layer and packed inputs.
[Final native view](screenshots/final-native-profile.png) preserves that state.
The UI continues to label capture qualification unverified because independent
qualification reports are external evidence, rather than a fabricated manifest badge.

The [spacing receipt](form-spacing.json) measures Repository, Precision, both
revision fields and SSH port on the production UI. Controls have a 6 px top
margin and a 7 px gap below the rendered label text. Optional revisions have no
required attribute. [Model fields](screenshots/configuration-revisions.png) and
[SSH fields](screenshots/ssh-form-spacing.png) preserve the layout. Remote host
selection was used only to inspect that form and restored to Local machine;
this does not claim an automatic SSH bootstrap run. The actual bounded SSH
reachability probe is retained separately in [ssh-reachability.json](ssh-reachability.json).

An imported finalized cancellation capture retained all 82 host events, three
returned results and zero accepted outputs. Its profile stays explicitly partial
(three batches, 145.476 ms native extent); importing it reset native time to zero.
Inspection during another capture exposed a misleading Follow live control for
that unrelated run. Both the control and automatic follow effect now require the
currently displayed capture to be live. Rebuilt production UI and all 52 web
checks pass after that change; the imported cancellation recording shows Replay
at its host tail without a live control.

The actual UI initiated `run-eaa2e906f9bb` with model revision, tokenizer revision
and tokenizer repository left blank. It returned automatically to Explore after
the owned execution process exited successfully. Both revisions resolved to
`cdbee75f17c01a7cc42f958dc650907174af0554`; all six requests completed with length
terminals. This is an actual optional-revision check on the pre-STOP-fix source,
retained separately from the final source qualification.

## Qualified final-source browser journeys

Visualization implementation `8f0ca92d2ee2741723ba20e5637decb91982694b` completed the
actual browser-created capture `run-1021e5882f0d`. All three override inputs were
blank and optional; preparation resolved `main` to the canonical model/tokenizer
commit. The UI opened Explore automatically after worker 16545 exited with code
zero. The recording has 315 events and six completed eight-output requests.
The post-close NVIDIA receipt reports 3 MiB and no compute applications.

This initialized pool contains **5,414 physical blocks**, including the null
block. At event 27, after result event 26, block 1 / generation 1 has one reference
and five of 16 computed positions. The readable slots map sample-2 prompt
positions 0–4 to their actual retained token IDs/fragments; the remaining slots
stay unobserved. Block 5,413 is addressable, generation zero, zero references,
available, and has no computed positions. Pan and reset controls preserve block
selection. [Final pool screenshot](screenshots/final-kv-pool.png) shows both the
complete grid and computed token slots. Capacity differs from the older capture
because each initialized engine records its actual configured pool.

The final native recording `run-b57709a84086` imports with its declared normalized
attachment. Step 1 has 55 rows, now profile-b positions 0–27 followed by profile-a
positions 0–26; this is the runner's actual post-reorder order, rather than a
hardcoded scheduler order. At native 16.154 ms the selected profile-b / position
0 participates in step 1 and an actually correlated layer-0 attention interval
is active. Attention drilldown exposes 32 query heads sharing eight KV heads,
explicit dimension 128, and separate CPU/device measurements. Host replay stays
at event zero. [Final attention screenshot](screenshots/qualified-native-attention.png)
retains that state at 0.001×; token identity and batch participation do not establish
individual token/head latency or numerical tensor values.

The live-follow repair was also exercised while unrelated capture
`run-9ff4ff7f29ba` was preparing/capturing. The browser retained the finalized
`run-eaa2e906f9bb` recording and exposed neither Follow live nor Reconnect live.
The paired controller state, complete browser accessibility state and control
counts are retained with the final evidence archive.

## Final lifecycle delta

Revision `dcab49f6ff0e194711c031f4d72399b93f300877` changes only producer process synchronization and its regression tests. The frontend and replay are unchanged; the browser observations above remain tied to their original recording/source checkpoints. Four fresh actual A10 lifecycle cases and their exact source identity are retained separately in the [qualification report](README.md#final-lifecycle-repair-and-repeated-gpu-checks).
