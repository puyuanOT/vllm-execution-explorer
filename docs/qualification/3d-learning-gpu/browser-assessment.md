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
