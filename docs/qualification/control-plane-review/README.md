# Control Plane redesign review

Reviewed `cursor/control-plane-redesign-4549` at `ca9a516`, preserving the visual direction. Repairs live on `codex/control-plane-review`. The main checkout and the user's Cursor branch were left unchanged.

## Repairs

| Reproduction | Repair and verification |
| --- | --- |
| Angled clicks on the raised KV column selected a different physical block (fixture block 3 → 6). | Traverse floor cells and intersect actual instance bounds. Regression rays cover 8, 4,410 and 35,000 blocks, low/free and raised columns, three camera directions, transforms, gaps, padding and distance limits. A direct browser click on the raised block retained block 3. |
| Raw scheduler/completion events after a trace gap collapsed onto the scope's final column and could be stepped into. | Scope, timeline and playback use the replayable prefix; seeking is bounded. The imported gap-plus-tail fixture stops at event 7, shows one trusted step, and never displays the later completion or scheduler step. |
| Short arrival plans produced seconds labels such as `0.2.5 s`. | Generate whole-millisecond axis ticks; round-trip checks cover 0–120,000 ms. Browser seconds labels are `0.003`, `0.006`, `0.009`; an editable `0.002 s` becomes `2 ms`. |
| The request-lane autoscroll could place the current mobile column behind sticky labels. | Bound trailing space by the available track width. Forward/backward regression checks cover 294–1,000 px regions. At a 320 px viewport S2 remains visible; its inspector closes with Escape and restores focus. |
| Fixed workload columns/suffix padding clipped `512` to `51` and compressed four-digit output values. The arrival-batch grid could overflow a phone. | Use the actual editor width for the dense row layout; narrow rows keep visible labels and adequate numeric fields. Constrain the batch grid to a shrinking track. Check `512` and `4096`, both row layouts, and 320/390 px page bounds. |
| Mobile arrival graphics hid later labels and crowded ticks. | Maintain a readable 480 px chart inside an accessible horizontal scroller. Keyboard scrolling moves its scroll position; the page itself stays within its viewport. |
| The mobile KV cartridge's close button extended past the screen (x=326–350 at width 320). | Use a two-row grid header. The repaired card has no horizontal content overflow and its close button fits and works. |
| The new generic `Undo` label suggested edit history, but the existing operation restores a deletion snapshot. | Restore the accurate `Undo deletion` label. The operation itself is unchanged. |
| A failed GPU preparation still received the green `Observed GPU recording` badge because requested execution type was treated as observed provenance. | Share one classification for header and scene: GPU evidence needs explicit `real_gpu: true` provenance and recorded GPU hardware. Provisional preparation and failed initialization remain unverified. Importing the retained failed A100 recording shows the unverified badge, no observed badge and disabled playback; the successful capture retains its observed badge. |

The preliminary header `sr-only` suspicion was withdrawn after inspecting compiled CSS: responsive restoration rules are emitted later and the header works. No header CSS change was made.

## Actual A100 execution

The supplied `ubuntu@144.24.39.189` host was initially clean. The UI's automatic setup successfully installed an isolated vLLM 0.30.0 runtime and opened the loopback tunnel. A second check reused the prepared runtime. The key remained on the laptop and is excluded from all evidence.

Hardware: NVIDIA A100-SXM4-40GB, UUID `GPU-8898e671-6fc2-2209-4f79-f39593c29ac6`, driver 580.105.08. Isolated runtime: Python 3.12.3, torch 2.13.0+cu130, vLLM 0.30.0. Model and tokenizer: `Qwen/Qwen3-4B-Instruct-2507`, immutable revision `cdbee75f17c01a7cc42f958dc650907174af0554`.

| Case | Assessment |
| --- | --- |
| Diagnostic chunked prefill, `run-5328256bf349` | Pass. UI-authored 512/16-token inputs were verified on the host. Both requested and accepted 8 outputs and terminated by length. 16 scheduler submissions/results; a 64-token iteration cap; all 113 physical blocks captured. The bounded profile has four packed-row steps, all 36 layers and 2,136 correlated CUDA activities. Raw profiles and outputs are retained. |
| Ordinary prefix run with undersized test budget, `run-14001494e084` | Rejected during preparation, as expected for the effective 4096-context/256-MiB combination. vLLM requires about 0.56 GiB. The UI displayed the error and allowed correction. No requests were admitted and no outputs were produced. The failed recording and controller error are retained rather than counted as GPU qualification. |
| Ordinary prefix run after correction, `run-d0ec34078edc` | Pass. 1 GiB budget, 455 physical blocks, V2 serving path, profiling off. Both 277/278-token prompts and their eight-token output caps match the submitted plan. The second request reused 256 prefix tokens; its prefill begins at position 256. Slot 0 selected the second request while retaining the first request as content source after the new binding was observed. The last physical block (454) is addressable independently. |

The two successful captures' 38-file distributed-source fingerprints equal `c48b0fd1c8eccdc0b0d1d1d37e08b02f854a6e8bfd13531f81bf116bc524704d`, matching the unmodified runner/instrumentation source at `ca9a516`. Their declared Git revision is null; the fingerprint establishes capture-source parity. The repair changes frontend behavior only.

Inputs, raw outputs, normalized profiles and original compressed native traces for all three attempts are in [a100-captures.tar.gz](a100-captures.tar.gz), with per-file sizes/SHA-256 values in [artifact-index.json](artifact-index.json). The existing read-only analyzer was invoked against `ca9a516`: [diagnostic assessment](native-audit.json), [prefix assessment](prefix-audit.json), and [rejected-budget assessment](rejected-budget-audit.json). The latter intentionally fails successful-GPU qualification because initialization never completed.

The synthetic outputs are count/position tests, not semantic answers. The prefix answers are deliberately bounded at eight tokens; both stop by length. Full text and token IDs were inspected. Finalized workers close correctly: the final GPU receipt shows 4 MiB and no compute applications. LangSmith was not used, per the user's preference; complete local evidence is retained.

## Browser and checks

Production build served on the supported `127.0.0.1:5173` origin. An initial alternate-port preview hit the existing CORS origin restriction; qualification continued on the supported origin without changing server policy. Source editing also reset the development page's transient key/check state; the final capture journeys used the production preview.

Direct UI checks covered automatic SSH setup/reuse, settings summaries, actual preparation/success/failure/recovery, invalid output submission and field focus, learning-example confirmation, request duplication/deletion, input-mode changes, request movement, deletion restoration, time-unit conversion, recording import, trace-gap stop, scheduler seeking, 0.001× speed selection, physical block lookup/clicks, reused token selection, inspector Escape/focus restoration, camera pan/reset, and separate host/profiler clocks. The diagnostic UI retained host event 610 while advancing the profiler cursor into a mapped batch with 64 observed packed input rows. Its `qualification: unverified` badge is preserved; batch/layer measurements do not establish individual-token/head duration or tensor values.

Responsive checks include 320×844, 390×844, 1280×540, the default 1280×720, and a wide editor. Browser validation establishes visible controls/state transitions, not a frame-rate benchmark or numerical tensor qualification.

Required checks: **214 Python tests, 51 replay tests, 68 web tests, Python compile check and TypeScript/Vite production build pass**. [Web tests](web-tests.txt), [build](web-build.txt), [Python tests](python-tests.txt), [replay tests](replay-tests.txt). The failed picking regression is retained in [baseline-regression.txt](baseline-regression.txt).

`npm audit` reports six existing dependency findings. All affected package versions are identical to the implementation base (`25432ac`); the redesign only added static font packages. The affected paths are build tooling for repository-owned CSS/globs. No dependency changes were folded into this UI review.

## Visual receipts

- [Laptop workload values](laptop-inputs.png)
- [Mobile four-digit workload values](mobile-inputs.png)
- [Mobile deletion action wording](mobile-undo-label.png)
- [Mobile current lane and restored focus](mobile-lanes.png)
- [Mobile arrival chart](mobile-arrivals.png)
- [Mobile KV cartridge](mobile-kv.png)
- [Trace gap with retained raw tail](gap-tail.png)
- [Actual A100 native profile](a100-profile.png)
- [Actual A100 prefix block and slot origin](a100-prefix-kv.png)
- [Preparation-budget rejection](kv-budget-rejection.png)
- [Failed capture with unverified GPU provenance](failed-gpu-provenance.png)

Three independent final reviewers found no significant unresolved issues after implementation and integrated evaluation: [scene correctness](reviews/scene-correctness.md), [replay integration and evidence](reviews/replay-integration.md), and [forms and accessibility](reviews/forms-accessibility.md). Their final pass discovered the negative-provenance badge issue and the narrow Undo wording inconsistency; both were repaired, verified in the browser, and reassessed before the final verdicts.
