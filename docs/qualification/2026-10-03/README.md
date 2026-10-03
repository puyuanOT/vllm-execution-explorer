# New-laptop local qualification — 2026-10-03

This is a local setup and browser-verification checkpoint. All new captures
below used the synthetic fixture. The imported GPU recording is the preserved
September acceptance bundle. A user-provided A10 host was connected and its
CUDA/vLLM preflight verified; no new model execution capture occurred.

## Checkout and environment

- Primary checkout: `/Users/pliu/Documents/vllm-execution-explorer`.
- Verification worktree: `/Users/pliu/Documents/vllm-execution-explorer-qualification`,
  branch `codex/laptop-qualification`.
- Recovered main: `4c8e8198219eaf22b0644a38d96b008645dec485`.
- Retained newer work: `72fc1cbf8e9b75f8b8cc82d233508f6025f0a55a`
  (the existing fixture host-check PR).
- Verified functional revision: `23aece686c387e822adbc4f36021f670ec2f3ad3`.
- macOS 26.7.1, arm64; Python 3.11.15; Node 22.14.0; npm 10.9.2;
  uv 0.12.22. Git, GitHub CLI, SSH, rsync, and Command Line Tools are available.
- Repository visibility remains private. GitHub authentication and repository
  author identity were already configured.

Python was installed through uv. Its initial bundled-certificate download
failed with `UnknownIssuer`; selecting the existing macOS certificate store
resolved it. Node came from the [official 22.14.0 archive](https://nodejs.org/en/download/archive/v22.14.0),
and its tarball matched the vendor's SHA-256 file. Node/npm/npx links are in the
existing `~/.local/bin` PATH. npm's initial install failed certificate validation;
its user-level `cafile` now uses public certificates exported from the macOS
System keychain. TLS verification remains enabled. No Homebrew, CUDA, PyTorch,
or vLLM installation was needed locally.

Both checkouts received seeded `.venv` environments with CI's exact five Python
dependency pins and editable packaging, plus `npm ci --prefix web`. No lockfiles
or tracked lessons were regenerated.

Controller: `http://127.0.0.1:8765`; Vite: `http://127.0.0.1:5173`, started with
`--strictPort`; `/api/health` returned `{"ok":true}`. The UI endpoint was blank,
using Vite's local API proxy. Services used the qualification worktree.

## Automated checks

| Check | Recovered main | Functional revision |
| --- | --- | --- |
| Python pytest | 103 passed | 107 passed |
| Replay Node tests | 23 passed | 23 passed |
| Web Node tests | 20 passed | 34 passed |
| Python compileall | Passed | Passed |
| TypeScript + production build | Passed | Passed |
| Whitespace check | Passed | Passed |
| Preserved GPU integrity | 8/8 passed | See backend results |

Node's type-stripping experimental notice, a Starlette/httpx deprecation notice,
and the known Vite chunk-size warning were non-failing. The final bundle was
approximately 1.388 MB before gzip. npm audit reported six high-severity entries
in the existing PostCSS/Tailwind build-tool dependency tree. Dependency upgrades
were not folded into this CI-parity bootstrap; the lockfile is unchanged.

## Browser cases and assessed outcomes

These cases used the actual React UI and local FastAPI controller through the
Codex in-app browser. Locator failures caused by native AX/DOM role differences
or a dialog closing asynchronously were corrected by inspecting fresh state;
they are not application failures.

| Case | Observed result |
| --- | --- |
| Initial draft | Exactly three batches, 2/3/1 requests, at 0/100/120 ms; advanced sections collapsed; no automatic capture. |
| Draft editing | Request label and prompt changes survived reload; batch label/time edits, request/batch duplication, equal-time batches, moving a stable request ID, request/batch deletion and undo, and new batch/request authoring worked. |
| Numeric validation | Output limit 0 produced an inline positive-integer error. Sub-millisecond seconds produced a precise-time error. Correction removed the error and retained expanded fields. |
| Raw-value submission regression | Before the final fix, raw batch time `abc` launched `run-8a05a5982e46` using the old 0 ms offset. This was a failed user-intent case despite passing recording integrity. After the fix, `abc` and blank times, blank and zero output caps, sub-millisecond seconds, and a blank active synthetic length each blocked capture. An unrelated label edit clearing displayed issues did not bypass validation. |
| Disclosure regression | Before the fix, correcting output 0 to 12 collapsed request and batch and swallowed the next input focus. After the fix, the same correction and a second 0-to-8 case retained both `open` states and focused the next prompt field. Manual disclosure remains available. |
| Input Enter | Invalid and corrected numeric values validated/committed without capture. Enter in the batch-label input did not capture. Enter on the focused Prepare and capture button explicitly started capture. |
| File chooser Enter | Initial broad Enter suppression prevented keyboard activation of the SSH key chooser. Exempting native action inputs restored the chooser; the final check opened and dismissed it without choosing a private key. |
| Choice input Enter | Review then exposed an implicit capture from Enter on the ms radio (`run-b6113e1b1816`, a failed user-intent case). The final guard suppresses Enter for non-action inputs. Retesting both time-unit radios, both host radios, and automatic-setup checkbox kept Configure idle. Space still selected/toggled choices; file chooser Enter still opened; explicit capture-button Enter started the valid default run `run-8c190859dd4c`. |
| Advanced validation | Top-p 2 was rejected; correcting it to 0.9 kept Memory and sampling and its containing section open. Temperature, seed and multiline stop strings edited successfully, then were restored. |
| Repeated hidden errors | Before the final fix, closing both sections after a top-p error and resubmitting left the error hidden. After the fix, repeating top-p 2 with both ancestors manually closed reopened them and focused `setting-top-p`. |
| Adapter draft | Adding/naming an adapter, selecting a batch default, and confirming removal restored base routing and removed its controls. This is authoring verification, not LoRA execution acceptance. |
| Host checks | Live local mode reported missing CUDA. Demo fixture was Ready while explicitly disclaiming CUDA/vLLM certification. An unused loopback controller port produced failed setup stages; returning to the blank endpoint recovered. |
| Version and field spacing | Header explicitly identifies vLLM 0.30.0. Legacy global grid labels had stretched beside taller hints, enlarging internal label/control gaps. Block labels/controls and top-aligned field grids restore the standard 6 px margin, including SSH port, Repository and Precision. |
| Actual SSH host check | The user supplied a current host and key. Its initial automatic setup succeeded after approximately 158 seconds while the old UI misleadingly kept SSH reachability active. With real phase reporting, a UI-driven automatic recheck finished in 13.358 seconds, showing SSH complete during runtime preparation and then Connected. Remote hardware is NVIDIA A10 (23028 MiB), vLLM 0.30.0, Linux, CUDA compute capability 8.6. This verifies connection/runtime/tunnel availability, not workload execution. |
| SSH failure and recovery | Checking the same host on unreachable port 1 produced a bounded SSH timeout, Reach execution host Failed, later phases Pending, and retry controls. The previous target was cleared. Retrying port 22 with automatic setup disabled connected the prepared runtime, without a project-copy phase. No model capture ran. |
| Capture lifecycle | Fixture capture showed progress, locked host/source/endpoint controls, and opened Explore. A long fixture request was cancelled through the UI; cancellation was reported and submission controls returned. |
| Unsupported fixture input | A synthetic-token-length draft was rejected with “The demo fixture cannot verify synthetic token lengths with the selected tokenizer.” Switching back to a prompt allowed capture. Actual tokenizer-length qualification requires GPU access. |
| Prompt integrity | The bracketed large-number probe survived UI capture and appeared unchanged in the request inspector. See backend exact-match boolean. |
| Latest-value capture | Final regression `run-32d16275bf8f` captured authored offsets 15/120/120 ms and first output cap 11 after seconds-mode editing. The immutable configuration retained those values, and Explore showed all six completed requests and 51 accepted tokens. The default example and local fixture mode were restored afterward. |
| Overview/detail views | All five components, inspectors and focused views opened; breadcrumbs/Back to overview preserved event 13. Single/double clicks worked both on component buttons and directly on the 3D scheduler mesh. |
| Request access | Search filtered to input_6; keyboard activation selected its stable request identity and preserved its planned arrival as 1500 ms in the imported GPU evidence. |
| Camera | Orbit left/right, zoom in/out and reset were exercised; reset left event 13 unchanged. |
| Playback | Restart/replay, play/pause, every speed option, recorded events/time, forward/backward seek, previous/next event and scheduler step worked. Quarter-speed recorded-time playback advanced and paused at an observed prefix. Backward seek to zero removed future generated text, accepted counts, prompt attachment and scheduled ranges. |
| GPU-evidence import | Importing manifest+events together showed 270 events, 27 steps, six completions and 48 accepted tokens; missing raw content was explicit. This was replay of previous evidence. |
| Invalid import | Manifest alone produced a missing-events error without replacing the open recording; selecting the full pair recovered. |
| Partial evidence | The Incomplete capture lesson reached gap event 7 and showed “Later state is unknown” / “Unknown observed state,” without future output. |
| Edit as new | Historical recording copied to a separate draft with its 0/100/1500 ms offsets; reset restored the six-request example. |
| Layout/accessibility | Explore at 1280×720 and 1440×900, narrow 768×1024 inspector dialog, keyboard activation, Escape dismissal, visible focus, contained object scrolling, and Reduce motion checked. A 640×360 CSS viewport had no document horizontal overflow and reachable capture controls. It is a compact-layout proxy, not actual 200% zoom certification. |
| Console | Warn/error log inspection returned no captured browser warnings/errors at the inspected checkpoints. |

## Changes and evidence

The recovered host-check fix is retained. New code gives validation-sensitive
native disclosures persistent expansion state, validates current numeric
editor values on submission, and reveals/focuses hidden invalid fields on every
attempt. Enter during input editing commits without implicitly submitting
capture, while file chooser and explicit capture-button activation still work.
The header names the supported version, field grids no longer stretch short
labels, and SSH checks report real copy/runtime/tunnel phases through the
existing status endpoint. Operation-specific timeout errors identify the
failure; Vite's 35-minute proxy limit covers the aggregate bounded setup steps.
Remote hosts are Unchecked until their selected key/connection is verified.
README setup now matches CI and
describes Demo fixture startup; the acceptance README no longer requires a
`PYTHONPATH` workaround.

Backend reports use the actual loader, integrity evaluator and structural
exporter. New raw runs and exported event streams remain in ignored
`recordings/runs/`; tracked results retain only safe configuration/counts and
verification booleans. The original GPU manifest, events, report and checksums
remain unchanged.

- [Backend results](backend-results.json)
- [Integrity reports](backend-results.txt)
- [Local automated check output](automated-checks.txt)
- [Actual SSH host-check results](ssh-check-results.json)
- [Editor after correction](screenshots/editor-correction.jpg)
- [Historical GPU output at 1280×720](screenshots/gpu-evidence-output.jpg)
- [Explore at 1440×900](screenshots/explore-1440.jpg)
- [Narrow inspector](screenshots/explore-768-inspector.jpg)
- [Gapped fixture](screenshots/incomplete-capture.jpg)
- [Compact layout proxy](screenshots/configure-compact-proxy.jpg)
- [Final regression capture](screenshots/final-regression-capture.jpg)
- [SSH runtime progress](screenshots/ssh-runtime-progress.jpg)
- [Connected SSH host](screenshots/ssh-connected.jpg)
- [Expected SSH failure](screenshots/ssh-failure.jpg)
- [Version header and model-field spacing](screenshots/version-and-model-spacing.jpg)

Three independent reviewers reassessed the final functional revision for
correctness, integration/evidence, and simplification. They found no significant
unresolved issues after the raw-value, hidden-error and Enter regressions were
fixed. The separate SSH recovery artifact records 8.927 seconds and passing
remote preflight; new header/spacing checks at 1440×1200 and 768×1024 confirmed
6 px field margins and no narrow document overflow.

## Remaining verification

The user-provided current host and key now pass SSH/runtime/tunnel preflight.
The key was selected for the local controller, remains outside the repository,
and its bytes are absent from logs, artifacts and chat. Its source permissions
were tightened to 0600. No model was downloaded or cloud instance provisioned
by this qualification. The latest late-intake,
cancellation, reconnect and operator-local artifact transfer/retry fixes still
need their affected GPU paths rerun. Gated Hugging Face access, instrumentation
overhead and workload ceilings remain unqualified.

This browser did not expose actual zoom through the tested shortcut. Actual
200% zoom remains unverified. Successful live preflight cannot be reproduced on
this Mac; the real SSH host's successful Linux preflight was exercised instead.
Local live failure, successful fixture, unreachable local controller, failed
SSH connection and successful SSH recovery were also exercised through the
actual UI. A failure after model preparation begins was not exercised.
