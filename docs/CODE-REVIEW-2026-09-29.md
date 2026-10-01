# Code review and remediation — 2026-09-29

Reviewed the v0.1.0 baseline in three passes, with an edit-owning reviewer,
an independent trace reviewer, and integration verification.

## Pass 1: execution and remote operations

- Closed intake atomically with the final queue check so an accepted add-now
  request cannot disappear when the engine finishes its previous work.
- Excluded environment and private-key files from project setup transfers.
- Quoted remote setup destinations and rejected SSH option-like usernames
  and control characters in remote paths.
- Assigned separate staging directories to artifact transfers, preventing
  old model files from being mixed with a subsequent model. Retries retain
  their original staging directory; capture uses the returned artifact path.
- Rejected concurrent SSH connections, validated targets before replacing
  credentials, and cleared the previous artifact target after a failed
  reconnect.

## Pass 2: browser state and controller selection

- Ignored obsolete controller refreshes and polls after endpoint changes;
  reset live trace cursors when the endpoint changes.
- Disabled execution-host controls and new capture submission during host
  checks and captures, keeping cancellation available.
- Validated saved draft structure before hydration so malformed batches,
  samples, or settings cannot crash startup. Supported legacy defaults and
  incomplete editable fields remain supported.

## Pass 3: recording and replay truthfulness

- Preserved quoted prompt/output strings while parsing large JSON integers.
- Removed stale request block references after completion, cancellation,
  preemption, zero-reference release, and allocation-generation changes;
  omitted ambiguous links across block pools.
- Required terminal request evidence and scheduled-step results before a
  completed recording can pass integrity checks. Explicit partial runs
  remain supported.

## Verification scope

Each reproduced bug has a targeted regression check where practical.
Final automated checks: 103 Python tests, 23 replay tests, and 20 web tests
pass, together with the production build, Python compilation, and whitespace
checks. UI verification covers capture-to-Explore with the bracketed-number
prompt preserved, disabled host controls during capture, and an old
controller's delayed response arriving after a new controller is selected.
Cancellation displays its outcome and restores host and capture controls.
The existing sanitized GPU acceptance bundle and checksums remain unchanged.
This review uses local fixtures and mocked remote operations; it does not
claim a new GPU acceptance run. A trace truncated after its final terminal
and result events cannot be distinguished from a full trace without an
independently recorded expected final sequence or checksum.
