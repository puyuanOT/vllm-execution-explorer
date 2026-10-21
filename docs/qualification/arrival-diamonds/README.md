# Planned arrival diamonds

The Configure arrival sequencer now uses one diamond per request at its planned dispatch offset. Vertical brackets identify arrival batches; the plot makes no execution-duration or scheduler-batch claim. It follows the approved **Request diamonds** mockup. Request colors, workload identities, serialization, and dispatch behavior are unchanged.

Readable 32 px rows (40 px in narrow charts) replace compressed bars. A bounded vertical scroller keeps all request labels reachable while the time axis remains visible outside it. Native titles and the accessible description preserve full labels when the visible names truncate. Tick density adapts to the actual plot width and displayed unit. Empty batches create no marks or axis extent.

Base: `3766f6d5506d7f8d09720efe3c5f8d6448060259`, tested from `codex/arrival-diamonds` in the isolated arrival-diamonds worktree. UI: `http://127.0.0.1:5180/configure`; controller: `http://127.0.0.1:8765`. Tests used the actual application and controller, with the synthetic fixture execution source.

## Browser outcomes

[Raw DOM observations](browser-cases.json) and [per-case assessment](assessment.json) retain the measured widths, diamond paths, labels, and outcomes for 22 observations.

| Journey | Outcome |
| --- | --- |
| Default six-request example | Six labeled diamonds in 2/3/1 groups at proportional 0/100/120 ms offsets. |
| Milliseconds ↔ seconds | Labels convert to 0/0.1/0.12 s; diamond geometry stays unchanged. |
| Edit B to 150 ms, then reload | All three B diamonds move together; the edited plan and diagram survive reload. |
| 320 and 360 px viewport | Diagram fits without horizontal scrolling. Full page bounds match viewport after the draft-action repair. |
| Maximum 600000 ms / 600 s offset | Accurate annotation and readable tick density in both display units. |
| Long request and batch labels | Names truncate within their reserved columns; full titles and accessible descriptions are retained. |
| Duplicate batch at 100 ms | Original and copied batches retain separate annotations and brackets at the same x coordinate. |
| Add an empty batch | Existing diamonds and time scale remain unchanged. |
| 64 requests | Every request retains a readable row; keyboard End reaches scrollTop 1702, equal to the maximum. Time axis remains outside the scroller. |
| Empty draft and all-empty batches | Chart unmounts; adding the first request remounts it with the measured width. |
| One and two requests at zero | Separate point rows on a valid integer-millisecond axis. |
| Capture, then return to Configure | Original six-request plan and the diamond chart remain intact. |

Screenshots: [desktop](desktop.png), [320 px](mobile-320.png), [360 px](mobile-360.png), [64 requests at the end of the scroller](dense-64.png).

Phone testing discovered an existing draft-actions overflow: page scrollWidth was 375 at both 320 and 360 px. Constraining the draft-summary/action container to its available width allows the existing wrapping buttons to wrap. Final page scrollWidth equals 320 and 360 respectively. The earlier failing observations remain in the raw evidence. The first immediate keyboard-End observation occurred during native scroll animation; the settled observation confirms the final row is reachable.

## Integrated capture

Run `run-3c86ec542ca0` completed through the actual Configure capture flow. The [manifest](fixture-manifest.json) preserves arrival groups `(0 ms, 2)`, `(100 ms, 3)`, `(120 ms, 1)`. The [event trace](fixture-events.jsonl) contains all six request completions, 289 events, complete event coverage, and no dropped events/content. The fixture is explicitly synthetic with `real_gpu: false`; this verifies frontend/controller integration, not GPU execution performance. No GPU run or LangSmith service tracing was needed for this diagram change.

The local controller was initially stopped. It was restarted before the capture experiment; the [final browser warning/error log](browser-final-log.json) was empty.

## Checks

- [Web suite](web-tests.txt): 68 passed.
- [Replay suite](replay-tests.txt): 51 passed.
- [Python suite](python-tests.txt): 214 passed.
- [Web type check and production build](build.txt): passed. Vite retains the existing large-bundle warning.
- Python compileall for runner/instrumentation: passed.
- `git diff --check`: passed.

The [initial Python attempt](python-tests-before-sdk.txt) had two import failures because this laptop's shared `.venv` lacked the `langsmith` SDK. The same failures were reproduced on unchanged main. Installing CI's pinned `langsmith==0.14.4` into the test environment resolved them; the final full suite ran with `LANGSMITH_API_KEY` and `LANGCHAIN_API_KEY` unset. No Python source or dependency manifest changed.

## Independent review

[Correctness](reviews/correctness.md), [integration and evidence](reviews/integration-evidence.md), and [simplification](reviews/simplification.md) reviewers found no remaining actionable issues. The evidence reviewer initially overlooked the top-level `case.page` width observations, then rechecked them and withdrew that finding. No source changes were needed after review.
