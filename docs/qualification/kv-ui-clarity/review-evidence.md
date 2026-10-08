# Independent integration and evidence review

Reviewed 2026-10-08: the two-file working-tree diff against `867b039`, its surrounding selection flow, the completed browser case README/DOM/screenshots, and the final web test/build logs. This reviewer did not author the change or operate the browser.

**No significant integration or evidence issue found.**

- The Close handler selects the KV-cache component while `App.choose` retains token tracking. The new explicit block/token-selection gate prevents that retained tracking from reopening the cartridge. Explicit block/token resolution and tracked highlighting retain their previous paths.
- The actual saved-demo cases support Close after selecting position 46, explicit reopening, preserved tracking in Model execution, reserved-null and ordinary-block dismissal, and the phone-width repeat. The source is identified as real recording `run-a413ca908152`; these cases replay event 25 and do not imply new GPU inference or numerical values for the reserved token.
- Inspected the overview DOM and screenshot: the requested sentence is absent and the serving/model overview remains present. Source inspection confirms the profiler-clock, trace-gap and unknown-architecture captions retain their existing conditions; an empty `SceneText` is not rendered.
- Inspected the final logs: all 73 web tests pass and the TypeScript/Vite build succeeds with the disclosed existing bundle-size advisory.

The case README accurately separates the original failure observed in the tool transcript from retained post-fix evidence, identifies the unavailable live endpoint, and limits its claims to saved-recording UI behavior. A fresh GPU run is unnecessary for this rendering/selection fix. No additional evidence or repair is required by this review.
