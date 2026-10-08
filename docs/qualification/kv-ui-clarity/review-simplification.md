# Simplification review

No significant simplification issues found.

Independently inspected the current two-file diff against HEAD and surrounding source in `web/src/scene.tsx` and `web/src/architecture/CacheStage.tsx`.

- The requested overview caption is removed by returning `null` and rendering `SceneText` conditionally. Existing profiler-clock, trace-gap, and unsupported-architecture messages remain.
- The drawer resolves a selected block only for explicit block or token selection. Selecting the KV-cache component after Close therefore dismisses the drawer instead of reopening it from the tracked-token fallback. Explicit block/token inspection retains its previous resolution order; tracked-token highlighting is unchanged.
- The changes introduce no additional state, helper, dependency, or test abstraction. `git diff --check` passed.

This review evaluates source simplicity and scope. Browser behavior verification is recorded separately; no rendering or capture performance claim is made.
