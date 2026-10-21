# Independent simplification review — October 13

**Result: no significant issues found.** No application edits or additional abstractions are recommended for this change.

## Scope and method

Reviewed the five-file scoped diff and the corresponding current source: `App.tsx`, `BlockMarks.tsx`, `CacheStage.tsx`, `ArchitectureNavigation.tsx`, and `ExecutionWorkspace.tsx`. Reconstructed the diff against the retained `before` files and confirmed it exactly matches [scoped.diff](../../../output/playwright/oct13-block-tops/scoped.diff). The preexisting worktree changes outside this task were excluded.

Applied the refactor and software-development-standards review guidance. Inspected direct palette/roster dependencies to check ordering, the [qualification record](README.md), source cases, DOM captures, browser receipt, test/build logs, and the pool-detail and shared-selection screenshots. This was a source and completed-evidence review; no browser interaction, runtime changes, or test reruns were performed by this reviewer.

## Findings

- **No dead rendering path or duplicate geometry added.** `BlockMarks` replaces its existing strip mesh with one non-pickable instanced plane mesh. Matrix/color updates, count updates, and bounding-sphere refresh remain together. The same component continues to serve both views; there is no second top-surface implementation to keep synchronized.
- **Band construction remains direct.** Current bindings are deduplicated, ordered, and divided equally. The small gap is bounded relative to band width, and no holder cap, extra wrapper component, or ownership-percentage model was introduced. Cached/null/unknown marks and selection brackets retain their existing paths.
- **Palette ordering is centralized appropriately.** `App` derives request order from the timeline roster and uses it for both the color map and scene prop. `projectSamples` maps that same roster, so the change removes the previous mismatch without maintaining another independently ordered list. The browser receipt confirms matching Request 2/3/4 chip and legend colors. The existing neutral fallback covers IDs absent from the roster.
- **No stale physical-block strip copy remains in the five files.** The legends and finder now describe top colors and equal shared bands. `Slot strip identifies holder` correctly refers to the cartridge's existing inset slot stripe; the zero-holder branch explicitly states that there are no current request references.
- **No redundant tests were added.** The recorded 107 passing tests preserve distinct allocation, holder/writer, and picking behavior; the real UI evidence covers the new visible encoding and shared-block selection. A new unit test that merely repeats plane dimensions would add little value.

## Evidence and limits

The pool-detail screenshot shows the expected pink/orange equal bands alongside green single-holder tops, a neutral hatched cached block, the null cross, and dark available blocks. The shared-selection screenshot preserves the split top and external selection brackets while its inspector distinguishes current holders from the original writer. The [browser receipt](../../../output/playwright/oct13-block-tops/browser-receipt.json) records consistent palette swatches and an empty captured warning/error log. The [test log](../../../output/playwright/oct13-block-tops/web-tests.txt) reports 107/107 passing; the [build log](../../../output/playwright/oct13-block-tops/build.txt) records successful TypeScript and Vite checks with the existing bundle-size advisory.

The evidence establishes the recorded 227-block offline replay and two-holder shared examples. It does not establish large-pool performance or live-controller behavior. Unknown blocks and more than two simultaneous holders were not exercised in the browser; the implementation retains the existing state exclusion and uses a general equal-band division. There is no evidence-backed reason to add caching, a sorting abstraction, or broader rendering refactors for this scoped change.
