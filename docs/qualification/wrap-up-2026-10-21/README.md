# Wrap-up integration — 2026-10-21

This integration preserves the accepted Explore-first UI, configuration workspace, realistic H100 demos, recorded GPU layer activity, request-colored KV block surfaces, and memory inspection. The original development worktree remains untouched.

## Targeted repairs

Memory telemetry no longer runs a management command after every model iteration. The existing background sampler collects observations. Shutdown performs its final attempt on that worker with a bounded deadline, retires publication before closing the recording, and reports incomplete accounting as `gpu_memory.status: partial`. Valid earlier samples remain available, with an incomplete-capture explanation in Memory breakdown. Repeated close and late results cannot modify finalized event files.

Small refactors remove duplicate title ownership, unused step-highlight work, and a dead finalization helper with implementation-only tests. Live capture remains enabled in local builds. The public build uses `VITE_HOSTED_DEMO=true`, preserving the configuration workspace with local-controller guidance and disabled hosted capture actions.

## Data and preservation

All 88 canonical runtime assets match their original sizes and SHA-256 values. Thirty-three compressed parts are pinned to immutable public commit `129404b6c905811593063ded5fcaa1f9c2a1a71b`. `npm run recordings:download` uses normal OS TLS verification through curl, validates the compressed assets, and restores canonical files for offline playback/tests. Local playback prefers those files. Public builds exclude generated caches and raw profiler archives.

See [archive validation](ARCHIVE-VALIDATION.md) for the complete original capture trees and audit. Their [versioned release](https://github.com/puyuanOT/vllm-execution-explorer/releases/tag/wrap-up-2026-10-21) preserves all historical attempts, not only successful results. Existing tracked qualification evidence remains in Git; new scratch output is excluded.

## Verification

- Final local suite: 339 Python, 79 replay, and 129 frontend tests passed (547 total); TypeScript/Vite build and Python compilation passed.
- The before-fix regression demonstrated inference blocking; actual CPU subprocess/session/collector integrations verify responsive stepping, final accounting, bounded shutdown, publication retirement, and unchanged persisted bytes after a late query. These integrations use representative management commands, not CUDA measurements.
- An actual HTTPS cache miss downloaded and verified a compressed BF16 part and restored its original canonical bytes. All 88 files passed the original hashes.
- Codex in-app browser at 1920×1080 covered local BF16 and largest Qwen playback, pending selectors, measured activity, local Configure, hosted BF16 and largest multipart Qwen, hosted capture guards, and imported recordings. The incomplete-memory warning was checked with a derived real recording whose partial status/reason alone were authored; its measured counters remained unchanged. The CPU timeout fixture was labelled synthetic. No live GPU recapture was performed.
- Browser network isolation was unavailable. Offline readiness is supported by denied-remote-fetch decoder regressions, full local materialization, and local browser playback; no offline browser switch is claimed.
- Three independent final reviewers assessed correctness, integration/evidence, and simplification. The sampler shutdown finding was fixed and independently reproduced as resolved. No significant issue remained in the reviewed scope.

`wrap-up-verification-2026-10-21.tar.gz` in the release retains per-case DOM snapshots, screenshots, logs, regressions, and CPU integration scripts/recordings, including initial failures and their successful checks. GitHub CI and the deployed revision are checked during publication; final commit identities are recorded by the release/tag and the Site's `application-revision.json`.
