# Independent assessment helper review

The read-only reviewer `/root/memory_capture/assessment_review` examined
`assess.py`, focused tests and the completed BF16 recording. Its first pass
demonstrated these false PASS or false rejection cases:

- Requested native profiling could disappear from manifest configuration and
  become optional. The requirement now comes from the expected recipe's
  `profiling_configuration`, with recorded-mode parity required.
- Quantized weight format could conceal a wrong compute dtype, and legacy fields
  could conceal nested model configuration overrides. Production `_parse_model`
  now resolves precision and model source/revision precedence.
- Partial GPU retention could retain one activity in each layer and still qualify.
  Normalization must be complete or explicitly omit CPU detail while retaining
  all GPU activities and correlations with zero omitted GPU activities.
- A single KV layer could qualify the model's cache precision. Complete, unique
  coverage now follows dense layer count or observed hybrid attention indices.
- Adapter labels could qualify a native base route, config-only hash or wrong
  artifact source. Native route kind/integer identity, canonical source/revision,
  valid configuration hash and model-file hashes are now required. Production
  source canonicalization handles raw path/repository aliases and excludes
  authentication flags from artifact identity.
- A decrease capped at 512 MiB could be called loaded-storage release while most
  weights remained resident. Cleanup now requires framework detachment and a
  measured process decrease of at least 90% of deduplicated loaded GPU storage.
  The result is described as consistent with release; periodic management samples
  cannot identify individual freed allocations.

Regression tests retain all demonstrated cases. The final reviewer concluded
“No significant outstanding helper defect found after the repairs” and
independently confirmed all 12 focused tests pass. The memory collector's 13
focused tests also pass. This review covers the assessment helper; final frozen
source GPU captures and complete product reviews remain separate evidence.

The actual hybrid attempt `run-ffa6b814452a` exposed a dense-only assumption in
prefix lineage: recurrent groups lack per-token binding/write records. The
reviewer inspected native scheduler snapshot source and confirmed its earlier
returned `block_table_changes` retain structural ownership for every group.
Sequence 359 records Request 1 owning blocks 4/3/2/1 across three Mamba groups
and one full-attention group, after observed generation-1 allocations and before
later positive 784-token lookups. The assessor now keys all owner evidence by
cache group, pool, block and generation. Opaque recurrent structural ownership
does not become a token-write or numerical-content claim. Missing/zero generation,
reserved null blocks, missing group ownership and cross-adapter origins still
fail strict lineage checks. Fourteen focused tests pass, including the new
hybrid/group-aliasing regressions. This attempt now fails only its explicitly
unavailable native profile attachment and coverage; its memory, prefix and
precision evidence passes independently. The corrected helper was synced to the
execution host without changing capture source or launching GPU work.

The FP8 weights attempt `run-b8840655c01c` then failed only the pre-exit
process-counter release threshold. Its recorded last inference counter is
6,031,409,152 bytes and its last cleanup counter is 5,469,372,416 bytes, despite
5,014,291,096 bytes of unique loaded GPU storage. Those retained bytes remain in
the recording and the pre-exit check remains false. The capture owner's successful
exit and external management receipts provide separate lifecycle evidence:
PID 9277 exits at host monotonic time 3,928,133,534,687 ns; a receipt after exit
at 3,928,273,710,947 ns reports no compute PID and 4 MiB device usage on the same
GPU UUID and physical total. This matches the 4 MiB pre-run baseline observed
before owner spawn. The exact native reference owner of the pre-exit allocations
was not established, and no individual tensor release is claimed.

The helper now preserves the 90% pre-exit test as diagnostic evidence and requires
either that test or strict post-owner receipts for the mandatory cleanup lifecycle
guard. Post-owner proof checks observed management query success, PID absence,
matching GPU UUID and physical total, the same local execution target, ordered
host monotonic boundaries around owner spawn/exit, capture source parity, and
usage within an explicit 64 MiB allowance of the pre-run baseline. Missing,
wrong-host, wrong-UUID, wrong-PID, wrong-clock, wrong-source or unsuccessful
query receipts fail this alternative.

An independent correctness review also reproduced a missing final-state file
being treated as successful finalization. Finalization now positively requires a
receipt for the same run with `finalized: true`, completed status, the exact
`Recording finalized.` message, matching manifest/source/process identity and
a successful matching owner PID exit after the last recorded event. A retained
actual FP8 recording with a missing final-state path now fails both finalization
and post-owner cleanup. Nineteen focused tests pass. The original attempt's
`assessment.json` remains FAIL; `evidence-assessment.json` records the corrected
PASS with its explicit external post-owner basis. Production capture source and
recorded measurements were not modified.
