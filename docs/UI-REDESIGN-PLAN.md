# vLLM execution explorer: UI and batch-sequence implementation plan

Prepared September 24, 2026. This is an implementation handoff, not an implementation. It is based on inspecting the current source, running the current UI, and reviewing [LLM-Visualized](https://www.llm-visualized.com/).

## 1. Product outcome and scope

Make the explorer a workspace where a user can **compose timed arrival batches, run them, and follow every sample through observed execution**.

The primary acceptance scenario is authored entirely through the UI:

```text
t = 0 ms       input_1, input_2
t = 100 ms     input_3, input_4, input_5
t = 120 ms     input_6
```

Each sample has editable prompt text and a maximum output-token count. Users can inspect its arrival, queueing, scheduled prompt/generation work, accepted output, emitted text, and terminal state. Selecting a sample highlights every execution batch containing it; selecting an execution batch highlights its participants.

Deliver this in two usable increments:

1. **Compose → capture → inspect:** complete sequence authoring, a coherent dark workspace, truthful run progress, and detailed replay after capture.
2. **Follow execution live:** the same visualization consumes incremental trace records, with independent live following and replay controls.

Both belong to the full plan. Increment 1 is a useful checkpoint; it must not be described as live execution visualization. Manual timeline dragging, saved scenario collections, and more elaborate 3D presentation are later polish.

### Essential terminology

| Term | Meaning in this product |
| --- | --- |
| Sample | One prompt/request with a stable identity. Use “sample” in authoring; show its request ID in details. |
| Arrival batch | A user-authored group whose samples share a requested arrival time. |
| Execution batch / scheduler step | The actual set of samples and token ranges selected by the engine in a recorded step. |
| Planned arrival | Requested offset from controller execution start, after engine initialization/warmup. |
| Observed arrival | A specifically named recorded observation, such as scheduler receipt or admission-call return. |

An arrival batch is not an atomic vLLM execution batch. The scheduler can split one arrival batch across steps and combine samples from different arrival batches in one step. Make this visible in the UI with one short contextual explanation, then use consistent labels.

## 2. What needs improvement in the current app

The problem is hierarchy as much as styling. The current UI gives two wide sidebars to setup and prose, while the central diagram is small, dim, and mostly unlabeled. The cream serif controls and brown scene feel disconnected. The default screen is empty, and the authoring form does not provide a sequence-building workflow.

| Existing code | Finding | Planned action |
| --- | --- | --- |
| `web/src/styles.css` | Cream serif interface; fixed 320px sidebars; large wrapping transport; narrow-screen layout stacks everything. | Replace with shared dark tokens, compact controls, responsive panel behavior, and more room for execution. |
| `web/src/panels.tsx::ConnectionPanel` | Two fixed request editors; no batch/sample add, remove, duplicate, or move controls. Connection settings precede authoring. | Replace primary surface with a sequence composer; move connection settings into a drawer. |
| `ConnectionPanel` local state | Switching sidebar tabs unmounts and loses the draft. Editable request ID is part of the React key. | Lift draft state; use immutable internal IDs as keys. |
| `web/src/scene.tsx` | Bars require a legend to interpret; fixed band positions do not scale with sample count. | Use labeled, scrollable sample lanes as the main view; retain 3D as an optional spatial/KV view. |
| `web/src/App.tsx` | Loading, polling, mutations, playback, shortcuts, selection, and layout are intertwined. | Extract focused hooks and feature components while preserving behavior. |
| `App` playback effect | One event advances every `max(80, 700 / speed)` ms. This is paced event playback. | Explicitly distinguish event playback from recorded-time playback. |
| `runner/workload.py`, `runner/controller.py` | Flat requests already support millisecond arrival offsets. | Reuse this path. Batch authoring does not require a new scheduler. |
| `runner/serve.py::RunState.snapshot` | Live status contains text and run status, not structured execution state. | Add incremental trace delivery in the live-view increment. |
| `replay/src/reduce.ts` | Deterministic event-prefix replay, content joins, and trace-gap handling already exist. | Keep as the source of truth; add pure visualization selectors. |

Do not broaden this into a framework migration, new engine scheduler, or transformer-internals visualization.

## 3. Visual direction

The reference uses a near-black visualization field, compact floating controls, subtle panel outlines, restrained technical typography, and selective color in the visualized data. Adopt those qualities, including its sense that the visualization owns the screen. These observations come from [LLM-Visualized](https://www.llm-visualized.com/); the layout and tokens below are proposed specifically for this explorer.

### Proposed design tokens

Keep the existing React/Vite/CSS stack. Establish semantic CSS variables rather than introducing Tailwind solely for a restyle.

| Token | Starting value / rule |
| --- | --- |
| Background / canvas | `#090B10` / `#06080C` |
| Panel / raised panel | `#11151D` / `#181E29` |
| Subtle border / interactive border | `#29313F` / `#58677D`; verify against adjacent colors |
| Primary / secondary text | `#E8EDF5` / `#A2AFC2` |
| Action and focus accent | `#77B7FF` |
| Success / warning / error | Muted green / amber / coral, always paired with text or a symbol |
| Typography | System sans-serif for interface text; system monospace and tabular numerals for IDs, time, and token counts |
| Type scale | 12px annotations, 14px controls/body, 16px panel headings, 20px app title; do not render important data below 12px |
| Spacing | 4/8/12/16/24px; consistent panel padding and row rhythm |
| Corners | 8px controls, 12px panels; modest shadows only where depth is useful |

Use one blue action accent. Sample identity colors are categorical data, not extra action colors. Provide eight distinguishable, reasonably bright colors for the current eight-sample limit, and assign them from the frozen sample order. Pair each color with a stable label; do not encode phase and identity with competing full-row colors. Use patterns, outlines, and short phase labels for prefill/decode/mixed work.

Avoid decorative gradients, glow, heavy glass blur, constant pulsing, and animated camera movement. Preserve reduced-motion support. Native controls should use `color-scheme: dark`. Verify text/control contrast rather than assuming the proposed palette passes every use.

### Workspace layout

At a wide desktop size, use a roughly 56px header, a 288–320px composer, a flexible execution workspace, and an optional 300–340px inspector. Keep the bottom transport around 64–80px. The inspector opens when a sample/step is selected and can collapse.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ vLLM Explorer   Example ▾    Demo fixture · Ready   Recordings   Settings    │
├─────────────────┬──────────────────────────────────────┬───────────────────┤
│ Arrival sequence│ Execution           Timeline | 3D   │ input_4           │
│ 3 batches       │ Planned arrivals: 0 / 100 / 120 ms    │ Batch B · 100 ms  │
│ 6 / 8 samples   │                                      │ Queued            │
│                 │ Sample       Observed execution      │                   │
│ A   [  0] ms    │ input_1  ──[P][D][D]─────────✓       │ Prompt            │
│ input_1  Edit   │ input_2  ──[P][D][D][D]──────✓       │ …                 │
│ input_2  Edit   │ input_3  ·······[P][D]────────✓       │ Output            │
│ + Add sample    │ input_4  ·······[P][D][D]─────✓       │ …                 │
│                 │ input_5  ·········[P][D]──────✓       │ Accepted  0 / 16  │
│ B   [100] ms    │ input_6  ·········[P][D][D]───✓       │ Scheduled ranges  │
│ input_3 …       │                                      │ KV blocks         │
│ input_4 …       │ Actual steps  [1][2][3][4][5]…        │ Milestones        │
│ input_5 …       │ Selected step: input_2 + input_3      │                   │
│                 │                                      │                   │
│ C   [120] ms    ├──────────────────────────────────────┴───────────────────┤
│ input_6 …       │ ⏮  ◀  Play  ▶     Events ▾  1×    Event 16 · 124 ms       │
│ + Add batch     │ ──────────────────●───────────────────────────────────   │
│ [Run sequence]  │ Recorded host observations · Fixture                    │
└─────────────────┴──────────────────────────────────────────────────────────┘
```

This wireframe illustrates layout and relationships, not a predicted execution schedule. `P` and `D` denote recorded scheduled work; they do not imply measured kernel duration.

Below roughly 1280px, turn the inspector into an accessible drawer so the center retains usable width. Below roughly 900px, offer Compose/Observe section switches with preserved state; allow horizontal scrolling inside the time plot while sample labels remain readable. Avoid stacking every setup field above the visualization. Use `100dvh`, explicit `min-width: 0`/`min-height: 0`, and contained scrolling.

## 4. Authoring interactions

### First visit

Load an editable “Staggered arrivals” example with the exact 2/3/1 grouping at 0/100/120ms, short usable prompts, and sample labels `input_1` through `input_6`. Show all six planned rows immediately. Do not run automatically. Explain the three-step flow in one sentence: “Create arrivals, run the sequence, then follow each sample.”

Provide “Start empty” and the existing recording library as secondary actions. If a runner is unavailable, the composer still works, and bundled recordings remain viewable. Place the runner error next to Run sequence with a Configure runner action.

### Arrival batch cards

Each card has a stable batch ID, display label, editable arrival time, sample count, compact sample rows, Add sample, and a menu for Duplicate/Delete. Expand a sample inline to edit its prompt and maximum output tokens. Keep labels and time visible while expanded.

Required operations:

- Add a batch, with the next default offset set to the greatest current offset plus 100ms and focused for editing.
- Add/remove/duplicate samples, and move a sample to another batch through a menu or select control.
- Duplicate a batch with new batch/sample IDs, preserving prompt/settings and initially keeping its offset.
- Change a batch time once to update all its samples.
- Edit display labels without changing request identities or losing input focus.
- Delete with Undo for reversible draft edits. A batch with no samples remains an editable placeholder, but prevents running until filled or removed.

Use numeric time fields as the primary precise interaction. Default to milliseconds. If a seconds toggle is added, convert `0.1s` to 100ms and `0.12s` to 120ms exactly; reject sub-millisecond values rather than silently truncating. Switching units only changes presentation.

Commit and sort time edits on blur/Enter, not every keystroke. Maintain stable authoring order for equal offsets. Same-time batch cards may remain distinct authored groups; show their shared time in the overview without silently merging them. Reordering cards should never silently alter timestamps.

The small arrival overview uses proportional time spacing. Close markers such as 100ms and 120ms need zoom/clustered labels or a readable list fallback. Do not spread markers evenly while labeling the axis as time. Clicking a marker selects its batch. Drag-to-retime is optional later polish; it must retain the numeric alternative and never create fractional/negative offsets silently.

### Validation and persistence

- Validate nonempty prompts; unique stable IDs; finite, nonnegative integer offsets; positive integer output limits; valid engine settings; no empty batches.
- Read `/api/limits`; show total sample count against the current cap of eight, 4,000 characters per prompt, and 128 output tokens. Keep server validation authoritative and avoid presenting unavailable limits as confirmed.
- Establish a maximum arrival offset consistent with an enforced run-duration limit. The server currently advertises 120 seconds without enforcing it.
- Show errors beside the relevant field with `aria-invalid` and `aria-describedby`; focus the first invalid field on submit. Preserve all draft data on failure.
- Keep draft state above transient panels. Opening settings or recordings must not reset it.
- Add versioned browser-local draft persistence with a visible “Saved on this device” indicator and Clear draft action. Handle storage failures without blocking editing. An export of the scenario is optional; do not add cloud storage.
- On Run sequence, freeze an immutable submitted snapshot. The visualization must use that snapshot, not the current editable draft. While running, show the submitted plan read-only; “Edit as new sequence” creates a separate draft.

## 5. Execution and inspection

### Primary sample view

Use DOM/SVG for the main labeled timeline. This is easier to scroll, select, test, and make accessible than placing all interactions inside WebGL. Keep React Three Fiber available through a secondary 3D view for the existing request/batch/KV representation.

Each sample keeps the same row and identity color throughout the run. Group or label rows by their authored arrival batch without resorting rows as statuses change. The row displays its label, current evidence-based state, scheduled prompt/generation segments, accepted output count, and terminal marker.

Planned samples are visible before arrival, but they are view-model entries, not fabricated `request_received` events. Use “Planned” before the first receipt observation. If a requested time has passed but receipt has not been observed, show “Awaiting observed admission.” A cancelled run can leave samples “Not submitted”; do not manufacture per-request cancellation events.

Derive phases from recorded ranges:

| Evidence | Display |
| --- | --- |
| Plan only, no applied receipt event | Planned / awaiting admission |
| `request_received`, `request_queued` | Received / queued |
| Scheduled range with prompt tokens only | Prefill scheduled |
| Scheduled range with generation tokens only | Decode scheduled |
| Range crossing the prompt boundary | Mixed prompt + generation scheduled |
| `tokens_accepted` | Accepted output increases |
| `text_emitted` plus content | Visible generated text increases |
| `preemption` | Preempted marker; recomputation where supported by following ranges |
| `completion` / `cancellation` | Recorded terminal status |
| Trace gap | Unknown observed state after the gap |

A scheduled range reserves work; it is not proof that output was accepted. Request status in the existing reducer is too coarse to supply all phase labels directly.

### Actual execution batches

Show a compact step strip beneath the sample lanes. A step exposes its ID, participant chips, prompt/generation counts per participant, total scheduled tokens, and token budget. Clicking it seeks to its scheduling event and highlights the actual participants across arrival groups.

Offer “Next step for this sample” and milestone navigation in the inspector. A selected sample stays selected while stepping unless the recording changes. All surfaces share one selection model.

### Selected-sample inspector

Order details by usefulness:

1. Label, request ID, arrival batch, and current status.
2. Requested arrival and separately named observed times.
3. Prompt and generated output, with clear missing/redacted-content states.
4. Accepted tokens, scheduled prompt/generation ranges, and prefix reuse where recorded.
5. Clickable milestone history and participating scheduler steps.
6. Collapsed technical details: KV block pool/ID/generation, preemptions, event sequence, raw observation fields.

Use existing `explainRequest`, `explainStep`, and `explainBlock` selectively rather than printing every explanation all the time. KV identity must include pool and allocation generation; block number alone is not stable identity. Do not paint a shared block as exclusively owned by an arbitrary first request.

Use metric labels with precise definitions. For example, “Wait to first schedule” is scheduler receipt to first scheduling observation; “First emitted text” is receipt to first text-emission observation. Accepted-token and text-emission timing are distinct. Render missing values as unavailable, not zero. Do not add token probabilities, tokenizer duration, or GPU utilization: they are not in this trace.

### Time and playback contract

- Add an explicit controller execution origin. Engine/model loading is a separate Preparing state and occurs before `t=0`.
- Requested deadlines may be late: synchronous engine steps prevent admission until the controller regains control. Several due arrival batches can become eligible at one boundary.
- Keep planned time markers visually distinct from observed scheduler receipt and host step intervals.
- Keep event `sequence` authoritative, including equal timestamps. Never reorder replay by timestamp.
- Default “Events” playback advances at a teaching cadence; speed means event playback speed. Provide previous/next event and previous/next scheduler step.
- Add “Recorded time” mode only with a timestamp index that maps the playhead to the last applicable sequence. Display milliseconds, not a misleading claim of nanosecond precision in JavaScript numbers.
- For legacy recordings without an explicit controller origin, label time as “Since first recorded event” and keep planned offsets on a separate ruler. Event-triggered lesson arrivals must stay labeled as event-triggered.
- Seek/scrub pauses replay. Keyboard shortcuts must ignore inputs, textareas, selects, editable content, and open dialogs. Play at the end should provide explicit Replay behavior.
- Playback speed, scrubbing, and pause never change actual run admission or stop the running engine.

Only reveal text, accepted tokens, phases, and completion through the currently applied prefix. A full-recording index may support navigation, but future results must not appear as present facts. Stop extending observed state at a trace gap and show the existing unknown-state explanation prominently.

## 6. Draft model and API contract

Move run request types out of `panels.tsx` into `api.ts` or a dedicated contracts module. Introduce approximately:

```ts
type SampleDraft = {
  id: string;                 // stable internal/request identity
  label: string;              // editable display name, e.g. input_4
  prompt: string;
  maxOutputTokens: number;
};

type ArrivalBatchDraft = {
  id: string;
  label: string;
  arrivalOffsetMs: number;
  samples: SampleDraft[];
};

type ScenarioDraft = {
  version: 1;
  id: string;
  name: string;
  batches: ArrivalBatchDraft[];
};
```

Keep temporary invalid text values in the field editor until commit/validation. Do not force an unfinished numeric edit into zero or NaN in the canonical scenario.

Flatten the frozen scenario to the existing `POST /api/runs` request list:

```ts
const requests = orderedBatches.flatMap(batch =>
  batch.samples.map(sample => ({
    request_id: sample.id,
    prompt: sample.prompt,
    arrival_offset_ms: batch.arrivalOffsetMs,
    max_output_tokens: sample.maxOutputTokens,
  }))
);
// The acceptance scenario yields offsets [0, 0, 100, 100, 100, 120].
```

Add an optional `workload_plan` alongside `requests`, containing only version, batch/sample IDs and display labels, requested offsets, display/submission order, and maximum output counts. The server validates an exact correspondence between metadata and the submitted requests, then persists a canonical form in the manifest. Legacy callers can omit it; the server synthesizes a minimal plan from their request list. Keep engine settings in the existing run configuration.

Do not duplicate prompts, output, or token IDs in plan metadata. The structural export currently copies manifests, so textual content belongs in `content.jsonl`. Record planned prompt content once even for samples never admitted, with no fictitious structural receipt event. Whitelist metadata fields and follow the existing export treatment of identifiers/display labels; prompt content must never be used as an automatic label.

Persist the full plan before execution and retain it on completion, cancellation, and failure. Add optional `controller_started_monotonic_ns` plus its clock provenance when execution starts. Keep existing schema 1.0.0 compatibility: its manifest permits additional properties. Type/document new optional fields and do not introduce event types that the current reducer rejects.

On old recordings, derive the roster from available events and arrivals. Only group by numeric offsets when the policy is actually controller offsets. If original batch membership is unrecoverable, say so; do not invent it.

## 7. Focused runner changes

These are needed for the user-facing contract, not a replacement for vLLM scheduling.

1. **Strict validation:** Harden `parse_workload` and run-setting validation. Reject malformed objects, booleans where numbers are expected, nonfinite/fractional offsets, empty prompts, zero output limits, and out-of-range settings with structured field errors. The existing `int(...)` conversions can truncate fractions, and `value or 16` turns zero into a default. Reuse validation for Add now.
2. **Stable admission order:** In `drive_run`, preserve submission order among equal offsets. Current sorting by `(offset, request_id)` can reorder user-authored samples. Admit all due samples before the next engine step; do not promise simultaneous API calls.
3. **One clock contract:** Capture the origin and admission observations through the same injectable monotonic clock source. Preserve names such as admission-call return versus scheduler receipt. Never combine epoch-based request arrival fields with monotonic time. Retain progress metadata if execution raises before `drive_run` returns.
4. **Intake versus accepted plan:** Closing intake must stop new additions while allowing already accepted future samples to run. The current intake guard suppresses pending planned arrivals. Keep Close intake out of the main authoring workflow; the finite sequence should complete automatically when pending and unfinished work are empty.
5. **Responsive cancellation/deadline:** Replace the potentially long sleep before the next arrival with bounded interruptible waits, and enforce the advertised run deadline. Cancellation preserves planned-but-unadmitted rows without fabricating engine events.
6. **Run identity and state:** Expose preparing/capturing/finalizing/completed/failed/cancelled states clearly. Define an active-state predicate and update backend admission guards and frontend polling together: they currently recognize only the literal `running`. Preparation/finalization must not permit a second concurrent run. Scope client fetches to the accepted `run_id`, discard stale responses, and do not let an older run overwrite a newly opened recording.
7. **Failure-safe finalization:** Keep partial recordings inspectable. Move recording initialization inside guarded execution and handle session cleanup and manifest finalization independently. Currently `Recording.open()` is outside the try block and `session.close()` can throw before manifest writing. Publish a terminal status even if initialization/cleanup fails; preserve available plan/progress and clearly report any artifact that could not be finalized.

### Fixture/demo behavior

The current fixture is not a configurable vLLM simulation. It picks two requests, ignores `max_output_tokens`, generates roughly three tokens per request, and can finish before the 100ms batch arrives. Some manifest settings describe requested settings that the fixture does not actually apply.

For a useful default demo, add an explicitly labeled authored-run fixture configuration without changing the established lesson defaults:

- Honor output limits and record effective behavior accurately.
- Add documented teaching pacing, for example a 40ms fixture-step interval. Tests use an injected clock instead of sleeping.
- Support selected concurrency/token-budget settings only if implemented correctly. Otherwise disable those controls in fixture mode and show the effective fixed settings. Prefix-cache controls also need truthful support/availability.
- Keep `execution_kind: fixture`; show “Demo fixture” throughout. Pacing is synthetic and never a GPU performance measurement.
- Make the six-sample demo long enough to demonstrate queueing/overlap, while allowing actual admission to occur after requested deadlines at controller boundaries.

Do not impose pacing on the GPU execution path. Preserve existing scripted lessons and their trace-integrity tests. GPU acceptance requires access to the supported CUDA/vLLM host; the current README says that feasibility proof has not been run here.

## 8. Live trace delivery

Implement after recorded replay and selectors work. Keep polling initially; no WebSocket infrastructure is needed for this workload size.

Proposed endpoint:

```text
GET /api/runs/{run_id}/trace
    ?after_sequence=16
    &after_content_sequence=8
    &limit=500
```

Return complete event/content records, independent cursors, a bounded-page indicator, capture/finalization status, and provisional run metadata sufficient to build the planned roster. Clearly mark provisional completeness as pending; an active recording is not yet proven complete. Include the finalized manifest once available.

Implementation requirements:

- Safely read only complete flushed JSONL records or provide an equivalent bounded collector read model. The collector already flushes records individually. Never parse a partial trailing line or scan an ever-growing whole file on every poll. Cache append-only file-offset/sequence indices, not a shared destructive reader position. Honor each caller's supplied cursors independently, including retries and multiple clients; advance returned cursors only past returned records.
- Do not add network calls or blocking UI subscribers to scheduler hooks. Preserve existing buffer-overflow/trace-gap semantics.
- Deduplicate events by sequence and content by content sequence. Keep missing-content placeholders until late content arrives. Do not drop content just because its event was already rendered.
- Poll around every 250–500ms with one request in flight per run, abort stale requests, and back off on failures. Poll health/settings separately instead of refetching preflight every frame.
- A small deterministic incremental replay adapter can apply appended events; it must not finalize a partial stream as a finished recording. Reuse reducer semantics and preserve historical seek capability.
- Default to Follow live. Scrubbing or event stepping turns following off while collection continues. A clear Follow live action catches up again. Never label replay pause as pausing vLLM.
- On finalization, drain any remaining cursor pages or load the authoritative full recording before reconciling; a finalized flag must not discard unread records. Reconcile once, preserve a user's manual selection/seek where possible, and verify the final live state equals replaying the finalized data.
- Reconnect from cursors; show disconnected/catching-up state rather than inferred progress. A known gap freezes observed state consistently in live and replay modes.

## 9. Frontend decomposition

Suggested files are boundaries, not a requirement to create dozens of tiny components:

```text
web/src/
  App.tsx                        composition and top-level ownership
  api.ts                         contracts, checked fetches, errors
  theme.ts                       sample palette and shared scene colors
  styles.css                     semantic tokens, shell, typography
  scenario/
    types.ts
    reducer.ts                   authoring actions with stable identity
    validate.ts
    serialize.ts                 scenario ↔ API/recorded plan
    BatchSequenceComposer.tsx
    ArrivalBatchCard.tsx
    SampleEditor.tsx
    ArrivalOverview.tsx
  execution/
    types.ts                     shared selection/view types
    ExecutionWorkspace.tsx
    SampleSwimlanes.tsx
    ExecutionBatchStrip.tsx
    PlaybackTransport.tsx
    SelectionInspector.tsx
  shell/
    AppShell.tsx
    RunnerSettings.tsx
    RecordingLibrary.tsx
  hooks/
    useScenarioDraft.ts
    useRunner.ts
    useRecording.ts
    usePlayback.ts
  scene.tsx                      retained optional 3D view

replay/src/
  timeline.ts                    pure indices/selectors
  timeline.test.ts
```

`timeline.ts` should expose a planned roster, sample milestones, step membership, time-to-sequence lookup, and event-prefix view selectors. Build reusable indices once per recording or append to them for live data. Do not repeatedly scan every event for every lane on every render.

Keep `Selection` independent of `scene.tsx`. Use local reducers/hooks and context where necessary; a new global-state framework is unnecessary for eight samples. Avoid parallel copies of replay state in each visual component.

Use native form controls where they suffice. For dialog/menu/tooltip focus behavior, choose one accessible primitive system, such as Radix, and use it consistently. Do not hand-build keyboard navigation for custom tabs or menus. Confirm package compatibility at implementation time.

## 10. Accessibility and interaction quality

- Every authoring action works by keyboard; drag interactions have exact numeric/menu alternatives.
- Focus stays stable while editing names/times and moves to a sensible adjacent control after deletion. Drawers restore focus on close.
- All icon buttons have accessible labels; selected items expose state semantically. Provide usable targets even when the glyph is small.
- Status labels and patterns accompany color. Expose the same sample/step facts in semantic DOM controls or a compact accessible table; WebGL must never be the only source.
- Use a restrained live region for run transitions and validation failures, not every token/frame.
- Respect reduced motion and the explicit user preference. Do not add decorative animation. If later requested, limit micro-feedback to brief transform/opacity changes, and keep it independent of trace timing.
- Keep source/evidence visible as a compact badge. Incomplete capture is a substantive warning and deserves more prominence than ordinary fixture labeling.
- At small sizes and 200% zoom, keep the authoring and observation actions reachable without overlapping panels or obscured controls.

## 11. Implementation order and checkpoints

| Phase | Deliverable | Exit criteria |
| --- | --- | --- |
| 1. Contract and scenario state | Stable draft types/actions, flattening, strict validation, optional recorded plan, stable admission order and explicit origin | Six requests serialize to exact expected offsets; plan survives cancellation/failure; old recordings still load. |
| 2. Dark shell and composer | Shared theme, batch/sample authoring, local draft persistence, settings/library surfaces | User builds 2/3/1 scenario without JSON; focus/draft survive edits/navigation; desktop/narrow layouts are usable. |
| 3. Evidence selectors and playback | Roster, milestones, membership, time indices, clear playback modes | Deterministic forward/backward seeks; no future output leakage; planned/observed clocks are distinct. |
| 4. Execution workspace | Sample lanes, actual-step strip, cross-selection, inspector, truthful fixture experience | User can follow input_4 from plan through its actual scheduled work and emitted output in a captured run. Increment 1 is usable. |
| 5. Live trace | Bounded cursor API, streaming adapter, Follow live and final reconciliation | Live final state equals finished replay; scrub/pause never stop the engine; reconnect and late content work. |
| 6. Integration and visual QA | Responsive/accessibility fixes, regression suite, updated README | Acceptance checklist below passes; screenshots demonstrate the intended design at representative sizes. |

Phases 1 and the visual shell can be developed in parallel after agreeing on draft/plan types. Replay selectors can proceed against existing lessons in parallel with composer work. Live transport depends on a settled time/metadata contract. Integrate each phase before expanding optional interaction scope.

Keep manual arrival dragging, saved scenario libraries, split/merge batch shortcuts, custom camera choreography, and additional analytics out of the critical path. Retain the existing lesson library, local imports, fixture/GPU distinction, SSH setup, cancellation, and reduced-motion behavior.

## 12. Verification and definition of done

### Functional acceptance walkthrough

1. Start with an empty sequence. Add batches at 0, 100, and 120ms with 2, 3, and 1 samples. Set labels `input_1`–`input_6`, prompts, and output limits entirely through the UI.
2. Check that the submitted request offsets are exactly `[0, 0, 100, 100, 100, 120]`. Names/IDs and within-time order remain stable. Changing display units does not change these values.
3. Move a sample between batches, duplicate/remove another, and undo deletion. Reopen settings and the recording library; the draft remains intact. Name editing does not lose focus.
4. Run the scenario. The frozen submitted plan remains identifiable and all six rows are visible before arrival. Show preparation separately from controller execution time.
5. Select `input_4`. See its requested arrival, observed receipt if available, prompt, scheduled ranges, execution-step participation, accepted output, emitted text, and recorded terminal state.
6. Select a mixed execution batch and see every participant highlighted, including samples from different arrival groups where the recording contains that behavior. Do not require GPU schedules to match the illustrative wireframe.
7. Step backward before input_4 is admitted. Its generated text and observed progress disappear; planned metadata remains. Change replay speed and verify the state at a chosen sequence is identical.
8. Cancel before a future batch arrives. Those planned samples remain visible as not submitted. Close intake in an advanced scenario and verify accepted future samples still drain.
9. Reopen the saved recording and a structural-only export. Preserve grouping/timing metadata; redacted prompt/output text stays unavailable. Existing event-triggered lessons retain their own timing semantics.
10. In the live increment, scrub backward while the engine runs, then Follow live. Reconnect without duplicate events or text; finalized live state matches recorded replay.

### Automated tests with meaningful assertions

| Layer | Required coverage |
| --- | --- |
| Scenario model/UI | Batch/sample operations, stable keys/IDs, offset conversion, invalid intermediate fields, duplicate identities, limits, persistence, frozen run versus edited draft, exact flattened acceptance payload. |
| Controller/API | Fake-clock 0/100/120ms admissions; stable ties; a long step that crosses both later deadlines; all due samples before next step; strict initial/Add now validation; idle cancellation; deadline; close-intake draining. |
| Manifest/export | Canonical plan survives success/failure/cancel; origin provenance; prompt/output omitted from structural export; old metadata-free recordings work. |
| Replay selectors | Planned rows, prefix-limited output, prefill/decode/mixed ranges, preemption, block generation/sharing, terminal states, equal timestamps, backward seek, unknown state at gaps, missing content. |
| Fixture | Effective settings match manifest, output limits are honored when supported, teaching pacing isolated from GPU path, current scripted lessons preserved. |
| Live trace | Complete-line reads, bounded pages, cursor retries/deduplication, late content, disconnect/reconnect, gaps, finalization races, old-run response rejection, final state equals finished replay. |

Use the existing commands as baseline checks:

```bash
.venv/bin/pytest -q
npm test --prefix replay
npm run build --prefix web
```

The web package currently has no test harness. Add focused component tests for authoring and a browser integration test for the main acceptance flow using the project's chosen testing tools. Use injected clocks for timing tests; avoid flaky assertions about real sleep durations.

Visually review at approximately 1440×900, 1280×800, and 768×1024, plus 200% zoom. Include empty/draft, loaded recording, selected sample, running, cancelled, disconnected, missing-content, and trace-gap states. Check keyboard traversal and system reduced motion. Run existing continuous-batching, chunked-prefill, prefix-reuse, preemption, cancellation, and incomplete-capture lessons through the new view.

Do not claim GPU validation based on fixture results. If hardware is unavailable, finish all local checks and document the remaining real-GPU acceptance step explicitly.

## 13. Prompt to give the implementing agent

> Implement the UI and timed arrival-sequence workflow described in `docs/UI-REDESIGN-PLAN.md` in this repository. Read the current implementation before editing. Preserve the deterministic replay reducer, existing lessons/imports, trace-gap semantics, and fixture-versus-GPU evidence. Start with the scenario/recording contract and dark batch composer, then implement sample-level replay, cross-selection, and live trace following in the documented order. Reuse the existing flat timed-request API and add only the metadata/runner changes needed by the plan. Treat arrival batches and scheduler batches as different concepts. Keep the running/submitted plan immutable, and never derive execution facts from UI animation or future events. Complete the exact 0/100/120ms, 2/3/1 sample acceptance scenario, appropriate automated checks, and visual/accessibility QA. Do not turn this into a framework migration or a new vLLM scheduler. Report completed phases, verification, and any GPU validation still requiring hardware.
