# Independent integration and evidence review

**Result: no significant issues found in this task's scope.**

Reviewed the integrated production components, browser receipt, all four retained 1920 × 1080 screenshots, web-test/build logs, and underlying bundled BF16 and Qwen3.8-27B recording data. This review did not operate the browser, rerun the browser cases, or perform a live capture.

## Checks

- The screenshots visibly retain Configure/Explore and the existing recordings control. The primary controls are Iterations and GPU activity; Component summary is collapsed and the GPU memory display is text. No replacement chart or Walkthrough control appears.
- `PlaybackTransport` exposes Raw events and Host time through the existing Base UI menu pattern, including selected-mode text on the closed trigger. `App` isolates dialog/menu keyboard input from global playback shortcuts. The browser receipt records menu navigation without an unintended event change and Raw events moving 200 → 201.
- The memory dialog uses the existing modal, has an accessible title and close action, and its receipt demonstrates focus cycling, Escape focus restoration, and real scrolling. The scrolled screenshot visibly reaches the phase table and CPU-storage section. Canvas bounds are identical before/after opening.
- Independently recomputed the displayed values directly from successful raw memory events, honoring event sequence and sample completion time. BF16 event 200 selects sample 172: **9.26 / 79.65 GiB**, sample age **94.9 ms**; the recorded-run maximum is **9.49 GiB**. BF16 event 2610 gives **1.11 / 79.65 GiB** with age **0.0 ms**. Hybrid event 355 selects sample 353: **53.76 / 79.65 GiB**, age **62.0 ms**, maximum **55.29 GiB**. The phase sample counts and maxima agree with the retained dialog text.
- `memorySummary` filters by GPU UUID and successful status. Recorded maxima intentionally summarize the whole run; live maxima exclude future sequence/timestamps. Three focused tests cover those distinctions and missing-versus-zero values. Cleanup attribution is invalidated by the unchanged memory projection and shown explicitly in the dialog.
- Independently computed the BF16 profiler extent from the bundled activity starts/ends: **22,295,681.006958008 µs**, exactly matching the receipt's native slider maximum. The screenshot shows an active Overview layer at **3.692 ms** while host replay stays at **event 201**. `profileBoundaryTimes` continues to use all activity starts for activity stepping; no gap-compression or clock-remapping behavior was introduced here.
- The retained log reports **107/107 web tests passed**; TypeScript and Vite build passed with the disclosed chunk-size warning.

## Evidence limits

The README properly discloses that these are offline bundled-recording checks, with the pre-existing controller connection failure visible in the screenshots. It does not claim a fresh live or multi-device capture. The receipt's incorrectly named Host time axis query returned false; the README calls out the probe error rather than presenting it as a passing assertion. Production routing to `RecordedTimeAxis` remains intact. These limits do not block this presentation-only change.
