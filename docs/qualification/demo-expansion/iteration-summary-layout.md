# Iteration summary layout verification

The current iteration and request breakdown now appear in a stable interface band above the 3D canvas. The overview and model focus no longer contain the floating model summary card. Desktop canvas minimum height increased from 288 to 360 pixels; the mobile sizing rule is unchanged. Configure, request lanes, inspection, playback clocks and memory controls remain available.

The band uses the same verified participation projection. Host mode shows scheduler prefill/decode counts. Native inspection shows only observed packed rows, excludes padding, and does not borrow host counts. Walkthrough explicitly identifies illustrated layer flow. Starting it closes local native iteration inspection to avoid mixing those clocks. Request side strips encode identity; the neutral card background does not encode allocation status.

The actual bundled BF16 run at iteration 21 reports three requests / 13 input positions: Request 3 and Request 2 each decode one input position, while Request 4 prefills 11. At 1200×720, the summary spans y=103–190 and the 360px canvas starts at y=190, with no overlap. At 390×844, badges wrap without horizontal overflow; the 523px mobile canvas remains below the summary and camera controls remain accessible. Memory breakdown expands/collapses, and pan/reset preserves iteration 21, its participants, model focus and playhead event 1031. Reduced motion remains enabled after switching to illustrated playback.

Web production build and 13 scene tests passed. Browser console contained no errors. Screenshots:

- `output/playwright/iteration-summary-desktop-start.png`
- `output/playwright/iteration-summary-desktop-mixed.png`
- `output/playwright/iteration-summary-desktop-walkthrough.png`
- `output/playwright/iteration-summary-mobile-mixed.png`

The final focused cleanup also removes the redundant per-request world cards and captions below the model. Pickable scheduled-input geometry remains. Request badges in the stable band are clickable and show exact zero-based token positions; discontiguous native positions retain their gaps. The model focus keeps one model-name line, with repeated evidence/tracked-request world headings removed; clock evidence remains explicit in the band and global legend.

After this cleanup, the actual BF16 iteration-21 summary at 1200×720 spans y=103–236 and the 360px canvas starts at y=236. At 390×844, the summary wraps to 219px and the mobile canvas remains 523px; document width equals viewport width. Clicking Request 4 selects its inspector, and pan/reset retains event 1031 and the same participants. Native inspection displays packed positions without host prefill/decode labels; starting Walkthrough closes native inspection and retains reduced motion. Memory breakdown was verified to collapse from `aria-expanded=true` to false. Browser console: zero errors/warnings. Final screenshots: `output/playwright/iteration-cleanup-desktop-final.png`, `output/playwright/iteration-cleanup-mobile.png`.

## Toolbar overlap regression, final live verification

Later direct interaction exposed a separate flex sizing defect at 1200×717 with the final BF16 recording (`run-af950e986ddc`), event 255, illustrated iteration 4. The toolbar wrapped into two rows but its flex item shrank to the 46px minimum. The second-row **Edit as new** button occupied y=103–135 while the toolbar ended at y=103, so the following iteration summary intercepted the click. The failed clicks and original geometry are retained in `output/playwright/live-h100-edit-click.txt`, `live-h100-edit-as-new.txt`, and `iteration-overlap-before.json`.

The toolbar and its sibling submitted/warning notes now reserve their natural height with `shrink-0`. No positioning, stacking or forced pointer click workaround was added. The same desktop toolbar is now 85px tall, ends at y=142, and the summary starts at y=142. Direct **Edit as new** opens Configure; Reset view preserves event 255. The original regression receipt uses a nonexistent heading name for its final visibility probe, so that probe reports false; the title and URL confirm the navigation, and the subsequent desktop/mobile receipts verify the actual Configure heading.

Actual browser regressions passed at 1200×717 and 390×844. In both cases, toolbar buttons remain inside the toolbar, the summary starts below it, and the document has no horizontal overflow. Memory breakdown expands to the measured accounting content and collapses; Reset view preserves the playhead; direct **Edit as new** returns to the real Configure form. Desktop native iteration inspection and mobile Request lanes also retain accessible toolbar actions. The mobile toolbar reserves 154.75px when its controls wrap. The web production build and `git diff --check` passed after the fix; the existing large bundle warning remains.

Receipts:

- `output/playwright/iteration-overlap-desktop-edit-regression.json`
- `output/playwright/iteration-layout-desktop-memory.json`
- `output/playwright/iteration-layout-mobile-memory.json`
- `output/playwright/iteration-layout-inspection-timeline.json`
- `output/playwright/iteration-layout-desktop-inspection.png`
- `output/playwright/iteration-layout-mobile-toolbar.png`
- `output/playwright/iteration-layout-mobile-reset.json`
- `output/playwright/iteration-layout-mobile-reset.png`
