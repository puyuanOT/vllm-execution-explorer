import type { RunSnapshot } from "../api.ts"

/** Controller runs started elsewhere may update health, but must not cause this
 * UI to download their traces or native profile attachments. */
export function ownedLiveRun(expectedRunId: string, live: RunSnapshot | null, userOpenedRecording: boolean): RunSnapshot | null {
  return expectedRunId && !userOpenedRecording && live?.run_id === expectedRunId ? live : null
}

export function expectedRunFailure(
  expectedRunId: string,
  live: RunSnapshot | null,
  userOpenedRecording: boolean,
): string {
  if (!expectedRunId || userOpenedRecording || live?.run_id !== expectedRunId) return ""
  const finalizationFailed = live.status === "completed" && live.finalized === false
  if (live.status !== "failed" && live.status !== "cancelled" && !finalizationFailed) return ""
  const detail = live.message.trim()
  if (detail) return detail
  if (finalizationFailed) return "Capture completed, but the recording could not be finalized."
  return live.status === "cancelled" ? "Capture was cancelled." : "Capture failed during preparation or execution."
}
