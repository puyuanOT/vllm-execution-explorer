import type { ProfileActivity } from "../../replay/src/profiling.ts"

export function profileBoundaryTimes(activities: ProfileActivity[], boundary: "activity" | "batch"): number[] {
  if (boundary === "activity") return [...new Set(activities.map((activity) => activity.startUs))].sort((a, b) => a - b)
  const starts = new Map<string, number>()
  for (const activity of activities) {
    const step = activity.scope.stepId
    if (step == null || activity.kind !== "cpu_scope" || activity.name !== `explorer.step/${step}`) continue
    starts.set(step, Math.min(starts.get(step) ?? Infinity, activity.startUs))
  }
  return [...starts.values()].sort((a, b) => a - b)
}
