import type { ProfileActivity, ProfileCacheWrite, ProfileKvObservations, ProfileStep } from "./profiling.ts"

export interface ProfileObservationIndex {
  activitiesById: ReadonlyMap<string, ProfileActivity>
  stepsById: ReadonlyMap<string, ProfileStep>
  gpuActivitiesByStep: ReadonlyMap<string | null, readonly ProfileActivity[]>
  writesByStep: ReadonlyMap<string, readonly ProfileCacheWrite[]>
}

interface CachedIndex {
  activities: ProfileActivity[]
  steps: ProfileStep[]
  writes: ProfileCacheWrite[]
  index: ProfileObservationIndex
}

const indices = new WeakMap<ProfileKvObservations, CachedIndex>()

/** Published recording observations are immutable. Replaced collections or a new
 * validated attachment get a new index; native-time changes reuse source records. */
export function profileObservationIndex(observations: ProfileKvObservations | null | undefined): ProfileObservationIndex | null {
  if (!observations) return null
  const cached = indices.get(observations)
  if (cached && cached.activities === observations.activities && cached.steps === observations.steps && cached.writes === observations.cacheWrites) return cached.index
  const activitiesById = new Map<string, ProfileActivity>()
  const gpuActivitiesByStep = new Map<string | null, ProfileActivity[]>()
  for (const activity of observations.activities) {
    activitiesById.set(activity.id, activity)
    if (!activity.kind.startsWith("cuda_")) continue
    const step = activity.scope.stepId
    const group = gpuActivitiesByStep.get(step) ?? []
    group.push(activity)
    gpuActivitiesByStep.set(step, group)
  }
  const stepsById = new Map(observations.steps.map(step => [step.stepId, step]))
  const writesByStep = new Map<string, ProfileCacheWrite[]>()
  for (const write of observations.cacheWrites) {
    const group = writesByStep.get(write.stepId) ?? []
    group.push(write)
    writesByStep.set(write.stepId, group)
  }
  const index = { activitiesById, stepsById, gpuActivitiesByStep, writesByStep }
  indices.set(observations, { activities: observations.activities, steps: observations.steps, writes: observations.cacheWrites, index })
  return index
}
