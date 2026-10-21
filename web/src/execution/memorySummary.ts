import { sampledDeviceMaximum, type MemoryProjection } from "../../../replay/src/memory.ts"

export function summarizeDeviceMemory(memory: MemoryProjection, uuid: string, sequence: number, live: boolean) {
  const samples = memory.samples.filter((sample) => !live || sample.sequence <= sequence && memory.timeNs != null && sample.timeNs <= memory.timeNs)
  const phases = new Map<string, { phase: string; samples: number; maximum: number }>()
  for (const sample of samples) {
    if (sample.status !== "observed") continue
    const device = sample.devices.find((row) => row.uuid === uuid)
    if (!device) continue
    const phase = phases.get(sample.phase) ?? { phase: sample.phase, samples: 0, maximum: 0 }
    phase.samples += 1
    phase.maximum = Math.max(phase.maximum, device.used)
    phases.set(sample.phase, phase)
  }
  return { maximum: sampledDeviceMaximum(samples, uuid), phases: [...phases.values()] }
}

export type DeviceMemorySummary = ReturnType<typeof summarizeDeviceMemory>
