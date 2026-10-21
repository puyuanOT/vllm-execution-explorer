import type { RecordingManifest } from "../../../replay/src/reduce.ts"

export type EvidenceKind = "synthetic" | "gpu" | "unverified" | "draft" | "recorded"

export function recordingEvidence(manifest: RecordingManifest, draftPreview: boolean): { kind: EvidenceKind; text: string } {
  if (draftPreview) return { kind: "draft", text: "Draft preview. Values are planned inputs and are not execution evidence." }
  if (manifest.demo_fixture === true || manifest.execution_kind === "fixture") {
    return { kind: "synthetic", text: "Demo fixture. Teaching pace is synthetic and is not GPU evidence." }
  }
  if (manifest.execution_kind === "gpu") {
    const provenance = manifest.provenance as { real_gpu?: unknown } | null | undefined
    const hardware = manifest.hardware as { gpu?: unknown } | null | undefined
    if (provenance?.real_gpu === true && hardware?.gpu) {
      return { kind: "gpu", text: "Observed GPU recording. Host intervals are not GPU kernel durations." }
    }
    return { kind: "unverified", text: "GPU execution has not been verified for this capture." }
  }
  return { kind: "recorded", text: "Recorded execution evidence." }
}
