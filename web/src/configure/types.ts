export interface ConfigureRuntimeInputs {
  sshIdentity: File | null
  huggingFaceToken: string
}

export type ReadinessTone = "neutral" | "ready" | "warning" | "error"

export interface ReadinessState {
  label: string
  message: string
  tone: ReadinessTone
}

export interface SetupStage {
  id: string
  label: string
  status: "pending" | "active" | "complete" | "error"
  detail?: string
}
