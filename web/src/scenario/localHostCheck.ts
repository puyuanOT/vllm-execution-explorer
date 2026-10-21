import type { ReadinessState, SetupStage } from "../configure/types.ts"

export interface LocalPreflight {
  ok: boolean
  message: string
}

export interface LocalHostCheck {
  readiness: ReadinessState
  stages: SetupStage[]
}

const RUNTIME_LABEL = "Inspect CUDA and vLLM runtime"
const READY_LABEL = "Confirm loopback controller"

function stage(id: string, label: string, status: SetupStage["status"], detail?: string): SetupStage {
  return { id, label, status, detail }
}

function connectStage(status: SetupStage["status"], detail: string): SetupStage {
  return stage("connect", "Reach execution host", status, detail)
}

export function localHostCheckPresentation(
  engine: "fixture" | "vllm",
  health: string,
  preflight: LocalPreflight | null,
): LocalHostCheck {
  if (!preflight) {
    const detail = health || "The local controller did not respond."
    return {
      readiness: {
        label: "Unavailable",
        message: "Start the local controller and retry.",
        tone: "error",
      },
      stages: [
        connectStage("error", detail),
        stage("runtime", RUNTIME_LABEL, "error", "The controller did not return a runtime check."),
        stage("ready", READY_LABEL, "error", detail),
      ],
    }
  }

  if (engine === "fixture") {
    return {
      readiness: {
        label: "Ready",
        message: "The demo fixture can run on this controller. It is synthetic evidence and does not require CUDA.",
        tone: "ready",
      },
      stages: [
        connectStage("complete", "Local controller"),
        stage("runtime", RUNTIME_LABEL, "complete", "Demo fixture selected. This check does not certify CUDA or vLLM."),
        stage("ready", READY_LABEL, "complete", health),
      ],
    }
  }

  return {
    readiness: preflight.ok
      ? { label: "Ready", message: preflight.message, tone: "ready" }
      : { label: "Needs GPU host", message: preflight.message, tone: "warning" },
    stages: [
      connectStage("complete", "Local controller"),
      stage("runtime", RUNTIME_LABEL, preflight.ok ? "complete" : "error", preflight.message),
      stage("ready", READY_LABEL, "complete", health),
    ],
  }
}
