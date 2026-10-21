import type { SetupStage } from "../configure/types.ts"

export function targetConnectionStages(status: {
  status: string
  phase: string
  message: string
}): SetupStage[] | null {
  if (status.status !== "connecting") return null
  const active = status.phase === "checking_ssh" ? 0
    : ["copying_project", "preparing_runtime"].includes(status.phase) ? 1
      : ["opening_tunnel", "checking_controller"].includes(status.phase) ? 2
        : -1
  if (active < 0) return null
  return [
    { id: "connect", label: "Reach execution host" },
    { id: "runtime", label: "Prepare CUDA and vLLM runtime" },
    { id: "ready", label: "Confirm loopback controller" },
  ].map((stage, index) => ({
    ...stage,
    status: index < active ? "complete" : index === active ? "active" : "pending",
    detail: index === active ? status.message : undefined,
  }))
}

export function monitorTargetConnection(onProgress: (stages: SetupStage[]) => void): () => void {
  const controller = new AbortController()
  let inFlight = false
  async function poll() {
    if (inFlight || controller.signal.aborted) return
    inFlight = true
    try {
      const response = await fetch("/api/targets/status", { signal: controller.signal })
      if (!response.ok) return
      const stages = targetConnectionStages(await response.json())
      if (!controller.signal.aborted && stages) onProgress(stages)
    } catch {
      // The connection response remains authoritative when a progress poll fails.
    } finally {
      inFlight = false
    }
  }
  const timer = window.setInterval(() => { void poll() }, 1000)
  void poll()
  return () => {
    window.clearInterval(timer)
    controller.abort()
  }
}
