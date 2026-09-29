import { useCallback, useEffect, useState } from "react"

import {
  runnerPath,
  fetchCapabilities,
  type CapabilityReport,
  runIsActive,
  type Limits,
  type Preflight,
  type RunSnapshot,
} from "../api.ts"

export function useRunner(runnerUrl: string) {
  const [health, setHealth] = useState("")
  const [preflight, setPreflight] = useState<Preflight | null>(null)
  const [limits, setLimits] = useState<Limits | null>(null)
  const [live, setLive] = useState<RunSnapshot | null>(null)
  const [capabilities, setCapabilities] = useState<CapabilityReport | null>(null)

  const refresh = useCallback(async (base = runnerUrl) => {
    try {
      const healthResponse = await fetch(runnerPath(base, "/api/health"))
      if (!healthResponse.ok) throw new Error("Runner did not answer.")
      const [preflightBody, limitsBody, currentBody, capabilitiesBody] = await Promise.all([
        fetch(runnerPath(base, "/api/preflight")).then((response) => response.json() as Promise<Preflight>),
        fetch(runnerPath(base, "/api/limits")).then((response) => response.json() as Promise<Limits>),
        fetch(runnerPath(base, "/api/runs/current")).then((response) => response.json() as Promise<RunSnapshot>),
        fetchCapabilities(base),
      ])
      setHealth("Runner is reachable.")
      setPreflight(preflightBody)
      setLimits(limitsBody)
      setLive(currentBody)
      setCapabilities(capabilitiesBody)
      return {
        health: "Runner is reachable.",
        preflight: preflightBody,
        limits: limitsBody,
        live: currentBody,
        capabilities: capabilitiesBody,
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Runner is unreachable. Bundled recordings still open."
      setHealth(message)
      setPreflight(null)
      setLimits(null)
      setLive(null)
      setCapabilities(null)
      return { health: message, preflight: null, limits: null, live: null, capabilities: null }
    }
  }, [runnerUrl])

  useEffect(() => {
    void refresh()
  }, [refresh])

  useEffect(() => {
    const delay = runIsActive(live?.status) ? 1000 : 2500
    const id = window.setInterval(() => {
      void fetch(runnerPath(runnerUrl, "/api/runs/current"))
        .then((response) => response.json() as Promise<RunSnapshot>)
        .then((body) => setLive(body))
        .catch(() => undefined)
    }, delay)
    return () => window.clearInterval(id)
  }, [runnerUrl, live?.status])

  return { health, preflight, limits, live, capabilities, refresh, setHealth }
}
