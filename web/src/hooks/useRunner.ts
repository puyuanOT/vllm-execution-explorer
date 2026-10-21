import { useCallback, useEffect, useRef, useState } from "react"

import {
  runnerPath,
  fetchCapabilities,
  type CapabilityReport,
  runIsActive,
  type Limits,
  type Preflight,
  type RunSnapshot,
} from "../api.ts"

export function useRunner(runnerUrl: string, enabled = true) {
  const [health, setHealth] = useState("")
  const [preflight, setPreflight] = useState<Preflight | null>(null)
  const [limits, setLimits] = useState<Limits | null>(null)
  const [live, setLive] = useState<RunSnapshot | null>(null)
  const [capabilities, setCapabilities] = useState<CapabilityReport | null>(null)
  const endpoint = useRef(runnerUrl)
  endpoint.current = runnerUrl
  const refreshVersion = useRef(0)
  const pollVersion = useRef(0)
  const refreshing = useRef(false)

  const refresh = useCallback(async (base = runnerUrl) => {
    const version = ++refreshVersion.current
    const startedAtEndpoint = endpoint.current
    refreshing.current = true
    ++pollVersion.current
    const isCurrent = () => version === refreshVersion.current && endpoint.current === startedAtEndpoint
    try {
      const healthResponse = await fetch(runnerPath(base, "/api/health"))
      if (!isCurrent()) return { health: "", preflight: null, limits: null, live: null, capabilities: null }
      if (!healthResponse.ok) throw new Error("Runner did not answer.")
      const [preflightBody, limitsBody, currentBody, capabilitiesBody] = await Promise.all([
        readJson<Preflight>(runnerPath(base, "/api/preflight")),
        readJson<Limits>(runnerPath(base, "/api/limits")),
        readJson<RunSnapshot>(runnerPath(base, "/api/runs/current")),
        fetchCapabilities(base),
      ])
      if (isCurrent()) {
        setHealth("Runner is reachable.")
        setPreflight(preflightBody)
        setLimits(limitsBody)
        setLive(currentBody)
        setCapabilities(capabilitiesBody)
      }
      return {
        health: "Runner is reachable.",
        preflight: preflightBody,
        limits: limitsBody,
        live: currentBody,
        capabilities: capabilitiesBody,
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Runner is unreachable. Bundled recordings still open."
      if (isCurrent()) {
        setHealth(message)
        setPreflight(null)
        setLimits(null)
        setLive(null)
        setCapabilities(null)
      }
      return { health: message, preflight: null, limits: null, live: null, capabilities: null }
    } finally {
      if (version === refreshVersion.current) refreshing.current = false
    }
  }, [runnerUrl])

  useEffect(() => {
    if (enabled) {
      setHealth("")
      setPreflight(null)
      setLimits(null)
      setLive(null)
      setCapabilities(null)
      void refresh()
    }
    return () => {
      ++refreshVersion.current
      ++pollVersion.current
      refreshing.current = false
    }
  }, [enabled, refresh])

  useEffect(() => {
    if (!enabled) return
    let stopped = false
    let inFlight = false
    const delay = runIsActive(live?.status) ? 1000 : 2500
    const id = window.setInterval(() => {
      if (inFlight || refreshing.current) return
      inFlight = true
      const version = ++pollVersion.current
      void readJson<RunSnapshot>(runnerPath(runnerUrl, "/api/runs/current"))
        .then((body) => {
          if (!stopped && version === pollVersion.current && runnerUrl === endpoint.current) setLive(body)
        })
        .catch(() => undefined)
        .finally(() => { inFlight = false })
    }, delay)
    return () => {
      stopped = true
      window.clearInterval(id)
    }
  }, [enabled, runnerUrl, live?.status])

  return { health, preflight, limits, live, capabilities, refresh, setHealth }
}

async function readJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Runner request failed (${response.status}).`)
  return response.json() as Promise<T>
}
