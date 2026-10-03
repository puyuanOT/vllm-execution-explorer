import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import { attachContent, createPlayer, type RecordingManifest } from "../../replay/src/reduce.ts"
import {
  buildTimeline,
  elapsedMsAt,
  nextStepSequence,
  previousStepSequence,
  projectSamples,
  promptsFromContent,
  sequenceAtElapsed,
} from "../../replay/src/timeline.ts"
import {
  fetchArtifactTransfer,
  loadBundled,
  loadCatalog,
  loadFiles,
  loadRunnerRecording,
  prepareArtifactTransfer,
  runnerPath,
  runIsActive,
  startArtifactTransfer,
  type CatalogEntry,
  type LoadedRecording,
} from "./api.ts"
import {
  ConfigurePage,
  type ConfigureRuntimeInputs,
  type ReadinessState,
  type SetupStage,
} from "./configure/index.ts"
import { focusValidationField } from "./configure/focusValidation.ts"
import { ExecutionWorkspace } from "./execution/ExecutionWorkspace.tsx"
import { PlaybackTransport } from "./execution/PlaybackTransport.tsx"
import { SelectionInspector } from "./execution/SelectionInspector.tsx"
import type { PlaybackMode, Selection, WorkspaceView } from "./execution/types.ts"
import { useAppRoute } from "./hooks/useAppRoute.ts"
import { useLiveTrace } from "./hooks/useLiveTrace.ts"
import { useRunner } from "./hooks/useRunner.ts"
import { useScenarioDraft } from "./hooks/useScenarioDraft.ts"
import { eventAt, nextSequence, previousSequence, SPEEDS } from "./playback.ts"
import { defaultSettings } from "./scenario/example.ts"
import { localHostCheckPresentation } from "./scenario/localHostCheck.ts"
import type { ScenarioAction } from "./scenario/reducer.ts"
import { expectedRunFailure, shouldLoadExpectedRecording } from "./scenario/runCompletion.ts"
import { cloneScenario, scenarioFromRecordedPlan, toRunBody, workloadPlan } from "./scenario/serialize.ts"
import {
  createSshVerification,
  remoteRuntimeError,
  sshVerificationError,
  type SshVerification,
} from "./scenario/sshVerification.ts"
import { loadSettings, saveSettings } from "./scenario/storage.ts"
import type { EngineSettings, ScenarioDraft, TimeUnit } from "./scenario/types.ts"
import { validateScenario, type FieldIssue } from "./scenario/validate.ts"
import { AppHeader } from "./shell/AppHeader.tsx"
import { ConfirmDialog } from "./shell/ConfirmDialog.tsx"
import { Modal } from "./shell/Modal.tsx"
import { RecordingLibrary } from "./shell/RecordingLibrary.tsx"
import { requestColor } from "./theme.ts"

const EMPTY_STAGES: SetupStage[] = [
  { id: "connect", label: "Reach execution host", status: "pending" },
  { id: "runtime", label: "Inspect CUDA and vLLM runtime", status: "pending" },
  { id: "ready", label: "Confirm loopback controller", status: "pending" },
]

export function App() {
  const { route, navigate } = useAppRoute()
  const draft = useScenarioDraft()
  const [settings, setSettings] = useState<EngineSettings>(() => loadSettings() ?? defaultSettings())
  const runner = useRunner(settings.runnerUrl)
  const [recording, setRecording] = useState<LoadedRecording | null>(null)
  const [lastCaptured, setLastCaptured] = useState<LoadedRecording | null>(null)
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [catalogError, setCatalogError] = useState("")
  const [loadError, setLoadError] = useState("")
  const [runError, setRunError] = useState("")
  const [issues, setIssues] = useState<FieldIssue[]>([])
  const [unit, setUnit] = useState<TimeUnit>("ms")
  const [submitted, setSubmitted] = useState<ScenarioDraft | null>(null)
  const [sequence, setSequence] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1)
  const [mode, setMode] = useState<PlaybackMode>("events")
  const [followLive, setFollowLive] = useState(true)
  const [selection, setSelection] = useState<Selection | null>(null)
  const [view, setView] = useState<WorkspaceView>("spatial")
  const [compact, setCompact] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [forceReduced, setForceReduced] = useState(false)
  const [systemReduced, setSystemReduced] = useState(false)
  const [announce, setAnnounce] = useState("")
  const [validationMessage, setValidationMessage] = useState("")
  const [recordingsOpen, setRecordingsOpen] = useState(false)
  const [confirmResetOpen, setConfirmResetOpen] = useState(false)
  const [checkingHost, setCheckingHost] = useState(false)
  const [preparingRun, setPreparingRun] = useState(false)
  const [preparationStatus, setPreparationStatus] = useState("")
  const [setupStages, setSetupStages] = useState<SetupStage[]>(EMPTY_STAGES)
  const [hostCheck, setHostCheck] = useState<ReadinessState | null>(null)
  const [sshVerification, setSshVerification] = useState<SshVerification | null>(null)
  const expectedRun = useRef("")
  const userOpened = useRef(false)
  const autoLessonAttempted = useRef(false)
  const reduced = forceReduced || systemReduced

  useEffect(() => {
    document.title = `${route === "explore" ? "Explore" : "Configure"} · vLLM Execution Explorer`
  }, [route])

  const acceptLiveRecording = useCallback((loaded: LoadedRecording) => {
    if (!expectedRun.current || loaded.source !== expectedRun.current || userOpened.current) return
    const completion = typeof loaded.manifest.completion_status === "string"
      ? loaded.manifest.completion_status
      : ""
    if (completion !== "completed") {
      const message = runner.live?.message.trim()
        || (completion === "cancelled" ? "Capture was cancelled." : "Capture failed during preparation or execution.")
      setRunError(message)
      setValidationMessage(message)
      setAnnounce(message)
      setPlaying(false)
      setFollowLive(false)
      navigate("configure")
      return
    }
    expectedRun.current = ""
    setRecording(loaded)
    setLastCaptured(loaded)
    setLoadError("")
    setSequence(0)
    setPlaying(false)
    setFollowLive(false)
    setSelection(null)
    setAnnounce("Capture finalized. The recording is ready to explore.")
    navigate("explore")
  }, [navigate, runner.live?.message])

  const trace = useLiveTrace(settings.runnerUrl, runner.live, acceptLiveRecording)

  useEffect(() => saveSettings(settings), [settings])

  useEffect(() => {
    void loadCatalog().then(setCatalog).catch((error: unknown) => {
      setCatalogError(error instanceof Error ? error.message : "Could not load recordings.")
    })
  }, [])

  useEffect(() => {
    if (route !== "explore" || recording || expectedRun.current || autoLessonAttempted.current || catalog.length === 0) return
    autoLessonAttempted.current = true
    void loadBundled(catalog[0].id).then(openLoaded).catch((error: unknown) => {
      setLoadError(error instanceof Error ? error.message : "Could not open the starter recording.")
    })
  }, [catalog, recording, route])

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)")
    const apply = () => setSystemReduced(media.matches)
    apply()
    media.addEventListener("change", apply)
    return () => media.removeEventListener("change", apply)
  }, [])

  useEffect(() => {
    const media = window.matchMedia("(max-width: 1100px)")
    const apply = () => setCompact(media.matches)
    apply()
    media.addEventListener("change", apply)
    return () => media.removeEventListener("change", apply)
  }, [])

  useEffect(() => {
    if (recording && userOpened.current) return
    const status = runner.live?.status
    if (!status || status === "idle") return
    setAnnounce(`Run ${runner.live?.run_id ?? ""} is ${statusLabel(status)}.`)
  }, [recording, runner.live?.run_id, runner.live?.status])

  useEffect(() => {
    const message = expectedRunFailure(expectedRun.current, runner.live, userOpened.current)
    if (!message) return
    expectedRun.current = ""
    setRunError(message)
    setValidationMessage(message)
    setAnnounce(message)
    setPlaying(false)
    setFollowLive(false)
    navigate("configure")
  }, [navigate, runner.live?.finalized, runner.live?.message, runner.live?.run_id, runner.live?.status])

  useEffect(() => {
    if (!shouldLoadExpectedRecording(expectedRun.current, runner.live, userOpened.current)) return
    const runId = runner.live.run_id
    let stopped = false
    let timer = 0
    let retryDelay = 150
    const load = () => {
      if (stopped || expectedRun.current !== runId || userOpened.current) return
      void loadRunnerRecording(settings.runnerUrl, runId)
        .then((loaded) => {
          if (!stopped) acceptLiveRecording(loaded)
        })
        .catch(() => {
          if (stopped || expectedRun.current !== runId || userOpened.current) return
          timer = window.setTimeout(load, retryDelay)
          retryDelay = Math.min(retryDelay * 2, 2000)
        })
    }
    load()
    return () => {
      stopped = true
      window.clearTimeout(timer)
    }
  }, [acceptLiveRecording, runner.live?.finalized, runner.live?.run_id, runner.live?.status, settings.runnerUrl])

  const showingLive = !recording
    && expectedRun.current !== ""
    && runner.live?.run_id === expectedRun.current
    && (runIsActive(runner.live?.status) || trace.events.length > 0 || trace.finalized)
  const liveManifest = useMemo<RecordingManifest | null>(() => {
    if (!showingLive) return null
    if (trace.manifest) return trace.manifest
    return {
      run_id: runner.live?.run_id ?? undefined,
      workload_plan: trace.plan ?? (submitted ? { ...workloadPlan(submitted), synthesized: false } : undefined),
      controller_origin: trace.origin ?? runner.live?.controller_origin,
      arrivals: trace.arrivals,
      completion_status: runner.live?.status,
      execution_kind: settings.engine === "fixture" ? "fixture" : "gpu",
      demo_fixture: settings.engine === "fixture",
      content_retained: true,
    }
  }, [runner.live, settings.engine, showingLive, submitted, trace])

  const previewManifest = useMemo<RecordingManifest>(() => ({
    workload_plan: { ...workloadPlan(draft.scenario), synthesized: false },
    arrivals: [],
    completion_status: "idle",
  }), [draft.scenario])

  const manifest = recording?.manifest ?? liveManifest ?? previewManifest
  const events = recording?.events ?? (showingLive ? trace.events : [])
  const content = recording?.content ?? (showingLive ? trace.content : [])
  const showingDraft = !recording && !showingLive

  const built = useMemo(() => {
    try {
      return { player: createPlayer(events, manifest), error: "" }
    } catch (error) {
      return { player: null, error: error instanceof Error ? error.message : "Replay failed." }
    }
  }, [events, manifest])
  const viewState = useMemo(() => built.player ? attachContent(built.player.at(sequence), content) : null, [built.player, content, sequence])
  const index = useMemo(() => buildTimeline(events, manifest), [events, manifest])
  const samples = useMemo(() => {
    if (!viewState) return []
    return projectSamples(index, viewState, sequence, {
      completionStatus: typeof manifest.completion_status === "string" ? manifest.completion_status : null,
      contentPrompts: promptsFromContent(content),
      submittedPrompts: showingLive && submitted ? promptMap(submitted) : showingDraft ? promptMap(draft.scenario) : null,
      arrivals: Array.isArray(manifest.arrivals) ? manifest.arrivals as Array<Record<string, unknown>> : [],
    })
  }, [content, draft.scenario, index, manifest, sequence, showingDraft, showingLive, submitted, viewState])
  const colors = useMemo(() => {
    const order = samples.map((item) => item.sample.id)
    return Object.fromEntries(order.map((id) => [id, requestColor(order, id)]))
  }, [samples])
  const highlighted = useMemo(() => highlight(selection, viewState), [selection, viewState])
  const maxSequence = built.player?.maxSequence ?? 0
  const atEnd = sequence >= maxSequence && maxSequence > 0
  const currentEvent = eventAt(events, sequence)
  const elapsed = elapsedMsAt(index, sequence)
  const valueText = currentEvent
    ? `Event ${currentEvent.sequence}, ${currentEvent.event_type}${elapsed == null ? "" : `, ${elapsed} ms`}. ${index.roster.timeBasisLabel}`
    : `Start. ${index.roster.timeBasisLabel}`

  useEffect(() => {
    if (followLive && runIsActive(runner.live?.status)) setSequence(maxSequence)
  }, [followLive, maxSequence, runner.live?.status])

  useEffect(() => {
    if (!playing || mode !== "events" || events.length === 0 || followLive) return
    const timer = window.setInterval(() => {
      setSequence((current) => {
        const next = nextSequence(events, current)
        if (next === current) setPlaying(false)
        return next
      })
    }, Math.max(80, 700 / speed))
    return () => window.clearInterval(timer)
  }, [events, followLive, mode, playing, speed])

  const sequenceRef = useRef(sequence)
  sequenceRef.current = sequence
  useEffect(() => {
    if (!playing || mode !== "recorded") return
    const clock = { value: elapsedMsAt(index, sequenceRef.current) ?? 0, last: performance.now() }
    let frame = 0
    const tick = (now: number) => {
      clock.value += (now - clock.last) * speed
      clock.last = now
      const next = sequenceAtElapsed(index, clock.value)
      setSequence(next)
      if (next >= index.maxSequence) {
        setPlaying(false)
        return
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [index, mode, playing, speed])

  useEffect(() => {
    if (route !== "explore") return
    function onKey(event: KeyboardEvent) {
      if (isTypingTarget(event.target)) return
      if (event.code === "Escape" && inspectorOpen) {
        event.preventDefault()
        setInspectorOpen(false)
        return
      }
      const onControl = event.target instanceof HTMLElement && Boolean(event.target.closest("button, a, summary"))
      if (event.code === "Space" && onControl) return
      if (events.length === 0) return
      if (event.code === "Space") {
        event.preventDefault()
        setFollowLive(false)
        setPlaying((current) => !current)
      } else if (event.code === "ArrowRight") {
        event.preventDefault()
        seek(nextSequence(events, sequence))
      } else if (event.code === "ArrowLeft") {
        event.preventDefault()
        seek(previousSequence(events, sequence))
      } else if (event.code === "BracketRight") {
        event.preventDefault()
        const next = nextStepSequence(index.steps, sequence)
        if (next != null) seek(next)
      } else if (event.code === "BracketLeft") {
        event.preventDefault()
        const previous = previousStepSequence(index.steps, sequence)
        if (previous != null) seek(previous)
      } else if (event.code === "Home") {
        event.preventDefault()
        seek(0)
      } else if (event.code === "End") {
        event.preventDefault()
        seek(maxSequence)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [events, index.steps, inspectorOpen, maxSequence, route, sequence])

  function seek(next: number, nextSelection?: Selection) {
    setFollowLive(false)
    setPlaying(false)
    setSequence(Math.max(0, next))
    if (nextSelection) {
      setSelection(nextSelection)
      if (compact && nextSelection.kind !== "component") setInspectorOpen(true)
    }
  }

  function choose(next: Selection) {
    setSelection(next)
    if (compact && next.kind !== "component") setInspectorOpen(true)
    if (next.kind === "step") {
      const step = index.steps.find((item) => item.id === next.id) ?? viewState?.steps.find((item) => item.id === next.id)
      if (step) seek(step.sequence, next)
    }
  }

  function openLoaded(loaded: LoadedRecording) {
    userOpened.current = true
    setRecording(loaded)
    setSequence(0)
    setPlaying(false)
    setFollowLive(false)
    setSelection(null)
    setLoadError("")
    setRunError("")
    setValidationMessage("")
    setAnnounce("Recording opened.")
    setRecordingsOpen(false)
    navigate("explore")
  }

  async function checkHost(runtime: ConfigureRuntimeInputs) {
    if (checkingHost || preparingRun || runIsActive(runner.live?.status)) return
    setCheckingHost(true)
    setRunError("")
    setValidationMessage("")
    setAnnounce("Checking the execution host.")
    setSshVerification(null)
    setSetupStages(EMPTY_STAGES.map((stage, index) => ({ ...stage, status: index === 0 ? "active" : "pending" })))
    try {
      if (settings.hostKind === "ssh") {
        const identity = runtime.sshIdentity
        if (!identity) throw new Error("Choose the SSH private key for this host.")
        const form = new FormData()
        form.set("user", settings.sshUser.trim())
        form.set("host", settings.sshHost.trim())
        form.set("port", settings.sshPort.trim())
        form.set("automatic_setup", String(settings.automaticSetup))
        form.set("identity", identity)
        const response = await fetch("/api/targets/connect", { method: "POST", body: form })
        if (!response.ok) throw new Error(await responseDetail(response, "Could not connect to the GPU host."))
        const body = await response.json() as { runner_endpoint?: string; message?: string; gpu?: string | null; vllm_version?: string | null }
        if (!body.runner_endpoint) throw new Error("The GPU host connected without a runner endpoint.")
        const runtimeProblem = remoteRuntimeError(body.gpu, body.vllm_version)
        if (runtimeProblem) throw new Error(runtimeProblem)
        const checked = await runner.refresh(body.runner_endpoint)
        if (!checked.preflight || !checked.capabilities) {
          throw new Error(`The SSH tunnel opened, but the remote runner could not be verified. ${checked.health}`)
        }
        if (!checked.preflight.ok) throw new Error(checked.preflight.message || "The remote vLLM runtime is not ready.")
        setSettings((current) => ({ ...current, runnerUrl: body.runner_endpoint ?? current.runnerUrl }))
        setSshVerification(createSshVerification(settings, identity, body.runner_endpoint))
        setSetupStages([
          { id: "connect", label: "Reach execution host", status: "complete", detail: settings.sshHost },
          { id: "runtime", label: "Inspect CUDA and vLLM runtime", status: "complete", detail: [body.gpu, body.vllm_version && `vLLM ${body.vllm_version}`].filter(Boolean).join(" · ") },
          { id: "ready", label: "Confirm loopback controller", status: "complete", detail: body.runner_endpoint },
        ])
        setHostCheck({ label: "Connected", message: body.message ?? "The remote loopback runner is ready.", tone: "ready" })
        setAnnounce("The remote execution host is ready.")
      } else {
        if (settings.runnerUrl) setSettings((current) => ({ ...current, runnerUrl: "" }))
        const checked = await runner.refresh()
        const presentation = localHostCheckPresentation(
          settings.engine,
          checked.health,
          checked.preflight ? { ok: checked.preflight.ok, message: checked.preflight.message } : null,
        )
        setSetupStages(presentation.stages)
        setHostCheck(presentation.readiness)
        setAnnounce("The local execution host check finished.")
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Host check failed."
      setHostCheck({ label: "Check failed", message, tone: "error" })
      setSetupStages((current) => current.map((stage) => stage.status === "active" ? { ...stage, status: "error", detail: message } : stage))
      setRunError(message)
    } finally {
      setCheckingHost(false)
    }
  }

  async function runSequence(next: ScenarioDraft, runtime: ConfigureRuntimeInputs) {
    if (checkingHost || preparingRun || runIsActive(runner.live?.status)) return
    draft.replace(next)
    const quantizationCapabilities = settings.hostKind === "ssh" && !settings.runnerUrl
      ? undefined
      : runner.capabilities?.model_options.quantization
    const problems = validateScenario(next, runner.limits, settings, quantizationCapabilities)
    if (settings.engine === "vllm" && settings.hostKind === "local" && runner.preflight?.ok !== true) {
      problems.push({
        fieldId: "host-local",
        message: runner.preflight?.message ?? "Check a local CUDA Linux execution host before capture.",
      })
    }
    const sshProblem = sshVerificationError(settings, runtime.sshIdentity, sshVerification)
    if (sshProblem) problems.push({ fieldId: "ssh-key", message: sshProblem })
    if (problems.length > 0) {
      setIssues(problems)
      setValidationMessage(problems[0]?.message ?? "Check the configuration.")
      focusValidationField(problems[0]?.fieldId ?? "")
      return
    }
    setIssues([])
    setValidationMessage("")
    setRunError("")
    setPreparingRun(true)
    setPreparationStatus("Preparing model artifacts…")
    try {
      const effectiveSettings = await transferArtifactsIfNeeded(settings)
      setPreparationStatus("Starting capture…")
      const body = {
        ...toRunBody(next, effectiveSettings),
        huggingface_token: runtime.huggingFaceToken.trim() || null,
      }
      const response = await fetch(runnerPath(effectiveSettings.runnerUrl, "/api/runs"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      })
      if (!response.ok) throw new Error(await responseDetail(response, "The run was refused."))
      const result = await response.json() as { run_id?: string }
      if (!result.run_id) throw new Error("The runner did not return a run id.")
      expectedRun.current = result.run_id
      userOpened.current = false
      setSubmitted(cloneScenario(next, next.id))
      setRecording(null)
      setSequence(0)
      setPlaying(false)
      setFollowLive(true)
      setSelection(null)
      setAnnounce("Capture started. Preparation completes before workload time zero.")
      await runner.refresh()
    } catch (error) {
      const message = error instanceof Error ? error.message : "The runner is unreachable."
      setRunError(message)
      setValidationMessage(message)
    } finally {
      setPreparingRun(false)
      setPreparationStatus("")
    }
  }

  async function transferArtifactsIfNeeded(current: EngineSettings): Promise<EngineSettings> {
    if (current.hostKind !== "ssh") return current
    let next = current
    if (current.modelSource === "operator") {
      const destination = await completeTransfer(current.model, normalizeRemoteDestination(current.modelDestination, current.sshUser, "model"))
      next = { ...next, modelSource: "host", model: destination, modelDestination: destination }
    }
    for (const adapter of next.loraAdapters) {
      if (adapter.source !== "operator") continue
      const destination = await completeTransfer(adapter.location, normalizeRemoteDestination("", current.sshUser, `adapters/${adapter.id}`))
      next = {
        ...next,
        loraAdapters: next.loraAdapters.map((item) => item.id === adapter.id ? { ...item, source: "host", location: destination } : item),
      }
    }
    return next
  }

  async function completeTransfer(source: string, destination: string) {
    setAnnounce(`Inspecting ${source} before transfer.`)
    const prepared = await prepareArtifactTransfer("", source, destination)
    const size = formatBytes(prepared.total_bytes)
    setPreparationStatus(`${prepared.source_name}: transferring ${size} to the execution host…`)
    let transfer = await startArtifactTransfer("", prepared.transfer_id)
    while (transfer.status === "prepared" || transfer.status === "transferring") {
      await delay(500)
      transfer = await fetchArtifactTransfer("", transfer.transfer_id)
      setAnnounce(`${transfer.source_name}: ${transfer.message}`)
      setPreparationStatus(`${transfer.source_name}: ${transfer.message} (${size})`)
    }
    if (transfer.status !== "completed") throw new Error(transfer.message || "Artifact transfer failed.")
    setPreparationStatus(`${transfer.source_name}: transfer complete (${size}).`)
    return transfer.destination
  }

  async function cancelRun() {
    try {
      const response = await fetch(runnerPath(settings.runnerUrl, "/api/runs/current/cancel"), { method: "POST" })
      if (!response.ok) throw new Error(await responseDetail(response, "Could not cancel the capture."))
      await runner.refresh()
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "Could not cancel the capture.")
    }
  }

  function editAsNew() {
    const copied = scenarioFromRecordedPlan(manifest.workload_plan, promptsFromContent(content))
    if (!copied) {
      setAnnounce("This recording has no arrival plan to copy.")
      return
    }
    draft.replace(copied)
    navigate("configure")
    setAnnounce("A copy of the recorded sequence is ready to edit.")
  }

  function updateSettings(next: EngineSettings) {
    const sshConnectionChanged = settings.hostKind !== next.hostKind
      || settings.sshUser !== next.sshUser
      || settings.sshHost !== next.sshHost
      || settings.sshPort !== next.sshPort
      || settings.automaticSetup !== next.automaticSetup
      || settings.runnerUrl !== next.runnerUrl
    if (sshConnectionChanged && next.hostKind === "ssh" && settings.runnerUrl === next.runnerUrl) {
      next = { ...next, runnerUrl: "" }
    }
    if (settings.hostKind === "ssh" && next.hostKind === "local") next = { ...next, runnerUrl: "" }
    setSettings(next)
    setHostCheck(null)
    setIssues([])
    setRunError("")
    setValidationMessage("")
    if (sshConnectionChanged) {
      setSshVerification(null)
      setSetupStages(EMPTY_STAGES)
    }
  }

  function updateScenario(action: ScenarioAction) {
    draft.dispatch(action)
    setIssues([])
    setValidationMessage("")
    setRunError("")
  }

  function undoScenarioDeletion() {
    draft.undoDelete()
    setIssues([])
    setValidationMessage("")
    setRunError("")
  }

  function startEmptyScenario() {
    draft.startEmpty()
    setIssues([])
    setValidationMessage("")
    setRunError("")
  }

  function resetScenario() {
    draft.clearDraft()
    setIssues([])
    setValidationMessage("")
    setRunError("")
  }

  const evidence = evidenceText(manifest, showingDraft)
  const warning = [
    viewState?.trust === "unknown" ? `Trace gap at event ${viewState.gap?.sequence ?? "?"}. Later state is unknown.` : "",
    manifest.content_retained === false ? "Prompt and output text were not retained." : "",
    loadError || built.error,
    !trace.connected && runIsActive(runner.live?.status) ? "Live trace disconnected. Catching up from the last cursor." : "",
  ].filter(Boolean).join(" ")
  const runnerActive = runIsActive(runner.live?.status) && runner.live?.run_id === expectedRun.current
  const running = preparingRun || runnerActive
  const status = runnerActive
    ? `${statusLabel(runner.live?.status ?? "running")}${runner.live?.message ? ` · ${runner.live.message}` : ""}`
    : preparingRun ? preparationStatus || "Preparing capture"
    : recording ? `Recording · ${recording.source}` : runner.health || "Runner unchecked"
  const hostReadiness = hostCheck ?? deriveHostReadiness(settings, runner.health, runner.preflight?.ok, runner.preflight?.message)
  const runBlockedReason = settings.engine === "vllm" && settings.hostKind === "local" && runner.preflight?.ok !== true
    ? hostReadiness.message
    : undefined
  const quantizationCapabilities = settings.hostKind === "ssh" && !settings.runnerUrl
    ? undefined
    : runner.capabilities?.model_options.quantization
  const modelReadiness = deriveModelReadiness(settings, runner.capabilities, quantizationCapabilities)
  const vllmReadiness = deriveVllmReadiness(settings, runner.capabilities)
  const effectiveSettings = {
    "Execution source": settings.engine === "fixture" ? "Demo fixture" : "Instrumented vLLM",
    "Context length": Number(settings.maxModelLen),
    "Concurrent sequences": Number(settings.maxNumSeqs),
    "Scheduled-token budget": Number(settings.tokenBudget),
    "Chunked prefill": settings.chunkedPrefill,
    "Prefix cache": settings.prefixCaching,
    "Fresh cache contract": "Required; verified during preparation",
  }
  const editableScenario = scenarioFromRecordedPlan(manifest.workload_plan, promptsFromContent(content))

  return (
    <div className={route === "explore" ? "flex h-dvh min-h-0 flex-col overflow-hidden bg-slate-950 text-slate-100" : "min-h-dvh bg-slate-950 text-slate-100"}>
      <p className="sr-only" aria-live="polite">{announce}</p>
      <p className="sr-only" aria-live="assertive">{validationMessage}</p>
      <AppHeader route={route} status={status} onNavigate={navigate} onRecordings={() => setRecordingsOpen(true)} />

      {route === "configure" ? (
        <ConfigurePage
          scenario={draft.scenario}
          settings={settings}
          limits={runner.limits}
          issues={issues}
          unit={unit}
          storageNote={draft.storageNote}
          storageError={draft.storageError}
          hostReadiness={hostReadiness}
          modelReadiness={modelReadiness}
          quantizationCapabilities={quantizationCapabilities}
          vllmReadiness={vllmReadiness}
          setupStages={setupStages}
          effectiveSettings={effectiveSettings}
          checkingHost={checkingHost}
          running={running}
          runBlockedReason={runBlockedReason}
          runStatus={running ? status : undefined}
          runError={runError}
          onUnit={setUnit}
          onAction={updateScenario}
          onSettings={updateSettings}
          onInvalid={(issue) => {
            setIssues([issue])
            setValidationMessage(issue.message)
          }}
          onCheckHost={(runtime) => void checkHost(runtime)}
          onSshIdentityChange={() => {
            setSshVerification(null)
            setHostCheck(null)
            setSetupStages(EMPTY_STAGES)
          }}
          onRun={(scenario, runtime) => void runSequence(scenario, runtime)}
          onCancel={runnerActive ? () => void cancelRun() : undefined}
          canUndo={draft.canUndo}
          onUndo={undoScenarioDeletion}
          onStartEmpty={startEmptyScenario}
          onClearDraft={() => setConfirmResetOpen(true)}
        />
      ) : (
        <>
          <div className="flex min-h-11 flex-wrap items-center gap-2 border-b border-line bg-surface px-4 py-2 text-xs text-muted">
            <span className="mr-auto truncate">{recording ? `Open recording: ${recording.source}` : showingLive ? "Live capture" : "Draft preview"}</span>
            {editableScenario ? <button type="button" onClick={editAsNew}>Edit as new sequence</button> : null}
            {compact && selection ? <button type="button" onClick={() => setInspectorOpen(true)}>Open inspector</button> : null}
          </div>
          <main className="explore-grid min-h-0 flex-1">
            <ExecutionWorkspace
              samples={samples}
              state={viewState}
              requestOrder={built.player?.requestOrder ?? []}
              colors={colors}
              selection={selection}
              highlightedIds={highlighted.ids}
              highlightedStepIds={highlighted.steps}
              view={view}
              onView={setView}
              animate={playing && !reduced && view === "spatial"}
              timeBasisLabel={showingDraft
                ? "Draft plan. No execution evidence has been recorded."
                : `${index.roster.timeBasisLabel}${index.roster.groupingKnown ? "" : " Arrival groups were not recorded."}`}
              evidence={evidence}
              warning={warning}
              submittedNote={showingLive && submitted ? "Showing the immutable submitted sequence." : ""}
              onSelect={choose}
            />
            {!compact ? (
              <SelectionInspector
                selection={selection}
                samples={samples}
                state={viewState}
                index={index}
                manifest={manifest}
                sequence={sequence}
                showClose={false}
                onSeek={seek}
                onSelect={choose}
                onClose={() => setInspectorOpen(false)}
              />
            ) : null}
          </main>
          <PlaybackTransport
            disabled={events.length === 0 && samples.length === 0}
            playing={playing}
            atEnd={atEnd}
            speed={speed}
            mode={mode}
            recordedAvailable={index.timePoints.some((point) => point.elapsedMs != null)}
            sequence={sequence}
            maxSequence={maxSequence}
            valueText={valueText}
            followLive={followLive}
            followAvailable={runIsActive(runner.live?.status)}
            connected={trace.connected}
            reduced={forceReduced}
            onReduced={setForceReduced}
            onTogglePlay={() => { setFollowLive(false); setPlaying((current) => !current) }}
            onReplay={() => { setFollowLive(false); setSequence(0); setPlaying(true) }}
            onPreviousEvent={() => seek(previousSequence(events, sequence))}
            onNextEvent={() => seek(nextSequence(events, sequence))}
            onPreviousStep={() => { const previous = previousStepSequence(index.steps, sequence); if (previous != null) seek(previous) }}
            onNextStep={() => { const next = nextStepSequence(index.steps, sequence); if (next != null) seek(next) }}
            onScrub={seek}
            onSpeed={setSpeed}
            onMode={(next) => { setFollowLive(false); setPlaying(false); setMode(next) }}
            onFollow={() => { setFollowLive(true); setPlaying(false); setSequence(maxSequence) }}
          />
        </>
      )}

      <Modal
        open={recordingsOpen}
        onOpenChange={setRecordingsOpen}
        title="Recordings"
        description="Open a bundled lesson or import a captured recording."
      >
        <RecordingLibrary
          catalog={catalog}
          catalogError={catalogError}
          loadError={loadError}
          activeId={recording?.source ?? ""}
          recent={lastCaptured ? { id: lastCaptured.source, label: `Latest capture · ${lastCaptured.source}` } : null}
          onOpenRecent={() => {
            if (lastCaptured) openLoaded(lastCaptured)
          }}
          onOpen={(id) => {
            void loadBundled(id).then(openLoaded).catch((error: unknown) => setLoadError(error instanceof Error ? error.message : "Could not open that recording."))
          }}
          onFiles={(files) => {
            void loadFiles(files).then(openLoaded).catch((error: unknown) => setLoadError(error instanceof Error ? error.message : "Could not read those files."))
          }}
        />
      </Modal>

      <Modal
        open={compact && inspectorOpen}
        onOpenChange={setInspectorOpen}
        title="Execution inspector"
        description="Recorded evidence for the selected workflow object."
        placement="right"
      >
        <SelectionInspector
          selection={selection}
          samples={samples}
          state={viewState}
          index={index}
          manifest={manifest}
          sequence={sequence}
          showClose={false}
          onSeek={seek}
          onSelect={choose}
          onClose={() => setInspectorOpen(false)}
        />
      </Modal>

      <ConfirmDialog
        open={confirmResetOpen}
        onOpenChange={setConfirmResetOpen}
        title="Reset the saved draft?"
        description="This replaces the current browser draft with the six-request example."
        confirmLabel="Reset draft"
        onConfirm={resetScenario}
      />
    </div>
  )
}

function highlight(selection: Selection | null, state: ReturnType<typeof attachContent> | null): { ids: Set<string>; steps: Set<string> } {
  const ids = new Set<string>()
  const steps = new Set<string>()
  if (!selection || !state) return { ids, steps }
  if (selection.kind === "request") {
    ids.add(selection.id)
    for (const step of state.steps) if (step.requests.some((request) => request.requestId === selection.id)) steps.add(step.id)
  }
  if (selection.kind === "step") {
    steps.add(selection.id)
    const step = state.steps.find((item) => item.id === selection.id)
    for (const request of step?.requests ?? []) ids.add(request.requestId)
  }
  return { ids, steps }
}

function promptMap(scenario: ScenarioDraft): Record<string, string> {
  return Object.fromEntries(scenario.batches.flatMap((batch) => batch.samples.map((sample) => [sample.id, sample.prompt])))
}

function evidenceText(manifest: RecordingManifest, draftPreview: boolean): string {
  if (draftPreview) return "Draft preview. Values are planned inputs and are not execution evidence."
  if (manifest.execution_kind === "gpu") return "Observed GPU recording. Host intervals are not GPU kernel durations."
  if (manifest.demo_fixture === true || manifest.execution_kind === "fixture") return "Demo fixture. Teaching pace is synthetic and is not GPU evidence."
  return "Recorded execution evidence."
}

function statusLabel(status: string): string {
  if (status === "preparing") return "Preparing"
  if (status === "capturing") return "Capturing"
  if (status === "finalizing") return "Finalizing"
  if (status === "completed") return "Completed"
  if (status === "cancelled") return "Cancelled"
  if (status === "failed") return "Failed"
  if (status === "running") return "Running"
  return status
}

function deriveHostReadiness(
  settings: EngineSettings,
  health: string,
  preflightOk: boolean | undefined,
  preflightMessage: string | undefined,
): ReadinessState {
  if (!health) return { label: "Unchecked", message: "Check the selected host before capture.", tone: "neutral" }
  if (settings.engine === "fixture") return { label: "Ready", message: `${health} The fixture does not require CUDA.`, tone: "ready" }
  if (preflightOk) return { label: "Runtime found", message: preflightMessage ?? health, tone: "ready" }
  return { label: "Needs GPU host", message: preflightMessage ?? health, tone: "warning" }
}

function deriveModelReadiness(
  settings: EngineSettings,
  capabilities: ReturnType<typeof useRunner>["capabilities"],
  quantizationCapabilities?: NonNullable<ReturnType<typeof useRunner>["capabilities"]>["model_options"]["quantization"],
): ReadinessState {
  if (!settings.model.trim()) return { label: "Needs model", message: "Choose model weights before capture.", tone: "warning" }
  const quantizationCapability = quantizationCapabilities?.[settings.quantization]
  if (quantizationCapability?.selectable === false) {
    return { label: "Unavailable", message: quantizationCapability.message, tone: "error" }
  }
  const key = settings.modelSource === "huggingface" ? "huggingface" : settings.modelSource === "host" ? "execution_host" : "operator_local"
  const capability = capabilities?.model_sources[key]
  if (capability?.status === "blocked") return { label: "Unavailable", message: capability.message, tone: "error" }
  if (capability?.status === "implementation_pending") return { label: "Pending", message: capability.message, tone: "warning" }
  return { label: "Configured", message: capability?.message ?? "Model configuration will be resolved during preparation.", tone: "ready" }
}

function deriveVllmReadiness(settings: EngineSettings, capabilities: ReturnType<typeof useRunner>["capabilities"]): ReadinessState {
  if (settings.engine === "fixture") return { label: "Demo fixture", message: "This mode exercises the UI and replay path without GPU evidence.", tone: "warning" }
  const mode = settings.hostKind === "ssh" ? capabilities?.execution_modes.ssh_gpu : capabilities?.execution_modes.local_gpu
  if (!mode) return { label: "Unchecked", message: "Check the selected host to resolve runtime support.", tone: "neutral" }
  if (mode.status === "blocked") return { label: "Unavailable", message: mode.message, tone: "error" }
  return { label: "Configured", message: mode.message, tone: mode.status === "available" ? "ready" : "warning" }
}

function normalizeRemoteDestination(value: string, user: string, suffix: string): string {
  if (value.trim().startsWith("/")) return value.trim()
  const home = user.trim() === "root" ? "/root" : `/home/${user.trim() || "ubuntu"}`
  return `${home}/vllm-explorer/${suffix}`
}

async function responseDetail(response: Response, fallback: string): Promise<string> {
  const body = await response.json().catch(() => null) as { detail?: unknown } | null
  return body && typeof body.detail === "string" ? body.detail : fallback
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"]
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** index
  return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[index]}`
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  return Boolean(target.closest("input, textarea, select, [contenteditable='true'], [role='dialog']"))
}
