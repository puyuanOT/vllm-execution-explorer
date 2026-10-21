import { parseProfileAttachment, readProfileArtifact, type ProfileAttachment } from "../../replay/src/profiling.ts"
import {
  parseJsonl,
  parseRecordingJson,
  type ContentRecord,
  type RecordingManifest,
  type TraceEvent,
} from "../../replay/src/reduce.ts"
import { recordingConfiguration, type RecordingConfiguration } from "./shell/demoCatalog.ts"
import { fetchRecordingText } from "./recordingAssets.ts"

export interface CatalogEntry {
  id: string
  title: string
  summary: string
  evidence: string
  configuration?: RecordingConfiguration
  configurationError?: string
}

export interface LoadedRecording {
  source: string
  /** Display name for recordings opened from files, whose source is only the manifest file name. */
  name?: string
  manifest: RecordingManifest
  events: TraceEvent[]
  content: ContentRecord[]
  profile?: ProfileAttachment | null
  profileError?: string
}

export interface RunSnapshot {
  run_id: string | null
  status: string
  active?: boolean
  intake_open: boolean
  text: Record<string, string>
  message: string
  workload_plan?: unknown
  controller_origin?: unknown
  finalized?: boolean
}

export interface TracePage {
  run_id: string
  status: string
  capture_state: string
  finalized: boolean
  provisional: boolean
  events: TraceEvent[]
  content: ContentRecord[]
  next_event_sequence: number
  next_content_sequence: number
  events_truncated: boolean
  content_truncated: boolean
  workload_plan: unknown
  controller_origin: unknown
  arrivals: Array<Record<string, unknown>>
  message: string
  manifest: RecordingManifest | null
}

export function runIsActive(status: string | null | undefined): boolean {
  return status === "preparing" || status === "capturing" || status === "finalizing" || status === "running"
}

export interface Preflight {
  ok: boolean
  message: string
  platform: string
}

export interface Limits {
  max_requests: number
  max_prompt_characters: number
  max_output_tokens: number
  max_run_seconds: number
}

export type CapabilityStatus = "available" | "blocked" | "gpu_validation_required" | "implementation_pending"

export interface FeatureCapability {
  status: CapabilityStatus
  selectable: boolean
  message: string
}

export interface CapabilityReport {
  contract_version: number
  runtime: {
    platform: string
    cuda_available: boolean
    cuda_compute_capability: [number, number] | null
    vllm_version: string | null
    pinned_vllm: string
  }
  execution_modes: Record<string, FeatureCapability>
  capture_boundaries: Record<string, FeatureCapability>
  workload: Record<string, FeatureCapability>
  model_sources: Record<string, FeatureCapability>
  model_options: {
    lora: FeatureCapability
    quantization: Record<string, FeatureCapability>
  }
  fresh_prefix_cache: FeatureCapability
}

export interface ArtifactTransfer {
  transfer_id: string
  status: "prepared" | "transferring" | "completed" | "failed"
  source_name: string
  artifact_kind: "file" | "directory"
  destination: string
  total_bytes: number
  bytes_transferred: number
  file_count: number
  attempt: number
  message: string
}

export function runnerPath(base: string, path: string): string {
  return `${base.replace(/\/$/, "")}${path}`
}

export async function fetchCapabilities(base: string): Promise<CapabilityReport> {
  return fetchJson<CapabilityReport>(runnerPath(base, "/api/capabilities"))
}

export async function prepareArtifactTransfer(
  base: string,
  sourcePath: string,
  destination: string,
): Promise<ArtifactTransfer> {
  const response = await fetch(runnerPath(base, "/api/artifacts/operator-local/prepare"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ source_path: sourcePath, destination }),
  })
  if (!response.ok) throw new Error(await responseDetail(response, "Could not prepare the artifact transfer."))
  return parseRecordingJson<ArtifactTransfer>(await response.text())
}

export async function startArtifactTransfer(base: string, transferId: string): Promise<ArtifactTransfer> {
  const response = await fetch(runnerPath(base, `/api/artifacts/transfers/${encodeURIComponent(transferId)}/start`), {
    method: "POST",
  })
  if (!response.ok) throw new Error(await responseDetail(response, "Could not start the artifact transfer."))
  return parseRecordingJson<ArtifactTransfer>(await response.text())
}

export async function fetchArtifactTransfer(base: string, transferId: string): Promise<ArtifactTransfer> {
  return fetchJson<ArtifactTransfer>(runnerPath(base, `/api/artifacts/transfers/${encodeURIComponent(transferId)}`))
}

export async function loadCatalog(): Promise<CatalogEntry[]> {
  const response = await fetch("/recordings/index.json")
  if (!response.ok) throw new Error("Bundled recordings are not available.")
  const body = (await response.json()) as { recordings: CatalogEntry[] }
  return Promise.all(body.recordings.map(async (entry) => {
    try {
      const manifest = parseRecordingJson<RecordingManifest>(await fetchRecordingText(`/recordings/${encodeURIComponent(entry.id)}/manifest.json`))
      return { ...entry, configuration: recordingConfiguration(manifest) }
    } catch (error) {
      return { ...entry, configurationError: error instanceof Error ? error.message : "Recording configuration could not be read." }
    }
  }))
}

export async function loadBundled(id: string): Promise<LoadedRecording> {
  return loadFromBase(`/recordings/${id}`, id)
}

export async function loadRunnerRecording(base: string, runId: string, signal?: AbortSignal): Promise<LoadedRecording> {
  const root = runnerPath(base, `/api/recordings/${runId}`)
  const manifest = await fetchJson<RecordingManifest>(`${root}/manifest`, signal)
  const events = parseJsonl<TraceEvent>(await fetchText(`${root}/events`, signal))
  const content = parseJsonl<ContentRecord>(await fetchText(`${root}/content`, signal))
  const profile = await loadProfile(manifest, () => fetchText(`${root}/profile`, signal), signal)
  return { source: runId, manifest, events, content, ...profile }
}

export async function fetchTrace(
  base: string,
  runId: string,
  cursors: { event: number; content: number },
  signal?: AbortSignal,
): Promise<TracePage> {
  const params = new URLSearchParams({
    after_sequence: String(cursors.event),
    after_content_sequence: String(cursors.content),
    limit: "500",
  })
  const response = await fetch(runnerPath(base, `/api/runs/${runId}/trace?${params}`), { signal })
  if (!response.ok) throw new Error("The live trace could not be read.")
  return parseRecordingJson<TracePage>(await response.text())
}

export async function loadFiles(files: FileList): Promise<LoadedRecording> {
  const named = new Map<string, File>()
  for (const file of files) named.set(file.name, file)
  const manifestFile = named.get("manifest.json")
  const eventsFile = named.get("events.jsonl")
  if (!manifestFile || !eventsFile) {
    throw new Error("Choose manifest.json and events.jsonl from the same recording.")
  }
  const manifest = parseRecordingJson<RecordingManifest>(await manifestFile.text())
  const events = parseJsonl<TraceEvent>(await eventsFile.text())
  const contentFile = named.get("content.jsonl")
  const content = contentFile ? parseJsonl<ContentRecord>(await contentFile.text()) : []
  const profile = await loadProfile(manifest, async () => {
    const path = readProfileArtifact(manifest)!.path
    const file = named.get(path) ?? named.get(path.split("/").at(-1)!)
    if (!file) throw new Error("Select the declared normalized profile JSON together with this recording.")
    return file.text()
  })
  return { source: manifestFile.name, name: typeof manifest.run_id === "string" && manifest.run_id ? manifest.run_id : undefined, manifest, events, content, ...profile }
}

async function loadFromBase(base: string, source: string): Promise<LoadedRecording> {
  const manifest = parseRecordingJson<RecordingManifest>(await fetchRecordingText(`${base}/manifest.json`))
  const events = parseJsonl<TraceEvent>(await fetchRecordingText(`${base}/events.jsonl`))
  const contentText = await fetchRecordingText(`${base}/content.jsonl`)
  const profile = await loadProfile(manifest, () => fetchRecordingText(`${base}/${readProfileArtifact(manifest)!.path}`))
  return { source, manifest, events, content: parseJsonl<ContentRecord>(contentText), ...profile }
}

async function loadProfile(manifest: RecordingManifest, readText: () => Promise<string>, signal?: AbortSignal): Promise<{ profile: ProfileAttachment | null; profileError: string }> {
  try {
    const artifact = readProfileArtifact(manifest)
    if (!artifact) return { profile: null, profileError: "" }
    if (artifact.sizeBytes > 512 * 1024 * 1024) throw new Error("Normalized profile exceeds the browser’s 512 MiB attachment limit. Inspect the retained raw trace with a profiler.")
    const text = await readText()
    signal?.throwIfAborted()
    const bytes = new TextEncoder().encode(text)
    if (bytes.length !== artifact.sizeBytes) throw new Error("Measured profile attachment size does not match its manifest.")
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))
    signal?.throwIfAborted()
    const hex = Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("")
    if (hex !== artifact.sha256) throw new Error("Measured profile attachment digest does not match its manifest.")
    return { profile: parseProfileAttachment(text, manifest), profileError: "" }
  } catch (error) {
    if (signal?.aborted) throw error
    return { profile: null, profileError: error instanceof Error ? error.message : "Measured profile could not be loaded." }
  }
}

async function fetchText(url: string, signal?: AbortSignal): Promise<string> {
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`Could not read ${url}`)
  return response.text()
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`Could not read ${url}`)
  return parseRecordingJson<T>(await response.text())
}

async function responseDetail(response: Response, fallback: string): Promise<string> {
  const body = await response.json().catch(() => null) as { detail?: unknown } | null
  return body && typeof body.detail === "string" ? body.detail : fallback
}
