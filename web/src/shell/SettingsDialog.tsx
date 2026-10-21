import { useState } from "react"
import type { FormEvent } from "react"

import type { Limits, Preflight, RunSnapshot } from "../api.ts"
import { runIsActive } from "../api.ts"
import type { EngineSettings } from "../scenario/types.ts"
import type { FieldIssue } from "../scenario/validate.ts"

export function SettingsDialog({
  settings,
  onSettings,
  health,
  preflight,
  limits,
  live,
  issues,
  onRefresh,
  onCloseIntake,
  onCancel,
  onAdd,
}: {
  settings: EngineSettings
  onSettings: (settings: EngineSettings) => void
  health: string
  preflight: Preflight | null
  limits: Limits | null
  live: RunSnapshot | null
  issues: FieldIssue[]
  onRefresh: () => void
  onCloseIntake: () => void
  onCancel: () => void
  onAdd: (body: { request_id: string; prompt: string; max_output_tokens: number; arrival_offset_ms: number }) => void
}) {
  const [ssh, setSsh] = useState({ user: "ubuntu", host: "", port: "22" })
  const [keyFile, setKeyFile] = useState<File | null>(null)
  const [sshMessage, setSshMessage] = useState("")
  const [sshInvalid, setSshInvalid] = useState<"host" | "key" | "form" | null>(null)
  const [connecting, setConnecting] = useState(false)
  const [extra, setExtra] = useState({ request_id: "", prompt: "", max_output_tokens: "8" })
  const active = runIsActive(live?.status)
  const fixture = settings.engine === "fixture"

  async function connectHost(event: FormEvent) {
    event.preventDefault()
    const port = Number(ssh.port)
    if (!ssh.user.trim() || !ssh.host.trim() || !Number.isInteger(port) || port < 1 || port > 65535) {
      setSshInvalid("host")
      setSshMessage("Enter the SSH user, host address, and port.")
      return
    }
    if (!keyFile) {
      setSshInvalid("key")
      setSshMessage("Choose the private key file for that host.")
      return
    }
    setSshInvalid(null)
    setConnecting(true)
    setSshMessage("")
    const body = new FormData()
    body.set("user", ssh.user.trim())
    body.set("host", ssh.host.trim())
    body.set("port", String(port))
    body.set("identity", keyFile)
    try {
      const response = await fetch("/api/targets/connect", { method: "POST", body })
      const payload = (await response.json().catch(() => null)) as { detail?: unknown; message?: string; runner_endpoint?: string } | null
      if (!response.ok) {
        setSshInvalid("form")
        setSshMessage(payload && typeof payload.detail === "string" ? payload.detail : "Could not connect to that GPU host.")
        return
      }
      if (payload?.runner_endpoint) onSettings({ ...settings, runnerUrl: payload.runner_endpoint })
      setSshMessage(payload?.message || "Connected.")
    } catch (error) {
      setSshInvalid("form")
      setSshMessage(error instanceof Error ? error.message : "Could not connect to that GPU host.")
    } finally {
      setConnecting(false)
    }
  }

  function settingIssue(id: string): string {
    return issues.find((issue) => issue.fieldId === id)?.message ?? ""
  }

  return (
    <div className="dialog-body">
      <h2 id="settings-title">Settings</h2>
      <p className="lede">Runner connection and engine settings. The arrival sequence stays in the composer.</p>
      <label htmlFor="runner-url">
        Runner URL
        <input
          id="runner-url"
          value={settings.runnerUrl}
          placeholder="Same origin, or http://127.0.0.1:8765"
          onChange={(event) => onSettings({ ...settings, runnerUrl: event.target.value })}
        />
      </label>
      <button type="button" onClick={onRefresh}>Check runner</button>
      <p role="status">{health || "Runner status is unknown."}</p>
      {preflight ? <p>{preflight.message}</p> : null}
      {limits ? (
        <p className="meta">
          Limits: {limits.max_requests} requests, {limits.max_prompt_characters} prompt characters, {limits.max_output_tokens} output tokens, {limits.max_run_seconds}s.
        </p>
      ) : null}
      <form aria-busy={connecting} onSubmit={(event) => void connectHost(event)}>
        <fieldset>
          <legend>GPU host</legend>
          <p id="ssh-help" className="meta">
            Connect copies this project to the server, starts the loopback runner, and opens a tunnel. The key stays on this computer.
          </p>
          <label htmlFor="ssh-user">
            User
            <input id="ssh-user" value={ssh.user} autoComplete="off" onChange={(event) => setSsh({ ...ssh, user: event.target.value })} />
          </label>
          <label htmlFor="ssh-host">
            Host address
            <input
              id="ssh-host"
              value={ssh.host}
              autoComplete="off"
              aria-invalid={sshInvalid === "host"}
              aria-describedby={sshInvalid === "host" ? "ssh-status" : "ssh-help"}
              onChange={(event) => setSsh({ ...ssh, host: event.target.value })}
            />
          </label>
          <label htmlFor="ssh-port">
            Port
            <input id="ssh-port" value={ssh.port} inputMode="numeric" onChange={(event) => setSsh({ ...ssh, port: event.target.value })} />
          </label>
          <label htmlFor="ssh-key">
            Private key file
            <input
              id="ssh-key"
              type="file"
              accept=".pem,.key"
              aria-invalid={sshInvalid === "key"}
              aria-describedby={sshInvalid === "key" ? "ssh-status" : "ssh-help"}
              onChange={(event) => setKeyFile(event.target.files?.[0] ?? null)}
            />
          </label>
          {keyFile ? <p className="meta">Selected key file: {keyFile.name}</p> : null}
          <button type="submit" disabled={connecting}>{connecting ? "Connecting…" : "Connect"}</button>
          <p id="ssh-status" role={sshInvalid ? "alert" : "status"}>{connecting ? "Connecting to the GPU host." : sshMessage}</p>
        </fieldset>
      </form>
      <label htmlFor="engine-kind">
        Engine
        <select
          id="engine-kind"
          value={settings.engine}
          onChange={(event) => onSettings({
            ...settings,
            engine: event.target.value as EngineSettings["engine"],
            prefixCaching: event.target.value === "fixture" ? false : settings.prefixCaching,
          })}
        >
          <option value="fixture">Demo fixture</option>
          <option value="vllm" disabled={preflight != null && !preflight.ok}>Live vLLM 0.30</option>
        </select>
      </label>
      {fixture ? (
        <p className="meta">
          Demo fixture honors output limits, concurrency, and token budget. Prefix caching is not simulated. Teaching pace is not GPU time.
        </p>
      ) : (
        <>
          <label htmlFor="setting-model-revision">
            Model revision
            <input
              id="setting-model-revision"
              value={settings.modelRevision}
              aria-invalid={Boolean(settingIssue("setting-model-revision"))}
              aria-describedby={settingIssue("setting-model-revision") ? "setting-model-revision-error" : undefined}
              onChange={(event) => onSettings({ ...settings, modelRevision: event.target.value })}
            />
          </label>
          {settingIssue("setting-model-revision") ? <p className="field-error" id="setting-model-revision-error">{settingIssue("setting-model-revision")}</p> : null}
          <label htmlFor="setting-tokenizer-revision">
            Tokenizer revision
            <input
              id="setting-tokenizer-revision"
              value={settings.tokenizerRevision}
              aria-invalid={Boolean(settingIssue("setting-tokenizer-revision"))}
              aria-describedby={settingIssue("setting-tokenizer-revision") ? "setting-tokenizer-revision-error" : undefined}
              onChange={(event) => onSettings({ ...settings, tokenizerRevision: event.target.value })}
            />
          </label>
          {settingIssue("setting-tokenizer-revision") ? <p className="field-error" id="setting-tokenizer-revision-error">{settingIssue("setting-tokenizer-revision")}</p> : null}
        </>
      )}
      <label className="check" htmlFor="prefix-caching">
        <input
          id="prefix-caching"
          type="checkbox"
          checked={fixture ? false : settings.prefixCaching}
          disabled={fixture}
          onChange={(event) => onSettings({ ...settings, prefixCaching: event.target.checked })}
        />
        Prefix caching
      </label>
      <div className="pair">
        <label htmlFor="setting-max-seqs">
          Max sequences
          <input
            id="setting-max-seqs"
            className="mono"
            inputMode="numeric"
            value={settings.maxNumSeqs}
            aria-invalid={Boolean(settingIssue("setting-max-seqs"))}
            aria-describedby={settingIssue("setting-max-seqs") ? "setting-max-seqs-error" : undefined}
            onChange={(event) => onSettings({ ...settings, maxNumSeqs: event.target.value })}
          />
        </label>
        <label htmlFor="setting-token-budget">
          Token budget
          <input
            id="setting-token-budget"
            className="mono"
            inputMode="numeric"
            value={settings.tokenBudget}
            aria-invalid={Boolean(settingIssue("setting-token-budget"))}
            aria-describedby={settingIssue("setting-token-budget") ? "setting-token-budget-error" : undefined}
            onChange={(event) => onSettings({ ...settings, tokenBudget: event.target.value })}
          />
        </label>
      </div>
      {settingIssue("setting-max-seqs") ? <p className="field-error" id="setting-max-seqs-error">{settingIssue("setting-max-seqs")}</p> : null}
      {settingIssue("setting-token-budget") ? <p className="field-error" id="setting-token-budget-error">{settingIssue("setting-token-budget")}</p> : null}
      <details>
        <summary>Advanced intake</summary>
        <p className="meta">Close intake rejects new samples. Samples already accepted still run, including future arrivals.</p>
        {active ? (
          <div className="row-actions">
            <button type="button" onClick={onCloseIntake} disabled={!live?.intake_open}>Close intake</button>
            <button type="button" onClick={onCancel}>Cancel run</button>
          </div>
        ) : <p className="meta">No run is active.</p>}
        {live?.intake_open && active ? (
          <form
            onSubmit={(event) => {
              event.preventDefault()
              onAdd({
                request_id: extra.request_id.trim(),
                prompt: extra.prompt,
                max_output_tokens: Number(extra.max_output_tokens),
                arrival_offset_ms: 0,
              })
            }}
          >
            <label htmlFor="extra-id">
              Add now
              <input id="extra-id" value={extra.request_id} onChange={(event) => setExtra({ ...extra, request_id: event.target.value })} />
            </label>
            <label htmlFor="extra-prompt">
              Prompt
              <textarea id="extra-prompt" rows={2} value={extra.prompt} onChange={(event) => setExtra({ ...extra, prompt: event.target.value })} />
            </label>
            <label htmlFor="extra-output">
              Maximum output tokens
              <input id="extra-output" className="mono" value={extra.max_output_tokens} onChange={(event) => setExtra({ ...extra, max_output_tokens: event.target.value })} />
            </label>
            <button type="submit">Admit now</button>
          </form>
        ) : null}
      </details>
    </div>
  )
}
