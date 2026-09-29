import type { EngineSettings } from "../scenario/types.ts"
import type { FieldIssue } from "../scenario/validate.ts"
import type { ConfigureRuntimeInputs, ReadinessState, SetupStage } from "./types.ts"
import {
  buttonClass,
  CheckboxField,
  ChoiceCard,
  ConfigureSection,
  FieldError,
  FieldHint,
  inputClass,
  StatusPill,
} from "./ui.tsx"

export function ServerSection({
  settings,
  issues,
  readiness,
  stages,
  checking,
  sshIdentity,
  onSshIdentity,
  onSettings,
  onCheck,
}: {
  settings: EngineSettings
  issues: FieldIssue[]
  readiness?: ReadinessState
  stages?: SetupStage[]
  checking: boolean
  sshIdentity: File | null
  onSshIdentity: (file: File | null) => void
  onSettings: (settings: EngineSettings) => void
  onCheck?: (runtime: Pick<ConfigureRuntimeInputs, "sshIdentity">) => void
}) {
  const issueFor = (id: string) => issues.find((issue) => issue.fieldId === id)?.message ?? ""
  const target = settings.hostKind === "local"
    ? "This machine"
    : settings.sshHost.trim() || "Remote host"
  const summary = settings.hostKind === "local"
    ? "Run on a CUDA-ready Linux machine available to this controller."
    : `${settings.sshUser || "user"}@${target}:${settings.sshPort || "22"}`

  return (
    <ConfigureSection
      number={2}
      title="Visualization server"
      summary={summary}
      status={readiness?.label ?? "Unchecked"}
      statusTone={readiness?.tone ?? "neutral"}
      forceOpen={issues.some((issue) => issue.fieldId.startsWith("ssh-"))}
    >
      <div className="space-y-6" aria-busy={checking}>
        <fieldset>
          <legend className="text-sm font-semibold text-slate-200">Execution host</legend>
          <p className="mt-1 text-pretty text-sm text-slate-400">Choose the CUDA-ready Linux host where vLLM and the timed workload driver will run.</p>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <ChoiceCard
              name="host-kind"
              value="local"
              checked={settings.hostKind === "local"}
              title="Local machine"
              description="Use a local CUDA host reachable by the controller. Capability is checked before setup."
              onChange={() => onSettings({ ...settings, hostKind: "local" })}
            />
            <ChoiceCard
              name="host-kind"
              value="ssh"
              checked={settings.hostKind === "ssh"}
              title="Remote GPU host"
              description="Connect to an existing CUDA-ready Linux machine through SSH."
              onChange={() => onSettings({ ...settings, hostKind: "ssh" })}
            />
          </div>
        </fieldset>

        {settings.hostKind === "ssh" ? (
          <div className="grid gap-4 rounded-lg border border-slate-800 bg-slate-950 p-4 md:grid-cols-2">
            <label className="text-xs font-medium text-slate-400" htmlFor="ssh-user">
              SSH user
              <input
                id="ssh-user"
                className={inputClass}
                value={settings.sshUser}
                autoComplete="username"
                aria-required="true"
                onChange={(event) => onSettings({ ...settings, sshUser: event.target.value })}
              />
            </label>
            <label className="text-xs font-medium text-slate-400" htmlFor="ssh-host">
              Host address
              <input
                id="ssh-host"
                className={inputClass}
                value={settings.sshHost}
                placeholder="203.0.113.10 or gpu.example.com"
                autoComplete="off"
                aria-required="true"
                aria-invalid={Boolean(issueFor("ssh-host"))}
                aria-describedby={issueFor("ssh-host") ? "ssh-host-error" : undefined}
                onChange={(event) => onSettings({ ...settings, sshHost: event.target.value })}
              />
              <FieldError id="ssh-host-error" message={issueFor("ssh-host")} />
            </label>
            <label className="text-xs font-medium text-slate-400" htmlFor="ssh-port">
              SSH port
              <input
                id="ssh-port"
                className={`${inputClass} tabular-nums`}
                value={settings.sshPort}
                inputMode="numeric"
                aria-required="true"
                aria-invalid={Boolean(issueFor("ssh-port"))}
                aria-describedby={issueFor("ssh-port") ? "ssh-port-error" : undefined}
                onChange={(event) => onSettings({ ...settings, sshPort: event.target.value })}
              />
              <FieldError id="ssh-port-error" message={issueFor("ssh-port")} />
            </label>
            <label className="text-xs font-medium text-slate-400" htmlFor="ssh-key">
              Private key
              <input
                id="ssh-key"
                className={`${inputClass} file:mr-3 file:rounded file:border-0 file:bg-slate-800 file:px-2 file:py-1 file:text-xs file:text-slate-200`}
                type="file"
                aria-required="true"
                aria-invalid={Boolean(issueFor("ssh-key"))}
                aria-describedby={`${issueFor("ssh-key") ? "ssh-key-error " : ""}ssh-key-hint`.trim()}
                onChange={(event) => onSshIdentity(event.target.files?.[0] ?? null)}
              />
              <FieldHint id="ssh-key-hint">{sshIdentity ? `${sshIdentity.name} selected. The key stays on this computer.` : "Select the key used to reach this host. It is never stored in the draft or recording."}</FieldHint>
              <FieldError id="ssh-key-error" message={issueFor("ssh-key")} />
            </label>
          </div>
        ) : (
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
            <p className="text-pretty text-sm text-slate-300">The capability check verifies Linux, NVIDIA drivers, CUDA visibility, storage, and required tools.</p>
            <p className="mt-2 text-pretty text-xs leading-5 text-slate-400">Cloud provisioning and driver installation are outside this workflow.</p>
          </div>
        )}

        <CheckboxField
          id="automatic-setup"
          checked={settings.automaticSetup}
          label="Set up the execution environment automatically"
          description="Create an isolated runtime, install pinned dependencies and instrumentation, obtain artifacts, and start the loopback endpoint."
          onChange={(automaticSetup) => onSettings({ ...settings, automaticSetup })}
        />

        {readiness?.message ? (
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-4" role={readiness.tone === "error" ? "alert" : "status"}>
            <div className="flex flex-wrap items-center gap-2">
              <StatusPill tone={readiness.tone}>{readiness.label}</StatusPill>
              <p className="text-pretty text-sm text-slate-300">{readiness.message}</p>
            </div>
          </div>
        ) : null}

        {stages && stages.length > 0 ? (
          <ol className="grid gap-2" aria-label="Server setup progress">
            {stages.map((stage, index) => (
              <li key={stage.id} className="flex gap-3 rounded-md border border-slate-800 bg-slate-950 px-3 py-2">
                <span className="flex size-6 shrink-0 items-center justify-center rounded-full border border-slate-700 text-xs tabular-nums text-slate-400" aria-hidden="true">{index + 1}</span>
                <span className="min-w-0 flex-1">
                  <span className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium text-slate-300">{stage.label}</span>
                    <StatusPill tone={stageTone(stage.status)}>{stageLabel(stage.status)}</StatusPill>
                  </span>
                  {stage.detail ? <span className="mt-1 block text-pretty text-xs text-slate-400">{stage.detail}</span> : null}
                </span>
              </li>
            ))}
          </ol>
        ) : null}

        <div className="flex flex-wrap items-center gap-3">
          <button
            className={buttonClass}
            type="button"
            disabled={checking}
            onClick={() => onCheck?.({ sshIdentity })}
          >
            {checking ? "Checking host…" : readiness?.tone === "error" ? "Retry host check" : "Check host"}
          </button>
          <span className="text-pretty text-xs text-slate-400">No execution begins until you start capture.</span>
        </div>

        <details className="rounded-lg border border-slate-800 bg-slate-950">
          <summary className="cursor-pointer rounded-lg px-4 py-3 text-sm font-medium text-slate-300 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500">
            Connection details
          </summary>
          <div className="grid gap-4 border-t border-slate-800 p-4 md:grid-cols-2">
            <label className="text-xs font-medium text-slate-400" htmlFor="runner-url">
              Controller endpoint
              <input
                id="runner-url"
                className={inputClass}
                value={settings.runnerUrl}
                placeholder="Same origin"
                aria-describedby="runner-url-hint"
                onChange={(event) => onSettings({ ...settings, runnerUrl: event.target.value })}
              />
              <FieldHint id="runner-url-hint">Leave empty when the UI and controller share an origin.</FieldHint>
            </label>
            <label className="text-xs font-medium text-slate-400" htmlFor="engine-kind">
              Execution source
              <select
                id="engine-kind"
                className={inputClass}
                value={settings.engine}
                aria-describedby="engine-kind-hint"
                onChange={(event) => onSettings({
                  ...settings,
                  engine: event.target.value === "fixture" ? "fixture" : "vllm",
                  prefixCaching: event.target.value === "fixture" ? false : settings.prefixCaching,
                })}
              >
                <option value="vllm">Instrumented vLLM server</option>
                <option value="fixture">Demo fixture</option>
              </select>
              <FieldHint id="engine-kind-hint">The fixture is labeled as synthetic evidence and cannot certify GPU behavior.</FieldHint>
            </label>
          </div>
        </details>
      </div>
    </ConfigureSection>
  )
}

function stageTone(status: SetupStage["status"]): ReadinessState["tone"] {
  if (status === "complete") return "ready"
  if (status === "active") return "warning"
  if (status === "error") return "error"
  return "neutral"
}

function stageLabel(status: SetupStage["status"]) {
  if (status === "complete") return "Complete"
  if (status === "active") return "In progress"
  if (status === "error") return "Failed"
  return "Pending"
}
