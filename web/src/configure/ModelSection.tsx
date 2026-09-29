import { AlertDialog } from "@base-ui/react/alert-dialog"

import type { FeatureCapability } from "../api.ts"
import { createId } from "../scenario/ids.ts"
import type { EngineSettings, LoraAdapterDraft } from "../scenario/types.ts"
import type { FieldIssue } from "../scenario/validate.ts"
import type { ReadinessState } from "./types.ts"
import {
  buttonClass,
  ChoiceCard,
  ConfigureSection,
  dangerButtonClass,
  FieldError,
  FieldHint,
  inputClass,
  StatusPill,
} from "./ui.tsx"

export function ModelSection({
  settings,
  issues,
  readiness,
  quantizationCapabilities,
  huggingFaceToken,
  onSettings,
  onHuggingFaceToken,
  onRemoveAdapter,
}: {
  settings: EngineSettings
  issues: FieldIssue[]
  readiness?: ReadinessState
  quantizationCapabilities?: Record<string, FeatureCapability>
  huggingFaceToken: string
  onSettings: (settings: EngineSettings) => void
  onHuggingFaceToken: (token: string) => void
  onRemoveAdapter: (adapterId: string) => void
}) {
  const issueFor = (id: string) => issues.find((issue) => issue.fieldId === id)?.message ?? ""
  const sourceLabel = settings.modelSource === "huggingface"
    ? "Hugging Face"
    : settings.modelSource === "host" ? "Execution host" : "This computer"
  const summary = `${sourceLabel} · ${settings.model || "No model selected"} · ${settings.dtype}${settings.quantization === "none" ? "" : ` · ${settings.quantization}`}`
  const quantizationCapability = quantizationCapabilities?.[settings.quantization]
  const quantizationIssue = issueFor("model-quantization")

  function updateAdapter(id: string, update: Partial<LoraAdapterDraft>) {
    onSettings({
      ...settings,
      loraAdapters: settings.loraAdapters.map((adapter) => adapter.id === id ? { ...adapter, ...update } : adapter),
    })
  }

  function chooseSource(source: EngineSettings["modelSource"]) {
    onSettings({ ...settings, modelSource: source, model: "", modelRevision: "", tokenizerRevision: "" })
  }

  return (
    <ConfigureSection
      number={3}
      title="Model configuration"
      summary={summary}
      status={readiness?.label ?? (settings.model ? "Configured" : "Needs model")}
      statusTone={readiness?.tone ?? (settings.model ? "ready" : "warning")}
      forceOpen={issues.some((issue) => (
        issue.fieldId.startsWith("setting-model")
        || issue.fieldId === "setting-tokenizer-revision"
        || issue.fieldId === "model-destination"
        || issue.fieldId === "model-quantization"
        || issue.fieldId.startsWith("adapter-")
      ))}
    >
      <div className="space-y-6">
        <fieldset>
          <legend className="text-sm font-semibold text-slate-200">Model weights</legend>
          <p className="mt-1 text-pretty text-sm text-slate-400">Weights are resolved on the execution host before the server starts.</p>
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <ChoiceCard
              name="model-source"
              value="huggingface"
              checked={settings.modelSource === "huggingface"}
              title="Hugging Face"
              description="Download a repository and revision onto the execution host."
              onChange={() => chooseSource("huggingface")}
            />
            <ChoiceCard
              name="model-source"
              value="host"
              checked={settings.modelSource === "host"}
              title="Execution host"
              description="Use weights already present on the selected host."
              onChange={() => chooseSource("host")}
            />
            <ChoiceCard
              name="model-source"
              value="operator"
              checked={settings.modelSource === "operator"}
              title="This computer"
              description="Use locally for local execution or transfer to a remote host."
              onChange={() => chooseSource("operator")}
            />
          </div>
        </fieldset>

        {settings.modelSource === "huggingface" ? (
          <div className="grid gap-4 rounded-lg border border-slate-800 bg-slate-950 p-4 md:grid-cols-2">
            <label className="text-xs font-medium text-slate-400" htmlFor="setting-model">
              Repository
              <input
                id="setting-model"
                className={inputClass}
                value={settings.model}
                placeholder="Qwen/Qwen3-4B-Instruct-2507"
                aria-required="true"
                aria-invalid={Boolean(issueFor("setting-model"))}
                aria-describedby={issueFor("setting-model") ? "setting-model-error" : undefined}
                onChange={(event) => onSettings({ ...settings, model: event.target.value })}
              />
              <FieldError id="setting-model-error" message={issueFor("setting-model")} />
            </label>
            <label className="text-xs font-medium text-slate-400" htmlFor="huggingface-token">
              Hugging Face token <span className="font-normal text-slate-400">optional</span>
              <input
                id="huggingface-token"
                className={inputClass}
                type="password"
                value={huggingFaceToken}
                autoComplete="off"
                placeholder="Required for gated or private models"
                aria-describedby="huggingface-token-hint"
                onChange={(event) => onHuggingFaceToken(event.target.value)}
              />
              <FieldHint id="huggingface-token-hint">The token is sent only for this preparation and is not saved in the browser draft or recording.</FieldHint>
            </label>
          </div>
        ) : settings.modelSource === "host" ? (
          <label className="block text-xs font-medium text-slate-400" htmlFor="setting-model">
            Path on execution host
            <input
              id="setting-model"
              className={inputClass}
              value={settings.model}
              placeholder="/models/qwen3-4b"
              aria-required="true"
              aria-invalid={Boolean(issueFor("setting-model"))}
              aria-describedby={issueFor("setting-model") ? "setting-model-error" : undefined}
              onChange={(event) => onSettings({ ...settings, model: event.target.value })}
            />
            <FieldError id="setting-model-error" message={issueFor("setting-model")} />
          </label>
        ) : (
          <div className="grid gap-4 rounded-lg border border-slate-800 bg-slate-950 p-4 md:grid-cols-2">
            <label className="text-xs font-medium text-slate-400" htmlFor="setting-model">
              Path on this computer
              <input
                id="setting-model"
                className={inputClass}
                value={settings.model}
                placeholder="/Users/me/models/qwen3-4b"
                aria-required="true"
                aria-invalid={Boolean(issueFor("setting-model"))}
                aria-describedby={`${issueFor("setting-model") ? "setting-model-error " : ""}operator-model-path-hint`.trim()}
                onChange={(event) => onSettings({ ...settings, model: event.target.value })}
              />
              <FieldHint id="operator-model-path-hint">Enter a model file or directory that the local controller can read.</FieldHint>
              <FieldError id="setting-model-error" message={issueFor("setting-model")} />
            </label>
            {settings.hostKind === "ssh" ? (
              <label className="text-xs font-medium text-slate-400" htmlFor="model-destination">
                Remote destination
                <input
                  id="model-destination"
                  className={inputClass}
                  value={settings.modelDestination}
                  placeholder="/home/ubuntu/vllm-explorer/model"
                  aria-required="true"
                  aria-invalid={Boolean(issueFor("model-destination"))}
                  aria-describedby={`${issueFor("model-destination") ? "model-destination-error " : ""}model-destination-hint`.trim()}
                  onChange={(event) => onSettings({ ...settings, modelDestination: event.target.value })}
                />
                <FieldHint id="model-destination-hint">Transfer size and progress appear during preparation.</FieldHint>
                <FieldError id="model-destination-error" message={issueFor("model-destination")} />
              </label>
            ) : (
              <p className="self-center text-pretty text-sm text-slate-400">Local execution uses the selected files directly.</p>
            )}
          </div>
        )}

        <div className="grid gap-4 md:grid-cols-2">
          <label className="text-xs font-medium text-slate-400" htmlFor="model-dtype">
            Precision
            <select
              id="model-dtype"
              className={inputClass}
              value={settings.dtype}
              onChange={(event) => onSettings({ ...settings, dtype: event.target.value as EngineSettings["dtype"] })}
            >
              <option value="auto">Auto</option>
              <option value="bfloat16">bfloat16</option>
              <option value="float16">float16</option>
              <option value="float32">float32</option>
            </select>
          </label>
          <label className="text-xs font-medium text-slate-400" htmlFor="model-quantization">
            Weight quantization
            <select
              id="model-quantization"
              className={inputClass}
              value={settings.quantization}
              aria-invalid={Boolean(quantizationIssue) || quantizationCapability?.selectable === false}
              aria-describedby={`${quantizationIssue ? "model-quantization-error " : ""}model-quantization-hint`.trim()}
              onChange={(event) => onSettings({ ...settings, quantization: event.target.value as EngineSettings["quantization"] })}
            >
              {QUANTIZATION_OPTIONS.map((option) => {
                const capability = quantizationCapabilities?.[option.value]
                const unavailable = capability?.selectable === false
                return (
                  <option key={option.value} value={option.value} disabled={unavailable}>
                    {option.label}{unavailable ? " (unavailable on this host)" : ""}
                  </option>
                )
              })}
            </select>
            <FieldHint id="model-quantization-hint">
              {quantizationCapability?.message ?? "Check the execution host to resolve quantization support. The selected model must still pass preparation."}
            </FieldHint>
            <FieldError id="model-quantization-error" message={quantizationIssue} />
          </label>
        </div>

        <details
          className="rounded-lg border border-slate-800 bg-slate-950"
          open={Boolean(issueFor("setting-model-revision") || issueFor("setting-tokenizer-revision")) || undefined}
        >
          <summary className="cursor-pointer rounded-lg px-4 py-3 text-sm font-medium text-slate-300 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500">
            Model revisions
          </summary>
          <div className="grid gap-4 border-t border-slate-800 p-4 md:grid-cols-2">
            <label className="text-xs font-medium text-slate-400" htmlFor="setting-model-revision">
              Model revision {settings.modelSource === "huggingface" ? null : <span className="font-normal text-slate-400">optional</span>}
              <input
                id="setting-model-revision"
                className={inputClass}
                value={settings.modelRevision}
                aria-required={settings.modelSource === "huggingface"}
                aria-invalid={Boolean(issueFor("setting-model-revision"))}
                aria-describedby={issueFor("setting-model-revision") ? "setting-model-revision-error" : undefined}
                onChange={(event) => onSettings({ ...settings, modelRevision: event.target.value })}
              />
              <FieldError id="setting-model-revision-error" message={issueFor("setting-model-revision")} />
            </label>
            <label className="text-xs font-medium text-slate-400" htmlFor="setting-tokenizer-revision">
              Tokenizer revision {settings.modelSource === "huggingface" ? null : <span className="font-normal text-slate-400">optional</span>}
              <input
                id="setting-tokenizer-revision"
                className={inputClass}
                value={settings.tokenizerRevision}
                aria-required={settings.modelSource === "huggingface"}
                aria-invalid={Boolean(issueFor("setting-tokenizer-revision"))}
                aria-describedby={issueFor("setting-tokenizer-revision") ? "setting-tokenizer-revision-error" : undefined}
                onChange={(event) => onSettings({ ...settings, tokenizerRevision: event.target.value })}
              />
              <FieldError id="setting-tokenizer-revision-error" message={issueFor("setting-tokenizer-revision")} />
            </label>
          </div>
        </details>

        <details
          className="rounded-lg border border-slate-800 bg-slate-950"
          open={issues.some((issue) => issue.fieldId.startsWith("adapter-")) || undefined}
        >
          <summary className="cursor-pointer rounded-lg px-4 py-3 text-sm font-medium text-slate-300 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500">
            LoRA adapters <span className="ml-2 text-xs tabular-nums text-slate-400">{settings.loraAdapters.length} attached</span>
          </summary>
          <div className="space-y-4 border-t border-slate-800 p-4">
            <p className="text-pretty text-sm text-slate-400">Attached adapters can become a batch default or a per-request override in the sequence above.</p>
            {settings.loraAdapters.length === 0 ? (
              <div className="rounded-md border border-dashed border-slate-700 p-4 text-center">
                <p className="text-pretty text-sm text-slate-400">No adapters are attached.</p>
                <button className={`${buttonClass} mt-3`} type="button" onClick={() => onSettings({ ...settings, loraAdapters: [...settings.loraAdapters, newAdapter()] })}>
                  Attach an adapter
                </button>
              </div>
            ) : (
              <>
                <ul className="space-y-3" aria-label="Attached LoRA adapters">
                  {settings.loraAdapters.map((adapter, index) => (
                    <li key={adapter.id} className="rounded-md border border-slate-800 bg-slate-900 p-4">
                      <div className="flex items-start justify-between gap-3">
                        <h3 className="text-balance text-sm font-semibold text-slate-200">Adapter {index + 1}</h3>
                        <RemoveAdapterButton name={adapter.name || `Adapter ${index + 1}`} onRemove={() => onRemoveAdapter(adapter.id)} />
                      </div>
                      <div className="mt-3 grid gap-4 md:grid-cols-2">
                        <label className="text-xs font-medium text-slate-400" htmlFor={`adapter-name-${adapter.id}`}>
                          Name
                          <input
                            id={`adapter-name-${adapter.id}`}
                            className={inputClass}
                            value={adapter.name}
                            aria-required="true"
                            aria-invalid={Boolean(issueFor(`adapter-name-${adapter.id}`))}
                            aria-describedby={issueFor(`adapter-name-${adapter.id}`) ? `adapter-name-${adapter.id}-error` : undefined}
                            onChange={(event) => updateAdapter(adapter.id, { name: event.target.value })}
                          />
                          <FieldError id={`adapter-name-${adapter.id}-error`} message={issueFor(`adapter-name-${adapter.id}`)} />
                        </label>
                        <label className="text-xs font-medium text-slate-400" htmlFor={`adapter-source-${adapter.id}`}>
                          Source
                          <select
                            id={`adapter-source-${adapter.id}`}
                            className={inputClass}
                            value={adapter.source}
                            onChange={(event) => updateAdapter(adapter.id, {
                              source: event.target.value as LoraAdapterDraft["source"],
                              location: "",
                              revision: "",
                            })}
                          >
                            <option value="huggingface">Hugging Face</option>
                            <option value="host">Execution host</option>
                            <option value="operator">This computer</option>
                          </select>
                        </label>
                      </div>
                      {adapter.source === "operator" ? (
                        <label className="mt-4 block text-xs font-medium text-slate-400" htmlFor={`adapter-location-${adapter.id}`}>
                          Path on this computer
                          <input
                            id={`adapter-location-${adapter.id}`}
                            className={inputClass}
                            value={adapter.location}
                            placeholder="/Users/me/adapters/support"
                            aria-required="true"
                            aria-invalid={Boolean(issueFor(`adapter-location-${adapter.id}`))}
                            aria-describedby={`${issueFor(`adapter-location-${adapter.id}`) ? `adapter-location-${adapter.id}-error ` : ""}adapter-location-${adapter.id}-hint`.trim()}
                            onChange={(event) => updateAdapter(adapter.id, { location: event.target.value })}
                          />
                          <FieldHint id={`adapter-location-${adapter.id}-hint`}>The local controller inspects this path before transferring it.</FieldHint>
                          <FieldError id={`adapter-location-${adapter.id}-error`} message={issueFor(`adapter-location-${adapter.id}`)} />
                        </label>
                      ) : (
                        <label className="mt-4 block text-xs font-medium text-slate-400" htmlFor={`adapter-location-${adapter.id}`}>
                          {adapter.source === "host" ? "Path on execution host" : "Repository"}
                          <input
                            id={`adapter-location-${adapter.id}`}
                            className={inputClass}
                            value={adapter.location}
                            aria-required="true"
                            aria-invalid={Boolean(issueFor(`adapter-location-${adapter.id}`))}
                            aria-describedby={issueFor(`adapter-location-${adapter.id}`) ? `adapter-location-${adapter.id}-error` : undefined}
                            onChange={(event) => updateAdapter(adapter.id, { location: event.target.value })}
                          />
                          <FieldError id={`adapter-location-${adapter.id}-error`} message={issueFor(`adapter-location-${adapter.id}`)} />
                        </label>
                      )}
                      <label className="mt-4 block text-xs font-medium text-slate-400" htmlFor={`adapter-revision-${adapter.id}`}>
                        Revision <span className="font-normal text-slate-400">optional</span>
                        <input id={`adapter-revision-${adapter.id}`} className={inputClass} value={adapter.revision} onChange={(event) => updateAdapter(adapter.id, { revision: event.target.value })} />
                      </label>
                    </li>
                  ))}
                </ul>
                <button className={buttonClass} type="button" onClick={() => onSettings({ ...settings, loraAdapters: [...settings.loraAdapters, newAdapter()] })}>
                  Attach another adapter
                </button>
              </>
            )}
          </div>
        </details>

        {readiness?.message ? (
          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-slate-800 bg-slate-950 p-4" role={readiness.tone === "error" ? "alert" : "status"}>
            <StatusPill tone={readiness.tone}>{readiness.label}</StatusPill>
            <p className="text-pretty text-sm text-slate-300">{readiness.message}</p>
          </div>
        ) : null}
      </div>
    </ConfigureSection>
  )
}

const QUANTIZATION_OPTIONS: Array<{ value: EngineSettings["quantization"]; label: string }> = [
  { value: "none", label: "None" },
  { value: "awq", label: "AWQ" },
  { value: "gptq", label: "GPTQ" },
  { value: "bitsandbytes", label: "bitsandbytes" },
  { value: "fp8", label: "FP8" },
]

function RemoveAdapterButton({ name, onRemove }: { name: string; onRemove: () => void }) {
  return (
    <AlertDialog.Root>
      <AlertDialog.Trigger className={dangerButtonClass}>Remove</AlertDialog.Trigger>
      <AlertDialog.Portal>
        <AlertDialog.Backdrop className="fixed inset-0 z-40 bg-black/70" />
        <AlertDialog.Viewport className="safe-dialog-viewport fixed inset-0 z-50 flex items-center justify-center overflow-y-auto p-4">
          <AlertDialog.Popup className="flex max-h-[calc(100dvh-2rem)] w-full max-w-lg flex-col gap-4 overflow-auto rounded-lg border border-slate-700 bg-slate-950 p-5 text-slate-100 shadow-xl outline-none">
            <div>
              <AlertDialog.Title className="text-balance text-base font-semibold">Remove {name}?</AlertDialog.Title>
              <AlertDialog.Description className="mt-2 text-pretty text-sm text-slate-400">
                Requests assigned to this adapter will return to the base model. The adapter files are not deleted.
              </AlertDialog.Description>
            </div>
            <div className="flex justify-end gap-2">
              <AlertDialog.Close className={buttonClass}>Keep adapter</AlertDialog.Close>
              <AlertDialog.Close className={dangerButtonClass} onClick={onRemove}>Remove adapter</AlertDialog.Close>
            </div>
          </AlertDialog.Popup>
        </AlertDialog.Viewport>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  )
}

function newAdapter(): LoraAdapterDraft {
  return {
    id: createId("adapter"),
    name: "",
    source: "huggingface",
    location: "",
    revision: "",
  }
}
