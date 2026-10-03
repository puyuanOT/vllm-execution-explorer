import type { EngineSettings } from "../scenario/types.ts"
import type { FieldIssue } from "../scenario/validate.ts"
import type { ReadinessState } from "./types.ts"
import {
  CheckboxField,
  ConfigureSection,
  Details,
  FieldError,
  FieldHint,
  inputClass,
  StatusPill,
} from "./ui.tsx"

export function VllmSection({
  settings,
  issues,
  readiness,
  effectiveSettings,
  onSettings,
}: {
  settings: EngineSettings
  issues: FieldIssue[]
  readiness?: ReadinessState
  effectiveSettings?: Record<string, string | number | boolean | null>
  onSettings: (settings: EngineSettings) => void
}) {
  const issueFor = (id: string) => issues.find((issue) => issue.fieldId === id)?.message ?? ""
  const summary = `${settings.maxModelLen} context · ${settings.maxNumSeqs} concurrent · ${settings.tokenBudget} tokens/iteration`

  return (
    <ConfigureSection
      number={4}
      title="vLLM configuration"
      summary={summary}
      status={readiness?.label ?? "Configured"}
      statusTone={readiness?.tone ?? "ready"}
      forceOpen={issues.some((issue) => [
        "setting-max-model-len",
        "setting-max-seqs",
        "setting-token-budget",
        "setting-gpu-memory",
        "setting-temperature",
        "setting-top-p",
        "setting-seed",
        "setting-stop-strings",
      ].includes(issue.fieldId))}
    >
      <div className="space-y-6">
        <div className="grid items-start gap-4 md:grid-cols-3">
          <label className="block text-xs font-medium text-slate-400" htmlFor="setting-max-model-len">
            Maximum context length
            <input
              id="setting-max-model-len"
              className={`${inputClass} tabular-nums`}
              inputMode="numeric"
              value={settings.maxModelLen}
              aria-required="true"
              aria-invalid={Boolean(issueFor("setting-max-model-len"))}
              aria-describedby={`${issueFor("setting-max-model-len") ? "setting-max-model-len-error " : ""}setting-max-model-len-hint`.trim()}
              onChange={(event) => onSettings({ ...settings, maxModelLen: event.target.value })}
            />
            <FieldHint id="setting-max-model-len-hint">Maximum prompt and generated tokens per request.</FieldHint>
            <FieldError id="setting-max-model-len-error" message={issueFor("setting-max-model-len")} />
          </label>
          <label className="block text-xs font-medium text-slate-400" htmlFor="setting-max-seqs">
            Maximum concurrent sequences
            <input
              id="setting-max-seqs"
              className={`${inputClass} tabular-nums`}
              inputMode="numeric"
              value={settings.maxNumSeqs}
              aria-required="true"
              aria-invalid={Boolean(issueFor("setting-max-seqs"))}
              aria-describedby={`${issueFor("setting-max-seqs") ? "setting-max-seqs-error " : ""}setting-max-seqs-hint`.trim()}
              onChange={(event) => onSettings({ ...settings, maxNumSeqs: event.target.value })}
            />
            <FieldHint id="setting-max-seqs-hint">Upper bound for sequences scheduled together.</FieldHint>
            <FieldError id="setting-max-seqs-error" message={issueFor("setting-max-seqs")} />
          </label>
          <label className="block text-xs font-medium text-slate-400" htmlFor="setting-token-budget">
            Tokens per scheduler iteration
            <input
              id="setting-token-budget"
              className={`${inputClass} tabular-nums`}
              inputMode="numeric"
              value={settings.tokenBudget}
              aria-required="true"
              aria-invalid={Boolean(issueFor("setting-token-budget"))}
              aria-describedby={`${issueFor("setting-token-budget") ? "setting-token-budget-error " : ""}setting-token-budget-hint`.trim()}
              onChange={(event) => onSettings({ ...settings, tokenBudget: event.target.value })}
            />
            <FieldHint id="setting-token-budget-hint">Maximum scheduled tokens across requests in one iteration.</FieldHint>
            <FieldError id="setting-token-budget-error" message={issueFor("setting-token-budget")} />
          </label>
        </div>

        <div className="grid items-start gap-5 rounded-lg border border-slate-800 bg-slate-950 p-4 md:grid-cols-2">
          <CheckboxField
            id="chunked-prefill"
            checked={settings.chunkedPrefill}
            label="Chunked prefill"
            description="Allow large prefills to share scheduler iterations with other work."
            onChange={(chunkedPrefill) => onSettings({ ...settings, chunkedPrefill })}
          />
          <CheckboxField
            id="prefix-caching"
            checked={settings.engine === "fixture" ? false : settings.prefixCaching}
            disabled={settings.engine === "fixture"}
            label="Prefix caching"
            description="Reuse matching prompt-prefix blocks within this run."
            onChange={(prefixCaching) => onSettings({ ...settings, prefixCaching })}
          />
        </div>

        <div className="rounded-lg border border-blue-900 bg-blue-950/40 p-4">
          <h3 className="text-balance text-sm font-semibold text-blue-200">Fresh-cache run contract</h3>
          <p className="mt-1 text-pretty text-sm leading-6 text-blue-300/80">
            Preparation must finish model loading, warmup, and a verified cache reset before workload t = 0. The controller must refuse capture when it cannot verify that boundary.
          </p>
        </div>

        <Details
          className="rounded-lg border border-slate-800 bg-slate-950"
          forceOpen={issues.some((issue) => [
            "setting-gpu-memory",
            "setting-temperature",
            "setting-top-p",
            "setting-seed",
            "setting-stop-strings",
          ].includes(issue.fieldId))}
        >
          <summary className="cursor-pointer rounded-lg px-4 py-3 text-sm font-medium text-slate-300 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500">
            Memory and sampling
          </summary>
          <div className="space-y-5 border-t border-slate-800 p-4">
            <label className="block text-xs font-medium text-slate-400" htmlFor="setting-gpu-memory">
              GPU memory fraction
              <input
                id="setting-gpu-memory"
                className={`${inputClass} tabular-nums`}
                inputMode="decimal"
                value={settings.gpuMemoryUtilization}
                aria-required="true"
                aria-invalid={Boolean(issueFor("setting-gpu-memory"))}
                aria-describedby={`${issueFor("setting-gpu-memory") ? "setting-gpu-memory-error " : ""}setting-gpu-memory-hint`.trim()}
                onChange={(event) => onSettings({ ...settings, gpuMemoryUtilization: event.target.value })}
              />
              <FieldHint id="setting-gpu-memory-hint">Fraction of GPU memory available to the engine. KV-cache capacity appears after initialization.</FieldHint>
              <FieldError id="setting-gpu-memory-error" message={issueFor("setting-gpu-memory")} />
            </label>

            <div className="grid items-start gap-4 md:grid-cols-3">
              <label className="block text-xs font-medium text-slate-400" htmlFor="setting-temperature">
                Temperature
                <input
                  id="setting-temperature"
                  className={`${inputClass} tabular-nums`}
                  inputMode="decimal"
                  value={settings.temperature}
                  aria-required="true"
                  aria-invalid={Boolean(issueFor("setting-temperature"))}
                  aria-describedby={`${issueFor("setting-temperature") ? "setting-temperature-error " : ""}sampling-hint`.trim()}
                  onChange={(event) => onSettings({ ...settings, temperature: event.target.value })}
                />
                <FieldError id="setting-temperature-error" message={issueFor("setting-temperature")} />
              </label>
              <label className="block text-xs font-medium text-slate-400" htmlFor="setting-top-p">
                Top-p
                <input
                  id="setting-top-p"
                  className={`${inputClass} tabular-nums`}
                  inputMode="decimal"
                  value={settings.topP}
                  aria-required="true"
                  aria-invalid={Boolean(issueFor("setting-top-p"))}
                  aria-describedby={`${issueFor("setting-top-p") ? "setting-top-p-error " : ""}sampling-hint`.trim()}
                  onChange={(event) => onSettings({ ...settings, topP: event.target.value })}
                />
                <FieldError id="setting-top-p-error" message={issueFor("setting-top-p")} />
              </label>
              <label className="block text-xs font-medium text-slate-400" htmlFor="setting-seed">
                Seed <span className="font-normal text-slate-400">optional</span>
                <input
                  id="setting-seed"
                  className={`${inputClass} tabular-nums`}
                  inputMode="numeric"
                  value={settings.seed}
                  aria-invalid={Boolean(issueFor("setting-seed"))}
                  aria-describedby={`${issueFor("setting-seed") ? "setting-seed-error " : ""}sampling-hint`.trim()}
                  onChange={(event) => onSettings({ ...settings, seed: event.target.value })}
                />
                <FieldError id="setting-seed-error" message={issueFor("setting-seed")} />
              </label>
            </div>
            <FieldHint id="sampling-hint">Temperature 0 uses greedy sampling. A seed helps investigation but does not promise identical scheduling across runs.</FieldHint>

            <label className="block text-xs font-medium text-slate-400" htmlFor="setting-stop-strings">
              Stop strings <span className="font-normal text-slate-400">optional, one per line</span>
              <textarea
                id="setting-stop-strings"
                className={`${inputClass} min-h-24 resize-y`}
                value={settings.stopStrings.join("\n")}
                aria-invalid={Boolean(issueFor("setting-stop-strings"))}
                aria-describedby={issueFor("setting-stop-strings") ? "setting-stop-strings-error" : undefined}
                onChange={(event) => onSettings({ ...settings, stopStrings: event.target.value.split("\n") })}
              />
              <FieldError id="setting-stop-strings-error" message={issueFor("setting-stop-strings")} />
            </label>
          </div>
        </Details>

        {effectiveSettings && Object.keys(effectiveSettings).length > 0 ? (
          <details className="rounded-lg border border-slate-800 bg-slate-950">
            <summary className="cursor-pointer rounded-lg px-4 py-3 text-sm font-medium text-slate-300 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500">
              Planned launch settings
            </summary>
            <dl className="grid gap-x-6 gap-y-3 border-t border-slate-800 p-4 md:grid-cols-2">
              {Object.entries(effectiveSettings).map(([label, value]) => (
                <div key={label} className="flex items-baseline justify-between gap-4 border-b border-slate-800 pb-2">
                  <dt className="text-xs text-slate-400">{label}</dt>
                  <dd className="truncate text-right text-xs tabular-nums text-slate-300">{value == null ? "Unknown" : String(value)}</dd>
                </div>
              ))}
            </dl>
          </details>
        ) : null}

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
