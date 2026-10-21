import assert from "node:assert/strict"
import test from "node:test"
import { defaultSettings } from "./example.ts"
import { learningExample, matchingLearningExample } from "./learningExamples.ts"
import { toRunBody } from "./serialize.ts"
import { validateScenario, validateSettings } from "./validate.ts"
import { demoAdapterScenario } from "../shell/demoCatalog.ts"

test("the operator workload is one consistent editable capture across model and precision choices", () => {
  let prompts: string[] | null = null
  for (const model of ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3.8-27B"]) {
    for (const quantization of ["none", "fp8"] as const) {
      const original = { ...defaultSettings(), model, quantization, sshHost: "new-host.example", runnerUrl: "http://127.0.0.1:8771" }
      const { scenario, settings } = learningExample("operator-demo", original)
      assert.deepEqual(validateSettings(settings), [])
      assert.deepEqual(validateScenario(scenario, null, settings), [])
      assert.equal(matchingLearningExample(scenario, settings), "operator-demo")
      assert.equal(settings.model, model)
      assert.equal(settings.quantization, quantization)
      assert.equal(settings.sshHost, original.sshHost)
      assert.equal(settings.runnerUrl, original.runnerUrl)
      const body = toRunBody(scenario, settings)
      assert.deepEqual(body.requests.map(r => r.max_output_tokens), [64, 96, 128, 160, 192, 256])
      assert.deepEqual(body.workload_plan.batches.map(b => b.samples.length), [1, 2, 3])
      const texts = body.requests.map(r => r.prompt)
      if (prompts) assert.deepEqual(texts, prompts)
      else prompts = texts
      assert.ok(texts.every(prompt => prompt.includes("Meridian")))
      const context = texts[0].slice(0, texts[0].lastIndexOf("\n\n"))
      assert.ok(texts[1].startsWith(context))
      assert.ok(texts[3].startsWith(context))
      assert.equal(body.profiling_configuration.max_steps, 1024)
      assert.equal(body.vllm_configuration.max_num_scheduled_tokens, model.includes("3.8") ? 1024 : 64)
      assert.equal(original.profilingMaxSteps, "8")
    }
  }
})

test("adapter assignment preserves meaningful prompts and their output limits", () => {
  const { scenario, settings } = learningExample("operator-demo", { ...defaultSettings(), loraAdapters: [
    { id: "adapter-a", name: "Adapter A", source: "huggingface" as const, location: "example/adapter", revision: "pinned" },
  ] })
  for (const mode of ["none", "all", "mixed"] as const) {
    const routed = demoAdapterScenario(scenario, settings, mode)
    const before = scenario.batches.flatMap(b => b.samples)
    const after = routed.batches.flatMap(b => b.samples)
    assert.deepEqual(after.map(s => s.prompt), before.map(s => s.prompt))
    assert.deepEqual(after.map(s => s.maxOutputTokens), before.map(s => s.maxOutputTokens))
    assert.equal(after.filter(s => s.adapterId === "adapter-a").length, mode === "none" ? 0 : mode === "all" ? 6 : 3)
  }
  const mixed = learningExample("operator-demo", { ...settings, demoLoraMode: "mixed" })
  assert.deepEqual(validateScenario(mixed.scenario, null, mixed.settings), [])
  assert.deepEqual(toRunBody(mixed.scenario, mixed.settings).workload_plan.batches.map(batch => batch.arrival_offset_ms), [0, 40000, 40025])
  assert.deepEqual(mixed.scenario.batches.flatMap(b => b.samples).map(s => s.adapterId), ["__base__", "__base__", "adapter-a", "adapter-a", "__base__", "adapter-a"])
})
