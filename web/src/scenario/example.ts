import type { ArrivalBatchDraft, EngineSettings, SampleDraft, ScenarioDraft } from "./types.ts"

function sample(id: string, label: string, prompt: string, maxOutputTokens = 8): SampleDraft {
  return {
    id,
    label,
    prompt,
    inputMode: "prompt",
    syntheticInputTokens: 128,
    resolvedInputTokens: null,
    maxOutputTokens,
    adapterId: null,
    priority: 0,
  }
}

function batch(
  id: string,
  label: string,
  arrivalOffsetMs: number,
  createdOrder: number,
  samples: SampleDraft[],
): ArrivalBatchDraft {
  return { id, label, arrivalOffsetMs, createdOrder, defaultAdapterId: null, samples }
}

export function staggeredExample(): ScenarioDraft {
  return {
    version: 1,
    id: "scenario-staggered-example",
    name: "Staggered arrivals",
    nextOrder: 3,
    batches: [
      batch("batch-a", "A", 0, 0, [
        sample("sample-1", "input_1", "Name a primary color."),
        sample("sample-2", "input_2", "Name a European capital."),
      ]),
      batch("batch-b", "B", 100, 1, [
        sample("sample-3", "input_3", "Count to three."),
        sample("sample-4", "input_4", "Name a common fruit."),
        sample("sample-5", "input_5", "Name a weekday."),
      ]),
      batch("batch-c", "C", 120, 2, [
        sample("sample-6", "input_6", "Name a season."),
      ]),
    ],
  }
}

export function emptyScenario(): ScenarioDraft {
  return {
    version: 1,
    id: "scenario-empty",
    name: "Empty sequence",
    nextOrder: 0,
    batches: [],
  }
}

export function defaultSettings(): EngineSettings {
  return {
    runnerUrl: "",
    engine: "vllm",
    hostKind: "local",
    sshUser: "ubuntu",
    sshHost: "",
    sshPort: "22",
    automaticSetup: true,
    modelSource: "huggingface",
    modelDestination: "/home/ubuntu/vllm-explorer/model",
    prefixCaching: false,
    chunkedPrefill: true,
    maxModelLen: "4096",
    maxNumSeqs: "4",
    tokenBudget: "2048",
    gpuMemoryUtilization: "0.9",
    schedulingPolicy: "fcfs",
    longPrefillTokenThreshold: "0",
    kvCacheMemoryBytes: "",
    kvCacheDtype: "auto",
    profilingMode: "off",
    profilingMaxSteps: "8",
    profilingStartStep: "1",
    profilingTriggerRequestId: "",
    tensorCapture: false,
    tensorPreset: "small",
    tensorRequestId: "",
    tensorTokenPositions: "",
    tensorLayerIndices: "",
    tensorHeadIndices: "0",
    tensorDimensionStart: "0",
    tensorDimensionCount: "16",
    tensorMaxBytes: "1048576",
    model: "Qwen/Qwen3-4B-Instruct-2507",
    modelRevision: "",
    tokenizerRevision: "",
    tokenizer: "",
    dtype: "auto",
    quantization: "none",
    loraAdapters: [],
    demoLoraMode: null,
    temperature: "0",
    topP: "1",
    seed: "",
    stopStrings: [],
  }
}
