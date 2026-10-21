export type SampleInputMode = "prompt" | "synthetic"

export interface SampleDraft {
  id: string
  label: string
  prompt: string
  inputMode: SampleInputMode
  syntheticInputTokens: number
  resolvedInputTokens: number | null
  maxOutputTokens: number
  adapterId: string | null
  priority: number
}

export interface ArrivalBatchDraft {
  id: string
  label: string
  arrivalOffsetMs: number
  createdOrder: number
  defaultAdapterId: string | null
  samples: SampleDraft[]
}

export interface ScenarioDraft {
  version: 1
  id: string
  name: string
  nextOrder: number
  batches: ArrivalBatchDraft[]
}

export interface EngineSettings {
  runnerUrl: string
  engine: "fixture" | "vllm"
  hostKind: "local" | "ssh"
  sshUser: string
  sshHost: string
  sshPort: string
  automaticSetup: boolean
  modelSource: "huggingface" | "host" | "operator"
  modelDestination: string
  prefixCaching: boolean
  chunkedPrefill: boolean
  maxModelLen: string
  maxNumSeqs: string
  tokenBudget: string
  gpuMemoryUtilization: string
  schedulingPolicy: "fcfs" | "priority"
  longPrefillTokenThreshold: string
  kvCacheMemoryBytes: string
  kvCacheDtype: "auto" | "float16" | "bfloat16" | "fp8" | "fp8_e4m3" | "fp8_e5m2"
  profilingMode: "off" | "eager_diagnostic"
  profilingMaxSteps: string
  profilingStartStep: string
  profilingTriggerRequestId: string
  tensorCapture: boolean
  tensorPreset: "small" | "custom"
  tensorRequestId: string
  tensorTokenPositions: string
  tensorLayerIndices: string
  tensorHeadIndices: string
  tensorDimensionStart: string
  tensorDimensionCount: string
  tensorMaxBytes: string
  model: string
  modelRevision: string
  tokenizerRevision: string
  tokenizer: string
  dtype: "auto" | "float16" | "bfloat16" | "float32"
  quantization: "none" | "awq" | "gptq" | "bitsandbytes" | "fp8"
  loraAdapters: LoraAdapterDraft[]
  /** A demo capture intent; null leaves request routing under manual control. */
  demoLoraMode: "none" | "all" | "mixed" | null
  temperature: string
  topP: string
  seed: string
  stopStrings: string[]
}

export interface LoraAdapterDraft {
  id: string
  name: string
  source: "huggingface" | "host" | "operator"
  location: string
  revision: string
}

export type TimeUnit = "ms" | "s"

export interface WorkloadPlan {
  version: 1
  batches: Array<{
    id: string
    label: string
    arrival_offset_ms: number
    samples: Array<{
      id: string
      label: string
      input_mode: SampleInputMode
      synthetic_input_tokens: number | null
      target_input_tokens: number | null
      resolved_input_tokens: number | null
      max_output_tokens: number
      adapter_id: string | null
      priority: number
    }>
    default_adapter_id: string | null
    adapter_id: string
  }>
}

export interface ArtifactSourcePayload {
  kind: "huggingface" | "execution_host" | "operator_local"
  repository?: string
  path?: string
  revision?: string | null
  transfer?: { destination: string | null }
}

export interface RunRequestBody {
  engine: "fixture" | "vllm"
  requests: Array<{
    request_id: string
    prompt: string
    input_mode: SampleInputMode
    synthetic_input_tokens: number | null
    target_input_tokens: number | null
    resolved_input_tokens: number | null
    arrival_offset_ms: number
    max_output_tokens: number
    adapter_id: string | null
    priority: number
  }>
  workload_plan: WorkloadPlan
  max_num_seqs: number
  prefix_caching: boolean
  token_budget: number
  model: string
  model_revision: string | null
  tokenizer_revision: string | null
  model_source: EngineSettings["modelSource"]
  model_destination: string | null
  dtype: EngineSettings["dtype"]
  quantization: EngineSettings["quantization"] | null
  kv_cache_dtype: EngineSettings["kvCacheDtype"]
  lora_adapters: Array<{
    id: string
    name: string
    source: LoraAdapterDraft["source"]
    location: string
    revision: string | null
  }>
  max_model_len: number
  chunked_prefill: boolean
  gpu_memory_utilization: number
  temperature: number
  top_p: number
  seed: number | null
  stop: string[]
  model_configuration: {
    source: ArtifactSourcePayload
    tokenizer: { locator: string; revision: string | null }
    dtype: EngineSettings["dtype"]
    quantization: { kind: EngineSettings["quantization"] }
    lora_adapters: Array<{ id: string; source: ArtifactSourcePayload }>
  }
  vllm_configuration: {
    max_model_len: number
    max_num_seqs: number
    max_num_scheduled_tokens: number
    enable_chunked_prefill: boolean
    enable_prefix_caching: boolean
    gpu_memory_utilization: number
    scheduling_policy: "fcfs" | "priority"
    long_prefill_token_threshold: number
    kv_cache_memory_bytes: number | null
    kv_cache_dtype: EngineSettings["kvCacheDtype"]
    fresh_prefix_cache: true
  }
  profiling_configuration: {
    mode: "off" | "eager_diagnostic"; max_steps: number
    start_step?: number; trigger_request_id?: string | null
    tensor_capture?: {
      enabled: boolean; preset: "small" | "custom"; request_id: string | null
      token_positions: number[]; layer_indices: number[]; head_indices: number[]
      dimension_start: number; dimension_count: number; max_bytes: number
    }
  }
  sampling: {
    temperature: number
    top_p: number
    seed: number
    stop: string[]
  }
}
