# Demo facets and numerical FP8 qualification

The local existing Configure/Explore application was preserved. This work adds a recording catalogue filtered by observed model, weight storage, KV encoding and LoRA routing. A selector opens only a matching real GPU artifact. Requested compute dtype does not establish observed weight storage; historical recordings without measured storage remain `Not captured`.

## Local checks

- Web tests: 102 passed after adding catalogue matching, independent KV requests, persisted LoRA intent, legacy storage recovery and numerical representation checks.
- Replay tests: 67 passed, including optional FP8 snapshot parsing and rejection of corrupt raw bytes, encoding, scale, accounting and reconstruction metadata.
- Python KV diagnostics tests: 16 passed. Authored fixtures exercise capture ownership, byte bounds, invalid scales and unsupported encodings; they are not GPU evidence.
- Production build passed; Vite reports the existing large application chunk warning.
- Browser at `http://127.0.0.1:5173/`: legacy `Any` filters open recorded GPU artifacts; explicit unrecorded storage needs capture; NVFP4 is unsupported; FP8 weights and E5M2 cache populate independent Configure requests. Mixed without an artifact preserves the intent, opens the adapter field, marks Model as needing attention and disables capture. Entering a browser-QA adapter locator applies three base and three adapter routes and clears the stale missing-artifact error. This locator was a draft-only UI fixture and was never sent to the GPU. None clears the adapter list and all routes. Flash-Next on the current qualification host has a disabled capture action.

## Source qualification

The installed vLLM 0.30 FlashAttention update was inspected read-only on the supplied host. It splits the loaded cache view into K/V and calls the native update with `self.kv_cache_dtype`, `layer._k_scale` and `layer._v_scale`. The instrumentation retains selected native cache bytes and the selected actual K or V scale on the update stream before reuse. CUDA E4M3FN and E5M2 are distinct encodings; ROCm E4M3FNUZ is rejected by this qualification path. Higher precision legacy snapshots retain their original values contract.

- [Native cache-update kernels, vLLM 0.30](https://github.com/vllm-project/vllm/blob/v0.30.0/csrc/libtorch_stable/cache_kernels.cu)
- [CUDA FP8 conversion and scaling, vLLM 0.30](https://github.com/vllm-project/vllm/blob/v0.30.0/csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh)
- [Cache dtype configuration, vLLM 0.30](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/cache.py)

`fp8-torch-cpu-reference.json` is an independent read-only, CPU-only native PyTorch decode of all 256 byte patterns for each encoding, captured from torch 2.13.0+cu130 on the host. All 512 patterns match both the Python and TypeScript decoders. This confirms bit interpretation but does not substitute for validation of a real GPU snapshot or reconstruction scale.

The [official Flash-Next recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) requires a dedicated image and documents a 172.78 GiB FP8 checkpoint. Its H100 deployment is validated with four H100 80GB GPUs and PLE CPU offload. The supplied qualification host has one H100 and a PyPI vLLM 0.30 environment, so an absent Flash-Next recording is shown as Unsupported. A future actual matching qualified recording can still be opened.

Real GPU FP8 snapshot verification completed against the final H100 capture; details and bounds are recorded below. No GPU workload was launched by this verification subtask.

## Independent verification follow-up

`fp8-independent-source-review.json` records a fresh read-only inspection of the installed `FlashAttentionImpl.do_kv_cache_update` and native op wrapper, with their source hashes. CPU reinterpretation of every byte with installed PyTorch again matched the production Python decoder for all 512 E4M3FN/E5M2 patterns. The vLLM 0.30 CUDA implementation uses native slot quotient/remainder addressing, scalar or per-head scale indexing, quantizes `input / scale` and reconstructs `decode(byte) × scale`. The snapshot address, selected scale and same-stream clone follow that boundary; these checks established the encoding and source path before the final capture. Actual GPU slice verification is recorded in the final section below.

The dedicated headed browser session `facet-verification` independently exercised Mixed without an artifact, attached the actual teaching adapter path as a draft, verified three base and three adapter routes, reloaded, and confirmed both intent and routes persisted. Changing to All assigned all six requests to the real configured adapter. Changing to None removed the adapter and restored all six base routes. No assignment alerts remained. The session never launched a capture. Receipts are in `output/playwright/facets-mixed-after-reload.txt`, `facets-all-routes.txt`, and `facets-none-clears-routes.txt`.

One presentation defect was corrected: the Raw bytes heatmap previously reused the signed KV magnitude scale and negative legend. It now uses a sequential unsigned 0–255 scale, explicitly describing encoded bits. Decoded storage and reconstructed values retain their signed magnitude scales. Web tests passed 102/102, FP8 replay tests passed 3/3 including all 512 native reference patterns and corruption rejection, and the production build passed (`facet-verification-*.txt`). These initial checks preceded the final actual GPU numerical and six-case browser verification below.


## Final real GPU artifact and browser verification

The final FP8-KV artifact is `run-5fb78beeb09f` (bundled as `gpu-h100-fp8-kv`), and its BF16 comparison is `run-af950e986ddc`. The CPU-only script [verify_real_fp8.py](verify_real_fp8.py) reads the original retained recording on the host. [fp8-real-gpu-independent.json](fp8-real-gpu-independent.json) records:

- All 24 selected snapshots (positions 0/1 × layers 0/18/35 × heads 0/1 × K/V), totaling 384 components, exactly match native PyTorch E4M3FN byte reinterpretation.
- The recorded scale and same-stream boundary agree with each slice, reconstruction is exactly decoded value × captured scale, and retained bytes total 480 (384 raw bytes plus 24 float32 scales).
- Every snapshot matches its recorded native cache-write participant, manager/kernel slot addressing and loaded storage bounds. This verifies retained address metadata; it does not claim to read a released allocation’s current GPU bytes.
- All 128 first-layer K/V byte values equal the native CPU FP8 cast of the same first-step BF16 baseline vectors divided by the captured scale. Higher layers can differ because prior FP8 cache values change attention results, so higher-layer equality is not claimed.
- This real artifact uses scale 1. It establishes actual byte/scale capture and reconstruction, while nonunit scales and E5M2 remain source/contract-qualified rather than claimed as additional GPU recordings.

The headed browser opened all six actual final artifacts using exact observed model/weight/KV/LoRA filters; each matched one recording, showed its correct ID, stepped its real first iteration and displayed the measured GPU memory panel. [six-case picker receipt](../../../output/playwright/facets-six-real-cases-ui.txt) is the receipt. The Mixed case’s iteration 20 contains Request 2 on the base model and Requests 3/6 on the adapter, with independently inspected route labels ([Mixed route receipt](../../../output/playwright/lora-real-mixed-route-ui.txt), [Mixed iteration screenshot](../../../output/playwright/lora-real-mixed-iteration.png)).

The actual FP8 numerical browser check exercised all 12 position/layer/K-or-V frames in Raw bytes, FP8 values and Reconstructed modes. All 1,152 displayed values matched the retained snapshots, including both captured heads; the unsigned 0–255 raw legend and unknown hatching were visually inspected. Receipt: [FP8 numerical UI receipt](../../../output/playwright/fp8-real-numerical-ui.txt); screenshots: [FP8 raw byte heatmap](../../../output/playwright/fp8-real-raw-byte-heatmap.png) and [FP8 reconstructed table](../../../output/playwright/fp8-real-reconstructed-table.png). The reusable browser scripts operate only visible controls and compare their DOM output against the retained artifacts.

For the actual Qwen3.8-27B recording, the matrix presents 48 GDN rows explicitly as recurrent state with no attention KV blocks; its 128-position display bound is disclosed beside the 784-position iteration summary. The native profiler displays 48 recurrent, 16 attention and 64 feed-forward scoped layer summaries, 784 observed first-iteration packed rows and the lossless activity-name dictionary explanation. The default profiler speed is 0.00001×. Receipts: [27B GDN cell receipt](../../../output/playwright/qwen27-real-gdn-cell-ui.txt), [27B native layer structure receipt](../../../output/playwright/qwen27-real-native-layer-structure.txt); screenshot: [27B GDN matrix screenshot](../../../output/playwright/qwen27-real-gdn-matrix.png). The profile’s declared CPU-detail omissions remain visible and are not relabeled complete.

No new source defect was found in these final checks. The prior Raw bytes legend correction is the only change made by this verification pass. The browser console contained only React DevTools informational entries, with no application errors.
