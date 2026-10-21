# vLLM Execution Explorer

An operator tool for configuring a timed request sequence, capturing an instrumented vLLM 0.30.x execution, and replaying only the facts present in that recording.

The application has two pages:

- **Configure** builds arrival batches, chooses a local or SSH execution host, resolves model weights and optional LoRA adapters, and sets vLLM and sampling options.
- **Explore** replays the captured workflow across request intake, the scheduler, KV cache, model execution, and output. A single selection opens recorded details; component exploration exposes the observations inside that stage.

The explorer labels authored drafts and deterministic fixtures separately from GPU evidence. Playback never infers a scheduler decision after a trace gap. A live run starts its workload clock only after model loading, warmup, and a verified prefix-cache reset.

## Repository layout

- `instrumentation/vllm_030` records schema 1.0.0 events from the pinned vLLM process.
- `runner` owns setup, workload dispatch, the instrumented loopback HTTP boundary, recording, SSH bootstrap, and artifact transfer.
- `replay` is the deterministic TypeScript reducer and timeline projection.
- `web` is the React interface for Configure and Explore.
- `web/public/recordings` contains saved real GPU lessons for the demo library and retained synthetic regression fixtures.
- `trace-schema` contains the recording schema.

## Local setup

Use Python 3.11.15 and Node.js 22.14.0, matching CI. With `uv` and Node.js/npm
installed, run these commands from the repository root. Local controller,
fixture, and replay development does not require vLLM, PyTorch, CUDA, or a GPU.

```bash
node --version
uv python install 3.11.15
uv venv --python 3.11.15 --seed .venv
.venv/bin/python -m pip install \
  fastapi==0.141.1 \
  httpx==0.28.1 \
  pytest==9.1.1 \
  python-multipart==0.0.32 \
  uvicorn==0.53.0
.venv/bin/python -m pip install --no-deps -e .
npm ci --prefix web
```

Saved demo manifests and small fixtures are tracked in Git. Large recordings use lossless compressed assets pinned to an immutable public commit. Online playback downloads the selected recording automatically. For offline playback and fixture tests, install `curl` and run once:

```bash
npm run recordings:download
```

This verifies the compressed parts, restores the original recording files, and checks every original SHA-256 and byte count. The resulting local cache is ignored by Git. Repeating the command verifies and reuses it; no GPU or model download is needed. Replay has no separate npm dependencies.

Start the local controller and UI in separate terminals:

```bash
.venv/bin/python -m runner
```

```bash
npm run dev --prefix web -- --strictPort
```

Open `http://127.0.0.1:5173`. Leave the controller endpoint blank to use the
local controller at `http://127.0.0.1:8765` through Vite's `/api` proxy.
The default draft contains six requests in batches at 0 ms, 100 ms, and 120 ms.
Live execution starts only when **Prepare and capture** is pressed.

A fresh browser opens **Explore** with a real GPU demo. Choose a recorded model, weight precision, KV-cache dtype, and LoRA configuration, then press **Load demo**. Unsupported combinations are not presented as captured evidence. **Capture your own run** opens the configuration workspace. Demo playback preserves your editable draft and starts no live inference.

## Verification

Prepare the verified offline fixtures first:

```bash
npm run recordings:download
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q runner instrumentation
npm test --prefix replay
npm test --prefix web
npm run build --prefix web
git diff --check
```

Deterministic fixtures remain available to regression tests and the controller's
fixture execution mode. The user-facing demo uses a saved real GPU run;
synthetic fixture timings never serve as GPU measurements.

## Live execution

Live requests cross the explorer's instrumented loopback HTTP endpoint and enter one engine-owner thread. That thread exclusively initializes vLLM, admits work, steps the engine, cancels requests, and closes the engine. The endpoint is an explorer capture boundary; it is not the stock vLLM OpenAI-compatible server.

Model weights can be resolved from:

- a pinned Hugging Face repository on the execution host;
- an absolute path already on the execution host; or
- a path on the operator machine, copied to the connected SSH host with resumable `rsync` transfer state.

Hugging Face tokens are transient and excluded from settings, events, and manifests. Configured LoRA adapters receive stable numeric IDs, are resolved on the execution host, and are routed per request. Quantization values are passed to vLLM during preparation. Both features remain marked **GPU validation required** until they pass acceptance with the selected model and host.

## SSH GPU host

Choose **Remote GPU over SSH** on Configure, enter the user, host, and port, select the private key, then run the host check. The local controller:

1. copies the project to an isolated directory on the remote host;
2. creates a virtual environment and installs the controller dependencies;
3. installs or repairs vLLM 0.30.0 when needed and waits for that installation to finish;
4. starts a loopback-only remote runner; and
5. opens an SSH tunnel at `http://127.0.0.1:8766`.

The GPU instance must already provide Linux, an NVIDIA driver, CUDA visibility, Python, SSH, and enough disk space. The operator machine needs `ssh` and `rsync`. The private key is stored outside the project on the operator machine and is never written into a recording.

A successful host check requires a visible NVIDIA GPU, PyTorch CUDA access, and vLLM 0.30.x. Turning automatic setup off skips project copying and dependency installation, so the selected host must already contain a prepared explorer checkout and runtime.

## Acceptance status

The v0.1 workflow passed UI-driven acceptance on an NVIDIA A10 with vLLM 0.30.0. Verified paths include automatic SSH setup, real HTTP capture, execution-host and operator-local model weights, resumable transfer, LoRA loading and routing, AWQ and GPTQ startup, chunked prefill, prefix reuse, cancellation, failure reporting, deterministic replay, responsive layouts, and keyboard-accessible inspection.

The preserved structural evidence for the final operator-local transfer run contains 270 ordered events for six requests and passes all eight integrity checks. See [`docs/acceptance`](docs/acceptance/README.md) for the sanitized recording, checksums, environment, and limitations.

FP8 did not start for the tested OPT checkpoint on the A10 and remains an explicitly reported model/runtime incompatibility. Bitsandbytes stays disabled when its vLLM plugin is absent. Private or gated Hugging Face access, multi-GPU execution, instrumentation-overhead measurement, and CUDA-kernel tracing are not part of the accepted v0.1 evidence.

## Public website

[Open vLLM Execution Explorer](https://vllm-execution-explorer.vast-rhino-1030.chatgpt.site/). The public build preserves Explore and the configuration workspace; live cluster capture requires the local controller. It never accepts SSH credentials for a hosted capture.

`npm ci` installs the frontend dependencies; `npm run build` creates the public static build in `dist/` with local capture actions disabled. `npm run build --prefix web` creates the local application build with capture enabled. The static build excludes decoded recording caches and native profiler archives, and streams the same verified recording transports from the immutable public asset commit. `application-revision.json` identifies the application revision used for a publication.

Curated qualification reports, reproduction scripts, and small test fixtures live under `docs/qualification`. Full capture archives are published separately as versioned [release assets](https://github.com/puyuanOT/vllm-execution-explorer/releases). Scratch output and browser caches are not source artifacts.
