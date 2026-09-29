"""GPU feasibility entrypoint.

Running a proof requires vLLM 0.30.x, a CUDA device, and pinned model revisions.
A Mac, or a Linux machine without that stack, gets a preflight message and does
not start a job. The staggered pair, prefix-cache reset, and host step-time
comparison run only after that preflight passes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"


def describe_gpu_blocker() -> str | None:
    try:
        import vllm
    except ImportError:
        return (
            "vLLM 0.30.x is not installed here. Replay and the fixture engine still work. "
            "A live engine needs vLLM 0.30.x on a CUDA Linux host."
        )
    version = str(getattr(vllm, "__version__", ""))
    if not version.startswith("0.30."):
        return f"Found vLLM {version}. The live engine is pinned to vLLM 0.30.x."
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed next to vLLM."
    if not torch.cuda.is_available():
        return "CUDA is not available on this machine. Choose an SSH GPU host for a live run."
    return None


def main(argv: list[str] | None = None) -> int:
    bindir = str(Path(sys.executable).resolve().parent)
    path = os.environ.get("PATH", "")
    if bindir not in path.split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([bindir, path] if path else [bindir])
    args = sys.argv[1:] if argv is None else list(argv)
    blocker = describe_gpu_blocker()
    if blocker:
        print(blocker, file=sys.stderr)
        return 2
    parsed = _parser().parse_args(args)
    if not parsed.model_revision or not parsed.tokenizer_revision:
        print(
            "Pass --model-revision and --tokenizer-revision to start a feasibility run.",
            file=sys.stderr,
        )
        return 2
    try:
        return run_gpu_feasibility(parsed)
    except Exception as exc:  # noqa: BLE001 - the command should report the failure and exit
        print(str(exc), file=sys.stderr)
        return 1


def run_gpu_feasibility(args: Any) -> int:
    from runner.load import load_recording
    from runner.manifest import manifest_skeleton
    from runner.report import evaluate, instrumentation_overhead
    from runner.stagger import run_staggered
    from runner.vllm_session import VllmSession
    from vllm_030.collector import Recording

    directory = Path(args.output)
    recording = Recording.open(directory, run_id=directory.name)
    session = VllmSession.open(recording, args)
    try:
        result = run_staggered(session, recording, args.prompt_a, args.prompt_b)
        disabled = session.disabled_step_times(args.prompt_a, len(result.step_times_ns))
        manifest = manifest_skeleton(
            directory.name,
            execution_kind="gpu",
            execution_target=session.execution_target or {},
            arrivals=result.arrivals,
        )
        manifest["engine_output_token_ids"] = result.engine_output_token_ids
        manifest["instrumentation_overhead"] = instrumentation_overhead(result.step_times_ns, disabled)
        if session.cache_geometry is not None:
            manifest["cache_geometry"] = session.cache_geometry
        for source_name, manifest_key in (
            ("engine_record", "engine"),
            ("model_record", "model"),
            ("tokenizer_record", "tokenizer"),
            ("hardware_record", "hardware"),
            ("engine_settings", "engine_settings"),
            ("sampling_record", "sampling"),
            ("initial_cache_state", "initial_cache_state"),
        ):
            value = getattr(session, source_name, None)
            if value is not None:
                manifest[manifest_key] = value
        recording.close(manifest)
    finally:
        session.close()
    report = evaluate(load_recording(directory), profile="feasibility")
    print(report.text())
    return 0 if report.ok else 1


def _parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Record one staggered vLLM 0.30.0 feasibility run")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--output", default="recordings/runs/feasibility-stagger")
    parser.add_argument("--prompt-a", default="The capital of France is")
    parser.add_argument("--prompt-b", default="The capital of France is Paris, and")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
