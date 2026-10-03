"""Resolve movable Hub references once before loading any model artifacts."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Callable


def resolve_model_revisions(body: Any, *, model_info: Callable[..., Any] | None = None) -> dict[str, Any]:
    repository = str(getattr(body, "model", "") or "").strip()
    if not repository:
        raise RuntimeError("Select a model repository or path before preparation.")
    kind = getattr(body, "model_source_kind", "huggingface")
    tokenizer = str(getattr(body, "tokenizer", None) or repository)
    requested_model = getattr(body, "model_revision", None) or "main"
    requested_tokenizer = getattr(body, "tokenizer_revision", None)
    if kind != "huggingface":
        return {"model": {"repository": repository, "requested_revision": getattr(body, "model_revision", None), "resolved_revision": getattr(body, "model_revision", None)},
                "tokenizer": {"repository": tokenizer, "requested_revision": requested_tokenizer, "resolved_revision": requested_tokenizer},
                "source": "execution_host_artifact"}

    def resolve(repo: str, revision: str) -> str:
        if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            return revision.lower()
        nonlocal model_info
        if model_info is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise RuntimeError("Hugging Face revision resolution requires huggingface_hub on the execution host.") from exc
            model_info = HfApi().model_info
        try:
            info = model_info(repo, revision=revision, token=getattr(body, "huggingface_token", None))
        except Exception as exc:
            raise RuntimeError(f"Could not resolve the {repo!r} revision on Hugging Face ({type(exc).__name__}). Check repository access and the revision.") from exc
        sha = str(getattr(info, "sha", "") or "")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
            raise RuntimeError(f"Hugging Face did not return an immutable commit for {repo!r}.")
        return sha.lower()

    model_sha = resolve(repository, str(requested_model))
    tokenizer_ref = requested_tokenizer or (model_sha if tokenizer == repository else "main")
    tokenizer_sha = model_sha if tokenizer == repository and tokenizer_ref in {requested_model, model_sha} else resolve(tokenizer, str(tokenizer_ref))
    body.model_revision = model_sha
    body.tokenizer_revision = tokenizer_sha
    resolution = {
        "source": "huggingface_commit_resolution",
        "model": {"repository": repository, "requested_revision": requested_model, "resolved_revision": model_sha},
        "tokenizer": {"repository": tokenizer, "requested_revision": requested_tokenizer or (requested_model if tokenizer == repository else "main"), "resolved_revision": tokenizer_sha},
    }
    raw = deepcopy(getattr(body, "model_configuration", None) or {})
    source = dict(raw.get("source") or {"kind": "huggingface", "repository": repository})
    source["revision"] = model_sha
    raw["source"] = source
    raw["tokenizer"] = {**(raw.get("tokenizer") or {}), "locator": tokenizer, "revision": tokenizer_sha}
    body.model_configuration = raw
    return resolution
