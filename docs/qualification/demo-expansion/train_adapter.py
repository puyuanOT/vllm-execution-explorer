#!/usr/bin/env python3
"""Train a bounded teaching adapter; save real nonzero weights and receipts.

This small adapter demonstrates serving routes and memory, not model quality.
It is intentionally local, with immutable checksums retained with each capture.
"""
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DESTINATION = Path("/home/ubuntu/vllm-execution-explorer/teaching-adapter-a")

def main():
    if DESTINATION.exists():
        raise RuntimeError("Retain the existing adapter; choose a new output path for another training run.")
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.train()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.0005)
    examples = [
        ("What does the explorer record?", "The explorer records requests, iterations, model activity and cache blocks."),
        ("What is a KV cache block?", "A block holds keys and values for token positions."),
        ("What is continuous batching?", "The scheduler admits new requests while existing requests are active."),
        ("What is prefix reuse?", "A matching cached prefix avoids recomputing its attention keys and values."),
    ]
    losses = []
    for step in range(8):
        question, answer = examples[step % len(examples)]
        text = tokenizer.apply_chat_template([{"role": "user", "content": question}, {"role": "assistant", "content": answer}], tokenize=False, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        print("step", step + 1, "loss", losses[-1], flush=True)
    model.save_pretrained(DESTINATION, safe_serialization=True)
    receipt = {"kind": "local_teaching_adapter", "base_model": MODEL, "base_revision": REVISION,
               "rank": 8, "target_modules": ["q_proj", "v_proj"], "steps": 8, "seed": 0, "losses": losses,
               "quality_claim": "Serving demonstration only; no downstream quality evaluation.",
               "framework_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in DESTINATION.iterdir() if path.is_file()}}
    (DESTINATION / "training-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

if __name__ == "__main__":
    main()
