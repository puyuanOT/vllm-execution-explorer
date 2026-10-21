# Local tokenizer feasibility

The version 2 prepared prompts were encoded from the two pinned repositories' `tokenizer.json` files with `tokenizers==0.22.1`. These are the literal ChatML strings in the request plans, with `add_special_tokens=False`, no automatic chat template, no padding, and no truncation. Tokenizer artifacts and each prompt/token-ID sequence are hashed in `output/playwright/oct13-realistic-demos/tokenizer-evidence.json`.

| Request | Output cap | Qwen3-4B prompt tokens | Qwen3.8-27B prompt tokens |
| --- | ---: | ---: | ---: |
| Request 1 | 64 | 1,049 | 1,060 |
| Request 2 | 96 | 1,046 | 1,057 |
| Request 3 | 128 | 599 | 606 |
| Request 4 | 160 | 1,046 | 1,057 |
| Request 5 | 192 | 576 | 583 |
| Request 6 | 256 | 613 | 621 |

Every prompt plus its maximum output fits the configured 2,048-token context. The largest sum is 1,217 tokens. All five dense variants have identical prompts and caps; LoRA routing does not change those strings.

Requests 1, 2, and 4 have a common token prefix of **988 tokens** with Qwen3-4B, and **999 tokens** with Qwen3.8-27B. This permits 61 complete 16-token prefix blocks for the dense model and at least one complete 784-token prefix block for the hybrid model. These counts establish a matching-prefix opportunity. They do not establish an actual cache hit; execution order, completion timing, cache retention, and adapter identity still determine reuse.

Pinned tokenizer sources:

- [Qwen3-4B-Instruct-2507, cdbee75f…](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer.json)
- [Qwen3.8-27B, 1d4bf0f2…](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json)

Reproduce with a Python environment containing `tokenizers==0.22.1`:

```sh
python docs/qualification/realistic-demos/verify_tokenizers.py \
  --artifacts output/playwright/oct13-realistic-demos/tokenizer-artifacts \
  --output output/playwright/oct13-realistic-demos/tokenizer-evidence.json
```

If the environment needs an organization-provided trust bundle, pass `--ca-bundle` with that existing trusted certificate file. The local run used `/etc/ssl/certs/opentable-ca-bundle.pem`; TLS verification stayed enabled.

This local tokenizer check uses no model weights or GPU. Native prompt ID/count comparisons are retained separately as `native-tokenizer-assessment.json` in each assessed capture case; they establish parity only for that specific run. Server qualification also verifies memory and prefix reuse, inspects generated responses, and confirms that actual accepted outputs fall within 64–256 tokens. The output caps alone do not guarantee those lengths. Version 1 tokenizer notes and evidence remain archived separately.
