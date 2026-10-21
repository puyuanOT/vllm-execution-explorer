# Original capture and qualification archives

The release archives preserve the complete original `docs/qualification` and `web/public/recordings` trees from the `codex/kv-layer-inspection` worktree on 2026-10-21, including tracked and untracked files, historical attempts, canonical recording files and raw profiler traces. The source trees were read without modification. No files outside these two trees were included.

| Release asset | Files | Original bytes | Archive bytes |
| --- | ---: | ---: | ---: |
| `qualification-evidence-2026-10-21.tar.gz` | 842 | 2,372,811,570 | 865,500,324 |
| `original-recordings-2026-10-21.tar.gz` | 102 | 1,636,582,629 | 611,322,598 |

Archive paths are relative to the repository root. Both files are below GitHub's [2 GiB per release asset limit](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Integrity checks

`source-file-manifest.json` records the relative path, byte count and SHA-256 of all 944 original files. Every archived member was read back and compared with that inventory; all 944 matched. The source inventory and all source file hashes were checked again after packaging and remained unchanged. There were no source symlinks, nested archive links, omitted source paths or unexpected archive members.

`release-assets.json` records the archive byte counts and SHA-256 values; `SHA256SUMS` provides the conventional checksum list:

```text
ada8ed19d0afd2c3cc65918134a615aa4d9b43c5d541dc62b9022b1986d3c5b6  qualification-evidence-2026-10-21.tar.gz
98f71b12e2ffe3776ff2fe249460821faa940f55a8621f6a85aade9f6cb3d74e  original-recordings-2026-10-21.tar.gz
```

## Credential audit

Before packaging, the audit scanned all 944 source files and recursively decoded gzip and tar contents: 1,755 streams, 24,391,711,692 decoded bytes, 65 gzip containers, seven tar containers and 818 nested archive members. No credential candidates were found. The report stores file locations and rule names only; it never stores matched secret values.

`credential-audit.json` records the exact scope, container paths, rule families, counts and verification results. Rules cover private-key headers, GitHub, AWS, Google, Slack, Hugging Face and provider token formats, bearer credentials, URL passwords and common sensitive literal assignments. This is a content pattern audit, with the usual limitation that arbitrary encoding or secret text contained only in image pixels is not recognizable by these rules.

These archives contain the historical evidence as recorded. Their inclusion preserves earlier unsuccessful attempts as well as successful qualification; it does not expand the qualification claims made by each original recording.
