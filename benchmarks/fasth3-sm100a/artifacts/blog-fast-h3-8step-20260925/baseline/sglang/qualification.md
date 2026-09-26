# SGLang FastH3 8-Step V2 qualification

Canonical run: `20260926T055654Z` on four NVIDIA GB200 GPUs. SGLang upstream is `8772916e06e631f3834fc7ba86f36d9c88a23082`; the exact local patch and changed files are retained in the run directory. The checkpoint is FastVideo/FastVideo-FastH3-8-Step-V2 at `3da2ddfe1954d9cda4c05b643dc0f26007a655c5`.

Six disjoint warmup requests and six measured requests passed full media validation at concurrency one. The workload combines 5/10/15 seconds with 1K/10K input tokens at 1344×768, 24 FPS, and aligned 124/243/362 frames. Outputs contain H.264 video and stereo 32 kHz AAC, with nonzero image variance and audio RMS and audio/video durations within one frame. This is qualification evidence, not a formal latency or throughput tournament.

The bundled overlay preserves checkpoint tensor values while mapping the full video VAE to native names and fused projection order. Sampling uses the released nine-node ladder `[.999,.874,.749,.624,.5,.375,.25,.125,0]`, eight DiT forwards, video/audio shifts 10/3, guidance one, 80% sparse video routing, 64-token tiles and the trained compression gates.

The transformer alone uses `VIDEO_SPARSE_ATTN_H3`; the text encoder and VAEs retain dense attention. `FASTVIDEO_VSA_SM100A=1` selects the FastVideo native SM100a extension and fails if unavailable. The extension is the CUDA 13.0 source build of fastvideo-kernel 0.3.4 also qualified with vLLM-Omni. Ulysses is four, encoder parallelism is auto, components remain resident, and upstream H3 speed mode leaves torch.compile disabled. No approximate cache is enabled.

| Duration | Prompt tokens | Complete-response latency (s) |
| --- | ---: | ---: |
| 5s | 1000 | 8.405 |
| 5s | 10000 | 11.395 |
| 10s | 1000 | 17.412 |
| 10s | 10000 | 22.310 |
| 15s | 1000 | 28.866 |
| 15s | 10000 | 35.459 |

Original retained bodies: 45,973,072 bytes across all twelve outputs. Client peak RSS: 132,120,576 bytes. Peak GPU memory: 114,885 MiB on one GPU. Encoding settings are backend-native; encoded byte volumes differ across systems and these isolated samples do not establish a performance ranking.

## Reproduction

Use `20260926T055654Z/sm100a.toml`, resolve it with `uniserve-eval --config ... plan sglang-bf16-qualification-c1`, then execute the same profile with `run`. The source lock, patch including added files, environment freeze, kernel binary hash, server logs, per-request records, complete original MP4s and media checks are colocated. SGLang does not expose `/version`; the source lock supplies its identity.

Fifteen focused tests passed, including independent masked-softmax comparisons for odd/even native tile counts, ragged masks, learned compression, the exact eight-step schedule and V1 regression coverage. Upstream-selected Ruff checks and git diff whitespace checks passed. `uv pip check` reports the NVIDIA `manylinux2014_sbsa` wheel tag as a platform mismatch; the installed cuSPARSELt ELF is ARM aarch64, as recorded in environment-check.log.

## Excluded preparation evidence

`20260926T054249Z` stopped before requests to correct the server output location. `20260926T054743Z` failed warmup because global H3 VSA reached the Qwen3-VL text encoder. The supported command scopes H3 VSA to the transformer. Neither run supplies canonical timings.

SM100a requires SM100 hardware. The prepared eight-GB200 profiles remain unqualified; RTX PRO 6000 is SM120 and is not eligible for this SM100a benchmark policy. UniServe retains its independently implemented, shape-selected SM100 CUDA/CuTe provider.
