# FastH3 8-Step V2 on GB200

This branch contains the SGLang implementation and frozen qualification artifacts for `FastVideo/FastVideo-FastH3-8-Step-V2@3da2ddfe1954d9cda4c05b643dc0f26007a655c5`, based on upstream `8772916e06e631f3834fc7ba86f36d9c88a23082`. The full checkpoint needs no additional LoRA. The bundled overlay preserves tensor values while adapting the native VAE representation, and the released nine sigma nodes produce eight transformer forwards.

`FASTVIDEO_VSA_SM100A=1` selects the native FastVideo SM100a kernel. Attention must be scoped to the transformer with `--attention-backend fa --component-attention-backends transformer=VIDEO_SPARSE_ATTN_H3`. The [model recipe](../../python/sglang/multimodal_gen/model_overlays/fasth3_8_step/README.md) documents serving and request semantics, including SGLang's nine-node `num_inference_steps` convention.

## Frozen evidence

The [canonical run](artifacts/blog-fast-h3-8step-20260925/baseline/sglang/20260926T055654Z/) retains the exact TOML and resolved plan, source snapshot, environment, kernel identity, server logs, raw request records, media inspection and all twelve original MP4s. [FILES.sha256](FILES.sha256) covers the exact evidence bytes. The implementation is committed directly in this repository; archived patches preserve the identity of the measured working tree.

Qualification uses four GB200 GPUs, BF16, 1344×768 at 24 FPS, concurrency one, six excluded warmups and six disjoint measured requests: 5/10/15 seconds × 1K/10K prompt tokens. All twelve outputs passed full H.264/AAC decoding, frame counts, stereo 32 kHz audio, audiovisual alignment and nondegenerate media checks. The measured complete-response latencies in that order were 8.405, 11.395, 17.412, 22.310, 28.866 and 35.459 seconds. Fifteen regression tests passed, including actual GPU sparse-attention numerics and the V1/V2 schedule contracts; the original validation log is included.

These are qualification observations, not a formal latency/throughput campaign or a backend performance ranking. Native encoding differs across backends. Eight-GPU deployments are not qualified, and SM120 cannot execute this SM100a kernel.

## Reproduce

1. Install this branch with the `python[diffusion]` extra. Match the recorded environment: Python 3.12, PyTorch 2.13.0+cu130 and CUDA 13.0 on ARM64 GB200. Provision the pinned checkpoint and its referenced base components.
2. Install `fastvideo-kernel==0.3.4` built for SM100a. The included reproduction wheel is the tested ARM64/Python 3.12 build; the shared library SHA-256 is `bbdb129f92f1cff8c91728c4bb3205d458064730e9080e5610efe7e5b7566733`. Do not use that wheel on a different architecture or ABI.
3. Prepare the evaluator from `Boreas618/uniserve@9458bca425a96d5bc45f5e97da9d2986449fad3e`, apply `artifacts/blog-fast-h3-8step-20260925/reproduction/uniserve.patch`, and install its locked GPU environment. The reproduction directory also retains the evaluator source, lockfile and source identity. Copy this directory's `artifacts/` tree into that evaluator checkout without replacing existing runs.
4. The frozen profile is `artifacts/blog-fast-h3-8step-20260925/baseline/sglang/20260926T055654Z/sm100a.toml`. Preserve the recorded `/workspace/UniServe` layout, putting this branch at `refs/sglang-current`, or copy the profile and consistently relocate the repository, environment, checkpoint, dataset and output paths. Keep the request manifests, numerical settings and load protocol unchanged.
5. From the evaluator checkout, resolve `.venv/bin/uniserve-eval --config <profile> plan sglang-bf16-qualification-c1`, then execute the matching `run sglang-bf16-qualification-c1 --output-root <fresh-directory>`. Run only one measurement point at a time and retain warmups and original response bodies.

Prepared campaign profiles and corpus authoring inputs are included for reproducibility; their presence does not certify unexecuted campaign points.
