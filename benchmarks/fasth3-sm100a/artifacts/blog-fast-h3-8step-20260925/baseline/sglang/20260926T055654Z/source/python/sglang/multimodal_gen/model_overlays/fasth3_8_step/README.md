# FastH3 8-Step V2

This bundled overlay loads `FastVideo/FastVideo-FastH3-8-Step-V2` through the native H3 pipeline. Transformer weights retain their Diffusers values; the video VAE materializer changes tensor names, fused QKV row order and SwiGLU half order to the native loader's representation. The full video and audio VAEs are retained.

The release uses eight transformer forwards over nine sigma nodes, video/audio shifts 10/3, guidance 1, 80% sparse video attention with 64-token tiles, dense prefix attention and learned compression gates. Requests use `num_inference_steps=9` and the canonical `task="t2va"` target schema. The terminal zero is a sigma node, not an additional forward.

For NVIDIA SM100 GPUs, install `fastvideo-kernel==0.3.4` with a CUDA 13.0 build containing SM100a code. Set `FASTVIDEO_VSA_SM100A=1` before launching SGLang. Explicit selection fails if the extension or hardware is unsupported; it does not fall back to Triton. SM120 GPUs cannot execute this SM100a kernel.

```bash
FASTVIDEO_VSA_SM100A=1 sglang serve --model-type diffusion \
  --model-path FastVideo/FastVideo-FastH3-8-Step-V2 \
  --served-model-name FastH3 --num-gpus 4 --ulysses-degree 4 \
  --performance-mode speed --encoder-parallel auto \
  --attention-backend fa \
  --component-attention-backends transformer=VIDEO_SPARSE_ATTN_H3 \
  --attention-backend-config '{"VSA_sparsity":0.8,"vsa_tile_size":64}'
```

The server logs `VSA-H3 provider=sm100a` when selecting the native provider. Odd logical tile counts are padded with a zero-size transport tile after routing; padding does not change selected video tiles or the compression branch.

The tensor materializer is adapted from `kevin-mi/FastH3-4step-Preview-overlay` at `f769cb8001dae335089de7b250364335bc7cb183` under Apache-2.0. V2 release metadata supplies its own sigma ladder and shifts; V1 keeps its separate four-step contract.
