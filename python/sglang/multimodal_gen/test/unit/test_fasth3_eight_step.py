# SPDX-License-Identifier: Apache-2.0
"""Eight-step release admission and native VSA numerical contracts."""

import json
from pathlib import Path

import pytest
import torch
from sglang.multimodal_gen.configs.sample.minimax_h3 import (
    FastH3EightStepSamplingParams,
)
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3_kernels import (
    vsa_h3_block_sparse_attn_forward,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)


def test_eight_step_release_admission():
    info = get_model_info("FastVideo/FastVideo-FastH3-8-Step-V2")
    params = info.sampling_param_cls(prompt="A coastal landscape")
    assert params.num_inference_steps == 9
    assert params.guidance_scale == 1.0
    with pytest.raises(ValueError, match="nine sigma nodes"):
        FastH3EightStepSamplingParams(prompt="p", num_inference_steps=5)
    path = Path(__file__).parents[2] / "model_overlays/fasth3_8_step/model_index.json"
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(json.loads(path.read_text()))
    assert metadata.denoising_sigmas == (
        0.999,
        0.874,
        0.749,
        0.624,
        0.5,
        0.375,
        0.25,
        0.125,
        0.0,
    )
    assert metadata.sigma_shift_scales == {"video": 10.0, "audio": 3.0}


def test_eight_step_timestep_outputs():
    params = FastH3EightStepSamplingParams(
        prompt="A coastal landscape",
        task="t2va",
        conditions=[],
        target={
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
        },
    )
    batch = Req(sampling_params=params, extra=params.build_request_extra())
    stage = MiniMaxH3TimestepPreparationStage(
        sigma_shift_scales={"video": 10.0, "audio": 3.0},
        denoising_sigmas=(
            0.999,
            0.874,
            0.749,
            0.624,
            0.5,
            0.375,
            0.25,
            0.125,
            0.0,
        ),
    )
    result = stage.forward(batch, None)
    # Independent scalar reference from the release's integer DMD ladder.
    base = [999, 874, 749, 624, 500, 375, 250, 125, 0]
    expected = [10 * x / (1000 + 9 * x) for x in base]
    assert len(result.timesteps) == 8
    assert result.sigmas == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("tiles", [3, 4])
def test_sm100a_ragged_sparse_attention(monkeypatch, tiles):
    """Odd/even block transport must preserve the same masked softmax."""
    monkeypatch.setenv("FASTVIDEO_VSA_SM100A", "1")
    torch.manual_seed(81)
    q, k, v = [
        torch.randn(1, 2, tiles * 64, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    sizes = torch.tensor(
        [64, 17] + [33] * (tiles - 2), device="cuda", dtype=torch.int32
    )
    # Every query tile selects only first and last key tiles. Prefix and
    # interior ragged padding remain masked independently of query padding.
    index = (
        torch.tensor([0, tiles - 1], device="cuda", dtype=torch.int32)
        .expand(1, 2, tiles, 2)
        .contiguous()
    )
    count = torch.full((1, 2, tiles), 2, device="cuda", dtype=torch.int32)
    actual = vsa_h3_block_sparse_attn_forward(q, k, v, index, count, sizes)
    positions = torch.arange(tiles * 64, device="cuda")
    selected = ((positions // 64 == 0) | (positions // 64 == tiles - 1)) & (
        positions % 64 < sizes[positions // 64]
    )
    logits = (q.float() @ k.float().transpose(-2, -1)) / (128**0.5)
    expected = torch.softmax(logits.masked_fill(~selected, -torch.inf), -1) @ v.float()
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
