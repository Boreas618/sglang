# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 released VAE decode contract."""

import subprocess
import sys
import textwrap
from unittest import mock

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3EightStepPipelineConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders import vae_loader
from sglang.multimodal_gen.runtime.loader.utils import load_model_state_dict
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import (
    MiniMaxH3AudioVAE,
    MiniMaxH3VideoVAE,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae import (
    CausalAttention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
    _apply_qk_norm,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
    minimax_h3_encode_reference_audio_rows,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _init_kwargs(config: MiniMaxH3VideoVAEConfig):
    with mock.patch.object(
        AutoencoderKLLegacy, "__init__", autospec=True, return_value=None
    ) as init:
        model = MiniMaxH3VideoVAE(config)
    return model, init.call_args.kwargs


@pytest.mark.parametrize(
    "mode",
    [
        None,
        "auto",
        "tiled",
    ],
)
def test_decode_mode_uses_released_tiled_recipe(mode):
    config = (
        MiniMaxH3VideoVAEConfig()
        if mode is None
        else MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    )
    model, kwargs = _init_kwargs(config)

    assert model.parallel_decode_mode == "tiled"
    assert kwargs["decoder_tiling"] is True
    assert kwargs["parallel_tiling"] is True
    assert kwargs["decoder_parallel"] is False


@pytest.mark.parametrize("mode", ["spatial", "spatial_shard", "patch"])
def test_unvalidated_decode_modes_are_rejected(mode):
    config = MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    with pytest.raises(ValueError, match="use tiled"):
        config.resolved_parallel_decode_mode()


def test_vit_attention_uses_local_usp_backend_dispatch():
    module = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention"
    with (
        mock.patch(f"{module}.current_platform.is_cuda", return_value=True),
        mock.patch(f"{module}.USPAttention", autospec=True) as usp_attention,
    ):
        Attention(heads=2, dim_head=64)

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["skip_sequence_parallel"] is True
    assert kwargs["default_attention_backend"] == AttentionBackendEnum.TORCH_SDPA
    assert kwargs["supported_attention_backends"] == {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }


def test_vit_qk_norm_supports_affine_free_rmsnorm():
    norm = nn.RMSNorm(64, elementwise_affine=False)
    hidden_states = torch.randn(1, 2, 2, 64)

    output = _apply_qk_norm(norm, hidden_states)

    assert output.shape == hidden_states.shape


def test_audio_vae_attention_defaults_to_local_sdpa_and_allows_fa():
    class RecordingFA(nn.Module):
        backend = AttentionBackendEnum.FA
        dtype = torch.bfloat16

        def forward(self, query, key, value):
            self.input_dtype = query.dtype
            return query

    module = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae"
    recording_fa = RecordingFA()
    with (
        mock.patch(f"{module}.current_platform.is_cuda", return_value=True),
        mock.patch(
            f"{module}.USPAttention", autospec=True, return_value=recording_fa
        ) as usp_attention,
    ):
        attention = CausalAttention(in_dim=64, out_dim=32, num_heads=2)
        output = attention(torch.randn(1, 4, 64))

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["causal"] is True
    assert kwargs["skip_sequence_parallel"] is True
    assert kwargs["default_attention_backend"] == AttentionBackendEnum.TORCH_SDPA
    assert kwargs["supported_attention_backends"] == {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }
    assert recording_fa.input_dtype == torch.bfloat16
    assert output.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_audio_snake_first_call_matches_repeated_calls():
    # a fresh process prevents earlier tests from warming a profiling JIT graph
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import torch
                from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae import Snake1d

                torch.manual_seed(42)
                activation = Snake1d(64).cuda().eval()
                with torch.inference_mode():
                    activation.alpha.uniform_(0.1, 2.0)
                    x = torch.randn(2, 64, 4096, device="cuda")
                    original_x = x.clone()
                    original_alpha = activation.alpha.clone()
                    first = activation(x)
                    for _ in range(4):
                        torch.testing.assert_close(activation(x), first, rtol=0, atol=0)
                    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
                    torch.testing.assert_close(activation.alpha, original_alpha, rtol=0, atol=0)
                """
            ),
        ],
        check=True,
        timeout=120,
    )


def _encode_video(vae) -> None:
    vae.encode(torch.zeros(1, 3, 1, 16, 16, device="meta"))


def _encode_reference_audio(vae) -> None:
    minimax_h3_encode_reference_audio_rows(
        vae, "reference.wav", MiniMaxH3AudioVAEConfig().arch_config
    )


@pytest.mark.parametrize(
    "vae_cls, config_cls, encode",
    [
        (MiniMaxH3VideoVAE, MiniMaxH3VideoVAEConfig, _encode_video),
        (MiniMaxH3AudioVAE, MiniMaxH3AudioVAEConfig, _encode_reference_audio),
    ],
)
def test_decode_only_vae_strictly_loads_checkpoint_minus_encoder(
    vae_cls, config_cls, encode
):
    """load_encoder=False must leave unbuilt exactly the modules whose tensors
    the config tells the loader to skip, and encoding must then fail loudly."""
    decode_only_config = config_cls(load_encoder=False)
    # A full module's state is checkpoint-shaped (the native loader restores
    # it strictly); meta tensors keep the full-size layouts free.
    with torch.device("meta"):
        checkpoint = vae_cls(config_cls()).state_dict()
        decode_only = vae_cls(decode_only_config)

    skipped = decode_only_config.skipped_checkpoint_prefixes()
    assert all(
        any(name.startswith(prefix) for name in checkpoint) for prefix in skipped
    )
    kept = {
        name: tensor
        for name, tensor in checkpoint.items()
        if not name.startswith(skipped)
    }
    load_model_state_dict(decode_only, kept, strict=True, assign=True)
    assert config_cls().skipped_checkpoint_prefixes() == ()

    with pytest.raises(RuntimeError, match="load_encoder=false"):
        encode(decode_only)


class _AudioVAELayout(nn.Module):
    """The H3 audio VAE's top-level modules at toy width."""

    def __init__(self, config: MiniMaxH3AudioVAEConfig) -> None:
        super().__init__()
        if config.load_encoder:
            self.encoder = nn.Linear(2, 2)
            self.pre_block = nn.Linear(2, 2)
            self.mean_proj = nn.Linear(2, 2)
            self.logs_proj = nn.Linear(2, 2)
        self.dec_in_proj = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)


class _LoaderServerArgs:
    def __init__(self, pipeline_config) -> None:
        self.pipeline_config = pipeline_config
        self.model_paths = {}
        self.component_weights_paths = {}
        self.component_precisions = {}

    def should_direct_gpu_weight_load_component(self, _component_name):
        return False

    def should_start_component_on_cpu(self, _component_name):
        return False


@pytest.mark.parametrize("stray", [False, True])
def test_native_vae_loader_skips_only_declared_encoder_tensors(tmp_path, stray):
    """The strict native load must drop the unbuilt encoder's tensors and
    still reject any other tensor the module does not own."""
    pipeline_config = FastH3EightStepPipelineConfig()
    # The route a deployment takes: --pipeline-config-path JSON.
    pipeline_config.update_pipeline_config(
        {"audio_vae_config": {"load_encoder": False}}
    )
    component_config = {
        "_class_name": "MiniMaxH3AudioVAE",
        "latents_mean": [0.0] * 32,
        "latents_std": [1.0] * 32,
    }
    checkpoint = {
        f"{module}.{name}": torch.full(shape, float(index))
        for index, module in enumerate(
            ("encoder", "pre_block", "mean_proj", "logs_proj", "dec_in_proj", "decoder")
        )
        for name, shape in (("weight", (2, 2)), ("bias", (2,)))
    }
    if stray:
        checkpoint["stray.weight"] = torch.zeros(1)
    weight_path = tmp_path / "model.safetensors"
    save_file(checkpoint, weight_path)
    loader = vae_loader.VAELoader()

    with (
        mock.patch.object(
            vae_loader,
            "get_diffusers_component_config",
            return_value=component_config,
        ),
        mock.patch.object(
            vae_loader.ModelRegistry,
            "resolve_model_cls",
            return_value=(_AudioVAELayout, None),
        ),
        mock.patch.object(
            vae_loader, "_list_safetensors_files", return_value=[str(weight_path)]
        ),
        mock.patch.object(loader, "target_device", return_value=torch.device("cpu")),
        mock.patch.object(
            vae_loader.current_platform, "optimize_vae", side_effect=lambda vae: vae
        ),
    ):
        if stray:
            with pytest.raises(RuntimeError, match="stray.weight"):
                loader.load_customized(
                    str(tmp_path), _LoaderServerArgs(pipeline_config), "audio_vae"
                )
            return
        vae = loader.load_customized(
            str(tmp_path), _LoaderServerArgs(pipeline_config), "audio_vae"
        )

    assert not hasattr(vae, "encoder")
    assert torch.equal(vae.decoder.weight, checkpoint["decoder.weight"])
    assert torch.equal(vae.dec_in_proj.bias, checkpoint["dec_in_proj.bias"])
