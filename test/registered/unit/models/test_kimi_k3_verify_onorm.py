"""K3 verify norm preparation and the model/backend norm handoff."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestKimiK3VerifyOnorm(CustomTestCase):
    def test_prepare_verify_norm_independently_of_decode_shape(self):
        for heads, decode_ready in ((6, False), (12, True)):
            with self.subTest(heads=heads):
                dim = heads * 128
                owner = SimpleNamespace(
                    attn=SimpleNamespace(
                        head_v_dim=128,
                        conv_weights=torch.zeros(3 * dim, 4, dtype=torch.float32),
                        bias=None,
                        A_log=torch.zeros(heads, dtype=torch.float32),
                        dt_bias=torch.zeros(dim, dtype=torch.float32),
                    ),
                    o_norm=SimpleNamespace(
                        weight=torch.arange(128, dtype=torch.bfloat16), eps=1e-6
                    ),
                    _kda_fused_decode_ready=False,
                    _kda_fused_verify_onorm_ready=False,
                )
                with (
                    patch.object(kimi_k3, "_is_hip", False),
                    patch.object(kimi_k3, "_is_npu", False),
                    patch.object(kimi_k3, "rank0_log"),
                ):
                    kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(owner)

                self.assertEqual(owner._kda_fused_decode_ready, decode_ready)
                self.assertTrue(owner._kda_fused_verify_onorm_ready)
                weight, eps = owner.attn._k3_verify_onorm_args
                torch.testing.assert_close(weight, owner.o_norm.weight.float())
                self.assertTrue(weight.is_contiguous())
                self.assertEqual(eps, owner.o_norm.eps)
                self.assertEqual(
                    hasattr(owner.attn, "_k3_fused_decode_args"), decode_ready
                )

    def test_tp16_verify_handoff_preserves_separate_norm_fallback(self):
        for mode, consumed in (
            (ForwardMode.TARGET_VERIFY, True),
            (ForwardMode.TARGET_VERIFY, False),
            (ForwardMode.DECODE, False),
            (ForwardMode.EXTEND, False),
        ):
            with self.subTest(mode=mode, consumed=consumed):
                tokens, heads, head_dim = 8, 6, 128
                gate = torch.zeros(tokens, heads * head_dim, dtype=torch.bfloat16)
                output = torch.zeros(1, tokens, heads, head_dim, dtype=torch.bfloat16)

                class Attention:
                    _k3_onorm_gate = None
                    offered = False

                    def __call__(self, *args, **kwargs):
                        self.offered = self._k3_onorm_gate is gate
                        self._k3_onorm_consumed = consumed
                        return output

                attention = Attention()
                norm = Mock(return_value=output)
                owner = SimpleNamespace(
                    _kda_fused_decode_ready=False,
                    _kda_hip_fused_decode_ready=False,
                    _kda_fused_verify_onorm_ready=True,
                    do_fuse_qkvbfg=False,
                    use_full_rank_gate=False,
                    head_dim=head_dim,
                    all_reduce_fusion=False,
                    attn=attention,
                    forward_qkvbfg=lambda hidden: (
                        torch.zeros(tokens, 3 * heads * head_dim),
                        torch.zeros(tokens, heads),
                        torch.zeros(tokens, heads * head_dim),
                        gate,
                    ),
                    o_norm=norm,
                    o_proj=lambda value: (value, None),
                )
                batch = SimpleNamespace(forward_mode=mode)
                result = kimi_k3.KimiK3DeltaAttention.forward(
                    owner, None, None, batch, None
                )

                self.assertEqual(result.shape, (tokens, heads * head_dim))
                self.assertEqual(attention.offered, mode.is_target_verify())
                self.assertIsNone(attention._k3_onorm_gate)
                self.assertEqual(norm.call_count, 0 if consumed else 1)


if __name__ == "__main__":
    unittest.main()
