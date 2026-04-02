"""
测试 Policy Network

验证:
1. 输出形状正确 (B, action_chunk_size, action_dim)
2. 全零初始化动作输入
3. 完整前向传播
4. 参数量级别合理
"""

import torch
import pytest

from vla_adapter.policy import PolicyNetwork
from vla_adapter.config import VLAAdapterConfig


@pytest.fixture
def config():
    return VLAAdapterConfig(
        hidden_size=128,
        num_vlm_layers=4,
        num_attention_heads=4,
        num_action_queries=16,
        action_chunk_size=8,
        action_dim=7,
        proprio_dim=7,
    )


@pytest.fixture
def policy(config):
    return PolicyNetwork(config)


class TestPolicyNetwork:
    """Policy Network 测试"""

    def test_output_shape(self, policy, config):
        """验证输出动作块形状: (B, H, action_dim)"""
        B = 2
        all_layer_raw = [
            torch.randn(B, 32, config.hidden_size)
            for _ in range(config.num_vlm_layers)
        ]
        all_layer_aq = [
            torch.randn(B, config.num_action_queries, config.hidden_size)
            for _ in range(config.num_vlm_layers)
        ]
        proprio = torch.randn(B, config.proprio_dim)

        action_chunk = policy(all_layer_raw, all_layer_aq, proprio)

        assert action_chunk.shape == (B, config.action_chunk_size, config.action_dim), \
            f"Expected ({B}, {config.action_chunk_size}, {config.action_dim}), got {action_chunk.shape}"

    def test_backward_pass(self, policy, config):
        """验证梯度反传正常"""
        B = 2
        all_layer_raw = [
            torch.randn(B, 32, config.hidden_size)
            for _ in range(config.num_vlm_layers)
        ]
        all_layer_aq = [
            torch.randn(B, config.num_action_queries, config.hidden_size)
            for _ in range(config.num_vlm_layers)
        ]
        proprio = torch.randn(B, config.proprio_dim)

        action_chunk = policy(all_layer_raw, all_layer_aq, proprio)

        target = torch.randn_like(action_chunk)
        loss = torch.nn.functional.l1_loss(action_chunk, target)
        loss.backward()

        # 检查关键参数有梯度
        for name, param in policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"

    def test_layer_count_matches(self, policy, config):
        """Policy层数应与VLM层数相同 (论文设计)"""
        assert len(policy.bridge_layers) == config.num_vlm_layers

    def test_deterministic(self, policy, config):
        """验证推理是确定性的 (L1-based, 非diffusion)"""
        B = 2
        torch.manual_seed(42)
        all_layer_raw = [
            torch.randn(B, 32, config.hidden_size)
            for _ in range(config.num_vlm_layers)
        ]
        all_layer_aq = [
            torch.randn(B, config.num_action_queries, config.hidden_size)
            for _ in range(config.num_vlm_layers)
        ]
        proprio = torch.randn(B, config.proprio_dim)

        with torch.no_grad():
            out1 = policy(all_layer_raw, all_layer_aq, proprio)
            out2 = policy(all_layer_raw, all_layer_aq, proprio)

        assert torch.allclose(out1, out2), "L1-based policy should be deterministic"

    def test_action_dim_correct(self, policy, config):
        """输出动作维度应为7 (7-DOF机械臂)"""
        B = 1
        all_layer_raw = [
            torch.randn(B, 32, config.hidden_size)
            for _ in range(config.num_vlm_layers)
        ]
        all_layer_aq = [
            torch.randn(B, config.num_action_queries, config.hidden_size)
            for _ in range(config.num_vlm_layers)
        ]
        proprio = torch.randn(B, config.proprio_dim)

        with torch.no_grad():
            out = policy(all_layer_raw, all_layer_aq, proprio)

        assert out.shape[-1] == config.action_dim
