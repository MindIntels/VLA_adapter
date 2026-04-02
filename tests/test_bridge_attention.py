"""
测试 Bridge Attention 模块

验证:
1. 输出形状正确
2. 门控参数初始化为0 (tanh(0)=0, 初始时Raw特征不注入)
3. 门控参数可学习
4. 梯度正常反传
"""

import torch
import pytest

from vla_adapter.bridge_attention import BridgeAttentionLayer
from vla_adapter.config import VLAAdapterConfig


@pytest.fixture
def config():
    return VLAAdapterConfig(
        hidden_size=128,
        num_vlm_layers=4,
        num_attention_heads=4,
        action_chunk_size=8,
        action_dim=7,
    )


@pytest.fixture
def bridge_layer(config):
    return BridgeAttentionLayer(config)


class TestBridgeAttention:
    """Bridge Attention 核心功能测试"""

    def test_output_shape(self, bridge_layer, config):
        """验证输出形状: (B, H_steps, hidden_size)"""
        B, H = 2, config.action_chunk_size
        action_latent = torch.randn(B, H, config.hidden_size)
        c_raw = torch.randn(B, 32, config.hidden_size)       # 32个VL token
        c_aq = torch.randn(B, config.num_action_queries, config.hidden_size)
        proprio_embed = torch.randn(B, 1, config.hidden_size)

        output = bridge_layer(action_latent, c_raw, c_aq, proprio_embed)

        assert output.shape == (B, H, config.hidden_size), \
            f"Expected ({B}, {H}, {config.hidden_size}), got {output.shape}"

    def test_gating_initialization(self, bridge_layer):
        """
        论文关键设计: 门控参数g初始化为0
        tanh(0) = 0, 意味着训练初期Raw特征不注入
        """
        assert bridge_layer.gating_factor.item() == 0.0
        assert torch.tanh(bridge_layer.gating_factor).item() == 0.0

    def test_gating_is_learnable(self, bridge_layer, config):
        """验证门控参数g在训练过程中可学习"""
        B, H = 2, config.action_chunk_size
        action_latent = torch.randn(B, H, config.hidden_size)
        c_raw = torch.randn(B, 32, config.hidden_size)
        c_aq = torch.randn(B, config.num_action_queries, config.hidden_size)
        proprio_embed = torch.randn(B, 1, config.hidden_size)

        output = bridge_layer(action_latent, c_raw, c_aq, proprio_embed)
        loss = output.sum()
        loss.backward()

        assert bridge_layer.gating_factor.grad is not None, "Gating factor should have gradients"

    def test_gradient_flow(self, bridge_layer, config):
        """验证梯度能正确反传到所有组件"""
        B, H = 2, config.action_chunk_size
        action_latent = torch.randn(B, H, config.hidden_size, requires_grad=True)
        c_raw = torch.randn(B, 32, config.hidden_size, requires_grad=True)
        c_aq = torch.randn(B, config.num_action_queries, config.hidden_size, requires_grad=True)
        proprio_embed = torch.randn(B, 1, config.hidden_size, requires_grad=True)

        output = bridge_layer(action_latent, c_raw, c_aq, proprio_embed)
        loss = output.sum()
        loss.backward()

        assert action_latent.grad is not None, "Action latent should have gradients"
        assert c_raw.grad is not None, "Raw features should have gradients"
        assert c_aq.grad is not None, "ActionQuery features should have gradients"
        assert proprio_embed.grad is not None, "Proprio embedding should have gradients"

    def test_gating_effect(self, config):
        """
        测试门控参数的效果:
        当g=0时 (初始), CA1的贡献为0 (Raw特征不注入)
        当g!=0时, CA1开始贡献

        这对应论文Table 8的消融实验
        """
        torch.manual_seed(42)
        layer = BridgeAttentionLayer(config)
        B, H = 2, config.action_chunk_size

        action_latent = torch.randn(B, H, config.hidden_size)
        c_raw = torch.randn(B, 32, config.hidden_size)
        c_aq = torch.randn(B, config.num_action_queries, config.hidden_size)
        proprio_embed = torch.randn(B, 1, config.hidden_size)

        # g=0时的输出
        with torch.no_grad():
            output_g0 = layer(action_latent, c_raw, c_aq, proprio_embed)

        # 手动设置g=1
        with torch.no_grad():
            layer.gating_factor.fill_(1.0)
            output_g1 = layer(action_latent, c_raw, c_aq, proprio_embed)

        # 两种情况输出应不同 (Raw特征的注入使输出改变)
        assert not torch.allclose(output_g0, output_g1, atol=1e-5), \
            "Outputs should differ when gating changes from 0 to 1"

    def test_batch_independence(self, bridge_layer, config):
        """验证批次内样本间互不影响"""
        B, H = 4, config.action_chunk_size
        action_latent = torch.randn(B, H, config.hidden_size)
        c_raw = torch.randn(B, 32, config.hidden_size)
        c_aq = torch.randn(B, config.num_action_queries, config.hidden_size)
        proprio_embed = torch.randn(B, 1, config.hidden_size)

        with torch.no_grad():
            full_output = bridge_layer(action_latent, c_raw, c_aq, proprio_embed)
            single_output = bridge_layer(
                action_latent[:1], c_raw[:1], c_aq[:1], proprio_embed[:1],
            )

        assert torch.allclose(full_output[0], single_output[0], atol=1e-5), \
            "Each sample in batch should be processed independently"
