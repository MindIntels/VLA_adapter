"""
测试 Qwen2.5-0.5B 真实 VLM Backbone

注意: 这些测试需要安装 transformers, timm, peft 并能下载模型。
在没有这些依赖时, 测试会自动跳过 (pytest.mark.skipif)。

运行方式:
    # 仅运行真实 backbone 测试:
    python -m pytest tests/test_vlm_backbone_qwen.py -v

    # 跳过需要下载模型的测试:
    python -m pytest tests/test_vlm_backbone_qwen.py -v -k "not requires_download"
"""

import torch
import pytest

from vla_adapter.config import VLAAdapterConfig

# 检测依赖
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from vla_adapter.vlm_backbone_qwen import (
        Qwen25VLMBackbone,
        DINOv2Encoder,
        SigLIPEncoder,
    )
    HAS_QWEN_BACKBONE = True
except ImportError:
    HAS_QWEN_BACKBONE = False


requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS,
    reason="需要安装 transformers: pip install transformers>=4.37.0",
)

requires_peft = pytest.mark.skipif(
    not HAS_PEFT,
    reason="需要安装 peft: pip install peft>=0.7.0",
)

requires_download = pytest.mark.skipif(
    not (HAS_TRANSFORMERS and HAS_QWEN_BACKBONE),
    reason="需要安装 transformers, timm 并能够下载模型",
)


class TestQwen25Config:
    """配置层面测试 (不需要下载模型)"""

    def test_config_backbone_type(self):
        """backbone_type 配置字段"""
        config = VLAAdapterConfig(backbone_type="qwen2.5")
        assert config.backbone_type == "qwen2.5"
        assert config.qwen_model_name == "Qwen/Qwen2.5-0.5B"

    def test_config_defaults_match_paper(self):
        """验证默认配置匹配论文 Table F2"""
        config = VLAAdapterConfig()
        assert config.hidden_size == 896
        assert config.num_vlm_layers == 24
        assert config.num_action_queries == 64
        assert config.action_chunk_size == 8
        assert config.action_dim == 7
        assert config.use_lora is True
        assert config.lora_r == 16

    def test_config_lora_options(self):
        """LoRA 配置"""
        config = VLAAdapterConfig(
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            freeze_vlm=False,
        )
        assert config.lora_r == 8
        assert config.lora_alpha == 16

    def test_config_freeze_option(self):
        """冻结VLM配置 (论文 Table 3 场景)"""
        config = VLAAdapterConfig(freeze_vlm=True)
        assert config.freeze_vlm is True


class TestModelCreation:
    """模型创建测试 (需要 transformers 但可能需要下载模型)"""

    def test_simulated_fallback(self):
        """当 backbone_type='simulated' 时, 使用模拟 backbone"""
        from vla_adapter.model import VLAAdapter
        config = VLAAdapterConfig(
            backbone_type="simulated",
            hidden_size=128,
            num_vlm_layers=4,
            num_attention_heads=4,
            image_size=56,
        )
        model = VLAAdapter(config)
        from vla_adapter.vlm_backbone import SimulatedVLMBackbone
        assert isinstance(model.vlm, SimulatedVLMBackbone)


@requires_download
class TestQwen25Backbone:
    """
    真实 Qwen2.5-0.5B Backbone 测试

    这些测试需要:
    1. pip install transformers timm peft
    2. 能够下载 Qwen/Qwen2.5-0.5B, facebook/dinov2-base, google/siglip-base-patch16-224
    """

    @pytest.fixture
    def config(self):
        return VLAAdapterConfig(
            backbone_type="qwen2.5",
            hidden_size=896,
            num_vlm_layers=24,
            num_action_queries=64,
            action_chunk_size=8,
            action_dim=7,
        )

    @pytest.fixture
    def backbone(self, config):
        return Qwen25VLMBackbone(
            config,
            qwen_model_name="Qwen/Qwen2.5-0.5B",
            use_lora=True,
            freeze_vlm=False,
        )

    def test_output_format(self, backbone, config):
        """验证输出格式与 SimulatedVLMBackbone 一致"""
        B = 1
        images_third = torch.randn(B, 3, 224, 224)
        images_gripper = torch.randn(B, 3, 224, 224)
        lang_tokens = torch.randint(0, 1000, (B, 16))

        with torch.no_grad():
            all_layer_raw, all_layer_aq = backbone(
                images_third, images_gripper, lang_tokens,
            )

        # 层数应等于 Qwen2.5-0.5B 的层数 (24)
        assert len(all_layer_raw) == config.num_vlm_layers
        assert len(all_layer_aq) == config.num_vlm_layers

        # ActionQuery 维度
        for aq in all_layer_aq:
            assert aq.shape == (B, config.num_action_queries, config.hidden_size)

        # Raw 特征
        for raw in all_layer_raw:
            assert raw.shape[0] == B
            assert raw.shape[2] == config.hidden_size

    def test_tokenize_instruction(self, backbone):
        """指令 tokenize (论文格式)"""
        token_ids = backbone.tokenize_instruction("pick up the red block")
        assert token_ids.ndim == 2
        assert token_ids.shape[0] == 1
        assert token_ids.dtype == torch.long

    def test_param_summary(self, backbone):
        """参数统计"""
        summary = backbone.get_param_summary()
        assert "dino_encoder" in summary
        assert "siglip_encoder" in summary
        assert "qwen_model" in summary
        assert "action_query" in summary

        # DINOv2/SigLIP 应被冻结 (trainable=投影层参数少量)
        dino_info = summary["dino_encoder"]
        assert dino_info["trainable"] < dino_info["total"]

    def test_frozen_vlm(self, config):
        """冻结VLM时, ActionQuery仍可训练 (论文 Table 3)"""
        backbone = Qwen25VLMBackbone(
            config,
            freeze_vlm=True,
            use_lora=False,
        )

        # Qwen2.5 参数应全部冻结
        qwen_trainable = sum(
            p.numel() for p in backbone.qwen_model.parameters() if p.requires_grad
        )
        assert qwen_trainable == 0

        # ActionQuery 应可训练
        aq_trainable = sum(
            p.numel()
            for p in backbone.action_query_module.parameters()
            if p.requires_grad
        )
        assert aq_trainable > 0

    def test_lora_trainable_params(self, backbone):
        """LoRA 模式下只有少量参数可训练"""
        summary = backbone.get_param_summary()
        qwen_info = summary["qwen_model"]
        # LoRA 模式: trainable 远小于 total
        assert qwen_info["trainable"] < qwen_info["total"]

    def test_end_to_end_with_policy(self, config):
        """端到端: Qwen2.5 backbone + Policy"""
        from vla_adapter.model import VLAAdapter
        model = VLAAdapter(config)

        B = 1
        images_third = torch.randn(B, 3, 224, 224)
        images_gripper = torch.randn(B, 3, 224, 224)
        lang_tokens = torch.randint(0, 1000, (B, 16))
        proprio = torch.randn(B, config.proprio_dim)

        with torch.no_grad():
            action_chunk = model(images_third, images_gripper, lang_tokens, proprio)

        assert action_chunk.shape == (B, config.action_chunk_size, config.action_dim)
        assert torch.isfinite(action_chunk).all()
