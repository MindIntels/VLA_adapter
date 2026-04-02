"""
VLA-Adapter 完整模型

将 VLM Backbone + ActionQuery + Policy Network 组装为端到端模型。

Pipeline:
  1. VLM接收 [图像, 指令, ActionQuery] → 输出每层 C_R, C_AQ
  2. Policy Network 以 C_R, C_AQ 和本体感受为条件 → 生成动作块
  3. L1损失监督训练

论文训练配置:
  - 端到端训练: VLM (LoRA微调) + ActionQuery (从零训练) + Policy (从零训练)
  - 总可训练参数: 197.2M
  - 训练时长: ~8小时 (单张消费级GPU)
"""

import torch
import torch.nn as nn

from .config import VLAAdapterConfig
from .vlm_backbone import SimulatedVLMBackbone
from .policy import PolicyNetwork

try:
    from .vlm_backbone_qwen import Qwen25VLMBackbone
    HAS_QWEN_BACKBONE = True
except ImportError:
    HAS_QWEN_BACKBONE = False


class VLAAdapter(nn.Module):
    """
    VLA-Adapter: 完整的 Vision-Language-Action 模型

    核心创新:
    1. ActionQuery: 在VLM中插入可学习token, 聚合多模态信息
    2. Bridge Attention: 通过门控机制将Raw和ActionQuery条件注入动作空间
    3. 轻量化: 0.5B backbone + 97M Policy = SOTA性能
    """

    def __init__(self, config: VLAAdapterConfig | None = None):
        super().__init__()
        self.config = config or VLAAdapterConfig()

        # VLM Backbone (含ActionQuery)
        if self.config.backbone_type == "qwen2.5":
            if not HAS_QWEN_BACKBONE:
                raise ImportError(
                    "Qwen2.5 backbone requires: pip install transformers>=4.37.0 timm>=0.9.0"
                )
            self.vlm = Qwen25VLMBackbone(
                self.config,
                qwen_model_name=self.config.qwen_model_name,
                use_lora=self.config.use_lora,
                lora_r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                freeze_vlm=self.config.freeze_vlm,
            )
        else:
            self.vlm = SimulatedVLMBackbone(self.config)

        # Policy Network (L1-based, with Bridge Attention)
        self.policy = PolicyNetwork(self.config)

    def forward(
        self,
        images_third: torch.Tensor,    # (B, C, H, W)
        images_gripper: torch.Tensor,   # (B, C, H, W)
        lang_tokens: torch.Tensor,      # (B, seq_len)
        proprio_state: torch.Tensor,    # (B, proprio_dim)
    ) -> torch.Tensor:
        """
        完整前向传播。

        Returns:
            action_chunk: (B, action_chunk_size, action_dim)
        """
        # Step 1: VLM前向, 获取所有层的条件
        all_layer_raw, all_layer_aq = self.vlm(images_third, images_gripper, lang_tokens)

        # Step 2: Policy生成动作
        action_chunk = self.policy(all_layer_raw, all_layer_aq, proprio_state)

        return action_chunk

    def compute_loss(
        self,
        pred_actions: torch.Tensor,    # (B, H, action_dim)
        target_actions: torch.Tensor,  # (B, H, action_dim)
    ) -> torch.Tensor:
        """
        L1损失 (论文 Eq.2)

        J(θ) = E[||π_θ(...) - A_t||_1]
        """
        return nn.functional.l1_loss(pred_actions, target_actions)

    def get_trainable_params_info(self) -> dict:
        """统计可训练参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        vlm_params = sum(p.numel() for p in self.vlm.parameters())
        policy_params = sum(p.numel() for p in self.policy.parameters())
        aq_params = sum(p.numel() for p in self.vlm.action_query_module.parameters())

        return {
            "total_params": total,
            "trainable_params": trainable,
            "vlm_params": vlm_params,
            "policy_params": policy_params,
            "action_query_params": aq_params,
        }
