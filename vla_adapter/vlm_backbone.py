"""
模拟的VLM Backbone

在实际应用中，这里应替换为真实的VLM (如Prismatic VLM + Qwen2.5-0.5B)。
本模块提供一个模拟实现，用于演示VLA-Adapter的架构和数据流。

论文架构:
- 视觉编码器: DINOv2 + SigLIP (双视觉编码)
- 语言模型: Qwen2.5-0.5B (24层Transformer)
- 输入: [视觉token, 指令token, ActionQuery token]
- 输出: 每层的Raw特征 C_R 和 ActionQuery特征 C_AQ
"""

import torch
import torch.nn as nn

from .config import VLAAdapterConfig
from .action_query import ActionQueryModule


class SimulatedTransformerLayer(nn.Module):
    """模拟单层Transformer (简化版，用于演示数据流)"""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        return x


class SimulatedVLMBackbone(nn.Module):
    """
    模拟VLM骨干网络。

    功能:
    1. 接收图像(第三视角+抓手)和指令，生成token序列
    2. 将ActionQuery拼接到序列中
    3. 通过M层Transformer处理
    4. 从每层提取 C_R (Raw特征) 和 C_AQ (ActionQuery特征)

    实际VLM对应关系:
    - 视觉编码: DINOv2(patch=14) + SigLIP → 视觉token
    - 语言编码: tokenizer → 语言token
    - [视觉token, 语言token, ActionQuery] → 24层Transformer

    输出格式:
    - all_layer_raw: List[Tensor], len=M, 每个 (B, seq_vl, H)
    - all_layer_aq:  List[Tensor], len=M, 每个 (B, num_aq, H)
    """

    def __init__(self, config: VLAAdapterConfig):
        super().__init__()
        self.config = config

        # 模拟视觉编码: 将图像投影为token序列
        # 实际: DINOv2 + SigLIP dual encoder
        num_visual_tokens = (config.image_size // 14) ** 2  # ~256 tokens per image
        self.num_visual_tokens = num_visual_tokens * 2  # 两个视角 (第三视角 + 抓手)

        self.visual_proj = nn.Sequential(
            nn.Linear(config.image_channels * 14 * 14, config.hidden_size),
            nn.GELU(),
        )

        # 模拟语言编码
        self.max_lang_tokens = 32
        self.lang_embedding = nn.Embedding(32000, config.hidden_size)

        # ActionQuery模块
        self.action_query_module = ActionQueryModule(config)

        # M层Transformer
        self.layers = nn.ModuleList([
            SimulatedTransformerLayer(config.hidden_size, config.num_attention_heads)
            for _ in range(config.num_vlm_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size)

    def _encode_images(self, images_third: torch.Tensor, images_gripper: torch.Tensor) -> torch.Tensor:
        """
        编码视觉输入。
        实际VLM中使用DINOv2 + SigLIP双编码器。
        """
        B = images_third.shape[0]
        # 简化: 将图像reshape为 patch 序列
        # (B, C, H, W) -> (B, num_patches, patch_dim)
        patch_size = 14
        h_patches = images_third.shape[2] // patch_size
        w_patches = images_third.shape[3] // patch_size

        third_patches = images_third.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        third_patches = third_patches.contiguous().view(B, -1, self.config.image_channels * patch_size * patch_size)

        gripper_patches = images_gripper.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        gripper_patches = gripper_patches.contiguous().view(B, -1, self.config.image_channels * patch_size * patch_size)

        # 投影到hidden_size
        third_tokens = self.visual_proj(third_patches)
        gripper_tokens = self.visual_proj(gripper_patches)

        # 拼接两个视角
        visual_tokens = torch.cat([third_tokens, gripper_tokens], dim=1)
        return visual_tokens

    def _encode_language(self, lang_tokens: torch.Tensor) -> torch.Tensor:
        """编码语言指令"""
        return self.lang_embedding(lang_tokens)

    def forward(
        self,
        images_third: torch.Tensor,   # (B, C, H, W) 第三视角图像
        images_gripper: torch.Tensor,  # (B, C, H, W) 抓手图像
        lang_tokens: torch.Tensor,     # (B, seq_len) 语言token
    ) -> tuple:
        """
        前向传播, 返回所有层的Raw特征和ActionQuery特征。

        Returns:
            all_layer_raw: List[Tensor], 长度M, 每个(B, seq_vl, hidden_size)
            all_layer_aq:  List[Tensor], 长度M, 每个(B, num_aq, hidden_size)
        """
        B = images_third.shape[0]

        # Step 1: 编码视觉和语言
        visual_tokens = self._encode_images(images_third, images_gripper)
        lang_embeddings = self._encode_language(lang_tokens)

        # Step 2: 获取ActionQuery
        action_queries = self.action_query_module(B)

        # Step 3: 拼接完整序列: [visual, language, ActionQuery]
        # 这是VLA-Adapter的核心设计: ActionQuery作为可学习token嵌入VLM
        num_vl = visual_tokens.shape[1] + lang_embeddings.shape[1]
        full_sequence = torch.cat([visual_tokens, lang_embeddings, action_queries], dim=1)

        # Step 4: 逐层处理, 提取每层的C_R和C_AQ
        all_layer_raw = []
        all_layer_aq = []

        x = full_sequence
        for layer in self.layers:
            x = layer(x)

            # 分离Raw特征和ActionQuery特征
            c_raw = x[:, :num_vl, :]          # (B, seq_vl, H) — VL部分
            c_aq = x[:, num_vl:, :]           # (B, num_aq, H) — ActionQuery部分

            all_layer_raw.append(c_raw)
            all_layer_aq.append(c_aq)

        return all_layer_raw, all_layer_aq
