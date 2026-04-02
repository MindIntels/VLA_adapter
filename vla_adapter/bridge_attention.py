"""
Bridge Attention 模块

这是 VLA-Adapter 的核心创新组件。
每层 Bridge Attention 包含:
  1. CA1: 动作隐向量对Raw特征的交叉注意力 (注入程度由可学习gate g控制)
  2. CA2: 动作隐向量对[ActionQuery特征, 本体感受嵌入]的交叉注意力
  3. SA:  动作隐向量的自注意力

公式 (论文 Eq.1):
  Â_t^τ = [CA1(Ã_t^τ, σ1(C_R)) · tanh(g),
            CA2(Ã_t^τ, σ2[C_AQ, σ0(P_t)]),
            SA(Ã_t^τ, Ã_t^τ)]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VLAAdapterConfig


class BridgeAttentionLayer(nn.Module):
    """
    单层 Bridge Attention (对应论文 Figure 5 中的一层Policy)

    关键设计:
    - 共享Q投影 (VLA-Adapter基础版; Pro版分离投影)
    - 可学习gating参数 g, 初始化为0, tanh(g)∈[-1,1] 控制C_R的注入程度
    - C_AQ固定注入比例为1 (论文消融实验 Table 8 验证)
    """

    def __init__(self, config: VLAAdapterConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # --- 共享 Q 投影 (从动作隐向量) ---
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # --- CA1: 对Raw特征的交叉注意力 ---
        # σ1: MLP投影 Raw特征 -> K1, V1
        self.sigma1 = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.k_raw = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_raw = nn.Linear(config.hidden_size, config.hidden_size)

        # --- CA2: 对[ActionQuery, Proprio]的交叉注意力 ---
        # σ2: MLP投影 [C_AQ, σ0(P_t)] -> K2, V2
        self.sigma2 = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.k_aq = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_aq = nn.Linear(config.hidden_size, config.hidden_size)

        # --- SA: 自注意力 ---
        self.k_self = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_self = nn.Linear(config.hidden_size, config.hidden_size)

        # --- 输出投影 (从concat的3份注意力 -> hidden_size) ---
        self.o_proj = nn.Linear(config.hidden_size * 3, config.hidden_size)

        # --- 可学习Gate参数 (论文核心: 初始化为0, tanh控制) ---
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # --- LayerNorm ---
        self.norm_action = nn.LayerNorm(config.hidden_size)

        # --- FFN (残差连接后的前馈网络) ---
        self.ffn = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, H) -> (B, num_heads, L, head_dim)"""
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """标准缩放点积注意力"""
        # q, k, v: (B, num_heads, L_q/L_kv, head_dim)
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

    def forward(
        self,
        action_latent: torch.Tensor,   # Ã_t^τ: (B, H_steps, hidden_size)
        c_raw: torch.Tensor,           # C_R^τ: (B, seq_vl, hidden_size)
        c_aq: torch.Tensor,            # C_AQ^τ: (B, num_aq, hidden_size)
        proprio_embed: torch.Tensor,   # σ0(P_t): (B, 1, hidden_size)
    ) -> torch.Tensor:
        """
        Bridge Attention 前向传播

        Returns:
            action_latent_next: Ã_t^{τ+1}, shape (B, H_steps, hidden_size)
        """
        B = action_latent.shape[0]

        # LayerNorm on action latent
        action_normed = self.norm_action(action_latent)

        # --- 共享 Q ---
        q = self.q_proj(action_normed)  # (B, H_steps, hidden_size)
        q = self._reshape_for_heads(q)  # (B, heads, H_steps, head_dim)

        # ============== CA1: Cross-Attention with Raw Features ==============
        raw_proj = self.sigma1(c_raw)                       # (B, seq_vl, hidden_size)
        k1 = self._reshape_for_heads(self.k_raw(raw_proj))  # (B, heads, seq_vl, head_dim)
        v1 = self._reshape_for_heads(self.v_raw(raw_proj))

        ca1 = self._attention(q, k1, v1)  # (B, heads, H_steps, head_dim)
        ca1 = ca1.transpose(1, 2).contiguous().view(B, -1, self.hidden_size)

        # 关键: tanh(g) 门控 Raw 特征注入
        ratio = torch.tanh(self.gating_factor)
        ca1 = ca1 * ratio

        # ============== CA2: Cross-Attention with [ActionQuery, Proprio] ==============
        # 拼接 ActionQuery特征和本体感受嵌入
        aq_with_proprio = torch.cat([c_aq, proprio_embed], dim=1)  # (B, num_aq+1, hidden_size)
        aq_proj = self.sigma2(aq_with_proprio)
        k2 = self._reshape_for_heads(self.k_aq(aq_proj))
        v2 = self._reshape_for_heads(self.v_aq(aq_proj))

        ca2 = self._attention(q, k2, v2)
        ca2 = ca2.transpose(1, 2).contiguous().view(B, -1, self.hidden_size)

        # ============== SA: Self-Attention ==============
        k_s = self._reshape_for_heads(self.k_self(action_normed))
        v_s = self._reshape_for_heads(self.v_self(action_normed))

        sa = self._attention(q, k_s, v_s)
        sa = sa.transpose(1, 2).contiguous().view(B, -1, self.hidden_size)

        # ============== 拼接并投影 (论文 Eq.1) ==============
        # Â_t^τ = [CA1 · tanh(g), CA2, SA]
        combined = torch.cat([ca1, ca2, sa], dim=-1)  # (B, H_steps, hidden_size*3)
        combined = self.o_proj(combined)               # (B, H_steps, hidden_size)

        # ============== 残差 + FFN ==============
        action_latent = action_latent + combined
        action_latent = action_latent + self.ffn(action_latent)

        return action_latent
