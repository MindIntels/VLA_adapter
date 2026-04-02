"""
Policy Network (L1-based)

论文中的Policy网络由M层(与VLM层数相同) Bridge Attention 组成。
每层接收对应层VLM的 C_R 和 C_AQ 作为条件。

Architecture (Figure 5):
    A_t^0 (全零初始化, H步) → LN+MLP → Ã_t^0
    for τ in 0..M-1:
        Ã_t^{τ+1} = BridgeAttention(Ã_t^τ, C_R^τ, C_AQ^τ, σ0(P_t))
    A_t^{M-1} = MLP(LN(Ã_t^{M-1}))  # 最终动作输出

总参数量: 97.3M (当backbone为Qwen2.5-0.5B时)
"""

import torch
import torch.nn as nn

from .config import VLAAdapterConfig
from .bridge_attention import BridgeAttentionLayer


class PolicyNetwork(nn.Module):
    """
    L1-based Policy Network with Bridge Attention

    特点:
    - 与VLM层数相同 (M=24层)
    - 每层使用Bridge Attention: 2个交叉注意力 + 1个自注意力
    - 输入为全零初始化的动作序列
    - 输出为H步动作块 (action chunk)
    """

    def __init__(self, config: VLAAdapterConfig):
        super().__init__()
        self.config = config

        # --- σ0: 本体感受编码器 (两层MLP) ---
        self.proprio_encoder = nn.Sequential(
            nn.Linear(config.proprio_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # --- 初始动作编码 (LN + MLP) ---
        self.action_init_proj = nn.Sequential(
            nn.LayerNorm(config.action_dim),
            nn.Linear(config.action_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # --- M层 Bridge Attention ---
        self.bridge_layers = nn.ModuleList([
            BridgeAttentionLayer(config)
            for _ in range(config.num_vlm_layers)
        ])

        # --- 输出头: LN + MLP → 动作维度 ---
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.action_dim),
        )

    def forward(
        self,
        all_layer_raw: list,       # List[Tensor], len=M, 每个(B, seq_vl, H)
        all_layer_aq: list,        # List[Tensor], len=M, 每个(B, num_aq, H)
        proprio_state: torch.Tensor,  # (B, proprio_dim)
    ) -> torch.Tensor:
        """
        Policy前向传播。

        Args:
            all_layer_raw: 每层VLM的Raw特征
            all_layer_aq: 每层VLM的ActionQuery特征
            proprio_state: 本体感受状态 (关节角度等)

        Returns:
            action_chunk: (B, action_chunk_size, action_dim) H步动作预测
        """
        B = proprio_state.shape[0]
        device = proprio_state.device

        # Step 1: 初始化全零动作序列 A_t^0
        # (B, H, action_dim)
        action_init = torch.zeros(B, self.config.action_chunk_size, self.config.action_dim, device=device)

        # Step 2: LN + MLP → Ã_t^0
        action_latent = self.action_init_proj(action_init)  # (B, H, hidden_size)

        # Step 3: 本体感受编码 σ0(P_t)
        proprio_embed = self.proprio_encoder(proprio_state)  # (B, hidden_size)
        proprio_embed = proprio_embed.unsqueeze(1)           # (B, 1, hidden_size)

        # Step 4: 逐层Bridge Attention
        for layer_idx, bridge_layer in enumerate(self.bridge_layers):
            action_latent = bridge_layer(
                action_latent=action_latent,
                c_raw=all_layer_raw[layer_idx],
                c_aq=all_layer_aq[layer_idx],
                proprio_embed=proprio_embed,
            )

        # Step 5: 输出头 → 动作
        action_chunk = self.output_head(action_latent)  # (B, H, action_dim)

        return action_chunk
