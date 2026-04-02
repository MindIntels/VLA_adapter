"""
ActionQuery 模块

ActionQuery 是 VLA-Adapter 的核心创新之一。
与传统VLA使用VLM最后一层Raw特征不同，VLA-Adapter在VLM输入序列中插入
可学习的ActionQuery token。这些token在VLM各层的attention中聚合多模态信息，
并作为条件传递给Policy网络。
"""

import torch
import torch.nn as nn

from .config import VLAAdapterConfig


class ActionQueryModule(nn.Module):
    """
    ActionQuery嵌入模块。

    论文核心设计:
    - 64个可学习token (初始化为可训练参数)
    - 插入VLM的输入序列，参与所有层的attention
    - 每层的ActionQuery输出 C_AQ^τ 作为对应层Policy的条件
    - 即使VLM冻结，ActionQuery仍可从零训练
    """

    def __init__(self, config: VLAAdapterConfig):
        super().__init__()
        self.config = config

        # 可学习的ActionQuery嵌入 (论文: self.action_queries.weight)
        # shape: (num_action_queries, hidden_size)
        self.action_queries = nn.Embedding(
            config.num_action_queries,
            config.hidden_size,
        )

        # 初始化: 使用小的随机值
        nn.init.normal_(self.action_queries.weight, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        获取ActionQuery嵌入并扩展到batch维度。

        Args:
            batch_size: 批大小

        Returns:
            action_queries: (batch_size, num_action_queries, hidden_size)
        """
        # (num_queries, hidden_size) -> (1, num_queries, hidden_size) -> (B, num_queries, hidden_size)
        aq = self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return aq
