from dataclasses import dataclass


@dataclass
class VLAAdapterConfig:
    """VLA-Adapter 超参数配置 (对应论文 Table F2)"""

    # VLM Backbone
    hidden_size: int = 896          # VLM hidden dimension
    num_vlm_layers: int = 24        # VLM层数 M = 24 (Qwen2.5-0.5B)
    num_attention_heads: int = 8    # 注意力头数

    # ActionQuery
    num_action_queries: int = 64    # ActionQuery数量 (论文最优)

    # Policy Network
    action_chunk_size: int = 8      # H-step action chunk
    action_dim: int = 7             # 7-DOF动作维度 (机器人臂)
    proprio_dim: int = 7            # 本体感受维度

    # Training
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    max_training_steps: int = 150000
    batch_size: int = 16

    # Image
    image_size: int = 224
    image_channels: int = 3

    # Backbone selection: "simulated" or "qwen2.5"
    backbone_type: str = "simulated"
    # Qwen2.5 specific
    qwen_model_name: str = "Qwen/Qwen2.5-0.5B"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    freeze_vlm: bool = False
