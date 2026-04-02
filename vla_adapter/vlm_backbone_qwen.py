"""
真实 VLM Backbone: Prismatic VLM (DINOv2 + SigLIP + Qwen2.5-0.5B)

使用 HuggingFace transformers 加载真实的 Qwen2.5-0.5B 模型，
配合 DINOv2 和 SigLIP 视觉编码器，实现论文中描述的完整 VLM 架构。

Qwen2.5-0.5B 参数:
- 24 层 Transformer
- hidden_size = 896
- num_attention_heads = 14 (GQA, num_key_value_heads=2)
- intermediate_size = 4864
- vocab_size = 151936

使用前须安装:
    pip install transformers>=4.37.0 timm>=0.9.0 peft>=0.7.0

模型会自动从 HuggingFace Hub 下载:
    - Qwen/Qwen2.5-0.5B
    - facebook/dinov2-base (ViT-B/14, 768-d)
    - google/siglip-base-patch16-224 (768-d)
"""

import torch
import torch.nn as nn

from .config import VLAAdapterConfig
from .action_query import ActionQueryModule

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoModel,
        SiglipVisionModel,
        AutoImageProcessor,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class DINOv2Encoder(nn.Module):
    """
    DINOv2 视觉编码器 (facebook/dinov2-base)

    论文中与 SigLIP 组成双视觉编码器 (Prismatic VLM 架构)。
    - DINOv2 提供强语义和空间特征 (自监督预训练)
    - 输出: (B, num_patches, 768) → 投影到 hidden_size
    """

    def __init__(self, config: VLAAdapterConfig, model_name: str = "facebook/dinov2-base"):
        super().__init__()
        self.vision_model = AutoModel.from_pretrained(model_name)
        self.vision_model.eval()
        # DINOv2-base: 768-d
        self.proj = nn.Linear(768, config.hidden_size)

        # 冻结视觉编码器
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224)
        Returns:
            features: (B, num_patches, hidden_size)  num_patches=256 for 224/14
        """
        with torch.no_grad():
            outputs = self.vision_model(pixel_values)
            # 使用 patch token (去掉 CLS token)
            patch_features = outputs.last_hidden_state[:, 1:, :]  # (B, 256, 768)
        return self.proj(patch_features)


class SigLIPEncoder(nn.Module):
    """
    SigLIP 视觉编码器 (google/siglip-base-patch16-224)

    与 DINOv2 互补:
    - SigLIP 提供对齐到文本空间的视觉语义 (对比学习预训练)
    - 输出: (B, num_patches, 768) → 投影到 hidden_size
    """

    def __init__(self, config: VLAAdapterConfig, model_name: str = "google/siglip-base-patch16-224"):
        super().__init__()
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        self.vision_model.eval()
        # SigLIP-base: 768-d
        self.proj = nn.Linear(768, config.hidden_size)

        # 冻结视觉编码器
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224)
        Returns:
            features: (B, num_patches, hidden_size)  num_patches=196 for 224/16
        """
        with torch.no_grad():
            outputs = self.vision_model(pixel_values)
            patch_features = outputs.last_hidden_state  # (B, 196, 768)
        return self.proj(patch_features)


class Qwen25VLMBackbone(nn.Module):
    """
    基于 Qwen2.5-0.5B 的真实 VLM Backbone。

    架构 (Prismatic VLM 风格):
        图像 ──→ DINOv2  ──→ visual_tokens_dino (B, 256, 896)  ─┐
                                                                  ├──→ concat ──→ visual_tokens
        图像 ──→ SigLIP  ──→ visual_tokens_siglip (B, 196, 896) ─┘
        指令 ──→ Qwen2.5 Tokenizer ──→ Qwen2.5 Embedding ──→ lang_tokens
        ActionQuery ──→ action_query_tokens

        完整序列 = [visual_tokens, lang_tokens, action_query_tokens]
        ──→ Qwen2.5-0.5B 24层 Transformer (LoRA微调)
        ──→ 每层输出 C_R (VL部分) 和 C_AQ (ActionQuery部分)

    参数配置:
        - Qwen2.5-0.5B: 24 layers, hidden=896, heads=14
        - DINOv2-base:   ViT-B/14, 768-d → 投影到 896
        - SigLIP-base:   ViT-B/16, 768-d → 投影到 896
        - 总视觉 token: 256 (dino) + 196 (siglip) = 452 per image, ×2 images = 904
        - ActionQuery: 64 tokens

    输出格式 (与 SimulatedVLMBackbone 接口一致):
        - all_layer_raw: List[Tensor], len=24, 每个 (B, seq_vl, 896)
        - all_layer_aq:  List[Tensor], len=24, 每个 (B, 64, 896)
    """

    # 支持的 Qwen2.5 模型
    SUPPORTED_MODELS = {
        "Qwen/Qwen2.5-0.5B": {"layers": 24, "hidden": 896, "heads": 14},
        "Qwen/Qwen2.5-1.5B": {"layers": 28, "hidden": 1536, "heads": 12},
        "Qwen/Qwen2.5-3B":   {"layers": 36, "hidden": 2048, "heads": 16},
    }

    def __init__(
        self,
        config: VLAAdapterConfig,
        qwen_model_name: str = "Qwen/Qwen2.5-0.5B",
        dino_model_name: str = "facebook/dinov2-base",
        siglip_model_name: str = "google/siglip-base-patch16-224",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        freeze_vlm: bool = False,
    ):
        """
        Args:
            config: VLA-Adapter 配置
            qwen_model_name: Qwen2.5 模型名称
            dino_model_name: DINOv2 模型名称
            siglip_model_name: SigLIP 模型名称
            use_lora: 是否使用 LoRA 微调 Qwen2.5 (论文默认)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            freeze_vlm: 是否完全冻结 VLM (论文表明冻结时仍有86.4%成功率)
        """
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "需要安装 transformers: pip install transformers>=4.37.0 timm>=0.9.0"
            )

        self.config = config
        self.qwen_model_name = qwen_model_name
        self.freeze_vlm = freeze_vlm

        # ==================== 1. 视觉编码器 ====================
        self.dino_encoder = DINOv2Encoder(config, dino_model_name)
        self.siglip_encoder = SigLIPEncoder(config, siglip_model_name)

        # ==================== 2. Qwen2.5-0.5B ====================
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.float32,
            output_hidden_states=True,   # 关键: 输出所有中间层隐状态
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
        )

        # 验证层数与配置匹配
        actual_layers = self.qwen_model.config.num_hidden_layers
        if actual_layers != config.num_vlm_layers:
            print(
                f"[WARNING] Qwen2.5 has {actual_layers} layers, "
                f"but config.num_vlm_layers={config.num_vlm_layers}. "
                f"Overriding config to {actual_layers}."
            )
            config.num_vlm_layers = actual_layers

        # 验证 hidden_size
        actual_hidden = self.qwen_model.config.hidden_size
        if actual_hidden != config.hidden_size:
            print(
                f"[WARNING] Qwen2.5 hidden_size={actual_hidden}, "
                f"but config.hidden_size={config.hidden_size}. "
                f"Overriding config to {actual_hidden}."
            )
            config.hidden_size = actual_hidden
            # 重新初始化视觉投影层以匹配维度
            self.dino_encoder.proj = nn.Linear(768, actual_hidden)
            self.siglip_encoder.proj = nn.Linear(768, actual_hidden)

        # ==================== 3. LoRA 或冻结 ====================
        if freeze_vlm:
            for param in self.qwen_model.parameters():
                param.requires_grad = False
        elif use_lora and HAS_PEFT:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],  # 论文使用LoRA微调
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.qwen_model = get_peft_model(self.qwen_model, lora_config)
            print(f"[INFO] LoRA applied. Trainable params:")
            self.qwen_model.print_trainable_parameters()
        elif use_lora and not HAS_PEFT:
            print("[WARNING] peft not installed, skipping LoRA. Install: pip install peft>=0.7.0")

        # ==================== 4. ActionQuery ====================
        self.action_query_module = ActionQueryModule(config)

        # ==================== 5. 图像预处理器 (可选) ====================
        self._dino_processor = None
        self._siglip_processor = None

    @property
    def dino_processor(self):
        """懒加载 DINOv2 图像预处理器"""
        if self._dino_processor is None:
            self._dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        return self._dino_processor

    @property
    def siglip_processor(self):
        """懒加载 SigLIP 图像预处理器"""
        if self._siglip_processor is None:
            self._siglip_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        return self._siglip_processor

    def _get_qwen_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取 Qwen2.5 的 token embedding (不经过 Transformer 层)"""
        return self.qwen_model.get_input_embeddings()(input_ids)

    def _encode_images(
        self,
        images_third: torch.Tensor,
        images_gripper: torch.Tensor,
    ) -> torch.Tensor:
        """
        双视觉编码: DINOv2 + SigLIP

        Args:
            images_third: (B, 3, 224, 224) 第三视角RGB图像
            images_gripper: (B, 3, 224, 224) 抓手RGB图像

        Returns:
            visual_tokens: (B, num_visual_tokens, hidden_size)
                第三视角: dino(256) + siglip(196) = 452 tokens
                抓手:     dino(256) + siglip(196) = 452 tokens
                总计:     904 tokens
        """
        # DINOv2 编码
        dino_third = self.dino_encoder(images_third)       # (B, 256, H)
        dino_gripper = self.dino_encoder(images_gripper)   # (B, 256, H)

        # SigLIP 编码
        siglip_third = self.siglip_encoder(images_third)     # (B, 196, H)
        siglip_gripper = self.siglip_encoder(images_gripper) # (B, 196, H)

        # 拼接: [dino_third, siglip_third, dino_gripper, siglip_gripper]
        visual_tokens = torch.cat(
            [dino_third, siglip_third, dino_gripper, siglip_gripper],
            dim=1,
        )
        return visual_tokens

    def _encode_language(self, lang_tokens: torch.Tensor) -> torch.Tensor:
        """
        将 token ids 映射为 Qwen2.5 embedding

        Args:
            lang_tokens: (B, seq_len) 已tokenize的指令

        Returns:
            lang_embeddings: (B, seq_len, hidden_size)
        """
        return self._get_qwen_embeddings(lang_tokens)

    def tokenize_instruction(self, instruction: str) -> torch.Tensor:
        """
        将自然语言指令转为token ids

        论文格式:
            "In: What action should the robot take to {instruction}?\nOut:"

        Args:
            instruction: 如 "pick up the red block"

        Returns:
            token_ids: (1, seq_len) Long tensor
        """
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        return encoded["input_ids"]

    def _run_qwen_with_custom_embeds(
        self,
        inputs_embeds: torch.Tensor,
        num_vl: int,
    ) -> tuple:
        """
        用自定义 embedding 输入运行 Qwen2.5, 提取每层的 C_R 和 C_AQ

        核心逻辑:
        - 将 [visual_tokens, lang_tokens, action_query_tokens] 作为一个序列
          直接送入 Qwen2.5 的 Transformer 层 (绕过 token embedding 层)
        - 从每层的 hidden_states 中分离出 VL 部分 (C_R) 和 AQ 部分 (C_AQ)

        Args:
            inputs_embeds: (B, total_seq, hidden_size)
            num_vl: VL token 的数量 (visual + language)

        Returns:
            all_layer_raw: List[Tensor], len=M
            all_layer_aq:  List[Tensor], len=M
        """
        outputs = self.qwen_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states: tuple of (M+1) tensors, 第0个是embedding输出
        # 我们取第1层到第M层 (共M个)
        hidden_states = outputs.hidden_states  # len = num_layers + 1

        all_layer_raw = []
        all_layer_aq = []

        for layer_idx in range(1, len(hidden_states)):  # 跳过 embedding 层
            h = hidden_states[layer_idx]  # (B, total_seq, hidden_size)
            c_raw = h[:, :num_vl, :]      # VL 部分
            c_aq = h[:, num_vl:, :]       # ActionQuery 部分
            all_layer_raw.append(c_raw)
            all_layer_aq.append(c_aq)

        return all_layer_raw, all_layer_aq

    def forward(
        self,
        images_third: torch.Tensor,    # (B, 3, 224, 224)
        images_gripper: torch.Tensor,   # (B, 3, 224, 224)
        lang_tokens: torch.Tensor,      # (B, seq_len) token ids
    ) -> tuple:
        """
        前向传播, 接口与 SimulatedVLMBackbone 完全一致。

        Returns:
            all_layer_raw: List[Tensor], len=M, 每个 (B, seq_vl, hidden_size)
            all_layer_aq:  List[Tensor], len=M, 每个 (B, num_aq, hidden_size)
        """
        B = images_third.shape[0]

        # Step 1: 双视觉编码
        visual_tokens = self._encode_images(images_third, images_gripper)

        # Step 2: 语言编码 (通过 Qwen2.5 embedding 层)
        lang_embeddings = self._encode_language(lang_tokens)

        # Step 3: ActionQuery
        action_queries = self.action_query_module(B)

        # Step 4: 拼接完整序列
        num_vl = visual_tokens.shape[1] + lang_embeddings.shape[1]
        inputs_embeds = torch.cat(
            [visual_tokens, lang_embeddings, action_queries],
            dim=1,
        )

        # Step 5: 通过 Qwen2.5 Transformer → 提取每层 C_R, C_AQ
        all_layer_raw, all_layer_aq = self._run_qwen_with_custom_embeds(
            inputs_embeds, num_vl,
        )

        return all_layer_raw, all_layer_aq

    def get_param_summary(self) -> dict:
        """参数统计"""
        def _count(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return {"total": total, "trainable": trainable}

        return {
            "dino_encoder": _count(self.dino_encoder),
            "siglip_encoder": _count(self.siglip_encoder),
            "qwen_model": _count(self.qwen_model),
            "action_query": _count(self.action_query_module),
        }
