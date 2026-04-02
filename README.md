# VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model

> **论文**: arXiv:2509.09372 (2025)  
> **作者**: Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui 等  
> **机构**: 北京邮电大学, 西湖大学, 浙江大学, OpenHelix Team  
> **项目主页**: https://vla-adapter.github.io/  
> **GitHub**: https://github.com/OpenHelix-Team/VLA-Adapter

---

## 1. 论文核心问题

当前的 Vision-Language-Action (VLA) 模型通常需要：
- **大规模VLM骨干**（7B+参数）
- **海量机器人数据预训练**（如Open X-Embodiment）
- **高昂的GPU资源**和长时间微调

**核心问题**: 如何更有效地将视觉-语言(VL)表征桥接到动作(A)空间？

---

## 2. 论文原理详解

### 2.1 整体架构

VLA-Adapter 由三个核心组件组成：

```
输入图像 + 指令 + ActionQuery
         │
         ▼
┌─────────────────────┐
│   VLM Backbone      │  (Qwen2.5-0.5B, 24层, 冻结/LoRA微调)
│   + ActionQuery     │  (64个可学习token, 嵌入VLM序列)
│   + DINOv2/SigLIP   │  (视觉编码器)
│                     │
│  每层输出两种条件:    │
│  · C_R  (Raw特征)    │
│  · C_AQ (ActionQuery │
│         特征)        │
└─────────┬───────────┘
          │  24层 × 2种条件
          ▼
┌─────────────────────┐
│  Policy Network     │  (24层Bridge Attention, 97M参数)
│  with Bridge        │
│  Attention          │
│                     │
│  输入: 初始化全零    │
│  动作 A_t^0 (H步)   │
│  + 本体感受 P_t     │
│                     │
│  输出: H步动作块    │
│  A_t^{M-1}          │
└─────────────────────┘
```

### 2.2 条件探索 —— 哪种条件最适合桥接VL到A？

论文系统性地探索了四种桥接条件:

| 条件类型 | 层级 | 说明 |
|---------|------|------|
| 单层 C_R | 某一层VLM的Raw特征 | 中间层表现优于深层 |
| 单层 C_AQ | 某一层的ActionQuery特征 | 深层表现优于浅层 |
| 全层 C_R | 每层VLM的Raw特征 → 对应层Policy | 整体优于单层 |
| 全层 C_AQ | 每层ActionQuery特征 → 对应层Policy | 整体最优 |

**三个关键发现 (Key Findings)**:

1. **Key Finding 1**: 中间层Raw特征优于深层。深层偏向语义信息，中间层更好地融合了图像和文本的多模态细节。
2. **Key Finding 2**: 深层ActionQuery特征优于浅层。因为ActionQuery从零训练，深层聚合了更丰富的多模态信息。
3. **Key Finding 3**: 多层特征优于单层。使用所有层的特征不仅提升性能，还省去了选择最佳层的设计成本。

**最终决策**: 同时使用全层C_R和全层C_AQ，因为C_AQ整体更优，但中间层C_R在某些困难任务上表现更好。

### 2.3 Bridge Attention（核心创新）

每层Bridge Attention由**两个交叉注意力 + 一个自注意力**组成：

```
                   ┌──────────────┐
                   │ 动作隐向量    │
                   │ Ã_t^τ        │
                   └──────┬───────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │  CA1      │  │  CA2      │  │  SA       │
    │           │  │           │  │           │
    │ Q: Ã_t^τ │  │ Q: Ã_t^τ │  │ Q: Ã_t^τ │
    │ K,V:      │  │ K,V:      │  │ K,V:      │
    │ σ₁(C_R)  │  │σ₂[C_AQ,  │  │ Ã_t^τ    │
    │           │  │  σ₀(P_t)]│  │           │
    └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
          │              │              │
          │×tanh(g)      │×1            │
          │              │              │
          └──────┬───────┘──────────────┘
                 │ Concat
                 ▼
          ┌──────────────┐
          │   Â_t^τ      │
          │ (拼接结果)    │
          └──────┬───────┘
                 │
                 ▼ 
          ┌──────────────┐
          │ residual FFN │
          └──────────────┘
                 │
                 ▼
          ┌──────────────┐
          │ Ã_t^{τ+1}   │
          └──────────────┘
```

**关键公式**:

$$\hat{A}_t^\tau = [\text{CA}_1(\tilde{A}_t^\tau, \sigma_1(\mathcal{C}_t^R)) \cdot \tanh(g), \; \text{CA}_2(\tilde{A}_t^\tau, \sigma_2[\mathcal{C}_t^{AQ}, \sigma_0(\mathcal{P}_t)]), \; \text{SA}(\tilde{A}_t^\tau, \tilde{A}_t^\tau)]$$

其中:
- $g$ 是可学习参数，初始化为0，$\tanh(g) \in [-1, 1]$ 控制Raw特征的注入程度
- $\sigma_0, \sigma_1, \sigma_2$ 是MLP投影层
- $\mathcal{P}_t$ 是本体感受状态（关节角度等）

### 2.4 训练目标

使用**L1损失**（Mean Absolute Error）：

$$\min_\theta \mathcal{J}(\theta) = \mathbb{E}_{A_t, \mathcal{C}_t^R, \mathcal{C}_t^{AQ}, \sigma_0(\mathcal{P}_t), \tau} \left[ \| \pi_\theta(A_t^\tau, \mathcal{C}_t^R, \mathcal{C}_t^{AQ}, \sigma_0(\mathcal{P}_t), \tau) - A_t \|_1 \right]$$

**训练配置**:
| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| 学习率调度 | Cosine Annealing + Warmup (10%) |
| Batch Size | 16 |
| 最大训练步数 | 150,000 |
| VLM微调策略 | LoRA |

---

## 3. VLA-Adapter的核心优势

### 3.1 极致轻量化
| 指标 | VLA-Adapter | OpenVLA-OFT | π₀ |
|------|------------|-------------|-----|
| 骨干参数量 | **0.5B** | 7B | 3B |
| Policy参数 | **97M** | - | - |
| 总可训练参数 | **197M** | ~7B | ~3B |

### 3.2 SOTA级性能
在LIBERO基准上:
| 方法 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| OpenVLA-OFT (7B) | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π₀ (3B) | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| SmolVLA (2.2B) | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| **VLA-Adapter (0.5B)** | **97.8** | **99.2** | **97.2** | **95.0** | **97.3** |
| **VLA-Adapter-Pro (0.5B)** | **99.6** | **99.6** | **98.2** | **96.4** | **98.5** |

### 3.3 高推理效率
| 方法 | 吞吐量 (Hz) | 延迟 (秒) |
|------|-------------|----------|
| OpenVLA | 4.2 | 0.2396 |
| OpenVLA-OFT (wo gripper) | 71.4 | 0.1120 |
| OpenVLA-OFT | 109.7 | 0.0729 |
| **VLA-Adapter** | **219.2** | **0.0365** |

### 3.4 低训练成本
- 单张消费级GPU上 **8小时** 即可训练完成
- **无需机器人数据预训练**
- 即使VLM骨干完全冻结，仍然表现强劲（Success Rate 86.4%）

### 3.5 强泛化能力
在CALVIN ABC→D零样本泛化任务中:
| 方法 | 1 | 2 | 3 | 4 | 5 | Avg. len |
|------|---|---|---|---|---|----------|
| OpenVLA-OFT (7B) | 96.3 | 89.1 | 82.4 | 75.8 | 66.5 | 4.10 |
| **VLA-Adapter (0.5B)** | **99.1** | 94.6 | 88.8 | 82.8 | 76.5 | **4.42** |
| **VLA-Adapter-Pro (0.5B)** | 98.5 | **95.0** | **90.5** | **85.3** | **80.0** | **4.50** |

---

## 4. 代码结构

```
VLA_adapter/
├── README.md                      # 本文件
├── requirements.txt               # 依赖
├── vla_adapter/
│   ├── __init__.py
│   ├── model.py                  # VLA-Adapter 完整模型
│   ├── bridge_attention.py       # Bridge Attention 模块
│   ├── policy.py                 # Policy Network
│   ├── action_query.py           # ActionQuery 模块
│   ├── vlm_backbone.py           # 模拟VLM骨干 (无需下载模型)
│   ├── vlm_backbone_qwen.py      # 真实VLM骨干 (Qwen2.5-0.5B + DINOv2 + SigLIP)
│   └── config.py                 # 配置
└── tests/
    ├── __init__.py
    ├── test_bridge_attention.py
    ├── test_policy.py
    ├── test_model.py
    ├── test_training.py
    └── test_vlm_backbone_qwen.py  # Qwen2.5 backbone 测试
```

## 5. 快速开始

### 5.1 安装依赖

```bash
cd VLA_adapter

# 基础依赖 (使用模拟backbone, 无需联网)
pip install -r requirements.txt

# 真实 Qwen2.5-0.5B backbone 额外依赖
pip install transformers>=4.37.0 timm>=0.9.0 peft>=0.7.0
```

### 5.2 使用模拟 Backbone (默认, 无需下载模型)

```python
from vla_adapter import VLAAdapter, VLAAdapterConfig

config = VLAAdapterConfig(backbone_type="simulated")
model = VLAAdapter(config)
```

### 5.3 使用真实 Qwen2.5-0.5B Backbone

```python
from vla_adapter import VLAAdapter, VLAAdapterConfig

# LoRA 微调模式 (论文默认配置)
config = VLAAdapterConfig(
    backbone_type="qwen2.5",
    use_lora=True,       # LoRA微调 q_proj, v_proj
    lora_r=16,           # LoRA rank
    lora_alpha=32,       # LoRA alpha
    freeze_vlm=False,    # VLM可训练 (通过LoRA)
)
model = VLAAdapter(config)

# 冻结VLM模式 (论文 Table 3 场景, 仍有86.4%成功率)
config = VLAAdapterConfig(
    backbone_type="qwen2.5",
    use_lora=False,
    freeze_vlm=True,     # 完全冻结VLM, 仅训练ActionQuery + Policy
)
model = VLAAdapter(config)
```

### 5.4 推理示例

```python
import torch

config = VLAAdapterConfig(backbone_type="simulated", hidden_size=128, num_vlm_layers=4,
                           num_attention_heads=4, image_size=56)
model = VLAAdapter(config)
model.eval()

# 模拟输入
images_third = torch.randn(1, 3, 56, 56)      # 第三视角图像
images_gripper = torch.randn(1, 3, 56, 56)     # 抓手图像
lang_tokens = torch.randint(0, 100, (1, 16))    # 语言指令 token
proprio = torch.randn(1, 7)                     # 本体感受 (7-DOF关节角度)

with torch.no_grad():
    action_chunk = model(images_third, images_gripper, lang_tokens, proprio)
    # action_chunk: (1, 8, 7) → 8步动作, 每步7维 (x,y,z,rx,ry,rz,gripper)
    first_action = action_chunk[0, 0, :]  # 执行第一步
    print(f"Action: {first_action}")
```

### 5.5 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 仅运行 Qwen2.5 backbone 测试 (需要 transformers + 模型下载)
python -m pytest tests/test_vlm_backbone_qwen.py -v
```

---

## 6. 参考文献

```bibtex
@article{wang2025vlaadapter,
  title={VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model},
  author={Wang, Yihao and Ding, Pengxiang and Li, Lingxiao and Cui, Can and Ge, Zirui and Tong, Xinyang and Song, Wenxuan and Zhao, Han and Zhao, Wei and Hou, Pengxu and Huang, Siteng and Tang, Yifan and Wang, Wenhui and Zhang, Ru and Liu, Jianyi and Wang, Donglin},
  journal={arXiv preprint arXiv:2509.09372},
  year={2025}
}
```
