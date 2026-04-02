"""
测试 VLA-Adapter 完整模型

验证:
1. 端到端前向传播
2. L1损失计算
3. 端到端训练循环
4. 参数统计
5. 模拟推理流程
"""

import torch
import pytest

from vla_adapter.model import VLAAdapter
from vla_adapter.config import VLAAdapterConfig


@pytest.fixture
def config():
    """使用较小配置以加速测试"""
    return VLAAdapterConfig(
        hidden_size=128,
        num_vlm_layers=4,
        num_attention_heads=4,
        num_action_queries=16,
        action_chunk_size=8,
        action_dim=7,
        proprio_dim=7,
        image_size=56,    # 缩小图像
        image_channels=3,
    )


@pytest.fixture
def model(config):
    return VLAAdapter(config)


def _create_dummy_batch(config, batch_size=2):
    """创建模拟输入数据"""
    return {
        "images_third": torch.randn(batch_size, config.image_channels, config.image_size, config.image_size),
        "images_gripper": torch.randn(batch_size, config.image_channels, config.image_size, config.image_size),
        "lang_tokens": torch.randint(0, 100, (batch_size, 16)),
        "proprio_state": torch.randn(batch_size, config.proprio_dim),
        "target_actions": torch.randn(batch_size, config.action_chunk_size, config.action_dim),
    }


class TestVLAAdapterModel:
    """完整模型测试"""

    def test_forward_shape(self, model, config):
        """端到端前向传播输出形状"""
        batch = _create_dummy_batch(config)
        with torch.no_grad():
            action_chunk = model(
                batch["images_third"],
                batch["images_gripper"],
                batch["lang_tokens"],
                batch["proprio_state"],
            )
        assert action_chunk.shape == (2, config.action_chunk_size, config.action_dim)

    def test_l1_loss(self, model, config):
        """L1损失计算 (论文 Eq.2)"""
        batch = _create_dummy_batch(config)
        pred = model(
            batch["images_third"],
            batch["images_gripper"],
            batch["lang_tokens"],
            batch["proprio_state"],
        )
        loss = model.compute_loss(pred, batch["target_actions"])

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert loss.requires_grad, "Loss should be differentiable"

    def test_training_step(self, model, config):
        """模拟一个完整训练步骤"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch = _create_dummy_batch(config)

        # Forward
        pred = model(
            batch["images_third"],
            batch["images_gripper"],
            batch["lang_tokens"],
            batch["proprio_state"],
        )
        loss = model.compute_loss(pred, batch["target_actions"])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证损失变化
        pred2 = model(
            batch["images_third"],
            batch["images_gripper"],
            batch["lang_tokens"],
            batch["proprio_state"],
        )
        loss2 = model.compute_loss(pred2, batch["target_actions"])

        # 一步训练后损失不应完全相同 (参数已更新)
        assert loss2.item() != loss.item(), "Loss should change after one training step"

    def test_params_info(self, model):
        """参数统计"""
        info = model.get_trainable_params_info()
        assert info["total_params"] > 0
        assert info["policy_params"] > 0
        assert info["action_query_params"] > 0
        assert info["trainable_params"] == info["total_params"]  # 所有参数默认可训练

    def test_inference_deterministic(self, model, config):
        """推理应确定性 (L1-based, 非diffusion)"""
        batch = _create_dummy_batch(config, batch_size=1)
        model.eval()

        with torch.no_grad():
            out1 = model(
                batch["images_third"],
                batch["images_gripper"],
                batch["lang_tokens"],
                batch["proprio_state"],
            )
            out2 = model(
                batch["images_third"],
                batch["images_gripper"],
                batch["lang_tokens"],
                batch["proprio_state"],
            )

        assert torch.allclose(out1, out2), "Inference should be deterministic"


class TestTrainingLoop:
    """模拟完整训练流程测试"""

    def test_multi_step_training(self, model, config):
        """模拟多步训练, 验证损失下降趋势"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        # 固定数据 (模拟过拟合测试)
        torch.manual_seed(42)
        batch = _create_dummy_batch(config, batch_size=4)

        losses = []
        for step in range(20):
            pred = model(
                batch["images_third"],
                batch["images_gripper"],
                batch["lang_tokens"],
                batch["proprio_state"],
            )
            loss = model.compute_loss(pred, batch["target_actions"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # 验证损失有下降趋势
        assert losses[-1] < losses[0], \
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"

    def test_cosine_lr_schedule(self, model, config):
        """验证余弦退火学习率调度 (论文训练配置)"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        total_steps = 100
        warmup_steps = int(total_steps * config.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        lrs = []
        for step in range(total_steps):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Warmup阶段: LR从0递增
        assert lrs[0] < lrs[warmup_steps - 1], "LR should increase during warmup"
        # 余弦退火: 峰值后递减
        assert lrs[warmup_steps] > lrs[-1], "LR should decrease after warmup"
