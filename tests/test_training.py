"""
测试训练完整流程 & 模拟Push-T/LIBERO场景

包含:
1. 模拟机器人数据集
2. 完整training loop
3. 推理pipeline演示
"""

import torch
import pytest
from torch.utils.data import Dataset, DataLoader

from vla_adapter.model import VLAAdapter
from vla_adapter.config import VLAAdapterConfig


class SimulatedRobotDataset(Dataset):
    """
    模拟机器人操作数据集

    对应论文中的LIBERO/CALVIN数据:
    - 第三视角图像 (224x224x3 RGB)
    - 抓手图像 (224x224x3 RGB / 84x84在CALVIN中)
    - 语言指令
    - 本体感受状态 (7-DOF关节角度)
    - 目标动作 (7-DOF, action chunk=8步)
    """

    def __init__(self, num_samples: int, config: VLAAdapterConfig):
        self.num_samples = num_samples
        self.config = config

        # 预生成固定数据 (确保可复现)
        torch.manual_seed(0)
        self.images_third = torch.randn(num_samples, config.image_channels, config.image_size, config.image_size)
        self.images_gripper = torch.randn(num_samples, config.image_channels, config.image_size, config.image_size)
        self.lang_tokens = torch.randint(0, 100, (num_samples, 16))
        self.proprio_states = torch.randn(num_samples, config.proprio_dim)
        # 目标动作: 模拟简单的周期性运动
        t = torch.linspace(0, 2 * 3.14159, config.action_chunk_size).unsqueeze(0).unsqueeze(-1)
        self.target_actions = torch.sin(t).expand(num_samples, -1, config.action_dim) * 0.1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "images_third": self.images_third[idx],
            "images_gripper": self.images_gripper[idx],
            "lang_tokens": self.lang_tokens[idx],
            "proprio_state": self.proprio_states[idx],
            "target_actions": self.target_actions[idx],
        }


@pytest.fixture
def config():
    return VLAAdapterConfig(
        hidden_size=64,
        num_vlm_layers=2,
        num_attention_heads=4,
        num_action_queries=8,
        action_chunk_size=8,
        action_dim=7,
        proprio_dim=7,
        image_size=28,
        image_channels=3,
    )


class TestTrainingPipeline:
    """完整训练管道测试"""

    def test_dataloader(self, config):
        """数据加载测试"""
        dataset = SimulatedRobotDataset(32, config)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        assert batch["images_third"].shape == (4, 3, config.image_size, config.image_size)
        assert batch["target_actions"].shape == (4, config.action_chunk_size, config.action_dim)

    def test_full_training_loop(self, config):
        """完整训练循环 -- 模拟论文Table F1的训练流程"""
        model = VLAAdapter(config)
        dataset = SimulatedRobotDataset(32, config)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        epoch_losses = []
        for epoch in range(3):
            total_loss = 0
            for batch in loader:
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

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            epoch_losses.append(avg_loss)

        # 3个epoch后应有明显损失下降
        assert epoch_losses[-1] < epoch_losses[0], \
            f"Loss should decrease over epochs: {epoch_losses}"

    def test_inference_pipeline(self, config):
        """推理管道测试 -- 模拟真实部署场景"""
        model = VLAAdapter(config)
        model.eval()

        # 模拟单帧输入 (实际推理是逐帧的)
        images_third = torch.randn(1, 3, config.image_size, config.image_size)
        images_gripper = torch.randn(1, 3, config.image_size, config.image_size)
        lang_tokens = torch.randint(0, 100, (1, 16))
        proprio = torch.randn(1, config.proprio_dim)

        with torch.no_grad():
            action_chunk = model(images_third, images_gripper, lang_tokens, proprio)

        # 验证输出 action chunk
        assert action_chunk.shape == (1, config.action_chunk_size, config.action_dim)
        # 动作值应在合理范围 (不应为NaN或Inf)
        assert torch.isfinite(action_chunk).all(), "Actions should be finite"

    def test_action_chunking_sequential(self, config):
        """
        模拟 Action Chunking 执行

        论文设计: 每次预测H步动作, 执行第一步, 然后用新观测重新预测。
        在LIBERO中H=8, 7-DOF动作 = [x, y, z, rx, ry, rz, gripper]
        """
        model = VLAAdapter(config)
        model.eval()

        # 模拟10步机器人控制
        executed_actions = []
        for step in range(10):
            images_third = torch.randn(1, 3, config.image_size, config.image_size)
            images_gripper = torch.randn(1, 3, config.image_size, config.image_size)
            lang_tokens = torch.randint(0, 100, (1, 16))
            proprio = torch.randn(1, config.proprio_dim)

            with torch.no_grad():
                action_chunk = model(images_third, images_gripper, lang_tokens, proprio)

            # 执行第一步动作
            first_action = action_chunk[0, 0, :]  # (action_dim,)
            executed_actions.append(first_action)

            assert first_action.shape == (config.action_dim,)

        assert len(executed_actions) == 10

    def test_gating_values_during_training(self, config):
        """
        监控训练过程中门控参数g的变化

        论文发现:
        - g初始化为0 (tanh(0)=0, Raw特征不注入)
        - 训练后g会学习到合适的值，自动调节C_R注入比例
        """
        model = VLAAdapter(config)
        dataset = SimulatedRobotDataset(16, config)
        loader = DataLoader(dataset, batch_size=8)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.train()

        # 记录初始gate值
        initial_gates = []
        for layer in model.policy.bridge_layers:
            initial_gates.append(layer.gating_factor.item())

        # 训练几步
        for _ in range(10):
            for batch in loader:
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

        # 记录训练后gate值
        trained_gates = []
        for layer in model.policy.bridge_layers:
            trained_gates.append(layer.gating_factor.item())

        # 验证: gate值在训练后应发生变化
        assert initial_gates != trained_gates, \
            "Gating factors should change during training"

        # 验证: tanh(g) 在 [-1, 1] 范围内
        for g in trained_gates:
            assert -1 <= torch.tanh(torch.tensor(g)).item() <= 1
