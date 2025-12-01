"""
action_processor.py

处理动作的编码和解码，包括归一化、反归一化等数据预处理
"""
import torch
import numpy as np
from typing import Dict, Optional, List, Union
import json


class ActionProcessor:
    """
    动作处理器，负责：
    1. 训练时：连续动作 -> 归一化 -> 令牌 -> 添加到输入序列
    2. 推理时：生成的令牌 -> 连续动作 -> 反归一化
    """

    def __init__(
        self,
        action_tokenizer,
        statistics: Optional[Dict] = None,
        action_chunk_size: int = 1,
    ):
        """
        Args:
            action_tokenizer: SpatialActionTokenizer6D实例
            statistics: 数据集统计信息 {"action": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}}
            action_chunk_size: 动作序列长度
        """
        self.action_tokenizer = action_tokenizer
        self.statistics = statistics or {}
        self.action_chunk_size = action_chunk_size

    def set_statistics(self, statistics: Dict):
        """设置或更新统计信息"""
        self.statistics = statistics
        print(f"[ActionProcessor] Statistics updated for datasets: {list(statistics.keys())}")

    def normalize_actions(
        self,
        actions: np.ndarray,
        dataset_key: Optional[str] = None
    ) -> np.ndarray:
        """
        归一化动作到 [-1, 1] 范围

        Args:
            actions: (n, 6) 原始动作
            dataset_key: 数据集名称（用于获取对应的统计信息）

        Returns:
            归一化后的动作 (n, 6)
        """
        if dataset_key is None or dataset_key not in self.statistics:
            print(f"[Warning] No statistics found for dataset '{dataset_key}', skipping normalization")
            return actions

        action_stats = self.statistics[dataset_key]["action"]

        # 使用1%和99%分位数进行归一化（更鲁棒）
        action_low = np.array(action_stats["q01"])
        action_high = np.array(action_stats["q99"])

        # 归一化到[-1, 1]
        normalized = 2.0 * (actions - action_low) / (action_high - action_low) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized

    def unnormalize_actions(
        self,
        normalized_actions: np.ndarray,
        dataset_key: Optional[str] = None
    ) -> np.ndarray:
        """
        反归一化动作

        Args:
            normalized_actions: (n, 6) 归一化的动作 [-1, 1]
            dataset_key: 数据集名称

        Returns:
            原始尺度的动作 (n, 6)
        """
        if dataset_key is None or dataset_key not in self.statistics:
            print(f"[Warning] No statistics found for dataset '{dataset_key}', skipping unnormalization")
            return normalized_actions

        action_stats = self.statistics[dataset_key]["action"]
        action_low = np.array(action_stats["q01"])
        action_high = np.array(action_stats["q99"])

        # 从[-1, 1]反归一化
        actions = 0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low

        return actions

    def encode_actions_for_training(
        self,
        actions: Union[np.ndarray, torch.Tensor],
        dataset_key: Optional[str] = None
    ) -> List[str]:
        """
        训练时编码动作为令牌字符串

        Args:
            actions: (chunk_size, 6) 或 (6,) 原始动作
            dataset_key: 数据集名称

        Returns:
            动作令牌字符串列表
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)

        # 归一化
        normalized_actions = self.normalize_actions(actions, dataset_key)

        # 令牌化
        action_tokens = self.action_tokenizer(normalized_actions)  # (n, 2)

        # 转换为字符串列表
        token_strings = []
        for i in range(action_tokens.shape[0]):
            token_strings.extend(action_tokens[i].tolist())

        return token_strings

    def decode_actions_from_generation(
        self,
        generation_outputs: torch.Tensor,
        dataset_key: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        推理时解码生成的令牌为动作

        Args:
            generation_outputs: (1, seq_len) 生成的令牌ids
            dataset_key: 数据集名称

        Returns:
            {
                "actions": (chunk_size, 6) 反归一化后的动作,
                "normalized_actions": (chunk_size, 6) 归一化的动作
            }
        """
        action_token_num = 2  # 6D动作只需要2个token

        # 提取前 chunk_size * 2 个令牌
        total_tokens = action_token_num * self.action_chunk_size
        predicted_action_token_ids = generation_outputs[0, :total_tokens]

        # 重塑为 (chunk_size, 2)
        predicted_action_token_ids = predicted_action_token_ids.reshape(
            self.action_chunk_size, action_token_num
        ).cpu().numpy()

        # 解码为归一化的连续动作 [-1, 1]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(
            predicted_action_token_ids
        )  # (chunk_size, 6)

        # 反归一化
        actions = self.unnormalize_actions(normalized_actions, dataset_key)

        return {
            "actions": actions,
            "normalized_actions": normalized_actions
        }

    def prepare_training_inputs(
        self,
        tokenizer,
        text: str,
        actions: np.ndarray,
        dataset_key: Optional[str] = None,
        add_generation_prompt: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        准备训练时的输入

        Args:
            tokenizer: HuggingFace tokenizer
            text: 文本指令
            actions: (chunk_size, 6) 或 (6,) 动作
            dataset_key: 数据集名称
            add_generation_prompt: 是否添加生成提示

        Returns:
            {
                "input_ids": token ids,
                "attention_mask": attention mask,
                "labels": labels for loss computation
            }
        """
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)

        # 编码动作为令牌字符串
        action_token_strings = self.encode_actions_for_training(actions, dataset_key)
        action_text = "".join(action_token_strings)

        # 构建完整的输入文本
        if add_generation_prompt:
            # 类似于chat模板
            full_text = f"{text}\nAssistant:"
        else:
            full_text = text

        # Tokenize
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        # Tokenize动作部分
        action_inputs = tokenizer(
            action_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False  # 不添加BOS/EOS
        )

        # 合并
        input_ids = torch.cat([inputs["input_ids"], action_inputs["input_ids"]], dim=1)
        attention_mask = torch.cat([inputs["attention_mask"], action_inputs["attention_mask"]], dim=1)

        # 创建labels：只计算动作token的损失
        labels = input_ids.clone()
        labels[:, :-action_inputs["input_ids"].shape[1]] = -100  # 忽略非动作部分

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    @classmethod
    def from_json(cls, json_path: str, action_tokenizer):
        """从JSON文件加载统计信息"""
        with open(json_path, 'r') as f:
            statistics = json.load(f)
        return cls(action_tokenizer=action_tokenizer, statistics=statistics)

    def save_statistics(self, json_path: str):
        """保存统计信息到JSON文件"""
        with open(json_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        print(f"[ActionProcessor] Statistics saved to {json_path}")


def compute_action_statistics(
    actions: np.ndarray,
    dataset_name: str = "default"
) -> Dict:
    """
    计算动作的统计信息

    Args:
        actions: (n, 6) 动作数组
        dataset_name: 数据集名称

    Returns:
        统计信息字典
    """
    statistics = {
        dataset_name: {
            "action": {
                "mean": actions.mean(axis=0).tolist(),
                "std": actions.std(axis=0).tolist(),
                "min": actions.min(axis=0).tolist(),
                "max": actions.max(axis=0).tolist(),
                "q01": np.percentile(actions, 1, axis=0).tolist(),
                "q99": np.percentile(actions, 99, axis=0).tolist(),
            }
        }
    }

    print(f"[compute_action_statistics] Statistics for '{dataset_name}':")
    print(f"  Mean: {statistics[dataset_name]['action']['mean']}")
    print(f"  Std:  {statistics[dataset_name]['action']['std']}")
    print(f"  Q01:  {statistics[dataset_name]['action']['q01']}")
    print(f"  Q99:  {statistics[dataset_name]['action']['q99']}")

    return statistics


# 示例：如何计算和使用统计信息
if __name__ == "__main__":
    # 1. 计算统计信息（在训练前做一次）
    dummy_actions = np.random.randn(1000, 6)  # 模拟1000个动作
    stats = compute_action_statistics(dummy_actions, dataset_name="my_robot_dataset")

    # 保存
    import json
    with open("action_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("统计信息已保存到 action_statistics.json")
