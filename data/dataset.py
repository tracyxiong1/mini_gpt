from typing import Tuple

import torch

"""数据切分与批量采样。"""


def build_dataset(
    token_ids: torch.Tensor, split_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建数据集：按比例切分训练集与验证集。
    
    Args:
        token_ids: 编码后的ID序列张量
        split_ratio: 训练集占比（例如 0.9 表示 90% 训练集）
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (train_data, val_data)，分别为训练集和验证集的ID序列
    """
    n = int(split_ratio * len(token_ids))
    train_data = token_ids[:n]
    val_data = token_ids[n:]
    return train_data, val_data


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """采样批量数据：随机截取上下文对 (x, y)。
    
    Args:
        data: 输入ID序列张量
        batch_size: 批量大小
        block_size: 上下文窗口大小
        device: 计算设备（如 'cpu' 或 'cuda'）
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (x, y)
            - x: 输入张量，形状为 (batch_size, block_size)
            - y: 标签张量，形状为 (batch_size, block_size)，是 x 向右偏移一位的结果
    
    Raises:
        ValueError: 如果数据长度小于或等于 block_size
    """
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError("data length must be larger than block_size")

    # 随机选择 batch_size 个起点；每个起点都会截出一段长度为 block_size 的连续文本。
    ix = torch.randint(max_start, (batch_size,))

    # x 是当前上下文，y 是整体右移一位后的目标，训练目标就是“用前面的字符预测后一个字符”。
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
