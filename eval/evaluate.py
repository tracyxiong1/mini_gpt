import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from data.dataset import build_dataset, get_batch
from data.tokenizer import CharTokenizer
from model.gpt import GPTConfig, GPTModel
from utils.config import load_config


@torch.no_grad()
def estimate_loss(
    model: GPTModel,
    data: torch.Tensor,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> float:
    """估算损失：在验证集上计算平均损失。
    
    Args:
        model: GPT 模型实例
        data: 验证集 ID 序列张量
        eval_iters: 评估迭代次数
        batch_size: 批量大小
        block_size: 上下文窗口大小
        device: 计算设备
    
    Returns:
        float: 平均损失值
    """
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run(config_path: str) -> None:
    """评估入口：加载模型与词表，计算验证集困惑度。
    
    Args:
        config_path: 配置文件路径（默认为 configs/default.yaml）
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg["data"]["data_path"], "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer.load(cfg["infer"]["vocab_path"])
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = build_dataset(ids, cfg["data"]["split_ratio"])
    if len(val_data) < 2:
        raise ValueError("验证数据过短，请提供更长的语料或调整 split_ratio")
    eval_data = val_data
    effective_block_size = min(
        cfg["model"]["block_size"], len(eval_data) - 1, len(train_data) - 1
    )

    model_config_path = cfg["infer"]["model_config_path"]
    if os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        model_cfg = GPTConfig(
            vocab_size=saved["vocab_size"],
            block_size=saved["block_size"],
            n_layers=saved["n_layers"],
            n_heads=saved["n_heads"],
            n_embd=saved["n_embd"],
            dropout=saved["dropout"],
        )
    else:
        model_cfg = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=effective_block_size,
            n_layers=cfg["model"]["n_layers"],
            n_heads=cfg["model"]["n_heads"],
            n_embd=cfg["model"]["n_embd"],
            dropout=cfg["model"]["dropout"],
        )

    model = GPTModel(model_cfg).to(device)
    model.load_state_dict(torch.load(cfg["infer"]["model_path"], map_location=device))

    loss = estimate_loss(
        model,
        eval_data,
        cfg["train"]["eval_iters"],
        cfg["train"]["batch_size"],
        model_cfg.block_size,
        device,
    )
    perplexity = math.exp(loss)
    print(f"eval_loss {loss:.4f} perplexity {perplexity:.4f}")


def main() -> None:
    """命令行入口：解析参数并启动评估。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
