import argparse
import json
import os

import torch
import torch.nn.functional as F

"""训练入口：读取数据、训练 GPT、保存权重与配置。"""

from data.dataset import build_dataset, get_batch
from data.tokenizer import CharTokenizer
from model.gpt import GPTConfig, GPTModel
from utils.config import load_config
from utils.seed import set_seed


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

        # logits shape: (batch, seq_len, vocab_size)
        # y shape:      (batch, seq_len)
        # 这里把前两个维度展平，相当于把“一个 batch 里所有位置的预测”一起算交叉熵。
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run(config_path: str) -> None:
    """训练入口：读取配置、初始化模型、循环训练与评估。
    
    Args:
        config_path: 配置文件路径（默认为 configs/default.yaml）
    """
    cfg = load_config(config_path)
    set_seed(cfg["train"]["seed"])

    device_cfg = cfg["train"]["device"]
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    with open(cfg["data"]["data_path"], "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = build_dataset(ids, cfg["data"]["split_ratio"])
    if len(train_data) < 2:
        raise ValueError("训练数据过短，请提供更长的语料")

    # 验证集过短时，回退到训练集做评估，保证 demo 在小语料上也能跑通。
    eval_data = val_data if len(val_data) >= 2 else train_data

    # block_size 不能超过数据本身长度，否则无法构造 x -> y 的右移监督。
    effective_block_size = min(
        cfg["model"]["block_size"], len(train_data) - 1, len(eval_data) - 1
    )

    vocab_path = cfg["infer"]["vocab_path"]
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    tokenizer.save(vocab_path)

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=effective_block_size,
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        n_embd=cfg["model"]["n_embd"],
        dropout=cfg["model"]["dropout"],
    )
    model = GPTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"])

    for it in range(cfg["train"]["max_iters"]):
        x, y = get_batch(
            train_data, cfg["train"]["batch_size"], effective_block_size, device
        )
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # 典型训练三步：清梯度 -> 反向传播 -> 参数更新。
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % cfg["train"]["eval_interval"] == 0:
            train_loss = estimate_loss(
                model,
                train_data,
                cfg["train"]["eval_iters"],
                cfg["train"]["batch_size"],
                effective_block_size,
                device,
            )
            val_loss = estimate_loss(
                model,
                eval_data,
                cfg["train"]["eval_iters"],
                cfg["train"]["batch_size"],
                effective_block_size,
                device,
            )
            print(f"step {it} train_loss {train_loss:.4f} val_loss {val_loss:.4f}")

    out_dir = cfg["out"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    model_config_path = cfg["infer"]["model_config_path"]

    # 额外保存一份结构配置，推理时就不必依赖训练配置文件完全一致。
    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": model_cfg.vocab_size,
                "block_size": model_cfg.block_size,
                "n_layers": model_cfg.n_layers,
                "n_heads": model_cfg.n_heads,
                "n_embd": model_cfg.n_embd,
                "dropout": model_cfg.dropout,
            },
            f,
            ensure_ascii=False,
        )

    # 权重文件只保存参数值，不包含 Python 代码本身。
    torch.save(model.state_dict(), cfg["infer"]["model_path"])


def main() -> None:
    """命令行入口：解析参数并启动训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
