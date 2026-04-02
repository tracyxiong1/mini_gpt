import argparse
import json
import os

import torch
import torch.nn.functional as F

"""推理入口：加载模型与词表，按采样策略生成文本。"""

from data.tokenizer import CharTokenizer
from model.gpt import GPTConfig, GPTModel
from utils.config import load_config


def top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """采样过滤：结合 top-k 与 top-p 截断 logits。
    
    Args:
        logits: 模型输出的 logits，形状为 (batch_size, vocab_size)
        top_k: 保留概率最高的 k 个 token
        top_p: 保留累积概率不超过 p 的 token
    
    Returns:
        torch.Tensor: 过滤后的 logits
    """
    if top_k > 0:
        # top-k: 只保留分数最高的 k 个候选，其余位置设为 -inf，不再参与采样。
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < min_values, torch.full_like(logits, float("-inf")), logits
        )
    if 0 < top_p < 1.0:
        # top-p: 先按概率从高到低排序，只保留累积概率达到阈值 p 之前的候选。
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """生成：自回归预测新 token。
    
    Args:
        model: GPT 模型实例
        idx: 输入 ID 序列，形状为 (batch_size, seq_len)
        max_new_tokens: 生成的最大 token 数
        temperature: 采样温度（>0，值越大越随机）
        top_k: 采样过滤参数
        top_p: 采样过滤参数
    
    Returns:
        torch.Tensor: 生成后的 ID 序列，形状为 (batch_size, seq_len + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        # 如果上下文已经超过窗口大小，只保留最后 block_size 个 token 参与本轮预测。
        idx_cond = idx[:, -model.config.block_size :]
        logits = model(idx_cond)

        # 只取最后一个位置的输出，因为我们当前只关心“下一个 token”。
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        logits = top_k_top_p_filter(logits, top_k, top_p)
        probs = F.softmax(logits, dim=-1)

        # multinomial 会按概率分布随机采样一个 token，而不是永远取概率最大的那个。
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def run(config_path: str, prompt: str) -> None:
    """推理入口：加载模型与词表，生成文本。
    
    Args:
        config_path: 配置文件路径（默认为 configs/default.yaml）
        prompt: 生成的起始文本
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CharTokenizer.load(cfg["infer"]["vocab_path"])
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
            vocab_size=len(tokenizer.stoi),
            block_size=cfg["model"]["block_size"],
            n_layers=cfg["model"]["n_layers"],
            n_heads=cfg["model"]["n_heads"],
            n_embd=cfg["model"]["n_embd"],
            dropout=cfg["model"]["dropout"],
        )
    model = GPTModel(model_cfg).to(device)
    model.load_state_dict(torch.load(cfg["infer"]["model_path"], map_location=device))
    model.eval()

    # prompt 先编码成整数 ID，模型内部只处理数字，不直接处理字符串。
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = generate(
        model,
        idx,
        cfg["infer"]["max_new_tokens"],
        cfg["infer"]["temperature"],
        cfg["infer"]["top_k"],
        cfg["infer"]["top_p"],
    )
    text = tokenizer.decode(out[0].tolist())
    print(text)


def main() -> None:
    """命令行入口：解析参数并启动推理。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()
    run(args.config, args.prompt)


if __name__ == "__main__":
    main()
