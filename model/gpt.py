import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

"""最小版 GPT 模型：因果自注意力 + MLP + 残差块堆叠。"""


@dataclass
class GPTConfig:
    """GPT 模型配置。
    
    Args:
        vocab_size: 词表大小
        block_size: 上下文窗口大小
        n_layers: Transformer 层数
        n_heads: 注意力头数
        n_embd: 嵌入维度
        dropout:  dropout 概率
    """
    # 模型结构配置
    vocab_size: int
    block_size: int
    n_layers: int
    n_heads: int
    n_embd: int
    dropout: float


class CausalSelfAttention(nn.Module):
    """因果自注意力：只关注当前及之前的位置。"""
    def __init__(self, config: GPTConfig) -> None:
        """
        Args:
            config: GPT 配置对象，包含 n_heads、n_embd、dropout 等参数
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        # 一次性映射出 QKV
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 因果 mask，禁止看到未来
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("bias", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, n_embd)
        
        Returns:
            torch.Tensor: 注意力输出，形状与输入相同
        """
        b, t, c = x.size()
        # 线性层一次性产出 Q、K、V，随后按最后一维均分成三份。
        qkv = self.c_attn(x)
        q, k, v = qkv.split(c, dim=2)

        # 从 (b, t, c) 变成 (b, n_heads, t, head_dim)，便于并行计算多头注意力。
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数 shape: (b, n_heads, t, t)，表示每个位置对历史各位置的关注程度。
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 因果 mask 会把“未来位置”设为 -inf，softmax 后这些位置的概率就会变成 0。
        att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 用注意力权重对 V 做加权求和，得到每个头的新表示。
        y = att @ v

        # 把多头结果重新拼回 (b, t, c)，再做一次线性投影回模型主干维度。
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """多层感知机：特征变换。"""
    def __init__(self, config: GPTConfig) -> None:
        """
        Args:
            config: GPT 配置对象，包含 n_embd、dropout 等参数
        """
        super().__init__()
        # 前馈网络：升维 -> 激活 -> 降维
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, n_embd)
        
        Returns:
            torch.Tensor: MLP 输出，形状与输入相同
        """
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer 块：注意力 + MLP + 残差连接。"""
    def __init__(self, config: GPTConfig) -> None:
        """
        Args:
            config: GPT 配置对象
        """
        super().__init__()
        # Pre-LN 结构
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, n_embd)
        
        Returns:
            torch.Tensor: Block 输出，形状与输入相同
        """
        # 残差连接让每一层更像“在原表示上做增量修正”，训练通常更稳定。
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """GPT 模型：词嵌入 + 位置嵌入 + 多层 Block + 语言模型头。"""
    def __init__(self, config: GPTConfig) -> None:
        """
        Args:
            config: GPT 配置对象，包含 vocab_size、block_size、n_layers 等参数
        """
        super().__init__()
        self.config = config
        # 词嵌入与位置嵌入
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        # 多层 Transformer 块
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: 输入ID序列，形状为 (batch_size, seq_len)
        
        Returns:
            torch.Tensor: 语言模型输出 logits，形状为 (batch_size, seq_len, vocab_size)
        """
        b, t = idx.size()

        # token embedding 表示“这个字符是谁”，position embedding 表示“它在第几个位置”。
        pos = torch.arange(0, t, device=idx.device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        # 多层 Block 逐步混合上下文信息，让每个位置都能理解它前面的内容。
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # 每个位置都会输出一个 vocab_size 维分数向量，表示“下一个字符是谁”的倾向。
        logits = self.lm_head(x)
        return logits
