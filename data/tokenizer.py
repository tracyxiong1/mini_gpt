import json
from typing import Dict, List

"""字符级分词器，提供最小可用的编码/解码与保存加载。"""


class CharTokenizer:
    def __init__(self, text: str) -> None:
        """字符级分词器：从文本构建词表。

        Args:
            text: 用于构建词表的文本字符串

        Output:
            初始化后的 CharTokenizer 实例，包含 stoi 和 itos 映射
        """
        chars = sorted(list(set(text)))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}  # 字符到ID
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}  # ID到字符

    @property
    def vocab_size(self) -> int:
        """词表大小。

        Returns:
            int: 词表中字符的数量
        """
        return len(self.stoi)

    def encode(self, s: str) -> List[int]:
        """编码：文本转ID序列。

        Args:
            s: 输入文本字符串

        Returns:
            List[int]: 编码后的ID序列

        Raises:
            ValueError: 如果文本包含词表外的字符
        """
        unknown = [c for c in s if c not in self.stoi]
        if unknown:
            raise ValueError(
                f"prompt contains unknown characters: {sorted(set(unknown))}"
            )
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        """解码：ID序列转文本。

        Args:
            ids: 输入ID序列

        Returns:
            str: 解码后的文本字符串
        """
        return "".join([self.itos[i] for i in ids])

    def save(self, path: str) -> None:
        """保存词表到文件。

        Args:
            path: 保存路径（通常为 .json 文件）
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False)

    @staticmethod
    def load(path: str, text: str | None = None) -> "CharTokenizer":
        """从文件加载词表。

        Args:
            path: 词表文件路径
            text: 可选，用于构建分词器的文本（若不提供则使用词表中的字符）

        Returns:
            CharTokenizer: 加载后的分词器实例
        """
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tok = CharTokenizer(text or "".join(obj["stoi"].keys()))
        tok.stoi = obj["stoi"]
        tok.itos = {i: ch for ch, i in tok.stoi.items()}
        return tok
