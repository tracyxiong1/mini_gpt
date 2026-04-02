import random

import torch

"""随机种子设置，保证可复现。"""


def set_seed(seed: int) -> None:
    """固定随机种子以复现结果。
    
    Args:
        seed: 随机种子值
    """
    # 统一设置 Python 与 PyTorch 的随机种子
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
