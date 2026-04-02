import argparse

from eval.evaluate import run as run_eval
from infer.generate import run as run_infer
from train.train import run as run_train

"""命令行入口：集中管理 train / infer 子命令。"""


def main() -> None:
    """命令行入口：解析参数并调用对应子命令。

    子命令：
    - train: 训练模型
    - infer: 生成文本
    - eval: 评估模型
    """
    # 子命令解析
    parser = argparse.ArgumentParser(prog="mini-gpt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml")

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--config", type=str, default="configs/default.yaml")
    infer_parser.add_argument("--prompt", type=str, required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--config", type=str, default="configs/default.yaml")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args.config)
    elif args.command == "infer":
        run_infer(args.config, args.prompt)
    elif args.command == "eval":
        run_eval(args.config)


if __name__ == "__main__":
    main()
