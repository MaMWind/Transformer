"""
main.py — entry point for nmt-starter.
Do not modify this file. Your work goes in src/model.py, src/train.py, src/evaluate.py.

Usage:
    python main.py --config=configs/debug.yaml
    python main.py --config=configs/sample.yaml
"""

import argparse
import yaml
import torch

from src.model import build_model
from src.train import train


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    print(f"Model parameters: {model.count_parameters()}")

    train(model, cfg["train"], device)
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
