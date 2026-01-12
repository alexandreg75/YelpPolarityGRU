"""
Évaluation.

Usage :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
"""

import argparse
import yaml
import torch
import torch.nn as nn

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import get_device


@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch.input_ids.to(device)
        m = batch.mask.to(device)
        y = batch.labels.to(device)

        logits = model(x, m).view(-1)
        loss = crit(logits, y)

        total_loss += float(loss.item()) * y.numel()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += int((preds == y).sum().item())
        total += int(y.numel())

    return total_loss / max(1, total), correct / max(1, total)


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # support multiple formats
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "model_state_dict"]:
            if k in ckpt:
                model.load_state_dict(ckpt[k])
                return
    # fallback: raw state_dict
    model.load_state_dict(ckpt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = get_device(cfg.get("train", {}).get("device", "auto"))
    print("Device:", device)

    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)
    cfg.setdefault("model", {})
    cfg["model"]["vocab_size"] = meta["vocab_size"]
    cfg["model"]["pad_idx"] = meta["pad_idx"]

    model = build_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)

    test_loss, test_acc = eval_loop(model, test_loader, device)
    print(f"TEST loss: {test_loss:.4f} | TEST acc: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
