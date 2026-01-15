"""
Évaluation — Yelp Polarity GRU (Test).

Exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher les métriques de test (loss + accuracy)
"""

import argparse
import yaml
import torch
import torch.nn as nn

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import get_device, set_seed


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).long()
    return (preds == labels.long()).float().mean().item()


@torch.no_grad()
def evaluate_split(model, loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    acc_sum = 0.0
    n = 0

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        mask = batch.mask.to(device)
        labels = batch.labels.to(device)

        logits = model(input_ids, mask)
        loss = crit(logits, labels)

        bs = labels.size(0)
        loss_sum += loss.item() * bs
        acc_sum += accuracy_from_logits(logits, labels) * bs
        n += bs

    return loss_sum / max(1, n), acc_sum / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # ---- config ----
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- seed + device ----
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("train", {}).get("device", "auto"))
    print("Device:", device)

    # ---- dataloaders + meta ----
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)

    # ✅ IMPORTANT : inject meta info into config (like in train.py)
    cfg.setdefault("dataset", {})
    cfg["dataset"]["vocab_size_effective"] = int(meta["vocab_size"])
    cfg["dataset"]["pad_idx"] = int(meta["pad_idx"])

    # ---- model ----
    model = build_model(cfg).to(device)

    # ---- load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device)

    # ton train.py sauvegarde "model_state"
    model.load_state_dict(ckpt["model_state"], strict=True)

    # ---- evaluate ----
    test_loss, test_acc = evaluate_split(model, test_loader, device)

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc*100:.2f}%")
    print("==============================\n")


if __name__ == "__main__":
    main()
