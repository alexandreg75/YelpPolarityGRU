"""
Entraînement principal (projet Yelp Polarity GRU + Global MaxPool).

Exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)
"""

import argparse
import os
import time
import math

import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # stabilité (optionnel)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # logits: (B,) ; labels: (B,)
    preds = (torch.sigmoid(logits) >= 0.5).long()
    return (preds == labels.long()).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        mask = batch.mask.to(device)
        labels = batch.labels.to(device).view(-1)

        logits = model(input_ids, mask).view(-1)
        loss = crit(logits, labels)

        if not torch.isfinite(loss):
            continue

        bs = labels.size(0)
        loss_sum += float(loss.item()) * bs
        acc_sum += accuracy_from_logits(logits, labels) * bs
        n += bs

    return loss_sum / max(1, n), acc_sum / max(1, n)


def sanity_check(model, loader, device, steps: int) -> None:
    """Vérifie que la loss/acc sont cohérentes sur quelques mini-batches.
    Robuste même si le DataLoader contient moins de 'steps' batches.
    """
    model.train()
    crit = nn.BCEWithLogitsLoss()
    done = 0
    for batch in loader:
        input_ids = batch.input_ids.to(device)
        mask = batch.mask.to(device)
        labels = batch.labels.to(device).view(-1)

        logits = model(input_ids, mask).view(-1)
        loss = crit(logits, labels)
        acc = accuracy_from_logits(logits, labels)

        done += 1
        print(f"[SANITY] step {done}/{steps} | loss={loss.item():.4f} | acc={acc*100:.2f}%")

        if done >= steps:
            break

    if done < steps:
        print(f"[SANITY] Only {done} batches available (requested {steps}).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    # ---- read config ----
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- seed ----
    seed = args.seed if args.seed is not None else config.get("seed", config.get("train", {}).get("seed", 42))
    set_seed(int(seed))

    # ---- apply overfit flag into config so data_loading can see it ----
    if args.overfit_small:
        config.setdefault("train", {})
        config["train"]["overfit_small"] = True

    # ---- device ----
    device_cfg = config.get("train", {}).get("device", "auto")
    device = get_device(device_cfg)
    print("Device:", device)

    # ---- dataloaders ----
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    # Injecter les infos nécessaires au modèle (vocab_size/pad_idx) dans la config
    config.setdefault("dataset", {})
    config["dataset"]["vocab_size_effective"] = int(meta["vocab_size"])
    config["dataset"]["pad_idx"] = int(meta["pad_idx"])

    # ---- model ----
    model = build_model(config).to(device)

    # ---- training settings ----
    train_cfg = config["train"]
    lr = float(train_cfg.get("lr", train_cfg.get("optimizer", {}).get("lr", 1e-3)))
    weight_decay = float(train_cfg.get("weight_decay", train_cfg.get("optimizer", {}).get("weight_decay", 0.0)))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    log_every_steps = int(train_cfg.get("log_every_steps", 200))

    # epochs / steps override
    epochs = int(train_cfg.get("epochs", 3))
    if args.max_epochs is not None:
        epochs = int(args.max_epochs)

    max_steps = args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else None

    if args.overfit_small:
        epochs = int(train_cfg.get("overfit_epochs", max(epochs, 20)))
        print(f"⚙️ Overfit mode: epochs set to {epochs}")

    # ---- loss + optimizer ----
    crit = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- paths ----
    runs_dir = config["paths"]["runs_dir"]
    artifacts_dir = config["paths"]["artifacts_dir"]
    best_path = config["paths"].get("best_ckpt_path", os.path.join(artifacts_dir, "best.ckpt"))
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # ---- TensorBoard run name ----
    H = config["model"]["hidden_size"]
    L = config["dataset"]["max_len"]
    run_name = f"gru_maxpool_H{H}_L{L}_lr{lr}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(runs_dir, run_name))

    # ---- sanity check ----
    sanity_steps = int(train_cfg.get("sanity_steps", 0))
    if sanity_steps > 0:
        sanity_check(model, train_loader, device, sanity_steps)

    # ---- training loop ----
    best_val_acc = -1.0
    global_step = 0

    # seuil anti-explosion pour BCE (si dépasse, on skip)
    bad_loss_threshold = float(train_cfg.get("bad_loss_threshold", 50.0))

    for epoch in range(1, epochs + 1):
        model.train()

        loss_sum = 0.0
        acc_sum = 0.0
        n = 0
        skipped = 0

        for batch in train_loader:
            input_ids = batch.input_ids.to(device)
            mask = batch.mask.to(device)
            labels = batch.labels.to(device).view(-1)

            logits = model(input_ids, mask).view(-1)
            loss = crit(logits, labels)

            # skip NaN/Inf + skip loss énorme (explosion)
            if (not torch.isfinite(loss)) or (float(loss.item()) > bad_loss_threshold):
                skipped += 1
                print(f"[WARN] Bad train loss at step={global_step}: {loss.item()}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if max_steps is not None and global_step >= max_steps:
                    break
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = None
            if grad_clip and grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            # si grad_norm est inf/nan : on skip l'update
            if grad_norm is not None and (not torch.isfinite(torch.tensor(grad_norm))):
                skipped += 1
                print(f"[WARN] Non-finite grad_norm at step={global_step}: {grad_norm}. Skipping optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if max_steps is not None and global_step >= max_steps:
                    break
                continue

            optimizer.step()

            bs = labels.size(0)
            loss_sum += float(loss.item()) * bs
            acc_sum += accuracy_from_logits(logits.detach(), labels) * bs
            n += bs

            global_step += 1

            if global_step % log_every_steps == 0:
                train_loss_step = loss_sum / max(1, n)
                train_acc_step = acc_sum / max(1, n)
                writer.add_scalar("train/loss_step", train_loss_step, global_step)
                writer.add_scalar("train/acc_step", train_acc_step, global_step)

            if max_steps is not None and global_step >= max_steps:
                break

        train_loss = loss_sum / max(1, n)
        train_acc = acc_sum / max(1, n)

        val_loss, val_acc = evaluate(model, val_loader, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.add_scalar("train/skipped_batches", skipped, epoch)

        train_loss_str = f"{train_loss:.4f}" if math.isfinite(train_loss) else "nan"
        val_loss_str = f"{val_loss:.4f}" if math.isfinite(val_loss) else "nan"

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_loss_str} acc {train_acc*100:.2f}% | "
            f"val loss {val_loss_str} acc {val_acc*100:.2f}%"
        )

        # save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "model_state": model.state_dict(),
                "state_dict": model.state_dict(),
                "config": config,
                "meta": meta,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
                "global_step": global_step,
            }
            torch.save(ckpt, best_path)
            print(f"✓ Saved best -> {best_path} (val_acc={best_val_acc*100:.2f}%)")

        if max_steps is not None and global_step >= max_steps:
            print("Reached max_steps, stopping.")
            break

    writer.close()
    print("Training done. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
