"""
LR finder.

Usage :
    python3 -m src.lr_finder --config configs/config.yaml
"""

import argparse
import time
import yaml

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("train", {}).get("device", "auto"))
    print("Device:", device)

    train_loader, _, _, meta = get_dataloaders(cfg)

    # ✅ IMPORTANT : build_model() attend ces clés dans cfg["dataset"]
    cfg.setdefault("dataset", {})
    cfg["dataset"]["vocab_size_effective"] = int(meta["vocab_size"])
    cfg["dataset"]["pad_idx"] = int(meta["pad_idx"])

    model = build_model(cfg).to(device)
    crit = nn.BCEWithLogitsLoss()

    # LR sweep (plage plus safe)
    lr_start = 1e-5
    lr_end = 1e-1
    num_steps = 200
    gamma = (lr_end / lr_start) ** (1 / max(1, num_steps - 1))

    optim = torch.optim.Adam(model.parameters(), lr=lr_start)

    runs_dir = cfg.get("paths", {}).get("runs_dir", "runs")
    writer = SummaryWriter(log_dir=f"{runs_dir}/lr_finder_{int(time.time())}")

    model.train()
    step = 0
    lr = lr_start

    it = iter(train_loader)
    while step < num_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        for pg in optim.param_groups:
            pg["lr"] = lr

        x = batch.input_ids.to(device)
        m = batch.mask.to(device)
        y = batch.labels.to(device)

        optim.zero_grad(set_to_none=True)
        logits = model(x, m).view(-1)
        loss = crit(logits, y)

        # stop si NaN/Inf
        if not torch.isfinite(loss):
            print(f"[STOP] Non-finite loss at step={step}, lr={lr:.2e}")
            break

        loss.backward()
        optim.step()

        writer.add_scalar("lr_finder/lr", lr, step)
        writer.add_scalar("lr_finder/loss", float(loss.item()), step)

        if step % 20 == 0:
            print(f"[LR] step {step}/{num_steps} lr={lr:.2e} loss={loss.item():.4f}")

        lr *= gamma
        step += 1

    writer.close()
    print("✅ LR finder done. Open TensorBoard on runs/ to inspect lr_finder/loss vs lr_finder/lr.")


if __name__ == "__main__":
    main()
