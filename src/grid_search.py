"""
Mini grid search.

Usage :
    python -m src.grid_search --config configs/config.yaml
"""

import argparse
import itertools
import os
import time
import yaml
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    base = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    h = base.get("hparams", {})
    if not h:
        raise ValueError("No 'hparams' section found in config.yaml")

    # on supporte lr, hidden_size, max_len (si présents)
    grid_keys = list(h.keys())
    grid_vals = [h[k] for k in grid_keys]

    runs = list(itertools.product(*grid_vals))
    print(f"Grid search: {len(runs)} runs -> keys={grid_keys}")

    for i, values in enumerate(runs, 1):
        cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
        for k, v in zip(grid_keys, values):
            # routes typiques
            if k == "lr":
                cfg.setdefault("train", {})["lr"] = float(v)
            elif k == "batch_size":
                cfg.setdefault("train", {})["batch_size"] = int(v)
            elif k == "hidden_size":
                cfg.setdefault("model", {})["hidden_size"] = int(v)
            elif k == "max_len":
                cfg.setdefault("dataset", {})["max_len"] = int(v)
            else:
                cfg.setdefault("train", {})[k] = v

        # écrit config temporaire
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            tmp_path = f.name

        print(f"\n[{i}/{len(runs)}] Running with {dict(zip(grid_keys, values))}")
        try:
            subprocess.check_call(["python", "-m", "src.train", "--config", tmp_path])
        finally:
            os.remove(tmp_path)

    print("Grid search done.")

if __name__ == "__main__":
    main()
