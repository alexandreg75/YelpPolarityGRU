"""
Mini grid search — implémenté.

Exécutable via :
    python -m src.grid_search --config configs/config.yaml

Fonctions :
- lit la section 'hparams' de la config
- lance plusieurs runs en variant les hyperparamètres
- loggue les hparams + résultats (TensorBoard) et écrit un summary CSV

Notes :
- Chaque run utilise un sous-dossier unique dans runs/ et artifacts/
- On appelle `src.train` en sous-process (simple et robuste)
"""

import argparse
import copy
import itertools
import os
import subprocess
import time
import csv
import yaml

from torch.utils.tensorboard import SummaryWriter


def _flatten_grid(hparams: dict):
    """Retourne liste de dicts {key: value} pour toutes les combinaisons."""
    keys = list(hparams.keys())
    values_lists = [hparams[k] for k in keys]
    combos = []
    for values in itertools.product(*values_lists):
        combos.append(dict(zip(keys, values)))
    return combos


def _set_nested(cfg: dict, key: str, value):
    """
    Applique un hyperparam dans la config.
    On mappe quelques clés fréquentes vers leur chemin YAML.
    """
    # mapping simple pour ton projet
    mapping = {
        "lr": ("train", "lr"),
        "batch_size": ("train", "batch_size"),
        "weight_decay": ("train", "weight_decay"),
        "hidden_size": ("model", "hidden_size"),
        "max_len": ("dataset", "max_len"),
        "vocab_samples": ("dataset", "vocab_samples"),
        "vocab_size": ("dataset", "vocab_size"),
        "embed_dim": ("model", "embed_dim"),
    }

    if key in mapping:
        a, b = mapping[key]
        cfg.setdefault(a, {})
        cfg[a][b] = value
    else:
        # fallback: si user met directement "train.lr" etc.
        if "." in key:
            parts = key.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value
        else:
            # sinon on met à la racine
            cfg[key] = value


def _parse_best_val_acc(stdout: str):
    """
    Récupère Best val acc imprimé par train.py :
      "Training done. Best val acc: 0.93"
    """
    marker = "Training done. Best val acc:"
    for line in stdout.splitlines()[::-1]:
        if marker in line:
            try:
                return float(line.split(marker, 1)[1].strip())
            except Exception:
                return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    # options pratiques pour limiter la durée en local CPU
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="limiter le nombre de configs testées")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    hparams = base_cfg.get("hparams", {})
    if not hparams or not isinstance(hparams, dict):
        raise ValueError("Ta config doit contenir une section 'hparams' (dict).")

    combos = _flatten_grid(hparams)
    if args.limit is not None:
        combos = combos[: args.limit]

    # dossier global du grid search
    ts = int(time.time())
    parent_runs = base_cfg["paths"]["runs_dir"]
    parent_artifacts = base_cfg["paths"]["artifacts_dir"]

    gs_name = f"gridsearch_{ts}"
    gs_runs_dir = os.path.join(parent_runs, gs_name)
    gs_artifacts_dir = os.path.join(parent_artifacts, gs_name)
    os.makedirs(gs_runs_dir, exist_ok=True)
    os.makedirs(gs_artifacts_dir, exist_ok=True)

    # TensorBoard global (hparams + métriques)
    writer = SummaryWriter(log_dir=gs_runs_dir)

    # CSV résumé
    csv_path = os.path.join(gs_runs_dir, "summary.csv")
    csv_fields = ["run_name", "best_val_acc"] + list(hparams.keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        wcsv = csv.DictWriter(fcsv, fieldnames=csv_fields)
        wcsv.writeheader()

        print(f"== Grid search: {len(combos)} runs ==")
        print(f"Logs: {gs_runs_dir}")
        print(f"CSV : {csv_path}")

        best_overall = (-1.0, None)

        for i, hp in enumerate(combos, start=1):
            cfg = copy.deepcopy(base_cfg)

            # appliquer HP
            for k, v in hp.items():
                _set_nested(cfg, k, v)

            # seed override si demandé
            if args.seed is not None:
                cfg["seed"] = int(args.seed)

            # forcer des runs courts
            cfg.setdefault("train", {})
            cfg["train"]["epochs"] = int(args.max_epochs)

            # chemins uniques pour ne pas écraser
            run_name = "gs_" + "_".join([f"{k}={hp[k]}" for k in sorted(hp.keys())])
            # sanitize (espaces, etc.)
            run_name = run_name.replace("/", "_").replace(" ", "")

            cfg["paths"]["runs_dir"] = gs_runs_dir
            cfg["paths"]["artifacts_dir"] = os.path.join(gs_artifacts_dir, run_name)
            cfg["paths"]["best_ckpt_path"] = os.path.join(cfg["paths"]["artifacts_dir"], "best.ckpt")
            os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)

            # écrire une config temporaire pour cette run
            tmp_cfg_path = os.path.join(gs_runs_dir, f"{run_name}.yaml")
            with open(tmp_cfg_path, "w", encoding="utf-8") as ft:
                yaml.safe_dump(cfg, ft, sort_keys=False, allow_unicode=True)

            # lancer train en sous-process
            cmd = [
                "python3", "-m", "src.train",
                "--config", tmp_cfg_path,
                "--max_steps", str(args.max_steps),
                "--max_epochs", str(args.max_epochs),
            ]

            print(f"\n[{i}/{len(combos)}] RUN {run_name}")
            print("CMD:", " ".join(cmd))

            p = subprocess.run(cmd, capture_output=True, text=True)
            stdout = p.stdout + "\n" + p.stderr

            # log console + fichier
            log_path = os.path.join(gs_runs_dir, f"{run_name}.log")
            with open(log_path, "w", encoding="utf-8") as flog:
                flog.write(stdout)

            best_val_acc = _parse_best_val_acc(stdout)
            if best_val_acc is None:
                best_val_acc = -1.0

            # TensorBoard: hparams
            # add_hparams demande dict[str, (int/float/str/bool)] + métriques
            metrics = {"best/val_acc": best_val_acc}
            writer.add_hparams(hp, metrics, run_name=run_name)

            # CSV
            row = {"run_name": run_name, "best_val_acc": best_val_acc}
            row.update(hp)
            wcsv.writerow(row)

            print(f"-> best_val_acc = {best_val_acc:.4f} | log: {log_path}")

            if best_val_acc > best_overall[0]:
                best_overall = (best_val_acc, run_name)

        print(f"\n== DONE. Best overall: {best_overall[1]} with val_acc={best_overall[0]:.4f} ==")

    writer.close()
    print(f"Summary CSV saved at: {csv_path}")


if __name__ == "__main__":
    main()
