"""
Chargement des données – Yelp Polarity (robuste).

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from src.preprocessing import get_preprocess_transforms


# =========================
# Vocabulaire
# =========================
class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, counter: Counter, max_size: int = 50000, min_freq: int = 2):
        specials = [self.PAD, self.UNK]
        words = [w for w, c in counter.most_common() if c >= min_freq]
        words = words[: max(0, max_size - len(specials))]
        self.itos = specials + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_idx = self.stoi[self.PAD]
        self.unk_idx = self.stoi[self.UNK]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_idx) for t in tokens]


def build_vocab_from_hf(hf_train, preprocess, vocab_samples: int, vocab_size: int, min_freq: int) -> Vocab:
    n = min(len(hf_train), vocab_samples)
    counter = Counter()
    for i in range(n):
        counter.update(preprocess(hf_train[i]["text"]))
    return Vocab(counter, max_size=vocab_size, min_freq=min_freq)


# =========================
# Dataset Torch (wrap HF Dataset)
# =========================
class YelpTorchDataset(Dataset):
    def __init__(self, hf_ds, preprocess, vocab: Vocab, max_len: int):
        self.ds = hf_ds
        self.preprocess = preprocess
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        tokens = self.preprocess(item["text"])
        ids = self.vocab.encode(tokens)[: self.max_len]
        label = int(item["label"])
        if label not in (0, 1):
            raise ValueError(f"Unexpected label value: {label}")
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


# =========================
# Batch + collate
# =========================
@dataclass
class Batch:
    input_ids: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int, max_len: int) -> Batch:
    seqs, labels = zip(*batch)
    B = len(seqs)
    T = max_len
    x = torch.full((B, T), pad_idx, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)

    for i, s in enumerate(seqs):
        s = s[:T]
        x[i, : len(s)] = s
        mask[i, : len(s)] = True

    return Batch(input_ids=x, mask=mask, labels=torch.stack(labels))


# =========================
# DataLoaders
# =========================
def get_dataloaders(config: dict):
    seed = int(config.get("seed", config.get("train", {}).get("seed", 42)))
    random.seed(seed)
    torch.manual_seed(seed)

    ds_cfg = config["dataset"]
    train_cfg = config.get("train", {})

    max_len = int(ds_cfg.get("max_len", 256))
    vocab_size = int(ds_cfg.get("vocab_size", 50000))
    min_freq = int(ds_cfg.get("min_freq", 2))
    vocab_samples = int(ds_cfg.get("vocab_samples", 200000))
    val_ratio = float(ds_cfg.get("val_ratio", 0.05))

    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(ds_cfg.get("num_workers", 2))

    preprocess = get_preprocess_transforms(config)

    # ---- Load once ----
    hf = load_dataset("yelp_polarity")
    hf_train_full = hf["train"]
    hf_test = hf["test"]

    # ---- Stratified split MANUAL using label column (robuste) ----
    labels = hf_train_full["label"]  # list[int] of 0/1

    idx_pos = [i for i, y in enumerate(labels) if y == 1]
    idx_neg = [i for i, y in enumerate(labels) if y == 0]

    rng = random.Random(seed)
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_val_pos = int(len(idx_pos) * val_ratio)
    n_val_neg = int(len(idx_neg) * val_ratio)

    val_idx = idx_pos[:n_val_pos] + idx_neg[:n_val_neg]
    train_idx = idx_pos[n_val_pos:] + idx_neg[n_val_neg:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    # HF-native selection (important)
    hf_train = hf_train_full.select(train_idx)
    hf_val = hf_train_full.select(val_idx)

    # ---- vocab from TRAIN split only ----
    vocab = build_vocab_from_hf(hf_train, preprocess, vocab_samples, vocab_size, min_freq)

    train_ds = YelpTorchDataset(hf_train, preprocess, vocab, max_len=max_len)
    val_ds = YelpTorchDataset(hf_val, preprocess, vocab, max_len=max_len)
    test_ds = YelpTorchDataset(hf_test, preprocess, vocab, max_len=max_len)

    # ---- Overfit small (optional) ----
    if bool(train_cfg.get("overfit_small", False)):
        overfit_n = int(train_cfg.get("overfit_num_examples", 256))
        overfit_n = max(1, min(overfit_n, len(train_ds)))

        # sample indices inside the train split
        idx = torch.randperm(len(train_ds))[:overfit_n].tolist()
        # easiest: wrap with a tiny subset dataset
        train_ds = torch.utils.data.Subset(train_ds, idx)

        k = min(len(val_ds), max(256, overfit_n))
        vidx = torch.randperm(len(val_ds))[:k].tolist()
        val_ds = torch.utils.data.Subset(val_ds, vidx)

    coll = lambda b: collate_fn(b, pad_idx=vocab.pad_idx, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=coll)

    meta: Dict = {
        "num_classes": 2,
        "input_shape": (max_len,),
        "vocab_size": len(vocab),
        "pad_idx": vocab.pad_idx,
        "max_len": max_len,
    }

    return train_loader, val_loader, test_loader, meta
