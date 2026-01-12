"""
Chargement des données – Yelp Polarity (robuste).

Split stratifié MANUEL (fix définitif labels=0 uniquement).
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from datasets import load_dataset
from src.preprocessing import get_preprocess_transforms


# =========================
# Vocabulaire
# =========================
class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, counter: Counter, max_size=50000, min_freq=2):
        specials = [self.PAD, self.UNK]
        words = [w for w, c in counter.most_common() if c >= min_freq]
        words = words[: max(0, max_size - len(specials))]
        self.itos = specials + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_idx = self.stoi[self.PAD]
        self.unk_idx = self.stoi[self.UNK]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_idx) for t in tokens]


def build_vocab(train_ds, preprocess, vocab_samples, vocab_size, min_freq):
    counter = Counter()
    n = min(len(train_ds), vocab_samples)
    for i in range(n):
        counter.update(preprocess(train_ds[i]["text"]))
    return Vocab(counter, vocab_size, min_freq)


# =========================
# Dataset Torch
# =========================
class YelpTorchDataset(Dataset):
    def __init__(self, hf_ds, preprocess, vocab, max_len):
        self.ds = hf_ds
        self.preprocess = preprocess
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        tokens = self.preprocess(item["text"])
        ids = self.vocab.encode(tokens)[: self.max_len]
        label = int(item["label"])
        return torch.tensor(ids), torch.tensor(label, dtype=torch.float32)


# =========================
# Batch
# =========================
@dataclass
class Batch:
    input_ids: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor


def collate_fn(batch, pad_idx, max_len):
    seqs, labels = zip(*batch)
    B, T = len(seqs), max_len
    x = torch.full((B, T), pad_idx, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)

    for i, s in enumerate(seqs):
        s = s[:T]
        x[i, : len(s)] = s
        mask[i, : len(s)] = True

    return Batch(x, mask, torch.stack(labels))


# =========================
# DataLoaders
# =========================
def get_dataloaders(config: dict):

    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    ds_cfg = config["dataset"]
    train_cfg = config["train"]

    preprocess = get_preprocess_transforms(config)

    max_len = ds_cfg["max_len"]
    batch_size = train_cfg["batch_size"]
    vocab_size = ds_cfg["vocab_size"]
    min_freq = ds_cfg["min_freq"]
    vocab_samples = ds_cfg["vocab_samples"]

    # ---- Load HF dataset ONCE ----
    hf = load_dataset("yelp_polarity")
    hf_train = hf["train"]
    hf_test = hf["test"]

    # ---- MANUAL stratified split ----
    idx_pos = [i for i in range(len(hf_train)) if hf_train[i]["label"] == 1]
    idx_neg = [i for i in range(len(hf_train)) if hf_train[i]["label"] == 0]

    random.shuffle(idx_pos)
    random.shuffle(idx_neg)

    val_ratio = 0.05
    n_val_pos = int(len(idx_pos) * val_ratio)
    n_val_neg = int(len(idx_neg) * val_ratio)

    val_idx = idx_pos[:n_val_pos] + idx_neg[:n_val_neg]
    train_idx = idx_pos[n_val_pos:] + idx_neg[n_val_neg:]

    random.shuffle(train_idx)
    random.shuffle(val_idx)

    hf_train_split = Subset(hf_train, train_idx)
    hf_val_split = Subset(hf_train, val_idx)

    # ---- vocab ----
    vocab = build_vocab(hf_train_split, preprocess, vocab_samples, vocab_size, min_freq)

    train_ds = YelpTorchDataset(hf_train_split, preprocess, vocab, max_len)
    val_ds = YelpTorchDataset(hf_val_split, preprocess, vocab, max_len)
    test_ds = YelpTorchDataset(hf_test, preprocess, vocab, max_len)

    coll = lambda b: collate_fn(b, vocab.pad_idx, max_len)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, collate_fn=coll)

    meta = {
        "num_classes": 2,
        "input_shape": (max_len,),
        "pad_idx": vocab.pad_idx,
        "vocab_size": len(vocab),
    }

    return train_loader, val_loader, test_loader, meta
