"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter

from datasets import load_dataset

from src.preprocessing import get_preprocess_transforms


# -------------------------
# Vocabulaire
# -------------------------
class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, counter: Counter, max_size: int = 50000, min_freq: int = 2):
        specials = [self.PAD, self.UNK]
        words = [w for w, c in counter.most_common() if c >= min_freq]
        if max_size is not None:
            words = words[: max(0, max_size - len(specials))]
        self.itos = specials + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_idx = self.stoi[self.PAD]
        self.unk_idx = self.stoi[self.UNK]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_idx) for t in tokens]


def build_vocab_from_hf_train(hf_train, preprocess, vocab_samples: int, vocab_size: int, min_freq: int) -> Vocab:
    n = min(len(hf_train), vocab_samples)
    counter = Counter()
    for i in range(n):
        counter.update(preprocess(hf_train[i]["text"]))
    return Vocab(counter, max_size=vocab_size, min_freq=min_freq)


# -------------------------
# Dataset Torch (wrap HF Dataset)
# -------------------------
class YelpTorchDataset(Dataset):
    """
    Wrapper PyTorch autour d'un HF Dataset déjà sélectionné (train/val/test).
    Retourne (ids, label) où ids est une séquence d'indices (non paddée).
    """
    def __init__(self, hf_ds, preprocess, vocab: Vocab, max_len: int):
        self.ds = hf_ds
        self.preprocess = preprocess
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        text = item["text"]
        label = int(item["label"])

        # HuggingFace yelp_polarity: labels ∈ {0,1}
        if label not in (0, 1):
            raise ValueError(f"Unexpected label value: {label}")

        tokens = self.preprocess(text)
        ids = self.vocab.encode(tokens)[: self.max_len]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


# -------------------------
# Batch + collate
# -------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor  # (B, T)
    mask: torch.Tensor       # (B, T) bool
    labels: torch.Tensor     # (B,)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int, max_len: int) -> Batch:
    seqs, labels = zip(*batch)
    B = len(seqs)
    T = max_len

    input_ids = torch.full((B, T), fill_value=pad_idx, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)

    for i, seq in enumerate(seqs):
        s = seq[:T]
        input_ids[i, : len(s)] = s
        mask[i, : len(s)] = True

    return Batch(input_ids=input_ids, mask=mask, labels=torch.stack(labels))


# -------------------------
# DataLoaders
# -------------------------
def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    """
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

    # preprocess
    preprocess = get_preprocess_transforms(config)

    # ---- load once + stratified split ----
    ds = load_dataset("yelp_polarity")
    hf_train = ds["train"]
    hf_test = ds["test"]

    split = hf_train.train_test_split(
        test_size=val_ratio,
        seed=seed,
        stratify_by_column="label",
    )
    hf_train_split = split["train"]
    hf_val_split = split["test"]

    # vocab from train split
    vocab = build_vocab_from_hf_train(hf_train_split, preprocess, vocab_samples, vocab_size, min_freq)

    # wrap torch datasets
    train_ds = YelpTorchDataset(hf_train_split, preprocess, vocab, max_len=max_len)
    val_ds = YelpTorchDataset(hf_val_split, preprocess, vocab, max_len=max_len)
    test_ds = YelpTorchDataset(hf_test, preprocess, vocab, max_len=max_len)

    # ---- overfit_small : limiter le dataset (aléatoire, mais labels OK grâce au split stratifié) ----
    if bool(train_cfg.get("overfit_small", False)):
        overfit_n = int(train_cfg.get("overfit_num_examples", ds_cfg.get("overfit_num_examples", 256)))
        overfit_n = max(1, min(overfit_n, len(train_ds)))

        train_idx = torch.randperm(len(train_ds))[:overfit_n].tolist()
        train_ds = Subset(train_ds, train_idx)

        k = min(len(val_ds), max(256, overfit_n))
        val_idx = torch.randperm(len(val_ds))[:k].tolist()
        val_ds = Subset(val_ds, val_idx)

    coll = lambda b: collate_fn(b, pad_idx=vocab.pad_idx, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=coll)

    meta: Dict = {
        "num_classes": 2,
        "input_shape": (max_len,),
        "vocab_size": len(vocab),
        "pad_idx": vocab.pad_idx,
        "vocab_itos": vocab.itos,
        "max_len": max_len,
    }

    return train_loader, val_loader, test_loader, meta
