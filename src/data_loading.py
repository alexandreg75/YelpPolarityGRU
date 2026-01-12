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
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
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


def build_vocab_from_train(preprocess, vocab_samples: int, vocab_size: int, min_freq: int) -> Vocab:
    train_ds = load_dataset("yelp_polarity", split="train")
    n = min(len(train_ds), vocab_samples)
    counter = Counter()
    for i in range(n):
        counter.update(preprocess(train_ds[i]["text"]))
    return Vocab(counter, max_size=vocab_size, min_freq=min_freq)


# -------------------------
# Dataset Torch
# -------------------------
class YelpTorchDataset(Dataset):
    """
    Dataset PyTorch basé sur HuggingFace yelp_polarity.
    Retourne (ids, label) où ids est une séquence d'indices (non paddée).
    """
    def __init__(self, split: str, preprocess, vocab: Vocab, max_len: int):
        self.ds = load_dataset("yelp_polarity", split=split)
        self.preprocess = preprocess
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        text = item["text"]
        label = int(item["label"])

        # sécurité si label est {1,2} au lieu de {0,1}
        if label in (1, 2):
            label = label - 1

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
    # seed
    seed = config.get("seed", config.get("train", {}).get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    ds_cfg = config["dataset"]
    max_len = int(ds_cfg.get("max_len", 256))
    vocab_size = int(ds_cfg.get("vocab_size", 50000))
    min_freq = int(ds_cfg.get("min_freq", 2))
    vocab_samples = int(ds_cfg.get("vocab_samples", 200000))

    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(ds_cfg.get("num_workers", 2))

    # preprocess
    preprocess = get_preprocess_transforms(config)

    # vocab
    vocab = build_vocab_from_train(preprocess, vocab_samples, vocab_size, min_freq)

    # datasets
    full_train = YelpTorchDataset("train", preprocess, vocab, max_len=max_len)
    test_ds = YelpTorchDataset("test", preprocess, vocab, max_len=max_len)

    # train/val split (ex: 95/5)
    val_ratio = float(ds_cfg.get("val_ratio", 0.05))
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # ---- overfit_small : limiter le dataset ----
    if bool(train_cfg.get("overfit_small", False)):
    overfit_n = int(train_cfg.get("overfit_num_examples", ds_cfg.get("overfit_num_examples", 256)))
    overfit_n = max(1, min(overfit_n, len(train_ds)))

    # overfit: on force un petit train set
    train_ds = Subset(train_ds, list(range(overfit_n)))

    # val: si on veut réduire pour accélérer, on échantillonne au hasard (sinon biais énorme)
    k = min(len(val_ds), max(256, overfit_n))
    idx = torch.randperm(len(val_ds))[:k].tolist()
    val_ds = Subset(val_ds, idx)


    # collate
    coll = lambda b: collate_fn(b, pad_idx=vocab.pad_idx, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=coll)

    meta: Dict = {
        "num_classes": 2,
        "input_shape": (max_len,),
        "vocab_size": len(vocab),
        "pad_idx": vocab.pad_idx,
        "vocab_itos": vocab.itos,  # utile pour sauvegarder / debug
        "max_len": max_len,
    }

    return train_loader, val_loader, test_loader, meta
