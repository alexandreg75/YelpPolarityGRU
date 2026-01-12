"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

import torch
import torch.nn as nn


class GRUMaxPoolClassifier(nn.Module):
    """
    Embedding(200) -> GRU(H) -> Global Max Pooling temporel -> Linear(H->1)
    Sortie: logits (B,)
    """
    def __init__(self, vocab_size: int, pad_idx: int, embed_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T) indices
        mask: (B, T) bool, True pour tokens réels, False pour padding
        """
        x = self.embedding(input_ids)      # (B, T, E)
        out, _ = self.gru(x)               # (B, T, H)

        # Global Max Pooling sur la dimension temporelle en ignorant le padding
        minus_inf = torch.finfo(out.dtype).min
        out = out.masked_fill(~mask.unsqueeze(-1), minus_inf)  # pads -> -inf
        pooled, _ = torch.max(out, dim=1)  # (B, H)

        logits = self.fc(pooled).squeeze(-1)  # (B,)
        return logits


def build_model(config: dict) -> nn.Module:
    """
    Construit et retourne un nn.Module selon la config.

    Attendu dans config :
      - model.embed_dim (imposé: 200)
      - model.hidden_size (H)
      - dataset.vocab_size_effective (déduit du vocab) OU model.vocab_size
      - dataset.pad_idx (déduit)
    """
    embed_dim = int(config["model"]["embed_dim"])
    hidden_size = int(config["model"]["hidden_size"])

    # Ces champs seront injectés par train.py après création du vocab
    vocab_size = int(config["dataset"]["vocab_size_effective"])
    pad_idx = int(config["dataset"]["pad_idx"])

    return GRUMaxPoolClassifier(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
    )
