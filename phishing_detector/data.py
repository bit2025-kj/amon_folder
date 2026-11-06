"""Chargement des splits en tenseurs PyTorch (tokens + masque + features + label)."""
import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.utils.data import TensorDataset

from text_utils import extract_features


def encode_texts(texts: list[str], tokenizer: Tokenizer):
    enc = tokenizer.encode_batch(texts)
    ids = torch.tensor([e.ids for e in enc], dtype=torch.long)
    mask = torch.tensor([e.attention_mask for e in enc], dtype=torch.long)
    return ids, mask


def feature_matrix(texts: list[str]) -> np.ndarray:
    return np.stack([extract_features(t) for t in texts])


FEAT_CLIP = 5.0


def feature_stats(feats: np.ndarray):
    mean, std = feats.mean(0), feats.std(0)
    # Une feature jamais vue (écart-type ~0) exploserait à la division : plancher à 1
    std = np.where(std < 1e-3, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(feats, mean, std):
    """(x - moyenne) / écart-type, borné à ±5 pour qu'une valeur extrême
    ne domine pas toute la décision. Marche avec numpy ou torch."""
    z = (feats - mean) / std
    return z.clip(-FEAT_CLIP, FEAT_CLIP)


def load_split(cfg, name: str, tokenizer: Tokenizer, feat_mean=None, feat_std=None):
    """Retourne (TensorDataset, feat_mean, feat_std).

    La standardisation des features utilise la moyenne / l'écart-type du TRAIN
    (passés en argument pour val/test).
    """
    path = cfg.processed_dir / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} introuvable : lance d'abord  python prepare_data.py")
    df = pd.read_csv(path)
    if cfg.limit:
        df = df.sample(n=min(cfg.limit, len(df)), random_state=cfg.seed)
    texts = df["text"].astype(str).tolist()
    ids, mask = encode_texts(texts, tokenizer)

    feats = feature_matrix(texts)
    if feat_mean is None:
        feat_mean, feat_std = feature_stats(feats)
    feats = torch.tensor(standardize(feats, feat_mean, feat_std), dtype=torch.float32)

    labels = torch.tensor(df["label"].values, dtype=torch.float32)
    return TensorDataset(ids, mask, feats, labels), feat_mean, feat_std
