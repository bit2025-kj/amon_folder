"""Étape 1 : nettoyage, déduplication, découpage train/val/test, augmentation, tokenizer.

Usage :
    python prepare_data.py               # dataset + emails transactionnels synthétiques + bruit
    python prepare_data.py --no-augment  # dataset d'origine seul (version 1 du projet)

Le découpage du dataset d'origine est identique dans les deux cas (même graine) :
test.csv reste comparable d'une version à l'autre.
"""
import argparse
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from augment import add_noise, synthetic
from config import Config
from text_utils import normalize


def synth_df(n_legit, n_phish, split, seed, cfg):
    df = pd.DataFrame(synthetic(n_legit, n_phish, split, seed), columns=["text", "label"])
    df["text"] = df["text"].map(lambda t: normalize(t, cfg.max_raw_chars))
    df["source"] = "synthétique"
    return df.drop_duplicates(subset="text")


def augment(train, val, test, cfg):
    """Ajoute les emails synthétiques et remplace une partie du train par une version bruitée."""
    orig_texts = set(train["text"]) | set(val["text"]) | set(test["text"])
    s_train = synth_df(*cfg.synth_train, "train", cfg.seed + 1, cfg)
    s_val = synth_df(*cfg.synth_val, "train", cfg.seed + 2, cfg)
    s_test = synth_df(*cfg.synth_test, "test", cfg.seed + 3, cfg)
    # Aucun email synthétique en double entre les splits
    s_val = s_val[~s_val["text"].isin(s_train["text"])]
    s_test = s_test[~s_test["text"].isin(set(s_train["text"]) | set(s_val["text"]) | orig_texts)]

    train = pd.concat([train.assign(source="dataset"), s_train], ignore_index=True)
    val = pd.concat([val.assign(source="dataset"), s_val], ignore_index=True)

    # Bruit : on REMPLACE noise_frac des emails du train (les 2 classes, même proportion)
    rng = random.Random(cfg.seed)
    idx = train.sample(frac=cfg.noise_frac, random_state=cfg.seed).index
    train.loc[idx, "text"] = [normalize(add_noise(t, rng), cfg.max_raw_chars) for t in train.loc[idx, "text"]]
    train.loc[idx, "source"] = train.loc[idx, "source"] + " + bruit"
    train = train[train["text"].str.len() >= 20].drop_duplicates(subset="text")
    train = train.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    print("\nAugmentation :")
    print(f"  train : +{len(s_train)} synthétiques, {len(idx)} emails bruités")
    print(f"  val   : +{len(s_val)} synthétiques")
    print(f"  test_synth : {len(s_test)} emails (marques et formulations jamais vues)")
    print(train["source"].value_counts().to_string())
    return train, val, s_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-augment", action="store_true", help="dataset d'origine seul")
    args = ap.parse_args()
    cfg = Config()
    df = pd.read_csv(cfg.raw_csv)
    df = df.rename(columns={"Email Text": "text", "Email Type": "label"})[["text", "label"]]
    n0 = len(df)

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].map(lambda t: normalize(t, cfg.max_raw_chars))
    df = df[df["text"].str.len() >= 20]                 # emails vides / inutilisables
    df["label"] = (df["label"] == "Phishing Email").astype(int)

    # Déduplication APRÈS normalisation : sinon un même email peut se retrouver
    # en train ET en test, ce qui gonfle artificiellement les scores.
    df = df.drop_duplicates(subset="text")
    print(f"{n0} emails bruts -> {len(df)} après nettoyage/déduplication")
    print(df["label"].value_counts().rename({0: "safe", 1: "phishing"}).to_string())

    # Découpage stratifié 80 / 10 / 10
    train, tmp = train_test_split(df, test_size=cfg.val_frac + cfg.test_frac,
                                  stratify=df["label"], random_state=cfg.seed)
    val, test = train_test_split(tmp, test_size=cfg.test_frac / (cfg.val_frac + cfg.test_frac),
                                 stratify=tmp["label"], random_state=cfg.seed)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    splits = [("train", train), ("val", val), ("test", test)]
    synth_path = cfg.processed_dir / "test_synth.csv"
    if args.no_augment:
        synth_path.unlink(missing_ok=True)
    else:
        train, val, s_test = augment(train, val, test, cfg)
        splits = [("train", train), ("val", val), ("test", test), ("test_synth", s_test)]
    print()
    for name, part in splits:
        part.to_csv(cfg.processed_dir / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(part):6d} emails  ({part['label'].mean():.1%} phishing)")

    # Tokenizer BPE entraîné UNIQUEMENT sur le train (pas de fuite d'information)
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=cfg.vocab_size, min_frequency=2,
                                  special_tokens=["[PAD]", "[UNK]"])
    tok.train_from_iterator(train["text"].tolist(), trainer=trainer)
    tok.enable_truncation(cfg.max_len)
    tok.enable_padding(pad_id=0, pad_token="[PAD]", length=cfg.max_len)
    cfg.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(cfg.tokenizer_path))
    print(f"Tokenizer BPE : {tok.get_vocab_size()} tokens -> {cfg.tokenizer_path}")


if __name__ == "__main__":
    main()
