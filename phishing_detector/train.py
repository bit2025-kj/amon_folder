"""Étape 3 : entraînement du Transformer (avec reprise) + export du livrable.

Usage :
    python train.py            # démarre, ou REPREND là où on s'était arrêté
    python train.py --fresh    # repart de zéro (ignore la sauvegarde de reprise)
    python train.py --debug    # mini-run rapide sur 256 emails (artifacts/debug/)
    python train.py --no-test  # pas de test interactif à la fin

Ctrl+C à tout moment : la position est sauvegardée, relance  python train.py  pour reprendre.
À la fin : artifacts/phishing_model.pt = le livrable (voir detector.py / predict.py).
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import load_split
from detector import PhishingDetector, interactive
from metrics import print_report, report, threshold_for_recall
from model import PhishingTransformer, count_params


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    for ids, mask, feats, y in tqdm(loader, desc="  évaluation", leave=False, dynamic_ncols=True):
        probs.append(torch.sigmoid(model(ids, mask, feats)))
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy().astype(int)


def atomic_save(obj, path):
    """Écrit dans un fichier temporaire puis renomme : un Ctrl+C pendant
    l'écriture ne peut pas corrompre la sauvegarde précédente."""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_resume(cfg, model, opt, state):
    atomic_save({
        "model": model.state_dict(), "optimizer": opt.state_dict(), "state": state,
        "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(), "model_kwargs": cfg.model_kwargs(),
    }, cfg.last_ckpt)


def debug_first_batch(model, batch, tokenizer):
    ids, mask, feats, y = batch
    print("\n--- DEBUG : premier batch ---")
    print(f"ids {tuple(ids.shape)}  mask {tuple(mask.shape)}  feats {tuple(feats.shape)}  y {tuple(y.shape)}")
    print(f"tokens réels / email : min {int(mask.sum(1).min())}  max {int(mask.sum(1).max())}")
    print(f"labels : {int(y.sum())} phishing / {len(y)}")
    print(f"email 0 : {tokenizer.decode(ids[0, :30].tolist())[:200]}")
    for name, mod in model.named_children():
        print(f"  {name:8s} {count_params(mod):>10,} paramètres")
    print("-----------------------------\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignorer la reprise, repartir de zéro")
    ap.add_argument("--debug", action="store_true", help="mini-run rapide + logs détaillés")
    ap.add_argument("--no-test", action="store_true", help="pas de test interactif à la fin")
    args = ap.parse_args()

    cfg = Config()
    if args.debug:
        cfg.apply_debug()
        torch.autograd.set_detect_anomaly(True)   # signale les NaN / gradients cassés
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    tokenizer = Tokenizer.from_file(str(cfg.tokenizer_path))
    cfg.vocab_size = tokenizer.get_vocab_size()

    print("Encodage des données...")
    train_ds, mu, sd = load_split(cfg, "train", tokenizer)
    val_ds, _, _ = load_split(cfg, "val", tokenizer, mu, sd)
    test_ds, _, _ = load_split(cfg, "test", tokenizer, mu, sd)
    val_dl = DataLoader(val_ds, batch_size=128)
    test_dl = DataLoader(test_ds, batch_size=128)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    model = PhishingTransformer(**cfg.model_kwargs())
    print(f"Paramètres : {count_params(model):,}")

    y_train = train_ds.tensors[3]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()   # compense le déséquilibre
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---------------------------------------------------------------- Reprise
    state = {"epoch": 1, "step": 0, "loss_sum": 0.0, "best_auc": -1.0,
             "bad_epochs": 0, "finished": False, "history": []}
    if cfg.last_ckpt.exists() and not args.fresh:
        ck = torch.load(cfg.last_ckpt, weights_only=False)
        if ck["model_kwargs"] != cfg.model_kwargs():
            raise SystemExit("La config du modèle a changé depuis la sauvegarde : "
                             "relance avec --fresh pour repartir de zéro.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        state = ck["state"]
        torch.set_rng_state(ck["rng_torch"])
        np.random.set_state(ck["rng_numpy"])
        random.setstate(ck["rng_python"])
        if state["finished"]:
            print("Entraînement déjà terminé (utilise --fresh pour recommencer).")
        else:
            print(f">>> Reprise à l'epoch {state['epoch']}, étape {state['step']} "
                  f"(meilleure PR-AUC val : {state['best_auc']:.4f})")
    elif args.fresh:
        for p in (cfg.last_ckpt, cfg.best_weights):
            p.unlink(missing_ok=True)

    # ------------------------------------------------------------ Entraînement
    interrupted = False
    try:
        while not state["finished"] and state["epoch"] <= cfg.max_epochs:
            epoch = state["epoch"]
            # Ordre des batchs déterministe par epoch -> on peut sauter les étapes déjà faites
            g = torch.Generator().manual_seed(cfg.seed + epoch)
            train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
            n_steps, t0 = len(train_dl), time.time()

            model.train()
            # Barre de progression : démarre à l'étape déjà atteinte en cas de reprise
            with tqdm(total=n_steps, initial=state["step"], desc=f"Epoch {epoch}/{cfg.max_epochs}",
                      unit="étape", dynamic_ncols=True) as bar:
                for step, batch in enumerate(train_dl, 1):
                    if step <= state["step"]:
                        continue                               # déjà fait avant l'arrêt
                    if args.debug and epoch == 1 and step == 1:
                        debug_first_batch(model, batch, tokenizer)
                    ids, mask, feats, y = batch
                    loss = loss_fn(model(ids, mask, feats), y)
                    opt.zero_grad()
                    loss.backward()
                    gnorm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()

                    state["step"] = step
                    state["loss_sum"] += loss.item()
                    info = {"loss": f"{state['loss_sum'] / step:.4f}"}
                    if args.debug:
                        info["grad_norm"] = f"{gnorm:.3f}"
                    bar.set_postfix(info)
                    bar.update(1)
                    if step % cfg.save_every == 0:
                        save_resume(cfg, model, opt, state)

            # --- fin d'epoch : validation ---
            p_val, y_val = predict_probs(model, val_dl)
            r = report(y_val, p_val)
            train_loss = state["loss_sum"] / n_steps
            print(f"Epoch {epoch} | loss {train_loss:.4f} | val PR-AUC {r['pr_auc']:.4f} "
                  f"| rappel {r['recall']:.4f} | précision {r['precision']:.4f} "
                  f"| {time.time() - t0:.0f}s")
            state["history"].append({"epoch": epoch, "loss": train_loss, "val_pr_auc": r["pr_auc"]})

            if r["pr_auc"] > state["best_auc"]:
                state["best_auc"], state["bad_epochs"] = r["pr_auc"], 0
                atomic_save(model.state_dict(), cfg.best_weights)
                print("  -> meilleur modèle sauvegardé")
            else:
                state["bad_epochs"] += 1

            state.update(epoch=epoch + 1, step=0, loss_sum=0.0)
            if state["bad_epochs"] >= cfg.patience:
                print("Early stopping.")
                state["finished"] = True
            if state["epoch"] > cfg.max_epochs:
                state["finished"] = True
            save_resume(cfg, model, opt, state)

    except KeyboardInterrupt:
        interrupted = True
        save_resume(cfg, model, opt, state)
        print(f"\n>>> Interrompu à l'epoch {state['epoch']}, étape {state['step']}. "
              f"Position sauvegardée : relance  python train.py{' --debug' if args.debug else ''}  "
              f"pour reprendre.")

    # ------------------------------------------- Export du livrable (meilleurs poids)
    if not cfg.best_weights.exists():
        print("Aucune epoch terminée : pas encore de modèle à exporter.")
        return

    model.load_state_dict(torch.load(cfg.best_weights, weights_only=True))
    p_val, y_val = predict_probs(model, val_dl)
    thr = threshold_for_recall(y_val, p_val, cfg.target_recall)   # seuil choisi sur la VAL
    p_test, y_test = predict_probs(model, test_dl)
    r05, rthr = report(y_test, p_test, 0.5), report(y_test, p_test, thr)
    print_report("Transformer – test @0.5", r05)
    print_report(f"Transformer – test @seuil rappel {cfg.target_recall}", rthr)
    if (cfg.processed_dir / "test_synth.csv").exists():   # emails transactionnels jamais vus
        synth_ds, _, _ = load_split(cfg, "test_synth", tokenizer, mu, sd)
        p_s, y_s = predict_probs(model, DataLoader(synth_ds, batch_size=128))
        print_report(f"Transformer – test_synth (transactionnel) @seuil {thr:.3f}", report(y_s, p_s, thr))

    PhishingDetector.export(cfg.model_path, model, cfg, tokenizer, mu, sd, thr, rthr)
    status = "PROVISOIRE (entraînement non terminé)" if interrupted else "final"
    print(f"\nLivrable {status} : {cfg.model_path}")

    if not interrupted and not args.no_test:
        interactive(PhishingDetector.load(cfg.model_path), debug=args.debug)


if __name__ == "__main__":
    main()
