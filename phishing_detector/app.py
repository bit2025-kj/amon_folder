"""Interface web du détecteur de phishing (Gradio).

    python app.py                 # ouvre http://127.0.0.1:7860 dans le navigateur
    python app.py --port 8000 --no-browser

Onglets :
  Analyser    verdict, probabilité, logit face au seuil, temps de réponse par étape,
              mots et indices décisifs
  Session     historique des analyses + statistiques de latence
  Modèle      architecture lue dans le fichier chargé, paramètres par couche,
              données d'entraînement, déroulé de l'entraînement
  Évaluation  test d'origine ou test transactionnel : métriques, matrices de confusion,
              courbes, erreurs, comparaison avec la référence TF-IDF à son propre seuil
"""
import argparse
import html
import statistics
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tokenizers import Tokenizer

from config import Config
from data import encode_texts, feature_matrix, standardize
from detector import PhishingDetector
from metrics import threshold_for_recall
from model import count_params
from text_utils import FEATURE_NAMES, normalize

CFG = Config()

FEATURE_LABELS = {
    "log_n_urls": "Nombre de liens", "has_ip_url": "Lien vers une adresse IP",
    "has_short_url": "Lien raccourci", "log_n_emails": "Adresses email",
    "urgency": "Mots d'urgence", "credentials": "Identifiants / compte",
    "money": "Argent / gains", "click": "Appels à cliquer",
    "exclam_ratio": "Points d'exclamation", "digit_ratio": "Proportion de chiffres",
    "log_chars": "Longueur (caractères)", "log_words": "Longueur (mots)",
}

EXAMPLES = [
    "URGENT: Your PayPal account has been suspended due to unauthorized activity. "
    "Verify your password within 24 hours at http://192.168.4.21/paypal/login or your "
    "account will be permanently locked. Click here immediately!",
    "Hi team, the weekly sync is moved to Thursday at 3pm in room B. Please bring the "
    "updated project timeline and the budget figures. Thanks, Sarah",
    "Congratulations! You have won $1,000,000 in our international lottery. To claim your "
    "prize, reply with your full name, bank account number and a copy of your ID.",
    "Dear customer, your monthly statement is now available. You can view it by signing in "
    "to your online banking as usual. We will never ask for your password by email.",
]

# Jeux d'évaluation : nom affiché -> fichier de data/processed
EVAL_SETS = {"Test d'origine": "test", "Test transactionnel (synthétique)": "test_synth"}

# ---------------------------------------------------------------- chargement
_t0 = time.perf_counter()
DET = PhishingDetector.load()
LOAD_MS = (time.perf_counter() - _t0) * 1000
MODEL = DET.model
for _p in MODEL.parameters():          # inférence seulement : pas de gradient sur les poids
    _p.requires_grad_(False)
N_PARAMS = count_params(MODEL)
MODEL_MB = CFG.model_path.stat().st_size / 1e6
MODEL_DATE = datetime.fromtimestamp(CFG.model_path.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
DET.predict("warm up")                  # le 1er appel PyTorch est toujours plus lent

# Architecture lue dans le modèle CHARGÉ (et non dans config.py) : l'écran décrit
# toujours le fichier réellement utilisé.
ARCH = dict(vocab=MODEL.tok_emb.num_embeddings, max_len=MODEL.pos_emb.num_embeddings,
            d=MODEL.tok_emb.embedding_dim, layers=len(MODEL.blocks),
            heads=MODEL.blocks[0].attn.n_heads, d_ff=MODEL.blocks[0].ff[0].out_features,
            n_feat=MODEL.head[0].in_features - MODEL.tok_emb.embedding_dim,
            hidden=MODEL.head[0].out_features)
THR_LOGIT = float(np.log(DET.threshold / (1 - DET.threshold)))   # le seuil exprimé en logit

# Copie du tokenizer sans troncature ni remplissage : donne la vraie longueur d'un email
FULL_TOK = Tokenizer.from_str(DET.tokenizer.to_str())
FULL_TOK.no_truncation()
FULL_TOK.no_padding()

# Les données de data/processed ont-elles été préparées avec le tokenizer de ce modèle ?
DATA_MATCH = (CFG.tokenizer_path.exists()
              and Tokenizer.from_file(str(CFG.tokenizer_path)).to_str() == DET.tokenizer.to_str())


def fmt_n(v):
    """1859201 -> '1 859 201'."""
    return f"{v:,}".replace(",", " ")


def fmt_pct(p):
    """Probabilité en % avec assez de décimales pour distinguer 0,04 % de 0,4 %."""
    v = p * 100
    if v < 0.005:
        return "<0.01"          # jamais « 0 % » : le modèle n'est jamais certain à 100 %
    if v >= 99.995:
        return ">99.99"
    return f"{v:.2f}" if v < 1 or v > 99 else f"{v:.1f}"


# ================================================================= analyse
def run_pipeline(raw: str):
    """Une prédiction complète, chronométrée étape par étape (en ms)."""
    t = [time.perf_counter()]
    text = normalize(raw, DET.max_raw_chars)
    t.append(time.perf_counter())
    ids, mask = encode_texts([text], DET.tokenizer)
    t.append(time.perf_counter())
    raw_feats = torch.tensor(feature_matrix([text]))
    feats = standardize(raw_feats, DET.feat_mean, DET.feat_std)
    t.append(time.perf_counter())
    with torch.inference_mode():
        logit = MODEL(ids, mask, feats).item()
    t.append(time.perf_counter())
    prob = 1 / (1 + np.exp(-logit))
    t.append(time.perf_counter())
    steps = ["Normalisation du texte", "Tokenisation BPE", "Features structurelles",
             "Réseau de neurones", "Sigmoïde + décision"]
    timings = {s: (t[i + 1] - t[i]) * 1000 for i, s in enumerate(steps)}
    return dict(text=text, ids=ids, mask=mask, raw_feats=raw_feats, feats=feats,
                logit=logit, prob=float(prob), timings=timings,
                total_ms=(t[-1] - t[0]) * 1000)


def feature_contributions(ids, mask, feats, logit_base):
    """Effet de chaque indice, EN LOGIT : on le remet à sa valeur moyenne et on mesure la variation.
    En logit et non en probabilité : quand le modèle est sûr de lui (99,99 %), la probabilité
    ne bouge presque plus alors que le logit, lui, varie encore nettement."""
    k = len(FEATURE_NAMES)
    batch = feats.repeat(k, 1)
    batch[torch.arange(k), torch.arange(k)] = 0.0
    with torch.inference_mode():
        logits = MODEL(ids.repeat(k, 1), mask.repeat(k, 1), batch).numpy()
    return logit_base - logits     # > 0 : l'indice pousse vers « phishing »


def _logit_from_emb(emb, mask, feats):
    """Fin du réseau à partir des embeddings de tokens (même calcul que PhishingTransformer.forward)."""
    m = MODEL
    x = emb + m.pos_emb(torch.arange(emb.size(1)))
    for block in m.blocks:
        x = block(x, mask)
    x = m.ln_f(x)
    w = mask.unsqueeze(-1).float()
    pooled = (x * w).sum(1) / w.sum(1).clamp(min=1.0)
    return m.head(torch.cat([pooled, feats], dim=1)).squeeze(-1)


def token_saliency(ids, mask, feats, steps=16):
    """Gradients intégrés : contribution de chaque token au logit, en partant d'un email « vide »
    (embeddings de tokens à zéro, positions conservées). La somme des contributions vaut à peu
    près logit(email) - logit(email vide) : chaque valeur est donc en unités de logit."""
    emb = MODEL.tok_emb(ids).detach()                                   # 1 x T x d
    alphas = torch.linspace(1 / steps, 1.0, steps).view(-1, 1, 1)
    path = (alphas * emb).requires_grad_(True)                          # steps x T x d
    _logit_from_emb(path, mask.repeat(steps, 1), feats.repeat(steps, 1)).sum().backward()
    avg_grad = path.grad.mean(0, keepdim=True)
    return (emb * avg_grad).sum(-1)[0].numpy()


# ----------------------------------------------------------- rendu HTML
def card(label, value, sub=""):
    return (f"<div class='kpi'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>")


def kpis(cards):
    return "<div class='kpis'>" + "".join(cards) + "</div>"


def verdict_html(prob, threshold, total_ms):
    phish = prob >= threshold
    cls, label = ("danger", "PHISHING") if phish else ("safe", "LÉGITIME")
    margin = abs(prob - threshold)
    # distance au seuil rapportée à l'espace disponible de ce côté du seuil (0 → 1)
    rel = margin / ((1 - threshold) if phish else threshold)
    conf = "élevée" if rel > 0.6 else "moyenne" if rel > 0.25 else "faible"
    return f"""
<div class='verdict {cls}'>
  <div class='v-top'>
    <div><div class='v-label'>{label}</div>
      <div class='v-sub'>Confiance {conf} · écart au seuil {margin:.3f}</div></div>
    <div class='v-prob'>{fmt_pct(prob)}<span>%</span></div>
  </div>
  <div class='gauge'><div class='gauge-fill' style='width:{prob * 100:.2f}%'></div>
    <div class='gauge-thr' style='left:{threshold * 100:.2f}%'></div></div>
  <div class='gauge-legend'><span>0 % légitime</span>
    <span>seuil {threshold:.3f}</span><span>100 % phishing</span></div>
  <div class='v-time'>⏱ Temps de réponse : <b>{total_ms:.1f} ms</b></div>
</div>"""


def timings_html(timings, total):
    rows = ""
    for name, ms in timings.items():
        pct = ms / total * 100 if total else 0
        rows += (f"<div class='bar-row'><div class='bar-name'>{name}</div>"
                 f"<div class='bar-track'><div class='bar-fill' style='width:{max(pct, .5):.1f}%'></div></div>"
                 f"<div class='bar-val'>{ms:.2f} ms · {pct:.0f} %</div></div>")
    return f"<div class='panel'><div class='panel-title'>Temps par étape</div>{rows}</div>"


def contrib_html(contribs, raw_feats, z):
    order = np.argsort(-np.abs(contribs))
    peak = max(np.abs(contribs).max(), 1e-6)
    rows = ""
    for i in order:
        c = contribs[i]
        side = "pos" if c > 0 else "neg"
        flag = " <span class='tag'>inhabituel</span>" if abs(z[i]) > 2 else ""
        rows += (f"<div class='bar-row'><div class='bar-name'>{FEATURE_LABELS[FEATURE_NAMES[i]]}{flag}</div>"
                 f"<div class='bar-track split'><div class='bar-{side}' "
                 f"style='width:{abs(c) / peak * 50:.1f}%'></div></div>"
                 f"<div class='bar-val'>{c:+.2f} logit · brut {raw_feats[i]:.2f} · z {z[i]:+.2f}</div></div>")
    return ("<div class='panel'><div class='panel-title'>Indices structurels "
            "<small>(variation du logit si l'indice était à sa valeur moyenne)</small></div>"
            f"{rows}<div class='legend'><span class='dot pos'></span>pousse vers phishing "
            "<span class='dot neg'></span>pousse vers légitime</div></div>")


def tokens_html(tokens, sal, truncated, full_len, base_logit, logit):
    peak = max(np.abs(sal).max(), 1e-9)
    chips = ""
    for tok, s in zip(tokens, sal):
        a = min(abs(s) / peak, 1.0) ** 0.7
        color = f"rgba(220,38,38,{a:.2f})" if s > 0 else f"rgba(37,99,235,{a:.2f})"
        chips += (f"<span class='tok' style='background:{color}' "
                  f"title='{s:+.4f}'>{html.escape(tok)}</span>")
    note = (f"<div class='warn'>Email tronqué : il compte {full_len} tokens, le modèle ne lit que les "
            f"{ARCH['max_len']} premiers. La fin de l'email n'influence pas le verdict.</div>"
            if truncated else "")
    # Bilan des gradients intégrés : point de départ (email vide) + somme des mots = logit
    bilan = (f"<div class='note'>Point de départ : un email de même longueur sans texte donne un logit de "
             f"<b>{base_logit:+.2f}</b> ({'phishing' if base_logit > THR_LOGIT else 'légitime'}). "
             f"Le texte de l'email le déplace de <b>{logit - base_logit:+.2f}</b> → logit {logit:+.2f}. "
             "Les couleurs estiment la part de chaque mot dans ce déplacement (gradients intégrés).</div>")
    if peak < 0.05:
        bilan += ("<div class='warn'>Aucun mot ne pèse à lui seul : le texte ne s'écarte pas du point de "
                  "départ. Le modèle classe cet email ainsi parce qu'il n'y trouve aucun signal d'email "
                  "légitime, pas à cause d'un mot précis.</div>")
    return ("<div class='panel'><div class='panel-title'>Ce que le modèle a lu "
            "<small>(rouge = pousse vers phishing, bleu = vers légitime ; survoler pour la valeur)</small></div>"
            f"{note}<div class='toks'>{chips}</div>{bilan}</div>")


def analyze(raw, threshold, history):
    raw = (raw or "").strip()
    if not raw:
        raise gr.Error("Colle d'abord le texte d'un email.")
    r = run_pipeline(raw)
    prob, total = r["prob"], r["total_ms"]

    t_exp = time.perf_counter()
    contribs = feature_contributions(r["ids"], r["mask"], r["feats"], r["logit"])
    n_tok = int(r["mask"].sum())
    sal = token_saliency(r["ids"], r["mask"], r["feats"])[:n_tok]
    with torch.inference_mode():   # logit d'un email « vide » de même longueur : le point de départ
        base_logit = _logit_from_emb(torch.zeros(1, r["ids"].size(1), ARCH["d"]),
                                     r["mask"], r["feats"]).item()
    exp_ms = (time.perf_counter() - t_exp) * 1000

    tokens = [DET.tokenizer.id_to_token(i) for i in r["ids"][0, :n_tok].tolist()]
    full_len = len(FULL_TOK.encode(r["text"]).ids)      # longueur réelle, sans troncature
    truncated = full_len > n_tok
    unk = sum(t == "[UNK]" for t in tokens)
    n_words = len(raw.split())

    # Mot et indice qui pèsent le plus dans CE verdict
    by_tok = pd.Series(sal).groupby(pd.Series(tokens)).sum()
    top_tok = by_tok.abs().idxmax()
    top_feat = int(np.argmax(np.abs(contribs)))
    thr_logit = float(np.log(threshold / (1 - threshold)))

    stats = kpis([
        card("Logit", f"{r['logit']:+.2f}",
             f"seuil = {thr_logit:+.2f} en logit · écart {r['logit'] - thr_logit:+.2f}"),
        card("Tokens lus", f"{n_tok} / {ARCH['max_len']}",
             f"email complet : {full_len} tokens" + (" · TRONQUÉ" if truncated else "")),
        card("Texte", f"{fmt_n(len(raw))} car.", f"{n_words} mots · {unk} token(s) inconnu(s)"),
        (card("Mot le plus influent", html.escape(top_tok),
              f"{'→ phishing' if by_tok[top_tok] > 0 else '→ légitime'} · {by_tok[top_tok]:+.2f} logit")
         if abs(by_tok[top_tok]) >= 0.05 else
         card("Mot le plus influent", "aucun",
              f"aucun mot ne pèse seul · email vide : logit {base_logit:+.2f}")),
        card("Indice le plus influent", FEATURE_LABELS[FEATURE_NAMES[top_feat]],
             f"{'→ phishing' if contribs[top_feat] > 0 else '→ légitime'} · {contribs[top_feat]:+.2f} logit"
             f" · explications en {exp_ms:.0f} ms"),
    ])

    top = pd.DataFrame({"token": tokens, "contribution": sal})
    top = (top.groupby("token", as_index=False)["contribution"].sum()
              .sort_values("contribution", key=np.abs, ascending=False).head(12))
    top["effet"] = np.where(top["contribution"] > 0, "→ phishing", "→ légitime")
    top["contribution"] = top["contribution"].round(4)

    history = history + [{
        "heure": datetime.now().strftime("%H:%M:%S"),
        "aperçu": raw[:70].replace("\n", " ") + ("…" if len(raw) > 70 else ""),
        "P(phishing)": round(prob, 4), "seuil": round(threshold, 3),
        "verdict": "PHISHING" if prob >= threshold else "légitime",
        "latence (ms)": round(total, 2), "tokens": n_tok,
        **{f"{k} (ms)": round(v, 3) for k, v in r["timings"].items()},
    }]
    return (verdict_html(prob, threshold, total), stats, timings_html(r["timings"], total),
            contrib_html(contribs, r["raw_feats"][0].numpy(), r["feats"][0].numpy()),
            tokens_html(tokens, sal, truncated, full_len, base_logit, r["logit"]),
            top[["token", "effet", "contribution"]],
            history, *session_outputs(history))


def benchmark(raw, n_runs):
    raw = (raw or "").strip()
    if not raw:
        raise gr.Error("Colle d'abord le texte d'un email.")
    n_runs = int(n_runs)
    lat = [run_pipeline(raw)["total_ms"] for _ in range(n_runs)]
    # Débit en lot : 32 copies traitées en un seul passage du réseau
    text = normalize(raw, DET.max_raw_chars)
    t0 = time.perf_counter()
    ids, mask = encode_texts([text] * 32, DET.tokenizer)
    feats = standardize(torch.tensor(feature_matrix([text] * 32)), DET.feat_mean, DET.feat_std)
    with torch.inference_mode():
        MODEL(ids, mask, feats)
    batch_ms = (time.perf_counter() - t0) * 1000
    q = np.percentile(lat, [50, 95, 99])
    return kpis([
        card("Moyenne", f"{statistics.mean(lat):.1f} ms", f"sur {n_runs} exécutions"),
        card("Médiane (p50)", f"{q[0]:.1f} ms", f"écart-type {statistics.pstdev(lat):.1f} ms"),
        card("p95 / p99", f"{q[1]:.1f} / {q[2]:.1f} ms", "95 % / 99 % des réponses en dessous"),
        card("Min / Max", f"{min(lat):.1f} / {max(lat):.1f} ms", ""),
        card("Débit unitaire", f"{1000 / statistics.mean(lat):.1f} emails/s", "un email à la fois"),
        card("Débit en lot", f"{32000 / batch_ms:.1f} emails/s", f"lot de 32 en {batch_ms:.0f} ms"),
    ])


# ================================================================= session
def session_outputs(history):
    if not history:
        return "<div class='empty'>Aucune analyse pour l'instant.</div>", pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(history)
    lat = df["latence (ms)"].to_numpy()
    n_phish = int((df["verdict"] == "PHISHING").sum())
    stats = kpis([
        card("Analyses", str(len(df)), "depuis l'ouverture de la page"),
        card("Phishing détectés", f"{n_phish}", f"{n_phish / len(df) * 100:.0f} % des emails analysés"),
        card("P(phishing) moyenne", f"{df['P(phishing)'].mean() * 100:.1f} %", ""),
        card("Latence moyenne", f"{lat.mean():.1f} ms", f"médiane {np.median(lat):.1f} ms"),
        card("Latence p95", f"{np.percentile(lat, 95):.1f} ms", f"min {lat.min():.1f} · max {lat.max():.1f} ms"),
        card("Tokens moyens", f"{df['tokens'].mean():.0f}", f"sur {ARCH['max_len']} maximum"),
    ])
    plot = pd.DataFrame({"analyse": np.arange(1, len(df) + 1), "latence (ms)": lat})
    cols = ["heure", "aperçu", "P(phishing)", "seuil", "verdict", "latence (ms)", "tokens"]
    return stats, df[cols].iloc[::-1], plot


# ================================================================= modèle
def param_table():
    a = ARCH
    groups = {}
    for name, p in MODEL.named_parameters():
        parts = name.split(".")
        if parts[0] == "tok_emb":
            key = f"Embedding des tokens ({fmt_n(a['vocab'])} × {a['d']})"
        elif parts[0] == "pos_emb":
            key = f"Embedding des positions ({a['max_len']} × {a['d']})"
        elif parts[0] == "blocks":
            sub = {"attn": f"self-attention ({a['heads']} têtes, Q, K, V, sortie)",
                   "ff": f"feed-forward {a['d']}→{a['d_ff']}→{a['d']}",
                   "ln1": "LayerNorm", "ln2": "LayerNorm"}[parts[2]]
            key = f"Bloc {int(parts[1]) + 1} · {sub}"
        elif parts[0] == "ln_f":
            key = "LayerNorm finale"
        else:
            key = f"Tête : Dense {a['d'] + a['n_feat']}→{a['hidden']} + Dense {a['hidden']}→1"
        groups[key] = groups.get(key, 0) + p.numel()
    df = pd.DataFrame({"composant": list(groups), "paramètres": list(groups.values())})
    df["part"] = (df["paramètres"] / N_PARAMS * 100).round(2).astype(str) + " %"
    df["paramètres"] = df["paramètres"].map(fmt_n)
    return df


def arch_table():
    """La configuration du modèle CHARGÉ, lue dans ses couches."""
    a = ARCH
    return pd.DataFrame([
        ("Tokenizer", f"BPE, vocabulaire {fmt_n(a['vocab'])} tokens"),
        ("Longueur lue", f"{a['max_len']} tokens par email (au-delà : tronqué)"),
        ("Dimension des vecteurs", str(a["d"])),
        ("Attention", f"{a['heads']} têtes × {a['d'] // a['heads']} dimensions"),
        ("Blocs encodeur", f"{a['layers']} (pre-norm, connexions résiduelles)"),
        ("Feed-forward", f"{a['d']} → {a['d_ff']} → {a['d']}, GELU"),
        ("Résumé de l'email", "moyenne des vrais tokens (remplissage masqué)"),
        ("Indices manuels", f"{a['n_feat']}, standardisés avec les statistiques du train, bornés à ±5"),
        ("Tête", f"{a['d'] + a['n_feat']} → {a['hidden']} (ReLU) → 1 (sigmoïde)"),
        ("Seuil de décision", f"{DET.threshold:.3f} (logit {THR_LOGIT:+.2f}), choisi sur la validation"),
    ], columns=["élément", "valeur"])


def data_table():
    """Les jeux de données préparés, si ce sont bien ceux du modèle chargé."""
    rows = []
    for label, name in (("Entraînement", "train"), ("Validation", "val"),
                        ("Test d'origine", "test"), ("Test transactionnel", "test_synth")):
        path = CFG.processed_dir / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in ("label", "source"))
        detail = (" · ".join(f"{k} : {fmt_n(v)}" for k, v in df["source"].value_counts().items())
                  if "source" in df else "dataset d'origine")
        rows.append((label, fmt_n(len(df)), f"{df['label'].mean() * 100:.1f} %", detail))
    return pd.DataFrame(rows, columns=["jeu", "emails", "part de phishing", "composition"])


def training_history():
    """Déroulé du dernier entraînement (checkpoint de reprise), s'il correspond au modèle."""
    if not CFG.last_ckpt.exists():
        return None, "Aucun historique : le point de reprise de l'entraînement n'existe plus."
    # Fichier écrit par notre propre train.py : weights_only=False est nécessaire (état aléatoire)
    ck = torch.load(CFG.last_ckpt, map_location="cpu", weights_only=False)
    kw = ck.get("model_kwargs", {})
    if (kw.get("d_model"), kw.get("n_layers"), kw.get("vocab_size")) != (ARCH["d"], ARCH["layers"], ARCH["vocab"]):
        return None, "Le point de reprise ne correspond pas au modèle chargé : historique masqué."
    hist = ck["state"].get("history", [])
    if not hist:
        return None, "L'entraînement n'a encore terminé aucune époque."
    best = max(hist, key=lambda h: h["val_pr_auc"])["epoch"]
    df = pd.DataFrame([{"époque": h["epoch"], "perte (train)": round(h["loss"], 4),
                        "PR-AUC (validation)": round(h["val_pr_auc"], 4),
                        "": "★ modèle livré" if h["epoch"] == best else ""} for h in hist])
    status = "terminé" if ck["state"].get("finished") else "en cours"
    stop = (f"arrêt après {ck['state'].get('bad_epochs', 0)} époques sans amélioration"
            if ck["state"].get("finished") and len(hist) < CFG.max_epochs else "")
    note = (f"Entraînement {status} · {len(hist)} époques · meilleure : époque {best} "
            f"(PR-AUC validation {max(h['val_pr_auc'] for h in hist):.4f})" + (f" · {stop}" if stop else ""))
    return df, note


def model_overview():
    m = DET.metrics
    head = kpis([
        card("Paramètres", fmt_n(N_PARAMS), f"{ARCH['layers']} blocs · {ARCH['heads']} têtes · d={ARCH['d']}"),
        card("Fichier livrable", f"{MODEL_MB:.1f} Mo", f"{CFG.model_path.name} · {MODEL_DATE}"),
        card("Chargement", f"{LOAD_MS:.0f} ms", "poids + tokenizer + normalisation"),
        card("Seuil de décision", f"{DET.threshold:.3f}", "choisi sur la validation (rappel visé 97 %)"),
        card("Matériel", "CPU", f"{torch.get_num_threads()} threads · PyTorch {torch.__version__}"),
    ])
    test = kpis([
        card("PR-AUC", f"{m.get('pr_auc', 0):.4f}", "indépendante du seuil"),
        card("Rappel", f"{m.get('recall', 0) * 100:.1f} %", f"{int(m.get('phishing_missed (FN)', 0))} phishing ratés"),
        card("Précision", f"{m.get('precision', 0) * 100:.1f} %", f"{int(m.get('false_alarms (FP)', 0))} fausses alertes"),
        card("F1", f"{m.get('f1', 0) * 100:.1f} %", f"au seuil {m.get('threshold', 0):.3f}"),
    ])
    return head, test


# ================================================================= évaluation
EVAL = {}        # résultats par jeu d'évaluation
BASELINE = {}    # référence TF-IDF, entraînée une fois par session


def _model_stamp():
    return CFG.model_path.stat().st_mtime


def _cache_path(name):
    return CFG.run_dir / f"eval_cache_{name}.npz"


def _baseline(progress):
    """Entraîne TF-IDF + régression logistique et choisit SON seuil sur la validation."""
    if BASELINE:
        return BASELINE
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    progress(0.9, desc="Référence TF-IDF : entraînement")
    train = pd.read_csv(CFG.processed_dir / "train.csv")
    val = pd.read_csv(CFG.processed_dir / "val.csv")
    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(vec.fit_transform(train["text"]), train["label"])
    p_val = clf.predict_proba(vec.transform(val["text"]))[:, 1]
    BASELINE.update(vec=vec, clf=clf, train_s=time.perf_counter() - t0, n_train=len(train),
                    thr=threshold_for_recall(val["label"], p_val, CFG.target_recall))
    return BASELINE


def run_evaluation(set_label, progress=gr.Progress()):
    name = EVAL_SETS[set_label]
    path = CFG.processed_dir / f"{name}.csv"
    if not path.exists():
        raise gr.Error(f"{path.name} introuvable : lance d'abord  python prepare_data.py")
    if not DATA_MATCH:
        gr.Warning("Les données préparées n'ont pas été produites avec le tokenizer de ce modèle : "
                   "les scores ne sont pas ceux de son vrai jeu de test.")
    df = pd.read_csv(path)
    texts, y = df["text"].astype(str).tolist(), df["label"].to_numpy()

    cache = _cache_path(name)
    if cache.exists():
        c = np.load(cache)
        if float(c["stamp"]) == _model_stamp() and len(c["y"]) == len(y):
            EVAL[name] = dict({k: c[k] for k in c.files}, texts=texts, cached=True)
            return eval_view(set_label, DET.threshold) + (gr.update(value=round(DET.threshold, 3)),)

    probs, bs = [], 64
    t0 = time.perf_counter()
    for i in progress.tqdm(range(0, len(texts), bs), desc=f"Transformer sur « {set_label} »"):
        chunk = texts[i:i + bs]
        ids, mask = encode_texts(chunk, DET.tokenizer)
        feats = standardize(torch.tensor(feature_matrix(chunk)), DET.feat_mean, DET.feat_std)
        with torch.inference_mode():
            probs.append(torch.sigmoid(MODEL(ids, mask, feats)).numpy())
    tf_s = time.perf_counter() - t0

    bl = _baseline(progress)
    t0 = time.perf_counter()
    p_bl = bl["clf"].predict_proba(bl["vec"].transform(texts))[:, 1]
    bl_s = time.perf_counter() - t0

    data = dict(p_tf=np.concatenate(probs), p_bl=p_bl, y=y, tf_s=tf_s, bl_s=bl_s,
                bl_thr=bl["thr"], bl_train_s=bl["train_s"], n_train=bl["n_train"], stamp=_model_stamp())
    np.savez(cache, **data)
    EVAL[name] = dict(data, texts=texts, cached=False)
    return eval_view(set_label, DET.threshold) + (gr.update(value=round(DET.threshold, 3)),)


def metrics_at(y, p, thr):
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=rec, precision=prec,
                f1=2 * prec * rec / max(prec + rec, 1e-9), acc=(tp + tn) / len(y),
                fpr=fp / max(fp + tn, 1))


def confusion_html(m, title):
    return f"""
<div class='panel'><div class='panel-title'>{title}</div>
<table class='cm'>
<tr><th></th><th>Prédit légitime</th><th>Prédit phishing</th></tr>
<tr><th>Vrai légitime</th><td class='ok'>{m['tn']}<small>vrais négatifs</small></td>
    <td class='bad'>{m['fp']}<small>fausses alertes</small></td></tr>
<tr><th>Vrai phishing</th><td class='bad'>{m['fn']}<small>phishing ratés</small></td>
    <td class='ok'>{m['tp']}<small>détectés</small></td></tr>
</table></div>"""


def eval_view(set_label, thr):
    name = EVAL_SETS[set_label]
    if name not in EVAL:
        empty = "<div class='empty'>Lance l'évaluation pour afficher les résultats de ce jeu.</div>"
        return (empty, empty, empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame())
    E = EVAL[name]
    y, p_tf, p_bl, bl_thr = E["y"], E["p_tf"], E["p_bl"], float(E["bl_thr"])
    n = len(y)
    mt = metrics_at(y, p_tf, thr)
    mb = metrics_at(y, p_bl, bl_thr)
    tf_ms = float(E["tf_s"]) / n * 1000

    head = kpis([
        card(set_label, fmt_n(n), f"{int(y.sum())} phishing · {n - int(y.sum())} légitimes"),
        card("Rappel", f"{mt['recall'] * 100:.2f} %", f"{mt['fn']} phishing ratés"),
        card("Précision", f"{mt['precision'] * 100:.2f} %", f"{mt['fp']} fausses alertes"),
        card("F1", f"{mt['f1'] * 100:.2f} %", f"au seuil {thr:.3f}"),
        card("PR-AUC / ROC-AUC", f"{average_precision_score(y, p_tf):.4f}",
             f"ROC-AUC {roc_auc_score(y, p_tf):.4f} · indépendantes du seuil"),
        card("Temps d'inférence", f"{tf_ms:.1f} ms/email",
             f"{n / float(E['tf_s']):.0f} emails/s par lots de 64"
             + (" · résultat en cache" if E.get("cached") else "")),
    ])

    ap_tf, ap_bl = average_precision_score(y, p_tf), average_precision_score(y, p_bl)

    def row(name_, p, ap, thr_, s, how):
        m = metrics_at(y, p, thr_)
        return [name_, f"{thr_:.3f}", how, f"{ap:.4f}", f"{m['recall'] * 100:.2f} %",
                f"{m['precision'] * 100:.2f} %", f"{m['f1'] * 100:.2f} %", m["fn"], m["fp"],
                f"{s / n * 1000:.2f} ms"]
    tf_s, bl_s = float(E["tf_s"]), float(E["bl_s"])
    rows_ = [row("Transformer", p_tf, ap_tf, DET.threshold, tf_s, "seuil du livrable"),
             row("Transformer", p_tf, ap_tf, 0.5, tf_s, "seuil neutre"),
             row("TF-IDF + rég. logistique", p_bl, ap_bl, bl_thr, bl_s, "son seuil (validation)"),
             row("TF-IDF + rég. logistique", p_bl, ap_bl, 0.5, bl_s, "seuil neutre")]
    if abs(thr - DET.threshold) > 5e-4 and abs(thr - 0.5) > 5e-4:   # ligne du curseur si elle diffère
        rows_.insert(0, row("Transformer", p_tf, ap_tf, thr, tf_s, "curseur"))
    comp = pd.DataFrame(rows_, columns=["modèle", "seuil", "origine du seuil", "PR-AUC", "rappel",
                                        "précision", "F1", "phishing ratés", "fausses alertes",
                                        "temps / email"])

    cms = (confusion_html(mt, f"Transformer · seuil {thr:.3f} (curseur)")
           + confusion_html(mb, f"TF-IDF · seuil {bl_thr:.3f} (le sien, choisi sur la validation)"))
    cms = f"<div class='two'>{cms}</div>"
    note = ("<div class='note'>Chaque modèle est comparé à <b>son propre</b> seuil, choisi sur la "
            "validation pour viser 97 % de rappel : leurs probabilités ne sont pas calibrées de la même "
            f"façon. Référence TF-IDF entraînée en {float(E['bl_train_s']):.1f} s sur "
            f"{fmt_n(int(E['n_train']))} emails, le même train que le Transformer.</div>")

    # Courbes précision-rappel
    pr = []
    for nm, p in (("Transformer", p_tf), ("TF-IDF", p_bl)):
        prec, rec, _ = precision_recall_curve(y, p)
        idx = np.linspace(0, len(rec) - 1, min(300, len(rec))).astype(int)
        pr.append(pd.DataFrame({"rappel": rec[idx], "précision": prec[idx], "modèle": nm}))
    pr = pd.concat(pr)

    # Métriques du Transformer en fonction du seuil
    rows = []
    for t in np.linspace(0.02, 0.98, 49):
        m = metrics_at(y, p_tf, t)
        rows += [(t, "rappel", m["recall"]), (t, "précision", m["precision"]), (t, "F1", m["f1"])]
    curve = pd.DataFrame(rows, columns=["seuil", "valeur", "métrique"])

    # Distribution des probabilités par vraie classe
    bins = np.linspace(0, 1, 21)
    hist = []
    for label, cls in (("légitime", 0), ("phishing", 1)):
        counts, _ = np.histogram(p_tf[y == cls], bins=bins)
        hist += [(f"{bins[i]:.2f}", int(c), label) for i, c in enumerate(counts)]
    hist = pd.DataFrame(hist, columns=["P(phishing)", "emails", "vraie classe"])

    # Erreurs du Transformer au seuil du curseur
    err = np.where((p_tf >= thr) != (y == 1))[0]
    errors = pd.DataFrame({
        "type": ["phishing raté" if y[i] == 1 else "fausse alerte" for i in err],
        "P(phishing)": np.round(p_tf[err], 4),
        "P TF-IDF": np.round(p_bl[err], 4),
        "TF-IDF aussi faux": ["oui" if (p_bl[i] >= bl_thr) != (y[i] == 1) else "non" for i in err],
        "extrait": [E["texts"][i][:160] + "…" for i in err],
    }).sort_values("P(phishing)")
    return head, cms, note, comp, pr, curve, hist, errors


# ================================================================= interface
CSS = """
.gradio-container {max-width: 1280px !important; margin: auto}
.hero {padding: 28px 32px; border-radius: 18px; color: #fff;
  background: linear-gradient(120deg, #1e1b4b 0%, #4338ca 55%, #7c3aed 100%)}
.hero h1 {margin: 0; font-size: 30px; font-weight: 700; color: #fff}
.hero p {margin: 6px 0 0; opacity: .85; font-size: 15px}
.hero .chips {margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap}
.hero .chip {background: rgba(255,255,255,.14); padding: 4px 12px; border-radius: 999px; font-size: 13px}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px}
.kpi {background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: 14px; padding: 14px 16px}
.kpi-label {font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--body-text-color-subdued)}
.kpi-value {font-size: 24px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere}
.kpi-sub {font-size: 12px; color: var(--body-text-color-subdued); margin-top: 2px}
.verdict {border-radius: 18px; padding: 22px 24px; color: #fff}
.verdict.danger {background: linear-gradient(135deg, #991b1b, #dc2626)}
.verdict.safe {background: linear-gradient(135deg, #065f46, #059669)}
.v-top {display: flex; justify-content: space-between; align-items: center}
.v-label {font-size: 30px; font-weight: 800; letter-spacing: .04em}
.v-sub {opacity: .85; font-size: 14px}
.v-prob {font-size: 54px; font-weight: 800; font-variant-numeric: tabular-nums}
.v-prob span {font-size: 26px; opacity: .8}
.gauge {position: relative; height: 12px; background: rgba(255,255,255,.25); border-radius: 999px; margin-top: 16px}
.gauge-fill {height: 100%; background: #fff; border-radius: 999px}
.gauge-thr {position: absolute; top: -5px; width: 3px; height: 22px; background: #fde047; border-radius: 2px}
.gauge-legend {display: flex; justify-content: space-between; font-size: 12px; opacity: .85; margin-top: 6px}
.v-time {margin-top: 12px; font-size: 15px; background: rgba(0,0,0,.18); display: inline-block;
  padding: 6px 12px; border-radius: 10px}
.panel {background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: 14px; padding: 16px 18px}
.panel-title {font-weight: 700; margin-bottom: 10px}
.panel-title small {font-weight: 400; color: var(--body-text-color-subdued)}
.bar-row {display: grid; grid-template-columns: 210px 1fr 230px; gap: 10px; align-items: center; margin: 6px 0; font-size: 13px}
.bar-track {height: 10px; background: var(--background-fill-secondary); border-radius: 999px; position: relative; overflow: hidden}
.bar-fill {height: 100%; background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 999px}
.bar-track.split::after {content: ''; position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--body-text-color-subdued)}
.bar-pos {position: absolute; left: 50%; height: 100%; background: #dc2626}
.bar-neg {position: absolute; right: 50%; height: 100%; background: #2563eb}
.bar-val {font-variant-numeric: tabular-nums; color: var(--body-text-color-subdued); font-size: 12px}
.tag {background: #f59e0b; color: #111; border-radius: 6px; padding: 0 6px; font-size: 11px; margin-left: 4px}
.legend {font-size: 12px; color: var(--body-text-color-subdued); margin-top: 8px}
.dot {display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 4px 0 12px}
.dot.pos {background: #dc2626} .dot.neg {background: #2563eb}
.toks {line-height: 2.1; font-size: 14px; max-height: 320px; overflow-y: auto}
.tok {padding: 2px 4px; border-radius: 5px; margin: 0 1px; font-family: ui-monospace, monospace}
.warn {background: #fef3c7; color: #78350f; padding: 6px 10px; border-radius: 8px; margin-bottom: 8px; font-size: 13px}
.empty {padding: 24px; text-align: center; color: var(--body-text-color-subdued)}
.note {font-size: 13px; color: var(--body-text-color-subdued); margin-top: 6px}
.two {display: grid; grid-template-columns: 1fr 1fr; gap: 12px}
.cm {width: 100%; border-collapse: separate; border-spacing: 6px; text-align: center}
.cm th {font-size: 13px; color: var(--body-text-color-subdued); font-weight: 600}
.cm td {font-size: 28px; font-weight: 700; padding: 14px; border-radius: 10px}
.cm td small {display: block; font-size: 12px; font-weight: 400; opacity: .8}
.cm td.ok {background: rgba(5,150,105,.15)} .cm td.bad {background: rgba(220,38,38,.15)}
@media (max-width: 800px) {.two {grid-template-columns: 1fr} .bar-row {grid-template-columns: 1fr}}
"""

HERO = f"""
<div class='hero'>
  <h1>🛡️ Détecteur de phishing</h1>
  <p>Transformer écrit de zéro · emails en anglais · analyse locale, rien n'est envoyé sur Internet</p>
  <div class='chips'>
    <span class='chip'>{fmt_n(N_PARAMS)} paramètres</span>
    <span class='chip'>{ARCH['layers']} blocs · {ARCH['heads']} têtes · d={ARCH['d']}</span>
    <span class='chip'>BPE {fmt_n(ARCH['vocab'])} tokens · {ARCH['max_len']} lus</span>
    <span class='chip'>seuil {DET.threshold:.3f}</span>
    <span class='chip'>F1 test {DET.metrics.get('f1', 0) * 100:.1f} %</span>
    <span class='chip'>modèle du {MODEL_DATE}</span>
  </div>
</div>"""


def build():
    with gr.Blocks(title="Détecteur de phishing") as demo:
        gr.HTML(HERO)
        history = gr.State([])

        with gr.Tab("🔍 Analyser"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    email = gr.Textbox(label="Email à analyser (anglais)", lines=12,
                                       placeholder="Colle ici l'objet et le corps de l'email…")
                    threshold = gr.Slider(0.01, 0.99, value=round(DET.threshold, 3), step=0.001,
                                          label="Seuil de décision",
                                          info=f"Seuil du modèle : {DET.threshold:.3f}. Plus bas = moins "
                                               "de phishing ratés, plus de fausses alertes")
                    with gr.Row():
                        go = gr.Button("Analyser", variant="primary", size="lg")
                        gr.ClearButton(email, value="Effacer", size="lg")
                    gr.Examples(EXAMPLES, inputs=email, label="Exemples")
                with gr.Column(scale=6):
                    verdict = gr.HTML("<div class='empty'>Le verdict apparaîtra ici.</div>")
                    stats = gr.HTML()
                    timing = gr.HTML()
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    toks = gr.HTML()
                with gr.Column(scale=5):
                    contrib = gr.HTML()
            top_tokens = gr.Dataframe(label="Tokens les plus influents", interactive=False)
            with gr.Accordion("⏱ Mesurer la latence (benchmark)", open=False):
                with gr.Row():
                    n_runs = gr.Slider(5, 200, value=30, step=5, label="Nombre d'exécutions")
                    bench_btn = gr.Button("Lancer le benchmark")
                bench_out = gr.HTML()

        with gr.Tab("📈 Session"):
            sess_stats = gr.HTML("<div class='empty'>Aucune analyse pour l'instant.</div>")
            lat_plot = gr.LinePlot(x="analyse", y="latence (ms)", title="Temps de réponse par analyse",
                                   height=260)
            sess_table = gr.Dataframe(label="Historique (le plus récent en haut)", interactive=False)
            export_btn = gr.Button("Exporter l'historique en CSV")
            export_file = gr.File(label="Fichier CSV", visible=False)

        with gr.Tab("🧠 Modèle"):
            head, test = model_overview()
            gr.HTML(head)
            gr.Markdown(f"#### Scores enregistrés dans le livrable · test d'origine, seuil {DET.threshold:.3f}")
            gr.HTML(test)
            with gr.Row():
                gr.Dataframe(param_table(), label="Paramètres par composant (lus dans le modèle chargé)",
                             interactive=False)
                gr.Dataframe(arch_table(), label="Architecture du modèle chargé", interactive=False)
            gr.Markdown("#### Données d'entraînement et d'évaluation")
            if not DATA_MATCH:
                gr.Markdown("⚠️ *Les fichiers de `data/processed` n'ont pas été préparés avec le tokenizer "
                            "de ce modèle : les tableaux ci-dessous décrivent d'autres données.*")
            gr.Dataframe(data_table(), interactive=False, wrap=True)
            hist_df, hist_note = training_history()
            gr.Markdown(f"#### Déroulé de l'entraînement\n{hist_note}")
            if hist_df is not None:
                with gr.Row():
                    gr.Dataframe(hist_df, interactive=False)
                    gr.LinePlot(hist_df, x="époque", y="PR-AUC (validation)",
                                title="PR-AUC de validation par époque", height=260)

        with gr.Tab("📊 Évaluation"):
            gr.Markdown("Évalue le modèle sur des emails **jamais vus pendant l'entraînement** et le compare "
                        "à la référence TF-IDF. **Test d'origine** : 1 750 emails du dataset. "
                        "**Test transactionnel** : emails de banques, colis, abonnements… générés avec des "
                        "marques et des phrases réservées, le cas que la v2 devait corriger. "
                        "Premier lancement : environ 1 minute par jeu, puis le résultat est mis en cache.")
            with gr.Row():
                eval_set = gr.Radio([k for k, v in EVAL_SETS.items()
                                     if (CFG.processed_dir / f"{v}.csv").exists()],
                                    value="Test d'origine", label="Jeu d'évaluation")
                eval_btn = gr.Button("Lancer l'évaluation", variant="primary")
                eval_thr = gr.Slider(0.01, 0.99, value=round(DET.threshold, 3), step=0.001,
                                     label="Seuil du Transformer (appliqué en direct)",
                                     info=f"Seuil du livrable : {DET.threshold:.3f}")
            ev_head = gr.HTML("<div class='empty'>Lance l'évaluation pour afficher les résultats.</div>")
            ev_comp = gr.Dataframe(label="Comparaison des modèles", interactive=False)
            ev_cm = gr.HTML()
            ev_note = gr.HTML()
            with gr.Row():
                ev_pr = gr.LinePlot(x="rappel", y="précision", color="modèle",
                                    title="Courbe précision-rappel", height=320)
                ev_curve = gr.LinePlot(x="seuil", y="valeur", color="métrique",
                                       title="Transformer : métriques selon le seuil", height=320)
            ev_hist = gr.BarPlot(x="P(phishing)", y="emails", color="vraie classe",
                                 title="Distribution des probabilités du Transformer", height=300)
            ev_err = gr.Dataframe(label="Erreurs du Transformer au seuil du curseur", interactive=False, wrap=True)

        # --------------------------------------------------------- événements
        sess_outs = [sess_stats, sess_table, lat_plot]
        go.click(analyze, [email, threshold, history],
                 [verdict, stats, timing, contrib, toks, top_tokens, history, *sess_outs])
        email.submit(analyze, [email, threshold, history],
                     [verdict, stats, timing, contrib, toks, top_tokens, history, *sess_outs])
        bench_btn.click(benchmark, [email, n_runs], bench_out)

        def export(h):
            if not h:
                raise gr.Error("L'historique est vide.")
            path = Path(CFG.run_dir) / "historique_analyses.csv"
            pd.DataFrame(h).to_csv(path, index=False, encoding="utf-8-sig")
            return gr.update(value=str(path), visible=True)
        export_btn.click(export, history, export_file)

        ev_outs = [ev_head, ev_cm, ev_note, ev_comp, ev_pr, ev_curve, ev_hist, ev_err]
        eval_btn.click(run_evaluation, eval_set, ev_outs + [eval_thr])
        eval_thr.release(eval_view, [eval_set, eval_thr], ev_outs)
        eval_set.change(eval_view, [eval_set, eval_thr], ev_outs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Interface web du détecteur de phishing")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    build().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser,
                   theme=theme, css=CSS)
