"""Le livrable : un détecteur prêt à l'emploi, chargé depuis UN seul fichier.

    from detector import PhishingDetector
    det = PhishingDetector.load()                 # artifacts/phishing_model.pt
    det.predict("Your account is suspended...")   # {'probability': 0.98, 'is_phishing': True, ...}

Le fichier .pt contient : les poids, l'architecture, le tokenizer, la normalisation
des features et le seuil de décision. Aucun réentraînement, aucun autre fichier requis.
"""
import torch
from tokenizers import Tokenizer

from config import Config
from data import encode_texts, feature_matrix, standardize
from model import PhishingTransformer
from text_utils import FEATURE_NAMES, normalize

FORMAT_VERSION = 1


class PhishingDetector:
    def __init__(self, model, tokenizer, max_raw_chars, feat_mean, feat_std, threshold, metrics=None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_raw_chars = max_raw_chars
        self.feat_mean, self.feat_std = feat_mean, feat_std
        self.threshold = threshold
        self.metrics = metrics or {}

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def export(path, model, cfg: Config, tokenizer: Tokenizer, feat_mean, feat_std,
               threshold: float, metrics: dict):
        bundle = {
            "format_version": FORMAT_VERSION,
            "model_kwargs": cfg.model_kwargs(),
            "state_dict": model.state_dict(),
            "tokenizer": tokenizer.to_str(),
            "max_raw_chars": cfg.max_raw_chars,
            "feat_mean": torch.as_tensor(feat_mean, dtype=torch.float32),
            "feat_std": torch.as_tensor(feat_std, dtype=torch.float32),
            "threshold": float(threshold),
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        torch.save(bundle, path)

    @classmethod
    def load(cls, path=None):
        path = path or Config().model_path
        # weights_only=True : chargement sûr, le fichier ne peut pas exécuter de code
        b = torch.load(path, map_location="cpu", weights_only=True)
        model = PhishingTransformer(**b["model_kwargs"])
        model.load_state_dict(b["state_dict"])
        return cls(model, Tokenizer.from_str(b["tokenizer"]), b["max_raw_chars"],
                   b["feat_mean"], b["feat_std"], b["threshold"], b["metrics"])

    # ------------------------------------------------------------ prédiction
    def _tensors(self, raw: str):
        text = normalize(raw, self.max_raw_chars)
        ids, mask = encode_texts([text], self.tokenizer)
        raw_feats = torch.tensor(feature_matrix([text]))
        feats = standardize(raw_feats, self.feat_mean, self.feat_std)
        return text, ids, mask, raw_feats, feats

    @torch.no_grad()
    def predict(self, raw: str) -> dict:
        _, ids, mask, _, feats = self._tensors(raw)
        p = torch.sigmoid(self.model(ids, mask, feats)).item()
        return {"probability": p, "is_phishing": p >= self.threshold, "threshold": self.threshold}

    @torch.no_grad()
    def explain(self, raw: str) -> dict:
        """Mode debug : montre tout ce que le modèle "voit"."""
        text, ids, mask, raw_feats, feats = self._tensors(raw)
        n_tok = int(mask.sum())
        full_len = len(self.tokenizer.encode(text).ids) if n_tok == mask.size(1) else n_tok
        tokens = [self.tokenizer.id_to_token(i) for i in ids[0, :n_tok].tolist()]
        logit = self.model(ids, mask, feats).item()

        print("\n--- DEBUG ---")
        print(f"Texte normalisé ({len(text)} car.) : {text[:300]}{'...' if len(text) > 300 else ''}")
        print(f"Tokens utilisés : {n_tok}/{mask.size(1)}"
              + (f"  (tronqué : l'email en fait plus)" if full_len > n_tok else ""))
        print(f"Premiers tokens : {tokens[:40]}")
        unk = sum(t == "[UNK]" for t in tokens)
        print(f"Tokens inconnus [UNK] : {unk}")
        print("Features (brute -> standardisée) :")
        for name, r, s in zip(FEATURE_NAMES, raw_feats[0].tolist(), feats[0].tolist()):
            flag = "  <-- plafonné" if abs(s) >= 5 else ("  <-- inhabituel" if abs(s) > 2 else "")
            print(f"  {name:15s} {r:8.3f} -> {s:+6.2f}{flag}")
        print(f"Logit : {logit:+.3f}   P(phishing) = {torch.sigmoid(torch.tensor(logit)).item():.4f}"
              f"   seuil = {self.threshold:.4f}")
        print("-------------")
        return self.predict(raw)


def format_result(r: dict) -> str:
    verdict = "PHISHING" if r["is_phishing"] else "légitime"
    return f"P(phishing) = {r['probability']:.3f}  (seuil {r['threshold']:.3f})  ->  {verdict}"


def interactive(detector: PhishingDetector, debug: bool = False):
    """Boucle : on colle un email, on tape FIN sur une ligne seule pour l'analyser."""
    print("\nColle un email (plusieurs lignes possibles), puis tape FIN sur une ligne seule.")
    print("Tape Q pour quitter.\n")
    while True:
        lines = []
        try:
            while True:
                line = input("> " if not lines else "  ")
                if line.strip().upper() == "Q" and not lines:
                    return
                if line.strip().upper() == "FIN":
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print()
            return
        email = "\n".join(lines).strip()
        if not email:
            continue
        r = detector.explain(email) if debug else detector.predict(email)
        print(format_result(r) + "\n")
