"""Normalisation du texte + features structurelles manuelles.

Pourquoi normaliser ? Dans ce dataset, une partie des emails est déjà en minuscules
et "pré-tokenisée" ("doesn ' t", "http : / / www . x . com"), l'autre est brute.
On met TOUS les emails dans le même format pour que le modèle apprenne le contenu,
pas le format d'origine.
"""
import math
import re

import numpy as np

_PUNCT = re.compile(r"([^\w\s])")
_SPACES = re.compile(r"\s+")


def normalize(text: str, max_chars: int = 20_000) -> str:
    """minuscules, ponctuation isolée par des espaces, espaces fusionnés."""
    text = str(text)[:max_chars].lower()
    text = _PUNCT.sub(r" \1 ", text)
    return _SPACES.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Features : calculées sur une version "compacte" du texte normalisé
# ("http : / / www . x . com" -> "http://www.x.com") pour que les regex marchent.
# ---------------------------------------------------------------------------
_COMPACT = re.compile(r" ?([^\w\s]) ?")

_URL = re.compile(r"https?://\S+|www\.\S+")
_IP_URL = re.compile(r"(https?://)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
_SHORT_URL = re.compile(r"\b(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly)\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

_URGENCY = ["urgent", "immediately", "suspend", "suspended", "expire", "expires", "expired",
            "within 24 hours", "act now", "limited time", "final notice", "locked",
            "unauthorized", "warning", "alert", "asap", "deadline", "last chance"]
_CREDENTIALS = ["password", "login", "log in", "sign in", "verify your", "confirm your",
                "update your", "account", "social security", "ssn", "credit card",
                "bank", "pin", "billing", "paypal"]
_MONEY = ["$", "dollars", "free", "winner", "won", "prize", "cash", "million",
          "lottery", "investment", "offer", "discount", "%"]
_CLICK = ["click here", "click below", "click the link", "follow the link", "visit",
          "unsubscribe", "remove me"]

FEATURE_NAMES = [
    "log_n_urls", "has_ip_url", "has_short_url", "log_n_emails",
    "urgency", "credentials", "money", "click",
    "exclam_ratio", "digit_ratio", "log_chars", "log_words",
]


def _count(text: str, words: list[str]) -> int:
    return sum(text.count(w) for w in words)


def extract_features(norm_text: str) -> np.ndarray:
    """12 indices structurels, à partir du texte DÉJÀ normalisé."""
    compact = _COMPACT.sub(r"\1", norm_text)
    n_words = max(len(norm_text.split()), 1)
    n_chars = max(len(norm_text), 1)
    feats = [
        math.log1p(len(_URL.findall(compact))),
        float(bool(_IP_URL.search(compact))),
        float(bool(_SHORT_URL.search(compact))),
        math.log1p(len(_EMAIL.findall(compact))),
        math.log1p(_count(norm_text, _URGENCY)),
        math.log1p(_count(norm_text, _CREDENTIALS)),
        math.log1p(_count(norm_text, _MONEY)),
        math.log1p(_count(norm_text, _CLICK)),
        norm_text.count("!") / n_words,
        sum(c.isdigit() for c in norm_text) / n_chars,
        math.log1p(n_chars),
        math.log1p(n_words),
    ]
    return np.asarray(feats, dtype=np.float32)


assert len(FEATURE_NAMES) == 12
