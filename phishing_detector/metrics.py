"""Métriques orientées sécurité : le rappel sur le phishing est prioritaire."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)


def report(y_true, probs, threshold=0.5) -> dict:
    y_pred = (np.asarray(probs) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "pr_auc": average_precision_score(y_true, probs),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "phishing_missed (FN)": int(fn),
        "false_alarms (FP)": int(fp),
    }


def threshold_for_recall(y_true, probs, target_recall: float) -> float:
    """Le seuil le plus haut (donc la meilleure précision) qui atteint le rappel visé."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]]) if len(ok) else 0.5


def print_report(title: str, r: dict):
    print(f"\n== {title} ==")
    for k, v in r.items():
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
