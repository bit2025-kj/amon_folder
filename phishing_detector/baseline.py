"""Étape 2 : modèle de référence TF-IDF + régression logistique.

Le Transformer n'a d'intérêt que s'il bat ce score (surtout le rappel).
Usage :  python baseline.py
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import Config
from metrics import print_report, report, threshold_for_recall


def main():
    cfg = Config()
    train, val, test = (pd.read_csv(cfg.processed_dir / f"{n}.csv") for n in ("train", "val", "test"))

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)
    Xtr = vec.fit_transform(train["text"])
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(Xtr, train["label"])

    p_val = clf.predict_proba(vec.transform(val["text"]))[:, 1]
    p_test = clf.predict_proba(vec.transform(test["text"]))[:, 1]
    thr = threshold_for_recall(val["label"], p_val, cfg.target_recall)

    print_report("Baseline TF-IDF – test @0.5", report(test["label"], p_test, 0.5))
    print_report(f"Baseline TF-IDF – test @seuil rappel {cfg.target_recall}",
                 report(test["label"], p_test, thr))
    synth = cfg.processed_dir / "test_synth.csv"
    if synth.exists():   # emails transactionnels, marques et formulations jamais vues
        s = pd.read_csv(synth)
        p_s = clf.predict_proba(vec.transform(s["text"]))[:, 1]
        print_report(f"Baseline TF-IDF – test_synth (transactionnel) @seuil {thr:.3f}",
                     report(s["label"], p_s, thr))


if __name__ == "__main__":
    main()
