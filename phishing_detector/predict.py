"""Étape 4 : utiliser le livrable (aucun entraînement nécessaire).

Usage :
    python predict.py "Your account has been suspended, verify your password at http://..."
    python predict.py --file email.txt
    python predict.py                      # mode interactif : colle un email, puis FIN
    python predict.py --debug "..."        # montre tokens, features, logit
    python predict.py --model autre.pt ... # charger un autre fichier modèle
"""
import argparse
from pathlib import Path

from config import Config
from detector import PhishingDetector, format_result, interactive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="texte de l'email")
    ap.add_argument("--file", help="fichier texte contenant l'email")
    ap.add_argument("--model", default=str(Config().model_path), help="fichier .pt du modèle")
    ap.add_argument("--debug", action="store_true", help="afficher ce que voit le modèle")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"{args.model} introuvable : entraîne d'abord le modèle (python train.py)")
    det = PhishingDetector.load(args.model)
    if det.metrics:
        print(f"Modèle chargé (test : rappel {det.metrics.get('recall', 0):.3f}, "
              f"précision {det.metrics.get('precision', 0):.3f})")

    raw = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if raw is None:
        interactive(det, debug=args.debug)
        return
    r = det.explain(raw) if args.debug else det.predict(raw)
    print(format_result(r))


if __name__ == "__main__":
    main()
