# predict.py, ligne par ligne

**Rôle :** utiliser le modèle livré depuis le terminal, sans rien réentraîner. On peut lui donner un email en argument, un fichier, ou rien du tout pour le mode interactif.

Commande : `python predict.py "texte de l'email"`, après `train.py`.

---

## Lignes 1 à 9 : la docstring

```python
1  """Étape 4 : utiliser le livrable (aucun entraînement nécessaire).
2
3  Usage :
4      python predict.py "Your account has been suspended, verify your password at http://..."
5      python predict.py --file email.txt
6      python predict.py                      # mode interactif : colle un email, puis FIN
7      python predict.py --debug "..."        # montre tokens, features, logit
8      python predict.py --model autre.pt ... # charger un autre fichier modèle
9  """
```

Le mode d'emploi, avec les cinq façons de lancer le script.

## Lignes 10 à 14 : les imports

```python
10 import argparse
11 from pathlib import Path
12
13 from config import Config
14 from detector import PhishingDetector, format_result, interactive
```

- **Ligne 10** : `argparse` lit les arguments de la ligne de commande et génère l'aide (`python predict.py --help`).
- **Ligne 11** : `Path` pour vérifier l'existence du modèle et lire un fichier.
- **Ligne 13** : la configuration, pour connaître le chemin par défaut du modèle.
- **Ligne 14** : depuis `detector.py`, la classe qui charge et utilise le modèle, la fonction qui met en forme le résultat, et la boucle interactive.

## Lignes 17 à 23 : les arguments

```python
17 def main():
18     ap = argparse.ArgumentParser()
19     ap.add_argument("text", nargs="?", help="texte de l'email")
20     ap.add_argument("--file", help="fichier texte contenant l'email")
21     ap.add_argument("--model", default=str(Config().model_path), help="fichier .pt du modèle")
22     ap.add_argument("--debug", action="store_true", help="afficher ce que voit le modèle")
23     args = ap.parse_args()
```

- **Ligne 18** : crée le lecteur d'arguments.
- **Ligne 19** : un argument positionnel, le texte de l'email. `nargs="?"` le rend facultatif : sans lui, `args.text` vaut `None`.
- **Ligne 20** : `--file email.txt` pour lire l'email dans un fichier, plus pratique pour un long email.
- **Ligne 21** : `--model` choisit le fichier du modèle. Par défaut, c'est `artifacts/phishing_model.pt`. Exemple : `--model artifacts/v1/phishing_model.pt` pour tester l'ancienne version.
- **Ligne 22** : `--debug` est un interrupteur : `action="store_true"` le met à `True` s'il est présent, à `False` sinon.
- **Ligne 23** : lit la ligne de commande et range les valeurs dans `args`.

## Lignes 25 à 30 : charger le modèle

```python
25     if not Path(args.model).exists():
26         raise SystemExit(f"{args.model} introuvable : entraîne d'abord le modèle (python train.py)")
27     det = PhishingDetector.load(args.model)
28     if det.metrics:
29         print(f"Modèle chargé (test : rappel {det.metrics.get('recall', 0):.3f}, "
30               f"précision {det.metrics.get('precision', 0):.3f})")
```

- **Lignes 25 et 26** : si le fichier n'existe pas, on s'arrête avec un message qui explique quoi faire.
- **Ligne 27** : charge le livrable. Le modèle, le tokenizer, la standardisation et le seuil sont reconstruits à partir de ce seul fichier, en moins d'une seconde.
- **Lignes 28 à 30** : si le fichier contient les scores du test, on rappelle le rappel et la précision. `.get('recall', 0)` renvoie 0 si la clé manque. `:.3f` affiche 3 décimales.

## Lignes 32 à 37 : analyser l'email

```python
32     raw = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
33     if raw is None:
34         interactive(det, debug=args.debug)
35         return
36     r = det.explain(raw) if args.debug else det.predict(raw)
37     print(format_result(r))
```

- **Ligne 32** : si `--file` est donné, on lit le fichier en UTF-8 (accents et symboles). Sinon, on prend le texte passé en argument.
- **Lignes 33 à 35** : si ni texte ni fichier n'est fourni, on lance le mode interactif : on colle un email, on tape `FIN` pour l'analyser, et `Q` pour quitter. `return` termine `main()`.
- **Ligne 36** : en mode debug, `explain()` affiche tout ce que voit le modèle (tokens, indices, logit) avant de renvoyer le résultat. Sinon, `predict()` renvoie seulement le résultat.
- **Ligne 37** : affiche par exemple `P(phishing) = 0.998  (seuil 0.881)  ->  PHISHING`.

## Lignes 40 et 41 : le point d'entrée

```python
40 if __name__ == "__main__":
41     main()
```

`main()` ne s'exécute que si le fichier est lancé directement, pas s'il est importé.
