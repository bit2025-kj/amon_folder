# metrics.py, ligne par ligne

**Rôle :** mesurer la qualité d'un modèle (rappel, précision, F1, PR-AUC, erreurs) et choisir le seuil de décision. Les métriques sont pensées pour la sécurité, où rater un phishing coûte plus cher qu'une fausse alerte.

Utilisé par : `baseline.py`, `train.py`.

---

## Lignes 1 à 4 : les imports

```python
1  """Métriques orientées sécurité : le rappel sur le phishing est prioritaire."""
2  import numpy as np
3  from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
4                               precision_recall_curve, precision_score, recall_score)
```

- **Ligne 2** : `numpy` pour les calculs sur tableaux.
- **Lignes 3 et 4** : les fonctions de mesure de scikit-learn. Les parenthèses permettent d'écrire l'import sur deux lignes.

## Lignes 7 à 18 : `report()`

```python
7  def report(y_true, probs, threshold=0.5) -> dict:
8      y_pred = (np.asarray(probs) >= threshold).astype(int)
9      tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
```

- **Ligne 7** : reçoit les vraies classes, les probabilités prédites et un seuil (0,5 par défaut). Renvoie un dictionnaire de scores.
- **Ligne 8** : transforme les probabilités en décisions. Une probabilité supérieure ou égale au seuil donne 1 (phishing), sinon 0. `astype(int)` convertit vrai/faux en 1/0.
- **Ligne 9** : la matrice de confusion compte les quatre cas possibles. `labels=[0, 1]` fixe l'ordre des classes, même si l'une est absente. `.ravel()` aplatit la matrice 2 × 2 en quatre nombres :
  - `tn` (vrais négatifs) : légitimes reconnus comme légitimes ;
  - `fp` (faux positifs) : légitimes classés phishing, les **fausses alertes** ;
  - `fn` (faux négatifs) : phishing classés légitimes, les **phishing ratés** ;
  - `tp` (vrais positifs) : phishing détectés.

```python
10     return {
11         "threshold": round(float(threshold), 4),
12         "pr_auc": average_precision_score(y_true, probs),
13         "recall": recall_score(y_true, y_pred, zero_division=0),
14         "precision": precision_score(y_true, y_pred, zero_division=0),
15         "f1": f1_score(y_true, y_pred, zero_division=0),
16         "phishing_missed (FN)": int(fn),
17         "false_alarms (FP)": int(fp),
18     }
```

- **Ligne 11** : le seuil utilisé, arrondi à 4 décimales.
- **Ligne 12** : la **PR-AUC**, l'aire sous la courbe précision-rappel. Elle se calcule sur les probabilités, pas sur les décisions : elle mesure si le modèle classe bien les phishing au-dessus des emails légitimes, quel que soit le seuil. 1 est parfait. Elle est préférée à la ROC-AUC quand la classe positive est minoritaire.
- **Ligne 13** : le **rappel** = tp / (tp + fn), la part des vrais phishing détectés. `zero_division=0` renvoie 0 au lieu d'une erreur si le dénominateur est nul.
- **Ligne 14** : la **précision** = tp / (tp + fp), la part des alertes qui sont de vrais phishing.
- **Ligne 15** : le **F1** = 2 × précision × rappel / (précision + rappel), la moyenne harmonique. Il est bas dès que l'un des deux est bas.
- **Lignes 16 et 17** : les nombres bruts d'erreurs. « 16 phishing ratés » parle plus à un jury que « rappel 0,9755 ».

L'exactitude (*accuracy*) n'est pas calculée : avec 63 % d'emails légitimes, un modèle qui répond toujours « légitime » aurait déjà 63 %.

## Lignes 21 à 25 : `threshold_for_recall()`

```python
21 def threshold_for_recall(y_true, probs, target_recall: float) -> float:
22     """Le seuil le plus haut (donc la meilleure précision) qui atteint le rappel visé."""
23     precision, recall, thresholds = precision_recall_curve(y_true, probs)
24     ok = np.where(recall[:-1] >= target_recall)[0]
25     return float(thresholds[ok[-1]]) if len(ok) else 0.5
```

- **Ligne 21** : cherche un seuil de décision à partir des vraies classes et des probabilités de la **validation**.
- **Ligne 23** : `precision_recall_curve` essaie tous les seuils possibles et renvoie, pour chacun, la précision et le rappel obtenus. Les seuils sont rangés par ordre croissant. Plus le seuil monte, moins il y a d'alertes : la précision augmente et le rappel baisse.
- **Ligne 24** : garde les positions où le rappel atteint l'objectif (0,97). `recall[:-1]` retire le dernier élément, car scikit-learn renvoie un point de plus pour le rappel que pour les seuils. `np.where(...)[0]` donne les indices qui satisfont la condition.
- **Ligne 25** : `ok[-1]` est le dernier indice valide, donc **le seuil le plus haut** qui atteint encore 97 % de rappel. C'est celui qui donne le moins de fausses alertes sous cette contrainte. Si aucun seuil n'y parvient, on revient à 0,5.

Ce seuil est toujours choisi sur la validation, jamais sur le test : sinon le score de test mesurerait un réglage fait sur lui, et serait optimiste.

## Lignes 28 à 31 : `print_report()`

```python
28 def print_report(title: str, r: dict):
29     print(f"\n== {title} ==")
30     for k, v in r.items():
31         print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
```

- **Ligne 29** : affiche un titre encadré, précédé d'une ligne vide (`\n`).
- **Ligne 30** : parcourt les paires (nom, valeur) du rapport.
- **Ligne 31** : `{k:22s}` aligne les noms sur 22 caractères. Les nombres à virgule sont affichés avec 4 décimales (`{v:.4f}`), et les entiers (nombres d'erreurs) tels quels. C'est l'affichage que l'on voit à la fin de `train.py` et de `baseline.py`.
