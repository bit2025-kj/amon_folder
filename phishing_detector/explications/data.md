# data.py, ligne par ligne

**Rôle :** transformer un fichier CSV d'emails en tenseurs PyTorch, c'est-à-dire en tableaux de nombres que le réseau peut lire :
- les numéros de tokens ;
- le masque ;
- les 12 indices standardisés ;
- les étiquettes.

Utilisé par : `train.py` (entraînement et évaluation), `detector.py` et `app.py` (prédiction).

---

## Lignes 1 à 8 : les imports

```python
1  """Chargement des splits en tenseurs PyTorch (tokens + masque + features + label)."""
2  import numpy as np
3  import pandas as pd
4  import torch
5  from tokenizers import Tokenizer
6  from torch.utils.data import TensorDataset
7
8  from text_utils import extract_features
```

- **Ligne 2** : `numpy` sert aux calculs sur les indices.
- **Ligne 3** : `pandas` lit les fichiers CSV.
- **Ligne 4** : `torch` est PyTorch, la bibliothèque de réseaux de neurones.
- **Ligne 5** : le type `Tokenizer` de la bibliothèque `tokenizers` (Hugging Face).
- **Ligne 6** : `TensorDataset` regroupe plusieurs tenseurs de même longueur. Son élément n° i est le n-uplet (ids[i], mask[i], feats[i], label[i]).
- **Ligne 8** : la fonction qui calcule les 12 indices d'un email.

## Lignes 11 à 15 : `encode_texts()`

```python
11 def encode_texts(texts: list[str], tokenizer: Tokenizer):
12     enc = tokenizer.encode_batch(texts)
13     ids = torch.tensor([e.ids for e in enc], dtype=torch.long)
14     mask = torch.tensor([e.attention_mask for e in enc], dtype=torch.long)
15     return ids, mask
```

- **Ligne 12** : découpe tous les textes en tokens d'un coup. `encode_batch` utilise plusieurs cœurs du processeur. Le tokenizer est configuré (dans `prepare_data.py`) pour tronquer et compléter à 256 tokens exactement.
- **Ligne 13** : `e.ids` est la liste des 256 numéros de tokens d'un email. Le résultat est un tenseur d'entiers (`long`) de forme N × 256. Le remplissage vaut 0 (`[PAD]`).
- **Ligne 14** : `e.attention_mask` vaut 1 pour un vrai token et 0 pour le remplissage, aussi de forme N × 256. Le réseau s'en sert pour ignorer le remplissage.
- **Ligne 15** : renvoie les deux tenseurs.

Exemple : « verify your password » donne ids = [num(verify), num(your), num(password), 0, 0, …] et mask = [1, 1, 1, 0, 0, …].

## Lignes 18 et 19 : `feature_matrix()`

```python
18 def feature_matrix(texts: list[str]) -> np.ndarray:
19     return np.stack([extract_features(t) for t in texts])
```

Calcule les 12 indices de chaque email, puis empile les vecteurs (`np.stack`) en une matrice de N lignes et 12 colonnes.

## Ligne 22 : la borne

```python
22 FEAT_CLIP = 5.0
```

La valeur maximale, en valeur absolue, d'un indice standardisé. Elle est définie comme constante pour être modifiable à un seul endroit.

## Lignes 25 à 29 : `feature_stats()`

```python
25 def feature_stats(feats: np.ndarray):
26     mean, std = feats.mean(0), feats.std(0)
27     # Une feature jamais vue (écart-type ~0) exploserait à la division : plancher à 1
28     std = np.where(std < 1e-3, 1.0, std)
29     return mean.astype(np.float32), std.astype(np.float32)
```

- **Ligne 26** : la moyenne et l'écart-type de chaque colonne. Le `0` veut dire « le long des lignes », on obtient donc 12 moyennes et 12 écarts-types. Cette fonction n'est appelée que sur le **train**.
- **Ligne 28** : `np.where(condition, a, b)` choisit `a` là où la condition est vraie et `b` ailleurs. Un écart-type presque nul, par exemple celui d'un indice qui vaut toujours 0 dans le train, est remplacé par 1. Cette ligne corrige un vrai bug rencontré pendant le projet : avec un écart-type proche de 0, un email où cet indice valait 1 donnait (1 − 0) / 0,000001 = 1 000 000, et l'entraînement explosait au milieu d'une époque.
- **Ligne 29** : convertit en nombres 32 bits, le format de PyTorch.

## Lignes 32 à 36 : `standardize()`

```python
32 def standardize(feats, mean, std):
33     """(x - moyenne) / écart-type, borné à ±5 pour qu'une valeur extrême
34     ne domine pas toute la décision. Marche avec numpy ou torch."""
35     z = (feats - mean) / std
36     return z.clip(-FEAT_CLIP, FEAT_CLIP)
```

- **Ligne 35** : centre et réduit chaque indice. Après cela, chaque indice a en moyenne 0 et un écart-type de 1 sur le train. Sans cette étape, `log_chars` (environ 7) écraserait `has_ip_url` (0 ou 1) simplement parce que ses valeurs sont plus grandes. La soustraction et la division s'appliquent colonne par colonne grâce au *broadcasting* : un vecteur de 12 valeurs est appliqué à chaque ligne.
- **Ligne 36** : borne chaque valeur entre −5 et +5, pour qu'un email extrême (300 liens) ne domine pas la décision. `.clip` existe à la fois en numpy et en PyTorch, donc la fonction marche avec les deux.

## Lignes 39 à 44 : `load_split()`

```python
39 def load_split(cfg, name: str, tokenizer: Tokenizer, feat_mean=None, feat_std=None):
40     """Retourne (TensorDataset, feat_mean, feat_std).
41
42     La standardisation des features utilise la moyenne / l'écart-type du TRAIN
43     (passés en argument pour val/test).
44     """
```

- **Ligne 39** : charge un jeu (`name` = `"train"`, `"val"`, `"test"` ou `"test_synth"`). Les deux derniers arguments sont vides pour le train et reçoivent les statistiques du train pour les autres jeux.
- **Lignes 40 à 44** : la docstring rappelle la règle importante : la validation et le test sont standardisés avec les statistiques du train. Calculer une moyenne sur le test donnerait au modèle une information sur le test.

## Lignes 45 à 51 : lire le fichier

```python
45     path = cfg.processed_dir / f"{name}.csv"
46     if not path.exists():
47         raise SystemExit(f"{path} introuvable : lance d'abord  python prepare_data.py")
48     df = pd.read_csv(path)
49     if cfg.limit:
50         df = df.sample(n=min(cfg.limit, len(df)), random_state=cfg.seed)
51     texts = df["text"].astype(str).tolist()
```

- **Ligne 45** : construit le chemin, par exemple `data/processed/train.csv`. Le `f"..."` insère la valeur de `name` dans la chaîne.
- **Lignes 46 et 47** : si le fichier manque, on arrête avec un message qui dit quoi faire, plutôt qu'une erreur obscure.
- **Ligne 48** : lit le CSV dans un tableau pandas, avec les colonnes `text`, `label` et parfois `source`.
- **Lignes 49 et 50** : en mode debug (`limit` = 256), on tire 256 emails au hasard, toujours les mêmes grâce à `random_state`.
- **Ligne 51** : la colonne de texte devient une liste Python de chaînes. `astype(str)` protège contre une case vide.

## Lignes 52 à 57 : tokens et indices

```python
52     ids, mask = encode_texts(texts, tokenizer)
53
54     feats = feature_matrix(texts)
55     if feat_mean is None:
56         feat_mean, feat_std = feature_stats(feats)
57     feats = torch.tensor(standardize(feats, feat_mean, feat_std), dtype=torch.float32)
```

- **Ligne 52** : les tokens et le masque.
- **Ligne 54** : les 12 indices bruts de chaque email.
- **Lignes 55 et 56** : si aucune statistique n'a été fournie (cas du train), on les calcule ici.
- **Ligne 57** : standardise puis convertit en tenseur PyTorch de nombres à virgule.

## Lignes 59 et 60 : étiquettes et résultat

```python
59     labels = torch.tensor(df["label"].values, dtype=torch.float32)
60     return TensorDataset(ids, mask, feats, labels), feat_mean, feat_std
```

- **Ligne 59** : les étiquettes 0 ou 1, en nombres à virgule, car la fonction de perte `BCEWithLogitsLoss` les attend sous cette forme.
- **Ligne 60** : renvoie le jeu de données et les statistiques. `train.py` récupère celles du train pour les passer à la validation et au test, puis pour les enregistrer dans le livrable.
