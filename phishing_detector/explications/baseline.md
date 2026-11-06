# baseline.py, ligne par ligne

**Rôle :** entraîner un modèle de référence simple et rapide (TF-IDF + régression logistique) et afficher ses scores. Le Transformer n'a d'intérêt que s'il fait mieux. Il s'exécute en moins d'une minute.

Commande : `python baseline.py`, après `prepare_data.py`.

---

## Lignes 1 à 11 : docstring et imports

```python
1  """Étape 2 : modèle de référence TF-IDF + régression logistique.
2
3  Le Transformer n'a d'intérêt que s'il bat ce score (surtout le rappel).
4  Usage :  python baseline.py
5  """
6  import pandas as pd
7  from sklearn.feature_extraction.text import TfidfVectorizer
8  from sklearn.linear_model import LogisticRegression
9
10 from config import Config
11 from metrics import print_report, report, threshold_for_recall
```

- **Lignes 1 à 5** : ce que fait le script et comment le lancer.
- **Ligne 6** : `pandas` pour lire les CSV.
- **Ligne 7** : `TfidfVectorizer` transforme des textes en vecteurs TF-IDF.
- **Ligne 8** : `LogisticRegression` est le classifieur linéaire.
- **Lignes 10 et 11** : la configuration du projet et les fonctions de mesure partagées avec le Transformer, pour que les deux modèles soient mesurés exactement de la même façon.

## Lignes 14 à 16 : charger les données

```python
14 def main():
15     cfg = Config()
16     train, val, test = (pd.read_csv(cfg.processed_dir / f"{n}.csv") for n in ("train", "val", "test"))
```

- **Ligne 15** : charge la configuration (chemins, graine, rappel visé).
- **Ligne 16** : lit les trois fichiers préparés. L'expression entre parenthèses produit les trois tableaux un par un, et Python les répartit dans `train`, `val` et `test`. Les textes sont déjà normalisés par `prepare_data.py`.

## Lignes 18 et 19 : TF-IDF

```python
18     vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)
19     Xtr = vec.fit_transform(train["text"])
```

- **Ligne 18** : configure le vectoriseur.
  - `ngram_range=(1, 2)` : prend les mots seuls **et** les paires de mots consécutifs. « verify your » devient une colonne à part entière, plus parlante que « verify » et « your » séparément.
  - `min_df=2` : ignore les mots qui n'apparaissent que dans un seul email (fautes, numéros uniques).
  - `max_features=50_000` : garde au plus les 50 000 termes les plus fréquents.
  - `sublinear_tf=True` : remplace le nombre d'occurrences par 1 + log(occurrences), pour qu'un mot répété 20 fois ne pèse pas 20 fois plus.
- **Ligne 19** : `fit` apprend le vocabulaire et les poids IDF **sur le train seul**, puis `transform` convertit chaque email du train en vecteur. Le résultat est une matrice creuse : presque toutes les cases valent 0, car un email ne contient qu'une petite partie du vocabulaire.

Pour un terme t dans un email d : TF-IDF(t, d) = TF(t, d) × IDF(t). TF mesure si le mot est fréquent dans l'email, IDF s'il est rare dans le corpus. « password » pèse donc plus que « the ».

## Lignes 20 et 21 : la régression logistique

```python
20     clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
21     clf.fit(Xtr, train["label"])
```

- **Ligne 20** : configure le classifieur.
  - `max_iter=2000` : le nombre maximal d'itérations de l'optimisation. Avec 50 000 colonnes, la valeur par défaut (100) ne suffit pas toujours.
  - `C=4.0` : l'inverse de la force de régularisation. Plus `C` est grand, moins les poids sont contraints.
  - `class_weight="balanced"` : donne plus de poids à la classe minoritaire (le phishing), comme `pos_weight` pour le Transformer.
- **Ligne 21** : apprend un poids par terme. Un poids positif pousse vers phishing, un poids négatif vers légitime. La prédiction vaut sigmoïde(somme des poids × valeurs TF-IDF + biais).

Ce modèle ne voit pas l'ordre des mots, seulement leur présence.

## Lignes 23 à 25 : prédire et choisir le seuil

```python
23     p_val = clf.predict_proba(vec.transform(val["text"]))[:, 1]
24     p_test = clf.predict_proba(vec.transform(test["text"]))[:, 1]
25     thr = threshold_for_recall(val["label"], p_val, cfg.target_recall)
```

- **Lignes 23 et 24** : `vec.transform` convertit la validation et le test avec le vocabulaire **appris sur le train** (sans `fit` : rien n'est appris sur ces jeux). `predict_proba` renvoie deux colonnes (P(légitime), P(phishing)) ; `[:, 1]` garde la seconde.
- **Ligne 25** : choisit sur la validation le seuil qui atteint 97 % de rappel avec la meilleure précision, exactement comme pour le Transformer.

## Lignes 27 à 29 : afficher les scores du test

```python
27     print_report("Baseline TF-IDF – test @0.5", report(test["label"], p_test, 0.5))
28     print_report(f"Baseline TF-IDF – test @seuil rappel {cfg.target_recall}",
29                  report(test["label"], p_test, thr))
```

Deux mesures sur le test : au seuil standard de 0,5, puis au seuil choisi sur la validation. Les deux montrent le compromis entre phishing ratés et fausses alertes.

## Lignes 30 à 35 : le test transactionnel (v2)

```python
30     synth = cfg.processed_dir / "test_synth.csv"
31     if synth.exists():   # emails transactionnels, marques et formulations jamais vues
32         s = pd.read_csv(synth)
33         p_s = clf.predict_proba(vec.transform(s["text"]))[:, 1]
34         print_report(f"Baseline TF-IDF – test_synth (transactionnel) @seuil {thr:.3f}",
35                      report(s["label"], p_s, thr))
```

- **Lignes 30 et 31** : si `prepare_data.py` a créé le test transactionnel, on le mesure aussi. Sinon, en v1 ou avec `--no-augment`, on saute cette étape.
- **Lignes 32 à 35** : même procédure, avec le seuil de la validation, affiché à 3 décimales (`{thr:.3f}`).

## Lignes 38 et 39 : le point d'entrée

```python
38 if __name__ == "__main__":
39     main()
```

`__name__` vaut `"__main__"` uniquement quand le fichier est lancé directement (`python baseline.py`). S'il est importé par un autre fichier, `main()` ne s'exécute pas. Tous les scripts du projet suivent ce modèle.
