# prepare_data.py, ligne par ligne

**Rôle :** première étape du projet. Le script :
- lit le dataset brut, le nettoie, le normalise et retire les doublons ;
- le découpe en train, validation et test ;
- ajoute les emails synthétiques et le bruit (v2) ;
- entraîne le tokenizer BPE.

Commandes : `python prepare_data.py`, ou `python prepare_data.py --no-augment` pour retrouver les données de la v1.

Résultat : 18 650 emails bruts deviennent 17 495 emails uniques. En v2, le train compte 17 996 emails, la validation 2 249, le test 1 750 et le test transactionnel 899.

---

## Lignes 1 à 9 : la docstring

```python
1  """Étape 1 : nettoyage, déduplication, découpage train/val/test, augmentation, tokenizer.
...
7  Le découpage du dataset d'origine est identique dans les deux cas (même graine) :
8  test.csv reste comparable d'une version à l'autre.
9  """
```

Résume les étapes et les deux modes. Point important : avec ou sans augmentation, le découpage du dataset d'origine est le même. Le test de la v1 et celui de la v2 sont donc **exactement les mêmes emails**, et la comparaison entre les deux versions est juste.

## Lignes 10 à 19 : les imports

```python
10 import argparse
11 import random
12
13 import pandas as pd
14 from sklearn.model_selection import train_test_split
15 from tokenizers import Tokenizer, models, pre_tokenizers, trainers
16
17 from augment import add_noise, synthetic
18 from config import Config
19 from text_utils import normalize
```

- **Ligne 10** : lit l'option `--no-augment`.
- **Ligne 11** : `random` est le générateur de nombres aléatoires de Python, utilisé pour le bruit.
- **Ligne 13** : `pandas` pour manipuler le tableau d'emails.
- **Ligne 14** : `train_test_split` découpe un tableau en deux parties.
- **Ligne 15** : les briques de la bibliothèque `tokenizers` : le tokenizer, son modèle (BPE), son pré-découpage et son entraîneur.
- **Ligne 17** : le générateur d'emails synthétiques et la fonction de bruit, définis dans `augment.py`.
- **Lignes 18 et 19** : la configuration et la normalisation.

## Lignes 22 à 26 : `synth_df()`

```python
22 def synth_df(n_legit, n_phish, split, seed, cfg):
23     df = pd.DataFrame(synthetic(n_legit, n_phish, split, seed), columns=["text", "label"])
24     df["text"] = df["text"].map(lambda t: normalize(t, cfg.max_raw_chars))
25     df["source"] = "synthétique"
26     return df.drop_duplicates(subset="text")
```

- **Ligne 22** : fabrique un tableau d'emails synthétiques. `split` vaut `"train"` ou `"test"` et choisit les marques et formulations autorisées. `seed` rend la génération reproductible.
- **Ligne 23** : `synthetic()` renvoie une liste de paires (texte, étiquette), transformée en tableau à deux colonnes.
- **Ligne 24** : normalise chaque texte exactement comme les emails réels. `map` applique la fonction à chaque case, et `lambda t: ...` définit cette fonction en une ligne.
- **Ligne 25** : ajoute une colonne `source` pour tracer l'origine de chaque email.
- **Ligne 26** : retire les doublons, car deux tirages au hasard peuvent produire le même email.

## Lignes 29 à 37 : `augment()`, générer les emails synthétiques

```python
29 def augment(train, val, test, cfg):
30     """Ajoute les emails synthétiques et remplace une partie du train par une version bruitée."""
31     orig_texts = set(train["text"]) | set(val["text"]) | set(test["text"])
32     s_train = synth_df(*cfg.synth_train, "train", cfg.seed + 1, cfg)
33     s_val = synth_df(*cfg.synth_val, "train", cfg.seed + 2, cfg)
34     s_test = synth_df(*cfg.synth_test, "test", cfg.seed + 3, cfg)
35     # Aucun email synthétique en double entre les splits
36     s_val = s_val[~s_val["text"].isin(s_train["text"])]
37     s_test = s_test[~s_test["text"].isin(set(s_train["text"]) | set(s_val["text"]) | orig_texts)]
```

- **Ligne 31** : l'ensemble de tous les textes réels. `|` réunit des ensembles Python.
- **Ligne 32** : 2 400 légitimes et 1 600 phishing pour le train. `*cfg.synth_train` déplie le couple (2400, 1600) en deux arguments.
- **Ligne 33** : 300 légitimes et 200 phishing pour la validation, avec les mêmes formulations autorisées que le train (`"train"`) mais une autre graine, donc d'autres combinaisons.
- **Ligne 34** : 500 légitimes et 400 phishing pour le test transactionnel, construits avec les marques et formulations **réservées** (`"test"`), jamais utilisées à l'entraînement.
- **Ligne 36** : retire de la validation tout email déjà présent dans le train synthétique. `~` inverse le masque : on garde ce qui n'est **pas** dans le train.
- **Ligne 37** : retire du test tout email présent ailleurs. Il reste 899 emails. Un même email dans le train et le test fausserait la mesure.

## Lignes 39 et 40 : ajouter les synthétiques

```python
39     train = pd.concat([train.assign(source="dataset"), s_train], ignore_index=True)
40     val = pd.concat([val.assign(source="dataset"), s_val], ignore_index=True)
```

- **Lignes 39 et 40** : marque les emails réels avec `source = "dataset"`, puis les met bout à bout avec les synthétiques (`concat`). `ignore_index=True` renumérote les lignes de 0 à N − 1. Le test d'origine n'est pas modifié.

## Lignes 42 à 48 : le bruit

```python
42     # Bruit : on REMPLACE noise_frac des emails du train (les 2 classes, même proportion)
43     rng = random.Random(cfg.seed)
44     idx = train.sample(frac=cfg.noise_frac, random_state=cfg.seed).index
45     train.loc[idx, "text"] = [normalize(add_noise(t, rng), cfg.max_raw_chars) for t in train.loc[idx, "text"]]
46     train.loc[idx, "source"] = train.loc[idx, "source"] + " + bruit"
47     train = train[train["text"].str.len() >= 20].drop_duplicates(subset="text")
48     train = train.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
```

- **Ligne 43** : un générateur aléatoire propre à cette étape, initialisé avec la graine, donc reproductible.
- **Ligne 44** : tire au hasard 30 % des lignes du train (5 399 emails), toutes classes confondues : le bruit touche le phishing et les emails légitimes dans la même proportion. `.index` récupère leurs numéros de ligne.
- **Ligne 45** : remplace le texte de chacune de ces lignes par sa version bruitée, renormalisée ensuite. `train.loc[lignes, colonne]` désigne des cases précises du tableau.
- **Ligne 46** : met à jour la colonne `source`, par exemple « dataset + bruit », pour garder la trace.
- **Ligne 47** : le bruit peut raccourcir un email (troncature, mots supprimés) ou créer un doublon. On retire les textes de moins de 20 caractères et les doublons.
- **Ligne 48** : mélange toutes les lignes (`frac=1.0` = 100 % des lignes, dans un ordre aléatoire) pour que les synthétiques ne soient pas regroupés à la fin. `reset_index(drop=True)` renumérote.

La validation et les tests ne sont **jamais bruités** : ils mesurent le modèle sur des emails propres.

## Lignes 50 à 55 : le résumé

```python
50     print("\nAugmentation :")
51     print(f"  train : +{len(s_train)} synthétiques, {len(idx)} emails bruités")
52     print(f"  val   : +{len(s_val)} synthétiques")
53     print(f"  test_synth : {len(s_test)} emails (marques et formulations jamais vues)")
54     print(train["source"].value_counts().to_string())
55     return train, val, s_test
```

- **Lignes 50 à 54** : affiche ce qui a été ajouté, puis le nombre d'emails par source (`value_counts`) : dataset, dataset + bruit, synthétique, synthétique + bruit.
- **Ligne 55** : renvoie le nouveau train, la nouvelle validation et le test transactionnel.

## Lignes 58 à 65 : lire le dataset

```python
58 def main():
59     ap = argparse.ArgumentParser()
60     ap.add_argument("--no-augment", action="store_true", help="dataset d'origine seul")
61     args = ap.parse_args()
62     cfg = Config()
63     df = pd.read_csv(cfg.raw_csv)
64     df = df.rename(columns={"Email Text": "text", "Email Type": "label"})[["text", "label"]]
65     n0 = len(df)
```

- **Lignes 59 à 61** : lit l'option `--no-augment` de la ligne de commande.
- **Ligne 63** : lit le CSV brut, `data/raw/Phishing_Email.csv`.
- **Ligne 64** : renomme les colonnes d'origine en noms courts, puis ne garde que ces deux colonnes, car le fichier contient aussi une colonne d'index inutile.
- **Ligne 65** : retient le nombre d'emails de départ (18 650) pour l'affichage.

## Lignes 67 à 70 : nettoyer

```python
67     df = df.dropna(subset=["text"])
68     df["text"] = df["text"].map(lambda t: normalize(t, cfg.max_raw_chars))
69     df = df[df["text"].str.len() >= 20]                 # emails vides / inutilisables
70     df["label"] = (df["label"] == "Phishing Email").astype(int)
```

- **Ligne 67** : retire les lignes sans texte (`NaN`, la valeur vide de pandas).
- **Ligne 68** : normalise chaque email : coupure à 20 000 caractères, minuscules, ponctuation isolée, espaces fusionnés.
- **Ligne 69** : retire les emails de moins de 20 caractères, trop courts pour contenir une information exploitable.
- **Ligne 70** : convertit l'étiquette texte en nombre. La comparaison donne vrai ou faux, et `astype(int)` le transforme en 1 (phishing) ou 0 (légitime).

## Lignes 72 à 76 : retirer les doublons

```python
72     # Déduplication APRÈS normalisation : sinon un même email peut se retrouver
73     # en train ET en test, ce qui gonfle artificiellement les scores.
74     df = df.drop_duplicates(subset="text")
75     print(f"{n0} emails bruts -> {len(df)} après nettoyage/déduplication")
76     print(df["label"].value_counts().rename({0: "safe", 1: "phishing"}).to_string())
```

- **Ligne 74** : garde un seul exemplaire de chaque texte identique. Elle est placée **après** la normalisation exprès : « URGENT: Verify » et « urgent : verify » deviennent identiques et sont alors détectés comme doublons. Avant la normalisation, ils resteraient deux emails distincts et pourraient tomber l'un dans le train, l'autre dans le test. Le modèle serait alors testé sur un email qu'il a déjà vu. Cette étape retire 1 155 emails.
- **Ligne 75** : affiche `18650 emails bruts -> 17495 après nettoyage/déduplication`.
- **Ligne 76** : affiche le nombre d'emails par classe : 10 973 légitimes et 6 522 phishing.

## Lignes 78 à 82 : découper en trois jeux

```python
78     # Découpage stratifié 80 / 10 / 10
79     train, tmp = train_test_split(df, test_size=cfg.val_frac + cfg.test_frac,
80                                   stratify=df["label"], random_state=cfg.seed)
81     val, test = train_test_split(tmp, test_size=cfg.test_frac / (cfg.val_frac + cfg.test_frac),
82                                  stratify=tmp["label"], random_state=cfg.seed)
```

- **Lignes 79 et 80** : premier découpage, 80 % pour le train et 20 % mis de côté (`tmp`). `stratify` garde la même proportion de phishing (environ 37 %) des deux côtés. `random_state=42` rend le découpage identique à chaque exécution.
- **Lignes 81 et 82** : coupe ces 20 % en deux moitiés. 0,10 / 0,20 = 0,5 : 10 % pour la validation et 10 % pour le test, toujours stratifiés.

Résultat : 13 996 / 1 749 / 1 750 emails. Le test contient 652 phishing.

## Lignes 84 à 95 : augmenter puis écrire les fichiers

```python
84     cfg.processed_dir.mkdir(parents=True, exist_ok=True)
85     splits = [("train", train), ("val", val), ("test", test)]
86     synth_path = cfg.processed_dir / "test_synth.csv"
87     if args.no_augment:
88         synth_path.unlink(missing_ok=True)
89     else:
90         train, val, s_test = augment(train, val, test, cfg)
91         splits = [("train", train), ("val", val), ("test", test), ("test_synth", s_test)]
92     print()
93     for name, part in splits:
94         part.to_csv(cfg.processed_dir / f"{name}.csv", index=False)
95         print(f"{name:5s}: {len(part):6d} emails  ({part['label'].mean():.1%} phishing)")
```

- **Ligne 84** : crée le dossier `data/processed`, y compris les dossiers parents (`parents=True`). Ne fait rien s'il existe déjà (`exist_ok=True`).
- **Ligne 85** : la liste des fichiers à écrire, au départ les trois jeux de la v1.
- **Lignes 86 à 88** : avec `--no-augment`, on supprime un éventuel ancien `test_synth.csv`, pour que `baseline.py` et `train.py` ne le mesurent pas par erreur.
- **Lignes 89 à 91** : sinon, on lance l'augmentation et on ajoute le test transactionnel à la liste.
- **Lignes 93 à 95** : écrit chaque jeu en CSV, sans la colonne d'index (`index=False`), et affiche sa taille et sa part de phishing. La moyenne des étiquettes 0/1 est exactement cette part, et `.1%` l'affiche en pourcentage avec une décimale.

## Lignes 97 à 102 : entraîner le tokenizer BPE

```python
97     # Tokenizer BPE entraîné UNIQUEMENT sur le train (pas de fuite d'information)
98     tok = Tokenizer(models.BPE(unk_token="[UNK]"))
99     tok.pre_tokenizer = pre_tokenizers.Whitespace()
100    trainer = trainers.BpeTrainer(vocab_size=cfg.vocab_size, min_frequency=2,
101                                  special_tokens=["[PAD]", "[UNK]"])
102    tok.train_from_iterator(train["text"].tolist(), trainer=trainer)
```

- **Ligne 98** : crée un tokenizer de type **BPE** (*Byte Pair Encoding*). Tout caractère inconnu sera remplacé par le token `[UNK]`.
- **Ligne 99** : avant le BPE, le texte est pré-découpé sur les espaces et la ponctuation (`Whitespace`). Un token ne peut donc jamais chevaucher deux mots.
- **Lignes 100 et 101** : les règles d'apprentissage. Le vocabulaire monte jusqu'à 8 000 tokens, une fusion doit apparaître au moins 2 fois, et deux tokens spéciaux sont réservés : `[PAD]` (numéro 0, le remplissage) et `[UNK]` (numéro 1, l'inconnu).
- **Ligne 102** : l'apprentissage. Le BPE part des caractères seuls, compte toutes les paires de symboles voisins, fusionne la paire la plus fréquente en un nouveau symbole, et recommence jusqu'à 8 000 symboles. Les mots fréquents (« password ») deviennent un seul token, et les mots rares se découpent en morceaux connus (« unverifiable » peut donner « un » + « verif » + « iable »). Il se fait **sur le train uniquement**, pour que le vocabulaire ne contienne rien d'appris sur le test.

## Lignes 103 à 107 : longueur fixe et sauvegarde

```python
103    tok.enable_truncation(cfg.max_len)
104    tok.enable_padding(pad_id=0, pad_token="[PAD]", length=cfg.max_len)
105    cfg.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
106    tok.save(str(cfg.tokenizer_path))
107    print(f"Tokenizer BPE : {tok.get_vocab_size()} tokens -> {cfg.tokenizer_path}")
```

- **Ligne 103** : tout email de plus de 256 tokens est coupé : seuls les 256 premiers sont gardés.
- **Ligne 104** : tout email plus court est complété avec `[PAD]` (numéro 0) jusqu'à 256. Tous les emails ont ainsi la même longueur et peuvent être empilés dans un même tenseur.
- **Ligne 105** : crée le dossier `artifacts` si besoin.
- **Ligne 106** : enregistre le tokenizer, réglages de longueur compris, dans `artifacts/tokenizer.json`.
- **Ligne 107** : confirme la taille du vocabulaire (8 000).

## Lignes 110 et 111 : le point d'entrée

```python
110 if __name__ == "__main__":
111     main()
```

`main()` ne s'exécute que si le fichier est lancé directement.
