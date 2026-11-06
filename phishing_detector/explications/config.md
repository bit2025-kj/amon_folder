# config.py, ligne par ligne

**Rôle :** regrouper au même endroit tous les réglages du projet : chemins des fichiers, tailles du modèle, paramètres d'entraînement. Les autres fichiers ne contiennent aucun nombre « magique » : ils lisent tout ici. Pour changer la taille du modèle ou le nombre d'époques, on ne modifie que ce fichier.

Utilisé par : `prepare_data.py`, `data.py`, `baseline.py`, `train.py`, `detector.py`, `predict.py`, `app.py`.

---

## Lignes 1 à 5 : en-tête et dossier racine

```python
1  """Tous les hyperparamètres du projet, au même endroit."""
2  from dataclasses import dataclass
3  from pathlib import Path
4
5  ROOT = Path(__file__).resolve().parent
```

- **Ligne 1** : la *docstring* du module, une phrase qui décrit le fichier. Python l'ignore à l'exécution, mais elle documente le code.
- **Ligne 2** : importe `dataclass`, un décorateur qui crée automatiquement le constructeur d'une classe à partir de ses attributs. On écrit `Config()` sans avoir à écrire `__init__`.
- **Ligne 3** : importe `Path`, qui manipule les chemins de fichiers de façon portable (Windows, Linux, macOS). L'opérateur `/` assemble les morceaux de chemin.
- **Ligne 5** : `__file__` est le chemin de `config.py`. `.resolve()` le rend absolu, `.parent` donne son dossier. `ROOT` est donc le dossier `phishing_detector`, quel que soit le dossier depuis lequel on lance les commandes.

## Lignes 8 et 9 : la classe

```python
8  @dataclass
9  class Config:
```

- **Ligne 8** : le décorateur `@dataclass` transforme la classe en « classe de données ». Chaque attribut annoté (`nom: type = valeur`) devient un paramètre du constructeur, avec sa valeur par défaut.
- **Ligne 9** : déclare la classe `Config`. Chaque script crée sa propre instance avec `cfg = Config()`.

## Lignes 10 à 14 : les chemins

```python
10     # --- Chemins ---
11     raw_csv: Path = ROOT / "data" / "raw" / "Phishing_Email.csv"
12     processed_dir: Path = ROOT / "data" / "processed"
13     tokenizer_path: Path = ROOT / "artifacts" / "tokenizer.json"
14     run_dir: Path = ROOT / "artifacts"          # checkpoints + livrable
```

- **Ligne 10** : un commentaire qui sépare les sections.
- **Ligne 11** : le dataset brut téléchargé, avant tout traitement.
- **Ligne 12** : le dossier où `prepare_data.py` écrit `train.csv`, `val.csv`, `test.csv` et `test_synth.csv`.
- **Ligne 13** : le fichier du tokenizer BPE entraîné.
- **Ligne 14** : le dossier des résultats d'entraînement : points de reprise, meilleurs poids et livrable final.

## Lignes 16 à 21 : les données

```python
16     # --- Données ---
17     max_raw_chars: int = 20_000      # un email du dataset fait 17 Mo : on coupe
18     val_frac: float = 0.10
19     test_frac: float = 0.10
20     seed: int = 42
21     limit: int | None = None         # nb max d'emails par split (mode debug)
```

- **Ligne 17** : chaque email est coupé à 20 000 caractères. Un email du dataset fait 17 Mo à lui seul ; sans cette limite, la normalisation et les indices seraient très lents. De toute façon, le modèle ne lit que les 256 premiers tokens. `20_000` s'écrit avec un tiret bas pour la lisibilité : c'est le nombre 20000.
- **Lignes 18 et 19** : 10 % des emails pour la validation, 10 % pour le test. Il reste 80 % pour l'entraînement.
- **Ligne 20** : la graine aléatoire. La même graine donne toujours le même découpage, le même ordre des lots et les mêmes emails synthétiques : les résultats sont reproductibles.
- **Ligne 21** : `None` par défaut, c'est-à-dire tous les emails. En mode debug, cette valeur passe à 256 pour aller vite. `int | None` veut dire « un entier ou rien ».

## Lignes 23 à 27 : l'augmentation (v2)

```python
23     # --- Augmentation (voir augment.py) ---
24     synth_train: tuple = (2400, 1600)   # emails transactionnels synthétiques (légitimes, phishing)
25     synth_val: tuple = (300, 200)
26     synth_test: tuple = (500, 400)      # marques et formulations jamais vues -> test_synth.csv
27     noise_frac: float = 0.30            # part des emails du train remplacés par une version bruitée
```

- **Ligne 24** : nombre d'emails synthétiques ajoutés au train, sous la forme (légitimes, phishing).
- **Ligne 25** : même chose pour la validation.
- **Ligne 26** : nombre d'emails du test transactionnel, construit avec des marques et des formulations jamais vues à l'entraînement.
- **Ligne 27** : 30 % des emails du train sont remplacés par une version bruitée (fautes de frappe, transferts, etc.).

## Lignes 29 à 31 : le tokenizer

```python
29     # --- Tokenizer ---
30     vocab_size: int = 8_000
31     max_len: int = 256               # tokens par email (objet + début du corps)
```

- **Ligne 30** : le vocabulaire BPE compte 8 000 tokens. Plus il est grand, plus la table d'embedding est lourde : 8 000 × 128 = 1 024 000 paramètres, soit déjà 55 % du modèle.
- **Ligne 31** : chaque email est tronqué ou complété à 256 tokens. C'est assez pour l'objet et le début du message, où se trouvent presque toujours les indices de phishing. Le coût de l'attention croît avec le carré de cette longueur.

## Lignes 33 à 41 : le modèle

```python
33     # --- Modèle ---
34     d_model: int = 128               # taille des embeddings
35     n_heads: int = 4                 # 4 têtes x 32 dimensions
36     n_layers: int = 4                # blocs Transformer encodeur
37     d_ff: int = 512                  # neurones cachés du feed-forward
38     n_features: int = 12             # features structurelles manuelles
39     head_hidden: int = 64            # neurones de la tête de classification
40     dropout: float = 0.1
41     head_dropout: float = 0.3
```

- **Ligne 34** : chaque token est représenté par un vecteur de 128 nombres, dans tout le réseau.
- **Ligne 35** : l'attention a 4 têtes ; 128 / 4 = 32 dimensions par tête. Chaque tête peut se spécialiser sur un type de relation entre les mots.
- **Ligne 36** : 4 blocs encodeurs empilés.
- **Ligne 37** : le feed-forward de chaque bloc passe de 128 à 512 neurones puis revient à 128. Le rapport de 4 est le standard des Transformers.
- **Ligne 38** : les 12 indices calculés à la main par `text_utils.py`.
- **Ligne 39** : la couche cachée de la tête de classification a 64 neurones.
- **Ligne 40** : pendant l'entraînement, 10 % des valeurs sont mises à zéro au hasard dans les blocs. Cela empêche le réseau de dépendre d'un seul neurone et limite le sur-apprentissage.
- **Ligne 41** : un dropout plus fort, de 30 %, dans la tête. Elle est petite et facile à sur-entraîner.

## Lignes 43 à 50 : l'entraînement

```python
43     # --- Entraînement ---
44     batch_size: int = 32
45     lr: float = 3e-4
46     weight_decay: float = 0.01
47     max_epochs: int = 10
48     patience: int = 2                # early stopping (sur la PR-AUC de validation)
49     grad_clip: float = 1.0
50     save_every: int = 200            # sauvegarde de reprise toutes les N étapes
```

- **Ligne 44** : 32 emails par lot. Les poids sont mis à jour après chaque lot.
- **Ligne 45** : le taux d'apprentissage, 0,0003 : la taille des pas que fait l'optimiseur AdamW. C'est une valeur classique pour un petit Transformer.
- **Ligne 46** : le *weight decay* tire légèrement chaque poids vers zéro à chaque étape. C'est une régularisation.
- **Ligne 47** : au plus 10 passages complets sur le train.
- **Ligne 48** : on arrête si la PR-AUC de validation ne s'améliore pas pendant 2 époques de suite.
- **Ligne 49** : la norme du gradient est plafonnée à 1,0 pour éviter qu'un lot inhabituel provoque un pas énorme.
- **Ligne 50** : un point de reprise est écrit toutes les 200 étapes, en plus de la fin de chaque époque.

## Lignes 52 et 53 : la décision

```python
52     # --- Décision ---
53     target_recall: float = 0.97      # seuil choisi pour rater <= 3 % des phishing
```

- **Ligne 53** : le seuil de décision est choisi sur la validation pour détecter au moins 97 % des phishing. Parmi les seuils qui y parviennent, on prend le plus haut, donc celui qui donne le moins de fausses alertes.

## Lignes 55 à 66 : les fichiers produits

```python
55     # Fichiers produits
56     @property
57     def last_ckpt(self) -> Path:      # pour reprendre l'entraînement
58         return self.run_dir / "checkpoint_last.pt"
59
60     @property
61     def best_weights(self) -> Path:   # meilleurs poids vus sur la validation
62         return self.run_dir / "best_weights.pt"
63
64     @property
65     def model_path(self) -> Path:     # LE LIVRABLE : tout ce qu'il faut pour prédire
66         return self.run_dir / "phishing_model.pt"
```

- **Lignes 56, 60 et 64** : `@property` transforme une méthode en attribut calculé. On écrit `cfg.model_path` sans parenthèses. Le chemin est recalculé à chaque lecture : si `run_dir` change (mode debug), ces trois chemins suivent automatiquement.
- **Lignes 57 et 58** : le point de reprise : poids, état de l'optimiseur, position dans l'entraînement.
- **Lignes 61 et 62** : les poids de la meilleure époque, selon la PR-AUC de validation.
- **Lignes 65 et 66** : le livrable final, chargé par `detector.py`.

## Lignes 68 à 74 : le mode debug

```python
68     def apply_debug(self):
69         """Mini-run pour vérifier que tout le pipeline marche en ~1 minute."""
70         self.run_dir = ROOT / "artifacts" / "debug"   # n'écrase pas le vrai modèle
71         self.limit = 256
72         self.max_epochs = 2
73         self.n_layers = 2
74         self.save_every = 4
```

- **Ligne 68** : méthode appelée par `train.py --debug`. Elle modifie la configuration en place.
- **Ligne 70** : tout est écrit dans `artifacts/debug/`, pour ne jamais écraser le vrai modèle.
- **Ligne 71** : 256 emails par jeu au lieu de plusieurs milliers.
- **Ligne 72** : 2 époques seulement.
- **Ligne 73** : 2 blocs au lieu de 4, donc un calcul deux fois plus rapide.
- **Ligne 74** : un point de reprise toutes les 4 étapes, pour pouvoir tester la reprise rapidement.

## Lignes 76 à 79 : les paramètres du modèle

```python
76     def model_kwargs(self) -> dict:
77         keys = ["vocab_size", "max_len", "d_model", "n_heads", "n_layers", "d_ff",
78                 "n_features", "head_hidden", "dropout", "head_dropout"]
79         return {k: getattr(self, k) for k in keys}
```

- **Ligne 76** : renvoie un dictionnaire des dimensions du modèle.
- **Lignes 77 et 78** : la liste exacte des arguments attendus par `PhishingTransformer`.
- **Ligne 79** : `getattr(self, k)` lit l'attribut nommé `k`. Le résultat est par exemple `{"vocab_size": 8000, "max_len": 256, ...}`. Il sert à trois endroits :
  - créer le modèle : `PhishingTransformer(**cfg.model_kwargs())`, où `**` « déplie » le dictionnaire en arguments nommés ;
  - vérifier, à la reprise, que la configuration n'a pas changé depuis la sauvegarde ;
  - l'enregistrer dans le livrable, pour reconstruire exactement la même architecture au chargement.
