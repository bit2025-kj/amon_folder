# model.py, ligne par ligne

**Rôle :** définir le réseau de neurones : un encodeur Transformer écrit de zéro, plus une branche pour les 12 indices manuels. Il reçoit les tokens, le masque et les indices d'un lot d'emails, et renvoie un nombre par email, le **logit**, qui devient une probabilité de phishing après la sigmoïde.

Total : **1 859 201 paramètres**.

Utilisé par : `train.py` (entraînement), `detector.py` (prédiction), `app.py` (explications).

Notation des formes de tenseurs : **B** = nombre d'emails dans le lot, **T** = 256 tokens, **C** = 128 dimensions.

---

## Lignes 1 à 9 : la docstring

```python
1  """Transformer encodeur écrit from scratch + branche de features manuelles.
2
3      tokens (256) -> Embedding token + position (128)
4                   -> 4 x [LayerNorm -> Self-Attention (4 têtes) -> résiduel
5                           LayerNorm -> Feed-forward 128->512->128 -> résiduel]
6                   -> LayerNorm -> mean pooling (on ignore le padding) -> 128
7      features (12) ---------------------------------------------------> concat 140
8                   -> Dense 140 -> 64 (ReLU, dropout) -> Dense 64 -> 1 (logit)
9  """
```

Un schéma en texte de toute l'architecture :
- les 256 tokens deviennent des vecteurs de 128 nombres ;
- ces vecteurs traversent 4 blocs (attention, puis feed-forward) ;
- on fait leur moyenne pour obtenir 128 nombres ;
- on y ajoute les 12 indices, soit 140 nombres ;
- 64 neurones, puis 1 neurone de sortie.

## Lignes 10 à 14 : les imports

```python
10 import math
11
12 import torch
13 import torch.nn as nn
14 import torch.nn.functional as F
```

- **Ligne 10** : `math.sqrt` pour la racine carrée de l'attention.
- **Ligne 12** : PyTorch.
- **Ligne 13** : `nn` contient les couches (`Linear`, `Embedding`, `LayerNorm`, `Dropout`…) et la classe de base `nn.Module`.
- **Ligne 14** : `F` contient les fonctions sans paramètres, ici `softmax`.

---

## Classe `MultiHeadSelfAttention` (lignes 17 à 41)

Le mécanisme qui permet à chaque mot de « regarder » les autres mots de l'email et d'en récupérer de l'information.

### Lignes 17 à 27 : le constructeur

```python
17 class MultiHeadSelfAttention(nn.Module):
18     """Attention bidirectionnelle : chaque token regarde tous les autres (sauf le padding)."""
19
20     def __init__(self, d_model: int, n_heads: int, dropout: float):
21         super().__init__()
22         assert d_model % n_heads == 0
23         self.n_heads, self.head_dim = n_heads, d_model // n_heads
24         self.qkv = nn.Linear(d_model, 3 * d_model)   # Q, K, V en une seule projection
25         self.out = nn.Linear(d_model, d_model)
26         self.attn_drop = nn.Dropout(dropout)
27         self.resid_drop = nn.Dropout(dropout)
```

- **Ligne 17** : hérite de `nn.Module`, la base de toute couche PyTorch. PyTorch retrouve ainsi tous les poids à entraîner.
- **Ligne 18** : « bidirectionnelle » veut dire que chaque mot voit aussi les mots qui le suivent. Ce n'est pas le cas dans GPT, qui ne regarde que le passé pour générer du texte. Pour classer un email entier, on a tout le texte d'un coup.
- **Ligne 21** : appelle le constructeur de `nn.Module`. C'est obligatoire.
- **Ligne 22** : vérifie que 128 se divise bien par le nombre de têtes (4).
- **Ligne 23** : retient 4 têtes de 128 / 4 = 32 dimensions chacune. `//` est la division entière.
- **Ligne 24** : une seule couche linéaire (128 → 384) calcule d'un coup les trois vecteurs de chaque token : **Q** (*query*, ce qu'il cherche), **K** (*key*, ce qu'il offre aux autres), **V** (*value*, son contenu). Une couche au lieu de trois est plus rapide. Paramètres : 128 × 384 + 384 = 49 536.
- **Ligne 25** : la couche de sortie (128 → 128), qui mélange les résultats des 4 têtes. Paramètres : 128 × 128 + 128 = 16 512.
- **Lignes 26 et 27** : deux dropouts, l'un sur les poids d'attention, l'autre sur la sortie. Ils sont actifs seulement pendant l'entraînement.

### Lignes 29 à 33 : préparer Q, K et V

```python
29     def forward(self, x, pad_mask):
30         B, T, C = x.shape
31         q, k, v = self.qkv(x).split(C, dim=2)
32         # (B, T, C) -> (B, têtes, T, head_dim)
33         q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
```

- **Ligne 29** : `forward` est le calcul effectué quand on appelle la couche. `x` est de forme B × 256 × 128, et `pad_mask` de forme B × 256.
- **Ligne 30** : récupère les trois dimensions.
- **Ligne 31** : applique la couche `qkv` (on obtient B × 256 × 384), puis coupe en trois morceaux de 128 le long de la dernière dimension : Q, K et V.
- **Ligne 33** : répartit chaque vecteur de 128 en 4 têtes de 32. `view` redécoupe en B × 256 × 4 × 32, puis `transpose(1, 2)` échange deux dimensions pour obtenir B × 4 × 256 × 32. Chaque tête travaille ainsi indépendamment, en parallèle.

### Lignes 35 à 38 : calculer l'attention

```python
35         scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, têtes, T, T)
36         # pad_mask : 1 = vrai token, 0 = padding -> on interdit de regarder le padding
37         scores = scores.masked_fill(pad_mask[:, None, None, :] == 0, float("-inf"))
38         att = self.attn_drop(F.softmax(scores, dim=-1))
```

- **Ligne 35** : `@` est le produit matriciel. Q × Kᵀ compare la *query* de chaque token avec la *key* de tous les autres. On obtient une matrice 256 × 256 par tête : la case (i, j) dit à quel point le token i s'intéresse au token j. On divise par √32 ≈ 5,66 : sans cela, les scores seraient trop grands et le softmax donnerait 1 à un seul mot et 0 à tous les autres, ce qui bloque l'apprentissage.
- **Ligne 37** : `pad_mask[:, None, None, :]` ajoute deux dimensions pour passer de B × 256 à B × 1 × 1 × 256, afin que le masque s'applique à toutes les têtes et à toutes les lignes. Là où le masque vaut 0 (remplissage), le score devient −∞.
- **Ligne 38** : le softmax transforme chaque ligne de scores en poids positifs qui somment à 1. Comme e^(−∞) = 0, le remplissage reçoit un poids de exactement 0 : aucun mot ne récupère d'information du remplissage.

La formule complète : Attention(Q, K, V) = softmax(Q Kᵀ / √32 + masque) V.

### Lignes 40 et 41 : combiner et recoller

```python
40         y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)       # on recolle les têtes
41         return self.resid_drop(self.out(y))
```

- **Ligne 40** : `att @ v` donne à chaque token une moyenne pondérée des valeurs V de tous les tokens, selon ses poids d'attention. Par exemple, « password » peut récupérer surtout l'information de « verify » et « urgent ». Ensuite, `transpose(1, 2)` revient à B × 256 × 4 × 32, `contiguous()` réorganise la mémoire (exigé par `view`), et `view(B, T, C)` recolle les 4 têtes en un vecteur de 128.
- **Ligne 41** : la couche `out` mélange les informations des 4 têtes, puis le dropout est appliqué. Sortie : B × 256 × 128, la même forme qu'en entrée.

---

## Classe `EncoderBlock` (lignes 44 à 58)

Un bloc du Transformer : attention, puis feed-forward, chacun avec une connexion résiduelle. Il est répété 4 fois.

### Lignes 44 à 53 : le constructeur

```python
44 class EncoderBlock(nn.Module):
45     def __init__(self, d_model, n_heads, d_ff, dropout):
46         super().__init__()
47         self.ln1 = nn.LayerNorm(d_model)
48         self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
49         self.ln2 = nn.LayerNorm(d_model)
50         self.ff = nn.Sequential(
51             nn.Linear(d_model, d_ff), nn.GELU(),
52             nn.Linear(d_ff, d_model), nn.Dropout(dropout),
53         )
```

- **Lignes 47 et 49** : deux `LayerNorm`. Chacune ramène chaque vecteur de token à une moyenne de 0 et un écart-type de 1, puis applique un facteur et un décalage appris (128 + 128 = 256 paramètres). Cela garde les valeurs dans une plage stable d'un bloc à l'autre.
- **Ligne 48** : la couche d'attention définie plus haut.
- **Lignes 50 à 53** : le feed-forward, une suite de couches exécutées dans l'ordre (`nn.Sequential`) :
  - `Linear(128, 512)` élargit chaque token à 512 neurones (128 × 512 + 512 = 66 048 paramètres) ;
  - `GELU` est l'activation non linéaire. Elle ressemble à ReLU, mais en plus douce autour de 0, ce qui est standard dans les Transformers. Sans activation, deux couches linéaires équivaudraient à une seule ;
  - `Linear(512, 128)` revient à 128 dimensions (512 × 128 + 128 = 65 664 paramètres) ;
  - `Dropout` pour la régularisation.

### Lignes 55 à 58 : le calcul

```python
55     def forward(self, x, pad_mask):
56         x = x + self.attn(self.ln1(x), pad_mask)   # connexion résiduelle
57         x = x + self.ff(self.ln2(x))
58         return x
```

- **Ligne 56** : normalise, applique l'attention, puis **ajoute le résultat à l'entrée** (`x + ...`). C'est la connexion résiduelle : le bloc n'apprend qu'une *correction* à apporter, l'information d'origine n'est jamais perdue, et le gradient traverse facilement les 4 blocs pendant l'entraînement.
- **Ligne 57** : même principe avec le feed-forward, qui travaille sur chaque token séparément. L'attention fait circuler l'information *entre* les mots ; le feed-forward la transforme *à l'intérieur* de chaque mot.
- Le schéma est dit *pre-norm* : la LayerNorm est placée **avant** chaque sous-couche, et non après comme dans l'article original de 2017. Il est plus stable à entraîner.

Paramètres d'un bloc : 256 + 66 048 (attention) + 256 + 131 712 (feed-forward) = 198 272.

---

## Classe `PhishingTransformer` (lignes 61 à 98)

Le modèle complet.

### Lignes 61 à 67 : les embeddings

```python
61 class PhishingTransformer(nn.Module):
62     def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff,
63                  n_features, head_hidden, dropout, head_dropout):
64         super().__init__()
65         self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
66         self.pos_emb = nn.Embedding(max_len, d_model)
67         self.drop = nn.Dropout(dropout)
```

- **Lignes 62 et 63** : reçoit les dimensions fournies par `cfg.model_kwargs()`.
- **Ligne 65** : la table d'embedding des tokens, 8 000 lignes × 128 colonnes (1 024 000 paramètres, 55 % du modèle). Le token n° i lit la ligne i. Pendant l'entraînement, les mots de sens proche finissent par avoir des vecteurs proches. `padding_idx=0` désigne la ligne du remplissage : elle ne reçoit jamais de gradient, donc l'entraînement ne la modifie pas. Elle n'est cependant **pas nulle** : `_init_weights` (ligne 76) la remplit au hasard comme les autres. C'est sans conséquence, car le masque exclut le remplissage de l'attention et de la moyenne finale.
- **Ligne 66** : la table des positions, 256 × 128 (32 768 paramètres) : un vecteur appris par position. Sans elle, l'attention traiterait l'email comme un sac de mots, sans ordre.
- **Ligne 67** : un dropout appliqué juste après les embeddings.

### Lignes 68 à 70 : les blocs

```python
68         self.blocks = nn.ModuleList(
69             [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
70         self.ln_f = nn.LayerNorm(d_model)
```

- **Lignes 68 et 69** : crée 4 blocs indépendants, chacun avec ses propres poids. `nn.ModuleList` (et non une simple liste Python) est nécessaire pour que PyTorch voie leurs poids et les entraîne.
- **Ligne 70** : une LayerNorm finale après le dernier bloc, comme le veut le schéma *pre-norm*.

### Lignes 71 à 76 : la tête de classification

```python
71         self.head = nn.Sequential(
72             nn.Linear(d_model + n_features, head_hidden), nn.ReLU(),
73             nn.Dropout(head_dropout),
74             nn.Linear(head_hidden, 1),               # 1 neurone : logit de P(phishing)
75         )
76         self.apply(self._init_weights)
```

- **Ligne 72** : 128 (le résumé du texte) + 12 (les indices) = 140 entrées vers 64 neurones (140 × 64 + 64 = 9 024 paramètres), puis `ReLU` = max(0, x) : un neurone dont la somme est négative s'éteint.
- **Ligne 73** : un dropout de 30 %.
- **Ligne 74** : un seul neurone de sortie (64 + 1 = 65 paramètres). Sa valeur est le *logit* : positif plutôt phishing, négatif plutôt légitime.
- **Ligne 76** : `apply` appelle `_init_weights` sur chaque sous-couche du modèle.

### Lignes 78 à 84 : l'initialisation des poids

```python
78     @staticmethod
79     def _init_weights(m):
80         if isinstance(m, nn.Linear):
81             nn.init.normal_(m.weight, std=0.02)
82             nn.init.zeros_(m.bias)
83         elif isinstance(m, nn.Embedding):
84             nn.init.normal_(m.weight, std=0.02)
```

- **Ligne 78** : `@staticmethod` indique une méthode qui n'utilise pas `self`.
- **Lignes 80 à 82** : les poids des couches linéaires sont tirés au hasard selon une loi normale d'écart-type 0,02, et les biais partent de 0.
- **Lignes 83 et 84** : même tirage pour les embeddings. Ce sont les valeurs de GPT-2. Des poids de départ petits évitent que les signaux explosent ou s'éteignent en traversant les couches. Le `_` final des fonctions `normal_` et `zeros_` signifie que le tenseur est modifié en place.

### Lignes 86 à 92 : du texte aux vecteurs

```python
86     def forward(self, ids, pad_mask, feats):
87         T = ids.size(1)
88         pos = torch.arange(T, device=ids.device)
89         x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
90         for block in self.blocks:
91             x = block(x, pad_mask)
92         x = self.ln_f(x)
```

- **Ligne 86** : les entrées : `ids` (B × 256), `pad_mask` (B × 256) et `feats` (B × 12).
- **Ligne 87** : la longueur des séquences, 256.
- **Ligne 88** : les positions 0, 1, 2… 255, créées sur le même appareil (CPU ou GPU) que les données.
- **Ligne 89** : chaque token reçoit son vecteur de mot (B × 256 × 128) plus son vecteur de position (256 × 128, ajouté à chaque email du lot), puis le dropout.
- **Lignes 90 et 91** : passe dans les 4 blocs, l'un après l'autre. La forme reste B × 256 × 128, mais chaque vecteur tient désormais compte du contexte de tout l'email.
- **Ligne 92** : la normalisation finale.

### Lignes 94 à 98 : résumer et décider

```python
94         # Mean pooling : moyenne des vecteurs des vrais tokens seulement
95         m = pad_mask.unsqueeze(-1).float()
96         pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
97
98         return self.head(torch.cat([pooled, feats], dim=1)).squeeze(-1)  # logits (B,)
```

- **Ligne 95** : le masque passe de B × 256 à B × 256 × 1 (`unsqueeze(-1)`) et devient un nombre à virgule, pour pouvoir multiplier chaque vecteur par 1 ou 0.
- **Ligne 96** : met à zéro les vecteurs du remplissage, additionne les 256 vecteurs, puis divise par le nombre de vrais tokens. On obtient la **moyenne des vrais tokens** (B × 128) : un email court n'est pas dilué par son remplissage. `clamp(min=1.0)` évite une division par zéro pour un email vide.
- **Ligne 98** : `torch.cat` colle le résumé (128) et les indices (12) en 140 valeurs, la tête calcule le logit (B × 1), et `squeeze(-1)` retire la dimension inutile (B). La sigmoïde n'est pas appliquée ici : la fonction de perte `BCEWithLogitsLoss` la calcule elle-même, de façon plus stable numériquement. La prédiction l'applique séparément.

## Lignes 101 et 102 : `count_params()`

```python
101 def count_params(model: nn.Module) -> int:
102     return sum(p.numel() for p in model.parameters())
```

Additionne le nombre d'éléments (`numel`) de chaque tenseur de poids. Pour ce modèle : 1 859 201.

| Composant | Paramètres |
| --- | --- |
| Embedding des tokens | 1 024 000 |
| Embedding des positions | 32 768 |
| 4 blocs × 198 272 | 793 088 |
| LayerNorm finale | 256 |
| Tête | 9 089 |
| **Total** | **1 859 201** |
