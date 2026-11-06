# text_utils.py, ligne par ligne

**Rôle :** deux fonctions utilisées partout :
- `normalize()` met tous les emails dans le même format ;
- `extract_features()` calcule 12 indices numériques, par exemple le nombre de liens ou la présence de mots d'urgence.

Utilisé par : `prepare_data.py`, `data.py`, `augment.py` (indirectement), `detector.py`, `app.py`.

---

## Lignes 1 à 7 : la docstring

```python
1  """Normalisation du texte + features structurelles manuelles.
2
3  Pourquoi normaliser ? Dans ce dataset, une partie des emails est déjà en minuscules
4  et "pré-tokenisée" ("doesn ' t", "http : / / www . x . com"), l'autre est brute.
5  On met TOUS les emails dans le même format pour que le modèle apprenne le contenu,
6  pas le format d'origine.
7  """
```

Elle explique le problème que résout ce fichier. Dans le dataset, une partie des emails a déjà été traitée : minuscules, apostrophes et ponctuation séparées par des espaces. L'autre partie est brute. Si l'un des deux formats contenait plus de phishing que l'autre, le modèle apprendrait à reconnaître le format plutôt que le contenu. C'est une **fuite d'information** : de bons scores, mais pour une mauvaise raison.

## Lignes 8 à 11 : les imports

```python
8  import math
9  import re
10
11 import numpy as np
```

- **Ligne 8** : `math` fournit `log1p`, c'est-à-dire log(1 + x).
- **Ligne 9** : `re` est le module des expressions régulières, des motifs de recherche dans le texte.
- **Ligne 11** : `numpy` sert à renvoyer les indices sous forme de tableau de nombres.

## Lignes 13 et 14 : deux motifs compilés

```python
13 _PUNCT = re.compile(r"([^\w\s])")
14 _SPACES = re.compile(r"\s+")
```

- **Ligne 13** : `[^\w\s]` reconnaît un caractère qui n'est ni une lettre ou un chiffre (`\w`), ni un espace (`\s`), donc un signe de ponctuation ou un symbole. Les parenthèses le « capturent » pour pouvoir le réutiliser dans le remplacement. Le `r"..."` (*raw string*) empêche Python d'interpréter les `\`.
- **Ligne 14** : `\s+` reconnaît une suite d'un ou plusieurs espaces, tabulations ou retours à la ligne.
- `re.compile` prépare chaque motif une seule fois. C'est plus rapide que de le recompiler pour chacun des 18 000 emails. Le `_` en début de nom signale une variable interne au fichier.

## Lignes 17 à 21 : `normalize()`

```python
17 def normalize(text: str, max_chars: int = 20_000) -> str:
18     """minuscules, ponctuation isolée par des espaces, espaces fusionnés."""
19     text = str(text)[:max_chars].lower()
20     text = _PUNCT.sub(r" \1 ", text)
21     return _SPACES.sub(" ", text).strip()
```

- **Ligne 17** : la fonction reçoit un texte et une longueur maximale, et renvoie un texte. Les annotations `: str` et `-> str` documentent les types.
- **Ligne 19** : `str(text)` protège contre une valeur non textuelle, par exemple un nombre lu dans le CSV. `[:max_chars]` coupe à 20 000 caractères. `.lower()` met tout en minuscules : « URGENT » et « urgent » deviennent le même mot.
- **Ligne 20** : chaque signe de ponctuation est remplacé par lui-même entouré d'espaces. `\1` désigne le caractère capturé. Ainsi `password!` devient `password ! `. Les deux formats du dataset convergent vers le même résultat : `doesn't` et `doesn ' t` deviennent tous deux `doesn ' t`.
- **Ligne 21** : les suites d'espaces deviennent un seul espace, et `.strip()` retire ceux du début et de la fin.

Propriété utile : la fonction est **idempotente**. L'appliquer à un texte déjà normalisé ne change rien, on peut donc la rappeler sans risque.

## Lignes 24 à 28 : la version « compacte »

```python
24 # ---------------------------------------------------------------------------
25 # Features : calculées sur une version "compacte" du texte normalisé
26 # ("http : / / www . x . com" -> "http://www.x.com") pour que les regex marchent.
27 # ---------------------------------------------------------------------------
28 _COMPACT = re.compile(r" ?([^\w\s]) ?")
```

- **Lignes 24 à 27** : après la normalisation, un lien ressemble à `http : / / www . x . com`, et un motif qui cherche `http://` ne le trouverait plus. On calcule donc les indices sur une version où la ponctuation est recollée.
- **Ligne 28** : reconnaît un signe de ponctuation entouré d'espaces optionnels (` ?` = un espace ou aucun). Il sera remplacé par le signe seul, sans les espaces.

## Lignes 30 à 33 : les motifs des liens et adresses

```python
30 _URL = re.compile(r"https?://\S+|www\.\S+")
31 _IP_URL = re.compile(r"(https?://)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
32 _SHORT_URL = re.compile(r"\b(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly)\b")
33 _EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
```

- **Ligne 30** : un lien commence par `http://`, `https://` (le `s?` rend le « s » facultatif) ou `www.`, suivi de caractères non-espaces (`\S+`). Le `|` veut dire « ou ».
- **Ligne 31** : une adresse IP : quatre groupes de 1 à 3 chiffres séparés par des points, comme `192.168.4.21`, éventuellement précédés de `http://`. Un lien vers une adresse IP au lieu d'un nom de domaine est un signe classique de phishing.
- **Ligne 32** : les services de liens raccourcis, qui cachent la vraie destination. `\b` marque une limite de mot, pour ne pas reconnaître `t.co` à l'intérieur de `at.com`.
- **Ligne 33** : une adresse email, sous la forme `nom@domaine.extension`.

## Lignes 35 à 44 : les listes de mots-clés

```python
35 _URGENCY = ["urgent", "immediately", "suspend", ..., "deadline", "last chance"]
38 _CREDENTIALS = ["password", "login", "log in", ..., "billing", "paypal"]
41 _MONEY = ["$", "dollars", "free", "winner", ..., "discount", "%"]
43 _CLICK = ["click here", "click below", ..., "unsubscribe", "remove me"]
```

Quatre familles de signaux typiques du phishing :

- **Lignes 35 à 37**, urgence (18 expressions) : la pression pour agir vite sans réfléchir (« suspended », « within 24 hours », « final notice »).
- **Lignes 38 à 40**, identifiants (15 expressions) : la demande d'informations sensibles (« password », « verify your », « bank »).
- **Lignes 41 et 42**, argent (13 expressions) : l'appât du gain (« winner », « prize », « lottery »).
- **Lignes 43 et 44**, appel à cliquer (7 expressions) : « click here », « follow the link ».

Ces listes ne décident pas seules : elles donnent au réseau des signaux qu'il combine avec ce qu'il lit dans le texte.

## Lignes 46 à 50 : les noms des indices

```python
46 FEATURE_NAMES = [
47     "log_n_urls", "has_ip_url", "has_short_url", "log_n_emails",
48     "urgency", "credentials", "money", "click",
49     "exclam_ratio", "digit_ratio", "log_chars", "log_words",
50 ]
```

Les noms des 12 indices, dans le même ordre que le calcul des lignes 63 à 74. Ils servent à l'affichage (mode debug, interface).

## Lignes 53 et 54 : `_count()`

```python
53 def _count(text: str, words: list[str]) -> int:
54     return sum(text.count(w) for w in words)
```

Compte toutes les occurrences de toutes les expressions d'une liste dans le texte. `text.count(w)` compte les apparitions de `w`, et `sum(...)` additionne le tout. Par exemple, « urgent: your account is suspended, act now » donne 3 pour la liste urgence : *urgent*, *suspend* et *suspended*.

## Lignes 57 à 61 : préparer le calcul

```python
57 def extract_features(norm_text: str) -> np.ndarray:
58     """12 indices structurels, à partir du texte DÉJÀ normalisé."""
59     compact = _COMPACT.sub(r"\1", norm_text)
60     n_words = max(len(norm_text.split()), 1)
61     n_chars = max(len(norm_text), 1)
```

- **Ligne 57** : la fonction reçoit un texte **déjà normalisé** et renvoie un tableau de 12 nombres.
- **Ligne 59** : construit la version compacte : `http : / / www . x . com` redevient `http://www.x.com`.
- **Lignes 60 et 61** : le nombre de mots et de caractères. `max(..., 1)` évite une division par zéro pour un texte vide.

## Lignes 62 à 76 : les 12 indices

```python
62     feats = [
63         math.log1p(len(_URL.findall(compact))),
64         float(bool(_IP_URL.search(compact))),
65         float(bool(_SHORT_URL.search(compact))),
66         math.log1p(len(_EMAIL.findall(compact))),
67         math.log1p(_count(norm_text, _URGENCY)),
68         math.log1p(_count(norm_text, _CREDENTIALS)),
69         math.log1p(_count(norm_text, _MONEY)),
70         math.log1p(_count(norm_text, _CLICK)),
71         norm_text.count("!") / n_words,
72         sum(c.isdigit() for c in norm_text) / n_chars,
73         math.log1p(n_chars),
74         math.log1p(n_words),
75     ]
76     return np.asarray(feats, dtype=np.float32)
```

- **Ligne 63** : `findall` renvoie la liste de tous les liens trouvés, et `len` les compte. `log1p` calcule log(1 + x) : 0 lien donne 0, 1 lien donne 0,69, et 50 liens donnent 3,9 seulement. Le logarithme écrase les grandes valeurs, pour qu'un email de 50 liens ne pèse pas 50 fois plus qu'un email d'un seul lien.
- **Ligne 64** : `search` cherche une première adresse IP. `bool` donne vrai ou faux, et `float` le transforme en 1.0 ou 0.0.
- **Ligne 65** : même principe pour les liens raccourcis.
- **Ligne 66** : le nombre d'adresses email, en logarithme.
- **Lignes 67 à 70** : les quatre familles de mots-clés, comptées sur le texte normalisé, en logarithme.
- **Ligne 71** : le nombre de points d'exclamation par mot. C'est un ratio, donc indépendant de la longueur de l'email.
- **Ligne 72** : la proportion de caractères qui sont des chiffres. Elle est élevée dans les emails remplis de montants, de numéros ou de codes.
- **Lignes 73 et 74** : la longueur en caractères et en mots, en logarithme. Les arnaques sont souvent courtes.
- **Ligne 76** : convertit la liste en tableau numpy de nombres à virgule 32 bits, le format attendu par PyTorch.

## Ligne 79 : une vérification

```python
79 assert len(FEATURE_NAMES) == 12
```

Elle s'exécute au chargement du fichier. Si quelqu'un ajoute un indice sans mettre à jour la liste des noms, le programme s'arrête tout de suite au lieu d'afficher des noms décalés. Le nombre 12 doit aussi correspondre à `n_features` dans `config.py`.
