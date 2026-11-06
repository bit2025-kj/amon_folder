# detector.py, ligne par ligne

**Rôle :** le livrable. Ce fichier sait :
- **exporter** un modèle entraîné dans un seul fichier `.pt` ;
- le **recharger** sans réentraîner ;
- **prédire** sur un nouvel email ;
- **expliquer** ce que voit le modèle (mode debug) ;
- proposer une **boucle interactive** dans le terminal.

Utilisé par : `train.py` (export et test final), `predict.py`, `app.py`.

---

## Lignes 1 à 9 : la docstring

```python
1  """Le livrable : un détecteur prêt à l'emploi, chargé depuis UN seul fichier.
2
3      from detector import PhishingDetector
4      det = PhishingDetector.load()                 # artifacts/phishing_model.pt
5      det.predict("Your account is suspended...")   # {'probability': 0.98, 'is_phishing': True, ...}
6
7  Le fichier .pt contient : les poids, l'architecture, le tokenizer, la normalisation
8  des features et le seuil de décision. Aucun réentraînement, aucun autre fichier requis.
9  """
```

Un exemple d'utilisation en trois lignes, et le contenu du fichier livré.

## Lignes 10 à 18 : imports et version du format

```python
10 import torch
11 from tokenizers import Tokenizer
12
13 from config import Config
14 from data import encode_texts, feature_matrix, standardize
15 from model import PhishingTransformer
16 from text_utils import FEATURE_NAMES, normalize
17
18 FORMAT_VERSION = 1
```

- **Lignes 10 et 11** : PyTorch et le tokenizer.
- **Ligne 13** : la configuration, pour le chemin par défaut du modèle.
- **Ligne 14** : les **mêmes** fonctions d'encodage qu'à l'entraînement. C'est essentiel : un email doit être transformé exactement comme les emails d'entraînement, sinon le modèle reçoit des entrées qu'il n'a jamais vues.
- **Ligne 15** : la classe du réseau, pour reconstruire l'architecture.
- **Ligne 16** : la normalisation et les noms des 12 indices (pour l'affichage debug).
- **Ligne 18** : un numéro de version enregistré dans le fichier. Si le format change un jour, on pourra reconnaître les anciens fichiers.

## Lignes 21 à 28 : le constructeur

```python
21 class PhishingDetector:
22     def __init__(self, model, tokenizer, max_raw_chars, feat_mean, feat_std, threshold, metrics=None):
23         self.model = model.eval()
24         self.tokenizer = tokenizer
25         self.max_raw_chars = max_raw_chars
26         self.feat_mean, self.feat_std = feat_mean, feat_std
27         self.threshold = threshold
28         self.metrics = metrics or {}
```

Un détecteur regroupe tout ce qu'il faut pour prédire :

- **Ligne 23** : le réseau, passé en mode évaluation (`eval()`, dropout désactivé). `eval()` renvoie le modèle lui-même, d'où l'écriture en une ligne.
- **Ligne 24** : le tokenizer BPE utilisé à l'entraînement.
- **Ligne 25** : la longueur maximale du texte (20 000 caractères).
- **Ligne 26** : la moyenne et l'écart-type des 12 indices **calculés sur le train**.
- **Ligne 27** : le seuil de décision choisi sur la validation.
- **Ligne 28** : les scores du test. `metrics or {}` met un dictionnaire vide si rien n'est fourni.

## Lignes 31 à 45 : `export()`

```python
31     @staticmethod
32     def export(path, model, cfg: Config, tokenizer: Tokenizer, feat_mean, feat_std,
33                threshold: float, metrics: dict):
34         bundle = {
35             "format_version": FORMAT_VERSION,
36             "model_kwargs": cfg.model_kwargs(),
37             "state_dict": model.state_dict(),
38             "tokenizer": tokenizer.to_str(),
39             "max_raw_chars": cfg.max_raw_chars,
40             "feat_mean": torch.as_tensor(feat_mean, dtype=torch.float32),
41             "feat_std": torch.as_tensor(feat_std, dtype=torch.float32),
42             "threshold": float(threshold),
43             "metrics": {k: float(v) for k, v in metrics.items()},
44         }
45         torch.save(bundle, path)
```

- **Ligne 31** : `@staticmethod` : la méthode n'a pas besoin d'un détecteur existant. On l'appelle `PhishingDetector.export(...)` depuis `train.py`.
- **Ligne 34** : un dictionnaire qui contient tout :
  - **ligne 35** : la version du format ;
  - **ligne 36** : les dimensions de l'architecture (vocabulaire, 128, 4 têtes, 4 blocs…), pour reconstruire le même réseau ;
  - **ligne 37** : les 1 859 201 poids entraînés ;
  - **ligne 38** : le tokenizer complet, converti en texte JSON. Le livrable ne dépend donc pas du fichier `tokenizer.json`, qui changerait à chaque `prepare_data.py` ;
  - **ligne 39** : la longueur de coupure du texte ;
  - **lignes 40 et 41** : les statistiques des indices, converties en tenseurs ;
  - **ligne 42** : le seuil, converti en nombre Python simple ;
  - **ligne 43** : les scores du test, tous convertis en nombres simples.
- **Ligne 45** : écrit le tout dans un seul fichier, d'environ 7,7 Mo.

Pourquoi tout regrouper ? Si on rechargeait les poids avec un autre tokenizer ou d'autres moyennes, le modèle recevrait des entrées décalées. Il donnerait alors des prédictions fausses **sans aucun message d'erreur**. Un seul fichier rend cette erreur impossible.

On n'enregistre que des types simples (tenseurs, nombres, texte, dictionnaires) : c'est ce qui permet le chargement sûr de la ligne 51.

## Lignes 47 à 55 : `load()`

```python
47     @classmethod
48     def load(cls, path=None):
49         path = path or Config().model_path
50         # weights_only=True : chargement sûr, le fichier ne peut pas exécuter de code
51         b = torch.load(path, map_location="cpu", weights_only=True)
52         model = PhishingTransformer(**b["model_kwargs"])
53         model.load_state_dict(b["state_dict"])
54         return cls(model, Tokenizer.from_str(b["tokenizer"]), b["max_raw_chars"],
55                    b["feat_mean"], b["feat_std"], b["threshold"], b["metrics"])
```

- **Ligne 47** : `@classmethod` : la méthode reçoit la classe elle-même (`cls`) et sert de constructeur alternatif. On écrit `det = PhishingDetector.load()`.
- **Ligne 49** : sans chemin, on prend `artifacts/phishing_model.pt`.
- **Ligne 51** : lit le fichier.
  - `map_location="cpu"` : charge sur le processeur, même si le modèle avait été entraîné sur une carte graphique.
  - `weights_only=True` : **chargement sûr**. Un fichier `.pt` ordinaire peut contenir du code Python qui s'exécute à l'ouverture. Avec cette option, PyTorch n'accepte que des données (tenseurs, nombres, texte) : un fichier piégé serait refusé.
- **Ligne 52** : reconstruit un réseau vide avec les dimensions enregistrées.
- **Ligne 53** : y place les poids entraînés.
- **Lignes 54 et 55** : reconstruit le tokenizer depuis son texte JSON, puis crée et renvoie le détecteur complet.

## Lignes 58 à 63 : `_tensors()`

```python
58     def _tensors(self, raw: str):
59         text = normalize(raw, self.max_raw_chars)
60         ids, mask = encode_texts([text], self.tokenizer)
61         raw_feats = torch.tensor(feature_matrix([text]))
62         feats = standardize(raw_feats, self.feat_mean, self.feat_std)
63         return text, ids, mask, raw_feats, feats
```

Transforme un email brut en entrées du réseau, exactement comme à l'entraînement. Le `_` initial indique une méthode interne.

- **Ligne 59** : normalise le texte (minuscules, ponctuation isolée…).
- **Ligne 60** : découpe en 256 tokens et construit le masque. L'email est mis dans une liste, car la fonction traite des lots : on obtient des tenseurs de forme 1 × 256.
- **Ligne 61** : calcule les 12 indices bruts (1 × 12).
- **Ligne 62** : les standardise avec les statistiques **du train** enregistrées dans le fichier, puis les borne à ±5.
- **Ligne 63** : renvoie aussi le texte normalisé et les indices bruts, utiles au mode debug.

## Lignes 65 à 69 : `predict()`

```python
65     @torch.no_grad()
66     def predict(self, raw: str) -> dict:
67         _, ids, mask, _, feats = self._tensors(raw)
68         p = torch.sigmoid(self.model(ids, mask, feats)).item()
69         return {"probability": p, "is_phishing": p >= self.threshold, "threshold": self.threshold}
```

- **Ligne 65** : pas de calcul de gradient, car on ne fait que prédire.
- **Ligne 67** : prépare les entrées. Les `_` ignorent le texte et les indices bruts.
- **Ligne 68** : le réseau renvoie le logit, que la sigmoïde (1 / (1 + e^(−x))) transforme en probabilité entre 0 et 1. `.item()` donne un nombre Python.
- **Ligne 69** : renvoie la probabilité, le verdict (vrai si la probabilité atteint le seuil) et le seuil.

## Lignes 71 à 79 : `explain()`, préparer l'affichage

```python
71     @torch.no_grad()
72     def explain(self, raw: str) -> dict:
73         """Mode debug : montre tout ce que le modèle "voit"."""
74         text, ids, mask, raw_feats, feats = self._tensors(raw)
75         n_tok = int(mask.sum())
76         full_len = len(self.tokenizer.encode(text).ids) if n_tok == mask.size(1) else n_tok
77         tokens = [self.tokenizer.id_to_token(i) for i in ids[0, :n_tok].tolist()]
78         logit = self.model(ids, mask, feats).item()
```

- **Ligne 74** : les mêmes entrées que pour `predict`.
- **Ligne 75** : le nombre de vrais tokens (somme des 1 du masque).
- **Ligne 76** : si les 256 places sont pleines, l'email a peut-être été tronqué. On le réencode pour connaître sa vraie longueur, sinon on garde `n_tok`. Le tokenizer tronque toujours à 256, donc `full_len` vaut au plus 256 et cette vérification ne repère pas tous les cas de troncature. C'est une limite de cet affichage de debug, pas de la prédiction.
- **Ligne 77** : retraduit chaque numéro de token en texte, pour voir comment l'email a été découpé.
- **Ligne 78** : le logit brut.

## Lignes 80 à 94 : `explain()`, l'affichage

```python
80         print("\n--- DEBUG ---")
81         print(f"Texte normalisé ({len(text)} car.) : {text[:300]}{'...' if len(text) > 300 else ''}")
82         print(f"Tokens utilisés : {n_tok}/{mask.size(1)}"
83               + (f"  (tronqué : l'email en fait plus)" if full_len > n_tok else ""))
84         print(f"Premiers tokens : {tokens[:40]}")
85         unk = sum(t == "[UNK]" for t in tokens)
86         print(f"Tokens inconnus [UNK] : {unk}")
87         print("Features (brute -> standardisée) :")
88         for name, r, s in zip(FEATURE_NAMES, raw_feats[0].tolist(), feats[0].tolist()):
89             flag = "  <-- plafonné" if abs(s) >= 5 else ("  <-- inhabituel" if abs(s) > 2 else "")
90             print(f"  {name:15s} {r:8.3f} -> {s:+6.2f}{flag}")
91         print(f"Logit : {logit:+.3f}   P(phishing) = {torch.sigmoid(torch.tensor(logit)).item():.4f}"
92               f"   seuil = {self.threshold:.4f}")
93         print("-------------")
94         return self.predict(raw)
```

- **Ligne 81** : le texte tel que le modèle le reçoit, limité à 300 caractères, avec « ... » s'il est plus long.
- **Lignes 82 et 83** : combien des 256 places sont utilisées, et si l'email a été coupé.
- **Ligne 84** : les 40 premiers tokens, pour vérifier le découpage BPE.
- **Lignes 85 et 86** : le nombre de tokens inconnus. Il doit être proche de 0 ; sinon, le texte contient des caractères absents du vocabulaire, comme des emojis ou un alphabet non latin.
- **Lignes 88 à 90** : pour chaque indice, sa valeur brute et sa valeur standardisée. `zip` parcourt les trois listes en parallèle. Un indice à plus de 2 écarts-types de la moyenne est signalé « inhabituel », et à ±5 « plafonné ». `{s:+6.2f}` affiche toujours le signe.
- **Lignes 91 et 92** : le logit, la probabilité et le seuil.
- **Ligne 94** : renvoie le même résultat que `predict`.

## Lignes 97 à 99 : `format_result()`

```python
97  def format_result(r: dict) -> str:
98      verdict = "PHISHING" if r["is_phishing"] else "légitime"
99      return f"P(phishing) = {r['probability']:.3f}  (seuil {r['threshold']:.3f})  ->  {verdict}"
```

Transforme le résultat en une ligne lisible, par exemple `P(phishing) = 0.998  (seuil 0.881)  ->  PHISHING`.

## Lignes 102 à 118 : `interactive()`, lire l'email

```python
102 def interactive(detector: PhishingDetector, debug: bool = False):
103     """Boucle : on colle un email, on tape FIN sur une ligne seule pour l'analyser."""
104     print("\nColle un email (plusieurs lignes possibles), puis tape FIN sur une ligne seule.")
105     print("Tape Q pour quitter.\n")
106     while True:
107         lines = []
108         try:
109             while True:
110                 line = input("> " if not lines else "  ")
111                 if line.strip().upper() == "Q" and not lines:
112                     return
113                 if line.strip().upper() == "FIN":
114                     break
115                 lines.append(line)
116         except (EOFError, KeyboardInterrupt):
117             print()
118             return
```

- **Lignes 104 et 105** : les instructions. Un email tient sur plusieurs lignes, il faut donc un mot pour signaler sa fin.
- **Ligne 106** : boucle sans fin, un email par tour.
- **Ligne 107** : les lignes de l'email en cours.
- **Ligne 110** : lit une ligne au clavier. L'invite est `> ` pour la première ligne et deux espaces pour les suivantes.
- **Lignes 111 et 112** : `Q` sur la première ligne quitte. `strip().upper()` accepte aussi « q » ou « Q  ».
- **Lignes 113 et 114** : `FIN` termine l'email et sort de la boucle de lecture.
- **Ligne 115** : sinon, on ajoute la ligne à l'email.
- **Lignes 116 à 118** : Ctrl+C, ou la fin de l'entrée (Ctrl+Z sous Windows, Ctrl+D sous Linux), quitte proprement au lieu d'afficher une erreur.

## Lignes 119 à 123 : `interactive()`, analyser

```python
119         email = "\n".join(lines).strip()
120         if not email:
121             continue
122         r = detector.explain(email) if debug else detector.predict(email)
123         print(format_result(r) + "\n")
```

- **Ligne 119** : recolle les lignes avec des retours à la ligne.
- **Lignes 120 et 121** : un email vide est ignoré, et on recommence.
- **Ligne 122** : prédit, avec le détail en mode debug.
- **Ligne 123** : affiche le verdict, puis attend l'email suivant.
