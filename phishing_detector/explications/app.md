# app.py, ligne par ligne

**Rôle :** l'interface web du détecteur, construite avec **Gradio**, une bibliothèque Python qui crée une page web à partir de composants (zones de texte, boutons, graphiques). Elle charge le livrable une fois au démarrage, analyse un email en environ 17 ms et affiche tous les chiffres : verdict, temps de réponse par étape, mots et indices décisifs, statistiques de session, architecture et évaluation complète sur le test.

Commande : `python app.py`, qui ouvre `http://127.0.0.1:7860`. Tout tourne sur l'ordinateur : rien n'est envoyé sur Internet.

Organisation du fichier :

| Lignes | Partie |
| --- | --- |
| 1–65 | imports, constantes, chargement du modèle |
| 69–113 | prédiction chronométrée et calcul des explications |
| 117–188 | génération du HTML (cartes, verdict, barres, tokens colorés) |
| 191–262 | onglet Analyser : analyse et benchmark |
| 266–282 | onglet Session |
| 286–340 | onglet Modèle |
| 344–485 | onglet Évaluation |
| 489–557 | style CSS et bandeau |
| 560–650 | construction de la page et branchement des boutons |
| 653–661 | lancement du serveur |

---

## Lignes 1 à 12 : la docstring

Le mode d'emploi (deux commandes) et le contenu des quatre onglets.

## Lignes 13 à 30 : les imports

```python
13 import argparse
14 import html
15 import statistics
16 import time
17 from datetime import datetime
18 from pathlib import Path
19
20 import gradio as gr
21 import numpy as np
22 import pandas as pd
23 import torch
24 from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
25
26 from config import Config
27 from data import encode_texts, feature_matrix, standardize
28 from detector import PhishingDetector
29 from model import count_params
30 from text_utils import FEATURE_NAMES, normalize
```

- **Ligne 13** : les options `--port` et `--no-browser`.
- **Ligne 14** : `html.escape` protège l'affichage des tokens. Un token comme `<b>` serait sinon interprété comme une balise HTML.
- **Ligne 15** : moyenne et écart-type des temps de réponse.
- **Ligne 16** : `time.perf_counter()`, le chronomètre le plus précis de Python, de l'ordre de la microseconde.
- **Ligne 17** : l'heure de chaque analyse, dans l'historique.
- **Ligne 18** : le chemin du fichier CSV exporté.
- **Lignes 20 à 24** : Gradio (abrégé `gr`), numpy, pandas, PyTorch et trois métriques de scikit-learn.
- **Lignes 26 à 30** : les fichiers du projet. On réutilise **les mêmes fonctions** qu'à l'entraînement, pour que les emails soient traités à l'identique.

## Lignes 32 à 54 : les constantes

```python
32 CFG = Config()
33 EVAL_CACHE = CFG.run_dir / "eval_cache.npz"
```

- **Ligne 32** : la configuration, chargée une fois pour toute l'application.
- **Ligne 33** : le fichier où sont mis en cache les résultats de l'évaluation, pour ne pas la refaire à chaque ouverture.

```python
35 FEATURE_LABELS = {
36     "log_n_urls": "Nombre de liens", "has_ip_url": "Lien vers une adresse IP",
...
42 }
```

- **Lignes 35 à 42** : un nom lisible en français pour chacun des 12 indices, affiché dans l'interface.

```python
44 EXAMPLES = [
45     "URGENT: Your PayPal account has been suspended ... Click here immediately!",
48     "Hi team, the weekly sync is moved to Thursday ... Thanks, Sarah",
50     "Congratulations! You have won $1,000,000 ... a copy of your ID.",
52     "Dear customer, your monthly statement is now available ... by email.",
54 ]
```

- **Lignes 44 à 54** : quatre emails d'exemple, cliquables dans l'interface : un phishing évident, un email interne légitime, une arnaque à la loterie et une vraie notification bancaire. La v1 classait ce dernier phishing à tort. Les chaînes écrites sur plusieurs lignes sans virgule entre elles sont collées automatiquement par Python.

## Lignes 56 à 65 : charger le modèle

```python
57 _t0 = time.perf_counter()
58 DET = PhishingDetector.load()
59 LOAD_MS = (time.perf_counter() - _t0) * 1000
60 MODEL = DET.model
61 for _p in MODEL.parameters():          # inférence seulement : pas de gradient sur les poids
62     _p.requires_grad_(False)
63 N_PARAMS = count_params(MODEL)
64 MODEL_MB = CFG.model_path.stat().st_size / 1e6
65 DET.predict("warm up")                  # le 1er appel PyTorch est toujours plus lent
```

Ce code s'exécute **une seule fois**, au lancement de l'application, et non à chaque analyse.

- **Lignes 57 à 59** : chronomètre le chargement du livrable. La différence entre deux lectures du chronomètre, multipliée par 1 000, donne des millisecondes.
- **Ligne 60** : un raccourci vers le réseau.
- **Lignes 61 et 62** : indique à PyTorch que les poids ne seront jamais entraînés ici. Il n'a donc pas à garder de quoi calculer leurs gradients. Ce réglage sert aussi à la ligne 105 : on ne calculera le gradient que sur les embeddings.
- **Ligne 63** : le nombre de paramètres (1 859 201), affiché dans le bandeau.
- **Ligne 64** : la taille du fichier en mégaoctets (`st_size` est en octets).
- **Ligne 65** : une prédiction « à vide ». Le tout premier calcul de PyTorch est toujours plus lent, car il prépare la mémoire et les routines de calcul. Sans cette ligne, la première analyse de l'utilisateur afficherait un temps de réponse trompeur.

---

## Lignes 69 à 89 : `run_pipeline()`, la prédiction chronométrée

```python
69 def run_pipeline(raw: str):
70     """Une prédiction complète, chronométrée étape par étape (en ms)."""
71     t = [time.perf_counter()]
72     text = normalize(raw, DET.max_raw_chars)
73     t.append(time.perf_counter())
74     ids, mask = encode_texts([text], DET.tokenizer)
75     t.append(time.perf_counter())
76     raw_feats = torch.tensor(feature_matrix([text]))
77     feats = standardize(raw_feats, DET.feat_mean, DET.feat_std)
78     t.append(time.perf_counter())
79     with torch.inference_mode():
80         logit = MODEL(ids, mask, feats).item()
81     t.append(time.perf_counter())
82     prob = 1 / (1 + np.exp(-logit))
83     t.append(time.perf_counter())
```

La même chaîne que `PhishingDetector.predict()`, mais avec une mesure de temps **entre chaque étape**. `t` est une liste d'instants : on en ajoute un après chaque étape.

- **Ligne 72** : étape 1, la normalisation du texte.
- **Ligne 74** : étape 2, le découpage BPE en 256 tokens et le masque.
- **Lignes 76 et 77** : étape 3, le calcul et la standardisation des 12 indices.
- **Lignes 79 et 80** : étape 4, le passage dans le réseau. `torch.inference_mode()` est la version la plus rapide de « pas de gradient ». C'est l'étape la plus longue.
- **Ligne 82** : étape 5, la sigmoïde, qui transforme le logit en probabilité.

```python
84     steps = ["Normalisation du texte", "Tokenisation BPE", "Features structurelles",
85              "Réseau de neurones", "Sigmoïde + décision"]
86     timings = {s: (t[i + 1] - t[i]) * 1000 for i, s in enumerate(steps)}
87     return dict(text=text, ids=ids, mask=mask, raw_feats=raw_feats, feats=feats,
88                 logit=logit, prob=float(prob), timings=timings,
89                 total_ms=(t[-1] - t[0]) * 1000)
```

- **Ligne 86** : la durée de chaque étape, égale à l'écart entre deux instants consécutifs, en millisecondes.
- **Lignes 87 à 89** : renvoie tous les résultats intermédiaires (utiles aux explications), les durées par étape et le **temps de réponse total**, entre le premier et le dernier instant (`t[-1]` est le dernier élément).

## Lignes 92 à 99 : `feature_contributions()`

```python
92  def feature_contributions(ids, mask, feats, p_base):
93      """Effet de chaque indice : on le remet à sa valeur moyenne et on mesure la variation."""
94      batch = feats.repeat(len(FEATURE_NAMES), 1)
95      batch[torch.arange(len(FEATURE_NAMES)), torch.arange(len(FEATURE_NAMES))] = 0.0
96      with torch.inference_mode():
97          p = torch.sigmoid(MODEL(ids.repeat(len(FEATURE_NAMES), 1),
98                                  mask.repeat(len(FEATURE_NAMES), 1), batch)).numpy()
99      return p_base - p          # > 0 : l'indice pousse vers « phishing »
```

Question posée : **quel est l'effet de chaque indice sur la décision ?**

- **Ligne 94** : copie 12 fois le vecteur d'indices de l'email (12 lignes identiques).
- **Ligne 95** : sur la ligne i, met l'indice i à 0. Après standardisation, 0 est la **valeur moyenne** du train. La ligne i représente donc « le même email, avec un indice i banal ». `batch[[0, 1, …, 11], [0, 1, …, 11]]` désigne la diagonale.
- **Lignes 96 à 98** : passe les 12 variantes dans le réseau **en un seul lot**, avec le même texte répété 12 fois. C'est beaucoup plus rapide que 12 appels séparés.
- **Ligne 99** : la différence avec la probabilité d'origine. Positive : cet indice poussait vers phishing. Négative : il poussait vers légitime. Par exemple, +0,30 pour « Lien vers une adresse IP » veut dire que sans ce lien, la probabilité baisserait de 30 points.

## Lignes 102 à 113 : `token_saliency()`

```python
102 def token_saliency(ids, mask, feats):
103     """Gradient x entrée sur les embeddings : contribution signée de chaque token au logit."""
104     m = MODEL
105     emb = m.tok_emb(ids).detach().requires_grad_(True)
```

Question posée : **quels mots ont poussé vers phishing ?** La méthode s'appelle *gradient × entrée*.

- **Ligne 105** : récupère les vecteurs des tokens (1 × 256 × 128). `detach()` les coupe de la table d'embedding. `requires_grad_(True)` demande à PyTorch de calculer le gradient **par rapport à ces vecteurs**.

```python
106     x = emb + m.pos_emb(torch.arange(ids.size(1)))
107     for block in m.blocks:
108         x = block(x, mask)
109     x = m.ln_f(x)
110     w = mask.unsqueeze(-1).float()
111     pooled = (x * w).sum(1) / w.sum(1).clamp(min=1.0)
112     m.head(torch.cat([pooled, feats], dim=1)).sum().backward()
113     return (emb.grad * emb).sum(-1)[0].detach().numpy()
```

- **Lignes 106 à 111** : refait exactement le calcul de `PhishingTransformer.forward()` (positions, 4 blocs, normalisation, moyenne), mais à partir de `emb`, pour pouvoir remonter jusqu'à lui.
- **Ligne 112** : calcule le logit, puis lance la rétropropagation (`backward()`). PyTorch calcule de combien le logit changerait si on modifiait légèrement chaque nombre de chaque vecteur de token. `.sum()` transforme le logit en un seul nombre, ce qu'exige `backward()`.
- **Ligne 113** : multiplie chaque vecteur par son gradient, puis additionne ses 128 composantes. On obtient **un score signé par token**. Positif : ce mot pousse le logit vers le haut, donc vers phishing. Négatif : vers légitime. `[0]` prend le seul email du lot.

C'est cette valeur qui colore les mots en rouge ou en bleu dans l'interface.

---

## Lignes 117 à 123 : les cartes de chiffres

```python
117 def card(label, value, sub=""):
118     return (f"<div class='kpi'><div class='kpi-label'>{label}</div>"
119             f"<div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>")
120
121
122 def kpis(cards):
123     return "<div class='kpis'>" + "".join(cards) + "</div>"
```

- **Lignes 117 à 119** : fabrique le HTML d'une « carte » : un petit titre, un grand chiffre et une ligne de détail. Les classes `kpi…` sont définies dans le CSS (ligne 497 et suivantes).
- **Lignes 122 et 123** : place plusieurs cartes dans une grille qui s'adapte à la largeur de l'écran.

## Lignes 126 à 145 : `verdict_html()`

```python
126 def verdict_html(prob, threshold, total_ms):
127     phish = prob >= threshold
128     cls, label = ("danger", "PHISHING") if phish else ("safe", "LÉGITIME")
129     margin = abs(prob - threshold)
130     # distance au seuil rapportée à l'espace disponible de ce côté du seuil (0 → 1)
131     rel = margin / ((1 - threshold) if phish else threshold)
132     conf = "élevée" if rel > 0.6 else "moyenne" if rel > 0.25 else "faible"
```

- **Ligne 127** : la décision.
- **Ligne 128** : la couleur (rouge `danger` ou vert `safe`) et le libellé.
- **Ligne 129** : la distance entre la probabilité et le seuil.
- **Ligne 131** : cette distance rapportée à l'espace disponible de ce côté du seuil. Avec un seuil de 0,881, il n'y a que 0,119 de marge côté phishing : une probabilité de 0,999 est presque au maximum (rel ≈ 0,99). La première version comparait l'écart brut, et affichait à tort « confiance moyenne » pour 99,9 %.
- **Ligne 132** : trois niveaux de confiance.

```python
133     return f"""
134 <div class='verdict {cls}'>
...
140   <div class='gauge'><div class='gauge-fill' style='width:{prob * 100:.2f}%'></div>
141     <div class='gauge-thr' style='left:{threshold * 100:.2f}%'></div></div>
...
144   <div class='v-time'>⏱ Temps de réponse : <b>{total_ms:.1f} ms</b></div>
145 </div>"""
```

- **Lignes 133 à 145** : le bloc HTML du verdict.
  - **lignes 136 et 137** : le libellé, la confiance et l'écart au seuil ;
  - **ligne 138** : la probabilité en grand, en pourcentage ;
  - **lignes 140 et 141** : une jauge remplie à la hauteur de la probabilité, avec un trait jaune à la position du seuil ;
  - **lignes 142 et 143** : la légende de la jauge ;
  - **ligne 144** : le **temps de réponse** total.

## Lignes 148 à 155 : `timings_html()`

```python
148 def timings_html(timings, total):
149     rows = ""
150     for name, ms in timings.items():
151         pct = ms / total * 100 if total else 0
152         rows += (f"<div class='bar-row'><div class='bar-name'>{name}</div>"
153                  f"<div class='bar-track'><div class='bar-fill' style='width:{max(pct, .5):.1f}%'></div></div>"
154                  f"<div class='bar-val'>{ms:.2f} ms · {pct:.0f} %</div></div>")
155     return f"<div class='panel'><div class='panel-title'>Temps par étape</div>{rows}</div>"
```

Une barre par étape, dont la longueur est la part de cette étape dans le temps total.
- **Ligne 151** : le pourcentage, protégé contre une division par zéro.
- **Ligne 153** : `max(pct, .5)` garde une barre visible même pour une étape très rapide.
- **Ligne 154** : affiche la durée exacte et le pourcentage.

## Lignes 158 à 173 : `contrib_html()`

```python
158 def contrib_html(contribs, raw_feats, z):
159     order = np.argsort(-np.abs(contribs))
160     peak = max(np.abs(contribs).max(), 1e-6)
161     rows = ""
162     for i in order:
163         c = contribs[i]
164         side = "pos" if c > 0 else "neg"
165         flag = " <span class='tag'>inhabituel</span>" if abs(z[i]) > 2 else ""
```

- **Ligne 159** : trie les indices du plus influent au moins influent. `argsort` renvoie l'ordre des positions, et le signe moins inverse le tri.
- **Ligne 160** : la plus grande contribution sert d'échelle à toutes les barres. Le `1e-6` évite une division par zéro.
- **Ligne 164** : une barre rouge vers la droite (phishing) ou bleue vers la gauche (légitime).
- **Ligne 165** : un badge « inhabituel » si la valeur standardisée dépasse ±2 écarts-types.

```python
166         rows += (f"<div class='bar-row'><div class='bar-name'>{FEATURE_LABELS[FEATURE_NAMES[i]]}{flag}</div>"
167                  f"<div class='bar-track split'><div class='bar-{side}' "
168                  f"style='width:{abs(c) / peak * 50:.1f}%'></div></div>"
169                  f"<div class='bar-val'>{c * 100:+.2f} pts · brut {raw_feats[i]:.2f} · z {z[i]:+.2f}</div></div>")
170     return ("<div class='panel'><div class='panel-title'>Indices structurels "
...
173             "<span class='dot neg'></span>pousse vers légitime</div></div>")
```

- **Lignes 166 à 169** : une ligne par indice : son nom en français, une barre centrée (au plus 50 % de la largeur de chaque côté), puis l'effet en points de pourcentage, la valeur brute et la valeur standardisée `z`.
- **Lignes 170 à 173** : le cadre, le titre et la légende des couleurs.

## Lignes 176 à 188 : `tokens_html()`

```python
176 def tokens_html(tokens, sal, truncated):
177     peak = max(np.abs(sal).max(), 1e-9)
178     chips = ""
179     for tok, s in zip(tokens, sal):
180         a = min(abs(s) / peak, 1.0) ** 0.7
181         color = f"rgba(220,38,38,{a:.2f})" if s > 0 else f"rgba(37,99,235,{a:.2f})"
182         chips += (f"<span class='tok' style='background:{color}' "
183                   f"title='{s:+.4f}'>{html.escape(tok)}</span>")
```

Affiche le texte tel que le modèle l'a lu, chaque token coloré selon son influence.
- **Ligne 177** : le score le plus fort sert de référence.
- **Ligne 180** : l'intensité de la couleur, entre 0 et 1. La puissance 0,7 rend visibles les influences moyennes, qui seraient sinon trop pâles.
- **Ligne 181** : rouge (`rgba(220,38,38,…)`) si le token pousse vers phishing, bleu sinon. Le 4ᵉ nombre est l'opacité.
- **Lignes 182 et 183** : chaque token est une étiquette colorée. Survoler un token affiche sa valeur exacte (`title`). `html.escape` neutralise les caractères `<`, `>` et `&` du texte de l'email.

```python
184     note = ("<div class='warn'>Email tronqué : seuls les 256 premiers tokens sont lus par le modèle.</div>"
185             if truncated else "")
186     return ("<div class='panel'><div class='panel-title'>Ce que le modèle a lu "
...
188             f"{note}<div class='toks'>{chips}</div></div>")
```

- **Lignes 184 et 185** : un avertissement si l'email dépasse 256 tokens.
- **Lignes 186 à 188** : le cadre avec le titre et la légende.

---

## Lignes 191 à 237 : `analyze()`, le bouton « Analyser »

```python
191 def analyze(raw, threshold, history):
192     raw = (raw or "").strip()
193     if not raw:
194         raise gr.Error("Colle d'abord le texte d'un email.")
195     r = run_pipeline(raw)
196     prob, total = r["prob"], r["total_ms"]
```

- **Ligne 191** : reçoit le texte saisi, le seuil choisi avec le curseur et l'historique de la session.
- **Lignes 192 à 194** : refuse un champ vide. `gr.Error` affiche un message rouge dans la page.
- **Ligne 195** : la prédiction chronométrée.

```python
198     t_exp = time.perf_counter()
199     contribs = feature_contributions(r["ids"], r["mask"], r["feats"], prob)
200     n_tok = int(r["mask"].sum())
201     sal = token_saliency(r["ids"], r["mask"], r["feats"])[:n_tok]
202     exp_ms = (time.perf_counter() - t_exp) * 1000
```

- **Lignes 198 à 202** : calcule les deux explications et les chronomètre **séparément**. Le temps de réponse affiché reste celui de la prédiction seule. `[:n_tok]` ne garde que les vrais tokens, sans le remplissage.

```python
204     tokens = [DET.tokenizer.id_to_token(i) for i in r["ids"][0, :n_tok].tolist()]
205     full_len = len(DET.tokenizer.encode(r["text"]).ids) if n_tok == r["mask"].size(1) else n_tok
206     truncated = full_len > n_tok
207     unk = sum(t == "[UNK]" for t in tokens)
208     n_words = len(raw.split())
```

- **Ligne 204** : retraduit les numéros en tokens lisibles.
- **Lignes 205 et 206** : essaie de détecter si l'email a été tronqué. Comme le tokenizer coupe toujours à 256, `full_len` ne dépasse jamais 256 : cette détection ne se déclenche pas en pratique. C'est la même limite que dans `detector.py`.
- **Ligne 207** : le nombre de tokens inconnus.
- **Ligne 208** : le nombre de mots du texte saisi.

```python
210     stats = kpis([
211         card("Logit", f"{r['logit']:+.3f}", "sortie brute du dernier neurone"),
212         card("Tokens lus", f"{n_tok} / {CFG.max_len}", ...),
214         card("Texte", f"{len(raw):,} car.".replace(",", " "), ...),
215         card("Débit", f"{n_tok / total * 1000:,.0f}".replace(",", " ") + " tok/s", ...),
217         card("Explication", f"{exp_ms:.0f} ms", "calcul des contributions (hors temps de réponse)"),
218     ])
```

- **Lignes 210 à 218** : cinq cartes de chiffres : le logit, les tokens lus sur 256, la longueur du texte, le débit (tokens par seconde et emails par seconde) et le temps des explications. `.replace(",", " ")` remplace le séparateur de milliers anglais par une espace.

```python
220     top = pd.DataFrame({"token": tokens, "contribution": sal})
221     top = (top.groupby("token", as_index=False)["contribution"].sum()
222               .sort_values("contribution", key=np.abs, ascending=False).head(12))
223     top["effet"] = np.where(top["contribution"] > 0, "→ phishing", "→ légitime")
224     top["contribution"] = top["contribution"].round(4)
```

- **Lignes 220 à 224** : le tableau des 12 tokens les plus influents. Un mot qui apparaît plusieurs fois voit ses contributions additionnées (`groupby(...).sum()`). Le tri se fait par valeur absolue (`key=np.abs`), pour mélanger les deux sens. On ajoute une colonne « effet ».

```python
226     history = history + [{
227         "heure": datetime.now().strftime("%H:%M:%S"),
228         "aperçu": raw[:70].replace("\n", " ") + ("…" if len(raw) > 70 else ""),
229         "P(phishing)": round(prob, 4), "seuil": round(threshold, 3),
230         "verdict": "PHISHING" if prob >= threshold else "légitime",
231         "latence (ms)": round(total, 2), "tokens": n_tok,
232         **{f"{k} (ms)": round(v, 3) for k, v in r["timings"].items()},
233     }]
```

- **Lignes 226 à 233** : ajoute l'analyse à l'historique : l'heure, les 70 premiers caractères, la probabilité, le seuil, le verdict, la latence, le nombre de tokens et le temps de chaque étape (ligne 232, `**` fusionne ce dictionnaire dans l'entrée). On crée une nouvelle liste (`history + [...]`) au lieu de modifier l'ancienne, ce qui est la bonne pratique avec l'état de Gradio.

```python
234     return (verdict_html(prob, threshold, total), stats, timings_html(r["timings"], total),
235             contrib_html(contribs, r["raw_feats"][0].numpy(), r["feats"][0].numpy()),
236             tokens_html(tokens, sal, truncated), top[["token", "effet", "contribution"]],
237             history, *session_outputs(history))
```

- **Lignes 234 à 237** : renvoie, dans l'ordre, une valeur pour chaque composant à mettre à jour (liste ligne 634). L'onglet Session est mis à jour en même temps. `*` déplie les trois valeurs renvoyées par `session_outputs`.

## Lignes 240 à 262 : `benchmark()`

```python
240 def benchmark(raw, n_runs):
241     raw = (raw or "").strip()
242     if not raw:
243         raise gr.Error("Colle d'abord le texte d'un email.")
244     n_runs = int(n_runs)
245     lat = [run_pipeline(raw)["total_ms"] for _ in range(n_runs)]
```

- **Ligne 245** : analyse le même email `n_runs` fois (30 par défaut) et garde chaque temps de réponse. Une seule mesure ne veut rien dire : le système d'exploitation peut ralentir un appel au hasard.

```python
246     # Débit en lot : 32 copies traitées en un seul passage du réseau
247     text = normalize(raw, DET.max_raw_chars)
248     t0 = time.perf_counter()
249     ids, mask = encode_texts([text] * 32, DET.tokenizer)
250     feats = standardize(torch.tensor(feature_matrix([text] * 32)), DET.feat_mean, DET.feat_std)
251     with torch.inference_mode():
252         MODEL(ids, mask, feats)
253     batch_ms = (time.perf_counter() - t0) * 1000
```

- **Lignes 247 à 253** : mesure aussi le traitement de **32 emails d'un coup**. Le réseau traite un lot bien plus efficacement que 32 emails un par un. C'est le chiffre qui compte pour filtrer une boîte de réception entière.

```python
254     q = np.percentile(lat, [50, 95, 99])
255     return kpis([
256         card("Moyenne", f"{statistics.mean(lat):.1f} ms", f"sur {n_runs} exécutions"),
257         card("Médiane (p50)", f"{q[0]:.1f} ms", f"écart-type {statistics.pstdev(lat):.1f} ms"),
258         card("p95 / p99", f"{q[1]:.1f} / {q[2]:.1f} ms", "95 % / 99 % des réponses en dessous"),
259         card("Min / Max", f"{min(lat):.1f} / {max(lat):.1f} ms", ""),
260         card("Débit unitaire", f"{1000 / statistics.mean(lat):.1f} emails/s", "un email à la fois"),
261         card("Débit en lot", f"{32000 / batch_ms:.1f} emails/s", f"lot de 32 en {batch_ms:.0f} ms"),
262     ])
```

- **Ligne 254** : les percentiles. La médiane (p50) est le temps sous lequel tombent la moitié des réponses ; le p95, 95 % d'entre elles. En production, c'est le p95 ou le p99 qui compte, pas la moyenne : il dit combien de temps attendent les utilisateurs les moins chanceux.
- **Lignes 255 à 262** : six cartes. Le débit unitaire vaut 1 000 ms divisé par la moyenne, et le débit en lot 32 emails × 1 000 divisé par la durée du lot.

---

## Lignes 266 à 282 : `session_outputs()`, l'onglet Session

```python
266 def session_outputs(history):
267     if not history:
268         return "<div class='empty'>Aucune analyse pour l'instant.</div>", pd.DataFrame(), pd.DataFrame()
269     df = pd.DataFrame(history)
270     lat = df["latence (ms)"].to_numpy()
271     n_phish = int((df["verdict"] == "PHISHING").sum())
```

- **Lignes 267 et 268** : sans analyse, un message et deux tableaux vides.
- **Ligne 269** : l'historique devient un tableau pandas.
- **Lignes 270 et 271** : les latences et le nombre de phishing détectés.

```python
272     stats = kpis([
273         card("Analyses", str(len(df)), "depuis l'ouverture de la page"),
...
278         card("Tokens moyens", f"{df['tokens'].mean():.0f}", f"sur {CFG.max_len} maximum"),
279     ])
280     plot = pd.DataFrame({"analyse": np.arange(1, len(df) + 1), "latence (ms)": lat})
281     cols = ["heure", "aperçu", "P(phishing)", "seuil", "verdict", "latence (ms)", "tokens"]
282     return stats, df[cols].iloc[::-1], plot
```

- **Lignes 272 à 279** : six cartes : le nombre d'analyses, les phishing détectés et leur part, la probabilité moyenne, la latence moyenne et médiane, le p95 avec le minimum et le maximum, et le nombre moyen de tokens.
- **Ligne 280** : les données du graphique de latence (numéro d'analyse, latence).
- **Ligne 282** : le tableau d'historique, avec les colonnes principales, inversé (`iloc[::-1]`) pour afficher la plus récente en haut.

---

## Lignes 286 à 306 : `param_table()`, l'onglet Modèle

```python
286 def param_table():
287     groups = {}
288     for name, p in MODEL.named_parameters():
289         parts = name.split(".")
```

- **Ligne 288** : parcourt chaque tenseur de poids avec son nom, par exemple `blocks.2.attn.qkv.weight`.
- **Ligne 289** : coupe le nom aux points : `["blocks", "2", "attn", "qkv", "weight"]`.

```python
290         if parts[0] == "tok_emb":
291             key = "Embedding des tokens (8 000 × 128)"
292         elif parts[0] == "pos_emb":
293             key = "Embedding des positions (256 × 128)"
294         elif parts[0] == "blocks":
295             sub = {"attn": "self-attention (Q, K, V, sortie)", "ff": "feed-forward 128→512→128",
296                    "ln1": "LayerNorm", "ln2": "LayerNorm"}[parts[2]]
297             key = f"Bloc {int(parts[1]) + 1} · {sub}"
298         elif parts[0] == "ln_f":
299             key = "LayerNorm finale"
300         else:
301             key = "Tête : Dense 140→64 + Dense 64→1"
302         groups[key] = groups.get(key, 0) + p.numel()
```

- **Lignes 290 à 301** : range chaque tenseur dans un groupe lisible. Pour les blocs, le numéro est dans `parts[1]` (0 à 3, affiché 1 à 4) et la sous-couche dans `parts[2]`.
- **Ligne 302** : additionne le nombre de poids de chaque groupe. `groups.get(key, 0)` vaut 0 la première fois.

```python
303     df = pd.DataFrame({"composant": list(groups), "paramètres": list(groups.values())})
304     df["part"] = (df["paramètres"] / N_PARAMS * 100).round(2).astype(str) + " %"
305     df["paramètres"] = df["paramètres"].map(lambda v: f"{v:,}".replace(",", " "))
306     return df
```

- **Lignes 303 à 306** : le tableau final, avec la part de chaque composant et les nombres formatés (`1 024 000`).

## Lignes 309 à 325 : `model_overview()`

```python
309 def model_overview():
310     k = DET.model.__class__.__name__
311     m = DET.metrics
312     head = kpis([
313         card("Paramètres", f"{N_PARAMS:,}".replace(",", " "), k),
314         card("Fichier livrable", f"{MODEL_MB:.1f} Mo", CFG.model_path.name),
315         card("Chargement", f"{LOAD_MS:.0f} ms", "poids + tokenizer + normalisation"),
316         card("Seuil de décision", f"{DET.threshold:.3f}", "choisi sur la validation (rappel visé 97 %)"),
317         card("Matériel", "CPU", f"{torch.get_num_threads()} threads · PyTorch {torch.__version__}"),
318     ])
```

- **Ligne 310** : le nom de la classe du modèle, `PhishingTransformer`.
- **Ligne 311** : les scores du test enregistrés dans le livrable.
- **Lignes 312 à 318** : cinq cartes : le nombre de paramètres, la taille du fichier, le temps de chargement, le seuil, et le matériel (nombre de threads du processeur, version de PyTorch).

```python
319     test = kpis([
320         card("PR-AUC (test)", f"{m.get('pr_auc', 0):.4f}", "aire sous la courbe précision-rappel"),
321         card("Rappel", ..., f"{int(m.get('phishing_missed (FN)', 0))} phishing ratés"),
322         card("Précision", ..., f"{int(m.get('false_alarms (FP)', 0))} fausses alertes"),
323         card("F1", ..., f"au seuil {m.get('threshold', 0):.3f}"),
324     ])
325     return head, test
```

- **Lignes 319 à 325** : les quatre scores du test, tels que `train.py` les a enregistrés. `.get(..., 0)` évite une erreur si une clé manque.

## Lignes 328 à 340 : la table de configuration

```python
328 CONFIG_DF = pd.DataFrame([
329     ("Tokenizer", f"BPE, vocabulaire {CFG.vocab_size:,}".replace(",", " ")),
...
339     ("Lot / époques", f"{CFG.batch_size} / {CFG.max_epochs} max, early stopping (patience {CFG.patience})"),
340 ], columns=["élément", "valeur"])
```

Un tableau à deux colonnes qui résume l'architecture et l'entraînement. Tout est lu dans `config.py`, rien n'est recopié à la main.

---

## Lignes 344 à 348 : l'état de l'évaluation

```python
344 EVAL = {}
345
346
347 def _model_stamp():
348     return CFG.model_path.stat().st_mtime
```

- **Ligne 344** : un dictionnaire qui garde en mémoire les résultats de l'évaluation. Le curseur de seuil les relit sans tout recalculer.
- **Lignes 347 et 348** : la date de modification du fichier du modèle. Si le modèle est réentraîné, cette date change et le cache devient invalide.

## Lignes 351 à 359 : `run_evaluation()`, le cache

```python
351 def run_evaluation(progress=gr.Progress()):
352     test = pd.read_csv(CFG.processed_dir / "test.csv")
353     texts, y = test["text"].astype(str).tolist(), test["label"].to_numpy()
354
355     if EVAL_CACHE.exists():
356         c = np.load(EVAL_CACHE)
357         if float(c["stamp"]) == _model_stamp() and len(c["y"]) == len(y):
358             EVAL.update({k: c[k] for k in c.files}, texts=texts, cached=True)
359             return eval_view(DET.threshold) + (gr.update(value=DET.threshold),)
```

- **Ligne 351** : `gr.Progress()` donne accès à une barre de progression dans la page.
- **Lignes 352 et 353** : lit les 1 750 emails de test et leurs étiquettes.
- **Lignes 355 à 359** : si un cache existe **pour ce même modèle** et ce même test, on le recharge et on affiche directement les résultats. L'écran apparaît alors instantanément au lieu de prendre environ 47 secondes. La dernière valeur renvoyée remet le curseur au seuil du modèle.

## Lignes 361 à 369 : le Transformer sur le test

```python
361     probs, bs = [], 64
362     t0 = time.perf_counter()
363     for i in progress.tqdm(range(0, len(texts), bs), desc="Transformer sur le jeu de test"):
364         chunk = texts[i:i + bs]
365         ids, mask = encode_texts(chunk, DET.tokenizer)
366         feats = standardize(torch.tensor(feature_matrix(chunk)), DET.feat_mean, DET.feat_std)
367         with torch.inference_mode():
368             probs.append(torch.sigmoid(MODEL(ids, mask, feats)).numpy())
369     tf_s = time.perf_counter() - t0
```

- **Lignes 361 à 368** : prédit les 1 750 emails par lots de 64, avec une barre de progression dans la page (`progress.tqdm`). `range(0, len(texts), bs)` donne 0, 64, 128…
- **Ligne 369** : le temps total, pour calculer le temps par email.

## Lignes 371 à 382 : la référence TF-IDF

```python
371     progress(0.95, desc="Référence TF-IDF (entraînement + prédiction)")
372     from sklearn.feature_extraction.text import TfidfVectorizer
373     from sklearn.linear_model import LogisticRegression
374     train = pd.read_csv(CFG.processed_dir / "train.csv")
375     t0 = time.perf_counter()
376     vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)
377     clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
378     clf.fit(vec.fit_transform(train["text"]), train["label"])
379     bl_train_s = time.perf_counter() - t0
380     t0 = time.perf_counter()
381     p_bl = clf.predict_proba(vec.transform(texts))[:, 1]
382     bl_s = time.perf_counter() - t0
```

- **Ligne 371** : la barre passe à 95 %, avec un nouveau message.
- **Lignes 372 et 373** : ces imports sont placés ici, et non en haut du fichier, pour ne pas ralentir le démarrage de l'application quand l'évaluation n'est pas utilisée.
- **Lignes 374 à 379** : réentraîne la référence TF-IDF exactement comme `baseline.py`, sur le même train, en chronométrant l'entraînement.
- **Lignes 380 à 382** : prédit le test et chronomètre la prédiction seule (environ 0,6 ms par email contre environ 17 ms pour le Transformer).

## Lignes 384 à 388 : enregistrer le cache

```python
384     data = dict(p_tf=np.concatenate(probs), p_bl=p_bl, y=y, tf_s=tf_s, bl_s=bl_s,
385                 bl_train_s=bl_train_s, n_train=len(train), stamp=_model_stamp())
386     np.savez(EVAL_CACHE, **data)
387     EVAL.update(data, texts=texts, cached=False)
388     return eval_view(DET.threshold) + (gr.update(value=DET.threshold),)
```

- **Lignes 384 et 385** : rassemble les probabilités des deux modèles, les étiquettes, les temps mesurés, la taille du train et la date du modèle.
- **Ligne 386** : enregistre le tout dans un fichier numpy compressé (`.npz`).
- **Ligne 387** : garde aussi les résultats en mémoire, avec les textes (pour la liste des erreurs).
- **Ligne 388** : construit l'affichage au seuil du modèle.

## Lignes 391 à 399 : `metrics_at()`

```python
391 def metrics_at(y, p, thr):
392     pred = p >= thr
393     tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
394     fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
395     rec = tp / max(tp + fn, 1)
396     prec = tp / max(tp + fp, 1)
397     return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=rec, precision=prec,
398                 f1=2 * prec * rec / max(prec + rec, 1e-9), acc=(tp + tn) / len(y),
399                 fpr=fp / max(fp + tn, 1))
```

Une version rapide de `metrics.report()`, écrite avec numpy pour pouvoir être recalculée des dizaines de fois quand on bouge le curseur.
- **Ligne 392** : les décisions à ce seuil.
- **Lignes 393 et 394** : les quatre cases de la matrice de confusion. `&` est le « et » élément par élément, `~` le « non ».
- **Lignes 395 à 399** : rappel, précision, F1, exactitude et taux de fausses alertes. `max(..., 1)` évite les divisions par zéro.

## Lignes 402 à 411 : `confusion_html()`

Un tableau 2 × 2 en HTML : les lignes sont la vraie classe, les colonnes la prédiction. Les bonnes réponses ont un fond vert, les erreurs un fond rouge, et chaque case porte son nom (« fausses alertes », « phishing ratés »…).

## Lignes 414 à 485 : `eval_view()`, construire l'écran d'évaluation

```python
414 def eval_view(thr):
415     if not EVAL:
416         empty = "<div class='empty'>Lance l'évaluation pour afficher les résultats.</div>"
417         return (empty, empty, empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
418                 pd.DataFrame(), pd.DataFrame())
419     y, p_tf, p_bl = EVAL["y"], EVAL["p_tf"], EVAL["p_bl"]
420     n = len(y)
421     mt, mb, mb05 = metrics_at(y, p_tf, thr), metrics_at(y, p_bl, thr), metrics_at(y, p_bl, 0.5)
422     tf_ms = float(EVAL["tf_s"]) / n * 1000
```

Appelée au premier calcul, puis chaque fois qu'on relâche le curseur de seuil.
- **Lignes 415 à 418** : sans évaluation, un message et des tableaux vides.
- **Ligne 421** : les métriques du Transformer et de TF-IDF au seuil choisi, plus TF-IDF à 0,5, son seuil naturel.
- **Ligne 422** : le temps par email du Transformer.

```python
424     head = kpis([
425         card("Emails de test", ...),
...
429         card("PR-AUC / ROC-AUC", f"{average_precision_score(y, p_tf):.4f}",
430              f"ROC-AUC {roc_auc_score(y, p_tf):.4f}"),
431         card("Temps d'inférence", ...),
434     ])
```

- **Lignes 424 à 434** : six cartes. La PR-AUC et la ROC-AUC ne dépendent pas du seuil ; les autres chiffres oui. La ROC-AUC mesure la probabilité qu'un phishing pris au hasard reçoive un score plus haut qu'un email légitime pris au hasard.

```python
436     def row(name, m, p, thr_, s):
437         return [name, f"{thr_:.3f}", ...]
440     comp = pd.DataFrame([
441         row("Transformer", mt, p_tf, thr, float(EVAL["tf_s"])),
442         row("TF-IDF + rég. logistique", mb, p_bl, thr, float(EVAL["bl_s"])),
443         row("TF-IDF (seuil 0,5)", mb05, p_bl, 0.5, float(EVAL["bl_s"])),
444     ], columns=[...])
```

- **Lignes 436 à 445** : le tableau de comparaison, avec une ligne par modèle. `row` est une petite fonction locale qui met en forme une ligne.

```python
447     cms = (confusion_html(mt, f"Matrice de confusion · Transformer · seuil {thr:.3f}")
448            + confusion_html(mb, f"Matrice de confusion · TF-IDF · seuil {thr:.3f}"))
449     cms = f"<div class='two'>{cms}</div>"
450     note = (f"<div class='note'>Référence TF-IDF entraînée en {float(EVAL['bl_train_s']):.1f} s "
451             f"sur {int(EVAL.get('n_train', 0)):,} emails d'entraînement. ...</div>".replace(",", " "))
```

- **Lignes 447 à 449** : les deux matrices de confusion côte à côte.
- **Lignes 450 et 451** : une note sur le temps d'entraînement de TF-IDF et la taille réelle du train (17 996 en v2).

```python
453     # Courbes précision-rappel
454     pr = []
455     for name, p in (("Transformer", p_tf), ("TF-IDF", p_bl)):
456         prec, rec, _ = precision_recall_curve(y, p)
457         idx = np.linspace(0, len(rec) - 1, min(300, len(rec))).astype(int)
458         pr.append(pd.DataFrame({"rappel": rec[idx], "précision": prec[idx], "modèle": name}))
459     pr = pd.concat(pr)
```

- **Lignes 454 à 459** : la courbe précision-rappel de chaque modèle. Chaque point est un seuil possible. Une courbe qui reste en haut à droite est meilleure. On garde au plus 300 points par courbe (ligne 457), régulièrement espacés, pour un graphique fluide.

```python
461     # Métriques en fonction du seuil
462     rows = []
463     for t in np.linspace(0.02, 0.98, 49):
464         m = metrics_at(y, p_tf, t)
465         rows += [(t, "rappel", m["recall"]), (t, "précision", m["precision"]), (t, "F1", m["f1"])]
466     curve = pd.DataFrame(rows, columns=["seuil", "valeur", "métrique"])
```

- **Lignes 462 à 466** : calcule le rappel, la précision et le F1 du Transformer pour 49 seuils, de 0,02 à 0,98. Le graphique montre le compromis : en montant le seuil, la précision augmente et le rappel baisse.

```python
468     # Distribution des probabilités par vraie classe
469     bins = np.linspace(0, 1, 21)
470     hist = []
471     for label, cls in (("légitime", 0), ("phishing", 1)):
472         counts, _ = np.histogram(p_tf[y == cls], bins=bins)
473         hist += [(f"{bins[i]:.2f}", int(c), label) for i, c in enumerate(counts)]
474     hist = pd.DataFrame(hist, columns=["P(phishing)", "emails", "vraie classe"])
```

- **Lignes 469 à 474** : l'histogramme des probabilités du Transformer, en 20 tranches de 0,05, séparé selon la vraie classe. Un bon modèle met les emails légitimes près de 0 et les phishing près de 1. Celui-ci est très tranché : presque tout est sous 0,01 ou au-dessus de 0,99.

```python
476     # Erreurs du Transformer
477     err = np.where((p_tf >= thr) != (y == 1))[0]
478     errors = pd.DataFrame({
479         "type": ["phishing raté" if y[i] == 1 else "fausse alerte" for i in err],
480         "P(phishing)": np.round(p_tf[err], 4),
481         "P TF-IDF": np.round(p_bl[err], 4),
482         "TF-IDF aussi faux": ["oui" if (p_bl[i] >= thr) != (y[i] == 1) else "non" for i in err],
483         "extrait": [EVAL["texts"][i][:160] + "…" for i in err],
484     }).sort_values("P(phishing)")
485     return head, cms, note, comp, pr, curve, hist, errors
```

- **Ligne 477** : les positions où la décision du Transformer ne correspond pas à la vraie classe.
- **Lignes 478 à 484** : pour chaque erreur, son type, la probabilité des deux modèles, si TF-IDF se trompe aussi, et les 160 premiers caractères de l'email. Les erreurs communes aux deux modèles sont souvent des emails mal étiquetés dans le dataset, ou réellement ambigus.
- **Ligne 485** : renvoie les 8 éléments de l'écran, dans l'ordre de la liste de la ligne 647.

---

## Lignes 489 à 544 : le style CSS

Une chaîne de texte qui contient les règles de style de la page. Gradio l'injecte dans la page (ligne 661).

| Lignes | Ce qu'elles mettent en forme |
| --- | --- |
| 490 | largeur maximale de la page (1 280 pixels), centrée |
| 491–496 | le bandeau d'en-tête : dégradé indigo-violet, titre, pastilles |
| 497–502 | les cartes de chiffres : grille qui s'adapte à l'écran, grand chiffre, chiffres à largeur fixe (`tabular-nums`) pour qu'ils restent alignés |
| 503–516 | le bloc du verdict : fond rouge ou vert en dégradé, grande probabilité, jauge et trait jaune du seuil, pastille du temps de réponse |
| 517–520 | les panneaux encadrés et leurs titres |
| 521–527 | les barres horizontales : nom, piste, valeur. `split` trace un axe au centre, avec les barres rouges à droite et bleues à gauche |
| 528–531 | le badge « inhabituel » et les pastilles de légende |
| 532–534 | les tokens colorés (police à chasse fixe, zone défilante) et l'avertissement de troncature |
| 535–542 | messages vides, notes, disposition en deux colonnes, matrice de confusion |
| 543 | sur un écran de moins de 800 pixels (téléphone), tout passe sur une seule colonne |

Les couleurs de fond des panneaux utilisent les variables de Gradio (`var(--block-background-fill)`…), pour s'adapter automatiquement au mode clair et au mode sombre.

## Lignes 546 à 557 : le bandeau

```python
546 HERO = f"""
547 <div class='hero'>
548   <h1>🛡️ Détecteur de phishing</h1>
549   <p>Transformer écrit de zéro · emails en anglais · analyse locale, rien n'est envoyé sur Internet</p>
550   <div class='chips'>
551     <span class='chip'>{N_PARAMS:,} paramètres</span>
...
555     <span class='chip'>chargé en {LOAD_MS:.0f} ms</span>
556   </div>
557 </div>""".replace(",", " ")
```

Le bandeau en haut de la page : le titre, une phrase de présentation et cinq pastilles (paramètres, architecture, vocabulaire, seuil, temps de chargement). `.replace(",", " ")` transforme `1,859,201` en `1 859 201`.

---

## Lignes 560 à 591 : `build()`, l'onglet Analyser

```python
560 def build():
561     with gr.Blocks(title="Détecteur de phishing") as demo:
562         gr.HTML(HERO)
563         history = gr.State([])
```

- **Ligne 561** : `gr.Blocks` est la page. Tout composant créé à l'intérieur du bloc `with` y est ajouté, dans l'ordre. `title` est le titre de l'onglet du navigateur.
- **Ligne 562** : affiche le bandeau.
- **Ligne 563** : `gr.State` est une mémoire invisible, propre à chaque visiteur, qui contient l'historique des analyses.

```python
565         with gr.Tab("🔍 Analyser"):
566             with gr.Row(equal_height=False):
567                 with gr.Column(scale=5):
568                     email = gr.Textbox(label="Email à analyser (anglais)", lines=12, ...)
570                     threshold = gr.Slider(0.01, 0.99, value=round(DET.threshold, 3), step=0.001, ...)
573                     with gr.Row():
574                         go = gr.Button("Analyser", variant="primary", size="lg")
575                         gr.ClearButton(email, value="Effacer", size="lg")
576                     gr.Examples(EXAMPLES, inputs=email, label="Exemples")
```

- **Lignes 565 à 567** : un onglet, une rangée, et une colonne de gauche (largeur relative 5).
- **Ligne 568** : la zone de saisie de l'email, de 12 lignes.
- **Ligne 570** : le curseur de seuil, de 0,01 à 0,99, au seuil du modèle par défaut. Le baisser donne moins de phishing ratés et plus de fausses alertes.
- **Lignes 574 et 575** : le bouton « Analyser » (couleur principale) et un bouton qui vide la zone de texte.
- **Ligne 576** : les exemples cliquables, qui remplissent la zone de texte.

```python
577                 with gr.Column(scale=6):
578                     verdict = gr.HTML("<div class='empty'>Le verdict apparaîtra ici.</div>")
579                     stats = gr.HTML()
580                     timing = gr.HTML()
581             with gr.Row(equal_height=False):
582                 with gr.Column(scale=6):
583                     toks = gr.HTML()
584                 with gr.Column(scale=5):
585                     contrib = gr.HTML()
586             top_tokens = gr.Dataframe(label="Tokens les plus influents", interactive=False)
587             with gr.Accordion("⏱ Mesurer la latence (benchmark)", open=False):
588                 with gr.Row():
589                     n_runs = gr.Slider(5, 200, value=30, step=5, label="Nombre d'exécutions")
590                     bench_btn = gr.Button("Lancer le benchmark")
591                 bench_out = gr.HTML()
```

- **Lignes 577 à 580** : la colonne de droite reçoit le verdict, les cartes de chiffres et les temps par étape. Ce sont des zones HTML vides, remplies par `analyze()`.
- **Lignes 581 à 585** : une deuxième rangée : le texte coloré à gauche, les indices à droite.
- **Ligne 586** : le tableau des tokens influents, en lecture seule.
- **Lignes 587 à 591** : un panneau repliable (`Accordion`, fermé par défaut) pour le benchmark : le nombre d'exécutions (5 à 200), un bouton et une zone de résultat.

## Lignes 593 à 629 : les onglets Session, Modèle et Évaluation

```python
593         with gr.Tab("📈 Session"):
594             sess_stats = gr.HTML(...)
595             lat_plot = gr.LinePlot(x="analyse", y="latence (ms)", title="Temps de réponse par analyse", height=260)
597             sess_table = gr.Dataframe(label="Historique (le plus récent en haut)", interactive=False)
598             export_btn = gr.Button("Exporter l'historique en CSV")
599             export_file = gr.File(label="Fichier CSV", visible=False)
```

- **Lignes 593 à 599** : l'onglet Session : les cartes de statistiques, une courbe de la latence (`x` et `y` sont les noms des colonnes du tableau fourni), l'historique, un bouton d'export et un lien de téléchargement caché jusqu'au premier export.

```python
601         with gr.Tab("🧠 Modèle"):
602             head, test = model_overview()
603             gr.HTML(head)
604             gr.Markdown("#### Scores enregistrés dans le livrable (jeu de test, seuil du modèle)")
605             gr.HTML(test)
606             with gr.Row():
607                 gr.Dataframe(param_table(), label="Paramètres par composant", interactive=False)
608                 gr.Dataframe(CONFIG_DF, label="Configuration", interactive=False)
```

- **Lignes 601 à 608** : l'onglet Modèle, entièrement calculé une fois à la construction de la page : les cartes, les scores du livrable, et les deux tableaux (paramètres par composant, configuration) côte à côte.

```python
610         with gr.Tab("📊 Évaluation"):
611             gr.Markdown("Évalue le modèle sur les **1 750 emails de test** ...")
614             with gr.Row():
615                 eval_btn = gr.Button("Lancer l'évaluation", variant="primary")
616                 eval_thr = gr.Slider(0.01, 0.99, value=round(DET.threshold, 3), step=0.001, ...)
618             ev_head = gr.HTML(...)
619             ev_comp = gr.Dataframe(label="Comparaison des modèles", interactive=False)
620             ev_cm = gr.HTML()
621             ev_note = gr.HTML()
622             with gr.Row():
623                 ev_pr = gr.LinePlot(x="rappel", y="précision", color="modèle", ...)
625                 ev_curve = gr.LinePlot(x="seuil", y="valeur", color="métrique", ...)
627             ev_hist = gr.BarPlot(x="P(phishing)", y="emails", color="vraie classe", ...)
629             ev_err = gr.Dataframe(label="Erreurs du Transformer à ce seuil", interactive=False, wrap=True)
```

- **Lignes 610 à 629** : l'onglet Évaluation : une explication, le bouton et le curseur, puis les zones de résultat (cartes, comparaison, matrices, note), deux courbes côte à côte, l'histogramme et la liste des erreurs. `color=` trace une courbe ou une couleur de barre par valeur de cette colonne : une courbe par modèle, par métrique ou par classe. `wrap=True` fait passer les longs extraits à la ligne.

## Lignes 631 à 650 : brancher les boutons

```python
632         sess_outs = [sess_stats, sess_table, lat_plot]
633         go.click(analyze, [email, threshold, history],
634                  [verdict, stats, timing, contrib, toks, top_tokens, history, *sess_outs])
635         email.submit(analyze, [email, threshold, history],
636                      [verdict, stats, timing, contrib, toks, top_tokens, history, *sess_outs])
637         bench_btn.click(benchmark, [email, n_runs], bench_out)
```

Le principe de Gradio : `composant.événement(fonction, entrées, sorties)`. Quand l'événement se produit, Gradio lit la valeur des entrées, appelle la fonction, puis place chaque valeur renvoyée dans la sortie correspondante, **dans l'ordre**.
- **Lignes 633 et 634** : un clic sur « Analyser » appelle `analyze(email, seuil, historique)`. Ses 10 valeurs de retour remplissent le verdict, les cartes, les temps, les indices, les tokens, le tableau, l'historique et les trois éléments de l'onglet Session.
- **Lignes 635 et 636** : même chose quand on valide la zone de texte au clavier.
- **Ligne 637** : le bouton du benchmark.

```python
639         def export(h):
640             if not h:
641                 raise gr.Error("L'historique est vide.")
642             path = Path(CFG.run_dir) / "historique_analyses.csv"
643             pd.DataFrame(h).to_csv(path, index=False, encoding="utf-8-sig")
644             return gr.update(value=str(path), visible=True)
645         export_btn.click(export, history, export_file)
```

- **Lignes 639 à 645** : l'export. L'historique est écrit dans `artifacts/historique_analyses.csv`. `utf-8-sig` ajoute une marque au début du fichier pour qu'Excel affiche correctement les accents. `gr.update(...)` affiche le lien de téléchargement.

```python
647         ev_outs = [ev_head, ev_cm, ev_note, ev_comp, ev_pr, ev_curve, ev_hist, ev_err]
648         eval_btn.click(run_evaluation, None, ev_outs + [eval_thr])
649         eval_thr.release(eval_view, eval_thr, ev_outs)
650     return demo
```

- **Ligne 648** : le bouton d'évaluation, sans entrée. Il remplit les 8 zones et remet le curseur au seuil du modèle.
- **Ligne 649** : quand on **relâche** le curseur, `eval_view` recalcule toutes les métriques à ce seuil, sans refaire les prédictions. C'est instantané. On utilise `release` plutôt que `change` pour ne pas recalculer à chaque pixel du déplacement.
- **Ligne 650** : renvoie la page construite.

## Lignes 653 à 661 : le lancement

```python
653 if __name__ == "__main__":
654     ap = argparse.ArgumentParser(description="Interface web du détecteur de phishing")
655     ap.add_argument("--port", type=int, default=7860)
656     ap.add_argument("--no-browser", action="store_true")
657     args = ap.parse_args()
658     theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
659                            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
660     build().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser,
661                    theme=theme, css=CSS)
```

- **Lignes 654 à 657** : les options : le port (7860 par défaut) et la possibilité de ne pas ouvrir le navigateur automatiquement.
- **Lignes 658 et 659** : le thème : couleur principale indigo, gris ardoise, police Inter. Si la police n'est pas disponible (hors ligne), la police du système est utilisée.
- **Lignes 660 et 661** : construit la page et lance le serveur web local.
  - `server_name="127.0.0.1"` : le serveur n'est accessible que depuis cet ordinateur, pas depuis le réseau. Les emails analysés restent donc privés.
  - `inbrowser` : ouvre le navigateur sur la page.
  - `theme` et `css` : le style.
