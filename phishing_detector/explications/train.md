# train.py, ligne par ligne

**Rôle :** entraîner le Transformer. Le script :
- peut être interrompu à tout moment, puis reprendre exactement là où il s'est arrêté ;
- garde les poids de la meilleure époque et s'arrête quand le modèle ne progresse plus ;
- choisit le seuil de décision ;
- mesure le résultat et exporte le livrable `artifacts/phishing_model.pt`.

Commandes :
- `python train.py` démarre l'entraînement, ou le reprend s'il avait été interrompu ;
- `python train.py --fresh` repart de zéro ;
- `python train.py --debug` lance un mini-entraînement de vérification ;
- `python train.py --no-test` saute le test interactif final.

---

## Lignes 1 à 11 : la docstring

```python
1  """Étape 3 : entraînement du Transformer (avec reprise) + export du livrable.
...
9  Ctrl+C à tout moment : la position est sauvegardée, relance  python train.py  pour reprendre.
10 À la fin : artifacts/phishing_model.pt = le livrable (voir detector.py / predict.py).
11 """
```

Le mode d'emploi : les quatre façons de lancer le script, la reprise après Ctrl+C, et le fichier produit.

## Lignes 12 à 28 : les imports

```python
12 import argparse
13 import os
14 import random
15 import time
16
17 import numpy as np
18 import torch
19 import torch.nn as nn
20 from tokenizers import Tokenizer
21 from torch.utils.data import DataLoader
22 from tqdm import tqdm
23
24 from config import Config
25 from data import load_split
26 from detector import PhishingDetector, interactive
27 from metrics import print_report, report, threshold_for_recall
28 from model import PhishingTransformer, count_params
```

- **Ligne 12** : lit les options `--fresh`, `--debug` et `--no-test`.
- **Ligne 13** : `os.replace` pour l'écriture atomique des sauvegardes.
- **Ligne 14** : le générateur aléatoire de Python, dont l'état est sauvegardé pour la reprise.
- **Ligne 15** : `time` mesure la durée de chaque époque.
- **Lignes 17 à 19** : numpy, PyTorch et ses couches.
- **Ligne 20** : pour charger le tokenizer.
- **Ligne 21** : `DataLoader` découpe un jeu de données en lots et peut les mélanger.
- **Ligne 22** : `tqdm` affiche la barre de progression.
- **Lignes 24 à 28** : les fichiers du projet : configuration, chargement des données, livrable, métriques, modèle.

## Lignes 31 à 38 : `predict_probs()`

```python
31 @torch.no_grad()
32 def predict_probs(model, loader):
33     model.eval()
34     probs, labels = [], []
35     for ids, mask, feats, y in tqdm(loader, desc="  évaluation", leave=False, dynamic_ncols=True):
36         probs.append(torch.sigmoid(model(ids, mask, feats)))
37         labels.append(y)
38     return torch.cat(probs).numpy(), torch.cat(labels).numpy().astype(int)
```

- **Ligne 31** : le décorateur `@torch.no_grad()` désactive le calcul des gradients dans toute la fonction. On ne fait que prédire, ce qui économise mémoire et temps.
- **Ligne 33** : `model.eval()` passe le modèle en mode évaluation et **désactive le dropout**. Sans cela, les prédictions seraient bruitées au hasard.
- **Ligne 35** : parcourt les lots avec une barre de progression temporaire, effacée à la fin (`leave=False`). Chaque lot fournit les tokens, le masque, les indices et les étiquettes.
- **Ligne 36** : calcule les logits, les transforme en probabilités avec la sigmoïde, et les garde.
- **Ligne 37** : garde aussi les vraies étiquettes.
- **Ligne 38** : colle les lots bout à bout (`torch.cat`), puis convertit en tableaux numpy, le format attendu par scikit-learn dans `metrics.py`.

## Lignes 41 à 46 : `atomic_save()`

```python
41 def atomic_save(obj, path):
42     """Écrit dans un fichier temporaire puis renomme : un Ctrl+C pendant
43     l'écriture ne peut pas corrompre la sauvegarde précédente."""
44     tmp = str(path) + ".tmp"
45     torch.save(obj, tmp)
46     os.replace(tmp, path)
```

- **Ligne 44** : un nom temporaire, par exemple `checkpoint_last.pt.tmp`.
- **Ligne 45** : écrit d'abord la sauvegarde dans ce fichier temporaire. Elle pèse 22 Mo et prend un moment à écrire.
- **Ligne 46** : remplace l'ancien fichier par le nouveau. Ce renommage est **atomique** : il se fait d'un coup, jamais à moitié. Si on appuie sur Ctrl+C pendant l'écriture de la ligne 45, seul le `.tmp` est abîmé, et l'ancienne sauvegarde reste intacte.

## Lignes 49 à 54 : `save_resume()`

```python
49 def save_resume(cfg, model, opt, state):
50     atomic_save({
51         "model": model.state_dict(), "optimizer": opt.state_dict(), "state": state,
52         "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
53         "rng_python": random.getstate(), "model_kwargs": cfg.model_kwargs(),
54     }, cfg.last_ckpt)
```

Enregistre dans `checkpoint_last.pt` tout ce qu'il faut pour reprendre exactement :

- **Ligne 51** :
  - `model.state_dict()` : tous les poids du modèle ;
  - `opt.state_dict()` : l'état de l'optimiseur AdamW. Il garde pour chaque poids une moyenne de ses gradients passés, et les perdre changerait la suite de l'entraînement ;
  - `state` : la position (époque, étape, perte cumulée, meilleur score…).
- **Lignes 52 et 53** : l'état des trois générateurs aléatoires (PyTorch, numpy, Python). Le dropout tire ses neurones au hasard : en restaurant ces états, les tirages qui suivent la reprise sont les mêmes que sans interruption. On y ajoute la configuration du modèle, pour la vérifier à la reprise.

## Lignes 57 à 66 : `debug_first_batch()`

```python
57 def debug_first_batch(model, batch, tokenizer):
58     ids, mask, feats, y = batch
59     print("\n--- DEBUG : premier batch ---")
60     print(f"ids {tuple(ids.shape)}  mask {tuple(mask.shape)}  feats {tuple(feats.shape)}  y {tuple(y.shape)}")
61     print(f"tokens réels / email : min {int(mask.sum(1).min())}  max {int(mask.sum(1).max())}")
62     print(f"labels : {int(y.sum())} phishing / {len(y)}")
63     print(f"email 0 : {tokenizer.decode(ids[0, :30].tolist())[:200]}")
64     for name, mod in model.named_children():
65         print(f"  {name:8s} {count_params(mod):>10,} paramètres")
66     print("-----------------------------\n")
```

Appelée une seule fois en mode `--debug`, sur le premier lot. Elle affiche :

- **Ligne 60** : la forme des tenseurs, par exemple ids (32, 256), feats (32, 12). Une forme inattendue révèle tout de suite un bug.
- **Ligne 61** : le nombre de vrais tokens par email, le plus petit et le plus grand du lot (somme du masque par ligne).
- **Ligne 62** : le nombre de phishing dans le lot, pour vérifier que les étiquettes sont bien là.
- **Ligne 63** : le début du premier email, retraduit en texte (`decode`) depuis ses numéros de tokens. On vérifie ainsi que le tokenizer fonctionne.
- **Lignes 64 et 65** : le nombre de paramètres de chaque grande partie du modèle. `{:>10,}` aligne à droite avec un séparateur de milliers.

## Lignes 69 à 74 : les options

```python
69 def main():
70     ap = argparse.ArgumentParser()
71     ap.add_argument("--fresh", action="store_true", help="ignorer la reprise, repartir de zéro")
72     ap.add_argument("--debug", action="store_true", help="mini-run rapide + logs détaillés")
73     ap.add_argument("--no-test", action="store_true", help="pas de test interactif à la fin")
74     args = ap.parse_args()
```

Trois interrupteurs, faux par défaut. `--no-test` devient `args.no_test` : argparse remplace le tiret par un tiret bas.

## Lignes 76 à 84 : configuration et graines

```python
76     cfg = Config()
77     if args.debug:
78         cfg.apply_debug()
79         torch.autograd.set_detect_anomaly(True)   # signale les NaN / gradients cassés
80     cfg.run_dir.mkdir(parents=True, exist_ok=True)
81
82     torch.manual_seed(cfg.seed)
83     np.random.seed(cfg.seed)
84     random.seed(cfg.seed)
```

- **Lignes 77 et 78** : en mode debug, on réduit tout : 256 emails, 2 époques, 2 blocs, dossier `artifacts/debug`.
- **Ligne 79** : la détection d'anomalies de PyTorch. Si un calcul produit `NaN` (*not a number*), le programme s'arrête en indiquant l'opération en cause. C'est plus lent, donc réservé au debug.
- **Ligne 80** : crée le dossier de sortie si besoin.
- **Lignes 82 à 84** : fixe les trois graines aléatoires à 42. Les poids initiaux et le dropout sont alors identiques d'une exécution à l'autre : les résultats sont reproductibles.

## Lignes 86 à 95 : charger le tokenizer et les données

```python
86     tokenizer = Tokenizer.from_file(str(cfg.tokenizer_path))
87     cfg.vocab_size = tokenizer.get_vocab_size()
88
89     print("Encodage des données...")
90     train_ds, mu, sd = load_split(cfg, "train", tokenizer)
91     val_ds, _, _ = load_split(cfg, "val", tokenizer, mu, sd)
92     test_ds, _, _ = load_split(cfg, "test", tokenizer, mu, sd)
93     val_dl = DataLoader(val_ds, batch_size=128)
94     test_dl = DataLoader(test_ds, batch_size=128)
95     print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")
```

- **Ligne 86** : charge le tokenizer créé par `prepare_data.py`.
- **Ligne 87** : aligne la taille du vocabulaire du modèle sur celle du tokenizer réel. Si le BPE en a produit un peu moins que 8 000, la table d'embedding a quand même la bonne taille.
- **Ligne 90** : encode le train et récupère la moyenne (`mu`) et l'écart-type (`sd`) de ses 12 indices.
- **Lignes 91 et 92** : encode la validation et le test **avec les statistiques du train**. `_` désigne une valeur ignorée.
- **Lignes 93 et 94** : des lots de 128 pour l'évaluation. On peut prendre des lots plus gros qu'à l'entraînement, car il n'y a pas de gradients à stocker. Pas de mélange : l'ordre n'a pas d'importance pour mesurer.
- **Ligne 95** : affiche la taille des jeux, par exemple `train 17996 | val 2249 | test 1750`.

## Lignes 97 à 103 : modèle, perte et optimiseur

```python
97     model = PhishingTransformer(**cfg.model_kwargs())
98     print(f"Paramètres : {count_params(model):,}")
99
100    y_train = train_ds.tensors[3]
101    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()   # compense le déséquilibre
102    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
103    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
```

- **Ligne 97** : crée le modèle avec des poids aléatoires.
- **Ligne 98** : affiche `Paramètres : 1,859,201`.
- **Ligne 100** : les étiquettes du train, le 4ᵉ tenseur du jeu de données (index 3).
- **Ligne 101** : le nombre de légitimes divisé par le nombre de phishing, environ 1,6 en v2. Il y a 1,6 fois moins de phishing : chaque erreur sur un phishing comptera donc 1,6 fois plus.
- **Ligne 102** : la perte, l'**entropie croisée binaire**. Pour un email de classe y (0 ou 1) et de probabilité prédite p, elle vaut −[w·y·log(p) + (1 − y)·log(1 − p)], où w = `pos_weight`. Elle est faible quand le modèle est confiant et juste, et très forte quand il est confiant et faux. La version *WithLogits* reçoit directement le logit et applique la sigmoïde à l'intérieur, de façon numériquement stable.
- **Ligne 103** : l'optimiseur **AdamW**. Il adapte la taille du pas de chaque poids selon l'historique de ses gradients (moyenne et variance). Le *weight decay* (0,01) tire doucement chaque poids vers zéro, ce qui limite le sur-apprentissage. Taux d'apprentissage : 0,0003.

## Lignes 105 à 107 : l'état de l'entraînement

```python
105    # ---------------------------------------------------------------- Reprise
106    state = {"epoch": 1, "step": 0, "loss_sum": 0.0, "best_auc": -1.0,
107             "bad_epochs": 0, "finished": False, "history": []}
```

L'état d'un entraînement neuf :
- `epoch` : l'époque en cours, de 1 à 10 ;
- `step` : le dernier lot traité dans cette époque ;
- `loss_sum` : la somme des pertes de l'époque, pour afficher la moyenne ;
- `best_auc` : la meilleure PR-AUC de validation vue jusqu'ici. −1 pour que la première époque soit forcément meilleure ;
- `bad_epochs` : le nombre d'époques consécutives sans amélioration ;
- `finished` : vrai quand l'entraînement est terminé ;
- `history` : le résumé de chaque époque.

## Lignes 108 à 123 : reprendre une sauvegarde

```python
108    if cfg.last_ckpt.exists() and not args.fresh:
109        ck = torch.load(cfg.last_ckpt, weights_only=False)
110        if ck["model_kwargs"] != cfg.model_kwargs():
111            raise SystemExit("La config du modèle a changé depuis la sauvegarde : "
112                             "relance avec --fresh pour repartir de zéro.")
113        model.load_state_dict(ck["model"])
114        opt.load_state_dict(ck["optimizer"])
115        state = ck["state"]
116        torch.set_rng_state(ck["rng_torch"])
117        np.random.set_state(ck["rng_numpy"])
118        random.setstate(ck["rng_python"])
119        if state["finished"]:
120            print("Entraînement déjà terminé (utilise --fresh pour recommencer).")
121        else:
122            print(f">>> Reprise à l'epoch {state['epoch']}, étape {state['step']} "
123                  f"(meilleure PR-AUC val : {state['best_auc']:.4f})")
```

- **Ligne 108** : s'il existe une sauvegarde et que `--fresh` n'est pas demandé, on reprend.
- **Ligne 109** : charge la sauvegarde. `weights_only=False` est nécessaire ici, car l'état du générateur de Python n'est pas un simple tenseur. C'est sans risque : ce fichier a été écrit par ce même script. Le livrable, lui, se charge en mode sûr.
- **Lignes 110 à 112** : si les dimensions du modèle ont changé depuis la sauvegarde, les anciens poids ne rentrent plus dans le nouveau modèle. On s'arrête avec un message clair.
- **Lignes 113 et 114** : restaure les poids et l'état d'AdamW.
- **Ligne 115** : restaure la position.
- **Lignes 116 à 118** : restaure les trois générateurs aléatoires.
- **Lignes 119 à 123** : informe l'utilisateur. Si l'entraînement était déjà terminé, la boucle sera sautée et on passera directement à l'export.

## Lignes 124 à 126 : repartir de zéro

```python
124    elif args.fresh:
125        for p in (cfg.last_ckpt, cfg.best_weights):
126            p.unlink(missing_ok=True)
```

Avec `--fresh`, on supprime l'ancienne sauvegarde et les anciens meilleurs poids. `missing_ok=True` évite une erreur s'ils n'existent pas. C'est obligatoire après un nouveau `prepare_data.py`, car le tokenizer a changé.

## Lignes 129 à 136 : début d'une époque

```python
129    interrupted = False
130    try:
131        while not state["finished"] and state["epoch"] <= cfg.max_epochs:
132            epoch = state["epoch"]
133            # Ordre des batchs déterministe par epoch -> on peut sauter les étapes déjà faites
134            g = torch.Generator().manual_seed(cfg.seed + epoch)
135            train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
136            n_steps, t0 = len(train_dl), time.time()
```

- **Ligne 129** : passera à vrai si l'utilisateur appuie sur Ctrl+C.
- **Ligne 130** : le bloc `try` permet d'intercepter Ctrl+C (ligne 188).
- **Ligne 131** : boucle sur les époques tant que l'entraînement n'est pas fini et qu'on n'a pas dépassé 10 époques.
- **Lignes 134 et 135** : l'astuce de la reprise. Chaque époque mélange le train avec sa propre graine (42 + numéro d'époque). Après une reprise à l'époque 5, étape 200, on recrée exactement le même ordre de lots qu'avant l'interruption, et on peut sauter les 200 premiers. Des lots de 32 emails, mélangés à chaque époque pour que le modèle ne voie pas les emails toujours dans le même ordre.
- **Ligne 136** : le nombre de lots (563 en v2) et l'heure de début.

## Lignes 138 à 146 : la barre de progression

```python
138            model.train()
139            # Barre de progression : démarre à l'étape déjà atteinte en cas de reprise
140            with tqdm(total=n_steps, initial=state["step"], desc=f"Epoch {epoch}/{cfg.max_epochs}",
141                      unit="étape", dynamic_ncols=True) as bar:
142                for step, batch in enumerate(train_dl, 1):
143                    if step <= state["step"]:
144                        continue                               # déjà fait avant l'arrêt
145                    if args.debug and epoch == 1 and step == 1:
146                        debug_first_batch(model, batch, tokenizer)
```

- **Ligne 138** : `model.train()` réactive le dropout, éteint pendant la validation.
- **Lignes 140 et 141** : la barre de progression. `initial` la fait démarrer à l'étape déjà atteinte, `dynamic_ncols` l'adapte à la largeur du terminal.
- **Ligne 142** : parcourt les lots, numérotés à partir de 1.
- **Lignes 143 et 144** : saute les lots déjà traités avant l'interruption.
- **Lignes 145 et 146** : en debug, affiche le détail du tout premier lot.

## Lignes 147 à 152 : une étape d'apprentissage

```python
147                    ids, mask, feats, y = batch
148                    loss = loss_fn(model(ids, mask, feats), y)
149                    opt.zero_grad()
150                    loss.backward()
151                    gnorm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
152                    opt.step()
```

Le cœur de l'apprentissage, répété des milliers de fois :

- **Ligne 147** : sépare les quatre tenseurs du lot.
- **Ligne 148** : **passe avant**. Le modèle calcule les 32 logits, et la perte mesure l'erreur moyenne sur le lot.
- **Ligne 149** : remet les gradients à zéro, car PyTorch les additionne d'une étape à l'autre par défaut.
- **Ligne 150** : **rétropropagation**. PyTorch calcule automatiquement, pour chacun des 1 859 201 poids, la dérivée de la perte : dans quel sens et de combien il faut bouger ce poids pour réduire l'erreur.
- **Ligne 151** : si la norme globale des gradients dépasse 1,0, tous les gradients sont réduits proportionnellement. Un lot inhabituel ne peut donc pas provoquer un pas énorme qui détruirait l'apprentissage. La fonction renvoie la norme d'origine, affichée en debug.
- **Ligne 152** : AdamW met à jour tous les poids.

## Lignes 154 à 162 : suivi et sauvegarde

```python
154                    state["step"] = step
155                    state["loss_sum"] += loss.item()
156                    info = {"loss": f"{state['loss_sum'] / step:.4f}"}
157                    if args.debug:
158                        info["grad_norm"] = f"{gnorm:.3f}"
159                    bar.set_postfix(info)
160                    bar.update(1)
161                    if step % cfg.save_every == 0:
162                        save_resume(cfg, model, opt, state)
```

- **Ligne 154** : retient la position.
- **Ligne 155** : ajoute la perte du lot. `.item()` convertit le tenseur en nombre Python.
- **Ligne 156** : la perte moyenne depuis le début de l'époque, affichée à droite de la barre. Elle doit baisser.
- **Lignes 157 et 158** : en debug, affiche aussi la norme du gradient.
- **Lignes 159 et 160** : met à jour le texte et l'avancement de la barre.
- **Lignes 161 et 162** : toutes les 200 étapes, écrit une sauvegarde de reprise. `%` est le reste de la division.

## Lignes 164 à 171 : évaluer sur la validation

```python
164            # --- fin d'epoch : validation ---
165            p_val, y_val = predict_probs(model, val_dl)
166            r = report(y_val, p_val)
167            train_loss = state["loss_sum"] / n_steps
168            print(f"Epoch {epoch} | loss {train_loss:.4f} | val PR-AUC {r['pr_auc']:.4f} "
169                  f"| rappel {r['recall']:.4f} | précision {r['precision']:.4f} "
170                  f"| {time.time() - t0:.0f}s")
171            state["history"].append({"epoch": epoch, "loss": train_loss, "val_pr_auc": r["pr_auc"]})
```

- **Lignes 165 et 166** : prédit sur la validation, des emails jamais utilisés pour apprendre, et calcule les métriques au seuil de 0,5.
- **Ligne 167** : la perte moyenne de l'époque.
- **Lignes 168 à 170** : affiche le bilan, par exemple `Epoch 4 | loss 0.0414 | val PR-AUC 0.9934 | ... | 1280s`.
- **Ligne 171** : ajoute ce bilan à l'historique.

## Lignes 173 à 178 : garder le meilleur modèle

```python
173            if r["pr_auc"] > state["best_auc"]:
174                state["best_auc"], state["bad_epochs"] = r["pr_auc"], 0
175                atomic_save(model.state_dict(), cfg.best_weights)
176                print("  -> meilleur modèle sauvegardé")
177            else:
178                state["bad_epochs"] += 1
```

- **Ligne 173** : la PR-AUC de validation est-elle la meilleure jamais vue ?
- **Lignes 174 à 176** : si oui, on la retient, on remet le compteur d'échecs à 0, et on enregistre ces poids dans `best_weights.pt`.
- **Lignes 177 et 178** : sinon, on compte une époque sans progrès. Les poids continuent d'évoluer, mais les meilleurs sont déjà en sécurité.

Pourquoi la PR-AUC et pas la perte ? La PR-AUC mesure directement la capacité à classer les phishing au-dessus des emails légitimes, ce qui nous intéresse vraiment.

## Lignes 180 à 186 : préparer l'époque suivante

```python
180            state.update(epoch=epoch + 1, step=0, loss_sum=0.0)
181            if state["bad_epochs"] >= cfg.patience:
182                print("Early stopping.")
183                state["finished"] = True
184            if state["epoch"] > cfg.max_epochs:
185                state["finished"] = True
186            save_resume(cfg, model, opt, state)
```

- **Ligne 180** : passe à l'époque suivante et remet l'étape et la perte à zéro.
- **Lignes 181 à 183** : **early stopping**. Après 2 époques de suite sans progrès, on arrête : le modèle commence à apprendre le train par cœur (sur-apprentissage) au lieu de généraliser. En v1, l'arrêt a eu lieu après l'époque 8 ; la meilleure était l'époque 6.
- **Lignes 184 et 185** : on arrête aussi après la 10ᵉ époque.
- **Ligne 186** : une sauvegarde de reprise en fin d'époque.

## Lignes 188 à 193 : Ctrl+C

```python
188    except KeyboardInterrupt:
189        interrupted = True
190        save_resume(cfg, model, opt, state)
191        print(f"\n>>> Interrompu à l'epoch {state['epoch']}, étape {state['step']}. "
192              f"Position sauvegardée : relance  python train.py{' --debug' if args.debug else ''}  "
193              f"pour reprendre.")
```

- **Ligne 188** : Ctrl+C lève l'exception `KeyboardInterrupt`. Elle est interceptée ici au lieu d'arrêter brutalement le programme.
- **Ligne 190** : sauvegarde immédiatement la position exacte.
- **Lignes 191 à 193** : indique la commande pour reprendre, avec `--debug` si on était en mode debug.

## Lignes 196 à 198 : y a-t-il un modèle à exporter ?

```python
196    if not cfg.best_weights.exists():
197        print("Aucune epoch terminée : pas encore de modèle à exporter.")
198        return
```

Si l'interruption a eu lieu avant la fin de la première époque, il n'y a pas encore de meilleurs poids : on s'arrête là.

## Lignes 200 à 206 : seuil et mesure sur le test

```python
200    model.load_state_dict(torch.load(cfg.best_weights, weights_only=True))
201    p_val, y_val = predict_probs(model, val_dl)
202    thr = threshold_for_recall(y_val, p_val, cfg.target_recall)   # seuil choisi sur la VAL
203    p_test, y_test = predict_probs(model, test_dl)
204    r05, rthr = report(y_test, p_test, 0.5), report(y_test, p_test, thr)
205    print_report("Transformer – test @0.5", r05)
206    print_report(f"Transformer – test @seuil rappel {cfg.target_recall}", rthr)
```

- **Ligne 200** : recharge les **meilleurs** poids, pas ceux de la dernière époque, qui peuvent être moins bons.
- **Ligne 201** : prédit sur la validation.
- **Ligne 202** : choisit le seuil sur la validation : le plus haut qui détecte encore 97 % des phishing.
- **Ligne 203** : prédit sur le test, **pour la première et seule fois**. Le test n'a servi à aucune décision : son score est une estimation honnête.
- **Lignes 204 à 206** : mesure et affiche les scores au seuil de 0,5 et au seuil choisi.

## Lignes 207 à 210 : le test transactionnel (v2)

```python
207    if (cfg.processed_dir / "test_synth.csv").exists():   # emails transactionnels jamais vus
208        synth_ds, _, _ = load_split(cfg, "test_synth", tokenizer, mu, sd)
209        p_s, y_s = predict_probs(model, DataLoader(synth_ds, batch_size=128))
210        print_report(f"Transformer – test_synth (transactionnel) @seuil {thr:.3f}", report(y_s, p_s, thr))
```

Si le test transactionnel existe, il est mesuré aussi, au seuil choisi. Ce score montre si le modèle reconnaît maintenant les notifications légitimes de banques, de transporteurs ou de services en ligne.

## Lignes 212 à 214 : exporter le livrable

```python
212    PhishingDetector.export(cfg.model_path, model, cfg, tokenizer, mu, sd, thr, rthr)
213    status = "PROVISOIRE (entraînement non terminé)" if interrupted else "final"
214    print(f"\nLivrable {status} : {cfg.model_path}")
```

- **Ligne 212** : écrit `phishing_model.pt` avec, dans un seul fichier, les poids, la configuration, le tokenizer, les statistiques des indices, le seuil et les scores du test.
- **Lignes 213 et 214** : précise si le livrable est final ou provisoire (export après un Ctrl+C, à partir des meilleurs poids connus).

## Lignes 216 et 217 : le test interactif

```python
216    if not interrupted and not args.no_test:
217        interactive(PhishingDetector.load(cfg.model_path), debug=args.debug)
```

Si l'entraînement est allé au bout et que `--no-test` n'est pas demandé, on recharge le livrable **depuis le fichier**. Cela vérifie au passage que l'export fonctionne. On lance ensuite la boucle où l'on colle un email pour le tester.

## Lignes 220 et 221 : le point d'entrée

```python
220 if __name__ == "__main__":
221     main()
```
