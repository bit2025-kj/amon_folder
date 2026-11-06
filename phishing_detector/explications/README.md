# Explications du code, fichier par fichier

Chaque fichier `.md` de ce dossier reprend un fichier Python du projet **dans l'ordre**, par blocs de lignes numérotées. Il montre d'abord le code, puis explique ce que fait chaque ligne et pourquoi elle est écrite ainsi. Les numéros de ligne correspondent au code au 5 novembre 2026 (version v2, avec augmentation).

## Ordre de lecture conseillé

Le même ordre que celui où les données traversent le projet :

| # | Fichier | Explication | Ce qu'il fait |
| --- | --- | --- | --- |
| 1 | `config.py` | [config.md](config.md) | tous les réglages : chemins, tailles, hyperparamètres |
| 2 | `text_utils.py` | [text_utils.md](text_utils.md) | normalisation du texte et 12 indices manuels |
| 3 | `augment.py` | [augment.md](augment.md) | emails transactionnels synthétiques et bruit (v2) |
| 4 | `prepare_data.py` | [prepare_data.md](prepare_data.md) | étape 1 : nettoyage, déduplication, découpage, augmentation, tokenizer BPE |
| 5 | `data.py` | [data.md](data.md) | transformation des CSV en tenseurs PyTorch |
| 6 | `metrics.py` | [metrics.md](metrics.md) | rappel, précision, F1, PR-AUC, choix du seuil |
| 7 | `baseline.py` | [baseline.md](baseline.md) | étape 2 : référence TF-IDF + régression logistique |
| 8 | `model.py` | [model.md](model.md) | le Transformer : attention, blocs, tête de classification |
| 9 | `train.py` | [train.md](train.md) | étape 3 : entraînement, reprise, early stopping, export |
| 10 | `detector.py` | [detector.md](detector.md) | le livrable : export, chargement sûr, prédiction, debug |
| 11 | `predict.py` | [predict.md](predict.md) | étape 4 : utilisation en ligne de commande |
| 12 | `app.py` | [app.md](app.md) | interface web : analyse, temps de réponse, statistiques, évaluation |

## Le chemin des données en une phrase par étape

1. `prepare_data.py` lit 18 650 emails, les normalise (`text_utils.py`), retire les doublons, les découpe en train, validation et test, ajoute les emails synthétiques et le bruit (`augment.py`), puis entraîne le tokenizer BPE.
2. `baseline.py` mesure une référence simple, que le Transformer doit égaler ou battre.
3. `train.py` transforme les données en tenseurs (`data.py`), entraîne `model.py`, garde la meilleure époque selon la validation (`metrics.py`), choisit le seuil, puis écrit `phishing_model.pt` (`detector.py`).
4. `predict.py` et `app.py` chargent ce seul fichier et analysent de nouveaux emails, sans réentraîner.
