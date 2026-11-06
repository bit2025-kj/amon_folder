# Voix off : « Modélisation du réseau de neurones »

## Consigne de voix (pour l'outil de synthèse vocale)

> Voix française naturelle, posée et pédagogique, comme un professeur qui explique à un public curieux.
> Débit modéré (environ 150 mots par minute), articulation nette, ton chaleureux et assuré.
> Courtes pauses entre les phrases, pause plus longue entre les étapes numérotées.
> Pas de musique, pas d'effet. Prononcer les mots anglais à l'anglaise : « password », « login », « feed-forward ».

Les nombres sont écrits en toutes lettres pour que la voix les lise correctement.

---

## 01 · Le neurone artificiel (≈ 60 s)

Comment un réseau de neurones peut-il reconnaître un email de phishing ? Voici notre modèle, construit de zéro.

Tout commence par un seul neurone.

Un neurone reçoit des nombres en entrée : ici, x1, x2, x3 et x4.

Chaque connexion porte un poids, appris pendant l'entraînement. En bleu, les poids positifs. En rouge, les poids négatifs.

Le neurone multiplie chaque entrée par son poids, additionne le tout, puis ajoute un biais. Résultat : z égale zéro virgule quatre-vingt-douze.

Une fonction d'activation, ici ReLU, décide de ce qui passe. z est positif : le neurone s'allume.

Avec d'autres entrées, z vaut moins zéro virgule vingt-neuf. ReLU renvoie zéro : le neurone reste éteint.

Plusieurs neurones reliés aux mêmes entrées forment une couche : quatre entrées, cinq neurones, vingt poids.

Empilez les couches, et le signal circule de gauche à droite. Notre modèle contient un million huit cent cinquante-neuf mille paramètres de ce type.

---

## 02 · Le réseau complet (≈ 54 s)

Voici le modèle entier. À gauche, les mots de l'email. Chaque cercle bleu représente le vecteur d'un mot : cent vingt-huit nombres.

Dans l'attention, tous les mots sont connectés entre eux. Dans le feed-forward, chaque mot est traité seul.

Suivons le signal.

Un : chaque mot devient un vecteur. C'est l'embedding.

Deux : l'attention mélange l'information de tous les mots.

Trois : le feed-forward transforme chaque mot séparément.

Quatre : ce bloc est répété quatre fois.

Cinq : on fait la moyenne des mots, et on ajoute douze indices calculés à la main, comme les liens ou les mots d'urgence.

Six : soixante-quatre neurones combinent tout.

Sept : un seul neurone de sortie donne la probabilité de phishing.

Le signal a traversé tout le réseau en un seul passage. Le voici en accéléré.

---

## 03 · Embedding (≈ 29 s)

Zoom sur l'embedding. Le mot « password » porte le numéro trois mille neuf cent cinq.

Ce numéro désigne une ligne dans une table apprise de huit mille lignes. Cette ligne devient une colonne de cent vingt-huit valeurs.

On lui ajoute un vecteur de position : sans lui, le réseau ignorerait l'ordre des mots.

Chaque mot de l'email obtient ainsi sa colonne. Pour la suite, on résume chaque colonne par un seul cercle.

---

## 04 · Self-attention (≈ 47 s)

L'attention permet aux mots d'échanger de l'information.

Chaque mot passe d'abord dans trois petites couches : Q, ce qu'il cherche ; K, ce qu'il offre ; V, son contenu.

Le mot « password » compare sa requête aux clés des autres mots. Plus elles se ressemblent, plus la connexion est forte. Il récupère surtout l'information de « urgent », « login » et « http ».

Tous les mots font la même chose en même temps : vingt-cinq connexions ici, plus de soixante-cinq mille dans le vrai modèle.

Quatre têtes travaillent en parallèle, chacune avec ses propres connexions.

Enfin, les positions de remplissage sont masquées : aucune information ne vient d'elles.

---

## 05 · Feed-forward et résiduel (≈ 37 s)

Après l'attention, chaque mot passe dans un petit réseau : le feed-forward. Son vecteur s'élargit à cinq cent douze neurones, puis revient à cent vingt-huit.

Avec l'activation GELU, certains neurones s'allument fort, d'autres restent presque éteints.

Une connexion résiduelle ajoute l'entrée à la sortie : l'information d'origine n'est jamais perdue. Une normalisation, LayerNorm, garde les valeurs stables.

Ce bloc, attention plus feed-forward, est empilé quatre fois. Chaque passage affine le sens.

---

## 06 · Tête de classification (≈ 32 s)

Il faut maintenant passer de plusieurs mots à une seule réponse.

On fait la moyenne des vecteurs de tous les vrais mots. À côté, douze indices calculés à la main : liens, adresse IP, urgence, identifiants.

Soixante-quatre neurones combinent ces cent quarante valeurs. ReLU éteint ceux dont la somme est négative.

Un dernier neurone produit un score, que la sigmoïde transforme en probabilité. Au-dessus du seuil, l'email est signalé comme phishing.

---

## 07 · Apprentissage (≈ 43 s)

Comment le réseau apprend-il ? On lui montre un email dont on connaît la réponse : c'est un phishing.

Au premier passage, il prédit zéro virgule quarante. L'erreur est grande.

La rétropropagation renvoie cette erreur de la sortie vers l'entrée, et l'optimiseur AdamW ajuste chaque poids pour la réduire.

Deuxième passage : zéro virgule soixante et onze. Troisième : zéro virgule quatre-vingt-treize. L'erreur a presque disparu.

Ce cycle est répété sur des milliers d'emails : quatre cent trente-huit lots de trente-deux, à chaque époque.

---

## 08 · Le flux complet (≈ 31 s)

Voyons le flux complet, en démonstration.

Un premier email : « URGENT, vérifiez votre mot de passe ». Le signal traverse le réseau, et la sortie s'allume : phishing.

Un second email : « Bonjour l'équipe, la réunion est déplacée à jeudi ». Cette fois, la sortie reste éteinte : légitime.

Embedding, attention, feed-forward, tête de classification : un million huit cent cinquante-neuf mille paramètres au service de la détection.

Merci.
