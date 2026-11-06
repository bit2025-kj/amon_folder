# augment.py, ligne par ligne

**Rôle :** produire les données supplémentaires de la v2 :
1. des **emails transactionnels synthétiques** (banque, colis, abonnement…), chacun en version légitime et en version phishing ;
2. une fonction de **bruit** qui dégrade un email (fautes de frappe, transferts, signatures mobiles…).

Pourquoi : la v1 classait phishing à 99,9 % une vraie notification bancaire. Dans le dataset d'origine, presque tous les emails où une entreprise écrit à son client sont du phishing, et le modèle avait appris « banque = phishing ».

Utilisé par : `prepare_data.py`.

---

## Lignes 1 à 23 : la docstring

```python
1  """Données supplémentaires : emails transactionnels synthétiques + bruit d'augmentation.
...
14 Ainsi le modèle ne peut pas apprendre « banque = légitime » : il doit regarder les indices.
...
22 La casse et les espaces ne sont pas bruités : normalize() les efface de toute façon.
23 """
```

Elle résume la stratégie :
- **lignes 3 à 5** : le problème de la v1 ;
- **lignes 7 à 14** : le principe des **paires**. Pour chaque scénario, la version légitime et la version phishing partagent marques, salutations, signatures et bas de page. Seuls quatre vrais indices diffèrent : le domaine du lien, ce qu'on demande, la pression exercée et le canal de contact. Si seules les versions légitimes parlaient de banque, le modèle apprendrait le raccourci inverse, « banque = légitime » ;
- **lignes 16 et 17** : un élément sur cinq de chaque liste est réservé au test, pour mesurer la généralisation ;
- **lignes 19 à 22** : la liste des perturbations du bruit, appliquées au train seulement et de la même façon aux deux classes.

## Lignes 24 et 25 : les imports

```python
24 import random
25 import re
```

- **Ligne 24** : le générateur aléatoire de Python. Il est toujours utilisé à travers un objet `rng` créé avec une graine, pour que tout soit reproductible.
- **Ligne 25** : les expressions régulières, pour le mélange de phrases.

## Lignes 28 à 51 : les marques

```python
28 BRANDS = {
29     "bank": [("Chase", "chase.com"), ("Wells Fargo", "wellsfargo.com"), ...],
35     "shop": [("Amazon", "amazon.com"), ...],
39     "carrier": [("UPS", "ups.com"), ...],
42     "service": [("Netflix", "netflix.com"), ...],
47     "payroll": [("ADP", "adp.com"), ...],
49     "gov": [("IRS", "irs.gov"), ...],
51 }
```

Un dictionnaire de 60 marques réelles, rangées en 6 secteurs : 15 banques, 10 boutiques, 10 transporteurs, 15 services en ligne, 5 services de paie et 5 administrations. Chaque marque est un couple (nom, domaine officiel). Le domaine sert à fabriquer les vrais liens des emails légitimes et à imiter les faux liens des emails de phishing.

## Lignes 54 à 56 : `held_out()`

```python
54 def held_out(i: int) -> bool:
55     """1 élément sur 5 est réservé au test synthétique."""
56     return i % 5 == 4
```

Renvoie vrai pour les positions 4, 9, 14… de chaque liste (le reste de la division par 5 vaut 4). Ces éléments sont réservés au test transactionnel.

## Lignes 59 à 61 : `pick()`

```python
59 def pick(rng, pool, split):
60     items = [x for i, x in enumerate(pool) if held_out(i) == (split == "test")]
61     return rng.choice(items or pool)
```

- **Ligne 60** : pour le test (`split == "test"`), on ne garde que les éléments réservés ; pour le train et la validation, uniquement les autres. `enumerate` donne la position et la valeur de chaque élément.
- **Ligne 61** : tire un élément au hasard. `items or pool` est une sécurité : si le filtre ne laisse rien (liste de moins de 5 éléments), on prend dans la liste complète.

Résultat : le test contient des marques et des formulations **jamais vues pendant l'entraînement**. Il mesure si le modèle a compris les indices, et non s'il a mémorisé des phrases. Pour les listes de 5 phrases, le test n'en reçoit qu'une seule par liste, la 5ᵉ, ce qui le rend moins varié que le train.

## Lignes 64 à 77 : les éléments variables

```python
64 FIRST = ["James", "Mary", ...]           # 25 prénoms
67 LAST = ["Smith", "Johnson", ...]         # 20 noms de famille
69 CITIES = ["Chicago", "Houston", ...]     # 15 villes (connexion suspecte)
71 DEVICES = ["Chrome on Windows", ...]     # 7 appareils (nouvelle connexion)
73 ITEMS = ["wireless headphones", ...]     # 12 articles (commandes)
76 MONTHS = ["January", ..., "December"]    # les 12 mois
```

Les valeurs tirées au hasard pour remplir les emails, afin que deux emails générés ne se ressemblent pas mot pour mot. Ces listes ne sont pas filtrées par `pick` : un prénom ne donne aucun indice sur la classe.

## Lignes 80 à 98 : les parties communes aux deux classes

```python
80 GREETINGS = ["Dear Customer,", "Dear {first},", ...]           # 10 salutations
82 CLOSINGS = ["Thank you for choosing {brand}.", ...]             # 10 formules finales
86 SIGNATURES = ["Best regards,\n{brand} Customer Support", ...]   # 10 signatures
91 FOOTERS = ["", "", "", "This is an automated message...", ...]  # 10 bas de page
```

Ces morceaux apparaissent **à l'identique** dans les emails légitimes et dans les emails de phishing. C'est le cœur de la méthode : « Dear Customer » ou « Thank you for banking with us » ne permettent plus de deviner la classe.

- **Lignes 80 et 81** : les accroches entre `{...}` sont des emplacements remplis plus tard (ligne 409), par exemple `{first}` par un prénom.
- **Lignes 86 à 90** : `\n` est un retour à la ligne dans la signature.
- **Lignes 91 à 98** : les trois chaînes vides donnent 3 chances sur 10 de ne pas avoir de bas de page. La ligne 97, « {brand} will never ask you for your password », est volontairement partagée : sinon le modèle apprendrait que cette phrase garantit un email légitime, alors que de vrais phishings la recopient.

## Lignes 99 à 124 : les objets d'email

```python
99  SUBJECTS = {
100     "transaction": ["Transaction alert", "Your recent transaction", ...],
...
122     "shared": ["Document shared with you", "{first2} shared a file", ...],
124 }
```

Cinq objets possibles par scénario. Ils sont communs aux deux classes ; la version phishing peut seulement y ajouter un préfixe d'urgence (ligne 393).

## Lignes 128 à 333 : les 12 scénarios

```python
128 SCENARIOS = {
129     "transaction": dict(
130         sector="bank",
131         body=["Your recent transaction of {amount} was successfully processed on {date}.", ...],
136         legit=["You can review the transaction details in your online banking application.", ...],
141         phish=["If you do not recognize this transaction, cancel it immediately at {url_phish}.", ...]),
...
333 }
```

Chaque scénario a quatre champs :
- **`sector`** : le secteur où l'on tire la marque ;
- **`body`** : 5 phrases de contexte, **communes aux deux classes** (« Your recent transaction of $125.00 was processed… ») ;
- **`legit`** : 5 actions d'un email légitime ;
- **`phish`** : 5 actions d'un email de phishing.

| Scénario | Lignes | Secteur | Exemple d'action légitime | Exemple d'action phishing |
| --- | --- | --- | --- | --- |
| transaction | 129–145 | banque | « call the number on the back of your card » | « reply with your full card number, expiry date and CVV » |
| statement | 146–162 | banque | « open the {brand} app and go to Statements » | « re-confirm your details at {url_phish} » |
| signin | 163–179 | service | « If this was you, you can ignore this email » | « suspended in 24 hours unless you verify… » |
| password | 180–196 | service | « reset your password from the {brand} app » | « send us your old password and your phone number » |
| order | 197–213 | boutique | « track your order in Your Orders at {url_legit} » | « Update your card details… or your order will be cancelled » |
| shipping | 214–230 | transporteur | « We will try again on the next business day » | « A redelivery fee of $1.99 is required » |
| subscription | 231–247 | service | « manage or cancel your plan in your account settings » | « Reply with your card number and security code » |
| otp | 248–264 | service | « Never share this code with anyone » | « A support agent will call you. Please give them this code » |
| payroll | 265–281 | paie | « contact your HR department directly » | « Open the attached file and enable macros » |
| storage | 282–298 | service | « free up space or upgrade your plan » | « increase your quota for free: {url_phish} » |
| tax | 299–315 | administration | « call the number listed on your official notice » | « Failure to verify… will result in legal action » |
| shared | 316–332 | service | « You can find it in Shared with me » | « Enter your Microsoft 365 credentials at {url_phish} » |

Les actions légitimes renvoient vers un canal **que le destinataire connaît déjà** (l'application, le site officiel, le numéro de la carte) ou ne demandent rien. Les actions de phishing demandent un identifiant, une carte, un code ou un paiement, souvent avec un délai ou une menace, et presque toujours via un lien ou une pièce jointe. Ce sont exactement les indices qu'un humain regarde.

Dans le scénario `shared` (ligne 318 et suivantes), les `\"` sont des guillemets à l'intérieur d'une chaîne Python.

## Lignes 334 à 337 : les dernières listes

```python
334 DOCS = ["Q3 budget", "Invoice 2291", ...]          # noms de documents partagés
336 PLANS = ["Premium", "Standard", ...]               # formules d'abonnement
337 URGENCY = ["URGENT: ", "Action required: ", ...]   # préfixes d'objet, phishing uniquement
```

`URGENCY` est la seule liste propre au phishing dans l'en-tête. Elle n'est utilisée que dans 40 % des objets de phishing (ligne 392), donc beaucoup de phishings n'ont aucun mot d'urgence dans l'objet.

## Lignes 340 à 358 : `phish_url()`

```python
340 def phish_url(rng, name, domain, split):
341     host = domain.split(".")[0]
342     words = ["secure", "login", "verify", "account", "update", "support", "billing", "alert", "service"]
343     w1, w2 = rng.sample(words, 2)
344     tld = rng.choice(["com", "net", "info", "online", "xyz", "co", "top", "site"])
345     leet = host.replace("o", "0").replace("l", "1").replace("i", "1").replace("e", "3")
```

- **Ligne 341** : le nom de domaine sans extension, par exemple `chase` pour `chase.com`.
- **Ligne 342** : les mots que les fraudeurs collent aux noms de domaine pour paraître officiels.
- **Ligne 343** : en tire deux, différents.
- **Ligne 344** : une extension, souvent bon marché (`.xyz`, `.top`, `.site`).
- **Ligne 345** : la version « déguisée » du nom : `o` devient `0`, `l` devient `1`, `i` devient `1` et `e` devient `3`. Par exemple `chase` donne `chas3`, et `google` donne `g00g13`.

```python
346     patterns = [
347         f"http://{host}-{w1}-{w2}.{tld}/login",                     # chase-secure-login.top/login
348         f"http://{w1}-{host}.{tld}/{w2}",                           # verify-chase.net/update
349         f"http://{rng.randint(11, 223)}...{host}/{w1}",             # adresse IP
350         f"http://{host}.{w1}-{w2}.{tld}/",                          # chase.secure-login.xyz
351         f"https://{rng.choice(['bit.ly', ...])}/{...}x",            # lien raccourci
352         f"http://www.{leet}.{tld}/{w1}",                            # lettres déguisées
353         f"https://{host}{rng.randint(1, 99)}.{tld}/{w1}/{w2}",      # chase37.online
354         f"http://{domain}.{w1}.{tld}/session",                      # chase.com.verify.info
355         f"https://{w1}.{host}-{tld}.com/",                          # sous-domaine trompeur
356         f"http://{host}-{w2}.web.app/{w1}",                         # hébergement gratuit
357     ]
358     return pick(rng, patterns, split)
```

Dix techniques réelles de faux liens : nom de marque noyé dans un autre domaine, adresse IP, lien raccourci, lettres déguisées, vrai domaine utilisé comme sous-domaine d'un faux, hébergeur gratuit… Certaines utilisent `https`, pour que le modèle n'apprenne pas « https = sûr ». Le 5ᵉ et le 10ᵉ motif (positions 4 et 9 : `chase.secure-login.xyz` et l'hébergeur gratuit `web.app`) sont réservés au test : le modèle ne les voit jamais pendant l'entraînement.

Les commentaires à droite du bloc ci-dessus sont des exemples ajoutés pour cette explication ; ils ne figurent pas dans le fichier.

## Lignes 361 à 365 : `legit_url()`

```python
361 def legit_url(rng, domain, split):
362     paths = ["", "account", "help", "security", "activity", "orders", "settings", "billing",
363              "myaccount", "support"]
364     p = pick(rng, paths, split)
365     return f"https://www.{domain}/{p}" if p else f"https://www.{domain}"
```

Un vrai lien : toujours `https://www.` suivi du **domaine officiel**, plus éventuellement une page (`/account`, `/help`…). Le chemin vide donne la page d'accueil.

## Lignes 368 à 387 : `make_email()`, préparer les valeurs

```python
368 def make_email(rng, label: int, split: str) -> str:
369     kind = rng.choice(list(SCENARIOS))
370     sc = SCENARIOS[kind]
371     name, domain = pick(rng, BRANDS[sc["sector"]], split)
372     shop = pick(rng, BRANDS["shop"], split)[0]
373     day = rng.randint(1, 28)
374     ctx = dict(
375         brand=name, shop=shop, first=rng.choice(FIRST), last=rng.choice(LAST),
...
386         fake_email=f"{rng.choice(['security', 'verify', 'support'])}@{domain.split('.')[0]}-help.{...}",
387     )
```

- **Ligne 368** : génère un email de classe `label` (0 = légitime, 1 = phishing).
- **Lignes 369 et 370** : tire un scénario au hasard parmi les 12.
- **Ligne 371** : tire une marque du bon secteur (une banque pour une transaction), en respectant la réserve du test.
- **Ligne 372** : une boutique, pour « Your parcel from {shop} ». `[0]` garde le nom.
- **Ligne 373** : un jour de 1 à 28, valable pour tous les mois.
- **Lignes 374 à 387** : `ctx` contient une valeur pour chaque emplacement `{...}` :
  - `amount` : un montant de 4 à 2 500, en dollars (2 chances sur 4), livres ou euros. `:,.2f` ajoute le séparateur de milliers et 2 décimales, par exemple `$1,234.56` ;
  - `date`, `date2`, `month`, `year` : des dates plausibles ;
  - `last4` : les 4 derniers chiffres d'une carte. `:04d` complète avec des zéros (`0042`) ;
  - `order`, `track` : un numéro de commande et un numéro de suivi au format des transporteurs (`1Z…` pour UPS) ;
  - `code` : un code de vérification à 6 chiffres ;
  - `url_legit`, `url_phish` : les deux types de liens. Les deux sont préparés, mais seul celui de la classe demandée apparaîtra, car les phrases légitimes n'utilisent que `{url_legit}` et les phrases de phishing que `{url_phish}` ;
  - `fake_email` : une fausse adresse de « support », par exemple `security@chase-help.net`.

## Lignes 388 à 409 : `make_email()`, assembler

```python
388     action_pool = sc["legit"] if label == 0 else sc["phish"]
389     parts = []
390     if rng.random() < 0.5:
391         subj = pick(rng, SUBJECTS[kind], split)
392         if label == 1 and rng.random() < 0.4:
393             subj = rng.choice(URGENCY) + subj
394         parts.append("Subject: " + subj)
```

- **Ligne 388** : **la seule différence structurelle entre les deux classes** : la liste d'actions utilisée.
- **Lignes 390 à 394** : une fois sur deux, l'email commence par un objet. Pour un phishing, 4 fois sur 10, l'objet reçoit en plus un préfixe d'urgence (« URGENT: Transaction alert »).

```python
395     parts.append(pick(rng, GREETINGS, split))
396     parts.append(pick(rng, sc["body"], split))
397     parts.append(pick(rng, action_pool, split))
398     # Une 2e action (même classe) dans 40 % des cas : emails plus variés
399     if rng.random() < 0.4:
400         extra = rng.choice(action_pool)
401         if extra != parts[-1]:
402             parts.append(extra)
```

- **Ligne 395** : la salutation (commune).
- **Ligne 396** : la phrase de contexte (commune).
- **Ligne 397** : l'action, propre à la classe.
- **Lignes 399 à 402** : 4 fois sur 10, une deuxième action de la même classe, si elle diffère de la première. Cette deuxième action est tirée dans toute la liste (`rng.choice`, pas `pick`) : elle peut donc utiliser une formulation réservée au test. C'est une petite fuite entre les phrases du train et celles du test.

```python
403     parts.append(pick(rng, CLOSINGS, split))
404     parts.append(pick(rng, SIGNATURES, split))
405     footer = rng.choice(FOOTERS)       # mêmes bas de page pour les 2 classes (même la phrase
406     if footer:                         # « nous ne demandons jamais votre mot de passe »)
407         parts.append(footer)
408     text = "\n\n".join(parts)
409     return text.format(**ctx)
```

- **Lignes 403 à 407** : la formule finale, la signature et, 7 fois sur 10, un bas de page, tous communs aux deux classes.
- **Ligne 408** : assemble les morceaux en paragraphes, séparés par une ligne vide.
- **Ligne 409** : `format(**ctx)` remplace chaque `{...}` par sa valeur. Par exemple, `"Dear {first},"` devient `"Dear Sarah,"`.

## Lignes 412 à 417 : `synthetic()`

```python
412 def synthetic(n_legit: int, n_phish: int, split: str, seed: int):
413     rng = random.Random(seed)
414     rows = [(make_email(rng, 0, split), 0) for _ in range(n_legit)]
415     rows += [(make_email(rng, 1, split), 1) for _ in range(n_phish)]
416     rng.shuffle(rows)
417     return rows
```

- **Ligne 413** : un générateur aléatoire **avec graine** : la même graine produit exactement les mêmes emails.
- **Lignes 414 et 415** : génère `n_legit` emails légitimes (étiquette 0) puis `n_phish` emails de phishing (étiquette 1). `_` désigne une variable de boucle inutilisée.
- **Ligne 416** : mélange les deux classes.
- **Ligne 417** : renvoie une liste de couples (texte, étiquette).

---

## Le bruit

### Lignes 421 et 422 : les tables de substitution

```python
421 _HOMO = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "@", "s": "5"}
422 _KEYS = "abcdefghijklmnopqrstuvwxyz"
```

- **Ligne 421** : les lettres que les fraudeurs remplacent par des caractères ressemblants (*homoglyphes*) pour échapper aux filtres : `paypal` devient `p@yp@1`.
- **Ligne 422** : l'alphabet, pour simuler une frappe sur la mauvaise touche.

### Lignes 424 à 440 : les artefacts d'emails

```python
424 PREFIXES = ["fw :", "fwd :", "re :", "re : re :", "subject : fwd :"]
425 QUOTES = [
426     "- - - - - original message - - - - - from : {who} sent : monday , {date} to : {who2} subject : re : update",
427     "on {date} , {who} wrote :",
...
430 ]
431 SUFFIXES = [
432     "sent from my iphone", "sent from my android device", "get outlook for ios",
...
439     "view this email in your browser",
440 ]
```

Ce que les logiciels de messagerie ajoutent aux emails réels :
- **ligne 424** : les préfixes de réponse et de transfert ;
- **lignes 425 à 430** : les en-têtes de message cité ou transféré ;
- **lignes 431 à 440** : signatures mobiles, avertissements légaux, liens de désabonnement, restes de HTML (`< br >`, `& nbsp ;`).

Ces textes sont **déjà écrits au format normalisé** (minuscules, ponctuation entourée d'espaces), car le bruit s'applique à des textes déjà normalisés.

### Lignes 443 à 454 : `_typo()`

```python
443 def _typo(word, rng):
444     if len(word) < 4 or not word.isalpha():
445         return word
446     i = rng.randrange(len(word) - 1)
447     op = rng.randrange(4)
448     if op == 0:
449         return word[:i] + word[i + 1] + word[i] + word[i + 2:]     # inversion
450     if op == 1:
451         return word[:i] + word[i + 1:]                              # lettre manquante
452     if op == 2:
453         return word[:i] + word[i] + word[i:]                        # lettre doublée
454     return word[:i] + rng.choice(_KEYS) + word[i + 1:]              # mauvaise touche
```

- **Lignes 444 et 445** : ne touche ni aux mots courts ni aux mots qui contiennent autre chose que des lettres (nombres, ponctuation).
- **Ligne 446** : choisit une position dans le mot.
- **Ligne 447** : choisit l'un des quatre types de faute.
- **Ligne 449** : deux lettres voisines inversées : `password` devient `passwrod`.
- **Ligne 451** : une lettre supprimée : `pasword`.
- **Ligne 453** : une lettre doublée : `passsword`.
- **Ligne 454** : une lettre remplacée par une autre au hasard : `passwkrd`.

`word[:i]` est le début du mot jusqu'à la position i (exclue), `word[i + 1:]` la suite après i.

### Lignes 457 à 461 : `add_noise()`, tirer les perturbations

```python
457 def add_noise(text: str, rng: random.Random) -> str:
458     """Applique 1 à 3 perturbations à un texte DÉJÀ normalisé (minuscules, ponctuation isolée)."""
459     words = text.split()
460     ops = rng.sample(["typo", "homoglyph", "drop", "shuffle", "truncate", "prefix", "quote", "suffix"],
461                      k=rng.randint(1, 3))
```

- **Ligne 459** : découpe le texte en mots (sur les espaces).
- **Lignes 460 et 461** : tire de 1 à 3 perturbations **différentes** parmi les 8.

### Lignes 462 à 469 : fautes, lettres déguisées, mots supprimés

```python
462     for op in ops:
463         if op == "typo":
464             words = [_typo(w, rng) if rng.random() < 0.05 else w for w in words]
465         elif op == "homoglyph":
466             words = ["".join(_HOMO.get(c, c) if rng.random() < 0.3 else c for c in w)
467                      if w.isalpha() and len(w) > 3 and rng.random() < 0.04 else w for w in words]
468         elif op == "drop" and len(words) > 20:
469             words = [w for w in words if rng.random() > 0.05]
```

- **Ligne 464** : chaque mot a 5 % de chances de recevoir une faute de frappe.
- **Lignes 466 et 467** : chaque mot alphabétique de plus de 3 lettres a 4 % de chances d'être déguisé. Dans ce cas, chacune de ses lettres a 30 % de chances d'être remplacée si elle figure dans la table. `_HOMO.get(c, c)` renvoie le remplaçant, ou la lettre elle-même si elle n'en a pas.
- **Lignes 468 et 469** : pour les textes de plus de 20 mots, chaque mot a 5 % de chances de disparaître.

### Lignes 470 à 480 : mélanger les phrases

```python
470         elif op == "shuffle":
471             joined = " ".join(words)
472             # Après normalize(), « site.com » devient « site . com » : couper aux points
473             # casserait les liens en morceaux. On ne mélange que les emails sans lien.
474             if re.search(r"http|www|\. (com|net|org|co|uk|gov|io|us)\b", joined):
475                 continue
476             sents = re.split(r"(?<= [.!?]) ", joined)
477             if len(sents) > 2:
478                 head, rest = sents[0], sents[1:]
479                 rng.shuffle(rest)
480                 words = " ".join([head] + rest).split()
```

- **Ligne 471** : recolle les mots en un texte.
- **Lignes 474 et 475** : si le texte contient un lien, on ne mélange pas. Après la normalisation, le point de `chase . com` ressemble à une fin de phrase, et le mélange découpait les liens en morceaux. Ce bug a été repéré en relisant les données générées, puis corrigé par cette condition. `continue` passe à la perturbation suivante.
- **Ligne 476** : coupe le texte en phrases, après chaque « . », « ! » ou « ? » suivi d'un espace. `(?<= [.!?])` est un *lookbehind* : la coupure se fait juste après la ponctuation, sans la supprimer.
- **Lignes 477 à 480** : s'il y a au moins 3 phrases, on garde la première en place (souvent l'objet ou la salutation) et on mélange les autres.

### Lignes 481 à 493 : troncature, préfixe, citation, suffixe

```python
481         elif op == "truncate" and len(words) > 40:
482             words = words[: int(len(words) * rng.uniform(0.6, 0.9))]
483         elif op == "prefix":
484             words = rng.choice(PREFIXES).split() + words
485         elif op == "quote":
486             q = rng.choice(QUOTES).format(
487                 who=f"{rng.choice(FIRST)} {rng.choice(LAST)}".lower(),
488                 who2=f"{rng.choice(FIRST)} {rng.choice(LAST)}".lower(),
489                 date=f"{rng.choice(MONTHS)} {rng.randint(1, 28)} , {rng.randint(2001, 2026)}".lower())
490             words = (q.split() + words) if rng.random() < 0.5 else (words + q.split())
491         elif op == "suffix":
492             words = words + rng.choice(SUFFIXES).split()
493     return " ".join(words)
```

- **Lignes 481 et 482** : pour les textes de plus de 40 mots, on ne garde qu'entre 60 et 90 % du début, comme un email coupé.
- **Lignes 483 et 484** : ajoute un préfixe « fwd : » ou « re : » au début.
- **Lignes 485 à 490** : fabrique un en-tête de citation avec un nom et une date au hasard (en minuscules, comme le texte normalisé), puis le place au début ou à la fin, une fois sur deux.
- **Lignes 491 et 492** : ajoute une signature mobile, un avertissement légal ou un reste de HTML à la fin.
- **Ligne 493** : recolle les mots en texte.

**Pourquoi c'est sans risque pour l'apprentissage :** `prepare_data.py` choisit les emails à bruiter au hasard dans les deux classes, et toutes ces perturbations ont les mêmes probabilités pour le phishing et pour les emails légitimes. Elles n'apportent donc aucun indice sur la classe. Le modèle apprend simplement que « sent from my iphone » ou une faute de frappe ne change pas la nature d'un email, ce qui le rend plus robuste face à de vrais emails, plus désordonnés que ceux du dataset.
