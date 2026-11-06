"""Vidéo explicative du détecteur de phishing, réalisée avec Manim (Community Edition).

Rendu d'une scène (aperçu rapide) :  manim -ql phishing_explainer.py S06_Attention
Rendu complet + assemblage        :  python render_all.py          (voir ce fichier)

Les valeurs marquées « illustratif » à l'écran servent à expliquer le mécanisme ;
toutes les autres (ids de tokens, tailles, paramètres, scores TF-IDF) sont réelles.
"""
from manim import *
import numpy as np

# ---------------------------------------------------------------- Palette
BG = "#0F1B2A"
PANEL = "#1B2A3D"
ORANGE = "#F08A4B"   # phishing / signal d'alerte
BLUE_C = "#7FB3E6"   # légitime / vecteurs
GREEN_C = "#7BC8A4"  # valeurs V, mises à jour
TXT = "#E9EEF3"
SOFT = "#A9B8C8"
GREY_C = "#4A5A6E"
FONT = "Arial"
MONO = "Consolas"


# ---------------------------------------------------------------- Helpers
def T(s, size=28, color=TXT, weight=NORMAL, **kw):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight, **kw)


def M(s, size=26, color=TXT, **kw):
    return Text(s, font=MONO, font_size=size, color=color, **kw)


def box(label, w=2.4, h=0.8, stroke=SOFT, fill=PANEL, size=24, color=TXT):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=stroke,
                         stroke_width=2, fill_color=fill, fill_opacity=1)
    t = T(label, size, color)
    if t.width > w - 0.25:
        t.scale_to_fit_width(w - 0.25)
    t.move_to(r)
    return VGroup(r, t)


def pill(s, fill=GREY_C, color=TXT, size=24):
    t = M(s, size, color)
    r = RoundedRectangle(corner_radius=0.1, width=max(t.width + 0.3, 0.5), height=0.55,
                         fill_color=fill, fill_opacity=1, stroke_width=0)
    t.move_to(r)
    return VGroup(r, t)


def vec(n=8, seed=0, color=BLUE_C, size=0.26, horizontal=False):
    """Un vecteur dessiné comme une colonne de cases (intensité = valeur)."""
    rng = np.random.default_rng(seed)
    vals = rng.uniform(0.15, 1.0, n)
    sq = VGroup(*[Square(size, stroke_width=1, stroke_color=BG, fill_color=color, fill_opacity=v)
                  for v in vals])
    sq.arrange(RIGHT if horizontal else DOWN, buff=0)
    return sq


def grid(rows, cols, seed=0, color=BLUE_C, size=0.2):
    rng = np.random.default_rng(seed)
    g = VGroup()
    for i in range(rows):
        row = VGroup(*[Square(size, stroke_width=0.6, stroke_color=BG, fill_color=color,
                              fill_opacity=rng.uniform(0.15, 0.9)) for _ in range(cols)])
        row.arrange(RIGHT, buff=0)
        g.add(row)
    g.arrange(DOWN, buff=0)
    return g


def neuron_layer(n, color=BLUE_C, r=0.12, buff=0.12):
    return VGroup(*[Circle(r, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.35)
                    for _ in range(n)]).arrange(DOWN, buff=buff)


def connect(a, b, color=SOFT, opacity=0.25, width=1):
    return VGroup(*[Line(x.get_right(), y.get_left(), stroke_color=color, stroke_width=width,
                         stroke_opacity=opacity) for x in a for y in b])


def tag(s, color):
    return T(s, 20, color).set_opacity(0.9)


class Base(Scene):
    chapter = ""
    number = ""

    def setup(self):
        self.camera.background_color = BG
        self.cap = None
        self.head = None

    # Carte de chapitre puis bandeau en haut à gauche
    def chapter_card(self, subtitle=""):
        n = T(self.number, 96, ORANGE, weight=BOLD)
        t = T(self.chapter, 52, TXT, weight=BOLD)
        g = VGroup(n, t).arrange(DOWN, buff=0.3)
        if subtitle:
            g.add(T(subtitle, 28, SOFT).next_to(t, DOWN, buff=0.35))
        self.play(FadeIn(n, scale=0.8), Write(t), run_time=1.2)
        if subtitle:
            self.play(FadeIn(g[2]))
        self.wait(1.5)
        self.play(FadeOut(g))
        self.head = T(f"{self.number} · {self.chapter}", 22, SOFT).to_corner(UL, buff=0.35)
        self.play(FadeIn(self.head), run_time=0.4)

    # Sous-titre explicatif en bas de l'écran
    def say(self, s, hold=2.5, color=TXT):
        new = T(s, 28, color)
        if new.width > 13:
            new.scale_to_fit_width(13)
        new.to_edge(DOWN, buff=0.35)
        if self.cap is None:
            self.play(FadeIn(new, shift=UP * 0.15), run_time=0.5)
        else:
            self.play(FadeOut(self.cap, shift=UP * 0.15), FadeIn(new, shift=UP * 0.15), run_time=0.5)
        self.cap = new
        self.wait(hold)

    def clear(self, keep_head=True):
        mobs = [m for m in self.mobjects if not (keep_head and m is self.head)]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=0.7)
        self.cap = None

    def outro(self):
        self.clear(keep_head=False)
        self.wait(0.3)


# =====================================================================
# 01 · Le problème
# =====================================================================
class S01_Intro(Base):
    number, chapter = "01", "Le problème"

    def construct(self):
        t1 = T("Détecter le phishing", 64, TXT, weight=BOLD)
        t2 = T("avec un Transformer écrit from scratch", 40, ORANGE)
        t3 = T("NLP · PyTorch · 1 859 201 paramètres", 26, SOFT)
        g = VGroup(t1, t2, t3).arrange(DOWN, buff=0.35)
        self.play(Write(t1), run_time=1.4)
        self.play(FadeIn(t2, shift=UP * 0.2))
        self.play(FadeIn(t3))
        self.wait(2)
        self.play(FadeOut(g))
        self.chapter_card("Un email arrive : est-ce une attaque ?")

        lines = ["URGENT: your PayPal account has been suspended.",
                 "Verify your password within 24 hours at",
                 "http://192.168.4.2/login or it will be locked!"]
        colors = [{"URGENT": ORANGE, "suspended": ORANGE, "PayPal": BLUE_C},
                  {"Verify": BLUE_C, "password": BLUE_C, "24": ORANGE, "hours": ORANGE},
                  {"http://192.168.4.2/login": BLUE_C, "locked": ORANGE}]
        plain = VGroup(*[M(l, 26) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        colored = VGroup(*[M(l, 26, t2c=c) for l, c in zip(lines, colors)]).arrange(
            DOWN, aligned_edge=LEFT, buff=0.22)
        card = SurroundingRectangle(plain, buff=0.45, corner_radius=0.15, color=SOFT,
                                    fill_color=PANEL, fill_opacity=1)
        hdr = T("Nouveau message", 22, SOFT).next_to(card, UP, aligned_edge=LEFT, buff=0.15)
        mail = VGroup(card, hdr, plain).move_to(UP * 1.0)
        colored.move_to(plain)
        self.play(FadeIn(card), FadeIn(hdr))
        self.play(AddTextLetterByLetter(plain[0]), run_time=1.2)
        self.play(AddTextLetterByLetter(plain[1]), run_time=1.2)
        self.play(AddTextLetterByLetter(plain[2]), run_time=1.2)
        self.say("Le phishing ne pirate pas une machine : il manipule une personne.")
        self.play(Transform(plain, colored))

        tags = VGroup(
            box("Urgence et menace", 3.6, 0.7, ORANGE, size=24),
            box("Vol d'identifiants", 3.6, 0.7, BLUE_C, size=24),
            box("Lien suspect (adresse IP)", 3.6, 0.7, BLUE_C, size=24),
        ).arrange(RIGHT, buff=0.4).next_to(card, DOWN, buff=0.5)
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in tags], lag_ratio=0.3))
        self.say("Trois signaux typiques : urgence, demande d'identifiants, lien piégé.", 3)
        self.play(FadeOut(tags))

        model = box("Modèle", 2.4, 1.0, ORANGE, size=30)
        out = T("P(phishing) = ?", 34, ORANGE, weight=BOLD)
        self.play(mail.animate.scale(0.55).to_edge(LEFT, buff=0.5).shift(UP * 0.3))
        model.next_to(mail, RIGHT, buff=1.0)
        out.next_to(model, RIGHT, buff=1.0)
        a1 = Arrow(mail.get_right(), model.get_left(), buff=0.15, color=SOFT)
        a2 = Arrow(model.get_right(), out.get_left(), buff=0.15, color=SOFT)
        self.play(GrowArrow(a1), FadeIn(model))
        self.play(GrowArrow(a2), Write(out))
        self.say("Objectif : une probabilité entre 0 et 1 pour chaque email, avant le clic.", 3)

        self.clear()
        plan = VGroup(*[T(s, 26, TXT) for s in [
            "02  Le pipeline NLP", "03  Normalisation", "04  Tokenisation BPE",
            "05  Embeddings", "06  Self-attention", "07  Bloc encodeur",
            "08  Tête de classification", "09  Entraînement", "10  Évaluation",
            "11  Le livrable"]]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        title = T("Plan", 40, ORANGE, weight=BOLD).next_to(plan, UP, buff=0.4, aligned_edge=LEFT)
        VGroup(title, plan).move_to(ORIGIN)
        self.play(FadeIn(title), LaggedStart(*[FadeIn(p, shift=RIGHT * 0.2) for p in plan],
                                             lag_ratio=0.12))
        self.wait(3)
        self.outro()


# =====================================================================
# 02 · Le pipeline NLP
# =====================================================================
class S02_Pipeline(Base):
    number, chapter = "02", "Le pipeline NLP"

    def construct(self):
        self.chapter_card("Un réseau ne lit pas : il calcule sur des nombres")
        steps = ["1  Texte brut", "2  Normalisation", "3  Tokens", "4  Identifiants",
                 "5  Vecteurs", "6  Transformer", "7  Décision"]
        col = VGroup(*[box(s, 3.3, 0.62, GREY_C, size=24) for s in steps]).arrange(DOWN, buff=0.16)
        col.to_edge(LEFT, buff=0.6).shift(DOWN * 0.15)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in col], lag_ratio=0.1))

        stage = RIGHT * 2.6 + UP * 0.2
        toks = ["urgent", ":", "your", "account"]
        ids = ["4001", "43", "295", "1126"]

        reps = [
            M("URGENT: your account", 34),
            M("urgent : your account", 34),
            VGroup(*[pill(t, GREY_C, size=30) for t in toks]).arrange(RIGHT, buff=0.2),
            VGroup(*[pill(i, "#2B4A6B", size=30) for i in ids]).arrange(RIGHT, buff=0.2),
            VGroup(*[vec(8, s) for s in range(4)]).arrange(RIGHT, buff=0.35),
            VGroup(*[vec(8, s + 10, ORANGE) for s in range(4)]).arrange(RIGHT, buff=0.35),
            T("P(phishing) entre 0 et 1", 40, ORANGE, weight=BOLD),
        ]
        notes = [
            "L'email tel qu'il arrive dans la boîte de réception.",
            "Un format unique : minuscules, ponctuation séparée.",
            "Le tokenizer BPE découpe le texte en morceaux connus.",
            "Chaque token devient son numéro dans un vocabulaire de 8 000 (ids réels).",
            "Chaque numéro devient un vecteur de 128 nombres appris.",
            "L'attention mélange les vecteurs : chacun tient compte du contexte.",
            "Un résumé de l'email passe dans une tête qui sort une probabilité.",
        ]
        for r in reps:
            r.move_to(stage)
        cur = None
        for i, (r, n) in enumerate(zip(reps, notes)):
            hl = col[i][0].copy().set_fill(opacity=0).set_stroke(ORANGE, 4)
            anims = [Create(hl)]
            if cur is None:
                anims.append(FadeIn(r))
            else:
                anims.append(ReplacementTransform(cur, r))
            self.play(*anims, run_time=0.9)
            if i == 3:
                lab = VGroup(*[T(t, 20, SOFT).next_to(p, DOWN, buff=0.15) for t, p in zip(toks, r)])
                self.play(FadeIn(lab))
                self.say(n, 2.8)
                self.play(FadeOut(lab))
            elif i == 4:
                dim = T("4 tokens × 128 dimensions (8 cases dessinées par vecteur)", 22, SOFT)
                dim.next_to(r, DOWN, buff=0.3)
                self.play(FadeIn(dim))
                self.say(n, 2.8)
                self.play(FadeOut(dim))
            else:
                self.say(n, 2.4)
            self.play(FadeOut(hl), run_time=0.3)
            cur = r
        brace = Brace(col[:4], RIGHT, color=SOFT)
        b1 = T("règles fixes", 22, SOFT).next_to(brace, RIGHT, buff=0.15)
        brace2 = Brace(col[4:], RIGHT, color=ORANGE)
        b2 = T("appris par\nentraînement", 22, ORANGE).next_to(brace2, RIGHT, buff=0.15)
        self.play(FadeOut(cur), GrowFromCenter(brace), FadeIn(b1), GrowFromCenter(brace2), FadeIn(b2))
        self.say("Les étapes 1 à 4 sont des règles ; à partir de 5, tout est appris.", 3.5)
        self.outro()


# =====================================================================
# 03 · Normalisation
# =====================================================================
class S03_Normalisation(Base):
    number, chapter = "03", "Normalisation"

    def construct(self):
        self.chapter_card("Apprendre le contenu, pas le format")
        steps = [
            ("Email brut", "URGENT: Verify your password at http://192.168.4.2/login!"),
            ("1 · minuscules", "urgent: verify your password at http://192.168.4.2/login!"),
            ("2 · ponctuation isolée", "urgent : verify your password at http : / / 192 . 168 . 4 . 2 / login !"),
        ]
        label = T(steps[0][0], 26, ORANGE).move_to(UP * 1.6)
        txt = M(steps[0][1], 26).move_to(UP * 0.6)
        if txt.width > 13:
            txt.scale_to_fit_width(13)
        self.play(FadeIn(label), Write(txt), run_time=1.5)
        self.wait(1)
        for name, s in steps[1:]:
            new_l = T(name, 26, ORANGE).move_to(label)
            new_t = M(s, 26).move_to(txt)
            if new_t.width > 13:
                new_t.scale_to_fit_width(13)
            self.play(Transform(label, new_l), Transform(txt, new_t), run_time=1.2)
            self.wait(1.2)
        extra = VGroup(
            T("3 · espaces multiples fusionnés", 26, TXT),
            T("4 · coupe à 20 000 caractères (un email du dataset en fait 17 millions)", 26, TXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(txt, DOWN, buff=0.6)
        self.play(LaggedStart(*[FadeIn(e, shift=UP * 0.2) for e in extra], lag_ratio=0.4))
        self.say("Une seule fonction normalize(), utilisée à l'entraînement ET en production.", 3)
        self.clear()

        # Pourquoi : la fuite de format
        title = T("Pourquoi ? Le dataset mélange deux formats", 34, TXT, weight=BOLD).to_edge(UP, buff=1.0)
        self.play(FadeIn(title))
        axis = Line(LEFT * 3, RIGHT * 3, color=SOFT).shift(DOWN * 1.6)
        h_safe, h_ph = 34.9 / 14, 21.2 / 14
        b1 = Rectangle(width=1.6, height=h_safe, fill_color=BLUE_C, fill_opacity=0.85, stroke_width=0)
        b2 = Rectangle(width=1.6, height=h_ph, fill_color=ORANGE, fill_opacity=0.85, stroke_width=0)
        b1.next_to(axis, UP, buff=0).shift(LEFT * 1.4)
        b2.next_to(axis, UP, buff=0).shift(RIGHT * 1.4)
        v1 = T("34,9 %", 30, TXT, weight=BOLD).next_to(b1, UP, buff=0.15)
        v2 = T("21,2 %", 30, TXT, weight=BOLD).next_to(b2, UP, buff=0.15)
        l1 = T("emails sûrs", 24, BLUE_C).next_to(b1, DOWN, buff=0.25)
        l2 = T("phishing", 24, ORANGE).next_to(b2, DOWN, buff=0.25)
        q = T("Part des emails contenant des majuscules (mesuré sur le dataset brut)", 22, SOFT)
        q.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(q), Create(axis))
        self.play(GrowFromEdge(b1, DOWN), GrowFromEdge(b2, DOWN), FadeIn(l1), FadeIn(l2))
        self.play(FadeIn(v1), FadeIn(v2))
        self.say("Sans normalisation, le modèle pourrait apprendre « majuscules = email sûr ».", 3)
        self.say("Un raccourci qui marche sur ce dataset, et échoue sur de vrais emails.", 3, ORANGE)
        self.outro()


# =====================================================================
# 04 · Tokenisation BPE
# =====================================================================
class S04_BPE(Base):
    number, chapter = "04", "Tokenisation BPE"

    def construct(self):
        self.chapter_card("Byte Pair Encoding : fusionner les paires fréquentes")
        seqs = [
            ["s", "u", "s", "p", "e", "n", "d", "e", "d"],
            ["s", "u", "s", "p", "e", "n", "d", "ed"],
            ["su", "s", "p", "e", "n", "d", "ed"],
            ["sus", "p", "e", "n", "d", "ed"],
            ["susp", "e", "n", "d", "ed"],
            ["susp", "en", "d", "ed"],
            ["susp", "end", "ed"],
            ["susp", "ended"],
        ]

        def row(seq):
            return VGroup(*[pill(s, GREY_C, size=32) for s in seq]).arrange(RIGHT, buff=0.15).move_to(UP * 0.8)

        cur = row(seqs[0])
        lab = T("On part des caractères…", 26, SOFT).next_to(cur, DOWN, buff=0.6)
        self.play(LaggedStart(*[FadeIn(p, shift=DOWN * 0.2) for p in cur], lag_ratio=0.08))
        self.play(FadeIn(lab))
        self.wait(1)
        new_lab = T("…et on fusionne la paire la plus fréquente, encore et encore", 26, SOFT).move_to(lab)
        self.play(Transform(lab, new_lab))
        for s in seqs[1:]:
            nxt = row(s)
            self.play(ReplacementTransform(cur, nxt), run_time=0.7)
            cur = nxt
            self.wait(0.35)
        self.play(*[p[0].animate.set_fill(ORANGE) for p in cur])
        note = T("ordre des fusions illustratif ; le découpage final « susp | ended » est réel", 20, SOFT)
        note.next_to(lab, DOWN, buff=0.25)
        self.play(FadeIn(note))
        self.say("On s'arrête à 8 000 tokens. Le tokenizer est entraîné sur le train seulement.", 3)
        self.clear()

        # Découpage réel de l'email de test
        title = T("Découpage réel de l'email de test", 32, TXT, weight=BOLD).to_edge(UP, buff=1.0)
        toks = ["urgent", ":", "your", "paypal", "account", "has", "been", "susp", "ended", ".",
                "verify", "your", "password", "within", "24", "hours", "at", "http", ":", "/", "/",
                "192", ".", "16", "8", ".", "4", ".", "2", "/", "login", "or", "it", "will", "be",
                "lock", "ed", "!"]
        split = {"susp", "ended", "lock", "ed", "16", "8"}
        ps = VGroup(*[pill(t, ORANGE if t in split else GREY_C, size=22) for t in toks])
        ps.arrange_in_grid(cols=10, buff=(0.12, 0.18), flow_order="rd")
        ps.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(title))
        self.play(LaggedStart(*[FadeIn(p, scale=0.8) for p in ps], lag_ratio=0.03), run_time=2.5)
        info = T("38 tokens · 0 token inconnu [UNK] · en orange : mots découpés", 24, SOFT)
        info.next_to(ps, DOWN, buff=0.45)
        self.play(FadeIn(info))
        self.say("Les mots fréquents restent entiers, les mots rares réutilisent des morceaux.", 3)
        self.clear()

        # Padding et masque
        title = T("Longueur fixe : 256 positions", 32, TXT, weight=BOLD).to_edge(UP, buff=1.0)
        slots = ["urgent", ":", "your", "…", "lock", "ed", "!", "[PAD]", "[PAD]", "…", "[PAD]"]
        mask = ["1", "1", "1", "…", "1", "1", "1", "0", "0", "…", "0"]
        top = VGroup(*[pill(s, GREY_C if "PAD" not in s else "#2A3442",
                            SOFT if "PAD" in s else TXT, size=22) for s in slots]).arrange(RIGHT, buff=0.1)
        bot = VGroup(*[M(m, 28, ORANGE if m == "1" else SOFT) for m in mask])
        for b, t in zip(bot, top):
            b.next_to(t, DOWN, buff=0.4)
        lt = T("tokens", 22, SOFT).next_to(top, LEFT, buff=0.3)
        lb = T("masque", 22, SOFT).next_to(bot, LEFT, buff=0.3)
        g = VGroup(top, bot, lt, lb).move_to(ORIGIN)
        brace = Brace(top[:7], UP, color=SOFT)
        bt = T("38 vrais tokens", 20, SOFT).next_to(brace, UP, buff=0.1)
        brace2 = Brace(top[7:], UP, color=SOFT)
        bt2 = T("218 × [PAD] (id 0)", 20, SOFT).next_to(brace2, UP, buff=0.1)
        self.play(FadeIn(title), FadeIn(top), FadeIn(lt))
        self.play(GrowFromCenter(brace), FadeIn(bt), GrowFromCenter(brace2), FadeIn(bt2))
        self.play(FadeIn(bot), FadeIn(lb))
        self.say("Les emails courts sont complétés par [PAD], les longs sont coupés à 256.", 3)
        self.say("Le masque (1 = vrai token, 0 = padding) dira au modèle d'ignorer le remplissage.", 3)
        self.outro()


# =====================================================================
# 05 · Embeddings
# =====================================================================
class S05_Embeddings(Base):
    number, chapter = "05", "Embeddings"

    def construct(self):
        self.chapter_card("Chaque token devient un vecteur de 128 nombres")
        tok = pill("password", GREY_C, size=30).move_to(LEFT * 5.2 + UP * 1.5)
        tid = pill("id 3905", "#2B4A6B", size=30).next_to(tok, DOWN, buff=0.5)
        self.play(FadeIn(tok))
        self.play(TransformFromCopy(tok, tid))

        table = grid(10, 12, seed=1, size=0.24).move_to(LEFT * 0.6 + UP * 0.3)
        tl = T("Table des tokens : 8 000 × 128", 24, SOFT).next_to(table, UP, buff=0.25)
        dots = VGroup(*[Dot(radius=0.035, color=SOFT) for _ in range(3)]).arrange(DOWN, buff=0.08)
        dots.next_to(table, DOWN, buff=0.12)
        self.play(FadeIn(table), FadeIn(tl), FadeIn(dots))
        row_i = 6
        hl = SurroundingRectangle(table[row_i], color=ORANGE, buff=0.02)
        arr = Arrow(tid.get_right(), table[row_i].get_left(), buff=0.15, color=ORANGE)
        self.play(GrowArrow(arr), Create(hl))
        self.say("L'id sert simplement de numéro de ligne dans une grande table.", 2.5)

        v_tok = table[row_i].copy()
        self.play(v_tok.animate.arrange(DOWN, buff=0).move_to(RIGHT * 3.4 + UP * 0.3))
        vt_l = T("token", 22, BLUE_C).next_to(v_tok, UP, buff=0.2)
        self.play(FadeIn(vt_l))

        plus = T("+", 48, ORANGE).next_to(v_tok, RIGHT, buff=0.35)
        v_pos = vec(12, 42, ORANGE, size=0.24).next_to(plus, RIGHT, buff=0.35)
        vp_l = T("position 12", 22, ORANGE).next_to(v_pos, UP, buff=0.2)
        self.play(FadeIn(plus), FadeIn(v_pos, shift=LEFT * 0.3), FadeIn(vp_l))
        self.say("On ajoute un vecteur de position (table 256 × 128) : sinon l'ordre des mots serait perdu.", 3.2)

        eq = T("=", 48, TXT).next_to(v_pos, RIGHT, buff=0.35)
        v_sum = vec(12, 7, GREEN_C, size=0.24).next_to(eq, RIGHT, buff=0.35)
        self.play(FadeIn(eq), TransformFromCopy(VGroup(v_tok, v_pos), v_sum))
        self.say("Au départ ces nombres sont aléatoires ; l'entraînement les ajuste.", 2.8)
        self.clear()

        # Du token à la phrase, puis au lot
        mat = grid(12, 16, seed=3, color=GREEN_C, size=0.22).move_to(LEFT * 3)
        ml = T("1 email = 256 × 128", 24, SOFT).next_to(mat, UP, buff=0.25)
        self.play(FadeIn(mat), FadeIn(ml))
        stack = VGroup(*[Rectangle(width=3.5, height=2.6, stroke_color=GREEN_C, stroke_width=2,
                                   fill_color=PANEL, fill_opacity=1).shift(UR * 0.12 * i)
                         for i in range(6)]).move_to(RIGHT * 3)
        sl = M("(32, 256, 128)", 30, ORANGE).next_to(stack, DOWN, buff=0.35)
        st = T("un lot de 32 emails", 24, SOFT).next_to(stack, UP, buff=0.3)
        self.play(LaggedStart(*[FadeIn(s) for s in stack], lag_ratio=0.1), FadeIn(st))
        self.play(Write(sl))
        self.say("Le Transformer reçoit un tenseur : 32 emails × 256 tokens × 128 nombres.", 3)
        self.clear()

        # Intuition géométrique
        title = T("Intuition : des sens proches, des vecteurs proches", 32, TXT, weight=BOLD)
        title.to_edge(UP, buff=1.0)
        plane = NumberPlane(x_range=[-6, 6, 1], y_range=[-3, 3, 1], x_length=10, y_length=4.6,
                            background_line_style={"stroke_color": GREY_C, "stroke_opacity": 0.35}
                            ).shift(DOWN * 0.4)
        groups = {
            ORANGE: [("password", (-3.5, 1.4)), ("login", (-2.6, 2.0)), ("verify", (-4.2, 0.6)),
                     ("account", (-2.4, 0.9))],
            BLUE_C: [("meeting", (2.8, -1.2)), ("agenda", (3.8, -0.6)), ("thursday", (2.2, -2.0)),
                     ("report", (4.0, -1.8))],
            GREEN_C: [("free", (0.6, 2.2)), ("winner", (1.4, 1.6)), ("prize", (0.4, 1.3))],
        }
        pts = VGroup()
        for c, items in groups.items():
            for w, (x, y) in items:
                d = Dot(plane.c2p(x, y), color=c, radius=0.08)
                l = T(w, 20, c).next_to(d, UR, buff=0.05)
                pts.add(VGroup(d, l))
        self.play(FadeIn(title), Create(plane), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(p, scale=0.6) for p in pts], lag_ratio=0.08))
        note = T("projection 2D illustrative d'un espace à 128 dimensions", 20, SOFT).next_to(plane, DOWN, buff=0.15)
        self.play(FadeIn(note))
        self.say("Les mots utiles à la même décision finissent regroupés dans l'espace.", 3)
        self.outro()


# =====================================================================
# 06 · Self-attention
# =====================================================================
class S06_Attention(Base):
    number, chapter = "06", "Self-attention"

    def construct(self):
        self.chapter_card("Chaque token consulte tous les autres")
        words = ["verify", "your", "password", "at", "http", "login"]
        toks = VGroup(*[pill(w, GREY_C, size=28) for w in words]).arrange(RIGHT, buff=0.45).move_to(UP * 1.7)
        self.play(LaggedStart(*[FadeIn(t, shift=DOWN * 0.2) for t in toks], lag_ratio=0.1))
        focus = toks[2]
        self.play(focus[0].animate.set_fill(ORANGE))
        self.say("Prenons « password » : quels autres mots doit-il regarder ?", 2.5)

        # Q, K, V
        q = vec(6, 1, ORANGE, size=0.2).next_to(focus, DOWN, buff=0.4).shift(LEFT * 0.2)
        ql = T("Q", 24, ORANGE, weight=BOLD).next_to(q, LEFT, buff=0.1)
        ks = VGroup(*[vec(6, 10 + i, BLUE_C, size=0.2).next_to(t, DOWN, buff=0.4) for i, t in enumerate(toks)])
        ks[2].shift(RIGHT * 0.2)
        kl = T("K", 24, BLUE_C, weight=BOLD).next_to(ks[0], LEFT, buff=0.1)
        expl = VGroup(
            T("Q · requête : ce que je cherche", 26, ORANGE),
            T("K · clé : ce que chaque token contient", 26, BLUE_C),
            T("V · valeur : ce que chaque token transmet", 26, GREEN_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(DOWN * 1.5)
        self.play(FadeIn(q), FadeIn(ql), FadeIn(expl[0]))
        self.play(LaggedStart(*[FadeIn(k) for k in ks], lag_ratio=0.08), FadeIn(kl), FadeIn(expl[1]))
        self.play(FadeIn(expl[2]))
        self.say("Trois projections linéaires apprises transforment chaque vecteur en Q, K et V.", 3)
        self.play(FadeOut(expl))

        # Scores puis softmax (valeurs illustratives)
        scores = [2.1, 0.2, 1.0, -0.6, 1.5, 1.8]
        e = np.exp(np.array(scores))
        w = e / e.sum()
        ill = T("valeurs illustratives", 20, SOFT).to_corner(UR, buff=0.4)
        sc = VGroup(*[M(f"{s:+.1f}".replace(".", ","), 24, TXT).next_to(k, DOWN, buff=0.2)
                      for s, k in zip(scores, ks)])
        self.play(LaggedStart(*[FadeIn(s) for s in sc], lag_ratio=0.08), FadeIn(ill))
        self.say("Score = Q·K / √32 : l'intérêt de « password » pour chaque mot.", 2.8)
        bars = VGroup()
        for wi, s in zip(w, sc):
            b = Rectangle(width=0.5, height=max(wi * 3.5, 0.04), fill_color=ORANGE, fill_opacity=0.9,
                          stroke_width=0)
            b.move_to(s.get_bottom() + DOWN * 0.2, aligned_edge=UP)
            bars.add(b)
        pct = VGroup(*[T(f"{wi * 100:.0f} %", 20, ORANGE).next_to(b, DOWN, buff=0.08) for wi, b in zip(w, bars)])
        self.play(LaggedStart(*[GrowFromEdge(b, UP) for b in bars], lag_ratio=0.08), FadeIn(pct))
        self.say("Le softmax transforme les scores en poids qui somment à 100 %.", 2.8)

        # Arcs d'attention : épaisseur et opacité = poids
        arcs = VGroup()
        start = focus.get_top() + UP * 0.05
        for i, (t, wi) in enumerate(zip(toks, w)):
            if i == 2:
                continue
            end = t.get_top() + UP * 0.05
            ang = PI / 2 if end[0] < start[0] else -PI / 2   # l'arc passe toujours au-dessus
            a = ArcBetweenPoints(start, end, angle=ang, stroke_color=ORANGE,
                                 stroke_width=2 + 10 * wi / w.max())
            a.set_stroke(opacity=0.3 + 0.7 * wi / w.max())
            arcs.add(a)
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.12))
        self.say("« password » va surtout chercher l'information de « verify », « login » et « http ».", 3)

        # Somme pondérée des V
        vs = VGroup(*[vec(6, 30 + i, GREEN_C, size=0.2).move_to(k) for i, k in enumerate(ks)])
        vl = T("V", 24, GREEN_C, weight=BOLD).move_to(kl)
        self.play(ReplacementTransform(ks, vs), ReplacementTransform(kl, vl), FadeOut(VGroup(q, ql)))
        out = vec(6, 99, GREEN_C, size=0.3)
        ol = T("nouveau vecteur de « password »\n= Σ poids × V", 24, GREEN_C)
        res = VGroup(out, ol).arrange(RIGHT, buff=0.4).move_to(DOWN * 2.0)
        self.play(FadeOut(VGroup(sc, bars, pct)))
        self.play(TransformFromCopy(vs, out), run_time=1.2)
        self.play(FadeIn(ol))
        self.say("Résultat : un vecteur enrichi par le contexte de tout l'email.", 2.8)
        self.clear()

        # Formule
        f = M("Attention(Q, K, V) = softmax(Q·Kᵀ / √32) · V", 34, ORANGE)
        fb = SurroundingRectangle(f, buff=0.35, color=ORANGE, corner_radius=0.12)
        VGroup(f, fb).move_to(UP * 1.5)
        self.play(Write(f), Create(fb), run_time=1.8)
        parts = VGroup(
            T("Q·Kᵀ : tous les scores d'un coup (matrice 256 × 256)", 24, TXT),
            T("/ √32 : garde des valeurs stables avant le softmax", 24, TXT),
            T("· V : moyenne pondérée des valeurs", 24, TXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(fb, DOWN, buff=0.6)
        self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.2) for p in parts], lag_ratio=0.4))
        self.wait(3)
        self.clear()

        # Multi-têtes
        title = T("4 têtes en parallèle, 32 dimensions chacune", 32, TXT, weight=BOLD).to_edge(UP, buff=1.0)
        self.play(FadeIn(title))
        heads = VGroup()
        colors = [ORANGE, BLUE_C, GREEN_C, "#C9A0E8"]
        rng = np.random.default_rng(5)
        for c in colors:
            att = VGroup(*[Square(0.28, stroke_width=0.5, stroke_color=BG, fill_color=c,
                                  fill_opacity=rng.uniform(0.05, 1)) for _ in range(36)]).arrange_in_grid(6, 6, buff=0)
            heads.add(att)
        heads.arrange(RIGHT, buff=0.6).shift(UP * 0.2)
        hl = VGroup(*[T(f"tête {i + 1}", 22, c).next_to(h, UP, buff=0.15) for i, (h, c) in enumerate(zip(heads, colors))])
        self.play(LaggedStart(*[FadeIn(h, scale=0.8) for h in heads], lag_ratio=0.15), FadeIn(hl))
        cat = M("concat(4 × 32) = 128  →  projection", 28, TXT).next_to(heads, DOWN, buff=0.6)
        self.play(FadeIn(cat))
        self.say("Chaque tête apprend un type de relation différent ; on recolle les 4 résultats.", 3)
        self.clear()

        # Masques : bidirectionnel vs causal
        title = T("Quel token peut regarder quoi ?", 32, TXT, weight=BOLD).to_edge(UP, buff=1.0)
        self.play(FadeIn(title))
        n = 6

        def mask_grid(fn):
            return VGroup(*[Square(0.4, stroke_width=1, stroke_color=BG,
                                   fill_color=ORANGE if fn(i, j) else "#2A3442", fill_opacity=1)
                            for i in range(n) for j in range(n)]).arrange_in_grid(n, n, buff=0)
        g1 = mask_grid(lambda i, j: j < 4)          # 4 vrais tokens, 2 [PAD]
        g2 = mask_grid(lambda i, j: j <= i)         # causal (GPT)
        g1.move_to(LEFT * 3 + DOWN * 0.2)
        g2.move_to(RIGHT * 3 + DOWN * 0.2)
        l1 = T("Notre modèle : bidirectionnel", 24, ORANGE).next_to(g1, UP, buff=0.3)
        l2 = T("GPT : causal (pas de futur)", 24, SOFT).next_to(g2, UP, buff=0.3)
        p1 = T("colonnes grises = [PAD], score −∞ → poids 0", 20, SOFT).next_to(g1, DOWN, buff=0.3)
        p2 = T("pour générer du texte mot à mot", 20, SOFT).next_to(g2, DOWN, buff=0.3)
        self.play(FadeIn(g1), FadeIn(l1), FadeIn(p1))
        self.play(FadeIn(g2), FadeIn(l2), FadeIn(p2))
        self.say("On classe un email entier : chaque token voit tout l'email, sauf le padding.", 3.2)
        self.outro()


# =====================================================================
# 07 · Bloc encodeur
# =====================================================================
class S07_Encoder(Base):
    number, chapter = "07", "Bloc encodeur"

    def construct(self):
        self.chapter_card("Attention + feed-forward + connexions résiduelles")
        names = ["entrée x  (256 × 128)", "LayerNorm", "Attention 4 têtes", "⊕", "LayerNorm",
                 "Feed-forward 128 → 512 → 128", "⊕", "sortie  (256 × 128)"]
        items = VGroup()
        for n in names:
            if n == "⊕":
                c = Circle(0.25, color=ORANGE, stroke_width=3)
                items.add(VGroup(c, T("+", 30, ORANGE).move_to(c)))
            else:
                stroke = ORANGE if "Attention" in n or "Feed" in n else GREY_C
                items.add(box(n, 4.6, 0.55, stroke, size=22))
        items.arrange(DOWN, buff=0.22).move_to(LEFT * 2.6 + DOWN * 0.15)
        arrows = VGroup(*[Arrow(items[i].get_bottom(), items[i + 1].get_top(), buff=0.03,
                                color=SOFT, stroke_width=2, max_tip_length_to_length_ratio=0.35)
                          for i in range(len(items) - 1)])
        # connexions résiduelles (contournent le sous-bloc)
        r1 = VGroup(
            Line(items[0].get_right() + RIGHT * 0.05, items[0].get_right() + RIGHT * 0.6, color=GREEN_C),
            Line(items[0].get_right() + RIGHT * 0.6, [items[0].get_right()[0] + 0.6, items[3].get_center()[1], 0], color=GREEN_C),
            Arrow([items[0].get_right()[0] + 0.6, items[3].get_center()[1], 0], items[3].get_right(), buff=0.02, color=GREEN_C, stroke_width=3),
        )
        r2 = VGroup(
            Line(items[3].get_right() + RIGHT * 0.05, items[3].get_right() + RIGHT * 2.15, color=GREEN_C),
            Line(items[3].get_right() + RIGHT * 2.15, [items[3].get_right()[0] + 2.15, items[6].get_center()[1], 0], color=GREEN_C),
            Arrow([items[3].get_right()[0] + 2.15, items[6].get_center()[1], 0], items[6].get_right(), buff=0.02, color=GREEN_C, stroke_width=3),
        )
        self.play(LaggedStart(*[FadeIn(i) for i in items], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.08))
        self.say("Le vecteur traverse : normalisation, attention, puis un petit réseau par token.", 3)
        self.play(Create(r1), Create(r2), run_time=1.5)
        rl = T("résiduel :\non ajoute\nl'entrée\nà la sortie", 22, GREEN_C).next_to(r2, RIGHT, buff=0.3)
        self.play(FadeIn(rl))
        self.say("Les connexions résiduelles gardent l'information et facilitent l'apprentissage.", 3)

        # Flux d'un point de donnée
        path = VMobject().set_points_as_corners([i.get_center() for i in items])
        dot = Dot(items[0].get_center(), color=ORANGE, radius=0.1)
        self.play(FadeIn(dot))
        self.play(MoveAlongPath(dot, path), run_time=3, rate_func=linear)
        self.play(FadeOut(dot))
        self.say("Variante « pre-norm » (comme GPT-2) : LayerNorm avant chaque sous-bloc, plus stable.", 3)

        # Zoom feed-forward
        self.play(FadeOut(rl))
        l_in = neuron_layer(6, BLUE_C)
        l_hid = neuron_layer(12, ORANGE, r=0.1, buff=0.08)
        l_out = neuron_layer(6, BLUE_C)
        ff = VGroup(l_in, l_hid, l_out).arrange(RIGHT, buff=1.0).move_to(RIGHT * 4.3 + DOWN * 0.2)
        c1 = connect(l_in, l_hid)
        c2 = connect(l_hid, l_out)
        labs = VGroup(T("128", 22, BLUE_C).next_to(l_in, DOWN), T("512\nGELU", 22, ORANGE).next_to(l_hid, DOWN),
                      T("128", 22, BLUE_C).next_to(l_out, DOWN))
        self.play(FadeOut(VGroup(r1, r2)))
        self.play(Indicate(items[5], color=ORANGE))
        self.play(FadeIn(ff), Create(c1), Create(c2), FadeIn(labs), run_time=1.5)
        self.say("Le feed-forward élargit à 512 neurones puis revient à 128, pour chaque token.", 3)
        self.clear()

        # Empilement x4
        blocks = VGroup(*[box(f"Bloc encodeur {i + 1}", 4.2, 0.75, ORANGE, size=24) for i in range(4)]
                        ).arrange(UP, buff=0.3).move_to(LEFT * 2)
        ar = VGroup(*[Arrow(blocks[i].get_top(), blocks[i + 1].get_bottom(), buff=0.03, color=SOFT)
                      for i in range(3)])
        side = VGroup(
            T("Attention : 66 048 paramètres", 24, TXT),
            T("Feed-forward : 131 712", 24, TXT),
            T("LayerNorm × 2 : 512", 24, TXT),
            T("= 198 272 par bloc", 26, ORANGE, weight=BOLD),
            T("× 4 = 793 088", 26, ORANGE, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(blocks, RIGHT, buff=1.0)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in blocks], lag_ratio=0.2),
                  LaggedStart(*[GrowArrow(a) for a in ar], lag_ratio=0.2))
        self.play(LaggedStart(*[FadeIn(s) for s in side], lag_ratio=0.3))
        self.say("4 blocs identiques empilés : chacun affine la représentation du précédent.", 3)
        self.outro()


# =====================================================================
# 08 · Tête de classification
# =====================================================================
class S08_Head(Base):
    number, chapter = "08", "Tête de classification"

    def construct(self):
        self.chapter_card("Du tenseur à une seule probabilité")
        mat = grid(10, 8, seed=11, color=GREEN_C, size=0.2).move_to(LEFT * 5.3 + UP * 1.4)
        for r in mat[7:]:
            r.set_fill(GREY_C, opacity=0.5)
        ml = T("256 × 128", 20, SOFT).next_to(mat, UP, buff=0.15)
        pad = T("[PAD]", 18, SOFT).next_to(mat[8], LEFT, buff=0.1)
        self.play(FadeIn(mat), FadeIn(ml), FadeIn(pad))
        pooled = vec(8, 12, GREEN_C, size=0.2).move_to(LEFT * 3.9 + UP * 1.4)
        pl = T("moyenne\n128", 20, GREEN_C).next_to(pooled, UP, buff=0.15)
        self.play(TransformFromCopy(mat[:7], pooled), FadeIn(pl), run_time=1.3)
        self.say("Mean pooling : moyenne des vecteurs des vrais tokens, le padding est ignoré.", 3)

        # Features
        names = ["liens", "IP", "raccourci", "emails", "urgence", "identif.", "argent", "clic",
                 "!", "chiffres", "longueur", "mots"]
        raw = [0.69, 1.0, 0.0, 0.0, 1.79, 1.79, 0.0, 0.0, 0.03, 0.07, 5.01, 3.58]
        mx = [1.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.2, 0.3, 8.0, 6.0]
        fb = VGroup()
        for n, v, m in zip(names, raw, mx):
            lab = T(n, 18, SOFT)
            bar = Rectangle(width=max(0.9 * min(v / m, 1), 0.03), height=0.16, fill_color=ORANGE,
                            fill_opacity=0.9, stroke_width=0)
            bar.move_to(lab.get_left() + RIGHT * 1.15, aligned_edge=LEFT)
            fb.add(VGroup(lab, bar))
        fb.arrange_in_grid(6, 2, buff=(0.35, 0.14), col_alignments="ll", flow_order="dr")
        fb.move_to(LEFT * 4.6 + DOWN * 1.5)
        fl = T("12 features (email de test)", 20, ORANGE).next_to(fb, UP, buff=0.2)
        self.play(FadeIn(fl), LaggedStart(*[FadeIn(f) for f in fb], lag_ratio=0.05))
        self.say("En parallèle : 12 indices calculés à la main (liens, IP, urgence…).", 2.8)

        # Concaténation
        cat_top = vec(8, 12, GREEN_C, size=0.22)
        cat_bot = vec(3, 13, ORANGE, size=0.22)
        cat = VGroup(cat_top, cat_bot).arrange(DOWN, buff=0).move_to(LEFT * 1.6 + DOWN * 0.1)
        cl = T("140", 22, TXT).next_to(cat, UP, buff=0.2)
        self.play(TransformFromCopy(pooled, cat_top), TransformFromCopy(fb, cat_bot), FadeIn(cl), run_time=1.3)

        # Dense 64 -> 1
        dense = neuron_layer(9, BLUE_C, r=0.12, buff=0.1).move_to(LEFT * 0.1 + DOWN * 0.1)
        dl = T("64 ReLU\ndropout 0,3", 20, BLUE_C).next_to(dense, DOWN, buff=0.15)
        c1 = VGroup(*[Line(s.get_right(), n.get_left(), stroke_width=0.8, stroke_opacity=0.25, color=SOFT)
                      for s in cat for n in dense])
        outn = Circle(0.25, color=ORANGE, fill_color=ORANGE, fill_opacity=0.6).move_to(RIGHT * 1.3 + DOWN * 0.1)
        c2 = connect(dense, [outn])
        ol = T("1 neurone\nlogit z", 20, ORANGE).next_to(outn, DOWN, buff=0.15)
        self.play(Create(c1), FadeIn(dense), FadeIn(dl), run_time=1.2)
        self.play(Create(c2), FadeIn(outn), FadeIn(ol))
        self.say("Une couche de 64 neurones, puis un seul neurone : le logit z.", 2.8)

        # Sigmoïde
        ax = Axes(x_range=[-6, 6, 2], y_range=[0, 1, 0.5], x_length=3.5, y_length=2.4,
                  axis_config={"color": SOFT, "include_tip": False}).move_to(RIGHT * 4.6 + UP * 0.1)
        curve = ax.plot(lambda x: 1 / (1 + np.exp(-x)), color=ORANGE, stroke_width=4)
        thr = DashedLine(ax.c2p(-6, 0.5), ax.c2p(6, 0.5), color=SOFT)
        tl = T("seuil", 18, SOFT).next_to(thr, RIGHT, buff=0.08)
        sl = T("σ(z) = 1 / (1 + exp(−z))", 22, ORANGE).next_to(ax, UP, buff=0.25)
        y1 = T("1", 18, SOFT).next_to(ax.c2p(-6, 1), LEFT, buff=0.1)
        y0 = T("0", 18, SOFT).next_to(ax.c2p(-6, 0), LEFT, buff=0.1)
        self.play(Create(ax), FadeIn(y0), FadeIn(y1), Create(curve), FadeIn(sl))
        self.play(Create(thr), FadeIn(tl))
        z = ValueTracker(-4)
        dot = always_redraw(lambda: Dot(ax.c2p(z.get_value(), 1 / (1 + np.exp(-z.get_value()))),
                                        color=TXT, radius=0.09))

        def verdict():
            p = 1 / (1 + np.exp(-z.get_value()))
            s, c = ("PHISHING", ORANGE) if p >= 0.5 else ("légitime", BLUE_C)
            return T(f"P = {p:.2f} → {s}".replace(".", ","), 24, c, weight=BOLD).next_to(ax, DOWN, buff=0.3)
        v = always_redraw(verdict)
        self.play(FadeIn(dot), FadeIn(v))
        self.play(z.animate.set_value(3.5), run_time=3)
        self.say("La sigmoïde écrase z entre 0 et 1 ; au-dessus du seuil, l'email est signalé.", 3)
        self.play(z.animate.set_value(-1.5), run_time=1.5)
        self.wait(1)
        self.outro()


# =====================================================================
# 09 · Entraînement
# =====================================================================
class S09_Training(Base):
    number, chapter = "09", "Entraînement"

    def construct(self):
        self.chapter_card("Apprendre en corrigeant ses erreurs, lot après lot")
        names = ["Lot de 32 emails", "Propagation avant", "Perte BCE", "Rétropropagation",
                 "AdamW : mise à jour"]
        center = LEFT * 3.3 + DOWN * 0.1
        R = 2.4
        angs = [PI / 2 - i * TAU / 5 for i in range(5)]
        nodes = VGroup()
        for i, (n, ang) in enumerate(zip(names, angs)):
            b = box(n, 2.45, 0.6, ORANGE if i in (2, 4) else GREY_C, size=20)
            b.move_to(center + R * np.array([np.cos(ang), np.sin(ang), 0]))
            nodes.add(b)
        # Flèches le long du cercle, entre deux boîtes (sens horaire)
        arrows = VGroup()
        for ang in angs:
            a = Arc(radius=R, start_angle=ang - 0.5, angle=-(TAU / 5 - 1.0), arc_center=center,
                    stroke_color=ORANGE, stroke_width=3)
            a.add_tip(tip_length=0.18, tip_width=0.18)
            arrows.add(a)
        self.play(LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=0.15))
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.15))

        ax = Axes(x_range=[0, 10, 2], y_range=[0, 1, 0.5], x_length=5, y_length=3.2,
                  axis_config={"color": SOFT, "include_tip": False}).move_to(RIGHT * 3.6 + DOWN * 0.2)
        xl = T("étapes", 20, SOFT).next_to(ax, DOWN, buff=0.15)
        yl = T("perte", 20, SOFT).next_to(ax, LEFT, buff=0.15)
        il = T("courbe illustrative", 18, SOFT).next_to(ax, UP, buff=0.15)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(il))
        t = ValueTracker(0.01)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.03, 400)

        def loss(x):
            return 0.75 * np.exp(-x / 2.5) + 0.12 + noise[int(x * 39) % 400] * (1 - x / 12)
        curve = always_redraw(lambda: ax.plot(loss, x_range=[0, t.get_value(), 0.05], color=ORANGE))
        self.add(curve)
        captions = ["On prend 32 emails…", "…le modèle prédit…", "…on mesure l'erreur…",
                    "…on calcule le gradient de chaque poids…", "…et AdamW ajuste les 1,86 M de poids."]
        for k in range(5):
            hl = nodes[k][0].copy().set_fill(opacity=0).set_stroke(ORANGE, 5)
            self.play(Create(hl), t.animate.set_value(2 * (k + 1)), run_time=1.2)
            self.say(captions[k], 1.2)
            self.play(FadeOut(hl), run_time=0.3)
        self.say("438 tours par epoch ; à chaque tour la perte baisse un peu.", 2.5)
        self.clear()

        # BCE + pos_weight (calculs réels)
        title = T("La perte : punir les erreurs, surtout les phishing ratés", 32, TXT, weight=BOLD).to_edge(UP, buff=1.0)
        f = M("BCE = −[ y·log(p) + (1−y)·log(1−p) ]", 30, ORANGE).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(title), Write(f))
        rows = VGroup(
            T("Phishing (y = 1), le modèle dit p = 0,9   →   −log(0,9) = 0,11", 26, TXT),
            T("Phishing (y = 1), le modèle dit p = 0,2   →   −log(0,2) = 1,61", 26, ORANGE),
            T("pos_weight ≈ 1,7 : erreur sur un phishing × 1,7   →   2,74", 26, ORANGE, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(f, DOWN, buff=0.6)
        for r in rows:
            self.play(FadeIn(r, shift=UP * 0.2))
            self.wait(1.2)
        self.say("pos_weight = emails sûrs / phishing dans le train : il compense le déséquilibre.", 3)
        self.clear()

        # Early stopping
        title = T("Arrêt anticipé sur la PR-AUC de validation", 32, TXT, weight=BOLD).to_edge(UP, buff=1.0)
        self.play(FadeIn(title))
        vals = [0.90, 0.95, 0.97, 0.975, 0.972, 0.974]
        base = Line(LEFT * 4, RIGHT * 4, color=SOFT).shift(DOWN * 2)
        bars = VGroup()
        for i, v in enumerate(vals):
            h = (v - 0.85) * 25
            b = Rectangle(width=0.8, height=h, fill_color=BLUE_C, fill_opacity=0.85, stroke_width=0)
            b.next_to(base, UP, buff=0).set_x(-3.3 + i * 1.3)
            bars.add(b)
        labels = VGroup(*[T(f"ep {i + 1}", 20, SOFT).next_to(b, DOWN, buff=0.15 + 0) for i, b in enumerate(bars)])
        self.play(Create(base))
        for i, (b, l) in enumerate(zip(bars, labels)):
            self.play(GrowFromEdge(b, DOWN), FadeIn(l), run_time=0.5)
            if i == 3:
                star = VGroup(Star(5, outer_radius=0.14, color=ORANGE, fill_opacity=1),
                              T("meilleur → best_weights.pt", 22, ORANGE)).arrange(RIGHT, buff=0.12)
                star.next_to(b, UP, buff=0.15)
                self.play(b.animate.set_fill(ORANGE), FadeIn(star))
        stop = T("2 epochs sans progrès → arrêt", 24, ORANGE, weight=BOLD).next_to(bars[-1], UP, buff=0.9)
        il = T("valeurs illustratives", 18, SOFT).to_corner(UR, buff=0.4)
        self.play(FadeIn(stop), FadeIn(il))
        self.say("On garde le meilleur epoch, pas le dernier : c'est lui qui est exporté.", 3)
        self.clear()

        # Reprise après interruption
        title = T("Reprise après interruption", 32, TXT, weight=BOLD).to_edge(UP, buff=1.0)
        self.play(FadeIn(title))
        tl = NumberLine(x_range=[0, 438, 100], length=11, color=SOFT, include_numbers=False).shift(DOWN * 0.2)
        ends = VGroup(T("étape 0", 20, SOFT).next_to(tl.n2p(0), DOWN, buff=0.3),
                      T("438", 20, SOFT).next_to(tl.n2p(438), DOWN, buff=0.3))
        self.play(Create(tl), FadeIn(ends))
        saves = VGroup()
        for s in (200, 400):
            m = VGroup(Triangle(color=GREEN_C, fill_opacity=1).scale(0.12).rotate(PI).move_to(tl.n2p(s) + UP * 0.35),
                       T(f"sauvegarde {s}", 18, GREEN_C).next_to(tl.n2p(s), UP, buff=0.55))
            saves.add(m)
        prog = Line(tl.n2p(0), tl.n2p(0), color=ORANGE, stroke_width=8)
        self.add(prog)
        self.play(prog.animate.put_start_and_end_on(tl.n2p(0), tl.n2p(310)), FadeIn(saves[0]), run_time=2)
        cut = VGroup(T("Ctrl+C à l'étape 310", 22, ORANGE, weight=BOLD).next_to(tl.n2p(310), DOWN, buff=0.7),
                     Cross(scale_factor=0.15, stroke_color=ORANGE).move_to(tl.n2p(310)))
        self.play(FadeIn(cut))
        self.say("Ctrl+C : la position exacte est sauvegardée (poids, optimiseur, hasard, compteurs).", 3)
        resume = CurvedArrow(tl.n2p(310) + DOWN * 1.3, tl.n2p(310) + DOWN * 0.15, color=GREEN_C)
        self.play(Create(resume))
        self.play(prog.animate.put_start_and_end_on(tl.n2p(0), tl.n2p(438)), FadeIn(saves[1]), run_time=1.5)
        self.say("python train.py reprend à l'étape 310, avec le même ordre de lots (graine = seed + epoch).", 3.2)
        self.say("Écriture atomique : fichier .tmp puis renommage, une coupure ne corrompt rien.", 3)
        self.outro()


# =====================================================================
# 10 · Évaluation
# =====================================================================
class S10_Evaluation(Base):
    number, chapter = "10", "Évaluation"

    def construct(self):
        self.chapter_card("Rater un phishing coûte plus cher qu'une fausse alerte")
        title = T("Référence TF-IDF sur le test : 1 750 emails dont 652 phishing", 28, TXT, weight=BOLD)
        title.to_edge(UP, buff=1.0)
        self.play(FadeIn(title))
        cells = [("640", "phishing détectés", ORANGE, 0.85), ("12", "phishing ratés", ORANGE, 0.25),
                 ("13", "fausses alertes", BLUE_C, 0.25), ("1 085", "sûrs bien classés", BLUE_C, 0.85)]
        cm = VGroup()
        for v, l, c, o in cells:
            sq = Square(2.2, stroke_color=BG, stroke_width=4, fill_color=c, fill_opacity=o)
            txt = VGroup(T(v, 40, TXT, weight=BOLD), T(l, 18, TXT)).arrange(DOWN, buff=0.1).move_to(sq)
            cm.add(VGroup(sq, txt))
        cm.arrange_in_grid(2, 2, buff=0).move_to(LEFT * 2.2 + DOWN * 0.5)
        rl = VGroup(T("vrai : phishing", 20, ORANGE).next_to(cm[0], LEFT, buff=0.2),
                    T("vrai : sûr", 20, BLUE_C).next_to(cm[2], LEFT, buff=0.2))
        cl = VGroup(T("prédit : phishing", 20, SOFT).next_to(cm[0], UP, buff=0.15),
                    T("prédit : sûr", 20, SOFT).next_to(cm[1], UP, buff=0.15))
        self.play(LaggedStart(*[FadeIn(c, scale=0.9) for c in cm], lag_ratio=0.15), FadeIn(rl), FadeIn(cl))
        met = VGroup(
            T("Rappel = 640 / 652 = 98,2 %", 28, ORANGE, weight=BOLD),
            T("part des phishing attrapés", 20, SOFT),
            T("Précision = 640 / 653 = 98,0 %", 28, BLUE_C, weight=BOLD),
            T("part des alertes justifiées", 20, SOFT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(RIGHT * 3.7 + DOWN * 0.3)
        met[2].shift(DOWN * 0.3)
        met[3].shift(DOWN * 0.3)
        self.play(FadeIn(met[:2]))
        self.play(Indicate(cm[1], color=ORANGE))
        self.play(FadeIn(met[2:]))
        self.play(Indicate(cm[2], color=BLUE_C))
        self.say("Priorité au rappel : un phishing raté peut vider un compte bancaire.", 3)
        self.clear()

        # Seuil
        title = T("Le seuil règle le compromis", 32, TXT, weight=BOLD).to_edge(UP, buff=1.0)
        self.play(FadeIn(title))
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 4, 1], x_length=9, y_length=3.4,
                  axis_config={"color": SOFT, "include_tip": False}).shift(DOWN * 0.4)
        x0 = T("0", 20, SOFT).next_to(ax.c2p(0, 0), DOWN, buff=0.15)
        x1 = T("1", 20, SOFT).next_to(ax.c2p(1, 0), DOWN, buff=0.15)
        xl = T("probabilité prédite", 20, SOFT).next_to(ax, DOWN, buff=0.15)

        def g(mu, s):
            return lambda x: 3.5 * np.exp(-((x - mu) ** 2) / (2 * s ** 2))
        safe = ax.plot(g(0.15, 0.1), x_range=[0, 1], color=BLUE_C)
        phish = ax.plot(g(0.85, 0.1), x_range=[0, 1], color=ORANGE)
        sa = ax.get_area(safe, x_range=[0, 1], color=BLUE_C, opacity=0.25)
        pa = ax.get_area(phish, x_range=[0, 1], color=ORANGE, opacity=0.25)
        leg = VGroup(T("emails sûrs", 22, BLUE_C), T("phishing", 22, ORANGE)).arrange(RIGHT, buff=0.6).next_to(title, DOWN, buff=0.3)
        il = T("distributions illustratives", 18, SOFT).to_corner(UR, buff=0.4)
        self.play(Create(ax), FadeIn(x0), FadeIn(x1), FadeIn(xl), FadeIn(il))
        self.play(Create(safe), Create(phish), FadeIn(sa), FadeIn(pa), FadeIn(leg))
        th = ValueTracker(0.5)
        line = always_redraw(lambda: DashedLine(ax.c2p(th.get_value(), 0), ax.c2p(th.get_value(), 3.9), color=TXT))
        miss = always_redraw(lambda: ax.get_area(phish, x_range=[0, th.get_value()], color=ORANGE, opacity=0.9))
        fa = always_redraw(lambda: ax.get_area(safe, x_range=[th.get_value(), 1], color=BLUE_C, opacity=0.9))
        self.play(Create(line), FadeIn(miss), FadeIn(fa))
        self.say("À gauche du seuil : « sûr ». Les zones pleines sont les erreurs.", 2.5)
        self.play(th.animate.set_value(0.3), run_time=2)
        self.say("Seuil bas : moins de phishing ratés, plus de fausses alertes.", 2.5)
        self.play(th.animate.set_value(0.7), run_time=2)
        self.say("Seuil haut : moins de fausses alertes, plus de phishing ratés.", 2.5)
        self.clear()

        # Résultats réels
        title = T("Résultats réels du TF-IDF sur le test", 32, TXT, weight=BOLD).to_edge(UP, buff=1.0)
        hdr = ["Seuil", "Rappel", "Précision", "Phishing ratés", "Fausses alertes"]
        data = [["0,50", "98,2 %", "98,0 %", "12", "13"], ["0,62", "97,6 %", "99,1 %", "16", "6"],
                ["Transformer", "à venir", "—", "—", "—"]]
        tbl = VGroup()
        for r, row in enumerate([hdr] + data):
            line_ = VGroup(*[T(c, 22, SOFT if r == 0 else (ORANGE if r == 3 else TXT),
                               weight=BOLD if r == 0 else NORMAL) for c in row])
            for j, c in enumerate(line_):
                c.move_to(RIGHT * (j - 2) * 2.9)
            tbl.add(line_)
        tbl.arrange(DOWN, buff=0.45).next_to(title, DOWN, buff=0.7)
        for line_ in tbl:
            for j, c in enumerate(line_):
                c.set_x((j - 2) * 2.9)
        self.play(FadeIn(title), LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in tbl], lag_ratio=0.3))
        self.say("Le seuil est choisi sur la validation (rappel visé : 97 %), puis le test est mesuré une seule fois.", 3.5)
        self.say("La référence est déjà très forte : le Transformer doit faire au moins aussi bien.", 3)
        self.outro()


# =====================================================================
# 11 · Le livrable
# =====================================================================
class S11_Deliverable(Base):
    number, chapter = "11", "Le livrable"

    def construct(self):
        self.chapter_card("Un seul fichier, aucun réentraînement")
        file = VGroup(
            Polygon([-1, 1.3, 0], [0.6, 1.3, 0], [1, 0.9, 0], [1, -1.3, 0], [-1, -1.3, 0],
                    color=ORANGE, fill_color=PANEL, fill_opacity=1, stroke_width=3),
            M(".pt", 36, ORANGE),
        ).move_to(LEFT * 4.5)
        fl = M("phishing_model.pt", 26, ORANGE).next_to(file, DOWN, buff=0.3)
        self.play(DrawBorderThenFill(file[0]), Write(file[1]), FadeIn(fl))
        items = VGroup(*[T(s, 26, TXT) for s in [
            "• poids et architecture du Transformer",
            "• tokenizer BPE (8 000 tokens)",
            "• moyennes / écarts-types des 12 features",
            "• seuil de décision choisi sur la validation",
            "• scores de test",
        ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 1.6 + UP * 0.3)
        arrows = VGroup(*[Arrow(file.get_right(), it.get_left(), buff=0.2, color=SOFT, stroke_width=2,
                                max_tip_length_to_length_ratio=0.06) for it in items])
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(i)) for a, i in zip(arrows, items)],
                              lag_ratio=0.3))
        sec = T("Chargé avec weights_only=True : le fichier ne peut pas exécuter de code.", 24, GREEN_C)
        sec.next_to(VGroup(file, fl, items), DOWN, buff=0.6)
        self.play(FadeIn(sec))
        self.say("Tout ce qu'il faut pour prédire, et rien d'autre.", 2.5)
        self.clear()

        # Utilisation
        code = VGroup(
            M("from detector import PhishingDetector", 26),
            M("det = PhishingDetector.load()", 26),
            M('det.predict("URGENT: verify your…")', 26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        cb = SurroundingRectangle(code, buff=0.4, color=GREY_C, fill_color=PANEL, fill_opacity=1, corner_radius=0.12)
        g = VGroup(cb, code).move_to(UP * 1.2)
        self.play(FadeIn(cb), AddTextLetterByLetter(code[0]), run_time=1.2)
        self.play(AddTextLetterByLetter(code[1]), run_time=1)
        self.play(AddTextLetterByLetter(code[2]), run_time=1)
        cli = VGroup(M("python predict.py", 24, SOFT), M("python predict.py --file email.txt", 24, SOFT),
                     M('python predict.py --debug "…"', 24, SOFT)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        cli.next_to(g, DOWN, buff=0.6)
        self.play(FadeIn(cli))
        self.say("Trois lignes en Python, ou une commande dans le terminal.", 2.8)
        self.clear()

        # Récapitulatif
        title = T("Le modèle en chiffres", 40, ORANGE, weight=BOLD).to_edge(UP, buff=0.9)
        facts = [("256", "tokens par email"), ("8 000", "tokens de vocabulaire"), ("128", "dimensions"),
                 ("4 × 32", "têtes d'attention"), ("4", "blocs encodeurs"), ("512", "neurones feed-forward"),
                 ("12", "features manuelles"), ("64 → 1", "neurones de sortie"), ("1 859 201", "paramètres")]
        cards = VGroup()
        for v, l in facts:
            cards.add(VGroup(T(v, 40, TXT, weight=BOLD), T(l, 20, SOFT)).arrange(DOWN, buff=0.1))
        cards.arrange_in_grid(3, 3, buff=(1.4, 0.6)).next_to(title, DOWN, buff=0.6)
        cards[-1][0].set_color(ORANGE)
        self.play(FadeIn(title), LaggedStart(*[FadeIn(c, scale=0.85) for c in cards], lag_ratio=0.1))
        self.wait(3)
        self.clear(keep_head=False)
        end = VGroup(T("Merci", 72, TXT, weight=BOLD), T("Questions ?", 36, ORANGE)).arrange(DOWN, buff=0.4)
        self.play(Write(end[0]), FadeIn(end[1], shift=UP * 0.2))
        self.wait(2.5)
        self.play(FadeOut(end))


SCENES = ["S01_Intro", "S02_Pipeline", "S03_Normalisation", "S04_BPE", "S05_Embeddings",
          "S06_Attention", "S07_Encoder", "S08_Head", "S09_Training", "S10_Evaluation",
          "S11_Deliverable"]
