"""Vidéo : modélisation du réseau de neurones du détecteur de phishing (Manim CE).

Neurones, connexions et flux du signal, couche par couche.
    Aperçu d'une scène :  manim -ql reseau_neurones.py N02_Reseau
    Tout rendre        :  python render_all.py [--preview]

Les tailles (128, 512, 64, 1, 4 blocs, 4 têtes, ids de tokens) sont celles du vrai modèle.
Les activations et poids dessinés sont illustratifs (le texte à l'écran le précise).
"""
from manim import *
import numpy as np

BG = "#0F1B2A"
PANEL = "#1B2A3D"
ORANGE = "#F08A4B"
BLUE_C = "#7FB3E6"
GREEN_C = "#7BC8A4"
RED_C = "#E06C75"
TXT = "#E9EEF3"
SOFT = "#A9B8C8"
GREY_C = "#4A5A6E"
FONT = "Arial"
MONO = "Consolas"


# ------------------------------------------------------------------ helpers
def T(s, size=28, color=TXT, weight=NORMAL, **kw):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight, **kw)


def M(s, size=26, color=TXT, **kw):
    return Text(s, font=MONO, font_size=size, color=color, **kw)


def fr(x):
    """0.8 -> '0,8' ; -0.2 -> '−0,2'"""
    s = f"{x:.2f}".rstrip("0").rstrip(".") if abs(x) >= 0.005 else "0"
    return s.replace(".", ",").replace("-", "−")


def neuron(r=0.15, color=BLUE_C, opacity=0.08):
    return Circle(radius=r, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=opacity)


def column(n, x=0, color=BLUE_C, r=0.13, buff=0.12):
    return VGroup(*[neuron(r, color) for _ in range(n)]).arrange(DOWN, buff=buff).set_x(x)


def link(a, b, color=SOFT, width=1.0, opacity=0.3):
    """Connexion entre deux neurones (cercles), du bord de l'un au bord de l'autre."""
    pa, pb = a.get_center(), b.get_center()
    u = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-6)
    ra, rb = a.width / 2, b.width / 2
    return Line(pa + u * ra, pb - u * rb, stroke_color=color, stroke_width=width, stroke_opacity=opacity)


def fc(la, lb, color=SOFT, width=1.0, opacity=0.25):
    """Toutes les connexions entre deux couches (fully connected)."""
    return VGroup(*[link(a, b, color, width, opacity) for a in la for b in lb])


def straight(la, lb, color=SOFT, width=1.5, opacity=0.4):
    """Connexions une-à-une (même token d'une couche à l'autre)."""
    return VGroup(*[link(a, b, color, width, opacity) for a, b in zip(la, lb)])


def flow(edges, color=ORANGE, run_time=0.9, width=3.5):
    """Impulsions qui parcourent les connexions dans leur sens."""
    return ShowPassingFlash(edges.copy().set_stroke(color=color, width=width, opacity=1),
                            time_width=0.45, run_time=run_time)


def reverse(edges):
    return VGroup(*[e.copy().reverse_points() for e in edges])


def light(layer, seed=0, color=BLUE_C, low=0.15, high=0.95, vals=None):
    rng = np.random.default_rng(seed)
    vals = vals if vals is not None else rng.uniform(low, high, len(layer))
    return [n.animate.set_fill(color, opacity=float(v)) for n, v in zip(layer, vals)]


def pill(s, fill=GREY_C, color=TXT, size=24):
    t = M(s, size, color)
    r = RoundedRectangle(corner_radius=0.1, width=max(t.width + 0.3, 0.5), height=0.5,
                         fill_color=fill, fill_opacity=1, stroke_width=0)
    t.move_to(r)
    return VGroup(r, t)


class Base(Scene):
    number, chapter = "", ""

    def setup(self):
        self.camera.background_color = BG
        self.cap = None
        self.head = None

    def chapter_card(self, subtitle=""):
        n = T(self.number, 90, ORANGE, weight=BOLD)
        t = T(self.chapter, 50, TXT, weight=BOLD)
        g = VGroup(n, t).arrange(DOWN, buff=0.25)
        if subtitle:
            g.add(T(subtitle, 28, SOFT).next_to(t, DOWN, buff=0.3))
        self.play(FadeIn(g, shift=UP * 0.2), run_time=1)
        self.wait(1.2)
        self.play(FadeOut(g), run_time=0.6)
        self.head = T(f"{self.number} · {self.chapter}", 22, SOFT).to_corner(UL, buff=0.3)
        self.play(FadeIn(self.head), run_time=0.3)

    def say(self, s, hold=2.4, color=TXT):
        new = T(s, 27, color)
        if new.width > 13.2:
            new.scale_to_fit_width(13.2)
        new.to_edge(DOWN, buff=0.3)
        anims = [FadeIn(new, shift=UP * 0.15)]
        if self.cap is not None:
            anims.append(FadeOut(self.cap, shift=UP * 0.15))
        self.play(*anims, run_time=0.5)
        self.cap = new
        self.wait(hold)

    def clear(self, keep_head=True):
        mobs = [m for m in self.mobjects if not (keep_head and m is self.head)]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=0.6)
        self.cap = None


# ------------------------------------------------------------------ réseau complet (vue d'ensemble)
WORDS = ["urgent", "verify", "password", "http", "login"]
YS = [1.6, 0.8, 0.0, -0.8, -1.6]


class Macro:
    """Le modèle entier : 1 cercle bleu = le vecteur (128 nombres) d'un token."""

    def __init__(self, words=WORDS):
        r = 0.17
        xs = [-4.6, -3.4, -2.3, -1.1, 0.0]
        self.toks = VGroup(*[T(w, 20, TXT) for w in words])
        for t, y in zip(self.toks, YS):
            t.move_to([-5.3, y, 0], aligned_edge=RIGHT)
        self.cols = [VGroup(*[neuron(r, BLUE_C).move_to([x, y, 0]) for y in YS]) for x in xs]
        self.e_in = VGroup(*[Line(t.get_right() + RIGHT * 0.08, c.get_left(), stroke_color=SOFT,
                                  stroke_width=1.5, stroke_opacity=0.5) for t, c in zip(self.toks, self.cols[0])])
        # attention = tous les tokens vers tous les tokens ; feed-forward = chaque token seul
        self.edges = [fc(self.cols[0], self.cols[1], opacity=0.22),
                      straight(self.cols[1], self.cols[2]),
                      fc(self.cols[2], self.cols[3], opacity=0.22),
                      straight(self.cols[3], self.cols[4])]
        self.pool = neuron(0.2, GREEN_C).move_to([1.3, 1.0, 0])
        self.feats = VGroup(*[neuron(0.13, ORANGE).move_to([1.3, y, 0]) for y in (0.1, -0.4, -0.9)])
        self.e_pool = fc(self.cols[-1], [self.pool], opacity=0.4)
        self.concat = VGroup(self.pool, *self.feats)
        self.dense = column(8, 2.8, BLUE_C, r=0.13, buff=0.2)
        self.e_dense = fc(self.concat, self.dense, opacity=0.22)
        self.out = neuron(0.28, ORANGE).move_to([4.2, 0, 0])
        self.e_out = fc(self.dense, [self.out], opacity=0.35)

        lab = lambda s, x, c=SOFT: T(s, 16, c).move_to([x, 2.35, 0])
        self.labels = VGroup(lab("tokens", -5.8), lab("embedding", -4.6), lab("attention", -3.4),
                             lab("feed-\nforward", -2.3), lab("attention", -1.1), lab("feed-\nforward", 0.0),
                             lab("pooling", 1.3, GREEN_C), lab("dense 64", 2.8), lab("sortie", 4.2, ORANGE))
        self.flab = T("12 features\n(3 dessinées)", 14, ORANGE).next_to(self.feats, DOWN, buff=0.12)
        br = BraceBetweenPoints([-3.65, -2.05, 0], [0.25, -2.05, 0], DOWN, color=SOFT)
        self.brace = VGroup(br, T("bloc encodeur × 4  (2 dessinés)", 18, SOFT).next_to(br, DOWN, buff=0.1))
        self.legend = T("1 cercle bleu = le vecteur de 128 nombres d'un token", 18, SOFT).move_to([0, -3.0, 0])

        self.net = VGroup(self.e_in, *self.edges, self.e_pool, self.e_dense, self.e_out,
                          *self.cols, self.concat, self.dense, self.out)
        self.all = VGroup(self.toks, self.net, self.labels, self.flab, self.brace, self.legend)

    def stages(self):
        """(connexions, couche d'arrivée, couleur des neurones)"""
        return [(self.e_in, self.cols[0], BLUE_C), (self.edges[0], self.cols[1], BLUE_C),
                (self.edges[1], self.cols[2], BLUE_C), (self.edges[2], self.cols[3], BLUE_C),
                (self.edges[3], self.cols[4], BLUE_C), (self.e_pool, VGroup(self.pool), GREEN_C),
                (self.e_dense, self.dense, BLUE_C), (self.e_out, VGroup(self.out), ORANGE)]

    def forward(self, scene, color=ORANGE, rt=0.8, seed=0, out_opacity=0.9, captions=None):
        for i, (edges, layer, c) in enumerate(self.stages()):
            anims = [flow(edges, color, rt)]
            if i == 7:
                anims.append(self.out.animate.set_fill(ORANGE, opacity=out_opacity))
            else:
                anims += light(layer, seed + i, c)
            if i == 5:
                anims += light(self.feats, seed + 50, ORANGE, 0.4, 1.0)
            scene.play(*anims, run_time=rt)
            if captions and i in captions:
                scene.say(captions[i], 2.2)

    def backward(self, scene, rt=0.6):
        groups = [self.e_out, self.e_dense, self.e_pool, *reversed(self.edges), self.e_in]
        for g in groups[:-1]:
            scene.play(flow(reverse(g), GREEN_C, rt), run_time=rt)

    def update_weights(self, scene, seed=0):
        rng = np.random.default_rng(seed)
        anims = []
        for g in [*self.edges, self.e_pool, self.e_dense, self.e_out]:
            for e in g:
                anims.append(e.animate.set_stroke(width=float(rng.uniform(0.6, 2.6)),
                                                  opacity=float(rng.uniform(0.12, 0.5))))
        scene.play(*anims, run_time=0.8)

    def reset(self):
        anims = []
        for layer in [*self.cols, self.dense, self.feats]:
            anims += [n.animate.set_fill(opacity=0.08) for n in layer]
        anims += [self.pool.animate.set_fill(opacity=0.08), self.out.animate.set_fill(opacity=0.08)]
        return anims


# =====================================================================
# 01 · Le neurone artificiel
# =====================================================================
class N01_Neurone(Base):
    number, chapter = "01", "Le neurone artificiel"

    def construct(self):
        title = T("Modélisation du réseau de neurones", 56, TXT, weight=BOLD)
        sub = T("Détecteur de phishing · Transformer from scratch", 32, ORANGE)
        g = VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait(1.8)
        self.play(FadeOut(g))
        self.chapter_card("La brique de base : multiplier, additionner, activer")

        xs = [0.8, 0.2, 0.5, 0.9]
        ws = [0.6, -0.4, 0.9, 0.3]
        b = -0.2
        ins = VGroup(*[neuron(0.36, BLUE_C, 0.15 + 0.6 * x) for x in xs]).arrange(DOWN, buff=0.32)
        ins.move_to(LEFT * 5.2 + UP * 0.4)
        vals = VGroup(*[T(fr(x), 22).move_to(n) for n, x in zip(ins, xs)])
        il = VGroup(*[T(f"x{i + 1}", 22, SOFT).next_to(n, LEFT, buff=0.15) for i, n in enumerate(ins)])
        sig = Circle(0.6, stroke_color=ORANGE, stroke_width=3, fill_color=PANEL, fill_opacity=1)
        sig.move_to(LEFT * 1.4 + UP * 0.4)
        st = T("Σ", 44, ORANGE).move_to(sig)
        edges = VGroup(*[link(n, sig, BLUE_C if w > 0 else RED_C, 1.5 + 5 * abs(w), 0.85)
                         for n, w in zip(ins, ws)])
        wl = VGroup()
        for e, w in zip(edges, ws):
            t = T(f"w = {fr(w)}", 20, BLUE_C if w > 0 else RED_C)
            t.move_to(e.point_from_proportion(0.42) + UP * 0.22)
            wl.add(t)
        self.play(LaggedStart(*[GrowFromCenter(n) for n in ins], lag_ratio=0.1), FadeIn(vals), FadeIn(il))
        self.say("Un neurone reçoit des nombres en entrée : x1, x2, x3, x4.", 2.2)
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.1), FadeIn(sig), FadeIn(st))
        self.play(FadeIn(wl))
        self.say("Chaque connexion a un poids w, appris pendant l'entraînement (rouge = négatif).", 2.8)

        # Multiplications
        prods = VGroup(*[T(f"{fr(x)} × {fr(w)} = {fr(x * w)}", 22, TXT) for x, w in zip(xs, ws)])
        prods.add(T(f"+ biais b = {fr(b)}", 22, ORANGE))
        prods.arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(sig, DOWN, buff=0.45)
        for e, p in zip(edges, prods[:4]):
            self.play(flow(VGroup(e), ORANGE, 0.6), FadeIn(p, shift=LEFT * 0.2), run_time=0.6)
        self.play(FadeIn(prods[4]))
        z = sum(x * w for x, w in zip(xs, ws)) + b
        zt = T(f"z = {fr(z)}", 30, ORANGE, weight=BOLD).next_to(sig, RIGHT, buff=0.35).shift(UP * 0.55)
        self.play(sig.animate.set_fill(ORANGE, opacity=0.35), Write(zt))
        self.say("Le neurone additionne entrées × poids, plus un biais : z = 0,92.", 2.6)

        # Activation ReLU
        ax = Axes(x_range=[-2, 2, 1], y_range=[0, 2, 1], x_length=2.4, y_length=1.6,
                  axis_config={"color": SOFT, "include_tip": False}).move_to(RIGHT * 2.2 + UP * 0.4)
        relu = ax.plot(lambda x: max(x, 0), color=ORANGE, stroke_width=4)
        frame = SurroundingRectangle(ax, buff=0.2, color=GREY_C, corner_radius=0.1)
        al = T("activation ReLU : max(0, z)", 20, SOFT).next_to(frame, UP, buff=0.12)
        a1 = Arrow(sig.get_right(), frame.get_left(), buff=0.1, color=SOFT, stroke_width=3)
        self.play(GrowArrow(a1), Create(frame), Create(ax), Create(relu), FadeIn(al))
        d = Dot(ax.c2p(z, max(z, 0)), color=TXT)
        self.play(FadeIn(d, scale=2))
        out = neuron(0.42, BLUE_C, 0.85).move_to(RIGHT * 5.3 + UP * 0.4)
        ot = T(fr(max(z, 0)), 24).move_to(out)
        a2 = Arrow(frame.get_right(), out.get_left(), buff=0.1, color=SOFT, stroke_width=3)
        self.play(GrowArrow(a2), GrowFromCenter(out), FadeIn(ot))
        ol = T("sortie", 20, SOFT).next_to(out, DOWN, buff=0.15)
        self.play(FadeIn(ol))
        self.say("Puis une activation décide de ce qui passe : ici 0,92, le neurone s'allume.", 2.8)

        # Deuxième exemple : neurone éteint
        xs2 = [0.1, 0.9, 0.2, 0.1]
        z2 = sum(x * w for x, w in zip(xs2, ws)) + b
        new_vals = VGroup(*[T(fr(x), 22).move_to(n) for n, x in zip(ins, xs2)])
        new_prods = VGroup(*[T(f"{fr(x)} × {fr(w)} = {fr(x * w)}", 22, TXT) for x, w in zip(xs2, ws)])
        new_prods.add(T(f"+ biais b = {fr(b)}", 22, ORANGE))
        new_prods.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(prods, aligned_edge=UL)
        self.play(*[n.animate.set_fill(opacity=0.15 + 0.6 * x) for n, x in zip(ins, xs2)],
                  Transform(vals, new_vals), Transform(prods, new_prods))
        self.play(flow(edges, ORANGE, 0.8))
        zt2 = T(f"z = {fr(z2)}", 30, RED_C, weight=BOLD).move_to(zt)
        self.play(Transform(zt, zt2), d.animate.move_to(ax.c2p(z2, 0)))
        ot2 = T("0", 24).move_to(out)
        self.play(out.animate.set_fill(GREY_C, opacity=0.15), Transform(ot, ot2))
        self.say("Autres entrées : z = −0,29, ReLU donne 0. Le neurone reste éteint.", 2.8, RED_C)
        self.clear()

        # Une couche, puis un réseau
        inp = column(4, -4.5, BLUE_C, r=0.22, buff=0.35)
        hid = column(5, -1.0, BLUE_C, r=0.22, buff=0.3)
        lay_e = fc(inp, hid, SOFT, 1.2, 0.35)
        self.play(FadeIn(inp), FadeIn(hid))
        self.play(Create(lay_e), run_time=1.2)
        tt = T("Une couche : chaque neurone est relié à toutes les entrées\n4 × 5 = 20 poids + 5 biais",
               24, TXT).scale_to_fit_width(6).move_to(RIGHT * 3.4)
        self.play(FadeIn(tt))
        self.play(flow(lay_e, ORANGE, 1.2), *light(hid, 3), run_time=1.2)
        self.wait(1.2)
        self.play(FadeOut(tt))
        hid2 = column(5, 2.5, BLUE_C, r=0.22, buff=0.3)
        outl = column(1, 5.5, ORANGE, r=0.3)
        e2 = fc(hid, hid2, SOFT, 1.2, 0.35)
        e3 = fc(hid2, outl, SOFT, 1.2, 0.35)
        self.play(FadeIn(hid2), Create(e2), FadeIn(outl), Create(e3), run_time=1.2)
        self.say("Un réseau : des couches empilées, le signal circule de gauche à droite.", 1.6)
        for _ in range(2):
            self.play(flow(lay_e, ORANGE, 0.7), *light(hid, _ + 4), run_time=0.7)
            self.play(flow(e2, ORANGE, 0.7), *light(hid2, _ + 9), run_time=0.7)
            self.play(flow(e3, ORANGE, 0.7), outl[0].animate.set_fill(ORANGE, opacity=0.9), run_time=0.7)
            self.play(*[n.animate.set_fill(opacity=0.08) for n in [*hid, *hid2, *outl]], run_time=0.4)
        self.say("Notre modèle : 1 859 201 poids et biais de ce type, organisés en Transformer.", 3)
        self.clear(keep_head=False)


# =====================================================================
# 02 · Le réseau complet
# =====================================================================
class N02_Reseau(Base):
    number, chapter = "02", "Le réseau complet"

    def construct(self):
        self.chapter_card("Le chemin d'un email, de l'entrée à la décision")
        mc = Macro()
        self.play(FadeIn(mc.toks, shift=RIGHT * 0.2), FadeIn(mc.labels))
        self.play(*[FadeIn(c) for c in mc.cols], FadeIn(mc.concat), FadeIn(mc.dense), FadeIn(mc.out),
                  FadeIn(mc.flab), run_time=1)
        self.play(Create(mc.e_in), *[Create(e) for e in mc.edges], Create(mc.e_pool),
                  Create(mc.e_dense), Create(mc.e_out), run_time=2)
        self.play(FadeIn(mc.brace), FadeIn(mc.legend))
        self.say("Voici le modèle entier. Chaque cercle bleu représente le vecteur d'un token.", 3)
        self.say("Attention : tous les tokens se connectent entre eux. Feed-forward : chaque token seul.", 3.2)
        caps = {0: "1 · Chaque token devient un vecteur de 128 nombres (embedding).",
                1: "2 · L'attention mélange l'information de tous les tokens.",
                2: "3 · Le feed-forward transforme chaque token séparément.",
                4: "4 · Ce bloc est répété 4 fois (2 sont dessinés).",
                5: "5 · Moyenne des tokens + 12 indices calculés (liens, IP, urgence…).",
                6: "6 · 64 neurones combinent tout.",
                7: "7 · Un seul neurone de sortie : la probabilité de phishing."}
        mc.forward(self, ORANGE, 0.9, seed=1, captions=caps)
        p = T("P(phishing)", 26, ORANGE, weight=BOLD).next_to(mc.out, RIGHT, buff=0.3)
        self.play(Write(p), Flash(mc.out, color=ORANGE, flash_radius=0.5))
        self.say("Le signal a traversé tout le réseau en un seul passage, de gauche à droite.", 3)
        self.play(*mc.reset(), FadeOut(p))
        self.say("Même passage, en accéléré :", 1.2)
        mc.forward(self, ORANGE, 0.35, seed=7)
        self.wait(1)
        self.clear(keep_head=False)


# =====================================================================
# 03 · Embedding : du token au vecteur de neurones
# =====================================================================
class N03_Embedding(Base):
    number, chapter = "03", "Embedding"

    def construct(self):
        self.chapter_card("Un token devient une colonne de 128 valeurs")
        tok = pill("password", GREY_C, size=28).move_to(LEFT * 5.4 + UP * 1.6)
        tid = pill("id 3905", "#2B4A6B", size=28).next_to(tok, DOWN, buff=0.4)
        self.play(FadeIn(tok))
        self.play(TransformFromCopy(tok, tid))
        rng = np.random.default_rng(2)
        table = VGroup(*[VGroup(*[Square(0.22, stroke_width=0.6, stroke_color=BG, fill_color=BLUE_C,
                                         fill_opacity=rng.uniform(0.15, 0.9)) for _ in range(8)]).arrange(RIGHT, buff=0)
                         for _ in range(9)]).arrange(DOWN, buff=0).move_to(LEFT * 2.2 + UP * 0.6)
        tl = T("table apprise : 8 000 lignes × 128", 20, SOFT).next_to(table, UP, buff=0.2)
        self.play(FadeIn(table), FadeIn(tl))
        hl = SurroundingRectangle(table[5], color=ORANGE, buff=0.02)
        arr = Arrow(tid.get_right(), table[5].get_left(), buff=0.12, color=ORANGE, stroke_width=3)
        self.play(GrowArrow(arr), Create(hl))
        self.say("L'id du token désigne une ligne de la table.", 2)

        # La ligne devient une colonne de neurones
        vals = [sq.get_fill_opacity() for sq in table[5]]
        col = VGroup(*[neuron(0.16, BLUE_C, v) for v in vals]).arrange(DOWN, buff=0.08).move_to(RIGHT * 1.4 + UP * 0.6)
        self.play(*[TransformFromCopy(sq, n) for sq, n in zip(table[5], col)], run_time=1.3)
        cl = T("token", 20, BLUE_C).next_to(col, UP, buff=0.15)
        self.play(FadeIn(cl))
        plus = T("+", 40, ORANGE).next_to(col, RIGHT, buff=0.3)
        pos = VGroup(*[neuron(0.16, ORANGE, rng.uniform(0.15, 0.9)) for _ in range(8)]).arrange(DOWN, buff=0.08)
        pos.next_to(plus, RIGHT, buff=0.3)
        pl = T("position", 20, ORANGE).next_to(pos, UP, buff=0.15)
        self.play(FadeIn(plus), FadeIn(pos, shift=LEFT * 0.2), FadeIn(pl))
        eq = T("=", 40, TXT).next_to(pos, RIGHT, buff=0.3)
        res = VGroup(*[neuron(0.16, GREEN_C, rng.uniform(0.2, 0.95)) for _ in range(8)]).arrange(DOWN, buff=0.08)
        res.next_to(eq, RIGHT, buff=0.3)
        self.play(FadeIn(eq), TransformFromCopy(VGroup(col, pos), res), run_time=1.2)
        note = T("128 valeurs par colonne (8 dessinées) · valeurs illustratives", 18, SOFT).next_to(VGroup(col, res), DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.say("On ajoute un vecteur de position, sinon le réseau ignorerait l'ordre des mots.", 3)
        self.clear()

        # Tous les tokens -> colonnes -> cercles du schéma global
        cols = VGroup()
        labs = VGroup()
        for i, w in enumerate(WORDS):
            c = VGroup(*[neuron(0.13, GREEN_C, rng.uniform(0.2, 0.95)) for _ in range(8)]).arrange(DOWN, buff=0.07)
            cols.add(c)
        cols.arrange(RIGHT, buff=0.9).move_to(UP * 0.3)
        for c, w in zip(cols, WORDS):
            labs.add(T(w, 20, TXT).next_to(c, DOWN, buff=0.25))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cols], lag_ratio=0.15), FadeIn(labs))
        self.say("Chaque token de l'email obtient sa colonne : une matrice tokens × 128.", 2.8)
        circles = VGroup(*[neuron(0.3, BLUE_C, 0.6).move_to(c) for c in cols])
        self.play(*[ReplacementTransform(c, k) for c, k in zip(cols, circles)], run_time=1.3)
        self.say("Pour la suite, on résume chaque colonne par un seul cercle.", 2.5)
        self.clear(keep_head=False)


# =====================================================================
# 04 · Attention : les connexions entre tokens
# =====================================================================
class N04_Attention(Base):
    number, chapter = "04", "Self-attention"

    def construct(self):
        self.chapter_card("Les tokens s'échangent de l'information")
        xs = np.linspace(-4.4, 4.4, 5)
        bot = VGroup(*[neuron(0.3, BLUE_C, 0.6).move_to([x, -1.5, 0]) for x in xs])
        top = VGroup(*[neuron(0.3, BLUE_C, 0.08).move_to([x, 1.6, 0]) for x in xs])
        bl = VGroup(*[T(w, 22).next_to(c, DOWN, buff=0.2) for w, c in zip(WORDS, bot)])
        tl = VGroup(*[T(w, 22, SOFT).next_to(c, UP, buff=0.2) for w, c in zip(WORDS, top)])
        lb = T("avant", 20, SOFT).next_to(bot, LEFT, buff=0.4)
        lt = T("après", 20, SOFT).next_to(top, LEFT, buff=0.4)
        self.play(FadeIn(bot), FadeIn(bl), FadeIn(lb))
        self.play(FadeIn(top), FadeIn(tl), FadeIn(lt))

        # Zoom : Q, K, V sont trois petites couches de neurones
        f = 2
        zoom = VGroup(*[neuron(0.11, BLUE_C, 0.7) for _ in range(4)]).arrange(DOWN, buff=0.08).move_to([-1.0, 0.05, 0])
        groups = VGroup(*[VGroup(*[neuron(0.09, c, 0.6) for _ in range(3)]).arrange(DOWN, buff=0.06)
                          for c in (ORANGE, BLUE_C, GREEN_C)]).arrange(DOWN, buff=0.25).move_to([0.9, 0.05, 0])
        qe = VGroup(*[fc(zoom, g, c, 0.8, 0.4) for g, c in zip(groups, (ORANGE, BLUE_C, GREEN_C))])
        gl = VGroup(T("Q", 20, ORANGE), T("K", 20, BLUE_C), T("V", 20, GREEN_C))
        for t, g in zip(gl, groups):
            t.next_to(g, RIGHT, buff=0.15)
        frame = SurroundingRectangle(VGroup(zoom, groups, gl), color=GREY_C, buff=0.2, corner_radius=0.1)
        zl = T("zoom sur « password »", 18, SOFT).next_to(frame, UP, buff=0.1)
        self.play(FadeIn(frame), FadeIn(zl), TransformFromCopy(bot[f], zoom))
        self.play(Create(qe), FadeIn(groups), FadeIn(gl), run_time=1.2)
        self.play(flow(qe, ORANGE, 1.0))
        self.say("Chaque token passe dans 3 petites couches : Q (ce qu'il cherche), K (ce qu'il offre), V (son contenu).", 3.4)
        self.play(FadeOut(VGroup(frame, zl, zoom, groups, gl, qe)))

        # Les connexions de « password »
        scores = np.array([2.1, 0.2, 1.0, 1.5, 1.8])
        w = np.exp(scores) / np.exp(scores).sum()
        conns = VGroup(*[link(b, top[f], ORANGE, 1 + 9 * wi / w.max(), 0.25 + 0.75 * wi / w.max())
                         for b, wi in zip(bot, w)])
        pct = VGroup(*[T(f"{wi * 100:.0f} %", 20, ORANGE).move_to(c.point_from_proportion(0.3) + RIGHT * 0.35)
                       for c, wi in zip(conns, w)])
        self.play(LaggedStart(*[Create(c) for c in conns], lag_ratio=0.1))
        self.play(FadeIn(pct))
        ill = T("poids illustratifs", 18, SOFT).to_corner(UR, buff=0.35)
        self.play(FadeIn(ill))
        self.say("« password » compare son Q aux K des autres : plus c'est proche, plus la connexion est forte.", 3.2)
        self.play(flow(conns, ORANGE, 1.4), top[f].animate.set_fill(ORANGE, opacity=0.85), run_time=1.4)
        self.say("Il récupère surtout l'information de « urgent », « login » et « http ».", 2.6)
        self.play(FadeOut(pct))

        # Tous les tokens en même temps
        mesh = fc(bot, top, SOFT, 1.0, 0.3)
        self.play(FadeOut(conns), Create(mesh), run_time=1.2)
        self.play(flow(mesh, ORANGE, 1.6), *light(top, 3, BLUE_C, 0.5, 0.95), run_time=1.6)
        self.say("Tous les tokens le font en même temps : 5 × 5 connexions ici, 256 × 256 dans le vrai modèle.", 3.2)

        # 4 têtes
        cols = [ORANGE, BLUE_C, GREEN_C, "#C9A0E8"]
        heads = VGroup()
        rng = np.random.default_rng(4)
        for k, c in enumerate(cols):
            h = VGroup(*[Line(b.get_top(), t_.get_bottom(), stroke_color=c,
                              stroke_width=float(rng.uniform(0.5, 4)), stroke_opacity=float(rng.uniform(0.15, 0.8)))
                         for b in bot for t_ in top]).shift(RIGHT * (k - 1.5) * 0.06)
            heads.add(h)
        self.play(FadeOut(mesh), LaggedStart(*[Create(h) for h in heads], lag_ratio=0.3), run_time=2)
        hl = VGroup(*[T(f"tête {i + 1}", 20, c) for i, c in enumerate(cols)]).arrange(RIGHT, buff=0.5).move_to(UP * 2.75)
        self.play(FadeOut(tl), FadeIn(hl))
        self.say("4 têtes en parallèle : chacune apprend ses propres connexions (4 × 32 = 128).", 3)
        self.play(*[flow(h, c, 1.4) for h, c in zip(heads, cols)], run_time=1.4)

        # Padding : aucune connexion
        pads = VGroup(*[neuron(0.3, GREY_C, 0.1).move_to([x, -1.5, 0]) for x in (5.4, 6.3)])
        self.play(*[m.animate.shift(LEFT * 0.9) for m in (bot, bl, top, heads, lb, lt, hl)], run_time=0.8)
        pads.shift(LEFT * 0.9)
        pl = T("[PAD]", 18, SOFT).next_to(pads, DOWN, buff=0.2)
        crosses = VGroup(*[Cross(p, stroke_color=RED_C, stroke_width=3, scale_factor=0.6) for p in pads])
        self.play(FadeIn(pads), FadeIn(pl))
        self.play(Create(crosses))
        self.say("Les positions de remplissage [PAD] sont masquées : aucune connexion ne part d'elles.", 3)
        self.clear(keep_head=False)


# =====================================================================
# 05 · Feed-forward + connexion résiduelle
# =====================================================================
class N05_FeedForward(Base):
    number, chapter = "05", "Feed-forward et résiduel"

    def construct(self):
        self.chapter_card("Un petit réseau appliqué à chaque token")
        rng = np.random.default_rng(8)
        inp = column(8, -4.6, BLUE_C, r=0.16, buff=0.12).shift(DOWN * 0.1)
        hid = column(16, -0.4, ORANGE, r=0.1, buff=0.08).shift(DOWN * 0.1)
        outl = column(8, 3.8, BLUE_C, r=0.16, buff=0.12).shift(DOWN * 0.1)
        e1 = fc(inp, hid, SOFT, 0.7, 0.18)
        e2 = fc(hid, outl, SOFT, 0.7, 0.18)
        labs = VGroup(T("128", 22, BLUE_C).next_to(inp, UP, buff=0.2),
                      T("512 neurones · GELU", 22, ORANGE).next_to(hid, UP, buff=0.2),
                      T("128", 22, BLUE_C).next_to(outl, UP, buff=0.2))
        note = T("8, 16 et 8 neurones dessinés", 18, SOFT).to_corner(UR, buff=0.35)
        self.play(FadeIn(inp), FadeIn(hid), FadeIn(outl), FadeIn(labs), FadeIn(note))
        self.play(Create(e1), Create(e2), run_time=2)
        self.say("Le vecteur d'un token s'élargit à 512 neurones, puis revient à 128.", 2.6)

        hv = rng.uniform(0, 1, 16)
        hv[rng.choice(16, 6, replace=False)] = 0.03       # GELU : certains neurones quasi éteints
        self.play(*light(inp, 1), run_time=0.6)
        self.play(flow(e1, ORANGE, 1.2), *light(hid, vals=hv, color=ORANGE), run_time=1.2)
        self.say("GELU : certains neurones s'allument fort, d'autres restent presque éteints.", 2.6)
        self.play(flow(e2, ORANGE, 1.2), *light(outl, 2), run_time=1.2)

        # Connexion résiduelle
        plus = VGroup(Circle(0.28, color=GREEN_C, stroke_width=3), T("+", 30, GREEN_C)).move_to(RIGHT * 5.5 + DOWN * 0.1)
        to_plus = VGroup(*[Line(n.get_right(), plus.get_left(), stroke_color=SOFT, stroke_width=1, stroke_opacity=0.4)
                           for n in outl])
        y_low = hid.get_bottom()[1] - 0.35       # le chemin passe SOUS le réseau
        res_path = VMobject(stroke_color=GREEN_C, stroke_width=4).set_points_as_corners([
            inp.get_bottom() + DOWN * 0.1, [inp.get_x(), y_low, 0], [plus.get_x(), y_low, 0],
            plus.get_bottom()])
        rl = T("connexion résiduelle : on rajoute l'entrée", 20, GREEN_C).next_to(res_path, DOWN, buff=0.12)
        self.play(FadeIn(plus), Create(to_plus))
        self.play(Create(res_path), FadeIn(rl), run_time=1.2)
        self.play(ShowPassingFlash(res_path.copy().set_stroke(GREEN_C, 8), time_width=0.4, run_time=1.4),
                  flow(to_plus, ORANGE, 1.4), plus[0].animate.set_fill(GREEN_C, opacity=0.5), run_time=1.4)
        self.say("La sortie s'ajoute à l'entrée : l'information d'origine n'est jamais perdue.", 3)
        self.say("LayerNorm avant l'attention et avant le feed-forward garde les valeurs stables.", 2.8)
        self.clear()

        # 4 blocs empilés
        blocks = VGroup()
        for i in range(4):
            a = column(5, 0, BLUE_C, r=0.08, buff=0.06)
            b = column(5, 0.6, BLUE_C, r=0.08, buff=0.06)
            c = column(5, 1.2, BLUE_C, r=0.08, buff=0.06)
            blk = VGroup(fc(a, b, SOFT, 0.6, 0.3), VGroup(*[link(x, y, SOFT, 1, 0.5) for x, y in zip(b, c)]), a, b, c)
            frame = SurroundingRectangle(blk, color=ORANGE, buff=0.15, corner_radius=0.1)
            lab = T(f"bloc {i + 1}", 20, ORANGE).next_to(frame, UP, buff=0.1)
            blocks.add(VGroup(frame, blk, lab))
        blocks.arrange(RIGHT, buff=0.7).move_to(UP * 0.3)
        arrows = VGroup(*[Arrow(blocks[i][0].get_right(), blocks[i + 1][0].get_left(), buff=0.05, color=SOFT)
                          for i in range(3)])
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in blocks], lag_ratio=0.2), Create(arrows))
        for i, b in enumerate(blocks):
            self.play(flow(b[1][0], ORANGE, 0.6), *light(b[1][3], i), run_time=0.6)
            self.play(flow(b[1][1], ORANGE, 0.4), *light(b[1][4], i + 10), run_time=0.4)
        info = T("198 272 paramètres par bloc · 793 088 pour les 4", 24, TXT).next_to(blocks, DOWN, buff=0.6)
        self.play(FadeIn(info))
        self.say("Ce bloc (attention + feed-forward) est empilé 4 fois : chaque passage affine le sens.", 3)
        self.clear(keep_head=False)


# =====================================================================
# 06 · Tête de classification
# =====================================================================
class N06_Tete(Base):
    number, chapter = "06", "Tête de classification"

    def construct(self):
        self.chapter_card("Des vecteurs de tokens à une seule probabilité")
        rng = np.random.default_rng(12)
        toks = VGroup(*[VGroup(*[neuron(0.09, BLUE_C, rng.uniform(0.2, 0.9)) for _ in range(6)]).arrange(DOWN, buff=0.05)
                        for _ in range(5)]).arrange(RIGHT, buff=0.18).move_to(LEFT * 5.3 + UP * 0.9)
        tl = T("5 tokens × 128", 18, SOFT).next_to(toks, UP, buff=0.15)
        pooled = VGroup(*[neuron(0.11, GREEN_C, 0.08) for _ in range(6)]).arrange(DOWN, buff=0.06).move_to(LEFT * 3.3 + UP * 0.9)
        pe = VGroup(*[link(t[i], pooled[i], GREEN_C, 0.8, 0.35) for t in toks for i in range(6)])
        pl = T("moyenne", 18, GREEN_C).next_to(pooled, UP, buff=0.15)
        self.play(FadeIn(toks), FadeIn(tl))
        self.play(Create(pe), FadeIn(pooled), FadeIn(pl), run_time=1.2)
        self.play(flow(pe, GREEN_C, 1.2), *light(pooled, 5, GREEN_C, 0.4, 0.9), run_time=1.2)
        self.say("Mean pooling : chaque valeur est la moyenne sur tous les vrais tokens.", 2.6)

        feats = VGroup(*[neuron(0.11, ORANGE, v) for v in (0.6, 1.0, 0.9, 0.9)]).arrange(DOWN, buff=0.06).move_to(LEFT * 3.3 + DOWN * 1.4)
        fn = VGroup(*[T(s, 16, ORANGE).next_to(n, LEFT, buff=0.12) for s, n in
                      zip(["liens", "IP", "urgence", "identifiants"], feats)])
        fl = T("12 features (4 dessinées)", 18, ORANGE).next_to(feats, DOWN, buff=0.15)
        self.play(FadeIn(feats), FadeIn(fn), FadeIn(fl))
        self.say("À côté : 12 indices calculés à la main, ici pour l'email de test.", 2.4)

        concat = VGroup(*pooled, *feats)
        dense = column(10, 0.2, BLUE_C, r=0.13, buff=0.1).shift(DOWN * 0.1)
        dl = T("64 neurones ReLU\n(10 dessinés)", 18, BLUE_C).next_to(dense, UP, buff=0.15)
        de = fc(concat, dense, SOFT, 0.7, 0.2)
        self.play(Create(de), FadeIn(dense), FadeIn(dl), run_time=1.4)
        dv = rng.uniform(0.3, 1, 10)
        dv[[1, 4, 7]] = 0.02
        self.play(flow(de, ORANGE, 1.3), *light(dense, vals=dv), run_time=1.3)
        offs = VGroup(*[Cross(dense[i], stroke_color=RED_C, stroke_width=2, scale_factor=0.7) for i in (1, 4, 7)])
        self.play(Create(offs))
        self.say("Chaque neurone combine les 140 entrées ; ReLU éteint ceux dont la somme est négative.", 3)

        out = neuron(0.32, ORANGE).move_to(RIGHT * 2.3 + DOWN * 0.1)
        oe = fc(dense, [out], SOFT, 1, 0.35)
        ol = T("1 neurone\nlogit z", 18, ORANGE).next_to(out, DOWN, buff=0.15)
        self.play(Create(oe), FadeIn(out), FadeIn(ol))
        self.play(flow(oe, ORANGE, 1.0), out.animate.set_fill(ORANGE, opacity=0.85), run_time=1.0)

        ax = Axes(x_range=[-6, 6, 2], y_range=[0, 1, 0.5], x_length=3.2, y_length=2.2,
                  axis_config={"color": SOFT, "include_tip": False}).move_to(RIGHT * 5.2 + DOWN * 0.1)
        curve = ax.plot(lambda x: 1 / (1 + np.exp(-x)), color=ORANGE, stroke_width=4)
        thr = DashedLine(ax.c2p(-6, 0.5), ax.c2p(6, 0.5), color=SOFT)
        sl = T("sigmoïde", 20, ORANGE).next_to(ax, UP, buff=0.15)
        a = Arrow(out.get_right(), ax.get_left(), buff=0.1, color=SOFT, stroke_width=3)
        self.play(GrowArrow(a), Create(ax), Create(curve), FadeIn(sl), Create(thr))
        z = ValueTracker(-3)
        dot = always_redraw(lambda: Dot(ax.c2p(z.get_value(), 1 / (1 + np.exp(-z.get_value()))), color=TXT))

        def verdict():
            p = 1 / (1 + np.exp(-z.get_value()))
            s, c = ("PHISHING", ORANGE) if p >= 0.5 else ("légitime", BLUE_C)
            return T(f"P = {p:.2f} → {s}".replace(".", ","), 24, c, weight=BOLD).next_to(ax, DOWN, buff=0.3)
        v = always_redraw(verdict)
        self.play(FadeIn(dot), FadeIn(v))
        self.play(z.animate.set_value(3), run_time=2.5)
        self.say("La sigmoïde transforme z en probabilité ; au-dessus du seuil, l'email est signalé.", 3)
        self.clear(keep_head=False)


# =====================================================================
# 07 · Apprentissage
# =====================================================================
class N07_Apprentissage(Base):
    number, chapter = "07", "Apprentissage"

    def construct(self):
        self.chapter_card("Le signal avance, l'erreur recule, les poids changent")
        mc = Macro()
        self.add(mc.toks, mc.labels, mc.flab)
        self.play(FadeIn(mc.net), run_time=1.2)
        truth = T("vraie réponse : phishing (y = 1)", 22, TXT).move_to([4.6, 3.0, 0])
        self.play(FadeIn(truth))
        ill = T("valeurs illustratives", 18, SOFT).to_corner(UR, buff=0.35)
        self.play(FadeIn(ill))

        ps = [0.40, 0.71, 0.93]
        for k, p in enumerate(ps):
            mc.forward(self, ORANGE, 0.35, seed=20 + k, out_opacity=p)
            pt = T(f"prédit : p = {fr(p)}", 22, ORANGE).move_to([5.2, -0.9, 0])
            loss = -np.log(p)
            lt = T(f"erreur = −log(p) = {fr(round(loss, 2))}", 22, RED_C if loss > 0.3 else GREEN_C).move_to([5.2, -1.4, 0])
            self.play(FadeIn(pt), FadeIn(lt))
            if k == 0:
                self.say("Passage avant : le modèle prédit 0,40 alors que c'est un phishing. Erreur : 0,92.", 2.6)
                self.say("Rétropropagation : l'erreur repart de la sortie vers l'entrée…", 1.4, GREEN_C)
            mc.backward(self, 0.3 if k else 0.45)
            if k == 0:
                self.say("…et chaque poids est ajusté par AdamW pour réduire l'erreur.", 1.4, GREEN_C)
            mc.update_weights(self, seed=k)
            self.play(*mc.reset(), FadeOut(pt), FadeOut(lt), run_time=0.4)
        fin = VGroup(T("p : 0,40 → 0,71 → 0,93", 22, ORANGE), T("erreur : 0,92 → 0,34 → 0,07", 22, GREEN_C)
                     ).arrange(DOWN, buff=0.2).move_to([4.9, -1.3, 0])
        self.play(FadeIn(fin))
        self.say("Répété sur 438 lots de 32 emails par epoch : le réseau s'améliore pas à pas.", 3)
        self.clear(keep_head=False)


# =====================================================================
# 08 · Démonstration complète
# =====================================================================
class N08_Demo(Base):
    number, chapter = "08", "Le flux complet"

    def construct(self):
        self.chapter_card("Deux emails traversent le réseau")
        ill = T("démonstration du flux · sortie illustrative", 18, SOFT).to_corner(UR, buff=0.35)
        for words, mail, phishing in [
            (WORDS, "URGENT: verify your password at http://…/login", True),
            (["hi", "team", "meeting", "thursday", "agenda"], "Hi team, the meeting is moved to Thursday, see agenda", False),
        ]:
            mc = Macro(words)
            m = M(mail, 22, TXT).to_edge(UP, buff=0.75)
            self.play(FadeIn(m), FadeIn(ill), FadeIn(mc.toks), FadeIn(mc.labels), FadeIn(mc.net), FadeIn(mc.flab))
            mc.forward(self, ORANGE if phishing else BLUE_C, 0.5, seed=40 if phishing else 60,
                       out_opacity=0.9 if phishing else 0.1)
            res = T("PHISHING" if phishing else "légitime", 34, ORANGE if phishing else BLUE_C, weight=BOLD)
            res.next_to(mc.out, RIGHT, buff=0.3)
            self.play(Write(res), Flash(mc.out, color=ORANGE if phishing else BLUE_C, flash_radius=0.5))
            self.wait(1.5)
            self.play(*[FadeOut(x) for x in (m, mc.toks, mc.labels, mc.net, mc.flab, res)])

        recap = VGroup(*[T(s, 28, c) for s, c in [
            ("Embedding : 8 000 × 128", BLUE_C), ("Attention : 4 têtes × 32", ORANGE),
            ("Feed-forward : 128 → 512 → 128", ORANGE), ("4 blocs encodeurs", ORANGE),
            ("Pooling 128 + 12 features → 64 → 1", GREEN_C), ("1 859 201 paramètres", TXT)]]
        ).arrange(DOWN, buff=0.28)
        self.play(FadeOut(ill), LaggedStart(*[FadeIn(r, shift=UP * 0.2) for r in recap], lag_ratio=0.2))
        self.wait(3)
        self.play(FadeOut(recap), FadeOut(self.head))
        end = T("Merci", 72, TXT, weight=BOLD)
        self.play(Write(end))
        self.wait(2)
        self.play(FadeOut(end))


SCENES = ["N01_Neurone", "N02_Reseau", "N03_Embedding", "N04_Attention", "N05_FeedForward",
          "N06_Tete", "N07_Apprentissage", "N08_Demo"]
