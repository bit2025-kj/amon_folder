"""Tous les hyperparamètres du projet, au même endroit."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # --- Chemins ---
    raw_csv: Path = ROOT / "data" / "raw" / "Phishing_Email.csv"
    processed_dir: Path = ROOT / "data" / "processed"
    tokenizer_path: Path = ROOT / "artifacts" / "tokenizer.json"
    run_dir: Path = ROOT / "artifacts"          # checkpoints + livrable

    # --- Données ---
    max_raw_chars: int = 20_000      # un email du dataset fait 17 Mo : on coupe
    val_frac: float = 0.10
    test_frac: float = 0.10
    seed: int = 42
    limit: int | None = None         # nb max d'emails par split (mode debug)

    # --- Augmentation (voir augment.py) ---
    synth_train: tuple = (2400, 1600)   # emails transactionnels synthétiques (légitimes, phishing)
    synth_val: tuple = (300, 200)
    synth_test: tuple = (500, 400)      # marques et formulations jamais vues -> test_synth.csv
    noise_frac: float = 0.30            # part des emails du train remplacés par une version bruitée

    # --- Tokenizer ---
    vocab_size: int = 8_000
    max_len: int = 256               # tokens par email (objet + début du corps)

    # --- Modèle ---
    d_model: int = 128               # taille des embeddings
    n_heads: int = 4                 # 4 têtes x 32 dimensions
    n_layers: int = 4                # blocs Transformer encodeur
    d_ff: int = 512                  # neurones cachés du feed-forward
    n_features: int = 12             # features structurelles manuelles
    head_hidden: int = 64            # neurones de la tête de classification
    dropout: float = 0.1
    head_dropout: float = 0.3

    # --- Entraînement ---
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    patience: int = 2                # early stopping (sur la PR-AUC de validation)
    grad_clip: float = 1.0
    save_every: int = 200            # sauvegarde de reprise toutes les N étapes

    # --- Décision ---
    target_recall: float = 0.97      # seuil choisi pour rater <= 3 % des phishing

    # Fichiers produits
    @property
    def last_ckpt(self) -> Path:      # pour reprendre l'entraînement
        return self.run_dir / "checkpoint_last.pt"

    @property
    def best_weights(self) -> Path:   # meilleurs poids vus sur la validation
        return self.run_dir / "best_weights.pt"

    @property
    def model_path(self) -> Path:     # LE LIVRABLE : tout ce qu'il faut pour prédire
        return self.run_dir / "phishing_model.pt"

    def apply_debug(self):
        """Mini-run pour vérifier que tout le pipeline marche en ~1 minute."""
        self.run_dir = ROOT / "artifacts" / "debug"   # n'écrase pas le vrai modèle
        self.limit = 256
        self.max_epochs = 2
        self.n_layers = 2
        self.save_every = 4

    def model_kwargs(self) -> dict:
        keys = ["vocab_size", "max_len", "d_model", "n_heads", "n_layers", "d_ff",
                "n_features", "head_hidden", "dropout", "head_dropout"]
        return {k: getattr(self, k) for k in keys}
