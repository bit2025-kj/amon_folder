"""Rend toutes les scènes de la vidéo « réseau de neurones » puis les assemble.

    python render_all.py            # 1080p 30 i/s  -> reseau_neurones_1080p.mp4
    python render_all.py --preview  # 480p rapide   -> reseau_neurones_preview.mp4
    python render_all.py N04_Attention N05_FeedForward   # seulement certaines scènes
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reseau_neurones import SCENES  # noqa: E402

MODULE = "reseau_neurones"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    preview = "--preview" in sys.argv
    # -ql = 480p 15 i/s (aperçu) ; -qh + --fps 30 = 1080p 30 i/s (final)
    quality, fps, folder, suffix = (("-ql", [], "480p15", "preview") if preview
                                    else ("-qh", ["--fps", "30"], "1080p30", "1080p"))
    scenes = args or SCENES

    for s in scenes:
        print(f"=== Rendu {s} ===", flush=True)
        subprocess.run(["manim", quality, *fps, "--disable_caching", f"{MODULE}.py", s], cwd=HERE, check=True)

    if args:   # rendu partiel : pas d'assemblage
        return
    out_dir = HERE / "media" / "videos" / MODULE / folder
    listing = HERE / "concat.txt"
    listing.write_text("".join(f"file '{(out_dir / (s + '.mp4')).as_posix()}'\n" for s in SCENES), encoding="utf-8")
    final = HERE / f"{MODULE}_{suffix}.mp4"
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(final)],
                   check=True)
    print(f"\nVidéo finale : {final}")


if __name__ == "__main__":
    main()
