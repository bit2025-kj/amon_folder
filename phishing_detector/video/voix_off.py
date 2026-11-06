"""Génère la voix off (voix neuronale Microsoft via edge-tts) et l'assemble avec la vidéo.

    python voix_off.py                                  # voix par défaut (Rémy)
    python voix_off.py --voice fr-FR-VivienneMultilingualNeural
    python voix_off.py --rate -5%                       # parler un peu plus lentement

Entrées  : voix_off.md (texte) + media/videos/reseau_neurones/1080p30/*.mp4 (scènes rendues)
Sorties  : audio/<scène>.mp3, scenes_voix/<scène>.mp4, reseau_neurones_voix_1080p.mp4
Si la narration est plus longue que la scène, la dernière image est prolongée (la voix n'est jamais coupée).
"""
import argparse
import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path

# Variable d'environnement cassée sur cette machine (pointe vers un certificat PostgreSQL absent)
for var in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
    os.environ.pop(var, None)

import edge_tts  # noqa: E402

HERE = Path(__file__).resolve().parent
SCENES = ["N01_Neurone", "N02_Reseau", "N03_Embedding", "N04_Attention", "N05_FeedForward",
          "N06_Tete", "N07_Apprentissage", "N08_Demo"]
VIDEO_DIR = HERE / "media" / "videos" / "reseau_neurones" / "1080p30"
LEAD = 0.6    # secondes de silence avant que la voix commence
TAIL = 0.8    # secondes de marge après la fin de la voix
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def read_script(md_path: Path) -> dict[str, str]:
    """Découpe voix_off.md : chaque section '## 0N · …' -> texte de la scène N."""
    texts = {}
    for block in re.split(r"^## ", md_path.read_text(encoding="utf-8"), flags=re.M)[1:]:
        m = re.match(r"(\d\d) ·", block)
        if not m:
            continue                      # section « Consigne de voix »
        body = block.split("\n", 1)[1].split("\n---")[0]
        paras = [p.strip() for p in body.strip().split("\n\n") if p.strip()]
        texts[SCENES[int(m.group(1)) - 1]] = "\n\n".join(paras)
    return texts


def duration(path: Path) -> float:
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


async def synthesize(text: str, out: Path, voice: str, rate: str):
    await edge_tts.Communicate(text, voice, rate=rate).save(str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="fr-FR-RemyMultilingualNeural")
    ap.add_argument("--rate", default="+0%", help="vitesse, ex. -5%% ou +5%%")
    args = ap.parse_args()

    texts = read_script(HERE / "voix_off.md")
    audio_dir, scene_dir = HERE / "audio", HERE / "scenes_voix"
    audio_dir.mkdir(exist_ok=True)
    scene_dir.mkdir(exist_ok=True)

    for s in SCENES:
        video = VIDEO_DIR / f"{s}.mp4"
        if not video.exists():
            raise SystemExit(f"{video} introuvable : lance d'abord  python render_all.py")
        audio = audio_dir / f"{s}.mp3"
        print(f"[voix] {s} ...", flush=True)
        asyncio.run(synthesize(texts[s], audio, args.voice, args.rate))

        dv, da = duration(video), duration(audio)
        target = max(dv, LEAD + da + TAIL)
        pad = max(target - dv, 0)
        print(f"       vidéo {dv:5.1f}s | voix {da:5.1f}s | final {target:5.1f}s"
              + (f"  (+{pad:.1f}s d'image figée)" if pad > 0.05 else ""), flush=True)
        ms = int(LEAD * 1000)
        subprocess.run([
            FFMPEG, "-v", "error", "-y", "-i", str(video), "-i", str(audio),
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.3f}[v];[1:a]adelay={ms}|{ms},apad[a]",
            "-map", "[v]", "-map", "[a]", "-t", f"{target:.3f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            str(scene_dir / f"{s}.mp4")], check=True)

    listing = HERE / "concat_voix.txt"
    listing.write_text("".join(f"file '{(scene_dir / (s + '.mp4')).as_posix()}'\n" for s in SCENES),
                       encoding="utf-8")
    final = HERE / "reseau_neurones_voix_1080p.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
    print(f"\nVidéo avec voix off : {final}  ({duration(final):.0f} s)")


if __name__ == "__main__":
    main()
