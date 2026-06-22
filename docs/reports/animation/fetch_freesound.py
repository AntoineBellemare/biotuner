"""
Fetch the curated Freesound percussion kit used by the biorhythm reels.

Downloads each sound's HQ-ogg preview (token auth is read-only — previews only),
resamples to 44.1 kHz mono WAV under public/audio/samples/ (gitignored), and
writes SAMPLE_CREDITS.md. All picks are CC0, but we credit them anyway.

Requires a Freesound API token in the environment:
    FREESOUND_TOKEN=...   (get one at https://freesound.org/apiv2/apply)

Run: FREESOUND_TOKEN=xxx python fetch_freesound.py
"""
from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "audio" / "samples"
SR = 44_100
TOKEN = os.environ.get("FREESOUND_TOKEN", "")

# role → Freesound sound id (all Creative Commons 0)
KIT = {
    "bed": 411593,        # Mysterious Ambient Pad Loop 84 bpm Dm
    "low": 555527,        # deep_tom
    "kalimba": 536549,    # Kalimba C3
    "shaker": 199823,     # Egg Shaker - 1 Throw
    "bowl": 271370,       # singing bowl strike
    "heartbeat": 22440,   # Single Heartbeat Clean HQ (for the Heart×Brain duet)
}


def api(sound_id: int) -> dict:
    url = f"https://freesound.org/apiv2/sounds/{sound_id}/?" + urllib.parse.urlencode(
        {"fields": "id,name,username,license,url,previews"})
    req = urllib.request.Request(url, headers={"Authorization": "Token " + TOKEN})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def fetch_bytes(url: str) -> bytes:
    try:
        return urllib.request.urlopen(url, timeout=40).read()
    except urllib.error.HTTPError:
        req = urllib.request.Request(url, headers={"Authorization": "Token " + TOKEN})
        return urllib.request.urlopen(req, timeout=40).read()


def main() -> None:
    if not TOKEN:
        raise SystemExit("Set FREESOUND_TOKEN in the environment first.")
    OUT.mkdir(parents=True, exist_ok=True)
    creds = []
    for role, sid in KIT.items():
        meta = api(sid)
        prev = meta["previews"].get("preview-hq-ogg") or meta["previews"]["preview-hq-mp3"]
        x, sr = sf.read(io.BytesIO(fetch_bytes(prev)))
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            x = resample_poly(x, SR, sr)
        x = x / (np.max(np.abs(x)) or 1.0)
        sf.write(OUT / f"{role}.wav", x.astype("float32"), SR)
        creds.append((role, meta["id"], meta["name"], meta["username"], meta["license"], meta["url"]))
        print(f"  {role:8s} #{sid:<8d} {len(x)/SR:5.2f}s  {meta['name'][:40]}  [{meta['license'].split('/')[-2] if '/' in meta['license'] else meta['license']}]")

    lines = ["# Sample credits\n",
             "Percussion kit for the biorhythm reels, fetched from Freesound (all CC0).",
             "Re-fetch with `FREESOUND_TOKEN=… python fetch_freesound.py`.\n",
             "| role | id | name | author | license |",
             "|------|----|------|--------|---------|"]
    for role, sid, name, user, lic, url in creds:
        lines.append(f"| {role} | [{sid}]({url}) | {name} | {user} | {lic} |")
    (HERE / "SAMPLE_CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}  + SAMPLE_CREDITS.md")


if __name__ == "__main__":
    main()
