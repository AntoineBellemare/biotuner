# biotuner.harmonic_geometry — animations

Remotion (React + TypeScript) animations that turn `harmonic_geometry`
outputs into shareable video. There are two layers:

1. **GeometryV2 / GeometryV2-IG** — the flagship 94.5-second showcase
   (landscape + Instagram-portrait) with a chord-synced soundtrack. This is
   "Reel 01".
2. **Reels** — a lightweight per-reel pipeline for producing many short
   Instagram posts without writing new React each time. See
   [Reels](#reels) below.

The Python side computes geometry and writes JSON; the React side reads that
JSON and animates it; a Python audio renderer writes a synced `.wav`. The
three never import each other at runtime.

## Layout

```
animation/
├── export_geometry_data.py     # Python → public/geometry.json (flagship scenes)
├── render_audio.py             # Python → public/audio/score.wav (flagship soundtrack)
├── export_reels.py             # Python → public/reels/<name>.json + reel soundtracks
├── public/
│   ├── geometry.json           # flagship scene data (committed, ~6 MB)
│   ├── reels/                  # per-reel data (committed)
│   └── audio/                  # rendered .wav soundtracks (gitignored — regenerate)
├── out/                        # rendered .mp4 (gitignored — regenerate)
├── src/
│   ├── index.ts                # Remotion entry point
│   ├── Root.tsx                # registers every <Composition> (flagship + reels)
│   ├── MainV2.tsx              # flagship timeline (GeometryV2 / GeometryV2-IG)
│   ├── theme.ts                # palette + fonts + per-chord hues
│   ├── geometry.ts             # typed accessor for public/geometry.json
│   ├── projection.ts           # 3-D → 2-D projection helpers
│   ├── reels/                  # reel specs + the generic <Reel> composition
│   │   ├── reelData.ts         # typed accessor for public/reels/*.json
│   │   ├── Reel.tsx            # generic reel renderer (scene list + audio)
│   │   └── specs.ts            # registry: which scenes + audio each reel uses
│   ├── components/             # Backdrop, Stage, Caption, PedagogyCardIG, …
│   └── scenes/                 # individual scene components (reused across reels)
└── package.json
```

## Compositions

`Root.tsx` registers each renderable composition by id:

| id | size | duration | what |
|---|---|---|---|
| `GeometryV2`    | 1920×1080 | 94.5 s | flagship landscape showcase |
| `GeometryV2-IG` | 1080×1920 | 94.5 s | **Reel 01** — flagship Instagram portrait |
| `Reel02-Cymatics`  | 1080×1920 | 13.0 s | chord cymatics — "what a chord looks like" |
| `Reel03-Intervals` | 1080×1920 | 20.6 s | consonance vs dissonance — interval journey |
| `Reel04-HeyJude`   | 1080×1920 | 19.0 s | famous-song chords → geometry (Hey Jude) |
| `Reel05-LetItBe`   | 1080×1920 | 19.0 s | song variant (Let It Be) |
| `Reel06-Canon`     | 1080×1920 | 19.0 s | song variant (Canon in D) |
| `Reel07-BrainHeart`| 1080×1920 | 13.4 s | EEG vs ECG — inharmonic brain vs harmonic heart |
| `Reel08-ManyShapes`| 1080×1920 | 15.8 s | one chord through cymatics / Lissajous / harmonograph / interference |
| `Reel09-CanonHarmonograph` | 1080×1920 | 19.0 s | Canon in D drawn as a harmonograph (same music, new geometry) |

List them at any time with `npx remotion compositions src/index.ts`.

### Geometry types

The cymatics field is one of several live in-canvas generators
(`src/reels/geometries.ts`), all driven by a chord's ratios:

- **cymatics** — D4-symmetric Chladni nodal density (default scene)
- **lissajous** — rotating 3-D Lissajous knot (projected), x/y/z = first 3 ratios
- **harmonograph** — ratio-driven flower/rosette, gently drifting (one distinct
  figure per chord)
- **interference** — vortex-spiral interference; spiral-arm count/density = the
  chord's signature, visually distinct from the lattice/curve geometries

Each geometry has a deliberately different visual language so the four read
as four views of the same chord: lattice / 3-D knot / flower / spiral.

A reel's `scene: "multi"` + `geometries: [...]` (per segment) renders the
`MultiGeometryMorph` scene, showing the same music through different
geometries.

### Biosignal reels

`biosignal_chords.py` turns biotuner-extracted spectral peaks (from the
bundled `assets/biosignals/` example EEG/ECG) into cymatics chords — the
basis of the Brain vs Heart reel.

### Song reels

Song reels synthesise only the **chord progression** (harmony) of a famous
song — no copyrighted recording is used. `song_chords.py` maps each chord
(scale-degree + quality, in a key) to:

- small-integer cymatics wavenumbers, scaled globally across the song so
  each chord's pattern is distinct yet legible (harmonic motion shows), and
- octave-voiced audio frequencies with a bass root, so the pad progression
  sounds like the song's chords.

Add a song in `song_chords.py` (`SONGS` dict) and one `_song_reel(...)` line
in `export_reels.py`. The reel is branded with the song title so the
progression is recognisable; overlay the real track in-app if desired.

## Sound

`render_audio.py` builds the flagship soundtrack. It pulls the chord events
straight out of `public/geometry.json` so the audio is locked to the visual
timeline (no manual sync). Each chord is voiced by a 3-voice detuned-unison ×
8-partial additive stack with shimmer + vibrato, portamento glides between
chords, and a resonant band-pass sweep over the Chladni section.
`MainV2.tsx` plays it via `<Audio src={staticFile("audio/score.wav")} />`.

Reel soundtracks are produced the same way by `export_reels.py` (each reel
declares its own chord sequence and synth/percussion config).

> **Outputs are not committed.** `public/audio/*.wav` and `out/*.mp4` are
> gitignored — they are large and fully reproducible. The committed JSON +
> the two Python scripts + npm are everything needed to rebuild them.

## Build the flagship (Reel 01)

```bash
cd docs/reports/animation

# 1. (optional) regenerate scene data — needs the biotuner env
python export_geometry_data.py            # → public/geometry.json

# 2. render the synced soundtrack — pure Python (numpy + scipy)
python render_audio.py                    # → public/audio/score.wav

# 3. install JS deps once
npm install

# 4. render
npm run render:ig                         # → out/GeometryV2-IG.mp4  (portrait)
npm run render:geometry                   # → out/GeometryV2.mp4     (landscape)

# Live editing
npm run dev                               # Remotion Studio at localhost:3000
```

## Reels

The reel pipeline turns "a new Instagram post" into config rather than code.
Each reel is:

1. a **data builder** in `export_reels.py` that writes `public/reels/<name>.json`
   and (optionally) a `public/audio/<name>.wav` soundtrack;
2. a **spec** in `src/reels/specs.ts` — the ordered scenes, captions, audio
   file, fps and duration;
3. an automatic **`<Composition>`** registered from that spec in `Root.tsx`.

Most reels reuse existing scene components, so adding one is ~1 hour of config.

```bash
# build all reel data + soundtracks
python export_reels.py                    # → public/reels/*.json, public/audio/*.wav

# render a specific reel (id from src/reels/specs.ts)
npx remotion render src/index.ts Reel02-Cymatics out/Reel02-Cymatics.mp4
```

To add a new reel: add a builder to `export_reels.py`, add a spec entry to
`src/reels/specs.ts`, run `python export_reels.py`, then render. No new React
unless the reel needs a genuinely new visual.

## Specs

- Flagship: 1920×1080 and 1080×1920, 30 fps, 94.5 s (2835 frames)
- Reels: 1080×1920 portrait, 30 fps, length per spec
- Palette: dark background (`#0a0e1a`), warm gold accent, cool blue secondary;
  per-chord hues in `theme.ts`
- Chladni / cymatics fields are downsampled in the Python exporters to keep
  the JSON small; the canvas scenes upscale with bilinear smoothing.
- `delayRender` / `continueRender` gate every canvas-painting scene so Remotion
  waits for the paint before snapshotting each frame.
