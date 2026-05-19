# biotuner.harmonic_geometry — animation

A Remotion (React + TypeScript) animation showcasing Phase 1 – 3 outputs of the
`harmonic_geometry` module: Lissajous curves, harmonograph traces, star polygons,
the tuning circle, and Chladni nodal fields.

## Architecture

```
animation/
├── export_geometry_data.py       # Python: builds public/geometry.json from harmonic_geometry
├── public/
│   └── geometry.json             # ~1 MB; pre-computed coordinates / fields per scene
├── src/
│   ├── index.ts                  # Remotion entry point
│   ├── Root.tsx                  # registers <Composition>
│   ├── Main.tsx                  # timeline assembly via <Series>
│   ├── theme.ts                  # color palette + fonts
│   ├── geometry.ts               # typed accessor for public/geometry.json
│   ├── components/
│   │   ├── Backdrop.tsx          # animated radial-glow background
│   │   ├── Stage.tsx             # centered SVG using normalized [-1, 1] coords
│   │   └── Caption.tsx           # lower-thirds title + monospace subtitle
│   └── scenes/
│       ├── Title.tsx             # opening — drifting unit-circle Lissajous + title
│       ├── LissajousMorph.tsx    # crossfades through six 2-D Lissajous ratios
│       ├── Harmonograph.tsx      # progressive reveal of a damped harmonograph
│       ├── StarPolygons.tsx      # rotating Schläfli {n/k} family
│       ├── TuningCircle.tsx      # JI diatonic landing on the octave-equave
│       ├── ChladniMorph.tsx      # canvas-rendered rectangular plate fields
│       └── Outro.tsx             # fade-out title slate
└── out/
    └── harmonic-geometry.mp4     # rendered video
```

The Python side never touches the JS side — it dumps coordinates / fields once.
The React side reads that JSON and animates it. To regenerate after changing
the source data, rerun the Python step before re-rendering.

## Re-render

```bash
# 1. (optional) regenerate the data with the biotuner conda env
PYTHONPATH=../../.. python export_geometry_data.py

# 2. install once
npm install

# 3. render to MP4
npm run build           # writes out/harmonic-geometry.mp4
# or for a smaller WebM:
npm run build-webm

# Live editing
npm run dev             # opens the Remotion Studio at localhost:3000
```

## Specs

- 1920 × 1080, 30 fps
- ~40 s total (1192 frames)
- Seven scenes via `<Series.Sequence>`; durations live in `Main.tsx`
- Color palette: dark background (`#0a0e1a`), warm gold accent (`#e8d68a`),
  cool blue secondary (`#6da3d8`)

## Notes

- Chladni fields are pre-downsampled to 64×64 in `export_geometry_data.py` to
  keep `geometry.json` small (~1 MB). The canvas component then upscales them
  with bilinear smoothing.
- All curves use SVG strokes with a soft drop-shadow for the gold trace plus a
  wider semi-transparent cool stroke underneath, giving each line a subtle
  glow without hand-rolled shaders.
- `delayRender` / `continueRender` is used in `ChladniMorph.tsx` so Remotion
  waits for the canvas paint to finish before snapshotting each frame.
