import React, { useEffect, useRef } from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  delayRender,
  continueRender,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { theme, fonts } from "../theme";
import { cymaticsDensity, lerpRatios, type Chord } from "../reels/cymatics";
import { morphedStops, sampleStops } from "../reels/palettes";
import type { ReelData } from "../reels/reelData";

const FIELD_N = 300;

/**
 * Meditative reel: a single large cymatics plate that morphs SMOOTHLY and
 * continuously through a long sequence of real EEG-derived chords, while the
 * colour palette itself slowly cross-fades through a cycle. No chord labels —
 * just a slow, breathing, hypnotic field. ~1 minute.
 */
export const MeditativeMorph: React.FC<{ data: ReelData }> = ({ data }) => {
  const frame = useCurrentFrame();
  const { width, fps, durationInFrames } = useVideoConfig();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const chords: Chord[] = data.chords;
  const n = chords.length;
  const seg = data.frames_per_segment;
  const rawSeg = Math.floor(frame / seg);
  const segIdx = rawSeg % n;
  const local = (frame % seg) / seg;
  const fromChord = chords[segIdx];
  const toChord = chords[(segIdx + 1) % n];
  // Continuous cosine-eased morph (no holds) for a flowing field.
  const ratios = lerpRatios(fromChord.ratios, toChord.ratios, local);

  // Palette slowly cross-fades — exactly one full cycle over the reel
  // (≈10 s per palette), looping seamlessly.
  const palettePhase = frame / durationInFrames;
  // Gentle gamma breathing.
  const gamma = 0.82 + 0.16 * Math.sin((frame / fps) * 0.5);

  useEffect(() => {
    const handle = delayRender("meditative");
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const stops = morphedStops(palettePhase);
        const dens = cymaticsDensity(ratios, FIELD_N, { symmetry: "d4_max" });
        const img = ctx.createImageData(FIELD_N, FIELD_N);
        for (let k = 0; k < dens.length; k++) {
          const [r, g, b] = sampleStops(stops, dens[k], gamma);
          img.data[k * 4 + 0] = r;
          img.data[k * 4 + 1] = g;
          img.data[k * 4 + 2] = b;
          img.data[k * 4 + 3] = 255;
        }
        const off = document.createElement("canvas");
        off.width = FIELD_N;
        off.height = FIELD_N;
        off.getContext("2d")!.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
      }
    }
    continueRender(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  // Slow breathing scale.
  const breathe = 1 + 0.02 * Math.sin((frame / fps) * 0.6);
  const plate = Math.min(width * 0.9, 1000) * breathe;

  // Whole-reel fade in/out.
  const fade = interpolate(
    frame,
    [0, 30, durationInFrames - 40, durationInFrames - 1],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  // Subtle caption fades in then out early.
  const capOp = interpolate(frame, [40, 80, 180, 230], [0, 0.7, 0.7, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ opacity: fade, backgroundColor: "#04060d" }}>
      <Backdrop />
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: plate,
          height: plate,
          borderRadius: 14,
          overflow: "hidden",
          boxShadow: "0 0 120px rgba(80,120,180,0.25)",
        }}
      >
        <canvas
          ref={canvasRef}
          width={FIELD_N * 4}
          height={FIELD_N * 4}
          style={{ width: "100%", height: "100%", display: "block" }}
        />
      </div>

      {/* one quiet caption, then silence */}
      <div
        style={{
          position: "absolute",
          bottom: 150,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.display,
          fontSize: 34,
          fontWeight: 300,
          letterSpacing: 3,
          color: theme.ink,
          opacity: capOp,
        }}
      >
        a mind, resonating
      </div>
      <div
        style={{
          position: "absolute",
          bottom: 70,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.mono,
          fontSize: 22,
          letterSpacing: 4,
          color: theme.muted,
          opacity: 0.5,
        }}
      >
        biotuner · EEG
      </div>
    </AbsoluteFill>
  );
};
