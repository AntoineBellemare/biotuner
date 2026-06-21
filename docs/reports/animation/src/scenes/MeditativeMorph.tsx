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
import { MEDITATIVE_HUES, sampleCyclic } from "../reels/palettes";
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
  const { width, fps } = useVideoConfig();
  // The morph's own length (NOT the composition's) — so timing stays correct
  // whether or not a brand intro is prepended (inside ReelTimeline `frame` is
  // sequence-relative but useVideoConfig().durationInFrames is the whole reel).
  const durationInFrames = data.morph_frames;
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
  // Slow plate animation on TOP of the morph — the mode pairs breathe so the
  // nodal lattice is always gently flowing, not just changing at chord edges.
  const animTime = (frame / fps) * 0.45;

  // Hue is driven by position (concentric muted bands) and slowly rotates
  // through the ring over the reel, so the colour MIX drifts without ever
  // becoming a single flat tint.
  const hueDrift = frame / durationInFrames;
  // Gentle gamma breathing — keeps the thin bright lattice on a dark ground.
  const gamma = 0.82 + 0.16 * Math.sin((frame / fps) * 0.5);

  useEffect(() => {
    const handle = delayRender("meditative");
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const dens = cymaticsDensity(ratios, FIELD_N, {
          symmetry: "d4_max",
          time: animTime,
        });
        const img = ctx.createImageData(FIELD_N, FIELD_N);
        const c0 = (FIELD_N - 1) / 2;
        for (let k = 0; k < dens.length; k++) {
          // brightness from density (dark ground, bright lattice)…
          const bright = Math.pow(dens[k], gamma);
          // …hue from radius (concentric muted bands), drifting over time.
          const x = k % FIELD_N;
          const y = (k - x) / FIELD_N;
          const dx = (x - c0) / c0;
          const dy = (y - c0) / c0;
          const rad = Math.min(1, Math.sqrt(dx * dx + dy * dy) / 1.414);
          const [hr, hg, hb] = sampleCyclic(MEDITATIVE_HUES, rad * 1.25 + hueDrift);
          // brightest cores lift gently toward pale so nodes still glow.
          const core = Math.pow(dens[k], 6) * 0.4;
          img.data[k * 4 + 0] = Math.round((hr + (232 - hr) * core) * bright);
          img.data[k * 4 + 1] = Math.round((hg + (232 - hg) * core) * bright);
          img.data[k * 4 + 2] = Math.round((hb + (236 - hb) * core) * bright);
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
