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
import { theme, fonts, getChordHue } from "../theme";
import { lerpRatios, type Chord } from "../reels/cymatics";
import type { ReelData } from "../reels/reelData";
import {
  drawCymatics,
  drawLissajous,
  drawHarmonograph,
  drawInterference,
  GEOM_LABEL,
  type GeomKind,
} from "../reels/geometries";

/**
 * Cycles a chord through several harmonic-geometry generators — the same
 * music shown as cymatics, then Lissajous, then harmonograph, then
 * interference — with a brief fade between each so the geometry TYPE reads
 * as the thing that changes.
 */
export const MultiGeometryMorph: React.FC<{ data: ReelData }> = ({ data }) => {
  const frame = useCurrentFrame();
  const { width, fps } = useVideoConfig();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const chords: Chord[] = data.chords;
  const geoms = (data.geometries ?? []) as GeomKind[];
  const n = Math.max(chords.length, geoms.length);
  const seg = data.frames_per_segment;

  const rawSeg = Math.floor(frame / seg);
  const segIdx = data.loop ? rawSeg % n : Math.min(rawSeg, n - 1);
  const local = (frame % seg) / seg;
  const chord = chords[Math.min(segIdx, chords.length - 1)];
  const toChord = chords[Math.min((segIdx + 1) % n, chords.length - 1)];
  const kind: GeomKind = geoms[Math.min(segIdx, geoms.length - 1)] ?? "cymatics";
  const prevKind: GeomKind =
    geoms[((segIdx - 1 + n) % n) % geoms.length] ?? kind;
  const nextKind: GeomKind = geoms[((segIdx + 1) % n) % geoms.length] ?? kind;

  // When consecutive segments share a geometry TYPE (e.g. an all-harmonograph
  // song reel), morph the chord ratios continuously and DON'T fade. When the
  // type changes, fade through so the swap reads cleanly.
  const hold = data.hold_fraction ?? 0;
  const morphLocal =
    hold > 0 ? Math.max(0, Math.min(1, (local - hold) / (1 - hold))) : local;
  const sameAsNext = nextKind === kind;
  const ratios = sameAsNext
    ? lerpRatios(chord.ratios, toChord.ratios, morphLocal)
    : chord.ratios;

  const fadeIn =
    prevKind === kind
      ? 1
      : interpolate(local, [0, 0.12], [0, 1], { extrapolateRight: "clamp" });
  const fadeOut =
    nextKind === kind
      ? 1
      : interpolate(local, [0.86, 1], [1, 0], { extrapolateLeft: "clamp" });
  const fade = fadeIn * fadeOut;
  const gamma = 0.72 + 0.4 * Math.sin(Math.PI * (sameAsNext ? morphLocal : local));

  useEffect(() => {
    const handle = delayRender("multi-geometry");
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        // Use the GLOBAL frame for line-art animation phase so the figure
        // evolves continuously across segment boundaries (no phase reset).
        if (kind === "cymatics") drawCymatics(ctx, ratios, gamma);
        else if (kind === "lissajous") drawLissajous(ctx, ratios, frame, fps);
        else if (kind === "harmonograph")
          drawHarmonograph(ctx, ratios, frame, fps);
        else if (kind === "interference")
          drawInterference(ctx, ratios, frame, fps);
      }
    }
    continueRender(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  const plate = Math.min(width * 0.84, 920);
  const hue = getChordHue(chord.label);
  const introFade = interpolate(frame, [0, 12], [0, 1], {
    extrapolateRight: "clamp",
  });
  const isField = kind === "cymatics" || kind === "interference";

  return (
    <AbsoluteFill style={{ opacity: introFade }}>
      <Backdrop />

      {/* Hook */}
      <div
        style={{
          position: "absolute",
          top: 150,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.display,
          fontSize: 40,
          fontWeight: 300,
          letterSpacing: 1,
          color: theme.ink,
          opacity: 0.85,
          padding: "0 60px",
        }}
        dangerouslySetInnerHTML={{ __html: data.hook ?? "one sound, many shapes" }}
      />

      {/* Geometry canvas */}
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: plate,
          height: plate,
          borderRadius: 10,
          overflow: "hidden",
          opacity: fade,
          boxShadow: isField
            ? `0 0 80px ${hue.glow}`
            : "none",
        }}
      >
        <canvas
          ref={canvasRef}
          width={isField ? 1280 : 1100}
          height={isField ? 1280 : 1100}
          style={{ width: "100%", height: "100%", display: "block" }}
        />
      </div>

      {/* Geometry-type label + chord */}
      <div
        style={{
          position: "absolute",
          top: `calc(50% + ${plate / 2 + 60}px)`,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.display,
          opacity: fade,
        }}
      >
        <div
          style={{
            fontSize: 72,
            fontWeight: 700,
            color: hue.primary,
            textShadow: `0 0 28px ${hue.glow}`,
            letterSpacing: -1,
          }}
        >
          {GEOM_LABEL[kind]}
        </div>
        <div
          style={{
            fontSize: 30,
            fontFamily: fonts.mono,
            color: hue.soft,
            letterSpacing: 3,
            marginTop: 6,
          }}
        >
          {chord.name}
        </div>
      </div>

      {/* Footer */}
      <div
        style={{
          position: "absolute",
          bottom: 90,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.mono,
          fontSize: 24,
          letterSpacing: 3,
          color: theme.muted,
          opacity: 0.8,
        }}
      >
        biotuner · harmonic geometry
      </div>
    </AbsoluteFill>
  );
};
