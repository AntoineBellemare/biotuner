import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { theme, fonts } from "../theme";
import { FlowerOfLife } from "./motifs/FlowerOfLife";

/** Per-reel intro configuration (the part that "slightly changes each time"). */
export type IntroConfig = {
  title?: string; // brand line, default "BIOTUNER"
  tagline?: string; // default brand tagline
  topic: string; // this reel's subject, e.g. "Harmonic Geometry"
  motif: "flower_of_life"; // which background motif to draw
  accent?: string; // motif + topic colour
};

const DEFAULT_TITLE = "BIOTUNER";
const DEFAULT_TAGLINE = "Visualizing and sonifying biological signals";

/**
 * Reusable reel opening: brand title + tagline + this reel's topic, over an
 * animated geometric motif. Shared across every reel so the channel has a
 * consistent open; `topic` + `motif` change per reel.
 */
export const ReelIntro: React.FC<{ config: IntroConfig }> = ({ config }) => {
  const frame = useCurrentFrame();
  const { durationInFrames, width } = useVideoConfig();
  const accent = config.accent ?? "#7ad6c1";

  const title = config.title ?? DEFAULT_TITLE;
  const tagline = config.tagline ?? DEFAULT_TAGLINE;

  // Reveal timings (frames).
  const titleY = interpolate(frame, [0, 18], [28, 0], {
    extrapolateRight: "clamp",
  });
  const titleOp = interpolate(frame, [0, 18], [0, 1], {
    extrapolateRight: "clamp",
  });
  const taglineOp = interpolate(frame, [12, 32], [0, 1], {
    extrapolateRight: "clamp",
  });
  const topicOp = interpolate(frame, [40, 60], [0, 1], {
    extrapolateRight: "clamp",
  });
  const motifProgress = interpolate(frame, [0, 70], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Dissolve the whole intro out over the final 12 frames into the next scene.
  const exitOp = interpolate(
    frame,
    [durationInFrames - 12, durationInFrames - 1],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const motifSize = Math.min(width * 0.78, 860);

  return (
    <AbsoluteFill style={{ opacity: exitOp }}>
      <Backdrop />

      {/* Motif, centred as the hero. */}
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          opacity: 0.95,
        }}
      >
        {config.motif === "flower_of_life" ? (
          <FlowerOfLife
            size={motifSize}
            color={accent}
            glow={`${hexToRgba(accent, 0.45)}`}
            progress={motifProgress}
          />
        ) : null}
      </div>

      {/* Brand title + tagline, above the motif. */}
      <div
        style={{
          position: "absolute",
          top: 250,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.display,
        }}
      >
        <div
          style={{
            fontSize: 96,
            fontWeight: 800,
            letterSpacing: 14,
            color: theme.ink,
            opacity: titleOp,
            transform: `translateY(${titleY}px)`,
            textShadow: `0 0 30px ${hexToRgba(accent, 0.35)}`,
          }}
        >
          {title}
        </div>
        <div
          style={{
            marginTop: 18,
            fontSize: 30,
            fontWeight: 300,
            letterSpacing: 1.5,
            color: theme.muted,
            opacity: taglineOp,
          }}
        >
          {tagline}
        </div>
      </div>

      {/* This reel's topic, below the motif. */}
      <div
        style={{
          position: "absolute",
          bottom: 320,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.display,
          opacity: topicOp,
        }}
      >
        <div
          style={{
            display: "inline-block",
            padding: "12px 34px",
            borderRadius: 999,
            border: `1px solid ${hexToRgba(accent, 0.5)}`,
            fontSize: 46,
            fontWeight: 600,
            letterSpacing: 2,
            color: accent,
            textShadow: `0 0 24px ${hexToRgba(accent, 0.4)}`,
            background: hexToRgba(accent, 0.06),
          }}
        >
          {config.topic}
        </div>
      </div>
    </AbsoluteFill>
  );
};

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
