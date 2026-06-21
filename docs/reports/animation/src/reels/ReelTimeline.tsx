import React from "react";
import {
  AbsoluteFill,
  Series,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { ReelIntro, type IntroConfig } from "./ReelIntro";

/**
 * Standard reel timeline: the shared brand intro, then the reel's main
 * scene, joined by an elegant fade-through-black at the cut so the Flower
 * of Life dissolves cleanly into the visualisation.
 */
export const ReelTimeline: React.FC<{
  introFrames: number;
  mainFrames: number;
  intro: IntroConfig;
  main: React.ReactNode;
}> = ({ introFrames, mainFrames, intro, main }) => {
  return (
    <AbsoluteFill>
      <Series>
        <Series.Sequence durationInFrames={introFrames}>
          <ReelIntro config={intro} />
        </Series.Sequence>
        <Series.Sequence durationInFrames={mainFrames}>{main}</Series.Sequence>
      </Series>
      {/* Fade-through-black across the intro→viz boundary. */}
      <CutToBlack at={introFrames} preFrames={14} postFrames={16} />
    </AbsoluteFill>
  );
};

/**
 * Full-frame black that ramps up just before frame `at`, sits solid across
 * the hard Series cut, then ramps back down — hiding the boundary behind a
 * brief, deliberate fade to black.
 */
const CutToBlack: React.FC<{
  at: number;
  preFrames: number;
  postFrames: number;
}> = ({ at, preFrames, postFrames }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(
    frame,
    [at - preFrames, at - 2, at + 2, at + postFrames],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  if (opacity <= 0) return null;
  return (
    <AbsoluteFill
      style={{ backgroundColor: "#000000", opacity, pointerEvents: "none" }}
    />
  );
};
