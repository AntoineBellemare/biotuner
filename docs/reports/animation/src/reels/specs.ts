/**
 * Reel registry. Each entry is one renderable Instagram reel: its
 * dimensions, fps, duration, and a parameterless React component (the
 * shared brand intro + the reel's main scene, wrapped in the generic
 * <Reel> frame with its soundtrack).
 *
 * Adding a reel = (1) a builder in export_reels.py, (2) a data import in
 * reelData.ts, (3) one entry here. Root.tsx registers a <Composition> for
 * every entry automatically.
 */
import React from "react";
import { Reel } from "./Reel";
import { ReelTimeline } from "./ReelTimeline";
import { CymaticsChordMorph } from "../scenes/CymaticsChordMorph";
import { REEL_DATA } from "./reelData";

export type ReelSpec = {
  id: string;
  width: number;
  height: number;
  fps: number;
  durationInFrames: number;
  Component: React.FC;
};

const reel02 = REEL_DATA["Reel02-Cymatics"];

export const REEL_SPECS: ReelSpec[] = [
  {
    id: "Reel02-Cymatics",
    width: 1080,
    height: 1920,
    fps: reel02.fps,
    durationInFrames: reel02.total_frames,
    Component: () =>
      React.createElement(Reel, {
        audio: reel02.audio,
        children: React.createElement(ReelTimeline, {
          introFrames: reel02.intro_frames,
          mainFrames: reel02.morph_frames,
          intro: reel02.intro!,
          main: React.createElement(CymaticsChordMorph, { data: reel02 }),
        }),
      }),
  },
];
