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
import { MultiGeometryMorph } from "../scenes/MultiGeometryMorph";
import { REEL_DATA } from "./reelData";

export type ReelSpec = {
  id: string;
  width: number;
  height: number;
  fps: number;
  durationInFrames: number;
  Component: React.FC;
};

/** Build a standard reel composition (brand intro → viz scene). The viz
 *  scene is chosen by `data.scene`: "multi" → MultiGeometryMorph, else the
 *  Reel-02 cymatics scene. */
function cymaticsReel(id: string): ReelSpec {
  const d = REEL_DATA[id];
  const Scene = d.scene === "multi" ? MultiGeometryMorph : CymaticsChordMorph;
  return {
    id,
    width: 1080,
    height: 1920,
    fps: d.fps,
    durationInFrames: d.total_frames,
    Component: () =>
      React.createElement(Reel, {
        audio: d.audio,
        children: React.createElement(ReelTimeline, {
          introFrames: d.intro_frames,
          mainFrames: d.morph_frames,
          intro: d.intro!,
          main: React.createElement(Scene, { data: d }),
        }),
      }),
  };
}

export const REEL_SPECS: ReelSpec[] = [
  cymaticsReel("Reel02-Cymatics"),
  cymaticsReel("Reel03-Intervals"),
  cymaticsReel("Reel04-HeyJude"),
  cymaticsReel("Reel05-LetItBe"),
  cymaticsReel("Reel06-Canon"),
  cymaticsReel("Reel07-BrainHeart"),
  cymaticsReel("Reel08-ManyShapes"),
  cymaticsReel("Reel09-CanonHarmonograph"),
  cymaticsReel("Reel10-LetItBeShapes"),
];
