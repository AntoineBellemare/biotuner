import React from "react";
import { Composition } from "remotion";
import { MainV2, TOTAL_FRAMES_V2 } from "./MainV2";
import { MainV3, TOTAL_FRAMES_V3 } from "./MainV3";
import { HarmonicSimilarity, TOTAL_HARMSIM } from "./scenes/HarmonicSimilarity";
import { SubharmonicTension, TOTAL_SUBHARM } from "./scenes/SubharmonicTension";
import { SetharesDissonance, TOTAL_SETHARES } from "./scenes/SetharesDissonance";
import { DissonancePipeline, TOTAL_DISSPIPE } from "./scenes/DissonancePipeline";
import { BrainPolyrhythm, TOTAL_BRAINPOLY } from "./scenes/BrainPolyrhythm";
import { HeartBrainDuet, TOTAL_HEARTBRAIN } from "./scenes/HeartBrainDuet";
import { BrainGrooves, TOTAL_BRAINGROOVES } from "./scenes/BrainGrooves";
import { PitchRhythm, TOTAL_PITCHRHYTHM } from "./scenes/PitchRhythm";
import { FractalRhythm, TOTAL_FRACTAL } from "./scenes/FractalRhythm";
import { REEL_SPECS } from "./reels/specs";

// GeometryV2 is the flagship showcase; GeometryV2-IG is its Instagram
// portrait crop (Reel 01). The v1 "MainVideo" composition has been retired.
// Short-form reels are registered from src/reels/specs.ts.

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="GeometryV2"
        component={MainV2}
        durationInFrames={TOTAL_FRAMES_V2}
        fps={30}
        width={1920}
        height={1080}
      />
      {/* Instagram Reels / TikTok / Shorts — 9:16 portrait at 1080×1920. */}
      <Composition
        id="GeometryV2-IG"
        component={MainV2}
        durationInFrames={TOTAL_FRAMES_V2}
        fps={30}
        width={1080}
        height={1920}
      />
      {/* GeometryV3-IG — same reel, Chladni section in the nodal-density
          "sand" style of the short-form reels. */}
      <Composition
        id="GeometryV3-IG"
        component={MainV3}
        durationInFrames={TOTAL_FRAMES_V3}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Didactic — harmonic similarity (dyad_similarity comb alignment). */}
      <Composition
        id="HarmonicSimilarity-IG"
        component={HarmonicSimilarity}
        durationInFrames={TOTAL_HARMSIM}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Didactic — subharmonic tension (common-fundamental convergence). */}
      <Composition
        id="SubharmonicTension-IG"
        component={SubharmonicTension}
        durationInFrames={TOTAL_SUBHARM}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Didactic — Sethares / Plomp-Levelt sensory dissonance curve. */}
      <Composition
        id="SetharesDissonance-IG"
        component={SetharesDissonance}
        durationInFrames={TOTAL_SETHARES}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Didactic — signal → peaks → dissonance curve → scale (incl. EEG). */}
      <Composition
        id="DissonancePipeline-IG"
        component={DissonancePipeline}
        durationInFrames={TOTAL_DISSPIPE}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Biorhythm — EEG peaks → 5:7:11 polyrhythm phase wheel. */}
      <Composition
        id="BrainPolyrhythm-IG"
        component={BrainPolyrhythm}
        durationInFrames={TOTAL_BRAINPOLY}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Biorhythm — ECG heartbeat × EEG polyrhythm duet (coincidence groove). */}
      <Composition
        id="HeartBrainDuet-IG"
        component={HeartBrainDuet}
        durationInFrames={TOTAL_HEARTBRAIN}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Biorhythm — EEG → Euclidean rhythms → named world grooves. */}
      <Composition
        id="BrainGrooves-IG"
        component={BrainGrooves}
        durationInFrames={TOTAL_BRAINGROOVES}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Rhythm — beat_envelope: pitch ↔ rhythm continuum. */}
      <Composition
        id="PitchRhythm-IG"
        component={PitchRhythm}
        durationInFrames={TOTAL_PITCHRHYTHM}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Rhythm — second_order_polyrhythm: rhythm nested inside rhythm. */}
      <Composition
        id="FractalRhythm-IG"
        component={FractalRhythm}
        durationInFrames={TOTAL_FRACTAL}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Short-form reels — one <Composition> per spec. */}
      {REEL_SPECS.map((spec) => (
        <Composition
          key={spec.id}
          id={spec.id}
          component={spec.Component}
          durationInFrames={spec.durationInFrames}
          fps={spec.fps}
          width={spec.width}
          height={spec.height}
        />
      ))}
    </>
  );
};
