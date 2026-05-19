import React from "react";
import { AbsoluteFill, Audio, Series, staticFile } from "remotion";
import { Title } from "./scenes/Title";
import { Outro } from "./scenes/Outro";
import { HarmonographRealtime, FRAMES_PER_VARIANT_RT } from "./scenes/HarmonographRealtime";
import { LissajousKnot3D, FRAMES_PER_KNOT } from "./scenes/LissajousKnot3D";
import { LSystemTree3D, FRAMES_PER_TREE } from "./scenes/LSystemTree3D";
import { PointCloud3D, FRAMES_PER_CLOUD } from "./scenes/PointCloud3D";
import { ChladniExpanded, FRAMES_PER_CHORD_EXPANDED } from "./scenes/ChladniExpanded";
import { geometryData } from "./geometry";

// Number of items in each new scene (drives durations)
const N_HARMONOGRAPH = geometryData.scenes.harmonograph_variants.variants.length;       // 4
const N_KNOTS         = geometryData.scenes.lissajous_3d_knots.items.length;            // 4
const N_TREES         = geometryData.scenes.lsystem_3d_variants.items.length;           // 4
const N_CLOUDS        = geometryData.scenes.harmonic_point_clouds.items.length;         // 3
const N_CHLADNI       = geometryData.scenes.chladni_expanded.items.length;              // 7

export const SCENE_FRAMES_V2 = {
  title:        90,                                                     //  3.0 s
  harmonograph: N_HARMONOGRAPH * FRAMES_PER_VARIANT_RT,                 // 16.0 s
  knots:        N_KNOTS         * FRAMES_PER_KNOT,                      // 20.0 s
  chladni:      N_CHLADNI       * FRAMES_PER_CHORD_EXPANDED,            // 17.5 s
  trees:        N_TREES         * FRAMES_PER_TREE,                      // 20.0 s
  clouds:       N_CLOUDS        * FRAMES_PER_CLOUD,                     // 15.0 s
  outro:        90,                                                     //  3.0 s
};

export const TOTAL_FRAMES_V2 =
  Object.values(SCENE_FRAMES_V2).reduce((a, b) => a + b, 0);

/**
 * GeometryV2 — pedagogical animation focused on the harmonic_geometry
 * module's flagship generators:
 *   • harmonograph (real-time pen reveal, fixed minor-spiral density)
 *   • 3-D Lissajous knots (rotating)
 *   • Chladni 4-plate grid (rectangular / circular / pentagon / box-3d slice)
 *   • 3-D L-system fractal harmonic growth (rotating)
 *   • harmonic point clouds (rotating, coloured by field amplitude)
 *
 * Each scene carries a non-technical pedagogical caption.
 */
export const MainV2: React.FC = () => {
  return (
    <AbsoluteFill>
      {/* Synced soundtrack rendered by docs/reports/animation/render_audio.py.
          Chord events / portamento glides / Chladni filter sweep all align
          with the visual timeline below — see render_audio.py for the synth
          design (3-voice detuned unison + 8-partial additive + shimmer). */}
      <Audio src={staticFile("audio/score.wav")} />
      <Series>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.title}>
          <Title />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.harmonograph}>
          <HarmonographRealtime />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.knots}>
          <LissajousKnot3D />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.chladni}>
          <ChladniExpanded />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.trees}>
          <LSystemTree3D />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.clouds}>
          <PointCloud3D />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.outro}>
          <Outro />
        </Series.Sequence>
      </Series>
    </AbsoluteFill>
  );
};
