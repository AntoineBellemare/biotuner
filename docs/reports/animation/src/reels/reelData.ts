/**
 * Typed accessors for the per-reel data written by export_reels.py into
 * public/reels/<id>.json. Imported at build time (the files are tiny —
 * just chord sequences + timing).
 */
import reel02 from "../../public/reels/Reel02-Cymatics.json";
import reel03 from "../../public/reels/Reel03-Intervals.json";
import reel07 from "../../public/reels/Reel07-BrainHeart.json";
import reel08 from "../../public/reels/Reel08-ManyShapes.json";
import reel12 from "../../public/reels/Reel12-Meditative.json";
import reel13 from "../../public/reels/Reel13-Forbidden.json";
import type { Chord } from "./cymatics";
import type { IntroConfig } from "./ReelIntro";

export type ReelData = {
  id: string;
  fps: number;
  frames_per_segment: number;
  intro_frames: number;
  /** 0 = none; >0 reserves a closing outro at the end of the main scene. */
  outro_frames?: number;
  morph_frames: number;
  total_frames: number;
  symmetry: "d4_max" | "d4_sum" | "none";
  loop: boolean;
  /** 0 = continuous morph; >0 holds each chord for that fraction of the
      segment, then morphs to the next over the remainder. */
  hold_fraction: number;
  /** Top-of-frame hook line for the viz scene. */
  hook: string | null;
  /** Which viz scene to render: "cymatics" (default) or "multi". */
  scene?: string;
  /** Optional per-segment geometry type (for the multi-geometry scene). */
  geometries?: string[] | null;
  /** Optional gallery phases (for the gallery scene): each phase is a grid. */
  gallery_phases?: Array<{
    title: string;
    subtitle: string;
    accent: string;
    cells: Chord[];
  }> | null;
  intro: IntroConfig | null;
  chords: Chord[];
  audio: string; // staticFile-relative path, e.g. "audio/Reel02-Cymatics.wav"
};

export const REEL_DATA: Record<string, ReelData> = {
  "Reel02-Cymatics": reel02 as unknown as ReelData,
  "Reel03-Intervals": reel03 as unknown as ReelData,
  "Reel07-BrainHeart": reel07 as unknown as ReelData,
  "Reel08-ManyShapes": reel08 as unknown as ReelData,
  "Reel12-Meditative": reel12 as unknown as ReelData,
  "Reel13-Forbidden": reel13 as unknown as ReelData,
};
