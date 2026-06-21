/**
 * Typed accessors for the per-reel data written by export_reels.py into
 * public/reels/<id>.json. Imported at build time (the files are tiny —
 * just chord sequences + timing).
 */
import reel02 from "../../public/reels/Reel02-Cymatics.json";
import type { Chord } from "./cymatics";

export type ReelData = {
  id: string;
  fps: number;
  frames_per_segment: number;
  total_frames: number;
  symmetry: "d4_max" | "d4_sum" | "none";
  loop: boolean;
  chords: Chord[];
  audio: string; // staticFile-relative path, e.g. "audio/Reel02-Cymatics.wav"
};

export const REEL_DATA: Record<string, ReelData> = {
  "Reel02-Cymatics": reel02 as unknown as ReelData,
};
