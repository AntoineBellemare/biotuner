import React, { useMemo } from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { PedagogyCaption } from "../components/PedagogyCaption";
import { Stage } from "../components/Stage";
import { theme, getChordHue, plasma, lerpRgb } from "../theme";
import { geometryData, Vec3 } from "../geometry";
import { tumbleProject, depthOpacity } from "../projection";

export const FRAMES_PER_CLOUD = 150; // 5 s

// Portrait / Reels pedagogy is scene-static: ONE title + ONE body for
// the whole 15-second slot. Three surfaces cycle visually so the viewer
// sees how the chord projects onto sphere / torus / Klein, but the text
// stays put so they can finish reading.
const PEDAGOGY_IG_TITLE = "Harmonic point clouds";
const PEDAGOGY_IG_BODY =
  "A chord painted on a surface. Bright bands are **constructive " +
  "resonance**.";

const PEDAGOGY: Record<string, string> = {
  sphere:
    "On a sphere, the chord's harmonic field bunches points into bright " +
    "bands of constructive resonance. Brightness = local field amplitude; " +
    "darker patches are nodal regions where the standing wave cancels out.",
  torus:
    "On a torus, both winding directions contribute. The chord's seventh " +
    "adds a long-period spiral that wraps around the inner ring as the " +
    "cloud rotates.",
  klein:
    "A non-orientable surface, Lawson's immersion of the Klein bottle in " +
    "R³. The same chord is forced through a one-sided geometry, producing " +
    "a dramatically different resonance pattern.",
};

/**
 * Rotating 3-D harmonic point cloud, coloured by field amplitude.
 */
export const PointCloud3D: React.FC = () => {
  const frame = useCurrentFrame();
  const items = geometryData.scenes.harmonic_point_clouds.items;

  const itemIdx = Math.min(
    Math.floor(frame / FRAMES_PER_CLOUD),
    items.length - 1
  );
  const localFrame = frame % FRAMES_PER_CLOUD;
  const item = items[itemIdx];

  const yaw = (localFrame / FRAMES_PER_CLOUD) * Math.PI * 2;
  const pitch = 0.28;

  const projected = useMemo(
    () => tumbleProject(item.vertices as Vec3[], yaw, pitch, 1.8),
    [item.vertices, yaw, pitch]
  );

  // Sort by depth so back points render first
  const order = useMemo(() => {
    const idx = projected.map((_, i) => i);
    idx.sort((a, b) => projected[a][2] - projected[b][2]);
    return idx;
  }, [projected]);

  // Two-tier colour scheme:
  //   - low-weight points use a plasma-style sequential ramp (deep purple →
  //     red → yellow) so the bright resonance bands really stand out
  //     against dark "node" regions where the field cancels.
  //   - high-weight points blend toward the chord's identity hue so the
  //     scene still reads as "Major" / "Dom7" / "Dim7" at a glance.
  const hue = getChordHue(item.name);

  function colorFor(w: number): string {
    // plasma ramp for w < 0.6, then blend toward chord-identity hue
    if (w < 0.6) {
      const [r, g, b] = plasma(w / 0.6);
      return `rgb(${r},${g},${b})`;
    }
    // w in [0.6, 1] → mix from plasma top into chord hue
    const t = (w - 0.6) / 0.4;
    const [r, g, b] = plasma(1.0);
    return lerpRgb([r, g, b], hue.rgb, t);
  }

  return (
    <AbsoluteFill>
      <Backdrop />
      {/* Portrait: anchor the cloud Stage low (y=580) so it sits in the
          lower 60 % of the frame, clear of the IG pedagogy card. */}
      <Stage portraitSize={1080} portraitTopOffset={580}>
        {order.map((i) => {
          const p = projected[i];
          const w = item.weights[i] ?? 0.5;
          const col = colorFor(w);
          const dop = depthOpacity(p[2], 0.95, 0.25);
          const r = 0.0075 + 0.013 * w;
          return (
            <circle
              key={i}
              cx={p[0]}
              cy={p[1]}
              r={r}
              fill={col}
              opacity={dop * (0.55 + 0.45 * w)}
            />
          );
        })}
      </Stage>
      <PedagogyCaption
        title={`Harmonic point cloud · ${item.name}`}
        body={PEDAGOGY[item.surface] ?? item.subtitle}
        meta={`${item.vertices.length} pts · ${item.surface} surface`}
        accent={hue.primary}
        igTitle={PEDAGOGY_IG_TITLE}
        igBody={PEDAGOGY_IG_BODY}
      />
    </AbsoluteFill>
  );
};
