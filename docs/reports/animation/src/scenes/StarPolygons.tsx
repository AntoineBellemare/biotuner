import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { Caption } from "../components/Caption";
import { Stage } from "../components/Stage";
import { theme } from "../theme";
import { geometryData, Vec2 } from "../geometry";

const FRAMES_PER_ITEM = 40;

export const StarPolygons: React.FC = () => {
  const frame = useCurrentFrame();
  const items = geometryData.scenes.star_polygons.items;
  const activeIdx = Math.min(
    Math.floor(frame / FRAMES_PER_ITEM),
    items.length - 1
  );
  const local = (frame % FRAMES_PER_ITEM) / FRAMES_PER_ITEM;
  // Slow continuous rotation tied to the absolute frame.
  const rotation = (frame / 30) * 12; // degrees per second

  return (
    <AbsoluteFill>
      <Backdrop />
      <Stage>
        <g transform={`rotate(${rotation})`}>
          {items.map((it, i) => {
            const dist = i - activeIdx;
            if (Math.abs(dist) > 1) return null;
            let op = 0;
            if (dist === 0) op = 1;
            else if (dist === -1) op = 1 - local;
            else if (dist === 1) op = local;
            return (
              <StarPath key={`${it.label}`} points={it.points} opacity={op} />
            );
          })}
        </g>
      </Stage>
      <Caption
        title={items[activeIdx]?.label}
        subtitle="star_polygon(n, k)"
      />
    </AbsoluteFill>
  );
};

const StarPath: React.FC<{ points: Vec2[]; opacity: number }> = ({
  points,
  opacity,
}) => {
  const closed = [...points, points[0]];
  const d = closed
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`)
    .join(" ");
  return (
    <>
      <path
        d={d}
        fill="none"
        stroke={theme.cool}
        strokeWidth={0.012}
        opacity={opacity * 0.4}
      />
      <path
        d={d}
        fill="none"
        stroke={theme.accent}
        strokeWidth={0.006}
        opacity={opacity}
        style={{ filter: `drop-shadow(0 0 8px ${theme.glow})` }}
      />
      {points.map((p, i) => (
        <circle
          key={i}
          cx={p[0]}
          cy={p[1]}
          r={0.018}
          fill={theme.ink}
          opacity={opacity}
        />
      ))}
    </>
  );
};
