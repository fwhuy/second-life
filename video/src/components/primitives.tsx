import React from "react";
import {
  Easing,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { c, f } from "../theme";

/** Fade + rise. Every element on screen enters through this so the motion feels authored. */
export const Reveal: React.FC<{
  delay?: number;
  y?: number;
  damping?: number;
  style?: React.CSSProperties;
  children: React.ReactNode;
}> = ({ delay = 0, y = 28, damping = 200, style, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = spring({
    frame: frame - delay,
    fps,
    config: { damping, mass: 0.7 },
  });
  return (
    <div
      style={{
        opacity: p,
        transform: `translateY(${(1 - p) * y}px)`,
        ...style,
      }}
    >
      {children}
    </div>
  );
};

/** Small uppercase kicker. */
export const Label: React.FC<{
  children: React.ReactNode;
  color?: string;
  style?: React.CSSProperties;
}> = ({ children, color = c.gold, style }) => (
  <div
    style={{
      fontFamily: f.mono,
      fontSize: 24,
      letterSpacing: "0.22em",
      textTransform: "uppercase",
      color,
      ...style,
    }}
  >
    {children}
  </div>
);

/** Animated integer. Separators are applied after rounding so digits never jitter. */
export const CountUp: React.FC<{
  to: number;
  delay?: number;
  dur?: number;
  decimals?: number;
  suffix?: string;
  style?: React.CSSProperties;
}> = ({ to, delay = 0, dur = 45, decimals = 0, suffix = "", style }) => {
  const frame = useCurrentFrame();
  const v = interpolate(frame - delay, [0, dur], [0, to], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });
  const text =
    decimals > 0
      ? v.toFixed(decimals)
      : Math.round(v).toLocaleString("en-US");
  return <span style={style}>{text}{suffix}</span>;
};

/** Horizontal rule that draws itself left-to-right. */
export const DrawRule: React.FC<{
  delay?: number;
  dur?: number;
  color?: string;
  height?: number;
  style?: React.CSSProperties;
}> = ({ delay = 0, dur = 26, color = c.rule, height = 2, style }) => {
  const frame = useCurrentFrame();
  const w = interpolate(frame - delay, [0, dur], [0, 100], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  return (
    <div style={{ height, background: "transparent", ...style }}>
      <div style={{ width: `${w}%`, height: "100%", background: color }} />
    </div>
  );
};
