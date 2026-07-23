import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { c, f, DURATION, SAFE_X, SAFE_Y, SCENES } from "../theme";

/**
 * Persistent frame furniture: brand, position marker, and a progress rule.
 * The rule is functional — "strict time control" is graded, and it lets the
 * presenter see their pace without looking at a clock.
 */
export const Chrome: React.FC = () => {
  const frame = useCurrentFrame();

  const active = [...SCENES].reverse().find((s) => frame >= s.from);
  const idx = SCENES.findIndex((s) => s.id === active?.id);
  const isClose = active?.id === "close";

  // Fade the whole chrome out over the final beat so the closing frame is clean.
  const chromeOpacity = interpolate(frame, [1735, 1760], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const progress = interpolate(frame, [0, DURATION], [0, 100], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        opacity: chromeOpacity,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: SAFE_Y - 42,
          left: SAFE_X,
          right: SAFE_X,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
        }}
      >
        <div
          style={{
            fontFamily: f.serif,
            fontWeight: 900,
            fontSize: 26,
            color: c.ink,
            letterSpacing: "0.04em",
          }}
        >
          万物皆有新生
          <span
            style={{
              fontFamily: f.mono,
              fontWeight: 400,
              fontSize: 19,
              letterSpacing: "0.16em",
              color: c.goldPale,
              marginLeft: 16,
            }}
          >
            SECOND LIFE AI
          </span>
        </div>

        {!isClose && active ? (
          <div
            style={{
              fontFamily: f.mono,
              fontSize: 21,
              letterSpacing: "0.2em",
              color: c.gold,
            }}
          >
            {String(idx + 1).padStart(2, "0")}
            <span style={{ color: c.rule }}> / </span>
            {active.label}
          </div>
        ) : null}
      </div>

      <div
        style={{
          position: "absolute",
          bottom: SAFE_Y - 44,
          left: SAFE_X,
          right: SAFE_X,
          height: 3,
          background: c.rule,
        }}
      >
        <div
          style={{ width: `${progress}%`, height: "100%", background: c.green }}
        />
      </div>
    </div>
  );
};

/** Page background + safe-area padding shared by every scene. */
export const Stage: React.FC<{
  children: React.ReactNode;
  style?: React.CSSProperties;
}> = ({ children, style }) => (
  <div
    style={{
      position: "absolute",
      inset: 0,
      color: c.ink,
      padding: `${SAFE_Y}px ${SAFE_X}px`,
      display: "flex",
      flexDirection: "column",
      justifyContent: "center",
      ...style,
    }}
  >
    {children}
  </div>
);
