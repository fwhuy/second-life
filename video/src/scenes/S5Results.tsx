import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { Stage } from "../components/Chrome";
import { Label, Reveal } from "../components/primitives";
import { c, f } from "../theme";

/**
 * Fold-0 validation confusion matrix, four-view TTA.
 * Source: model/reports/baseline_resnet50/confusion_baseline_resnet50_val_fold0_tta.png
 * Rows sum to 434; diagonal sums to 412 = 94.93%.
 */
const CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];
const CM = [
  [64, 0, 1, 2, 1, 1],
  [0, 78, 3, 1, 4, 0],
  [0, 2, 67, 0, 1, 1],
  [0, 0, 0, 101, 1, 0],
  [0, 0, 0, 1, 82, 0],
  [0, 0, 0, 1, 2, 20],
];
const CELL = 74;

const NUMBERS = [
  { v: "93.09%", l: "Validation, single view", note: "404 / 434", tone: c.inkSoft },
  { v: "94.93%", l: "Validation, four-view TTA", note: "412 / 434", tone: c.green, hero: true },
  { v: "89.20%", l: "Test — one pre-registered measurement", note: "322 / 361 · never re-run", tone: c.alarm },
];

export const S5Results: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <Stage>
      <Reveal delay={0}>
        <Label color={c.green}>What an honest number looks like</Label>
      </Reveal>

      <div style={{ display: "flex", gap: 80, marginTop: 30, alignItems: "center" }}>
        {/* ---- confusion matrix ---- */}
        <div>
          <div style={{ display: "flex" }}>
            <div style={{ width: 190 }} />
            {CLASSES.map((cl, j) => (
              <Reveal
                key={cl}
                delay={10 + j * 3}
                y={10}
                style={{
                  width: CELL,
                  fontFamily: f.mono,
                  fontSize: 17,
                  color: c.gold,
                  textAlign: "center",
                  paddingBottom: 8,
                  letterSpacing: "0.04em",
                }}
              >
                {cl.slice(0, 5)}
              </Reveal>
            ))}
            <Reveal
              delay={40}
              y={10}
              style={{
                width: 150,
                fontFamily: f.mono,
                fontSize: 17,
                color: c.gold,
                paddingBottom: 8,
                paddingLeft: 18,
                letterSpacing: "0.04em",
              }}
            >
              recall
            </Reveal>
          </div>

          {CM.map((row, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center" }}>
              <Reveal
                delay={12 + i * 4}
                y={8}
                style={{
                  width: 190,
                  fontFamily: f.mono,
                  fontSize: 23,
                  color: c.ink,
                  textAlign: "right",
                  paddingRight: 18,
                }}
              >
                {CLASSES[i]}
              </Reveal>
              {row.map((val, j) => {
                const diag = i === j;
                const appear = interpolate(
                  frame,
                  [16 + (i + j) * 3, 30 + (i + j) * 3],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );
                const intensity = diag
                  ? 0.16 + (val / 101) * 0.84
                  : val === 0
                    ? 0
                    : 0.14 + (val / 4) * 0.5;
                const bg = diag
                  ? `rgba(47, 90, 60, ${intensity * appear})`
                  : `rgba(196, 96, 47, ${intensity * appear})`;
                return (
                  <div
                    key={j}
                    style={{
                      width: CELL,
                      height: CELL,
                      background: bg,
                      border: `1px solid ${c.rule}`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontFamily: f.mono,
                      fontSize: 25,
                      color:
                        diag && intensity > 0.55
                          ? c.cream
                          : val === 0
                            ? c.goldPale
                            : c.ink,
                      opacity: appear,
                    }}
                  >
                    {val}
                  </div>
                );
              })}
              <Reveal
                delay={40 + i * 4}
                y={8}
                style={{
                  width: 150,
                  fontFamily: f.mono,
                  fontSize: 21,
                  color: c.gold,
                  paddingLeft: 18,
                }}
              >
                {row[i]}/{row.reduce((a, b) => a + b, 0)}
              </Reveal>
            </div>
          ))}

          <Reveal delay={70} y={10}>
            <div
              style={{
                fontFamily: f.mono,
                fontSize: 19,
                color: c.gold,
                marginTop: 16,
                marginLeft: 190,
                letterSpacing: "0.08em",
              }}
            >
              TRUE (rows) × PREDICTED (columns) · FOLD-0 VALIDATION, TTA
            </div>
          </Reveal>
        </div>

        {/* ---- the three numbers ---- */}
        <div style={{ flex: 1 }}>
          {NUMBERS.map((n, i) => (
            <Reveal key={n.v} delay={40 + i * 16} y={30} style={{ marginBottom: 34 }}>
              <div
                style={{
                  borderLeft: `5px solid ${n.tone}`,
                  paddingLeft: 24,
                }}
              >
                <div
                  style={{
                    fontFamily: f.serif,
                    fontWeight: 900,
                    fontSize: n.hero ? 108 : 74,
                    lineHeight: 1,
                    color: n.tone,
                  }}
                >
                  {n.v}
                </div>
                <div
                  style={{
                    fontFamily: f.sans,
                    fontSize: 30,
                    color: c.ink,
                    marginTop: 10,
                  }}
                >
                  {n.l}
                </div>
                <div
                  style={{
                    fontFamily: f.mono,
                    fontSize: 22,
                    color: c.gold,
                    marginTop: 6,
                  }}
                >
                  {n.note}
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </Stage>
  );
};
