import React from "react";
import { Stage } from "../components/Chrome";
import { Label, Reveal } from "../components/primitives";
import { c, f } from "../theme";

const ROWS = [
  {
    x: "Progressive resize, 224 → 384px",
    r: "412 / 434",
    net: "±0",
    why: "10 fixes, 10 regressions",
  },
  {
    x: "Modern augmentation",
    r: "415 / 434",
    net: "+3",
    why: "trash recall fell 20/23 → 18/23",
  },
  {
    x: "Weight decay 0.05 → 0.10",
    r: "411 / 434",
    net: "−1",
    why: "macro-F1 fell too",
  },
];

export const S6Ablation: React.FC = () => (
  <Stage>
    <Reveal delay={0}>
      <Label color={c.green}>One variable at a time</Label>
      <div
        style={{
          fontFamily: f.serif,
          fontWeight: 900,
          fontSize: 60,
          lineHeight: 1.2,
          marginTop: 14,
        }}
      >
        Three experiments, each gated{" "}
        <span style={{ fontStyle: "italic" }}>before</span> it ran.
      </div>
    </Reveal>

    <div style={{ marginTop: 48 }}>
      <Reveal delay={12} y={12}>
        <div
          style={{
            display: "flex",
            fontFamily: f.mono,
            fontSize: 21,
            letterSpacing: "0.16em",
            color: c.gold,
            paddingBottom: 14,
            borderBottom: `2px solid ${c.rule}`,
          }}
        >
          <div style={{ flex: 3 }}>ISOLATED VARIABLE</div>
          <div style={{ flex: 1 }}>TTA RESULT</div>
          <div style={{ flex: 1 }}>NET IMAGES</div>
          <div style={{ flex: 1.4 }}>VERDICT</div>
        </div>
      </Reveal>

      {ROWS.map((r, i) => (
        <Reveal key={r.x} delay={22 + i * 14} y={22}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              padding: "26px 0",
              borderBottom: `1px solid ${c.rule}`,
              fontFamily: f.sans,
              fontSize: 32,
            }}
          >
            <div style={{ flex: 3, paddingRight: 24 }}>
              {r.x}
              <div
                style={{ fontSize: 24, color: c.inkSoft, marginTop: 6 }}
              >
                {r.why}
              </div>
            </div>
            <div style={{ flex: 1, fontFamily: f.mono, color: c.inkSoft }}>
              {r.r}
            </div>
            <div
              style={{
                flex: 1,
                fontFamily: f.mono,
                fontSize: 38,
                color: c.ink,
              }}
            >
              {r.net}
            </div>
            <div
              style={{
                flex: 1.4,
                fontFamily: f.mono,
                fontSize: 26,
                letterSpacing: "0.14em",
                color: c.alarm,
              }}
            >
              REJECTED
            </div>
          </div>
        </Reveal>
      ))}
    </div>

    <Reveal delay={82} y={24} style={{ marginTop: 44 }}>
      <div
        style={{
          fontFamily: f.sans,
          fontSize: 38,
          lineHeight: 1.6,
          color: c.inkSoft,
          maxWidth: 1540,
        }}
      >
        The acceptance threshold — <span style={{ color: c.ink, fontWeight: 700 }}>+7 images</span>{" "}
        — was fixed in advance. None of the three cleared it, so none was adopted.
        Stopping was the finding.
      </div>
    </Reveal>
  </Stage>
);
