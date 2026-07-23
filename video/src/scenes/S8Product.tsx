import React from "react";
import { Stage } from "../components/Chrome";
import { Label, Reveal } from "../components/primitives";
import { c, f } from "../theme";

const STEPS = [
  { n: "一", zh: "洞察实验室", en: "Insight Lab", d: "Photograph an object. Real softmax, real confidence.", tint: c.greenBright },
  { n: "二", zh: "命运分岔口", en: "Fork in the Road", d: "Choose its fate. Watch both timelines play out.", tint: c.amber },
  { n: "三", zh: "万物博物馆", en: "Museum of Everything", d: "Materials, collections, campus stories.", tint: c.sage },
];

export const S8Product: React.FC = () => (
  <Stage>
    <Reveal delay={0}>
      <Label color={c.green}>What the classifier powers</Label>
      <div
        style={{
          fontFamily: f.serif,
          fontWeight: 900,
          fontSize: 64,
          lineHeight: 1.2,
          marginTop: 14,
        }}
      >
        Offline. Bilingual. Running on the real checkpoint.
      </div>
    </Reveal>

    <div style={{ display: "flex", gap: 30, marginTop: 56 }}>
      {STEPS.map((s, i) => (
        <Reveal key={s.en} delay={14 + i * 12} y={36} style={{ flex: 1 }}>
          <div
            style={{
              background: c.creamDeep,
              borderRadius: 12,
              padding: "34px 32px 38px",
              height: "100%",
            }}
          >
            <div
              style={{
                width: 62,
                height: 62,
                borderRadius: 16,
                background: s.tint,
                color: c.cream,
                fontFamily: f.serif,
                fontWeight: 900,
                fontSize: 30,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {s.n}
            </div>
            <div
              style={{
                fontFamily: f.serif,
                fontWeight: 900,
                fontSize: 44,
                marginTop: 24,
              }}
            >
              {s.zh}
            </div>
            <div
              style={{
                fontFamily: f.sans,
                fontSize: 30,
                color: s.tint,
                marginTop: 8,
                fontWeight: 700,
              }}
            >
              {s.en}
            </div>
            <div
              style={{
                fontFamily: f.sans,
                fontSize: 27,
                lineHeight: 1.56,
                color: c.inkSoft,
                marginTop: 16,
              }}
            >
              {s.d}
            </div>
          </div>
        </Reveal>
      ))}
    </div>
  </Stage>
);
