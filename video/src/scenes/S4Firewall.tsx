import React from "react";
import { Stage } from "../components/Chrome";
import { Label, Reveal } from "../components/primitives";
import { c, f } from "../theme";

const GATES = [
  { n: "01", t: "Provenance", d: "Every image mapped to its origin. 2,527/2,527, exact, not statistical." },
  { n: "02", t: "Duplicate groups", d: "Perceptual hash + learned embeddings link near-identical photographs." },
  { n: "03", t: "Group-aware split", d: "A whole group lands in one fold. Never across two." },
  { n: "04", t: "Train-only augmentation", d: "Split first, augment after — and only the training set." },
  { n: "05", t: "Test quarantine", d: "All 361 spent-test images locked out, re-checked on every load." },
];

export const S4Firewall: React.FC = () => (
  <Stage>
    <Reveal delay={0}>
      <Label color={c.green}>Provenance to quarantine</Label>
      <div
        style={{
          fontFamily: f.serif,
          fontWeight: 900,
          fontSize: 68,
          lineHeight: 1.2,
          marginTop: 14,
        }}
      >
        Five gates, enforced at load time.
      </div>
    </Reveal>

    <div style={{ display: "flex", gap: 18, marginTop: 58 }}>
      {GATES.map((g, i) => (
        <Reveal key={g.n} delay={14 + i * 11} y={34} style={{ flex: 1 }}>
          <div
            style={{
              borderTop: `4px solid ${i === 4 ? c.alarm : c.green}`,
              paddingTop: 20,
              height: "100%",
            }}
          >
            <div
              style={{
                fontFamily: f.mono,
                fontSize: 22,
                letterSpacing: "0.18em",
                color: i === 4 ? c.alarm : c.gold,
              }}
            >
              {g.n}
            </div>
            <div
              style={{
                fontFamily: f.serif,
                fontWeight: 900,
                fontSize: 38,
                lineHeight: 1.2,
                margin: "12px 0 14px",
                minHeight: 92, // keeps the five descriptions on a shared baseline
              }}
            >
              {g.t}
            </div>
            <div
              style={{
                fontFamily: f.sans,
                fontSize: 26,
                lineHeight: 1.58,
                color: c.inkSoft,
              }}
            >
              {g.d}
            </div>
          </div>
        </Reveal>
      ))}
    </div>

    <Reveal delay={98} y={26} style={{ marginTop: 60 }}>
      <div
        style={{
          background: c.dark,
          borderRadius: 8,
          padding: "26px 32px",
          fontFamily: f.mono,
          fontSize: 30,
          lineHeight: 1.6,
          color: c.darkInk,
        }}
      >
        <span style={{ color: c.darkSage }}>assert_old_test_quarantined()</span>{" "}
        — raises <span style={{ color: c.alarm }}>LEAKAGE</span> or{" "}
        <span style={{ color: c.alarm }}>QUARANTINE BREACH</span>.
        <span style={{ color: c.darkSoft }}> It fails the run. It never warns.</span>
      </div>
    </Reveal>
  </Stage>
);
