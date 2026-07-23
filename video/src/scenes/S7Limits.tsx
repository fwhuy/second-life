import React from "react";
import { Img, staticFile } from "remotion";
import { Stage } from "../components/Chrome";
import { CountUp, Label, Reveal } from "../components/primitives";
import { c, f } from "../theme";

const Thumb: React.FC<{ src: string; caption: string; tone: string }> = ({
  src,
  caption,
  tone,
}) => (
  <div style={{ textAlign: "center" }}>
    <div
      style={{
        width: 176,
        height: 176,
        borderRadius: 6,
        overflow: "hidden",
        border: `3px solid ${tone}`,
        background: c.creamDeep,
      }}
    >
      <Img
        src={src}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
    </div>
    <div
      style={{
        fontFamily: f.mono,
        fontSize: 20,
        color: tone,
        marginTop: 12,
        lineHeight: 1.45,
        whiteSpace: "pre-line",
      }}
    >
      {caption}
    </div>
  </div>
);

export const S7Limits: React.FC = () => (
  <Stage>
    <Reveal delay={0}>
      <Label color={c.alarm}>What we still get wrong</Label>
    </Reveal>

    <div style={{ display: "flex", gap: 92, marginTop: 32, alignItems: "flex-start" }}>
      <div style={{ flex: "0 0 50%" }}>
        <Reveal delay={8} y={34}>
          <div
            style={{
              fontFamily: f.serif,
              fontWeight: 900,
              fontSize: 170,
              lineHeight: 0.95,
              color: c.green,
            }}
          >
            <CountUp to={14} delay={12} dur={28} />
            <span style={{ color: c.goldPale, fontSize: 96 }}> / 30</span>
          </div>
        </Reveal>
        <Reveal delay={34} y={24}>
          <div
            style={{
              fontFamily: f.sans,
              fontSize: 34,
              lineHeight: 1.6,
              marginTop: 22,
              color: c.inkSoft,
              maxWidth: 720,
            }}
          >
            of the highest-loss test errors we audited by hand were{" "}
            <span style={{ color: c.ink, fontWeight: 700 }}>
              mislabelled or genuinely ambiguous
            </span>{" "}
            — not model failures. Three pixel-identical pairs carry contradictory
            labels; one disagrees with TrashNet inside TrashNet itself.
          </div>
        </Reveal>
      </div>

      <div style={{ flex: 1 }}>
        <Reveal delay={54} y={30}>
          <div style={{ display: "flex", gap: 40 }}>
            <Thumb
              src={staticFile("bottle.jpeg")}
              caption={"in distribution\nplastic"}
              tone={c.green}
            />
            <Thumb
              src={staticFile("cat.jpg")}
              caption={"out of distribution\nnot waste at all"}
              tone={c.alarm}
            />
          </div>
        </Reveal>
        <Reveal delay={78} y={24}>
          <div
            style={{
              fontFamily: f.sans,
              fontSize: 32,
              lineHeight: 1.58,
              color: c.inkSoft,
              marginTop: 30,
              maxWidth: 660,
            }}
          >
            Softmax confidence alone did not separate these two. A classifier with
            six classes will answer{" "}
            <span style={{ color: c.ink, fontWeight: 700 }}>confidently</span> even
            when the right answer is none of them.
          </div>
        </Reveal>
      </div>
    </div>

    <Reveal delay={106} y={22} style={{ marginTop: 48 }}>
      <div
        style={{
          fontFamily: f.mono,
          fontSize: 28,
          letterSpacing: "0.06em",
          color: c.green,
          borderTop: `2px solid ${c.rule}`,
          paddingTop: 22,
        }}
      >
        NEXT → ConvNeXtV2 Tiny on an 11,299-image unified corpus.
        <span style={{ color: c.gold }}>
          {" "}
          Corpus measured and quarantined; model not yet trained.
        </span>
      </div>
    </Reveal>
  </Stage>
);
