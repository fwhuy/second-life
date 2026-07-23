import React from "react";
import { Stage } from "../components/Chrome";
import { Exhibit } from "../components/Exhibit";
import { CountUp, Label, Reveal } from "../components/primitives";
import { c, f } from "../theme";

export const S3Discovery: React.FC = () => (
  <Stage>
    <Reveal delay={0}>
      <Label color={c.alarm}>What we found in our own corpus</Label>
    </Reveal>

    <div
      style={{
        display: "flex",
        gap: 84,
        alignItems: "flex-start",
        marginTop: 26,
      }}
    >
      <div style={{ flex: "0 0 52%" }}>
        <Reveal delay={8} y={40} damping={160}>
          <div
            style={{
              fontFamily: f.serif,
              fontWeight: 900,
              fontSize: 216,
              lineHeight: 0.92,
              color: c.green,
              letterSpacing: "-0.02em",
            }}
          >
            <CountUp to={2218} delay={10} dur={52} />
          </div>
        </Reveal>
        <Reveal delay={44} y={26}>
          <div
            style={{
              fontFamily: f.serif,
              fontWeight: 900,
              fontSize: 46,
              lineHeight: 1.34,
              marginTop: 22,
            }}
          >
            of the 2,527 TrashNet images were
            <br />
            sitting inside another public dataset
            <br />
            <span style={{ color: c.alarm }}>under different filenames.</span>
          </div>
        </Reveal>
      </div>

      <Exhibit title="Quarantine check" delay={78} width="100%">
        <div
          style={{
            fontFamily: f.mono,
            fontSize: 92,
            color: c.darkInk,
            lineHeight: 1,
          }}
        >
          <CountUp to={361} delay={86} dur={30} />
          <span style={{ color: c.darkSoft }}>/361</span>
        </div>
        <div
          style={{
            fontFamily: f.sans,
            fontSize: 30,
            color: c.darkSage,
            marginTop: 14,
            lineHeight: 1.5,
          }}
        >
          spent test images recovered
          <br />
          and forced back into quarantine
        </div>

        <div style={{ display: "flex", gap: 3, marginTop: 30 }}>
          <div style={{ flex: 318, height: 14, background: c.alarm }} />
          <div style={{ flex: 43, height: 14, background: c.darkRule }} />
        </div>
        <div
          style={{
            fontFamily: f.mono,
            fontSize: 23,
            color: c.alarm,
            marginTop: 14,
            lineHeight: 1.55,
          }}
        >
          318 disguised under another
          <br />
          dataset&rsquo;s filename
        </div>
      </Exhibit>
    </div>

    <Reveal delay={150} y={26} style={{ marginTop: 52 }}>
      <div
        style={{
          fontFamily: f.sans,
          fontSize: 38,
          lineHeight: 1.6,
          color: c.inkSoft,
          maxWidth: 1560,
        }}
      >
        Filename comparison could not see it. Training on the corpus as
        downloaded would have meant training on{" "}
        <span style={{ color: c.ink, fontWeight: 700 }}>
          100% of our own test set
        </span>{" "}
        — the exact failure this project exists to criticise.
      </div>
    </Reveal>
  </Stage>
);
