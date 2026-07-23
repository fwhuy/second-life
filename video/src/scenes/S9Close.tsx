import React from "react";
import { Stage } from "../components/Chrome";
import { Reveal } from "../components/primitives";
import { c, f } from "../theme";

export const S9Close: React.FC = () => (
  <Stage style={{ alignItems: "center", textAlign: "center" }}>
    <Reveal delay={0} y={30} damping={170}>
      <div
        style={{
          fontFamily: f.serif,
          fontWeight: 900,
          fontSize: 140,
          lineHeight: 1.1,
          letterSpacing: "0.06em",
        }}
      >
        万物皆有新生
      </div>
    </Reveal>

    <Reveal delay={10} y={22}>
      <div
        style={{
          fontFamily: f.serif,
          fontWeight: 900,
          fontSize: 54,
          color: c.green,
          marginTop: 18,
        }}
      >
        Every object deserves a second life
      </div>
    </Reveal>

    <Reveal delay={22} y={18}>
      <div
        style={{
          fontFamily: f.sans,
          fontSize: 34,
          color: c.inkSoft,
          marginTop: 34,
        }}
      >
        A number you can defend is worth more than a number you can&rsquo;t.
      </div>
    </Reveal>
  </Stage>
);
