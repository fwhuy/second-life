import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { Stage } from "../components/Chrome";
import { DrawRule, Label, Reveal } from "../components/primitives";
import { c, f } from "../theme";

export const S1Claim: React.FC = () => {
  const frame = useCurrentFrame();

  // The headline number desaturates the moment it is struck through.
  const struck = interpolate(frame, [58, 82], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <Stage>
      <Reveal delay={0}>
        <Label>Commonly reported on TrashNet</Label>
      </Reveal>

      <div style={{ position: "relative", width: "fit-content", marginTop: 18 }}>
        <Reveal delay={6} y={44} damping={150}>
          <div
            style={{
              fontFamily: f.serif,
              fontWeight: 900,
              fontSize: 272,
              lineHeight: 1,
              letterSpacing: "-0.02em",
              color: `rgb(${interpolate(struck, [0, 1], [47, 169])}, ${interpolate(
                struck,
                [0, 1],
                [90, 161]
              )}, ${interpolate(struck, [0, 1], [60, 131])})`,
            }}
          >
            99–100%
          </div>
        </Reveal>
        <DrawRule
          delay={58}
          dur={24}
          color={c.alarm}
          height={9}
          style={{
            position: "absolute",
            top: "52%",
            left: -10,
            right: -10,
          }}
        />
      </div>

      <Reveal delay={80} y={34}>
        <div
          style={{
            fontFamily: f.serif,
            fontWeight: 900,
            fontSize: 76,
            lineHeight: 1.24,
            marginTop: 34,
          }}
        >
          is not a result.
          <br />
          <span style={{ color: c.alarm }}>It is usually a leak.</span>
        </div>
      </Reveal>
    </Stage>
  );
};
