import React from "react";
import {
  AbsoluteFill,
  interpolate,
  Sequence,
  useCurrentFrame,
} from "remotion";
import { Chrome } from "./components/Chrome";
import { c, SCENES } from "./theme";
import { S1Claim } from "./scenes/S1Claim";
import { S2Leak } from "./scenes/S2Leak";
import { S3Discovery } from "./scenes/S3Discovery";
import { S4Firewall } from "./scenes/S4Firewall";
import { S5Results } from "./scenes/S5Results";
import { S6Ablation } from "./scenes/S6Ablation";
import { S7Limits } from "./scenes/S7Limits";
import { S8Product } from "./scenes/S8Product";
import { S9Close } from "./scenes/S9Close";

const SCENE_COMPONENTS: Record<string, React.FC> = {
  claim: S1Claim,
  leak: S2Leak,
  discovery: S3Discovery,
  firewall: S4Firewall,
  results: S5Results,
  ablation: S6Ablation,
  limits: S7Limits,
  product: S8Product,
  close: S9Close,
};

/**
 * Scenes hard-cut in but fade out over the last 8 frames, so the cream page
 * underneath is never interrupted — the effect is a page turn, not a cut to black.
 */
const SceneWrap: React.FC<{ dur: number; children: React.ReactNode }> = ({
  dur,
  children,
}) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [dur - 9, dur - 1], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return <AbsoluteFill style={{ opacity }}>{children}</AbsoluteFill>;
};

export const SecondLifeVideo: React.FC = () => (
  <AbsoluteFill style={{ background: c.cream }}>
    {SCENES.map((s) => {
      const Comp = SCENE_COMPONENTS[s.id];
      return (
        <Sequence key={s.id} from={s.from} durationInFrames={s.dur} name={s.label || "CLOSE"}>
          <SceneWrap dur={s.dur}>
            <Comp />
          </SceneWrap>
        </Sequence>
      );
    })}
    <Chrome />
  </AbsoluteFill>
);
