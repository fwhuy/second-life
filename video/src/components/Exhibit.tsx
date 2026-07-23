import React from "react";
import { c, f } from "../theme";
import { Reveal } from "./primitives";

/**
 * The dark "evidence" register. Used only where a claim is backed by a committed
 * artifact (a guard file, an assertion, a report) so the visual shift always means
 * the same thing: this is the receipt.
 */
export const Exhibit: React.FC<{
  title: string;
  delay?: number;
  width?: number | string;
  style?: React.CSSProperties;
  children: React.ReactNode;
}> = ({ title, delay = 0, width, style, children }) => (
  <Reveal delay={delay} y={34} style={{ width }}>
    <div
      style={{
        background: c.dark,
        borderRadius: 10,
        padding: "30px 34px 34px",
        boxShadow: "0 22px 60px rgba(35,55,43,0.26)",
        ...style,
      }}
    >
      <div
        style={{
          fontFamily: f.mono,
          fontSize: 21,
          letterSpacing: "0.2em",
          color: c.darkSage,
          paddingBottom: 14,
          borderBottom: `1px solid ${c.darkRule}`,
          marginBottom: 22,
        }}
      >
        {title}
      </div>
      {children}
    </div>
  </Reveal>
);
