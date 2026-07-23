import React from "react";
import { Img, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Stage } from "../components/Chrome";
import { Label, Reveal } from "../components/primitives";
import { c, f } from "../theme";

/**
 * Real near-duplicate pairs from TrashNet, taken straight from the committed
 * grouping in model/data/splits/groups.csv. Each row is one physical object that
 * the dataset stores as two separate images under two unrelated filenames.
 *
 *   groups.csv group 2078 → plastic293.jpg + plastic384.jpg
 *   groups.csv group  556 → glass24.jpg    + glass74.jpg
 *   groups.csv group  911 → metal11.jpg    + metal221.jpg
 *
 * These must stay real: a fabricated example would be exactly the sin the video
 * is accusing other people of.
 */
const PAIRS = [
  { a: "plastic293.jpg", b: "plastic384.jpg", label: "plastic" },
  { a: "glass24.jpg", b: "glass74.jpg", label: "glass" },
  { a: "metal11.jpg", b: "metal221.jpg", label: "metal" },
];

const TILE = 140;
const NAME_W = 232;
const MID = 190;
const ROW_GAP = 18;
const ROW_W = TILE * 2 + NAME_W * 2 + MID + 28;

const Tile: React.FC<{ file: string }> = ({ file }) => (
  <div
    style={{
      width: TILE,
      height: TILE,
      borderRadius: 6,
      overflow: "hidden",
      border: `1px solid ${c.rule}`,
      background: c.creamDeep,
      flex: `0 0 ${TILE}px`,
    }}
  >
    <Img
      src={staticFile(`pairs/${file}`)}
      style={{ width: "100%", height: "100%", objectFit: "cover" }}
    />
  </div>
);

export const S2Leak: React.FC = () => {
  const frame = useCurrentFrame();

  const cut = interpolate(frame, [104, 128], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // The connector greys out once the split severs it.
  const severed = interpolate(frame, [120, 140], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const rowsH = PAIRS.length * TILE + (PAIRS.length - 1) * ROW_GAP;

  return (
    <Stage>
      <Reveal delay={0}>
        <Label>Anatomy of a fake number</Label>
        <div
          style={{
            fontFamily: f.serif,
            fontWeight: 900,
            fontSize: 62,
            lineHeight: 1.2,
            marginTop: 12,
          }}
        >
          The same object, photographed twice.
        </div>
      </Reveal>

      <div
        style={{
          position: "relative",
          width: ROW_W,
          margin: "38px auto 0",
        }}
      >
        {/* column headers */}
        <div style={{ display: "flex", height: 40, alignItems: "center" }}>
          <Reveal
            delay={132}
            y={10}
            style={{ width: TILE + NAME_W + 14, textAlign: "center" }}
          >
            <Label color={c.green}>Train</Label>
          </Reveal>
          <div style={{ width: MID }} />
          <Reveal
            delay={140}
            y={10}
            style={{ width: TILE + NAME_W + 14, textAlign: "center" }}
          >
            <Label color={c.alarm}>Validation</Label>
          </Reveal>
        </div>

        <div style={{ position: "relative" }}>
          {PAIRS.map((p, i) => (
            <Reveal
              key={p.a}
              delay={10 + i * 9}
              y={26}
              style={{
                display: "flex",
                alignItems: "center",
                marginBottom: i === PAIRS.length - 1 ? 0 : ROW_GAP,
              }}
            >
              <Tile file={p.a} />
              <div
                style={{
                  width: NAME_W,
                  paddingLeft: 14,
                  fontFamily: f.mono,
                  fontSize: 21,
                  color: c.inkSoft,
                }}
              >
                {p.a}
                <div style={{ color: c.goldPale, fontSize: 18, marginTop: 4 }}>
                  {p.label}
                </div>
              </div>

              {/* connector */}
              <div
                style={{
                  width: MID,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 7,
                }}
              >
                <div
                  style={{
                    fontFamily: f.mono,
                    fontSize: 17,
                    letterSpacing: "0.12em",
                    color: severed > 0.5 ? c.goldPale : c.gold,
                  }}
                >
                  SAME OBJECT
                </div>
                <div
                  style={{
                    width: "100%",
                    borderTop: `2px dashed ${
                      severed > 0.5 ? c.rule : c.gold
                    }`,
                  }}
                />
              </div>

              <div
                style={{
                  width: NAME_W,
                  paddingRight: 14,
                  textAlign: "right",
                  fontFamily: f.mono,
                  fontSize: 21,
                  color: c.inkSoft,
                }}
              >
                {p.b}
                <div style={{ color: c.goldPale, fontSize: 18, marginTop: 4 }}>
                  {p.label}
                </div>
              </div>
              <Tile file={p.b} />
            </Reveal>
          ))}

          {/* the split slices every connector at once */}
          <div
            style={{
              position: "absolute",
              left: ROW_W / 2 - 1.5,
              top: -26,
              height: rowsH + 52,
              width: 3,
              background: c.alarm,
              transform: `scaleY(${cut})`,
              transformOrigin: "top",
            }}
          />
        </div>
      </div>

      <Reveal delay={182} y={24} style={{ marginTop: 30 }}>
        <div
          style={{
            fontFamily: f.sans,
            fontSize: 35,
            lineHeight: 1.58,
            color: c.inkSoft,
            maxWidth: 1560,
          }}
        >
          Unrelated filenames, so a random split cannot tell they belong together.
          The model trains on the left and is graded on the right — on objects it
          has already memorised.{" "}
          <span style={{ color: c.ink, fontWeight: 700 }}>That is the 99%.</span>
        </div>
      </Reveal>
    </Stage>
  );
};
