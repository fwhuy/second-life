/**
 * Design tokens lifted from website/index.html so the video, the site and the
 * poster read as one system. Do not introduce colours outside this file.
 */

export const c = {
  cream: "#FAF7EF",
  creamDeep: "#F2EDE0",
  ink: "#23372B",
  inkSoft: "#4A5A4B",
  green: "#2F5A3C",
  greenBright: "#3E7C4F",
  sage: "#7C8A73",
  gold: "#A9A183",
  goldPale: "#C9BD9F",
  rule: "#DED5BE",
  alarm: "#C4602F",
  alarmDeep: "#A8542A",
  amber: "#D97B29",
  // dark "exhibit" register — used only for evidence panels
  dark: "#12160F",
  darkRule: "#3A4436",
  darkInk: "#E8E4D6",
  darkSoft: "#6E7A66",
  darkSage: "#8FA98B",
} as const;

export const f = {
  serif:
    "'Songti SC', 'STSong', 'Noto Serif SC', Georgia, 'Times New Roman', serif",
  sans: "'PingFang SC', 'Noto Sans SC', system-ui, -apple-system, 'Helvetica Neue', sans-serif",
  mono: "'SF Mono', Menlo, Monaco, ui-monospace, monospace",
} as const;

/** 1920x1080 canvas. Nothing below 22px — the poster rubric wants legibility at 2m. */
export const SAFE_X = 132;
export const SAFE_Y = 96;

export const FPS = 30;
export const DURATION = 1800; // exactly 60.000s

/** Scene table. `from` is absolute; markers double as the presenter's pacing cues. */
export const SCENES = [
  { id: "claim", label: "THE CLAIM", from: 0, dur: 170 },
  { id: "leak", label: "THE LEAK", from: 170, dur: 300 },
  { id: "discovery", label: "THE DISCOVERY", from: 470, dur: 300 },
  { id: "firewall", label: "THE FIREWALL", from: 770, dur: 240 },
  { id: "results", label: "RESULTS", from: 1010, dur: 240 },
  { id: "ablation", label: "ABLATION", from: 1250, dur: 180 },
  { id: "limits", label: "LIMITS", from: 1430, dur: 180 },
  { id: "product", label: "PRODUCT", from: 1610, dur: 130 },
  { id: "close", label: "", from: 1740, dur: 60 },
] as const;
