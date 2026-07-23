# Second Life AI — 60-second elevator-pitch video

A silent, exactly-60.000s 1920×1080 backdrop for the **1-minute elevator pitch** slot of
the defense. It is designed to be *narrated live*, not to talk for you: on-screen text
carries anchors and evidence, the presenter carries the argument.

Built with [Remotion](https://remotion.dev) (React → MP4).

## Run it

```bash
npm install
npm start          # Remotion Studio — live preview, scrub, edit
npm run build      # renders out/second-life-ai.mp4 (silent, exactly 60.000s)
```

`npm start` opens a browser studio. Edits to `src/` hot-reload.

## The spine — "Two Numbers"

The video argues one thing: **the distance between a fake 100% and an honest 94.93%.**
This matters because the track is ranked on raw test accuracy, so the 89.20% test figure
will sit below teams whose splits leak. The video makes that case before a judge can
silently mark it down.

| # | Scene | Frames | Beat |
|---|---|---|---|
| 01 | THE CLAIM | 0–170 | 99–100% is not a result, it is usually a leak |
| 02 | THE LEAK | 170–470 | Three real near-duplicate pairs, split across train/validation |
| 03 | THE DISCOVERY | 470–770 | 2,218 renamed images; 361/361 spent-test recovered |
| 04 | THE FIREWALL | 770–1010 | Five gates, enforced at load time |
| 05 | RESULTS | 1010–1250 | Confusion matrix + 93.09 / 94.93 / 89.20 |
| 06 | ABLATION | 1250–1430 | Three pre-gated experiments, all rejected |
| 07 | LIMITS | 1430–1610 | Label noise (14/30) and OOD bottle-vs-cat |
| 08 | PRODUCT | 1610–1740 | Insight Lab → Fork in the Road → Museum |
| 09 | close | 1740–1800 | 万物皆有新生 |

Scene boundaries live in one place — `src/theme.ts` → `SCENES`. Changing a `dur` there
reflows everything; keep the total at `DURATION` (1800) so the file stays exactly 60s.

## Structure

```
src/
├── theme.ts              palette, type stack, scene table — the single source of truth
├── Root.tsx              registers the SecondLifeAI composition
├── Video.tsx             sequences the scenes, handles scene-to-scene fades
├── components/
│   ├── Chrome.tsx        persistent brand, position marker, progress rule + Stage
│   ├── Exhibit.tsx       the dark "evidence" panel
│   └── primitives.tsx    Reveal, Label, CountUp, DrawRule
└── scenes/               S1Claim … S9Close, one file per beat
public/
├── pairs/                three real TrashNet near-duplicate pairs (scene 02)
├── bottle.jpeg           in-distribution example (scene 07)
└── cat.jpg               out-of-distribution example (scene 07)
```

The **progress rule** along the bottom is functional, not decoration — "strict time
control" is graded, and it lets the presenter see their pace without a clock.

## Design rules

- Two registers only. **Cream/green editorial** is the voice; the **dark panel** means
  *this claim has a receipt* (a guard file, an assertion, a committed report). Never use
  the dark register for decoration.
- Colours come from `theme.ts`, which mirrors `website/index.html`. Don't introduce new ones.
- Nothing below 22px — the poster session is judged on legibility at 2 m.
- Scene kickers must not repeat the top-right marker.

## Numbers on screen

All figures trace to committed artifacts, and the claims follow
`PROJECT_PROCESS_README.md` §13 ("claims that are safe to make"):

- Confusion matrix — `model/reports/baseline_resnet50/confusion_baseline_resnet50_val_fold0_tta.png`.
  Rows sum to 434, diagonal to 412 = 94.93%.
- The scene-02 duplicate pairs are **real rows from `model/data/splits/groups.csv`** —
  groups 2078 (plastic293/plastic384), 556 (glass24/glass74) and 911 (metal11/metal221),
  extracted from `trashnet-train-FINAL-20260720.zip` and centre-cropped to 480×480.
  There are 37 same-label multi-image groups in TrashNet; these are three of them.
  **Do not substitute mock-ups here** — a fabricated duplicate would be exactly the sin
  the video accuses others of, and it is the first thing a judge could check.
- 93.09 / 94.93 / 89.20 and the three rejected experiments — `model/FINAL_SUMMARY.md`.
- 2,218 renamed, 361/361 quarantined, 318 under a foreign filename, the 14/30 audit,
  and the 11,299-image corpus — `PROJECT_PROCESS_README.md` §5.1, §5.4, §13.

**The 89.20% test figure is captioned on screen as a single pre-registered measurement,
never re-run.** Keep that caption if you edit the scene. The unified-corpus beat says
"model not yet trained" for the same reason — the corpus is measured, the model isn't.

## Gotcha

`typescript` is pinned to `^5.9`. TypeScript 7.x (the native port) drops the `ts.sys`
API that Remotion's webpack loader calls, and bundling fails with
`Cannot read properties of undefined (reading 'readFile')`. Don't let it drift up.
