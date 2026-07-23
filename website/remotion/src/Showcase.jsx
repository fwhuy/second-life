import React from 'react';
import {
  AbsoluteFill,
  Img,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

const C = {
  bg: '#111714',
  cream: '#FAF7EF',
  forest: '#1E3A2B',
  panel: 'rgba(30,58,43,.78)',
  line: 'rgba(216,199,168,.28)',
  text: '#FAF7EF',
  muted: '#D8C7A8',
  green: '#69D6AC',
  green2: '#3E7C4F',
  red: '#D97B29',
  amber: '#D97B29',
};

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'};
const ease = (f, a, b) => interpolate(f, [a, b], [0, 1], clamp);
const enter = (f, delay = 0, distance = 28) => ({
  opacity: ease(f, delay, delay + 18),
  transform: `translateY(${interpolate(f, [delay, delay + 22], [distance, 0], clamp)}px)`,
});

const Grid = () => (
  <AbsoluteFill
    style={{
      opacity: 0.22,
      backgroundImage:
        'linear-gradient(rgba(105,214,172,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(105,214,172,.12) 1px,transparent 1px)',
      backgroundSize: '54px 54px',
      maskImage: 'radial-gradient(circle at center,black,transparent 82%)',
    }}
  />
);

const Noise = () => (
  <AbsoluteFill
    style={{
      pointerEvents: 'none',
      opacity: 0.035,
      backgroundImage:
        'url("data:image/svg+xml,%3Csvg viewBox=%220 0 180 180%22 xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cfilter id=%22n%22%3E%3CfeTurbulence type=%22fractalNoise%22 baseFrequency=%22.9%22 numOctaves=%223%22 stitchTiles=%22stitch%22/%3E%3C/filter%3E%3Crect width=%22100%25%22 height=%22100%25%22 filter=%22url(%23n)%22/%3E%3C/svg%3E")',
    }}
  />
);

const Glass = ({children, style = {}}) => (
  <div
    style={{
      background: `linear-gradient(145deg,${C.panel},rgba(17,23,20,.82))`,
      color: C.text,
      border: `1px solid ${C.line}`,
      borderRadius: 24,
      boxShadow: '0 30px 90px rgba(0,0,0,.35), inset 0 1px rgba(255,255,255,.05)',
      backdropFilter: 'blur(18px)',
      ...style,
    }}
  >
    {children}
  </div>
);

const Kicker = ({children}) => (
  <div style={{fontSize: 14, fontWeight: 800, letterSpacing: 4, color: C.green, textTransform: 'uppercase'}}>
    {children}
  </div>
);

const Pill = ({children, color = C.green}) => (
  <span
    style={{
      display: 'inline-block',
      border: `1px solid ${color}66`,
      color,
      background: `${color}12`,
      borderRadius: 999,
      padding: '9px 14px',
      fontSize: 15,
      fontWeight: 700,
    }}
  >
    {children}
  </span>
);

const Terminal = ({frame, opacity = 1}) => {
  const lines = [
    'groups = build_duplicate_groups(phash_hamming=6)',
    'assert train_groups.isdisjoint(valid_groups)',
    'cosine_similarity(features_a, features_b) >= 0.96',
    'quarantine.add(source_match.sha256)',
    'model = convnextv2_tiny(num_classes=6)',
    'guard.score(penultimate_features)',
    'return {"class": label, "confidence": confidence}',
    'audit.hard_class("trash", recall=20/23)',
  ];
  return (
    <AbsoluteFill style={{opacity, overflow: 'hidden'}}>
      <div
        style={{
          position: 'absolute',
          inset: '-100px 50px',
          transform: `translateY(${-((frame * 0.65) % 58)}px)`,
          font: '18px ui-monospace, SFMono-Regular, Menlo, monospace',
          lineHeight: '58px',
          color: 'rgba(105,214,172,.17)',
        }}
      >
        {Array.from({length: 28}, (_, i) => (
          <div key={i}>
            <span style={{color: 'rgba(255,255,255,.08)', marginRight: 24}}>{String(i + 41).padStart(3, '0')}</span>
            {lines[i % lines.length]}
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

const Scene0 = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const selected = spring({frame: f - 66, fps, config: {damping: 17, stiffness: 110}});
  const fadeOthers = interpolate(f, [55, 90], [1, 0], clamp);
  const cards = [
    '01. Dynamic Dialog Box System',
    '02. Second-Hand Product Detection',
    '03. Second Life AI: Waste & Future',
  ];
  return (
    <AbsoluteFill style={{background: `radial-gradient(circle at 50% 45%,${C.forest},${C.bg} 70%)`, color: C.text}}>
      <Terminal frame={f} />
      <Grid />
      <div style={{position: 'absolute', inset: '80px 0', display: 'grid', placeItems: 'center'}}>
        <div style={{width: 850}}>
          {cards.map((label, i) => {
            const active = i === 2;
            return (
              <Glass
                key={label}
                style={{
                  marginBottom: 13,
                  padding: '24px 30px',
                  fontSize: 23,
                  fontWeight: 650,
                  color: active ? C.text : C.muted,
                  opacity: active ? 1 : fadeOthers * 0.48,
                  filter: active ? 'none' : 'blur(1.8px)',
                  borderColor: active ? `${C.green}88` : C.line,
                  boxShadow: active ? `0 0 55px ${C.green}24` : 'none',
                  transform: active ? `scale(${1 + selected * 0.075})` : 'none',
                }}
              >
                {label}
              </Glass>
            );
          })}
        </div>
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, bottom: 102, textAlign: 'center', ...enter(f, 105)}}>
        <Kicker>Selected direction</Kicker>
        <div style={{fontSize: 24, marginTop: 11, fontWeight: 600}}>
          Beyond Interface. Beyond Commerce. Designing the Future of Waste.
        </div>
      </div>
    </AbsoluteFill>
  );
};

const BottleTile = ({i}) => (
  <div
    style={{
      width: 126,
      height: 126,
      overflow: 'hidden',
      borderRadius: 16,
      border: `1px solid ${C.green}55`,
      background: '#171A1A',
    }}
  >
    <Img
      src={staticFile('bottle.jpeg')}
      style={{
        width: '100%',
        height: '100%',
        objectFit: 'cover',
        transform: `scale(${1.06 + i * 0.035}) rotate(${[-3, 2, -1, 4][i]}deg)`,
        filter: `brightness(${0.86 + i * 0.05})`,
      }}
    />
  </div>
);

const Scene1 = () => {
  const f = useCurrentFrame();
  const warning = ease(f, 95, 115);
  return (
    <AbsoluteFill style={{background: `linear-gradient(135deg,${C.bg},${C.forest})`, color: C.text}}>
      <Terminal frame={f + 180} opacity={0.65} />
      <div style={{position: 'absolute', inset: '80px 56px', display: 'grid', gridTemplateColumns: '1.2fr .8fr', gap: 22}}>
        <Glass style={{padding: 30, ...enter(f, 0)}}>
          <Kicker>Group-aware split firewall</Kicker>
          <div style={{display: 'flex', gap: 10, margin: '18px 0 23px'}}>
            <Pill>Perceptual Hash ≤ 6</Pill>
            <Pill>ResNet Cosine Similarity ≥ 0.96</Pill>
          </div>
          <div style={{display: 'flex', gap: 12}}>
            {[0, 1, 2, 3].map((i) => <BottleTile i={i} key={i} />)}
          </div>
          <div style={{marginTop: 18, color: C.muted, fontSize: 16}}>
            One visual group → one split. No duplicate leakage.
          </div>
        </Glass>
        <Glass style={{padding: 26, ...enter(f, 24)}}>
          <Kicker>OOD diagnostic</Kicker>
          <div style={{display: 'flex', gap: 20, alignItems: 'center', marginTop: 18}}>
            <Img src={staticFile('cat.jpg')} style={{width: 158, height: 158, borderRadius: 18, objectFit: 'cover'}} />
            <div>
              <div style={{fontSize: 15, color: C.muted}}>Penultimate-layer distance</div>
              <div style={{fontSize: 38, fontWeight: 800, margin: '7px 0'}}>Cat 0.78</div>
              <div style={{fontSize: 18, color: C.green}}>Bottle 0.30</div>
            </div>
          </div>
          <div
            style={{
              marginTop: 18,
              padding: '16px 17px',
              borderRadius: 14,
              background: `${C.amber}16`,
              border: `1px solid ${C.amber}`,
              color: C.amber,
              font: '700 15px ui-monospace, monospace',
              opacity: warning,
              transform: `scale(${0.96 + warning * 0.04})`,
            }}
          >
            ⚠ UNFAMILIAR INPUT DETECTED (&gt;0.70 Threshold)
          </div>
        </Glass>
      </div>
      <div style={{position: 'absolute', bottom: 95, left: 0, right: 0, textAlign: 'center', fontSize: 22, fontWeight: 650}}>
        Data Safeguards &amp; Out-of-Distribution Precision.
      </div>
    </AbsoluteFill>
  );
};

const Transition = () => {
  const f = useCurrentFrame();
  const burst = ease(f, 0, 58);
  return (
    <AbsoluteFill style={{background: `radial-gradient(circle at center,${C.green2},${C.forest} 52%,${C.bg})`, color: C.text, display: 'grid', placeItems: 'center'}}>
      {Array.from({length: 38}, (_, i) => {
        const angle = (i / 38) * Math.PI * 2;
        const r = burst * (170 + (i % 7) * 58);
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: 640 + Math.cos(angle) * r,
              top: 360 + Math.sin(angle) * r,
              width: 4 + (i % 3) * 2,
              height: 4 + (i % 3) * 2,
              borderRadius: '50%',
              background: i % 4 ? C.green : C.text,
              opacity: 1 - burst * 0.7,
              boxShadow: `0 0 18px ${C.green}`,
            }}
          />
        );
      })}
      <div style={{fontSize: 50, fontWeight: 760, letterSpacing: -1.5, transform: `scale(${0.88 + burst * 0.12})`}}>
        Full-Stack System Architecture.
      </div>
    </AbsoluteFill>
  );
};

const Node = ({title, sub, x, delay, frame}) => {
  const s = spring({frame: frame - delay, fps: 30, config: {damping: 16}});
  return (
    <Glass style={{width: 305, padding: '27px 24px', transform: `translateY(${(1 - s) * 26}px)`, opacity: s}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 12}}>
        <div style={{width: 12, height: 12, borderRadius: '50%', background: C.green, boxShadow: `0 0 20px ${C.green}`}} />
        <div style={{fontSize: 21, fontWeight: 750}}>{title}</div>
      </div>
      <div style={{color: C.muted, marginTop: 10, fontSize: 16}}>{sub}</div>
    </Glass>
  );
};

const Scene2 = () => {
  const f = useCurrentFrame();
  const count = Math.round(interpolate(f, [115, 235], [0, 11299], clamp)).toLocaleString('en-US');
  const acc = interpolate(f, [225, 285], [0, 95.62], clamp).toFixed(2);
  const rejected = spring({frame: f - 294, fps: 30, config: {damping: 10, stiffness: 180}});
  const pulse = ((f * 7) % 770) - 30;
  return (
    <AbsoluteFill style={{background: `radial-gradient(circle at 50% 5%,#FFFFFF,${C.cream} 48%,#F0E9DA)`, color: C.forest, padding: '48px 62px'}}>
      <Grid />
      <div style={{...enter(f, 0)}}>
        <Kicker>System architecture</Kicker>
        <div style={{fontSize: 42, fontWeight: 760, marginTop: 10}}>Three layers. One trustworthy loop.</div>
      </div>
      <div style={{position: 'relative', display: 'flex', gap: 76, marginTop: 48}}>
        <div style={{position: 'absolute', left: 100, right: 100, top: 58, height: 2, background: C.line}}>
          <div style={{width: 70, height: 2, background: C.green, transform: `translateX(${pulse}px)`, boxShadow: `0 0 14px ${C.green}`}} />
        </div>
        <Node frame={f} delay={25} title="Client Layer" sub="React / Vite + Tailwind CSS" />
        <Node frame={f} delay={38} title="Server Layer" sub="Flask RESTful API" />
        <Node frame={f} delay={51} title="AI Engine" sub="ConvNeXtV2 Tiny · <30M parameters" />
      </div>
      <div style={{display: 'grid', gridTemplateColumns: '1fr 1.18fr', gap: 22, marginTop: 30}}>
        <Glass style={{padding: 26, ...enter(f, 105)}}>
          <div style={{color: C.muted, fontSize: 16}}>Images cleansed across 5 public sources</div>
          <div style={{fontSize: 59, fontWeight: 820, letterSpacing: -3, marginTop: 8}}>{count}+</div>
        </Glass>
        <Glass style={{padding: 26, position: 'relative', overflow: 'hidden', ...enter(f, 205)}}>
          <div style={{color: C.muted}}>Candidate accuracy</div>
          <div style={{fontSize: 51, fontWeight: 820, marginTop: 5}}>{acc}%</div>
          <div
            style={{
              position: 'absolute',
              right: 25,
              top: 25,
              padding: '14px 21px',
              border: `4px solid ${C.red}`,
              color: C.red,
              fontSize: 25,
              fontWeight: 900,
              letterSpacing: 3,
              transform: `rotate(-7deg) scale(${rejected})`,
            }}
          >
            ✕ REJECTED
          </div>
          <div style={{fontSize: 15, color: C.cream, marginTop: 10, opacity: ease(f, 312, 330)}}>
            Trash recall fell 20/23 → 18/23. Zero compromise on hard-to-classify waste.
          </div>
        </Glass>
      </div>
      <div style={{position: 'absolute', bottom: 31, left: 62, color: '#6B786F', fontSize: 18}}>
        ConvNeXtV2 Tiny Engine. Uncompromising Engineering Rigor.
      </div>
    </AbsoluteFill>
  );
};

const Phone = ({frame}) => {
  const upload = ease(frame, 20, 55);
  const analyzed = ease(frame, 90, 125);
  return (
    <div style={{width: 320, height: 585, borderRadius: 42, padding: 12, background: '#050606', border: '1px solid #3E4542', boxShadow: '0 35px 100px #000'}}>
      <div style={{height: '100%', borderRadius: 31, overflow: 'hidden', background: '#F6F2E8', color: '#243329', position: 'relative'}}>
        <div style={{height: 54, display: 'flex', alignItems: 'center', padding: '0 20px', fontWeight: 900, color: '#2F5A3C'}}>
          SECOND LIFE <span style={{fontWeight: 400, marginLeft: 6}}>AI</span>
        </div>
        <div style={{margin: '0 15px', height: 245, borderRadius: 20, background: '#DCE7D3', overflow: 'hidden', position: 'relative'}}>
          <Img src={staticFile('bottle.jpeg')} style={{width: '100%', height: '100%', objectFit: 'cover', opacity: upload}} />
          <div style={{position: 'absolute', inset: 0, display: upload > 0.1 ? 'none' : 'grid', placeItems: 'center', color: '#607064'}}>Snap or upload</div>
          <div style={{position: 'absolute', inset: '18px 36px', border: '2px solid #69D6AC', opacity: analyzed}} />
        </div>
        <div style={{padding: '17px 20px'}}>
          <div style={{fontSize: 12, color: '#6B786F', fontWeight: 700, letterSpacing: 2}}>INSIGHT LAB</div>
          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'end', marginTop: 7, opacity: analyzed}}>
            <div style={{fontSize: 30, fontWeight: 850}}>Plastic</div>
            <div style={{fontSize: 28, fontWeight: 850, color: '#3E7C4F'}}>98%</div>
          </div>
          <div style={{height: 8, borderRadius: 9, background: '#D9D6CB', marginTop: 10, overflow: 'hidden'}}>
            <div style={{width: `${analyzed * 98}%`, height: '100%', background: '#3E7C4F'}} />
          </div>
          <div style={{fontSize: 14, lineHeight: 1.45, marginTop: 16, opacity: analyzed, color: '#566158'}}>
            Durable, lightweight, and derived from fossil feedstocks. Its next chapter depends on your choice.
          </div>
        </div>
      </div>
    </div>
  );
};

const Future = ({good, frame}) => {
  const p = ease(frame, 260, 390);
  const points = good
    ? ['Correct sorting', 'Material recovery', 'Resource rebirth']
    : ['Incorrect disposal', 'Landfill accumulation', '450 years + microplastics'];
  return (
    <Glass style={{padding: 24, borderColor: good ? `${C.green}66` : `${C.red}55`, background: good ? 'rgba(24,72,54,.52)' : 'rgba(48,26,29,.52)'}}>
      <div style={{display: 'flex', justifyContent: 'space-between'}}>
        <Kicker>{good ? 'Circular future' : 'Linear future'}</Kicker>
        <div style={{fontSize: 30}}>{good ? '♻' : '↘'}</div>
      </div>
      {points.map((pnt, i) => (
        <div key={pnt} style={{display: 'flex', alignItems: 'center', gap: 12, marginTop: 19, opacity: ease(frame, 275 + i * 25, 295 + i * 25)}}>
          <div style={{width: 9, height: 9, borderRadius: '50%', background: good ? C.green : C.red, boxShadow: `0 0 16px ${good ? C.green : C.red}`}} />
          <div style={{fontSize: 18}}>{pnt}</div>
        </div>
      ))}
      <div style={{height: 3, marginTop: 28, background: C.line, overflow: 'hidden'}}>
        <div style={{width: `${p * 100}%`, height: '100%', background: good ? C.green : C.red}} />
      </div>
    </Glass>
  );
};

const Scene3 = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{background: `radial-gradient(circle at 20% 45%,#DDE8DF,${C.cream} 54%,#F0E9DA)`, color: C.forest, padding: '42px 60px'}}>
      <Grid />
      <div style={{display: 'grid', gridTemplateColumns: '360px 1fr', gap: 45, height: '100%'}}>
        <div style={{...enter(f, 0), display: 'grid', placeItems: 'center'}}><Phone frame={f} /></div>
        <div style={{paddingTop: 34}}>
          <Kicker>Live platform walkthrough</Kicker>
          <div style={{fontSize: 48, fontWeight: 780, lineHeight: 1.05, margin: '12px 0 24px'}}>Snap. Analyze.<br />Simulate Two Futures.</div>
          <div style={{display: 'flex', gap: 10, marginBottom: 23}}>
            <Pill>01 Upload</Pill><Pill>02 Insight Lab</Pill><Pill>03 Fork in the Road</Pill>
          </div>
          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, opacity: ease(f, 225, 255)}}>
            <Future good frame={f} />
            <Future frame={f} />
          </div>
          <div style={{marginTop: 19, color: '#6B786F', fontSize: 16, opacity: ease(f, 425, 455)}}>
            One object. One decision. Two radically different material stories.
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const TypeLine = ({text, frame, start, style = {}}) => {
  const chars = Math.floor(interpolate(frame, [start, start + text.length * 2], [0, text.length], clamp));
  return <div style={style}>{text.slice(0, chars)}<span style={{opacity: frame % 20 < 10 ? 1 : 0}}>▌</span></div>;
};

const Scene4 = () => {
  const f = useCurrentFrame();
  const fade = interpolate(f, [0, 25, 270, 300], [0, 1, 1, 0], clamp);
  return (
    <AbsoluteFill style={{background: `radial-gradient(circle at center,${C.forest},${C.bg} 72%)`, color: C.text, display: 'grid', placeItems: 'center', opacity: fade}}>
      <Grid />
      <div style={{textAlign: 'center', width: 1000}}>
        <TypeLine text="Trustworthy AI. Empathetic Design." frame={f} start={28} style={{fontSize: 35, color: C.muted, minHeight: 52}} />
        <TypeLine text="Second Life AI." frame={f} start={112} style={{fontSize: 82, fontWeight: 820, letterSpacing: -4, minHeight: 112}} />
        <div style={{width: 90, height: 3, background: C.green, margin: '25px auto 31px', transform: `scaleX(${ease(f, 155, 190)})`}} />
        <div style={{display: 'flex', justifyContent: 'center', gap: 32, fontSize: 17, color: '#B6BFBB', opacity: ease(f, 175, 205)}}>
          <span>github.com/fwhuy/second-life</span>
          <span style={{color: C.green}}>•</span>
          <span>bit.ly/second-life-ai</span>
        </div>
        <div style={{marginTop: 17, fontSize: 15, color: C.muted, letterSpacing: 2, opacity: ease(f, 195, 225)}}>
          NYU SHANGHAI AI SUMMER PROGRAM
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Letterbox = () => {
  const frame = useCurrentFrame();
  const height = interpolate(frame, [360, 420], [80, 0], clamp);
  return (
    <>
      <div style={{position: 'absolute', zIndex: 100, left: 0, right: 0, top: 0, height, background: '#000'}} />
      <div style={{position: 'absolute', zIndex: 100, left: 0, right: 0, bottom: 0, height, background: '#000'}} />
    </>
  );
};

export const WebsiteShowcase = () => (
  <AbsoluteFill style={{fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', background: '#000'}}>
    <Sequence durationInFrames={180}><Scene0 /></Sequence>
    <Sequence from={180} durationInFrames={180}><Scene1 /></Sequence>
    <Sequence from={360} durationInFrames={60}><Transition /></Sequence>
    <Sequence from={420} durationInFrames={420}><Scene2 /></Sequence>
    <Sequence from={840} durationInFrames={660}><Scene3 /></Sequence>
    <Sequence from={1500} durationInFrames={300}><Scene4 /></Sequence>
    <Letterbox />
    <Noise />
  </AbsoluteFill>
);
