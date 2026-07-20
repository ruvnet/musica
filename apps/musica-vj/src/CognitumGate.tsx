import { useEffect, useRef } from "react";
import { KeyRound, Loader2, ShieldCheck, Terminal } from "lucide-react";

export interface CognitumGateLogEntry {
  at: string;
  text: string;
  tone?: "info" | "ok" | "warn";
}

interface CognitumGateProps {
  authHost: string;
  pending: boolean;
  busy: boolean;
  reason?: string;
  log: CognitumGateLogEntry[];
  manualCodeMode: boolean;
  manualCode: string;
  signInAvailable: boolean;
  onSignIn: () => void;
  onManualStart: () => void;
  onManualComplete: () => void;
  onManualCodeChange: (value: string) => void;
  onSkip?: () => void;
}

const CYAN = "53,220,255";
const VIOLET = "176,107,242";
const MINT = "117,244,197";

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

// Sets up a DPR-correct 2D canvas that auto-resizes; returns a teardown.
function mountCanvas(
  canvas: HTMLCanvasElement,
  render: (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => void,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return () => {};
  let raf = 0;
  let w = 0;
  let h = 0;
  const resize = () => {
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    w = canvas.clientWidth;
    h = canvas.clientHeight;
    canvas.width = Math.max(1, Math.round(w * ratio));
    canvas.height = Math.max(1, Math.round(h * ratio));
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  };
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  // Ambient canvas motion always runs so the gate reads as alive; under
  // reduced-motion it slows to a calm drift rather than freezing to a frame.
  const speed = prefersReducedMotion() ? 0.4 : 1;
  const start = performance.now();
  const loop = () => {
    render(ctx, w, h, (performance.now() - start) * speed);
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
  return () => {
    cancelAnimationFrame(raf);
    observer.disconnect();
  };
}

// Full-bleed backdrop: slow flowing wave bands + drifting spectral particles.
function GateField() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    // Stable pseudo-particles (index-seeded, no Math.random in render).
    const particles = Array.from({ length: 46 }, (_, i) => ({
      x: (i * 97.13) % 100 / 100,
      y: (i * 51.7) % 100 / 100,
      r: 0.6 + ((i * 7) % 5) * 0.4,
      speed: 0.4 + ((i * 13) % 7) * 0.05,
      hue: i % 3,
    }));
    const bands = [
      { amp: 0.14, freq: 1.4, y: 0.34, speed: 0.00013, color: CYAN, alpha: 0.5 },
      { amp: 0.1, freq: 2.1, y: 0.5, speed: 0.00019, color: VIOLET, alpha: 0.42 },
      { amp: 0.08, freq: 3.0, y: 0.66, speed: 0.00025, color: MINT, alpha: 0.34 },
    ];
    return mountCanvas(canvas, (ctx, w, h, t) => {
      ctx.clearRect(0, 0, w, h);
      // Flowing wave bands.
      for (const b of bands) {
        ctx.beginPath();
        for (let x = 0; x <= w; x += 3) {
          const nx = x / w;
          const y = h * b.y + Math.sin(nx * Math.PI * b.freq + t * b.speed) * h * b.amp;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = `rgba(${b.color},${b.alpha})`;
        ctx.shadowColor = `rgba(${b.color},0.5)`;
        ctx.shadowBlur = 14;
        ctx.stroke();
      }
      ctx.shadowBlur = 0;
      // Drifting particles.
      for (const p of particles) {
        const py = (p.y + (t * 0.00004 * p.speed)) % 1;
        const px = (p.x + Math.sin(t * 0.0002 * p.speed + p.y * 6) * 0.02 + 1) % 1;
        const col = p.hue === 0 ? CYAN : p.hue === 1 ? VIOLET : MINT;
        ctx.beginPath();
        ctx.arc(px * w, py * h, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${col},0.5)`;
        ctx.shadowColor = `rgba(${col},0.7)`;
        ctx.shadowBlur = 8;
        ctx.fill();
      }
      ctx.shadowBlur = 0;
    });
  }, []);
  return <canvas className="cognitum-gate-field" ref={ref} aria-hidden="true" />;
}

// A seamless glowing ribbon above the card: three phase-locked sine layers on a
// period that divides 2π so the motion never visibly seams, plus a travelling
// highlight that tracks left→right and wraps.
function WaveRibbon({ active }: { active: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const PERIOD = 9000;
    return mountCanvas(canvas, (ctx, w, h, elapsed) => {
      const boost = active ? 1.5 : 1.35;
      const phase = ((elapsed % PERIOD) / PERIOD) * Math.PI * 2;
      ctx.clearRect(0, 0, w, h);
      const layers = [
        { amp: 0.44, freq: 2, shift: 0, color: CYAN, alpha: 0.95 },
        { amp: 0.32, freq: 3, shift: Math.PI / 2, color: VIOLET, alpha: 0.72 },
        { amp: 0.22, freq: 4, shift: Math.PI, color: MINT, alpha: 0.52 },
      ];
      for (const layer of layers) {
        ctx.beginPath();
        for (let x = 0; x <= w; x += 2) {
          const nx = x / w;
          const y =
            h / 2 +
            Math.sin(nx * Math.PI * layer.freq + phase + layer.shift) *
              h * 0.5 * layer.amp * boost *
              Math.sin(nx * Math.PI); // taper toward the edges
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.lineWidth = 1.7;
        ctx.strokeStyle = `rgba(${layer.color},${layer.alpha})`;
        ctx.shadowColor = `rgba(${layer.color},0.65)`;
        ctx.shadowBlur = 9;
        ctx.stroke();
      }
      // Travelling highlight dot riding the top layer.
      const hx = ((elapsed % PERIOD) / PERIOD) * w;
      const hy = h / 2 + Math.sin((hx / w) * Math.PI * 2 + phase) * h * 0.5 * 0.44 * boost * Math.sin((hx / w) * Math.PI);
      ctx.beginPath();
      ctx.arc(hx, hy, 3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.shadowColor = `rgba(${CYAN},1)`;
      ctx.shadowBlur = 12;
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  }, [active]);
  return <canvas className="cognitum-gate-wave" ref={ref} aria-hidden="true" />;
}

export function CognitumGate({
  authHost,
  pending,
  busy,
  reason,
  log,
  manualCodeMode,
  manualCode,
  signInAvailable,
  onSignIn,
  onManualStart,
  onManualComplete,
  onManualCodeChange,
  onSkip,
}: CognitumGateProps) {
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [log]);

  const working = busy || pending;

  return (
    <section className="cognitum-gate" role="dialog" aria-modal="true" aria-label="Sign in to Musica">
      <GateField />
      <div className="cognitum-gate-shade" />

      <div className="cognitum-gate-stage">
        <div className="cognitum-gate-wordmark">
          <strong>MUSICA</strong>
          <span>LIVE AI MUSIC + VISUALS</span>
        </div>

        <div className={`cognitum-gate-card ${working ? "working" : ""}`}>
          <WaveRibbon active={working} />
          <div className="cognitum-gate-body">
            <div className="cognitum-gate-mark">
              <ShieldCheck size={14} />
              <span>STEP 01 · SIGN IN</span>
            </div>
            <h1>Sign in with Cognitum One</h1>
            <p>
              One account starts your set — it authorizes Lyria audio and unlocks the
              state-of-the-art AI enhancements.
            </p>

            <button
              type="button"
              className="cognitum-gate-primary"
              onClick={onSignIn}
              disabled={busy || pending || !signInAvailable}
            >
              <span className="cognitum-gate-primary-label">
                {pending ? <Loader2 size={16} className="spin" /> : <ShieldCheck size={16} />}
                {pending ? "WAITING FOR BROWSER…" : "SIGN IN WITH COGNITUM ONE"}
              </span>
            </button>

            {signInAvailable && (!manualCodeMode ? (
              <button type="button" className="cognitum-gate-manual-toggle" onClick={onManualStart} disabled={busy}>
                <KeyRound size={12} /> PASTE A CODE INSTEAD (HEADLESS / SSH)
              </button>
            ) : (
              <div className="cognitum-gate-manual">
                <input
                  value={manualCode}
                  placeholder="CGN-XXXXXX"
                  maxLength={64}
                  spellCheck={false}
                  onChange={(event) => onManualCodeChange(event.target.value)}
                  aria-label="Cognitum manual sign-in code"
                />
                <button type="button" onClick={onManualComplete} disabled={busy || !manualCode.trim()}>
                  CONNECT
                </button>
              </div>
            ))}

            {!signInAvailable && (
              <small className="cognitum-gate-reason">{reason ?? "Cognitum sign-in requires the desktop app."}</small>
            )}
            {signInAvailable && reason && <small className="cognitum-gate-reason">{reason}</small>}

            <div className="cognitum-gate-log">
              <div className="cognitum-gate-log-head">
                <Terminal size={11} /> <span>LOG</span>
                <i className="cognitum-gate-log-dot" />
              </div>
              <div className="cognitum-gate-log-lines" ref={logRef} role="log" aria-live="polite">
                {log.map((entry, index) => (
                  <p key={index} className={entry.tone ?? "info"}>
                    <b>{entry.at}</b> {entry.text}
                  </p>
                ))}
              </div>
            </div>

            {onSkip && (
              <button type="button" className="cognitum-gate-skip" onClick={onSkip}>
                Continue without an account
              </button>
            )}
          </div>
        </div>

        <div className="cognitum-gate-host">{authHost}</div>
      </div>
    </section>
  );
}
