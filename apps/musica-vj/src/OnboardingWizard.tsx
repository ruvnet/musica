import { useEffect, useRef, useState } from "react";
import { Aperture, ArrowLeft, ArrowRight, AudioLines, Box, Check, ChevronDown, Pause, Play, SlidersHorizontal, Sparkles, Volume2, Zap } from "lucide-react";
import * as THREE from "three";
import { VISUAL_SCENES } from "./core/presets";
import { LYRIA_REALTIME_STYLE_PRESETS, lyriaRealtimeStyleById } from "./core/lyriaRealtime";
import { VISUAL_ANIMATION_STYLES, VISUAL_COLOR_PALETTES } from "./visual/VisualEngine";
import {
  bpmForPace,
  normalizeOnboardingPreferences,
  type MusicFormat,
  type MusicPace,
  type OnboardingPreferences,
  type VocalRole,
  type VocalStyle,
} from "./core/onboarding";

export type OnboardingView = "welcome" | "setup" | "launching";

interface OnboardingWizardProps {
  view: OnboardingView;
  firstRun: boolean;
  preferences: OnboardingPreferences;
  lyriaAvailable: boolean;
  lyriaStatusLabel: string;
  cognitum?: { signedIn: boolean; pending: boolean; account?: string };
  onCognitumSignIn?: () => void;
  onChange: (preferences: OnboardingPreferences) => void;
  onEdit: () => void;
  onLaunch: (preferences: OnboardingPreferences, includeWelcomeCue: boolean) => Promise<void>;
  onClose?: () => void;
}

const FORMAT_OPTIONS: Array<{ id: MusicFormat; label: string; detail: string }> = [
  { id: "instrumental", label: "Instrumental", detail: "One main music stream" },
  { id: "hybrid", label: "Music + voice", detail: "Voice as a supporting layer" },
  { id: "vocal-led", label: "Vocal led", detail: "Voice carries the arrangement" },
];

const PACE_OPTIONS: Array<{ id: MusicPace; label: string }> = [
  { id: "slow", label: "Slow" },
  { id: "mid", label: "Mid" },
  { id: "fast", label: "Fast" },
];

const VOCAL_OPTIONS: Array<{ id: VocalStyle; label: string }> = [
  { id: "none", label: "None" },
  { id: "male", label: "Male" },
  { id: "female", label: "Female" },
  { id: "other", label: "Other" },
];

const VOCAL_ROLE_OPTIONS: Array<{ id: VocalRole; label: string }> = [
  { id: "sparse", label: "Sparse" },
  { id: "chorus", label: "Chorus" },
  { id: "experimental", label: "Experimental" },
];

interface NetworkNode {
  base: THREE.Vector3;
  phase: number;
  speed: number;
  swing: number;
}

function buildNetworkNodes(count: number): NetworkNode[] {
  const nodes: NetworkNode[] = [];
  for (let index = 0; index < count; index += 1) {
    const angle = Math.random() * Math.PI * 2;
    const edgeBias = Math.pow(Math.random(), 0.55);
    const radius = 2.9 + edgeBias * 7.6;
    const base = new THREE.Vector3(
      Math.cos(angle) * radius * (0.75 + Math.random() * 0.4),
      Math.sin(angle) * radius * 0.6 * (0.75 + Math.random() * 0.4),
      -3.6 + Math.random() * 3,
    );
    nodes.push({ base, phase: Math.random() * Math.PI * 2, speed: 0.1 + Math.random() * 0.2, swing: 0.12 + Math.random() * 0.16 });
  }
  return nodes;
}

function buildNetworkLinks(nodes: NetworkNode[], maxDistance: number, maxPerNode: number): [number, number][] {
  const links: [number, number][] = [];
  const linkSet = new Set<string>();
  for (let i = 0; i < nodes.length; i += 1) {
    const candidates = nodes
      .map((node, j) => ({ j, distance: i === j ? Infinity : nodes[i].base.distanceTo(node.base) }))
      .filter((entry) => entry.distance < maxDistance)
      .sort((a, b) => a.distance - b.distance)
      .slice(0, maxPerNode);
    for (const { j } of candidates) {
      const key = i < j ? `${i}:${j}` : `${j}:${i}`;
      if (linkSet.has(key)) continue;
      linkSet.add(key);
      links.push([i, j]);
    }
  }
  return links;
}

function WelcomeField({ active }: { active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(0, 0, 8.6);

    const nodes = buildNetworkNodes(96);
    const links = buildNetworkLinks(nodes, 2.9, 3);

    const nodePositions = new Float32Array(nodes.length * 3);
    const nodeColors = new Float32Array(nodes.length * 3);
    const nodeGeometry = new THREE.BufferGeometry();
    nodeGeometry.setAttribute("position", new THREE.BufferAttribute(nodePositions, 3));
    nodeGeometry.setAttribute("color", new THREE.BufferAttribute(nodeColors, 3));
    const cyan = new THREE.Color(0x35dcff);
    const violet = new THREE.Color(0xb06bf2);
    for (let index = 0; index < nodes.length; index += 1) {
      (Math.random() < 0.18 ? violet : cyan).toArray(nodeColors, index * 3);
    }
    const nodePoints = new THREE.Points(nodeGeometry, new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.88,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }));
    scene.add(nodePoints);

    const linkPositions = new Float32Array(links.length * 2 * 3);
    const linkGeometry = new THREE.BufferGeometry();
    linkGeometry.setAttribute("position", new THREE.BufferAttribute(linkPositions, 3));
    const linkLines = new THREE.LineSegments(linkGeometry, new THREE.LineBasicMaterial({
      color: 0x35dcff,
      transparent: true,
      opacity: 0.11,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }));
    scene.add(linkLines);

    const WAVE_PERIOD_SECONDS = 30;
    const waveGeometry = new THREE.PlaneGeometry(20, 12, 96, 58);
    const waveBase = Float32Array.from((waveGeometry.getAttribute("position") as THREE.BufferAttribute).array as ArrayLike<number>);
    const waveMaterial = new THREE.MeshBasicMaterial({
      color: 0x35dcff,
      wireframe: true,
      transparent: true,
      opacity: 0.075,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const waveMesh = new THREE.Mesh(waveGeometry, waveMaterial);
    waveMesh.position.set(0, 0, -4.6);
    scene.add(waveMesh);

    const echoGeometry = new THREE.PlaneGeometry(20, 12, 96, 58);
    const echoMaterial = new THREE.MeshBasicMaterial({
      color: 0xb06bf2,
      wireframe: true,
      transparent: true,
      opacity: 0.04,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const echoMesh = new THREE.Mesh(echoGeometry, echoMaterial);
    echoMesh.position.set(0, 0, -4.9);
    scene.add(echoMesh);

    let frame = 0;
    const clock = new THREE.Timer();
    clock.connect(document);
    const resize = () => {
      const width = canvas.clientWidth || window.innerWidth;
      const height = canvas.clientHeight || window.innerHeight;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.6));
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
    };
    const render = () => {
      clock.update();
      const time = clock.getElapsed();
      const drift = active ? 1.9 : 1;
      const nodeAttribute = nodeGeometry.getAttribute("position") as THREE.BufferAttribute;
      for (let index = 0; index < nodes.length; index += 1) {
        const node = nodes[index];
        const wobbleX = Math.sin(time * node.speed * drift + node.phase) * node.swing;
        const wobbleY = Math.cos(time * node.speed * drift * 0.84 + node.phase) * node.swing * 0.82;
        nodeAttribute.setXYZ(index, node.base.x + wobbleX, node.base.y + wobbleY, node.base.z);
      }
      nodeAttribute.needsUpdate = true;

      const linkAttribute = linkGeometry.getAttribute("position") as THREE.BufferAttribute;
      for (let index = 0; index < links.length; index += 1) {
        const [a, b] = links[index];
        linkAttribute.setXYZ(index * 2, nodeAttribute.getX(a), nodeAttribute.getY(a), nodeAttribute.getZ(a));
        linkAttribute.setXYZ(index * 2 + 1, nodeAttribute.getX(b), nodeAttribute.getY(b), nodeAttribute.getZ(b));
      }
      linkAttribute.needsUpdate = true;

      const loopPhase = ((time % WAVE_PERIOD_SECONDS) / WAVE_PERIOD_SECONDS) * Math.PI * 2;
      const amplitude = active ? 0.5 : 0.36;
      const waveAttribute = waveGeometry.getAttribute("position") as THREE.BufferAttribute;
      const echoAttribute = echoGeometry.getAttribute("position") as THREE.BufferAttribute;
      for (let index = 0; index < waveAttribute.count; index += 1) {
        const x = waveBase[index * 3];
        const y = waveBase[index * 3 + 1];
        const radial = Math.sqrt(x * x * 0.5 + y * y * 0.8);
        const primary = Math.sin(x * 0.34 + loopPhase) * Math.cos(y * 0.4 - loopPhase);
        const ripple = Math.sin(radial * 0.5 - loopPhase * 2) * 0.5;
        const z = primary * amplitude + ripple * amplitude * 0.6;
        waveAttribute.setZ(index, z);
        echoAttribute.setZ(index, z * 0.82);
      }
      waveAttribute.needsUpdate = true;
      echoAttribute.needsUpdate = true;
      waveMesh.rotation.z = Math.sin(loopPhase) * 0.012;
      echoMesh.rotation.z = Math.cos(loopPhase * 0.5) * 0.014;

      scene.rotation.y = Math.sin(time * 0.05) * 0.045;
      scene.rotation.x = Math.cos(time * 0.04) * 0.02;
      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };
    resize();
    window.addEventListener("resize", resize);
    render();
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      clock.dispose();
      nodeGeometry.dispose();
      linkGeometry.dispose();
      waveGeometry.dispose();
      echoGeometry.dispose();
      (nodePoints.material as THREE.Material).dispose();
      (linkLines.material as THREE.Material).dispose();
      waveMaterial.dispose();
      echoMaterial.dispose();
      renderer.dispose();
    };
  }, [active]);

  return <canvas className="welcome-field" ref={canvasRef} aria-hidden="true" />;
}

export function OnboardingWizard({
  view,
  firstRun,
  preferences,
  lyriaAvailable,
  lyriaStatusLabel,
  cognitum,
  onCognitumSignIn,
  onChange,
  onEdit,
  onLaunch,
  onClose,
}: OnboardingWizardProps) {
  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState(false);
  const [previewingStyleId, setPreviewingStyleId] = useState<string>();
  const [previewErrorStyleId, setPreviewErrorStyleId] = useState<string>();
  const previewAudioRef = useRef<HTMLAudioElement | undefined>(undefined);
  const loadingAudioRef = useRef<HTMLAudioElement | undefined>(undefined);
  const introChimePlayedRef = useRef(false);
  const selectedStyle = lyriaRealtimeStyleById(preferences.styleId);

  useEffect(() => {
    if (introChimePlayedRef.current) return;
    introChimePlayedRef.current = true;
    const chime = new Audio(new URL("previews/lyria/intro-chime.mp3", document.baseURI).href);
    chime.volume = 0.55;
    void chime.play().catch(() => undefined);
  }, []);

  const update = (next: Partial<OnboardingPreferences>) => onChange(normalizeOnboardingPreferences({ ...preferences, ...next }));
  const stopPreview = () => {
    previewAudioRef.current?.pause();
    previewAudioRef.current = undefined;
    setPreviewingStyleId(undefined);
  };
  const stopLoadingSound = () => {
    const audio = loadingAudioRef.current;
    if (!audio) return;
    const startVolume = audio.volume;
    const startedAt = performance.now();
    const fade = () => {
      const progress = Math.min(1, (performance.now() - startedAt) / 500);
      audio.volume = startVolume * (1 - progress);
      if (progress < 1) requestAnimationFrame(fade);
      else {
        audio.pause();
        loadingAudioRef.current = undefined;
      }
    };
    fade();
  };
  const startLoadingSound = () => {
    if (loadingAudioRef.current) return;
    const audio = new Audio(new URL("previews/lyria/welcome.mp3", document.baseURI).href);
    audio.loop = true;
    audio.volume = 0.18;
    loadingAudioRef.current = audio;
    void audio.play().catch(() => { loadingAudioRef.current = undefined; });
  };
  const togglePreview = (styleId: string) => {
    setPreviewErrorStyleId(undefined);
    if (previewingStyleId === styleId) {
      stopPreview();
      return;
    }
    stopPreview();
    const audio = new Audio(new URL("previews/lyria/" + styleId + ".mp3", document.baseURI).href);
    audio.loop = true;
    audio.volume = 0.78;
    const failPreview = () => {
      stopPreview();
      setPreviewErrorStyleId(styleId);
    };
    audio.addEventListener("error", failPreview, { once: true });
    previewAudioRef.current = audio;
    setPreviewingStyleId(styleId);
    void audio.play().catch(failPreview);
  };
  useEffect(() => () => {
    previewAudioRef.current?.pause();
    previewAudioRef.current = undefined;
    loadingAudioRef.current?.pause();
    loadingAudioRef.current = undefined;
  }, []);
  const launch = async (includeWelcomeCue: boolean) => {
    stopPreview();
    startLoadingSound();
    setBusy(true);
    try {
      await onLaunch(preferences, includeWelcomeCue);
    } finally {
      if (includeWelcomeCue) stopLoadingSound();
      setBusy(false);
    }
  };

  return (
    <section className="onboarding-shell" role="dialog" aria-modal="true" aria-label={view === "setup" ? "Music setup" : "Welcome to Musica"}>
      <WelcomeField active={busy || view === "launching"} />
      <div className="onboarding-shade" />

      {(view === "welcome" || view === "launching") && (
        <div className="welcome-content">
          <div className="welcome-wordmark"><strong>MUSICA</strong><span>LIVE AI MUSIC + VISUALS</span></div>
          <div className="welcome-divider"><i /></div>
          <div className="welcome-session">
            <span>{selectedStyle.label.toUpperCase()} · {preferences.bpm} BPM</span>
            <p>{preferences.direction || selectedStyle.description}</p>
            <div className={`welcome-provider ${lyriaAvailable ? "online" : ""}`}><i /> {lyriaStatusLabel}</div>
            <div className="welcome-actions">
              <button type="button" className="secondary" onClick={() => { stopLoadingSound(); onEdit(); }} aria-label="Edit music setup">
                <SlidersHorizontal size={16} /> EDIT SOUND
              </button>
              <button type="button" className="launch" onClick={() => void launch(true)} disabled={busy || !lyriaAvailable}>
                {busy ? <Volume2 size={17} /> : <AudioLines size={17} />} {busy ? "CONNECTING LYRIA" : "START SESSION"}
              </button>
            </div>
            {!lyriaAvailable && <button className="enter-without-audio" type="button" onClick={() => void launch(false)}>ENTER WITHOUT AUDIO</button>}
          </div>
          <div className="welcome-capabilities" aria-label="Musica capabilities">
            <div><Zap size={18} /><span>REALTIME AI</span></div>
            <div><Aperture size={19} /><span>LIVE VISUALS</span></div>
            <div><SlidersHorizontal size={19} /><span>ADAPTIVE MIX</span></div>
            <div><Box size={18} /><span>IMMERSIVE</span></div>
          </div>
          {cognitum && onCognitumSignIn && (
            <button
              type="button"
              className={`welcome-cognitum ${cognitum.signedIn ? "online" : ""}`}
              onClick={onCognitumSignIn}
              disabled={cognitum.pending || cognitum.signedIn}
              title="Optional Cognitum One account unlocks advanced AI capabilities"
            >
              <i />
              {cognitum.signedIn
                ? `COGNITUM ONE · ${(cognitum.account ?? "CONNECTED").toUpperCase()}`
                : cognitum.pending
                  ? "WAITING FOR BROWSER…"
                  : "SIGN IN WITH COGNITUM ONE · OPTIONAL"}
            </button>
          )}
          <ChevronDown className="welcome-chevron" size={21} aria-hidden="true" />
        </div>
      )}

      {view === "setup" && (
        <div className="onboarding-panel">
          <header>
            <div><span>{firstRun ? "FIRST RUN" : "MUSIC SETUP"}</span><strong>Shape your live set</strong></div>
            {!firstRun && onClose && <button type="button" onClick={onClose} aria-label="Close music setup">×</button>}
          </header>
          <div className="onboarding-progress" aria-label={`Step ${step + 1} of 4`}>
            {[0, 1, 2, 3].map((index) => <i key={index} className={index <= step ? "active" : ""} />)}
          </div>

          {step === 0 && (
            <div className="onboarding-step">
              <div className="step-heading"><span>01</span><div><strong>Music direction</strong><p>Choose the center of the set.</p></div></div>
              <div className="choice-grid format-grid">
                {FORMAT_OPTIONS.map((option) => (
                  <button key={option.id} type="button" className={preferences.format === option.id ? "selected" : ""} onClick={() => update({ format: option.id, vocalStyle: option.id === "instrumental" ? "none" : preferences.vocalStyle === "none" ? "other" : preferences.vocalStyle })}>
                    <strong>{option.label}</strong><span>{option.detail}</span>
                  </button>
                ))}
              </div>
              <label className="onboarding-label">GENRE</label>
              <div className="genre-grid">
                {LYRIA_REALTIME_STYLE_PRESETS.map((style) => (
                  <div className={`genre-option ${preferences.styleId === style.id ? "selected" : ""}`} key={style.id}>
                    <button type="button" className="genre-select" onClick={() => update({ styleId: style.id, bpm: style.config.bpm ?? preferences.bpm, direction: style.description })}>{style.label}</button>
                    <button type="button" className={`genre-preview ${previewErrorStyleId === style.id ? "error" : ""}`} onClick={() => togglePreview(style.id)} aria-label={`${previewingStyleId === style.id ? "Stop" : "Preview"} ${style.label}`} title={previewErrorStyleId === style.id ? "Preview is not available yet" : `30 second Lyria ${style.label} preview`}>
                      {previewingStyleId === style.id ? <Pause size={12} /> : <Play size={12} />}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {step === 1 && (
            <div className="onboarding-step">
              <div className="step-heading"><span>02</span><div><strong>Speed + character</strong><p>Set the performance energy.</p></div></div>
              <div className="segmented onboarding-segmented">
                {PACE_OPTIONS.map((option) => (
                  <button key={option.id} type="button" className={preferences.pace === option.id ? "active" : ""} onClick={() => update({ pace: option.id, bpm: bpmForPace(selectedStyle, option.id) })}>{option.label}</button>
                ))}
              </div>
              <label className="bpm-slider"><span>BPM</span><input type="range" min="60" max="200" value={preferences.bpm} onChange={(event) => update({ bpm: Number(event.target.value) })} /><b>{preferences.bpm}</b></label>
              <label className="experimental-toggle">
                <input type="checkbox" checked={preferences.experimental} onChange={(event) => update({ experimental: event.target.checked })} />
                <span><Sparkles size={15} /><strong>Experimental motion</strong><small>Controlled timbral variation inside stable phrases</small></span>
              </label>
              <label className="direction-input"><span>PERSONAL DIRECTION</span><textarea maxLength={180} value={preferences.direction} onChange={(event) => update({ direction: event.target.value })} placeholder="Describe the tone, instruments, mix, and energy..." /><b>{preferences.direction.length}/180</b></label>
            </div>
          )}

          {step === 2 && (
            <div className="onboarding-step">
              <div className="step-heading"><span>03</span><div><strong>Vocal layer</strong><p>Optional synchronized voice-only stream.</p></div></div>
              <div className="choice-grid vocal-grid">
                {VOCAL_OPTIONS.map((option) => (
                  <button key={option.id} type="button" className={preferences.vocalStyle === option.id ? "selected" : ""} onClick={() => update({ vocalStyle: option.id, format: option.id === "none" ? "instrumental" : preferences.format === "instrumental" ? "hybrid" : preferences.format })}>{option.label}</button>
                ))}
              </div>
              {preferences.vocalStyle !== "none" && (
                <>
                  <label className="onboarding-label">VOCAL ROLE</label>
                  <div className="segmented onboarding-segmented">
                    {VOCAL_ROLE_OPTIONS.map((option) => <button key={option.id} type="button" className={preferences.vocalRole === option.id ? "active" : ""} onClick={() => update({ vocalRole: option.id })}>{option.label}</button>)}
                  </div>
                  <button type="button" className={`vocal-preview-button ${previewErrorStyleId === `vocal-${preferences.vocalStyle}` ? "error" : ""}`} onClick={() => togglePreview(`vocal-${preferences.vocalStyle}`)}>
                    {previewingStyleId === `vocal-${preferences.vocalStyle}` ? <Pause size={13} /> : <Play size={13} />}
                    {previewingStyleId === `vocal-${preferences.vocalStyle}` ? "STOP VOICE PREVIEW" : `PREVIEW ${preferences.vocalStyle.toUpperCase()} VOICE`}
                  </button>
                </>
              )}
              <div className="onboarding-summary">
                <span>AUDIO</span><strong>{selectedStyle.label} · {preferences.bpm} BPM</strong><p>{preferences.vocalStyle === "none" ? "Instrumental main stream" : `${preferences.vocalStyle} ${preferences.vocalRole} vocal companion`} · {preferences.experimental ? "experimental variation" : "controlled variation"}</p>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="onboarding-step">
              <div className="step-heading"><span>04</span><div><strong>Visual direction</strong><p>Choose the initial live canvas.</p></div></div>
              <label className="onboarding-label">SCENE</label>
              <div className="visual-scene-grid">
                {VISUAL_SCENES.map((scene) => (
                  <button key={scene.id} type="button" className={preferences.visualScene === scene.id ? "selected" : ""} onClick={() => update({ visualScene: scene.id })}>
                    <i style={{ background: scene.color }} /><strong>{scene.name}</strong>
                  </button>
                ))}
              </div>
              <label className="onboarding-label">COLOR</label>
              <div className="visual-palette-options">
                {VISUAL_COLOR_PALETTES.map((palette) => (
                  <button key={palette.id} type="button" className={preferences.visualPalette === palette.id ? "selected" : ""} onClick={() => update({ visualPalette: palette.id })}>
                    <i style={{ background: palette.color ?? "#89909c", borderColor: palette.accent ?? "#c8ccd4" }} /><span>{palette.label}</span>
                  </button>
                ))}
              </div>
              <label className="onboarding-label">MOTION</label>
              <div className="visual-motion-options">
                {VISUAL_ANIMATION_STYLES.map((style) => <button key={style.id} type="button" className={preferences.visualAnimation === style.id ? "selected" : ""} onClick={() => update({ visualAnimation: style.id })}><strong>{style.label}</strong><span>{style.description}</span></button>)}
              </div>
              <label className="bpm-slider visual-intensity"><span>LEVEL</span><input type="range" min="5" max="100" value={Math.round(preferences.visualIntensity * 100)} onChange={(event) => update({ visualIntensity: Number(event.target.value) / 100 })} /><b>{Math.round(preferences.visualIntensity * 100)}</b></label>
            </div>
          )}

          <footer>
            <button type="button" className="icon-action" onClick={() => setStep((current) => Math.max(0, current - 1))} disabled={step === 0 || busy} title="Previous step"><ArrowLeft size={17} /></button>
            <span>{step + 1} / 4</span>
            {step < 3 ? (
              <button type="button" className="next" onClick={() => setStep((current) => Math.min(3, current + 1))}>NEXT <ArrowRight size={15} /></button>
            ) : (
              <button type="button" className="next" onClick={() => void launch(false)} disabled={busy}>
                {busy ? <Volume2 size={15} /> : <Check size={15} />} {busy ? "SAVING" : firstRun ? "SAVE + CONTINUE" : "APPLY"}
              </button>
            )}
          </footer>
        </div>
      )}
    </section>
  );
}
