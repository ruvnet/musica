import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  DJ_CONTROL_PROFILES,
  normalizeDjControlLayout,
  sendDjControlCommand,
  subscribeDjControlState,
  type DjControlProfileId,
  type DjControlState,
  type DjControlWidgetId,
  type DjControlWidgetLayout,
} from "./core/djControls";
import { installMagneticWindowSnapping } from "./core/djWindows";
import { CUSTOM_LYRIA_STYLES_STORAGE_KEY, LYRIA_REALTIME_STYLE_PRESETS, loadCustomLyriaStyles } from "./core/lyriaRealtime";
import { DEFAULT_LYRIA_DECK_CONTROLS, DEFAULT_LYRIA_DECK_SCENES, cloneLyriaDeckScene, type LyriaDeckControl } from "./core/lyriaDeckScenes";
import { VISUAL_SCENES } from "./core/presets";
import { DEFAULT_VISUAL_COLOR_CONTROLS, VISUAL_COLOR_PALETTES } from "./visual/VisualEngine";

const WIDGET_LABELS: Record<DjControlWidgetId, string> = {
  transport: "Transport",
  "deck-scenes": "Deck Scenes",
  "deck-mixer": "Stream Mixer",
  styles: "Styles",
  visuals: "Visual Bank",
  color: "Color / Look",
  master: "Master",
};

function fallbackState(): DjControlState {
  return {
    playing: false,
    bpm: 118,
    masterVolume: 0.82,
    styleId: "house",
    deckScenes: DEFAULT_LYRIA_DECK_SCENES.map(cloneLyriaDeckScene),
    deckEnabled: { main: true, sequence: false, vocal: false },
    deckControls: {
      main: { ...DEFAULT_LYRIA_DECK_CONTROLS.main },
      sequence: { ...DEFAULT_LYRIA_DECK_CONTROLS.sequence },
      vocal: { ...DEFAULT_LYRIA_DECK_CONTROLS.vocal },
    },
    visualScene: "tunnel",
    visualIntensity: 0.72,
    visualColor: { ...DEFAULT_VISUAL_COLOR_CONTROLS },
  };
}

function loadLayout(profile: DjControlProfileId): DjControlWidgetLayout[] {
  try {
    return normalizeDjControlLayout(profile, JSON.parse(window.localStorage.getItem(`musica.dj-window.${profile}.v1`) ?? "null"));
  } catch {
    return normalizeDjControlLayout(profile);
  }
}

export function DjControlWindow({ profile }: { profile: DjControlProfileId }) {
  const [state, setState] = useState<DjControlState>(fallbackState);
  const [connected, setConnected] = useState(false);
  const [customizing, setCustomizing] = useState(false);
  const [layout, setLayout] = useState<DjControlWidgetLayout[]>(() => loadLayout(profile));
  const allStyles = useMemo(() => {
    try {
      return [...LYRIA_REALTIME_STYLE_PRESETS, ...loadCustomLyriaStyles(window.localStorage.getItem(CUSTOM_LYRIA_STYLES_STORAGE_KEY))];
    } catch {
      return [...LYRIA_REALTIME_STYLE_PRESETS];
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(`musica.dj-window.${profile}.v1`, JSON.stringify(layout));
  }, [layout, profile]);

  useEffect(() => {
    let disposed = false;
    let unsubscribe: () => void = () => undefined;
    void subscribeDjControlState((next) => {
      if (disposed) return;
      setState(next);
      setConnected(true);
    }).then((stop) => {
      if (disposed) stop();
      else {
        unsubscribe = stop;
        void sendDjControlCommand({ type: "request-state" });
      }
    });
    return () => {
      disposed = true;
      unsubscribe();
    };
  }, []);

  useEffect(() => {
    let disposed = false;
    let uninstall: () => void = () => undefined;
    void installMagneticWindowSnapping().then((stop) => {
      if (disposed) stop();
      else uninstall = stop;
    });
    return () => {
      disposed = true;
      uninstall();
    };
  }, []);

  const sendAction = useCallback((action: Parameters<typeof sendDjControlCommand>[0] & { type: "action" }) => {
    void sendDjControlCommand(action);
  }, []);

  const updateDeck = useCallback((deck: "main" | "sequence" | "vocal", update: Partial<LyriaDeckControl>) => {
    setState((current) => ({
      ...current,
      activeDeckSceneId: undefined,
      deckControls: { ...current.deckControls, [deck]: { ...current.deckControls[deck], ...update } },
    }));
    void sendDjControlCommand({ type: "deck-control", deck, update });
  }, []);

  const visibleWidgets = useMemo(() => layout.filter((widget) => widget.visible), [layout]);

  const moveWidget = (index: number, delta: number) => {
    const target = index + delta;
    if (target < 0 || target >= layout.length) return;
    setLayout((current) => {
      const next = [...current];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  };

  const renderWidget = (id: DjControlWidgetId): ReactNode => {
    if (id === "transport") return (
      <section className="dj-widget dj-transport">
        <header><span>TRANSPORT</span><b>{state.bpm} BPM</b></header>
        <div className="dj-transport-main">
          <button className={state.playing ? "active" : ""} onClick={() => sendAction({ type: "action", action: "transport.toggle" })}>{state.playing ? "PAUSE" : "PLAY"}</button>
          <button onClick={() => sendAction({ type: "action", action: "transport.stop" })}>STOP</button>
          <button onClick={() => sendAction({ type: "action", action: "tempo.tap" })}>TAP</button>
        </div>
        <div className="dj-bpm-control">
          <button onClick={() => void sendDjControlCommand({ type: "bpm", value: state.bpm - 1 })}>-</button>
          <input aria-label="DJ BPM" type="number" min="60" max="200" value={state.bpm} onChange={(event) => void sendDjControlCommand({ type: "bpm", value: Number(event.target.value) })} />
          <button onClick={() => void sendDjControlCommand({ type: "bpm", value: state.bpm + 1 })}>+</button>
        </div>
      </section>
    );

    if (id === "deck-scenes") return (
      <section className="dj-widget">
        <header><span>DECK SCENES</span><b>SHIFT 1-4</b></header>
        <div className="dj-scene-grid">
          {state.deckScenes.map((scene, index) => (
            <button key={scene.id} className={scene.id === state.activeDeckSceneId ? "active" : ""} onClick={() => sendAction({ type: "action", action: "lyria.deck-scene.select", value: index })}>
              <em>{index + 1}</em><strong>{scene.name}</strong><small>{scene.bpm}</small>
            </button>
          ))}
        </div>
      </section>
    );

    if (id === "deck-mixer") return (
      <section className="dj-widget">
        <header><span>LYRIA STREAMS</span><b>LIVE MIX</b></header>
        <div className="dj-deck-mixer">
          {(["main", "sequence", "vocal"] as const).map((deck) => {
            const control = state.deckControls[deck];
            return (
              <article key={deck}>
                <header>
                  <strong>{deck === "vocal" ? "VOCALIZE" : deck.toUpperCase()}</strong>
                  <button className={state.deckEnabled[deck] ? "power active" : "power"} onClick={() => void sendDjControlCommand({ type: "deck-enabled", deck, enabled: !state.deckEnabled[deck] })}>{state.deckEnabled[deck] ? "ON" : "OFF"}</button>
                  <button className={control.muted ? "active" : ""} onClick={() => updateDeck(deck, { muted: !control.muted })}>M</button>
                </header>
                <label><span>VOL</span><input type="range" min="0" max="1" step="0.01" value={control.volume} onChange={(event) => updateDeck(deck, { volume: Number(event.target.value) })} /><b>{Math.round(control.volume * 100)}</b></label>
                <label><span>PITCH</span><input type="range" min="-7" max="7" step="1" value={control.pitchSemitones} onChange={(event) => updateDeck(deck, { pitchSemitones: Number(event.target.value) })} /><b>{control.pitchSemitones > 0 ? `+${control.pitchSemitones}` : control.pitchSemitones}</b></label>
                <label><span>BEAT</span><input type="range" min="-250" max="250" step="5" value={control.beatNudgeMs} onChange={(event) => updateDeck(deck, { beatNudgeMs: Number(event.target.value) })} /><b>{control.beatNudgeMs}</b></label>
              </article>
            );
          })}
        </div>
      </section>
    );

    if (id === "styles") return (
      <section className="dj-widget">
        <header><span>MUSICAL STYLE</span><b>LYRIA</b></header>
        <div className="dj-style-grid">
          {allStyles.map((style) => (
            <button
              key={style.id}
              className={`${style.id === state.styleId ? "active" : ""} ${style.id.startsWith("custom-") ? "custom-style" : ""}`}
              title={style.description}
              onClick={() => void sendDjControlCommand({ type: "style", styleId: style.id })}
            >
              {style.label}
            </button>
          ))}
        </div>
      </section>
    );

    if (id === "visuals") return (
      <section className="dj-widget">
        <header><span>VISUAL BANK</span><b>{state.visualScene.toUpperCase()}</b></header>
        <div className="dj-visual-grid">
          {VISUAL_SCENES.map((scene, index) => <button key={scene.id} className={scene.id === state.visualScene ? "active" : ""} onClick={() => sendAction({ type: "action", action: "visual.scene.select", value: index })}>{scene.label}</button>)}
        </div>
      </section>
    );

    if (id === "color") return (
      <section className="dj-widget">
        <header><span>COLOR / LOOK</span><b>{state.visualColor.palette.toUpperCase()}</b></header>
        <div className="dj-palette-grid">
          {VISUAL_COLOR_PALETTES.map((palette) => (
            <button
              key={palette.id}
              className={palette.id === state.visualColor.palette ? "active" : ""}
              onClick={() => void sendDjControlCommand({ type: "visual-color", update: { palette: palette.id } })}
            ><i style={{ background: palette.color ?? "#6876a8", boxShadow: `8px 0 0 ${palette.accent ?? "#49aab3"}` }} />{palette.label}</button>
          ))}
        </div>
        <div className="dj-color-sliders">
          {(["hue", "saturation", "contrast", "diversity"] as const).map((key) => (
            <label key={key}><span>{key.toUpperCase()}</span><input type="range" min="0" max="1" step="0.01" value={state.visualColor[key]} onChange={(event) => void sendDjControlCommand({ type: "visual-color", update: { [key]: Number(event.target.value) } })} /><b>{Math.round(state.visualColor[key] * 100)}</b></label>
          ))}
        </div>
      </section>
    );

    return (
      <section className="dj-widget">
        <header><span>MASTER</span><b>{Math.round(state.masterVolume * 100)}</b></header>
        <div className="dj-master-grid">
          <button onClick={() => sendAction({ type: "action", action: "master.delta", value: -1 })}>VOL -</button>
          <button onClick={() => sendAction({ type: "action", action: "master.delta", value: 1 })}>VOL +</button>
          <button onClick={() => sendAction({ type: "action", action: "visual.intensity.delta", value: -1 })}>VIS -</button>
          <button onClick={() => sendAction({ type: "action", action: "visual.intensity.delta", value: 1 })}>VIS +</button>
        </div>
      </section>
    );
  };

  return (
    <main className="dj-control-shell">
      <header className="dj-control-header">
        <div><span>MUSICA DJ</span><strong>{DJ_CONTROL_PROFILES[profile].label.toUpperCase()}</strong></div>
        <i className={connected ? "online" : ""} title={connected ? "Connected to main deck" : "Waiting for main deck"} />
        <b>SNAP</b>
        <button className={customizing ? "active" : ""} onClick={() => setCustomizing((value) => !value)}>EDIT</button>
      </header>
      {customizing && (
        <section className="dj-customizer">
          {layout.map((widget, index) => (
            <div key={widget.id}>
              <label><input type="checkbox" checked={widget.visible} onChange={(event) => setLayout((current) => current.map((candidate) => candidate.id === widget.id ? { ...candidate, visible: event.target.checked } : candidate))} /> {WIDGET_LABELS[widget.id]}</label>
              <button className={widget.wide ? "active" : ""} onClick={() => setLayout((current) => current.map((candidate) => candidate.id === widget.id ? { ...candidate, wide: !candidate.wide } : candidate))}>WIDE</button>
              <button onClick={() => moveWidget(index, -1)} disabled={index === 0}>UP</button>
              <button onClick={() => moveWidget(index, 1)} disabled={index === layout.length - 1}>DOWN</button>
            </div>
          ))}
          <button className="dj-reset-layout" onClick={() => setLayout(normalizeDjControlLayout(profile))}>RESET LAYOUT</button>
        </section>
      )}
      <div className="dj-widget-grid">
        {visibleWidgets.map((widget) => <div className={widget.wide ? "wide" : ""} key={widget.id}>{renderWidget(widget.id)}</div>)}
      </div>
    </main>
  );
}
