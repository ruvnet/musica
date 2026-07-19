import { Midi } from "@tonejs/midi";
import { TRACK_IDS, type TrackId, type TrackTemplate } from "./types";

export interface MidiImportResult {
  name: string;
  bpm?: number;
  tracks: Partial<Record<TrackId, TrackTemplate>>;
}

const ROLE_ORDER: TrackId[] = ["drums", "bass", "chords", "lead", "voice", "texture"];

function roleForMidiTrack(index: number, channel: number, averageNote: number): TrackId {
  if (channel === 9) return "drums";
  if (averageNote < 48) return "bass";
  if (index === 0) return "chords";
  return ROLE_ORDER[Math.min(index, ROLE_ORDER.length - 1)] ?? "texture";
}

function clampMidi(note: number, role: TrackId): number {
  if (role === "drums") {
    if (note <= 36) return 36;
    if (note <= 40) return 38;
    if (note <= 45) return 42;
    if (note <= 49) return 46;
    return 49;
  }
  if (role === "bass") return Math.min(48, Math.max(24, note));
  if (role === "chords") return Math.min(76, Math.max(45, note));
  if (role === "lead") return Math.min(96, Math.max(56, note));
  if (role === "voice") return Math.min(84, Math.max(50, note));
  return Math.min(72, Math.max(32, note));
}

export function importMidiPerformance(bytes: ArrayBuffer, name = "Imported MIDI"): MidiImportResult {
  const midi = new Midi(bytes);
  const bpm = midi.header.tempos[0]?.bpm === undefined ? undefined : Math.round(midi.header.tempos[0].bpm);
  const tracks: Partial<Record<TrackId, TrackTemplate>> = {};

  midi.tracks
    .filter((track) => track.notes.length > 0)
    .slice(0, ROLE_ORDER.length)
    .forEach((track, index) => {
      const averageNote = track.notes.reduce((sum, note) => sum + note.midi, 0) / track.notes.length;
      const role = roleForMidiTrack(index, track.channel, averageNote);
      const steps = new Set<number>();
      const notes: number[] = [];
      for (const note of track.notes.slice(0, 96)) {
        const step = Math.round(note.ticks / midi.header.ppq / 0.25) % 16;
        steps.add(step);
        notes.push(clampMidi(note.midi, role));
      }
      if (steps.size === 0 || notes.length === 0) return;
      const pattern = [...steps].sort((left, right) => left - right);
      const current = tracks[role];
      tracks[role] = {
        pattern: current ? [...new Set([...current.pattern, ...pattern])].sort((left, right) => left - right) : pattern,
        notes: current ? [...current.notes, ...notes].slice(0, 64) : notes.slice(0, 64),
        volume: role === "drums" ? 0.82 : role === "bass" ? 0.78 : role === "texture" ? 0.44 : 0.56,
        pan: role === "lead" ? 0.18 : role === "voice" ? -0.24 : role === "texture" ? 0.28 : 0,
      };
    });

  return { name, bpm, tracks };
}
