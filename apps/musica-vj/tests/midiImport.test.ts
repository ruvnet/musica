import { Midi } from "@tonejs/midi";
import { describe, expect, it } from "vitest";
import { importMidiPerformance } from "../src/core/midiImport";

describe("midi import", () => {
  it("maps a standard MIDI file into editable performance tracks", () => {
    const midi = new Midi();
    midi.header.setTempo(128);
    const drums = midi.addTrack();
    drums.channel = 9;
    drums.addNote({ midi: 36, ticks: 0, durationTicks: 120 });
    drums.addNote({ midi: 38, ticks: 480, durationTicks: 120 });
    const bass = midi.addTrack();
    bass.channel = 1;
    bass.addNote({ midi: 36, ticks: 0, durationTicks: 240 });
    bass.addNote({ midi: 43, ticks: 720, durationTicks: 240 });

    const bytes = midi.toArray();
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    const result = importMidiPerformance(buffer, "fixture.mid");

    expect(result.bpm).toBe(128);
    expect(result.tracks.drums?.pattern).toContain(0);
    expect(result.tracks.drums?.notes).toEqual([36, 38]);
    expect(result.tracks.bass?.notes).toEqual([36, 43]);
  });
});
