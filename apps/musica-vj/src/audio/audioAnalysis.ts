import type { VisualSceneId } from "../core/types";

export type EncodedAudioCodec = "wav" | "mp3";

export interface EncodedAudioMetadata {
  codec: EncodedAudioCodec;
  mimeType: "audio/wav" | "audio/mpeg";
  durationSeconds: number;
  sampleRateHz: number;
  channels: number;
  bitRateKbps: number;
  bitDepth?: number;
  frameCount?: number;
  audioDataBytes: number;
}

export interface DecodedPcmAudio {
  sampleRateHz: number;
  channels: readonly Float32Array[];
}

export interface WaveformBucket {
  startSeconds: number;
  endSeconds: number;
  minimum: number;
  maximum: number;
  rms: number;
}

export interface OnsetPoint {
  timeSeconds: number;
  strength: number;
}

export interface SpectralProfile {
  lowEnergy: number;
  midEnergy: number;
  highEnergy: number;
  centroidHz: number;
}

export type MeasuredSectionType = "intro" | "build" | "drop" | "breakdown" | "groove" | "outro";

export interface MeasuredSection {
  type: MeasuredSectionType;
  start: number;
  end: number;
  meanRms: number;
}

export interface VisualMapping {
  bass: "camera_displacement";
  kick: "radial_pulse";
  highFrequencyEnergy: "particle_density";
  sectionChange: "scene_transition";
}

export interface AudioAnalysisResult {
  durationSeconds: number;
  sampleRateHz: number;
  channels: number;
  loudnessLufs: number;
  bpm: number | null;
  key: string | null;
  waveform: WaveformBucket[];
  onsetMap: OnsetPoint[];
  beatGridSeconds: number[];
  spectralProfile: SpectralProfile;
  sections: MeasuredSection[];
  visualMapping: VisualMapping;
  recommendedScene: VisualSceneId;
  visualIntensity: number;
}

export interface AnalysisOptions {
  waveformBuckets?: number;
  minimumBpm?: number;
  maximumBpm?: number;
}

interface Mp3FrameHeader {
  version: 1 | 2 | 2.5;
  layer: 1 | 2 | 3;
  bitRateKbps: number;
  sampleRateHz: number;
  channels: 1 | 2;
  samplesPerFrame: number;
  frameLength: number;
}

interface EnvelopeAnalysis {
  values: Float64Array;
  hopSize: number;
  peaks: Array<{ index: number; strength: number }>;
}

function bytesToAscii(bytes: Uint8Array, offset: number, length: number): string {
  let value = "";
  for (let index = 0; index < length; index += 1) value += String.fromCharCode(bytes[offset + index] ?? 0);
  return value;
}

function normalizeMimeType(mimeType?: string): string | undefined {
  return mimeType?.split(";", 1)[0]?.trim().toLowerCase();
}

function isWavMime(mimeType: string): boolean {
  return mimeType === "audio/wav" || mimeType === "audio/wave" || mimeType === "audio/x-wav";
}

function isMp3Mime(mimeType: string): boolean {
  return mimeType === "audio/mpeg" || mimeType === "audio/mp3";
}

function round(value: number, digits = 6): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function inspectWav(bytes: Uint8Array): EncodedAudioMetadata {
  if (bytes.length < 12 || bytesToAscii(bytes, 0, 4) !== "RIFF" || bytesToAscii(bytes, 8, 4) !== "WAVE") {
    throw new Error("Invalid WAV signature");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let format: { channels: number; sampleRateHz: number; byteRate: number; blockAlign: number; bitDepth: number } | undefined;
  let dataBytes: number | undefined;
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const chunkId = bytesToAscii(bytes, offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const dataOffset = offset + 8;
    if (chunkSize > bytes.length - dataOffset) throw new Error(`Truncated WAV ${chunkId || "unknown"} chunk`);
    if (chunkId === "fmt ") {
      if (chunkSize < 16) throw new Error("Invalid WAV format chunk");
      const audioFormat = view.getUint16(dataOffset, true);
      if (audioFormat !== 1 && audioFormat !== 3 && audioFormat !== 0xfffe) {
        throw new Error(`Unsupported WAV audio format ${audioFormat}`);
      }
      format = {
        channels: view.getUint16(dataOffset + 2, true),
        sampleRateHz: view.getUint32(dataOffset + 4, true),
        byteRate: view.getUint32(dataOffset + 8, true),
        blockAlign: view.getUint16(dataOffset + 12, true),
        bitDepth: view.getUint16(dataOffset + 14, true),
      };
    } else if (chunkId === "data") {
      dataBytes = chunkSize;
    }
    offset = dataOffset + chunkSize + (chunkSize & 1);
  }
  if (!format || dataBytes === undefined) throw new Error("WAV is missing its format or data chunk");
  if (
    format.channels < 1 ||
    format.channels > 32 ||
    format.sampleRateHz < 8_000 ||
    format.sampleRateHz > 384_000 ||
    format.byteRate <= 0 ||
    format.blockAlign <= 0
  ) {
    throw new Error("WAV contains invalid audio parameters");
  }
  const durationSeconds = dataBytes / format.byteRate;
  return {
    codec: "wav",
    mimeType: "audio/wav",
    durationSeconds,
    sampleRateHz: format.sampleRateHz,
    channels: format.channels,
    bitRateKbps: (format.byteRate * 8) / 1_000,
    bitDepth: format.bitDepth,
    frameCount: Math.floor(dataBytes / format.blockAlign),
    audioDataBytes: dataBytes,
  };
}

const MPEG1_BITRATES: Record<number, readonly number[]> = {
  1: [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
  2: [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
  3: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
};

const MPEG2_BITRATES: Record<number, readonly number[]> = {
  1: [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
  2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
  3: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
};

function parseMp3FrameHeader(bytes: Uint8Array, offset: number): Mp3FrameHeader | undefined {
  if (offset + 4 > bytes.length) return undefined;
  const header =
    (((bytes[offset] ?? 0) << 24) | ((bytes[offset + 1] ?? 0) << 16) | ((bytes[offset + 2] ?? 0) << 8) | (bytes[offset + 3] ?? 0)) >>> 0;
  if ((header >>> 21) !== 0x7ff) return undefined;
  const versionBits = (header >>> 19) & 0b11;
  const layerBits = (header >>> 17) & 0b11;
  const bitRateIndex = (header >>> 12) & 0b1111;
  const sampleRateIndex = (header >>> 10) & 0b11;
  const padding = (header >>> 9) & 1;
  if (versionBits === 1 || layerBits === 0 || bitRateIndex === 0 || bitRateIndex === 15 || sampleRateIndex === 3) {
    return undefined;
  }
  const version: 1 | 2 | 2.5 = versionBits === 3 ? 1 : versionBits === 2 ? 2 : 2.5;
  const layer: 1 | 2 | 3 = layerBits === 3 ? 1 : layerBits === 2 ? 2 : 3;
  const baseSampleRates = [44_100, 48_000, 32_000] as const;
  const sampleRateHz = baseSampleRates[sampleRateIndex] / (version === 1 ? 1 : version === 2 ? 2 : 4);
  const bitRateKbps = (version === 1 ? MPEG1_BITRATES : MPEG2_BITRATES)[layer]?.[bitRateIndex];
  if (!bitRateKbps) return undefined;
  const samplesPerFrame = layer === 1 ? 384 : layer === 3 && version !== 1 ? 576 : 1152;
  const bitRate = bitRateKbps * 1_000;
  const frameLength =
    layer === 1
      ? Math.floor((12 * bitRate) / sampleRateHz + padding) * 4
      : Math.floor(((layer === 3 && version !== 1 ? 72 : 144) * bitRate) / sampleRateHz + padding);
  if (frameLength < 4) return undefined;
  return {
    version,
    layer,
    bitRateKbps,
    sampleRateHz,
    channels: ((header >>> 6) & 0b11) === 3 ? 1 : 2,
    samplesPerFrame,
    frameLength,
  };
}

function id3PayloadEnd(bytes: Uint8Array): number {
  if (bytes.length < 10 || bytesToAscii(bytes, 0, 3) !== "ID3") return 0;
  const sizeBytes = [bytes[6] ?? 0, bytes[7] ?? 0, bytes[8] ?? 0, bytes[9] ?? 0];
  if (sizeBytes.some((value) => value > 0x7f)) throw new Error("Invalid ID3 size");
  const payloadSize = sizeBytes.reduce((size, value) => (size << 7) | value, 0);
  const footerSize = ((bytes[5] ?? 0) & 0x10) !== 0 ? 10 : 0;
  const end = 10 + payloadSize + footerSize;
  if (end > bytes.length) throw new Error("Truncated ID3 tag");
  return end;
}

function inspectMp3(bytes: Uint8Array): EncodedAudioMetadata {
  let firstOffset = id3PayloadEnd(bytes);
  const searchEnd = Math.min(bytes.length - 4, firstOffset + 65_536);
  let firstHeader: Mp3FrameHeader | undefined;
  while (firstOffset <= searchEnd) {
    const candidate = parseMp3FrameHeader(bytes, firstOffset);
    if (candidate && firstOffset + candidate.frameLength <= bytes.length) {
      firstHeader = candidate;
      break;
    }
    firstOffset += 1;
  }
  if (!firstHeader) throw new Error("Invalid MP3 frame signature");

  let cursor = firstOffset;
  let frameCount = 0;
  let sampleCount = 0;
  let audioDataBytes = 0;
  let weightedBitRate = 0;
  while (cursor + 4 <= bytes.length) {
    const frame = parseMp3FrameHeader(bytes, cursor);
    if (
      !frame ||
      frame.version !== firstHeader.version ||
      frame.layer !== firstHeader.layer ||
      frame.sampleRateHz !== firstHeader.sampleRateHz ||
      cursor + frame.frameLength > bytes.length
    ) {
      break;
    }
    frameCount += 1;
    sampleCount += frame.samplesPerFrame;
    audioDataBytes += frame.frameLength;
    weightedBitRate += frame.bitRateKbps * frame.samplesPerFrame;
    cursor += frame.frameLength;
  }
  if (frameCount === 0 || sampleCount === 0) throw new Error("MP3 contains no complete audio frames");
  return {
    codec: "mp3",
    mimeType: "audio/mpeg",
    durationSeconds: sampleCount / firstHeader.sampleRateHz,
    sampleRateHz: firstHeader.sampleRateHz,
    channels: firstHeader.channels,
    bitRateKbps: weightedBitRate / sampleCount,
    frameCount,
    audioDataBytes,
  };
}

export function inspectEncodedAudio(
  input: ArrayBuffer | Uint8Array,
  declaredMimeType?: string,
): EncodedAudioMetadata {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const mimeType = normalizeMimeType(declaredMimeType);
  const hasWavSignature = bytes.length >= 12 && bytesToAscii(bytes, 0, 4) === "RIFF" && bytesToAscii(bytes, 8, 4) === "WAVE";
  const hasMp3Signature =
    (bytes.length >= 3 && bytesToAscii(bytes, 0, 3) === "ID3") || parseMp3FrameHeader(bytes, 0) !== undefined;
  if (hasWavSignature) {
    if (mimeType !== undefined && !isWavMime(mimeType)) throw new Error("Declared MIME type does not match WAV signature");
    return inspectWav(bytes);
  }
  if (hasMp3Signature) {
    if (mimeType !== undefined && !isMp3Mime(mimeType)) throw new Error("Declared MIME type does not match MP3 signature");
    return inspectMp3(bytes);
  }
  throw new Error("Unsupported or invalid audio signature");
}

function validateDecodedPcm(audio: DecodedPcmAudio): number {
  if (!Number.isInteger(audio.sampleRateHz) || audio.sampleRateHz < 8_000 || audio.sampleRateHz > 384_000) {
    throw new Error("Decoded PCM sample rate must be between 8 kHz and 384 kHz");
  }
  if (audio.channels.length < 1 || audio.channels.length > 8) {
    throw new Error("Decoded PCM must contain between one and eight channels");
  }
  const sampleCount = audio.channels[0]?.length ?? 0;
  if (sampleCount === 0) throw new Error("Decoded PCM is empty");
  if (audio.channels.some((channel) => channel.length !== sampleCount)) {
    throw new Error("Decoded PCM channels must contain the same number of samples");
  }
  return sampleCount;
}

function createPhaseSafeAnalysisSignal(audio: DecodedPcmAudio, sampleCount: number): Float32Array {
  // A conventional arithmetic downmix can turn audible anti-phase stereo into
  // silence. Use the channel with the greatest measured energy as the stable
  // analysis signal; integrated loudness still measures every channel below.
  let dominantChannel = audio.channels[0]!;
  let dominantEnergy = -1;
  for (const channel of audio.channels) {
    let energy = 0;
    for (let index = 0; index < sampleCount; index += 1) {
      const value = channel[index] ?? 0;
      const finiteValue = Number.isFinite(value) ? clamp(value, -1, 1) : 0;
      energy += finiteValue * finiteValue;
    }
    if (energy > dominantEnergy) {
      dominantEnergy = energy;
      dominantChannel = channel;
    }
  }

  const signal = new Float32Array(sampleCount);
  for (let index = 0; index < sampleCount; index += 1) {
    const value = dominantChannel[index] ?? 0;
    signal[index] = Number.isFinite(value) ? clamp(value, -1, 1) : 0;
  }
  return signal;
}

function calculateWaveform(
  mono: Float32Array,
  sampleRateHz: number,
  requestedBuckets: number,
): WaveformBucket[] {
  const bucketCount = Math.min(mono.length, clamp(Math.floor(requestedBuckets), 16, 2_048));
  const waveform: WaveformBucket[] = [];
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = Math.floor((bucket * mono.length) / bucketCount);
    const end = Math.max(start + 1, Math.floor(((bucket + 1) * mono.length) / bucketCount));
    let minimum = 1;
    let maximum = -1;
    let sumSquares = 0;
    for (let index = start; index < end && index < mono.length; index += 1) {
      const value = mono[index] ?? 0;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      sumSquares += value * value;
    }
    const count = Math.max(1, Math.min(end, mono.length) - start);
    waveform.push({
      startSeconds: start / sampleRateHz,
      endSeconds: Math.min(end, mono.length) / sampleRateHz,
      minimum,
      maximum,
      rms: Math.sqrt(sumSquares / count),
    });
  }
  return waveform;
}

interface BiquadCoefficients {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

interface BiquadState {
  x1: number;
  x2: number;
  y1: number;
  y2: number;
}

function kWeightingCoefficients(sampleRateHz: number): [BiquadCoefficients, BiquadCoefficients] {
  // ITU-R BS.1770 K-weighting: the first stage is the De Man high shelf and
  // the second is the revised low-frequency B-curve high-pass stage. These
  // equations reproduce the published 48 kHz coefficients and adapt them to
  // the decoded sample rate rather than assuming 44.1 or 48 kHz.
  const shelfFrequency = 1_681.974450955533;
  const shelfGainDb = 3.999843853973347;
  const shelfQ = 0.7071752369554196;
  const shelfK = Math.tan((Math.PI * shelfFrequency) / sampleRateHz);
  const shelfVh = 10 ** (shelfGainDb / 20);
  const shelfVb = shelfVh ** 0.4996667741545416;
  const shelfA0 = 1 + shelfK / shelfQ + shelfK * shelfK;
  const shelf: BiquadCoefficients = {
    b0: (shelfVh + (shelfVb * shelfK) / shelfQ + shelfK * shelfK) / shelfA0,
    b1: (2 * (shelfK * shelfK - shelfVh)) / shelfA0,
    b2: (shelfVh - (shelfVb * shelfK) / shelfQ + shelfK * shelfK) / shelfA0,
    a1: (2 * (shelfK * shelfK - 1)) / shelfA0,
    a2: (1 - shelfK / shelfQ + shelfK * shelfK) / shelfA0,
  };

  const highPassFrequency = 38.13547087602444;
  const highPassQ = 0.5003270373238773;
  const highPassK = Math.tan((Math.PI * highPassFrequency) / sampleRateHz);
  const highPassA0 = 1 + highPassK / highPassQ + highPassK * highPassK;
  const highPass: BiquadCoefficients = {
    b0: 1,
    b1: -2,
    b2: 1,
    a1: (2 * (highPassK * highPassK - 1)) / highPassA0,
    a2: (1 - highPassK / highPassQ + highPassK * highPassK) / highPassA0,
  };
  return [shelf, highPass];
}

function filterSample(sample: number, coefficients: BiquadCoefficients, state: BiquadState): number {
  const output =
    coefficients.b0 * sample +
    coefficients.b1 * state.x1 +
    coefficients.b2 * state.x2 -
    coefficients.a1 * state.y1 -
    coefficients.a2 * state.y2;
  state.x2 = state.x1;
  state.x1 = sample;
  state.y2 = state.y1;
  state.y1 = output;
  return output;
}

function loudnessFromEnergy(energy: number): number {
  return energy > 1e-12 ? -0.691 + 10 * Math.log10(energy) : -120;
}

function calculateIntegratedLoudness(audio: DecodedPcmAudio, sampleCount: number): number {
  const blockLength = Math.min(sampleCount, Math.max(1, Math.round(audio.sampleRateHz * 0.4)));
  const stepLength = Math.max(1, Math.round(audio.sampleRateHz * 0.1));
  const blockCount = sampleCount <= blockLength ? 1 : Math.floor((sampleCount - blockLength) / stepLength) + 1;
  const blockEnergies = new Float64Array(blockCount);
  const [shelf, highPass] = kWeightingCoefficients(audio.sampleRateHz);

  for (const channel of audio.channels) {
    const shelfState: BiquadState = { x1: 0, x2: 0, y1: 0, y2: 0 };
    const highPassState: BiquadState = { x1: 0, x2: 0, y1: 0, y2: 0 };
    const energyWindow = new Float64Array(blockLength);
    let runningEnergy = 0;
    let windowIndex = 0;
    let blockIndex = 0;
    let nextBlockEnd = blockLength - 1;

    for (let index = 0; index < sampleCount; index += 1) {
      const value = channel[index] ?? 0;
      const input = Number.isFinite(value) ? clamp(value, -1, 1) : 0;
      const shelfOutput = filterSample(input, shelf, shelfState);
      const weighted = filterSample(shelfOutput, highPass, highPassState);
      const squared = weighted * weighted;
      runningEnergy += squared - (energyWindow[windowIndex] ?? 0);
      energyWindow[windowIndex] = squared;
      windowIndex = (windowIndex + 1) % blockLength;

      if (index === nextBlockEnd && blockIndex < blockCount) {
        // Mono and stereo have unit channel weights in BS.1770. Layout-aware
        // surround weighting is intentionally deferred until channel labels
        // accompany decoded PCM rather than being guessed from channel count.
        blockEnergies[blockIndex] += runningEnergy / blockLength;
        blockIndex += 1;
        nextBlockEnd += stepLength;
      }
    }
  }

  const absoluteGated = [...blockEnergies].filter((energy) => loudnessFromEnergy(energy) > -70);
  if (absoluteGated.length === 0) return -120;
  const absoluteMean = absoluteGated.reduce((sum, energy) => sum + energy, 0) / absoluteGated.length;
  const relativeThreshold = loudnessFromEnergy(absoluteMean) - 10;
  const relativeGated = absoluteGated.filter((energy) => loudnessFromEnergy(energy) > relativeThreshold);
  if (relativeGated.length === 0) return -120;
  const integratedEnergy = relativeGated.reduce((sum, energy) => sum + energy, 0) / relativeGated.length;
  return clamp(loudnessFromEnergy(integratedEnergy), -120, 6);
}

function calculateEnvelope(mono: Float32Array, sampleRateHz: number, minimumBpm: number, maximumBpm: number): EnvelopeAnalysis {
  const hopSize = sampleRateHz >= 32_000 ? 512 : 256;
  const frameCount = Math.floor(mono.length / hopSize);
  const energy = new Float64Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sumSquares = 0;
    const start = frame * hopSize;
    for (let index = start; index < start + hopSize; index += 1) {
      const sample = mono[index] ?? 0;
      sumSquares += sample * sample;
    }
    energy[frame] = Math.sqrt(sumSquares / hopSize);
  }
  const values = new Float64Array(frameCount);
  let previousSmoothed = energy[0] ?? 0;
  for (let frame = 1; frame < frameCount; frame += 1) {
    const current = energy[frame] ?? 0;
    values[frame] = Math.max(0, current - previousSmoothed);
    previousSmoothed = previousSmoothed * 0.75 + current * 0.25;
  }
  let mean = 0;
  for (const value of values) mean += value;
  mean /= Math.max(1, values.length);
  let variance = 0;
  for (const value of values) variance += (value - mean) ** 2;
  const deviation = Math.sqrt(variance / Math.max(1, values.length));
  const threshold = mean + deviation * 0.8;
  const refractoryFrames = Math.max(1, Math.floor((60 / maximumBpm) * (sampleRateHz / hopSize) * 0.65));
  const peaks: Array<{ index: number; strength: number }> = [];
  let lastPeak = -refractoryFrames;
  for (let frame = 2; frame < values.length - 2; frame += 1) {
    const value = values[frame] ?? 0;
    if (
      value >= threshold &&
      value >= (values[frame - 1] ?? 0) &&
      value > (values[frame + 1] ?? 0) &&
      frame - lastPeak >= refractoryFrames
    ) {
      peaks.push({ index: frame, strength: deviation > 0 ? (value - mean) / deviation : 0 });
      lastPeak = frame;
    }
  }
  // Avoid returning thousands of insignificant events from noisy material.
  const maximumPeaks = Math.ceil((mono.length / sampleRateHz) * (maximumBpm / 60) * 2);
  return { values, hopSize, peaks: peaks.slice(0, maximumPeaks) };
}

function median(values: number[]): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2 : sorted[middle];
}

function foldTempo(bpm: number, minimumBpm: number, maximumBpm: number): number {
  let folded = bpm;
  while (folded < minimumBpm) folded *= 2;
  while (folded > maximumBpm) folded /= 2;
  return folded;
}

function estimateBpm(
  envelope: EnvelopeAnalysis,
  sampleRateHz: number,
  minimumBpm: number,
  maximumBpm: number,
): number | null {
  if (envelope.values.length < 8) return null;
  const secondsPerFrame = envelope.hopSize / sampleRateHz;
  const intervals = envelope.peaks
    .slice(1)
    .map((peak, index) => (peak.index - (envelope.peaks[index]?.index ?? peak.index)) * secondsPerFrame)
    .filter((interval) => interval >= 60 / maximumBpm * 0.75 && interval <= 60 / minimumBpm * 2.1);
  const interval = median(intervals);
  if (interval !== undefined && interval > 0) {
    return round(foldTempo(60 / interval, minimumBpm, maximumBpm), 3);
  }

  let signalPower = 0;
  for (const value of envelope.values) signalPower += value * value;
  if (signalPower <= 1e-12) return null;
  const minimumLag = Math.max(1, Math.floor(60 / maximumBpm / secondsPerFrame));
  const maximumLag = Math.min(envelope.values.length - 2, Math.ceil(60 / minimumBpm / secondsPerFrame));
  let bestLag = 0;
  let bestScore = -Infinity;
  for (let lag = minimumLag; lag <= maximumLag; lag += 1) {
    let numerator = 0;
    let leftPower = 0;
    let rightPower = 0;
    for (let index = lag; index < envelope.values.length; index += 1) {
      const left = envelope.values[index] ?? 0;
      const right = envelope.values[index - lag] ?? 0;
      numerator += left * right;
      leftPower += left * left;
      rightPower += right * right;
    }
    const correlation = numerator / Math.sqrt(Math.max(1e-18, leftPower * rightPower));
    const shorterLagPreference = 1 + ((maximumLag - lag) / Math.max(1, maximumLag - minimumLag)) * 0.015;
    const score = correlation * shorterLagPreference;
    if (score > bestScore) {
      bestScore = score;
      bestLag = lag;
    }
  }
  return bestLag > 0 && bestScore > 0.08 ? round(60 / (bestLag * secondsPerFrame), 3) : null;
}

function createBeatGrid(
  bpm: number | null,
  peaks: EnvelopeAnalysis["peaks"],
  hopSize: number,
  sampleRateHz: number,
  durationSeconds: number,
): number[] {
  if (bpm === null) return [];
  const interval = 60 / bpm;
  const firstOnset = peaks[0] ? (peaks[0].index * hopSize) / sampleRateHz : 0;
  const phase = firstOnset % interval;
  const grid: number[] = [];
  for (let time = phase; time < durationSeconds; time += interval) grid.push(round(time));
  return grid;
}

function goertzelPower(samples: Float32Array, offset: number, length: number, frequency: number, sampleRateHz: number): number {
  const coefficient = 2 * Math.cos((2 * Math.PI * frequency) / sampleRateHz);
  let previous = 0;
  let previousPrevious = 0;
  for (let index = 0; index < length; index += 1) {
    const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / Math.max(1, length - 1));
    const current = (samples[offset + index] ?? 0) * window + coefficient * previous - previousPrevious;
    previousPrevious = previous;
    previous = current;
  }
  return Math.max(0, previous * previous + previousPrevious * previousPrevious - coefficient * previous * previousPrevious);
}

function calculateSpectralProfile(mono: Float32Array, sampleRateHz: number): SpectralProfile {
  const windowLength = Math.min(2_048, mono.length);
  const windowCount = Math.min(8, Math.max(1, Math.floor(mono.length / windowLength)));
  const nyquistLimit = Math.min(16_000, sampleRateHz * 0.45);
  const frequencies = Array.from({ length: 48 }, (_, index) => {
    const position = index / 47;
    return 40 * (nyquistLimit / 40) ** position;
  });
  let low = 0;
  let mid = 0;
  let high = 0;
  let weightedFrequency = 0;
  let total = 0;
  for (let windowIndex = 0; windowIndex < windowCount; windowIndex += 1) {
    const offset = Math.min(
      mono.length - windowLength,
      Math.floor(((mono.length - windowLength) * windowIndex) / Math.max(1, windowCount - 1)),
    );
    for (const frequency of frequencies) {
      const power = goertzelPower(mono, offset, windowLength, frequency, sampleRateHz);
      if (frequency < 250) low += power;
      else if (frequency < 2_000) mid += power;
      else high += power;
      weightedFrequency += frequency * power;
      total += power;
    }
  }
  if (total <= 1e-18) return { lowEnergy: 0, midEnergy: 0, highEnergy: 0, centroidHz: 0 };
  return {
    lowEnergy: low / total,
    midEnergy: mid / total,
    highEnergy: high / total,
    centroidHz: weightedFrequency / total,
  };
}

const PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;
const MAJOR_KEY_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88] as const;
const MINOR_KEY_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17] as const;

function profileCorrelation(chroma: Float64Array, template: readonly number[], root: number): number {
  let chromaMean = 0;
  let templateMean = 0;
  for (let pitchClass = 0; pitchClass < 12; pitchClass += 1) {
    chromaMean += chroma[pitchClass] ?? 0;
    templateMean += template[pitchClass] ?? 0;
  }
  chromaMean /= 12;
  templateMean /= 12;
  let numerator = 0;
  let chromaPower = 0;
  let templatePower = 0;
  for (let pitchClass = 0; pitchClass < 12; pitchClass += 1) {
    const chromaValue = (chroma[pitchClass] ?? 0) - chromaMean;
    const templateIndex = (pitchClass - root + 12) % 12;
    const templateValue = (template[templateIndex] ?? 0) - templateMean;
    numerator += chromaValue * templateValue;
    chromaPower += chromaValue * chromaValue;
    templatePower += templateValue * templateValue;
  }
  const denominator = Math.sqrt(chromaPower * templatePower);
  return denominator > 1e-18 ? numerator / denominator : 0;
}

function estimateMusicalKey(mono: Float32Array, sampleRateHz: number): string | null {
  // Representative windows keep work constant for a three-minute track. Chroma is measured from PCM only.
  const windowLength = Math.min(4_096, mono.length);
  if (windowLength < 512) return null;
  const windowCount = Math.min(12, Math.max(1, Math.floor(mono.length / windowLength)));
  const chroma = new Float64Array(12);
  let analyzedWindows = 0;
  for (let windowIndex = 0; windowIndex < windowCount; windowIndex += 1) {
    const offset = Math.min(
      mono.length - windowLength,
      Math.floor(((mono.length - windowLength) * windowIndex) / Math.max(1, windowCount - 1)),
    );
    let windowPower = 0;
    for (let index = 0; index < windowLength; index += 1) {
      const sample = mono[offset + index] ?? 0;
      windowPower += sample * sample;
    }
    if (windowPower / windowLength < 1e-8) continue;
    analyzedWindows += 1;
    // C2 through B6 captures harmonic context while avoiding unstable sub-bass estimates.
    for (let midiNote = 36; midiNote <= 95; midiNote += 1) {
      const frequency = 440 * 2 ** ((midiNote - 69) / 12);
      if (frequency >= sampleRateHz * 0.45) break;
      const power = goertzelPower(mono, offset, windowLength, frequency, sampleRateHz);
      // Square-root compression stops one loud partial from overwhelming the chord profile.
      chroma[midiNote % 12] += Math.sqrt(power);
    }
  }
  if (analyzedWindows === 0) return null;
  let total = 0;
  let maximum = 0;
  for (const energy of chroma) {
    total += energy;
    maximum = Math.max(maximum, energy);
  }
  if (total <= 1e-8 || maximum <= 0) return null;
  const activePitchClasses = [...chroma].filter((energy) => energy >= maximum * 0.12).length;
  const tonalContrast = maximum / (total / 12);
  if (activePitchClasses < 3 || tonalContrast < 1.35) return null;

  const candidates: Array<{ root: number; mode: "major" | "minor"; score: number }> = [];
  for (let root = 0; root < 12; root += 1) {
    candidates.push({ root, mode: "major", score: profileCorrelation(chroma, MAJOR_KEY_PROFILE, root) });
    candidates.push({ root, mode: "minor", score: profileCorrelation(chroma, MINOR_KEY_PROFILE, root) });
  }
  candidates.sort((left, right) => right.score - left.score);
  const best = candidates[0];
  const runnerUp = candidates[1];
  if (!best || !runnerUp || best.score < 0.5 || best.score - runnerUp.score < 0.08) return null;
  return `${PITCH_CLASS_NAMES[best.root]} ${best.mode}`;
}

interface SectionFeature {
  start: number;
  end: number;
  rms: number;
  zeroCrossingRate: number;
}

function calculateSectionFeatures(mono: Float32Array, sampleRateHz: number): SectionFeature[] {
  const windowSamples = Math.max(1, Math.round(sampleRateHz));
  const features: SectionFeature[] = [];
  for (let start = 0; start < mono.length; start += windowSamples) {
    const end = Math.min(mono.length, start + windowSamples);
    let sumSquares = 0;
    let crossings = 0;
    let previous = mono[start] ?? 0;
    for (let index = start; index < end; index += 1) {
      const value = mono[index] ?? 0;
      sumSquares += value * value;
      if ((value >= 0) !== (previous >= 0)) crossings += 1;
      previous = value;
    }
    const count = Math.max(1, end - start);
    features.push({
      start: start / sampleRateHz,
      end: end / sampleRateHz,
      rms: Math.sqrt(sumSquares / count),
      zeroCrossingRate: crossings / count,
    });
  }
  return features;
}

function classifySection(
  index: number,
  totalSections: number,
  meanRms: number,
  previousRms: number | undefined,
  globalMeanRms: number,
): MeasuredSectionType {
  if (index === 0) return "intro";
  if (index === totalSections - 1) return "outro";
  if (meanRms >= globalMeanRms * 1.18) return "drop";
  if (meanRms <= globalMeanRms * 0.68) return "breakdown";
  if (previousRms !== undefined && meanRms > previousRms * 1.12) return "build";
  return "groove";
}

function detectSections(mono: Float32Array, sampleRateHz: number, durationSeconds: number): MeasuredSection[] {
  const features = calculateSectionFeatures(mono, sampleRateHz);
  if (features.length <= 2) {
    const meanRms = features.reduce((sum, value) => sum + value.rms, 0) / Math.max(1, features.length);
    return [{ type: "intro", start: 0, end: durationSeconds, meanRms }];
  }
  const changes = features.map((feature, index) => {
    if (index === 0) return 0;
    const previous = features[index - 1] ?? feature;
    const energyChange = Math.abs(Math.log10((feature.rms + 1e-5) / (previous.rms + 1e-5)));
    const textureChange = Math.abs(feature.zeroCrossingRate - previous.zeroCrossingRate) * 4;
    return energyChange + textureChange;
  });
  const meanChange = changes.reduce((sum, value) => sum + value, 0) / changes.length;
  const variance = changes.reduce((sum, value) => sum + (value - meanChange) ** 2, 0) / changes.length;
  const threshold = Math.max(0.12, meanChange + Math.sqrt(variance) * 0.65);
  const minimumSectionSeconds = Math.max(2, Math.min(8, durationSeconds / 10));
  const boundaries = [0];
  for (let index = 1; index < changes.length - 1; index += 1) {
    const time = features[index]?.start ?? index;
    const lastBoundary = boundaries[boundaries.length - 1] ?? 0;
    if (
      changes[index]! >= threshold &&
      changes[index]! >= (changes[index - 1] ?? 0) &&
      changes[index]! > (changes[index + 1] ?? 0) &&
      time - lastBoundary >= minimumSectionSeconds &&
      durationSeconds - time >= minimumSectionSeconds
    ) {
      boundaries.push(time);
    }
  }
  boundaries.push(durationSeconds);

  const rawSections = boundaries.slice(0, -1).map((start, index) => {
    const end = boundaries[index + 1] ?? durationSeconds;
    const included = features.filter((feature) => feature.start < end && feature.end > start);
    const meanRms = included.reduce((sum, value) => sum + value.rms, 0) / Math.max(1, included.length);
    return { start, end, meanRms };
  });
  const globalMeanRms = rawSections.reduce((sum, section) => sum + section.meanRms, 0) / rawSections.length;
  return rawSections.map((section, index) => ({
    type: classifySection(index, rawSections.length, section.meanRms, rawSections[index - 1]?.meanRms, globalMeanRms),
    start: round(section.start),
    end: round(section.end),
    meanRms: section.meanRms,
  }));
}

function recommendScene(profile: SpectralProfile, bpm: number | null, loudnessLufs: number): { scene: VisualSceneId; intensity: number } {
  const scene: VisualSceneId =
    profile.lowEnergy > profile.highEnergy * 1.15 ? "tunnel" : profile.highEnergy > profile.lowEnergy * 1.1 ? "bloom" : "terrain";
  const tempoEnergy = bpm === null ? 0.45 : clamp((bpm - 60) / 140, 0, 1);
  const loudnessEnergy = clamp((loudnessLufs + 36) / 30, 0, 1);
  return { scene, intensity: round(clamp(0.25 + tempoEnergy * 0.35 + loudnessEnergy * 0.4, 0.25, 1), 3) };
}

export function analyzeDecodedPcm(audio: DecodedPcmAudio, options: AnalysisOptions = {}): AudioAnalysisResult {
  const sampleCount = validateDecodedPcm(audio);
  const minimumBpm = clamp(options.minimumBpm ?? 60, 40, 200);
  const maximumBpm = clamp(options.maximumBpm ?? 200, minimumBpm + 1, 260);
  const mono = createPhaseSafeAnalysisSignal(audio, sampleCount);
  const durationSeconds = sampleCount / audio.sampleRateHz;
  const loudnessLufs = calculateIntegratedLoudness(audio, sampleCount);
  const envelope = calculateEnvelope(mono, audio.sampleRateHz, minimumBpm, maximumBpm);
  const bpm = estimateBpm(envelope, audio.sampleRateHz, minimumBpm, maximumBpm);
  const key = estimateMusicalKey(mono, audio.sampleRateHz);
  const spectralProfile = calculateSpectralProfile(mono, audio.sampleRateHz);
  const recommendation = recommendScene(spectralProfile, bpm, loudnessLufs);
  const maximumOnsets = 4_096;
  const onsetMap = envelope.peaks.slice(0, maximumOnsets).map((peak) => ({
    timeSeconds: round((peak.index * envelope.hopSize) / audio.sampleRateHz),
    strength: round(peak.strength, 4),
  }));
  return {
    durationSeconds,
    sampleRateHz: audio.sampleRateHz,
    channels: audio.channels.length,
    loudnessLufs: round(loudnessLufs, 3),
    bpm,
    key,
    waveform: calculateWaveform(mono, audio.sampleRateHz, options.waveformBuckets ?? 256),
    onsetMap,
    beatGridSeconds: createBeatGrid(bpm, envelope.peaks, envelope.hopSize, audio.sampleRateHz, durationSeconds),
    spectralProfile: {
      lowEnergy: round(spectralProfile.lowEnergy, 6),
      midEnergy: round(spectralProfile.midEnergy, 6),
      highEnergy: round(spectralProfile.highEnergy, 6),
      centroidHz: round(spectralProfile.centroidHz, 3),
    },
    sections: detectSections(mono, audio.sampleRateHz, durationSeconds),
    visualMapping: {
      bass: "camera_displacement",
      kick: "radial_pulse",
      highFrequencyEnergy: "particle_density",
      sectionChange: "scene_transition",
    },
    recommendedScene: recommendation.scene,
    visualIntensity: recommendation.intensity,
  };
}

export const analyzePcm = analyzeDecodedPcm;
