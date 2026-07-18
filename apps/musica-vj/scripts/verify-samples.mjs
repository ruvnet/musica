import { readdirSync } from "node:fs";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sampleDirectory = resolve(root, "samples");
const files = readdirSync(sampleDirectory)
  .filter((file) => extname(file) === ".mp4")
  .map((file) => resolve(sampleDirectory, file));

if (files.length < 2) throw new Error("Expected at least two MP4 sample renders");

for (const file of files) {
  const result = spawnSync("ffprobe", [
    "-v", "error",
    "-count_frames",
    "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels,duration,nb_read_frames",
    "-of", "json",
    file,
  ], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(`ffprobe rejected ${file}: ${result.stderr}`);
  const probe = JSON.parse(result.stdout);
  const video = probe.streams.find((stream) => stream.codec_type === "video");
  const audio = probe.streams.find((stream) => stream.codec_type === "audio");
  const duration = Number(probe.format.duration);
  if (video?.codec_name !== "h264" || video.width !== 360 || video.height !== 640 || video.r_frame_rate !== "30/1") {
    throw new Error(`${file} violates the H.264 360x640 at 30 fps preview contract`);
  }
  if (audio?.codec_name !== "aac" || audio.sample_rate !== "48000" || audio.channels !== 2) {
    throw new Error(`${file} violates the AAC 48 kHz stereo contract`);
  }
  if (duration < 5.95 || duration > 6.05) throw new Error(`${file} has invalid duration ${duration}`);
  if (Number(video.nb_read_frames) !== 180) throw new Error(`${file} does not contain exactly 180 delivered video frames`);
  const drift = Math.abs(Number(video.duration) - Number(audio.duration));
  if (!Number.isFinite(drift) || drift > 0.02) throw new Error(`${file} audio/video drift ${drift} exceeds 20 ms`);
  if (Number(probe.format.size) > 1_000_000) throw new Error(`${file} exceeds the one MB repository preview budget`);
}

console.log(`Verified ${files.length} social video samples`);
