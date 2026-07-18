import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDirectory = process.env.MUSICA_VJ_SAMPLE_OUTPUT
  ? resolve(process.env.MUSICA_VJ_SAMPLE_OUTPUT)
  : resolve(root, "samples");
mkdirSync(outputDirectory, { recursive: true });

const width = 360;
const height = 640;
const fps = 30;
const duration = 6;

const audio = [
  "aevalsrc=exprs=",
  "'0.30*sin(2*PI*55*t)*(0.35+0.65*gt(sin(2*PI*2*t)\\,0))+0.14*sin(2*PI*220*t)+0.09*sin(2*PI*330*t)",
  "|0.30*sin(2*PI*56*t)*(0.35+0.65*gt(sin(2*PI*2*t+0.2)\\,0))+0.14*sin(2*PI*223*t)+0.09*sin(2*PI*333*t)'",
  `:s=48000:d=${duration}`,
].join("");

const commonOutput = (path) => [
  "-map", "[video]",
  "-map", "[audio]",
  "-t", String(duration),
  "-r", String(fps),
  "-c:v", "libx264",
  "-preset", "medium",
  "-crf", "33",
  "-profile:v", "high",
  "-pix_fmt", "yuv420p",
  "-g", String(fps * 2),
  "-c:a", "aac",
  "-b:a", "192k",
  "-ar", "48000",
  "-threads", "1",
  "-fflags", "+bitexact",
  "-flags:v", "+bitexact",
  "-flags:a", "+bitexact",
  "-map_metadata", "-1",
  "-movflags", "+faststart",
  path,
];

const samples = [
  {
    path: resolve(outputDirectory, "signal-bloom-vertical.mp4"),
    gradient: `gradients=s=${width}x${height}:r=${fps}:c0=0x020308:c1=0x120019:c2=0x001712:c3=0x11062b:n=4:type=spiral:speed=0.018:seed=6101:d=${duration}`,
    graph: [
      "[0:a]aecho=0.8:0.7:240|480:0.24|0.12,alimiter=limit=0.85,asplit=3[audio][scopein][wavein]",
      `[scopein]avectorscope=s=${width}x${height}:r=${fps}:mode=polar:draw=aaline:scale=sqrt:rc=255:gc=90:bc=205:rf=8:gf=12:bf=5:zoom=1.55[scope]`,
      `[wavein]showwaves=s=${width}x120:r=${fps}:mode=cline:colors=0x75f4c5|0xff4f86:scale=sqrt,format=rgba,colorkey=0x000000:0.08:0.12[wave]`,
      `[1:v][scope]blend=all_mode=screen:all_opacity=0.78[base]`,
      `[base][wave]overlay=0:${height - 150}:shortest=1,noise=alls=3:allf=t+u:all_seed=6101,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='MUSICA  /  SIGNAL BLOOM':fontcolor=white@0.72:fontsize=10:x=20:y=24,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='112 BPM   LIVE GENERATIVE SET':fontcolor=0x75f4c5@0.85:fontsize=7:x=20:y=42,format=yuv420p[video]`,
    ].join(";"),
  },
  {
    path: resolve(outputDirectory, "spectral-field-vertical.mp4"),
    gradient: `gradients=s=${width}x${height}:r=${fps}:c0=0x02050a:c1=0x001d28:c2=0x170422:c3=0x061438:n=4:type=circular:speed=0.012:seed=6102:d=${duration}`,
    graph: [
      "[0:a]aecho=0.8:0.65:180|360:0.18|0.09,alimiter=limit=0.85,asplit=3[audio][specin][scopein]",
      `[specin]showspectrum=s=${width}x${height}:fps=${fps}:slide=scroll:mode=combined:color=terrain:scale=sqrt:fscale=log:saturation=1.5:opacity=0.68[spec]`,
      `[scopein]avectorscope=s=${width}x${height}:r=${fps}:mode=lissajous_xy:draw=aaline:scale=sqrt:rc=80:gc=255:bc=215:rf=5:gf=12:bf=8:zoom=1.35[scope]`,
      `[1:v][spec]blend=all_mode=screen:all_opacity=0.22[layer]`,
      `[layer][scope]blend=all_mode=screen:all_opacity=0.68,noise=alls=2:allf=t+u:all_seed=6102,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='MUSICA  /  SPECTRAL FIELD':fontcolor=white@0.72:fontsize=10:x=20:y=24,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='48 KHZ   AUDIO REACTIVE':fontcolor=0x70a9ff@0.9:fontsize=7:x=20:y=42,format=yuv420p[video]`,
    ].join(";"),
  },
];

for (const sample of samples) {
  const args = [
    "-y", "-hide_banner", "-loglevel", "warning",
    "-filter_complex_threads", "1",
    "-f", "lavfi", "-i", audio,
    "-f", "lavfi", "-i", sample.gradient,
    "-filter_complex", sample.graph,
    ...commonOutput(sample.path),
  ];
  const result = spawnSync("ffmpeg", args, { stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

console.log(`Rendered ${samples.length} reproducible sample fixtures to ${outputDirectory}`);
