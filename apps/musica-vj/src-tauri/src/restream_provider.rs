use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;

const DEFAULT_RESTREAM_URL: &str = "rtmps://live.restream.io/live";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RestreamStartRequest {
    pub ingest_url: String,
    pub stream_key: String,
    pub source: String,
    pub video_bitrate_kbps: u32,
    pub fps: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RestreamStatus {
    pub available: bool,
    pub active: bool,
    pub source: Option<String>,
    pub encoder: Option<String>,
    pub reason: Option<String>,
}

struct ActiveBroadcast {
    child: Child,
    stdin: Option<ChildStdin>,
    source: String,
}

#[derive(Default)]
pub(crate) struct RestreamProvider {
    active: Mutex<Option<ActiveBroadcast>>,
}

fn validate_restream_url(value: &str) -> Result<String, String> {
    let value = value.trim().trim_end_matches('/');
    let authority_and_path = value
        .strip_prefix("rtmps://")
        .or_else(|| value.strip_prefix("rtmp://"))
        .ok_or_else(|| "Restream ingest URL must use RTMPS or RTMP".to_string())?;
    let authority = authority_and_path.split('/').next().unwrap_or_default();
    if authority.is_empty() || authority.contains('@') {
        return Err("Restream ingest URL has an invalid host".to_string());
    }
    let host = authority
        .split(':')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if host != "restream.io" && !host.ends_with(".restream.io") {
        return Err("Only official restream.io ingest hosts are allowed".to_string());
    }
    Ok(value.to_string())
}

fn validate_stream_key(value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.len() < 8
        || value.len() > 512
        || value.chars().any(char::is_whitespace)
        || value.contains('/')
    {
        return Err("Restream stream key is invalid".to_string());
    }
    Ok(value.to_string())
}

fn ffmpeg_version() -> Result<String, String> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map_err(|_| "FFmpeg was not found on PATH".to_string())?;
    if !output.status.success() {
        return Err("FFmpeg is installed but unavailable".to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or("ffmpeg")
        .chars()
        .take(120)
        .collect())
}

fn status_for(provider: &RestreamProvider) -> RestreamStatus {
    let source = provider
        .active
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|broadcast| broadcast.source.clone()));
    match ffmpeg_version() {
        Ok(encoder) => RestreamStatus {
            available: true,
            active: source.is_some(),
            source,
            encoder: Some(encoder),
            reason: None,
        },
        Err(reason) => RestreamStatus {
            available: false,
            active: false,
            source: None,
            encoder: None,
            reason: Some(reason),
        },
    }
}

#[tauri::command]
pub(crate) fn restream_status(state: tauri::State<'_, RestreamProvider>) -> RestreamStatus {
    status_for(&state)
}

#[tauri::command]
pub(crate) fn restream_start(
    request: RestreamStartRequest,
    state: tauri::State<'_, RestreamProvider>,
) -> Result<RestreamStatus, String> {
    ffmpeg_version()?;
    let ingest_url = if request.ingest_url.trim().is_empty() {
        DEFAULT_RESTREAM_URL.to_string()
    } else {
        validate_restream_url(&request.ingest_url)?
    };
    let stream_key = validate_stream_key(&request.stream_key)?;
    if request.source != "program" && request.source != "window" {
        return Err("Restream source must be program or window".to_string());
    }
    let bitrate = request.video_bitrate_kbps.clamp(1_500, 8_000);
    let fps = request.fps.clamp(24, 60);
    let keyframe_interval = fps * 2;
    let destination = format!("{ingest_url}/{stream_key}");
    let fps_value = fps.to_string();
    let keyframe_value = keyframe_interval.to_string();
    let bitrate_value = format!("{bitrate}k");
    let buffer_value = format!("{}k", bitrate * 2);

    let mut active = state
        .active
        .lock()
        .map_err(|_| "Restream encoder state is unavailable".to_string())?;
    if active.is_some() {
        return Err("A Restream broadcast is already active".to_string());
    }
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "+genpts+discardcorrupt",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-r",
            &fps_value,
            "-g",
            &keyframe_value,
            "-keyint_min",
            &keyframe_value,
            "-sc_threshold",
            "0",
            "-b:v",
            &bitrate_value,
            "-maxrate",
            &bitrate_value,
            "-bufsize",
            &buffer_value,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-f",
            "flv",
            &destination,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("Could not start FFmpeg: {error}"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "FFmpeg input pipe is unavailable".to_string())?;
    *active = Some(ActiveBroadcast {
        child,
        stdin: Some(stdin),
        source: request.source,
    });
    drop(active);
    Ok(status_for(&state))
}

#[tauri::command]
pub(crate) fn restream_push_chunk(
    request: tauri::ipc::Request<'_>,
    state: tauri::State<'_, RestreamProvider>,
) -> Result<(), String> {
    let chunk = match request.body() {
        tauri::ipc::InvokeBody::Raw(bytes) => bytes,
        tauri::ipc::InvokeBody::Json(_) => {
            return Err("Restream media chunks must use binary IPC".to_string())
        }
    };
    if chunk.is_empty() {
        return Ok(());
    }
    if chunk.len() > 16 * 1024 * 1024 {
        return Err("Restream media chunk exceeds the 16 MB input limit".to_string());
    }
    let mut active = state
        .active
        .lock()
        .map_err(|_| "Restream encoder state is unavailable".to_string())?;
    let broadcast = active
        .as_mut()
        .ok_or_else(|| "No Restream broadcast is active".to_string())?;
    if let Some(status) = broadcast
        .child
        .try_wait()
        .map_err(|error| format!("Could not inspect FFmpeg: {error}"))?
    {
        return Err(format!("FFmpeg stopped unexpectedly with {status}"));
    }
    broadcast
        .stdin
        .as_mut()
        .ok_or_else(|| "FFmpeg input pipe is closed".to_string())?
        .write_all(&chunk)
        .map_err(|error| format!("Could not send media to FFmpeg: {error}"))
}

#[tauri::command]
pub(crate) fn restream_stop(
    state: tauri::State<'_, RestreamProvider>,
) -> Result<RestreamStatus, String> {
    let mut active = state
        .active
        .lock()
        .map_err(|_| "Restream encoder state is unavailable".to_string())?;
    if let Some(mut broadcast) = active.take() {
        broadcast.stdin.take();
        if broadcast
            .child
            .try_wait()
            .map_err(|error| format!("Could not inspect FFmpeg: {error}"))?
            .is_none()
        {
            broadcast
                .child
                .kill()
                .map_err(|error| format!("Could not stop FFmpeg: {error}"))?;
        }
        let _ = broadcast.child.wait();
    }
    drop(active);
    Ok(status_for(&state))
}

#[cfg(test)]
mod tests {
    use super::{validate_restream_url, validate_stream_key};

    #[test]
    fn accepts_official_restream_ingest_urls() {
        assert!(validate_restream_url("rtmps://live.restream.io/live").is_ok());
        assert!(validate_restream_url("rtmp://eu.restream.io/live/").is_ok());
    }

    #[test]
    fn rejects_non_restream_hosts_and_unsafe_keys() {
        assert!(validate_restream_url("rtmps://example.com/live").is_err());
        assert!(validate_restream_url("https://live.restream.io/live").is_err());
        assert!(validate_stream_key("short").is_err());
        assert!(validate_stream_key("re_valid_stream_key_123").is_ok());
        assert!(validate_stream_key("re_key/extra").is_err());
    }
}
