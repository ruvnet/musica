#![cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]

use serde::{Deserialize, Serialize};
use tauri::AppHandle;
use thiserror::Error;

const PROTOCOL_VERSION: u8 = 1;
const MAX_MESSAGE_BYTES: usize = 4_096;
const TOKEN_BYTES: usize = 64;
const MAX_CLOCK_SKEW_MS: u64 = 30_000;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMessage {
    v: u8,
    seq: u64,
    ts_ms: i64,
    action: String,
    #[serde(default)]
    value: Option<f64>,
    token: String,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ValidatedMessage {
    seq: u64,
    ts_ms: i64,
    action: String,
    value: Option<f64>,
    token: String,
}

#[derive(Clone, Serialize)]
#[allow(dead_code)]
struct ControllerEvent {
    action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<f64>,
    source: &'static str,
}

#[derive(Default)]
struct ReplayGuard {
    last_seq: u64,
}

impl ReplayGuard {
    fn accept(&mut self, seq: u64, ts_ms: i64, now_ms: i64) -> Result<(), BridgeError> {
        if seq == 0 || seq <= self.last_seq {
            return Err(BridgeError::Replay);
        }
        if ts_ms.abs_diff(now_ms) > MAX_CLOCK_SKEW_MS {
            return Err(BridgeError::Stale);
        }
        self.last_seq = seq;
        Ok(())
    }
}

fn parse_message(bytes: &[u8]) -> Result<ValidatedMessage, BridgeError> {
    if bytes.is_empty() || bytes.len() > MAX_MESSAGE_BYTES {
        return Err(BridgeError::InvalidMessage);
    }
    let message: WireMessage =
        serde_json::from_slice(bytes).map_err(|_| BridgeError::InvalidMessage)?;
    if message.v != PROTOCOL_VERSION || message.seq == 0 || !valid_wire_token(&message.token) {
        return Err(BridgeError::InvalidMessage);
    }
    let value = validate_action(&message.action, message.value)?;
    Ok(ValidatedMessage {
        seq: message.seq,
        ts_ms: message.ts_ms,
        action: message.action,
        value,
        token: message.token,
    })
}

fn valid_wire_token(token: &str) -> bool {
    token.len() == TOKEN_BYTES
        && token
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn validate_action(action: &str, value: Option<f64>) -> Result<Option<f64>, BridgeError> {
    let value = match action {
        "transport.toggle" | "transport.record" | "track.previous" | "track.next"
        | "track.mute" | "track.solo" | "visual.previous" | "visual.next" | "tempo.tap" => {
            if value.is_some_and(|value| !value.is_finite() || value != 0.0) {
                return Err(BridgeError::InvalidAction);
            }
            None
        }
        "visual.scene.select" | "performance.template.select" => {
            let value = value.ok_or(BridgeError::InvalidAction)?;
            if !value.is_finite() || value.fract() != 0.0 || !(0.0..=15.0).contains(&value) {
                return Err(BridgeError::InvalidAction);
            }
            Some(value)
        }
        "track.trigger" => {
            let value = value.ok_or(BridgeError::InvalidAction)?;
            if !value.is_finite() || value.fract() != 0.0 || !(0.0..=5.0).contains(&value) {
                return Err(BridgeError::InvalidAction);
            }
            Some(value)
        }
        "master.delta"
        | "visual.intensity.delta"
        | "visual.sculpture.delta"
        | "visual.motion.delta"
        | "visual.atmosphere.delta"
        | "visual.ribbon.delta"
        | "visual.temporal.speed.delta"
        | "visual.temporal.strobe.delta"
        | "visual.temporal.trail.delta"
        | "visual.temporal.morph.delta"
        | "visual.temporal.camera.delta"
        | "visual.temporal.phase.delta"
        | "tempo.delta" => {
            let value = value.ok_or(BridgeError::InvalidAction)?;
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(BridgeError::InvalidAction);
            }
            Some(value)
        }
        _ => return Err(BridgeError::InvalidAction),
    };
    Ok(value)
}

#[derive(Debug, Error)]
#[allow(dead_code)]
enum BridgeError {
    #[error("controller bridge is unavailable: {0}")]
    Unavailable(&'static str),
    #[error("controller message is invalid")]
    InvalidMessage,
    #[error("controller action is invalid")]
    InvalidAction,
    #[error("controller authentication failed")]
    Authentication,
    #[error("controller message was replayed")]
    Replay,
    #[error("controller message timestamp is stale")]
    Stale,
    #[error("controller bridge I/O failed")]
    Io(#[from] std::io::Error),
}

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::{
        env,
        fs::{self, File, OpenOptions},
        io::{Read, Write},
        os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt},
        os::unix::net::{UnixListener as StdUnixListener, UnixStream as StdUnixStream},
        path::{Path, PathBuf},
        sync::Arc,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use rand::{rngs::OsRng, RngCore};
    use subtle::ConstantTimeEq;
    use tauri::{Emitter, Manager};
    use tokio::{
        io::AsyncReadExt,
        net::{UnixListener, UnixStream},
        time::timeout,
    };

    const READ_TIMEOUT: Duration = Duration::from_millis(150);

    pub(super) fn start(app: &AppHandle) -> Result<(), BridgeError> {
        let user = env::var("USER").map_err(|_| BridgeError::Unavailable("USER is not set"))?;
        if !valid_user(&user) {
            return Err(BridgeError::Unavailable("USER is invalid"));
        }

        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|_| BridgeError::Unavailable("application data path is unavailable"))?;
        let token_path = app_data_dir.join("controller.token");
        let token = Arc::<[u8]>::from(load_or_create_token(&token_path)?);
        let socket_path = PathBuf::from(format!("/tmp/musica-vj-{user}.sock"));
        remove_stale_socket(&socket_path)?;
        let listener = StdUnixListener::bind(&socket_path)?;
        listener.set_nonblocking(true)?;
        fs::set_permissions(&socket_path, fs::Permissions::from_mode(0o600))?;

        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            if let Ok(listener) = UnixListener::from_std(listener) {
                run(listener, app, token).await;
            }
            let _ = remove_owned_socket(&socket_path);
        });
        Ok(())
    }

    async fn run(listener: UnixListener, app: AppHandle, token: Arc<[u8]>) {
        let mut replay = ReplayGuard::default();
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            // Process accepted connections in listener order. Spawning one task
            // per connection can let seq N+1 reach the replay guard before N,
            // which would reorder or drop legitimate controller actions.
            let _ = handle_connection(stream, &app, &token, &mut replay).await;
        }
    }

    async fn handle_connection(
        stream: UnixStream,
        app: &AppHandle,
        token: &[u8],
        replay: &mut ReplayGuard,
    ) -> Result<(), BridgeError> {
        let mut bytes = Vec::with_capacity(512);
        timeout(
            READ_TIMEOUT,
            stream
                .take((MAX_MESSAGE_BYTES + 1) as u64)
                .read_to_end(&mut bytes),
        )
        .await
        .map_err(|_| BridgeError::InvalidMessage)??;
        let message = parse_message(&bytes)?;
        if !bool::from(message.token.as_bytes().ct_eq(token)) {
            return Err(BridgeError::Authentication);
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| BridgeError::Unavailable("system clock is unavailable"))?
            .as_millis()
            .try_into()
            .map_err(|_| BridgeError::Unavailable("system clock is out of range"))?;
        replay.accept(message.seq, message.ts_ms, now_ms)?;

        app.emit(
            "controller://action",
            ControllerEvent {
                action: message.action,
                value: message.value,
                source: "logitech",
            },
        )
        .map_err(|_| BridgeError::Unavailable("event delivery failed"))?;
        Ok(())
    }

    fn valid_user(user: &str) -> bool {
        !user.is_empty()
            && user.len() <= 64
            && !matches!(user, "." | "..")
            && user
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    }

    fn load_or_create_token(path: &Path) -> Result<Vec<u8>, BridgeError> {
        let parent = path
            .parent()
            .ok_or(BridgeError::Unavailable("token path has no parent"))?;
        fs::create_dir_all(parent)?;
        if fs::symlink_metadata(parent)?.file_type().is_symlink() {
            return Err(BridgeError::Unavailable(
                "token directory must not be a symlink",
            ));
        }

        match open_existing_token(path) {
            Ok(mut file) => read_token(&mut file),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => create_token(path),
            Err(error) => Err(error.into()),
        }
    }

    fn open_existing_token(path: &Path) -> std::io::Result<File> {
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)
            .and_then(|file| {
                validate_token_metadata(&file)?;
                Ok(file)
            })
    }

    fn validate_token_metadata(file: &File) -> std::io::Result<()> {
        let metadata = file.metadata()?;
        if !metadata.file_type().is_file()
            || metadata.uid() != unsafe { libc::geteuid() }
            || metadata.mode() & 0o777 != 0o600
            || metadata.nlink() != 1
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "controller token file ownership or mode is unsafe",
            ));
        }
        Ok(())
    }

    fn read_token(file: &mut File) -> Result<Vec<u8>, BridgeError> {
        let mut token = Vec::with_capacity(TOKEN_BYTES);
        file.take((TOKEN_BYTES + 1) as u64)
            .read_to_end(&mut token)?;
        if !valid_wire_token(std::str::from_utf8(&token).unwrap_or_default()) {
            return Err(BridgeError::Unavailable("controller token file is invalid"));
        }
        Ok(token)
    }

    fn create_token(path: &Path) -> Result<Vec<u8>, BridgeError> {
        let mut random = [0_u8; 32];
        OsRng.fill_bytes(&mut random);
        let token = lower_hex(&random);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        file.write_all(token.as_bytes())?;
        file.sync_all()?;
        validate_token_metadata(&file)?;
        Ok(token.into_bytes())
    }

    fn lower_hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
        output
    }

    fn remove_stale_socket(path: &Path) -> Result<(), BridgeError> {
        match fs::symlink_metadata(path) {
            Ok(metadata) => {
                if !metadata.file_type().is_socket() || metadata.uid() != unsafe { libc::geteuid() }
                {
                    return Err(BridgeError::Unavailable("socket path is occupied unsafely"));
                }
                if StdUnixStream::connect(path).is_ok() {
                    return Err(BridgeError::Unavailable(
                        "another bridge instance is active",
                    ));
                }
                fs::remove_file(path)?;
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    fn remove_owned_socket(path: &Path) -> Result<(), BridgeError> {
        let metadata = fs::symlink_metadata(path)?;
        if metadata.file_type().is_socket() && metadata.uid() == unsafe { libc::geteuid() } {
            fs::remove_file(path)?;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn start(app: &AppHandle) -> Result<(), String> {
    platform::start(app).map_err(|error| error.to_string())
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn start(_app: &AppHandle) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn message(action: &str, value: &str) -> Vec<u8> {
        format!(
            r#"{{"v":1,"seq":4,"ts_ms":100000,"action":"{action}","value":{value},"token":"{TOKEN}"}}"#,
        )
        .into_bytes()
    }

    #[test]
    fn parses_strict_versioned_message() {
        let parsed = parse_message(&message("master.delta", "-0.5")).unwrap();
        assert_eq!(parsed.seq, 4);
        assert_eq!(parsed.value, Some(-0.5));
    }

    #[test]
    fn rejects_unknown_fields_actions_and_oversized_values() {
        assert!(parse_message(
            format!(
                r#"{{"v":1,"seq":4,"ts_ms":100000,"action":"system.shell","value":0,"token":"{TOKEN}"}}"#
            )
            .as_bytes()
        )
        .is_err());
        assert!(parse_message(&message("master.delta", "1.5")).is_err());
        assert!(parse_message(
            format!(
                r#"{{"v":1,"seq":4,"ts_ms":100000,"action":"tempo.tap","value":0,"token":"{TOKEN}","extra":true}}"#
            )
            .as_bytes()
        )
        .is_err());
        assert!(parse_message(&[b'x'; MAX_MESSAGE_BYTES + 1]).is_err());
    }

    #[test]
    fn validates_action_specific_values() {
        assert!(parse_message(&message("tempo.tap", "0")).is_ok());
        assert!(parse_message(&message("tempo.tap", "1")).is_err());
        assert!(parse_message(&message("track.trigger", "5")).is_ok());
        assert!(parse_message(&message("track.trigger", "5.5")).is_err());
        assert!(parse_message(&message("track.trigger", "6")).is_err());
    }

    #[test]
    fn token_must_be_exact_lowercase_hex() {
        assert!(valid_wire_token(TOKEN));
        assert!(!valid_wire_token(&TOKEN.to_uppercase()));
        assert!(!valid_wire_token("abcd"));
    }

    #[test]
    fn replay_guard_requires_fresh_increasing_sequence() {
        let mut guard = ReplayGuard::default();
        assert!(guard.accept(10, 1_000_000, 1_000_001).is_ok());
        assert!(matches!(
            guard.accept(10, 1_000_002, 1_000_002),
            Err(BridgeError::Replay)
        ));
        assert!(matches!(
            guard.accept(9, 1_000_003, 1_000_003),
            Err(BridgeError::Replay)
        ));
        assert!(matches!(
            guard.accept(11, 1_000_000 - MAX_CLOCK_SKEW_MS as i64 - 1, 1_000_000),
            Err(BridgeError::Stale)
        ));
        assert!(guard.accept(11, 1_000_004, 1_000_004).is_ok());
    }
}
