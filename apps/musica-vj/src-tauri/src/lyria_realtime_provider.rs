use std::{
    collections::{HashMap, VecDeque},
    env,
    sync::{Arc, Mutex},
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::process::Command;
use tokio::sync::mpsc;

use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, http::HeaderValue, Message},
};

const ENABLE_ENV: &str = "MUSICA_LYRIA_REALTIME_ENABLED";
const API_KEY_ENV: &str = "GEMINI_API_KEY";
const GCP_AUTH_ENV: &str = "MUSICA_GCP_AUTH";
const MODEL: &str = "models/lyria-realtime-exp";
const WS_ENDPOINT: &str = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateMusic";
const SAMPLE_RATE_HZ: u32 = 48_000;
const CHANNELS: u8 = 2;
const MAX_QUEUED_AUDIO_BYTES: usize = 48_000 * 2 * 2 * 8;
const MAX_POLL_BYTES: usize = 48_000 * 2 * 2;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LyriaRealtimeStatus {
    deck: LyriaRealtimeDeck,
    available: bool,
    provider: String,
    model: String,
    sample_rate_hz: u32,
    channels: u8,
    audio_format: String,
    instrumental_only: bool,
    reason: Option<String>,
    active_session_id: Option<String>,
    buffered_audio_bytes: usize,
    streamed_audio_bytes: usize,
    warning: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LyriaRealtimeDeck {
    Main,
    Sequence,
    Vocal,
}

impl LyriaRealtimeDeck {
    fn label(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Sequence => "sequence",
            Self::Vocal => "vocal",
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct WeightedPrompt {
    text: String,
    weight: f32,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum LyriaRealtimeScale {
    CMajorAMinor,
    DFlatMajorBFlatMinor,
    DMajorBMinor,
    EFlatMajorCMinor,
    EMajorDFlatMinor,
    FMajorDMinor,
    GFlatMajorEFlatMinor,
    GMajorEMinor,
    AFlatMajorFMinor,
    AMajorGFlatMinor,
    BFlatMajorGMinor,
    BMajorAFlatMinor,
    ScaleUnspecified,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum LyriaRealtimeMode {
    Quality,
    Diversity,
    Vocalization,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct LyriaRealtimeConfig {
    bpm: u16,
    guidance: f32,
    density: f32,
    brightness: f32,
    temperature: f32,
    top_k: u16,
    seed: Option<u32>,
    scale: LyriaRealtimeScale,
    mute_bass: bool,
    mute_drums: bool,
    only_bass_and_drums: bool,
    music_generation_mode: LyriaRealtimeMode,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct LyriaRealtimeStartRequest {
    weighted_prompts: Vec<WeightedPrompt>,
    config: LyriaRealtimeConfig,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LyriaRealtimeSession {
    deck: LyriaRealtimeDeck,
    id: String,
    provider: String,
    model: String,
    state: String,
    weighted_prompts: Vec<WeightedPrompt>,
    config: LyriaRealtimeConfig,
    sample_rate_hz: u32,
    channels: u8,
    audio_format: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LyriaRealtimeAudioPoll {
    deck: LyriaRealtimeDeck,
    session_id: Option<String>,
    sample_rate_hz: u32,
    channels: u8,
    audio_format: String,
    chunks: Vec<Vec<u8>>,
    buffered_audio_bytes: usize,
    streamed_audio_bytes: usize,
    warning: Option<String>,
}

struct ActiveSession {
    session: LyriaRealtimeSession,
    commands: mpsc::UnboundedSender<RealtimeCommand>,
    shared: Arc<RealtimeShared>,
}

#[derive(Default)]
struct RealtimeShared {
    audio: Mutex<VecDeque<Vec<u8>>>,
    buffered_audio_bytes: Mutex<usize>,
    streamed_audio_bytes: Mutex<usize>,
    warning: Mutex<Option<String>>,
    logged_audio: Mutex<bool>,
}

enum RealtimeCommand {
    Update(LyriaRealtimeStartRequest),
    Playback(&'static str),
    Close,
}

/// How the RealTime WebSocket authenticates to `generativelanguage.googleapis.com`.
#[derive(Clone)]
enum RealtimeAuth {
    /// `?key=<GEMINI_API_KEY>` query parameter.
    ApiKey(String),
    /// `Authorization: Bearer <token>` minted from gcloud application-default
    /// credentials — lets the operator's GCP login drive Lyria with no key.
    Gcloud,
}

pub(crate) struct LyriaRealtimeProvider {
    enabled: bool,
    auth: Option<RealtimeAuth>,
    /// A key brokered at runtime after Cognitum sign-in (ADR-179). When present
    /// it takes precedence and enables the provider, so a packaged app with no
    /// env config becomes live purely from signing in.
    runtime_key: Mutex<Option<String>>,
    active: Mutex<HashMap<LyriaRealtimeDeck, ActiveSession>>,
}

impl LyriaRealtimeProvider {
    pub(crate) fn from_env() -> Self {
        let api_key = env::var(API_KEY_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty());
        let gcloud = env::var(GCP_AUTH_ENV)
            .ok()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("gcloud"));
        // An explicit API key wins; otherwise fall back to gcloud ADC.
        let auth = match (api_key, gcloud) {
            (Some(key), _) => Some(RealtimeAuth::ApiKey(key)),
            (None, true) => Some(RealtimeAuth::Gcloud),
            (None, false) => None,
        };
        Self {
            enabled: env::var(ENABLE_ENV)
                .map(|value| matches!(value.as_str(), "true" | "1" | "yes"))
                .unwrap_or(false),
            auth,
            runtime_key: Mutex::new(None),
            active: Mutex::new(HashMap::new()),
        }
    }

    /// Injects (or clears) a runtime-brokered API key from the Cognitum flow.
    pub(crate) fn set_runtime_key(&self, key: Option<String>) {
        if let Ok(mut guard) = self.runtime_key.lock() {
            *guard = key
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());
        }
    }

    /// The auth to use: a brokered runtime key wins over env config.
    fn effective_auth(&self) -> Option<RealtimeAuth> {
        if let Ok(guard) = self.runtime_key.lock() {
            if let Some(key) = guard.as_ref() {
                return Some(RealtimeAuth::ApiKey(key.clone()));
            }
        }
        self.auth.clone()
    }

    /// A brokered key implicitly enables the provider even without the env flag.
    fn effective_enabled(&self) -> bool {
        self.enabled
            || self
                .runtime_key
                .lock()
                .map(|guard| guard.is_some())
                .unwrap_or(false)
    }

    fn status(&self, deck: LyriaRealtimeDeck) -> LyriaRealtimeStatus {
        let active = self.active.lock().ok();
        let active_session_id = active
            .as_ref()
            .and_then(|guard| guard.get(&deck).map(|active| active.session.id.clone()));
        let (buffered_audio_bytes, streamed_audio_bytes, warning) = active
            .as_ref()
            .and_then(|guard| guard.get(&deck))
            .map(|active| active.shared.snapshot())
            .unwrap_or_default();
        let enabled = self.effective_enabled();
        let auth = self.effective_auth();
        let reason = if !enabled {
            Some(format!("{ENABLE_ENV}=true is required"))
        } else if auth.is_none() {
            Some(format!(
                "{API_KEY_ENV}, {GCP_AUTH_ENV}=gcloud, or Cognitum sign-in is required"
            ))
        } else {
            None
        };
        LyriaRealtimeStatus {
            deck,
            available: enabled && auth.is_some(),
            provider: "lyria_realtime".into(),
            model: MODEL.into(),
            sample_rate_hz: SAMPLE_RATE_HZ,
            channels: CHANNELS,
            audio_format: "pcm16".into(),
            instrumental_only: true,
            reason,
            active_session_id,
            buffered_audio_bytes,
            streamed_audio_bytes,
            warning,
        }
    }

    fn start(
        &self,
        deck: LyriaRealtimeDeck,
        request: LyriaRealtimeStartRequest,
    ) -> Result<LyriaRealtimeSession, String> {
        if !self.effective_enabled() {
            return Err(format!("{ENABLE_ENV}=true is required"));
        }
        let auth = self.effective_auth().ok_or_else(|| {
            format!("{API_KEY_ENV}, {GCP_AUTH_ENV}=gcloud, or Cognitum sign-in is required")
        })?;
        validate_request(&request)?;
        self.stop(deck)?;
        let session = LyriaRealtimeSession {
            deck,
            id: format!("lrt-{}-{}", deck.label(), monotonic_millis()),
            provider: "lyria_realtime".into(),
            model: MODEL.into(),
            state: "streaming".into(),
            weighted_prompts: request.weighted_prompts.clone(),
            config: request.config.clone(),
            sample_rate_hz: SAMPLE_RATE_HZ,
            channels: CHANNELS,
            audio_format: "pcm16".into(),
        };
        let shared = Arc::new(RealtimeShared::default());
        let (commands, receiver) = mpsc::unbounded_channel();
        tokio::spawn(run_stream(
            deck,
            auth,
            request,
            receiver,
            Arc::clone(&shared),
        ));
        self.active
            .lock()
            .map_err(|_| "Lyria RealTime session lock failed")?
            .insert(
                deck,
                ActiveSession {
                    session: session.clone(),
                    commands,
                    shared,
                },
            );
        Ok(session)
    }

    fn update(
        &self,
        deck: LyriaRealtimeDeck,
        request: LyriaRealtimeStartRequest,
    ) -> Result<LyriaRealtimeSession, String> {
        validate_request(&request)?;
        let mut active = self
            .active
            .lock()
            .map_err(|_| "Lyria RealTime session lock failed")?;
        let active = active
            .get_mut(&deck)
            .ok_or("Lyria RealTime deck is not active")?;
        active
            .commands
            .send(RealtimeCommand::Update(request.clone()))
            .map_err(|_| "Lyria RealTime stream is not accepting updates")?;
        active.session.weighted_prompts = request.weighted_prompts;
        active.session.config = request.config;
        Ok(active.session.clone())
    }

    fn stop(&self, deck: LyriaRealtimeDeck) -> Result<(), String> {
        if let Some(active) = self
            .active
            .lock()
            .map_err(|_| "Lyria RealTime session lock failed")?
            .remove(&deck)
        {
            let _ = active.commands.send(RealtimeCommand::Playback("STOP"));
            let _ = active.commands.send(RealtimeCommand::Close);
        }
        Ok(())
    }

    fn poll_audio(&self, deck: LyriaRealtimeDeck) -> LyriaRealtimeAudioPoll {
        let active = self.active.lock().ok();
        if let Some(active) = active.as_ref().and_then(|guard| guard.get(&deck)) {
            let chunks = active.shared.drain_audio(MAX_POLL_BYTES);
            let (buffered_audio_bytes, streamed_audio_bytes, warning) = active.shared.snapshot();
            return LyriaRealtimeAudioPoll {
                deck,
                session_id: Some(active.session.id.clone()),
                sample_rate_hz: SAMPLE_RATE_HZ,
                channels: CHANNELS,
                audio_format: "pcm16".into(),
                chunks,
                buffered_audio_bytes,
                streamed_audio_bytes,
                warning,
            };
        }
        LyriaRealtimeAudioPoll {
            deck,
            session_id: None,
            sample_rate_hz: SAMPLE_RATE_HZ,
            channels: CHANNELS,
            audio_format: "pcm16".into(),
            chunks: vec![],
            buffered_audio_bytes: 0,
            streamed_audio_bytes: 0,
            warning: None,
        }
    }
}

impl RealtimeShared {
    fn push_audio(&self, bytes: Vec<u8>) {
        if bytes.is_empty() {
            return;
        }
        if let Ok(mut streamed) = self.streamed_audio_bytes.lock() {
            *streamed = streamed.saturating_add(bytes.len());
        }
        if let Ok(mut logged) = self.logged_audio.lock() {
            if !*logged {
                eprintln!("Lyria RealTime desktop stream received PCM audio");
                *logged = true;
            }
        }
        let Ok(mut queue) = self.audio.lock() else {
            return;
        };
        let Ok(mut buffered) = self.buffered_audio_bytes.lock() else {
            return;
        };
        *buffered = buffered.saturating_add(bytes.len());
        queue.push_back(bytes);
        while *buffered > MAX_QUEUED_AUDIO_BYTES {
            if let Some(dropped) = queue.pop_front() {
                *buffered = buffered.saturating_sub(dropped.len());
            } else {
                *buffered = 0;
                break;
            }
        }
    }

    fn drain_audio(&self, max_bytes: usize) -> Vec<Vec<u8>> {
        let Ok(mut queue) = self.audio.lock() else {
            return vec![];
        };
        let Ok(mut buffered) = self.buffered_audio_bytes.lock() else {
            return vec![];
        };
        let mut drained = vec![];
        let mut total = 0;
        while let Some(chunk) = queue.pop_front() {
            if total > 0 && total + chunk.len() > max_bytes {
                queue.push_front(chunk);
                break;
            }
            total += chunk.len();
            *buffered = buffered.saturating_sub(chunk.len());
            drained.push(chunk);
            if total >= max_bytes {
                break;
            }
        }
        drained
    }

    fn set_warning(&self, warning: impl Into<String>) {
        if let Ok(mut current) = self.warning.lock() {
            *current = Some(warning.into());
        }
    }

    fn snapshot(&self) -> (usize, usize, Option<String>) {
        (
            self.buffered_audio_bytes
                .lock()
                .map(|value| *value)
                .unwrap_or(0),
            self.streamed_audio_bytes
                .lock()
                .map(|value| *value)
                .unwrap_or(0),
            self.warning.lock().ok().and_then(|value| value.clone()),
        )
    }
}

async fn run_stream(
    deck: LyriaRealtimeDeck,
    auth: RealtimeAuth,
    initial: LyriaRealtimeStartRequest,
    mut commands: mpsc::UnboundedReceiver<RealtimeCommand>,
    shared: Arc<RealtimeShared>,
) {
    if let Err(error) = run_stream_inner(auth, initial, &mut commands, &shared).await {
        eprintln!("Lyria RealTime {} deck stopped: {error}", deck.label());
        shared.set_warning(error);
    }
}

/// Mints a gcloud application-default access token off the async runtime.
async fn gcloud_realtime_token() -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let mut command = Command::new("gcloud");
        command.args(["auth", "application-default", "print-access-token"]);
        crate::process_util::hide_console_window(&mut command);
        let output = command
            .output()
            .map_err(|_| "gcloud is not available on PATH".to_string())?;
        if !output.status.success() {
            return Err("gcloud application-default credentials are not set up".to_string());
        }
        let token = String::from_utf8(output.stdout)
            .map_err(|_| "gcloud returned an invalid token".to_string())?;
        let token = token.trim().to_string();
        if token.len() < 20
            || token.len() > 4096
            || !token
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err("gcloud returned an invalid token".to_string());
        }
        Ok(token)
    })
    .await
    .map_err(|_| "gcloud token task failed".to_string())?
}

async fn run_stream_inner(
    auth: RealtimeAuth,
    initial: LyriaRealtimeStartRequest,
    commands: &mut mpsc::UnboundedReceiver<RealtimeCommand>,
    shared: &RealtimeShared,
) -> Result<(), String> {
    // API key rides the query string; gcloud rides an Authorization header on
    // the handshake, matching how the batch provider authenticates to the same
    // generativelanguage.googleapis.com host.
    let request = match &auth {
        RealtimeAuth::ApiKey(key) => format!("{WS_ENDPOINT}?key={key}")
            .into_client_request()
            .map_err(|_| "Lyria RealTime WebSocket request could not be built")?,
        RealtimeAuth::Gcloud => {
            let token = gcloud_realtime_token().await?;
            let mut request = WS_ENDPOINT
                .into_client_request()
                .map_err(|_| "Lyria RealTime WebSocket request could not be built")?;
            let mut value = HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|_| "Lyria RealTime authorization header is invalid")?;
            value.set_sensitive(true);
            request.headers_mut().insert("authorization", value);
            request
        }
    };
    let (mut socket, _) = connect_async(request)
        .await
        .map_err(|_| "Lyria RealTime WebSocket connection failed")?;
    send_json(&mut socket, json!({ "setup": { "model": MODEL } })).await?;

    let mut setup_complete = false;
    while let Some(message) = socket.next().await {
        let message = message.map_err(|error| format!("Lyria RealTime setup failed: {error}"))?;
        if let Some(value) = message_json(&message)? {
            if value.get("setupComplete").is_some() || value.get("setup_complete").is_some() {
                setup_complete = true;
                break;
            }
            if let Some(warning) = value.get("warning").and_then(Value::as_str) {
                shared.set_warning(warning);
            }
            if let Some(error) = value.get("error") {
                return Err(format!("Lyria RealTime setup error: {error}"));
            }
        }
    }
    if !setup_complete {
        return Err("Lyria RealTime setup did not complete".into());
    }
    eprintln!("Lyria RealTime desktop stream setup complete");

    send_realtime_request(&mut socket, &initial).await?;
    send_json(&mut socket, json!({ "playback_control": "PLAY" })).await?;

    loop {
        tokio::select! {
            command = commands.recv() => {
                match command {
                    Some(RealtimeCommand::Update(request)) => send_realtime_request(&mut socket, &request).await?,
                    Some(RealtimeCommand::Playback(control)) => send_json(&mut socket, json!({ "playback_control": control })).await?,
                    Some(RealtimeCommand::Close) | None => {
                        let _ = socket.close(None).await;
                        return Ok(());
                    }
                }
            }
            message = socket.next() => {
                let Some(message) = message else {
                    return Err("Lyria RealTime WebSocket closed".into());
                };
                let message = message.map_err(|error| format!("Lyria RealTime stream failed: {error}"))?;
                match message {
                    Message::Text(text) => handle_server_text(&text, shared)?,
                    Message::Binary(bytes) => {
                        if let Ok(text) = std::str::from_utf8(&bytes) {
                            if serde_json::from_str::<Value>(text).is_ok() {
                                handle_server_text(text, shared)?;
                            } else {
                                shared.push_audio(bytes.to_vec());
                            }
                        } else {
                            shared.push_audio(bytes.to_vec());
                        }
                    }
                    Message::Close(_) => return Err("Lyria RealTime WebSocket closed".into()),
                    _ => {}
                }
            }
        }
    }
}

fn message_json(message: &Message) -> Result<Option<Value>, String> {
    let text = match message {
        Message::Text(text) => Some(text.as_str()),
        Message::Binary(bytes) => std::str::from_utf8(bytes).ok(),
        _ => None,
    };
    text.map(|text| {
        serde_json::from_str(text)
            .map_err(|_| "Lyria RealTime returned invalid JSON during setup".to_string())
    })
    .transpose()
}

async fn send_realtime_request(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    request: &LyriaRealtimeStartRequest,
) -> Result<(), String> {
    send_json(
        socket,
        json!({ "client_content": { "weightedPrompts": request.weighted_prompts } }),
    )
    .await?;
    send_json(socket, json!({ "music_generation_config": request.config })).await
}

async fn send_json(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    value: Value,
) -> Result<(), String> {
    socket
        .send(Message::Text(value.to_string().into()))
        .await
        .map_err(|error| format!("Lyria RealTime send failed: {error}"))
}

fn handle_server_text(text: &str, shared: &RealtimeShared) -> Result<(), String> {
    let value: Value = serde_json::from_str(text)
        .map_err(|_| "Lyria RealTime returned invalid JSON during streaming")?;
    if let Some(warning) = value.get("warning").and_then(Value::as_str) {
        eprintln!("Lyria RealTime warning: {warning}");
        shared.set_warning(warning);
    }
    if let Some(error) = value.get("error") {
        eprintln!("Lyria RealTime error: {error}");
        shared.set_warning(format!("Lyria RealTime error: {error}"));
    }
    if let Some(filtered) = value
        .get("filteredPrompt")
        .or_else(|| value.get("filtered_prompt"))
    {
        shared.set_warning(format!("Filtered prompt: {filtered}"));
    }
    let chunks = value
        .pointer("/serverContent/audioChunks")
        .or_else(|| value.pointer("/server_content/audio_chunks"))
        .and_then(Value::as_array);
    if let Some(chunks) = chunks {
        for chunk in chunks {
            if let Some(data) = chunk.get("data").and_then(Value::as_str) {
                let bytes = BASE64
                    .decode(data.as_bytes())
                    .map_err(|_| "Lyria RealTime returned invalid Base64 audio")?;
                shared.push_audio(bytes);
            }
        }
    }
    Ok(())
}

fn validate_request(request: &LyriaRealtimeStartRequest) -> Result<(), String> {
    if request.weighted_prompts.is_empty() || request.weighted_prompts.len() > 4 {
        return Err("Lyria RealTime requires one to four weighted prompts".into());
    }
    for prompt in &request.weighted_prompts {
        let text = prompt.text.trim();
        if text.is_empty() || text.chars().count() > 240 {
            return Err("Lyria RealTime prompts must be 1 to 240 characters".into());
        }
        if !prompt.weight.is_finite()
            || prompt.weight == 0.0
            || prompt.weight < -3.0
            || prompt.weight > 3.0
        {
            return Err(
                "Lyria RealTime prompt weights must be finite, non-zero, and between -3 and 3"
                    .into(),
            );
        }
    }
    let config = &request.config;
    if !(60..=200).contains(&config.bpm) {
        return Err("Lyria RealTime BPM must be 60 to 200".into());
    }
    if !unit(config.density) || !unit(config.brightness) {
        return Err("Lyria RealTime density and brightness must be 0 to 1".into());
    }
    if !config.guidance.is_finite() || !(0.0..=6.0).contains(&config.guidance) {
        return Err("Lyria RealTime guidance must be 0 to 6".into());
    }
    if !config.temperature.is_finite() || !(0.0..=3.0).contains(&config.temperature) {
        return Err("Lyria RealTime temperature must be 0 to 3".into());
    }
    if !(1..=1000).contains(&config.top_k) {
        return Err("Lyria RealTime topK must be 1 to 1000".into());
    }
    if config.only_bass_and_drums && (config.mute_bass || config.mute_drums) {
        return Err("onlyBassAndDrums cannot be combined with muted bass or drums".into());
    }
    Ok(())
}

fn unit(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn monotonic_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

#[tauri::command]
pub(crate) fn lyria_realtime_status(
    state: tauri::State<'_, LyriaRealtimeProvider>,
    deck: LyriaRealtimeDeck,
) -> LyriaRealtimeStatus {
    state.status(deck)
}

/// Injects a runtime-brokered API key (from the Cognitum sign-in flow) so a
/// packaged app with no env config becomes live purely from signing in. Pass an
/// empty string to clear it and revert to env/BYO-key behavior.
#[tauri::command]
pub(crate) fn lyria_realtime_configure_key(
    state: tauri::State<'_, LyriaRealtimeProvider>,
    key: String,
) {
    let trimmed = key.trim();
    state.set_runtime_key(if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    });
}

#[tauri::command]
pub(crate) async fn lyria_realtime_start(
    state: tauri::State<'_, LyriaRealtimeProvider>,
    deck: LyriaRealtimeDeck,
    request: LyriaRealtimeStartRequest,
) -> Result<LyriaRealtimeSession, String> {
    state.start(deck, request)
}

#[tauri::command]
pub(crate) async fn lyria_realtime_update(
    state: tauri::State<'_, LyriaRealtimeProvider>,
    deck: LyriaRealtimeDeck,
    request: LyriaRealtimeStartRequest,
) -> Result<LyriaRealtimeSession, String> {
    state.update(deck, request)
}

#[tauri::command]
pub(crate) async fn lyria_realtime_stop(
    state: tauri::State<'_, LyriaRealtimeProvider>,
    deck: LyriaRealtimeDeck,
) -> Result<(), String> {
    state.stop(deck)
}

#[tauri::command]
pub(crate) async fn lyria_realtime_poll_audio(
    state: tauri::State<'_, LyriaRealtimeProvider>,
    deck: LyriaRealtimeDeck,
) -> Result<LyriaRealtimeAudioPoll, String> {
    Ok(state.poll_audio(deck))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> LyriaRealtimeStartRequest {
        LyriaRealtimeStartRequest {
            weighted_prompts: vec![WeightedPrompt {
                text: "expressive prepared piano, ambient techno pulse".into(),
                weight: 1.0,
            }],
            config: LyriaRealtimeConfig {
                bpm: 120,
                guidance: 4.0,
                density: 0.55,
                brightness: 0.45,
                temperature: 1.1,
                top_k: 40,
                seed: Some(42),
                scale: LyriaRealtimeScale::EFlatMajorCMinor,
                mute_bass: false,
                mute_drums: false,
                only_bass_and_drums: false,
                music_generation_mode: LyriaRealtimeMode::Quality,
            },
        }
    }

    #[test]
    fn validates_realtime_controls() {
        assert!(validate_request(&request()).is_ok());
        let mut invalid = request();
        invalid.config.bpm = 220;
        assert!(validate_request(&invalid).is_err());
        let mut invalid = request();
        invalid.weighted_prompts[0].weight = 0.0;
        assert!(validate_request(&invalid).is_err());
        let mut invalid = request();
        invalid.config.only_bass_and_drums = true;
        invalid.config.mute_drums = true;
        assert!(validate_request(&invalid).is_err());
    }

    #[test]
    fn accepts_named_multistream_decks() {
        assert_eq!(
            serde_json::from_str::<LyriaRealtimeDeck>(r#""main""#).unwrap(),
            LyriaRealtimeDeck::Main
        );
        assert_eq!(
            serde_json::from_str::<LyriaRealtimeDeck>(r#""sequence""#).unwrap(),
            LyriaRealtimeDeck::Sequence
        );
        assert_eq!(
            serde_json::from_str::<LyriaRealtimeDeck>(r#""vocal""#).unwrap(),
            LyriaRealtimeDeck::Vocal
        );
    }

    #[test]
    fn accepts_frontend_realtime_payload_shape() {
        let payload = r#"{
          "weightedPrompts": [{"text": "prepared piano and ambient techno", "weight": 1.0}],
          "config": {
            "bpm": 118,
            "guidance": 4.0,
            "density": 0.52,
            "brightness": 0.42,
            "temperature": 1.1,
            "topK": 40,
            "seed": 42,
            "scale": "E_FLAT_MAJOR_C_MINOR",
            "muteBass": false,
            "muteDrums": false,
            "onlyBassAndDrums": false,
            "musicGenerationMode": "QUALITY"
          }
        }"#;
        let request: LyriaRealtimeStartRequest = serde_json::from_str(payload).unwrap();
        assert!(validate_request(&request).is_ok());
        assert!(matches!(
            request.config.scale,
            LyriaRealtimeScale::EFlatMajorCMinor
        ));
    }

    #[test]
    fn drains_audio_with_a_bounded_queue() {
        let shared = RealtimeShared::default();
        shared.push_audio(vec![1; 100]);
        shared.push_audio(vec![2; 100]);
        assert_eq!(shared.snapshot().0, 200);
        let chunks = shared.drain_audio(150);
        assert_eq!(chunks.len(), 1);
        assert_eq!(shared.snapshot().0, 100);
    }

    #[test]
    fn parses_server_audio_chunks() {
        let shared = RealtimeShared::default();
        let body = json!({
            "serverContent": {
                "audioChunks": [
                    { "data": BASE64.encode([1_u8, 2, 3, 4]), "mimeType": "audio/pcm" }
                ]
            }
        });
        handle_server_text(&body.to_string(), &shared).unwrap();
        assert_eq!(shared.drain_audio(1024), vec![vec![1, 2, 3, 4]]);
    }
}
