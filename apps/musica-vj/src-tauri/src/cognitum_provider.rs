use std::{
    env,
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    sync::Mutex,
    time::{Duration, Instant},
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::RngCore;
use reqwest::{redirect::Policy, Client, Url};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{Manager, State};

const AUTH_BASE_ENV: &str = "MUSICA_COGNITUM_AUTH_BASE";
const API_BASE_ENV: &str = "MUSICA_COGNITUM_API_BASE";
const CLIENT_ID_ENV: &str = "MUSICA_COGNITUM_CLIENT_ID";
const SCOPES_ENV: &str = "MUSICA_COGNITUM_SCOPES";
const DEFAULT_AUTH_BASE: &str = "https://auth.cognitum.one";
const DEFAULT_API_BASE: &str = "https://api.cognitum.one";
// The registered native client from identity seed migration 0014, shared with
// meta-proxy/metaharness. Flip to musica-vj-desktop via env once dashboard
// PR #90 (seed migration 0016) is deployed.
const DEFAULT_CLIENT_ID: &str = "meta-proxy";
// auth.cognitum.one advertises scopes_supported: ["inference"].
const DEFAULT_OAUTH_SCOPES: &str = "inference";
// RFC 8252 out-of-band sentinel — must match the identity service's
// FALLBACK_REDIRECT_URI exactly; used by the paste-code manual flow.
const OOB_REDIRECT_URI: &str = "urn:ietf:wg:oauth:2.0:oob";
const CALLBACK_TIMEOUT_SECONDS: u64 = 180;
const MAX_RESPONSE_BYTES: usize = 128 * 1024;
const KNOWN_CAPABILITIES: [&str; 4] = [
    "advanced-prompting",
    "autopilot",
    "learning",
    "realtime-vocals",
];

#[derive(Default)]
pub(crate) struct CognitumProvider {
    state: Mutex<AuthState>,
}

#[derive(Default)]
struct AuthState {
    pending_started_at: Option<Instant>,
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_at: Option<Instant>,
    manual_verifier: Option<String>,
    account: Option<String>,
    capabilities: Vec<String>,
    last_error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CognitumStatus {
    signed_in: bool,
    pending: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    account: Option<String>,
    capabilities: Vec<String>,
    auth_host: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CognitumAuthStart {
    auth_url: String,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    account_email: Option<String>,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
}

#[derive(Deserialize)]
struct CapabilitiesResponse {
    #[serde(default)]
    account: Option<String>,
    #[serde(default)]
    capabilities: Vec<String>,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: [ChatMessage; 2],
    temperature: f32,
    max_tokens: u16,
}

#[derive(Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StylePackPrompt {
    text: String,
    weight: f32,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StylePackConfig {
    bpm: u16,
    density: f32,
    brightness: f32,
    guidance: f32,
    scale: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StylePack {
    label: String,
    description: String,
    prompts: Vec<StylePackPrompt>,
    config: StylePackConfig,
}

fn auth_base() -> String {
    env::var(AUTH_BASE_ENV).unwrap_or_else(|_| DEFAULT_AUTH_BASE.into())
}

fn api_base() -> String {
    env::var(API_BASE_ENV).unwrap_or_else(|_| DEFAULT_API_BASE.into())
}

fn client_id() -> String {
    env::var(CLIENT_ID_ENV).unwrap_or_else(|_| DEFAULT_CLIENT_ID.into())
}

fn oauth_scopes() -> String {
    env::var(SCOPES_ENV).unwrap_or_else(|_| DEFAULT_OAUTH_SCOPES.into())
}

fn random_url_safe(bytes: usize) -> String {
    let mut buffer = vec![0u8; bytes];
    rand::thread_rng().fill_bytes(&mut buffer);
    URL_SAFE_NO_PAD.encode(buffer)
}

fn open_in_system_browser(url: &str) -> Result<(), String> {
    let parsed = Url::parse(url).map_err(|_| "Invalid authorization URL".to_string())?;
    if parsed.scheme() != "https" {
        return Err("Authorization URL must use HTTPS".into());
    }
    #[cfg(target_os = "macos")]
    let command = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "linux")]
    let command = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let command = std::process::Command::new("cmd")
        .args(["/C", "start", "", url])
        .spawn();
    command
        .map(|_| ())
        .map_err(|_| "Could not open the system browser".into())
}

fn parse_callback_query(request_line: &str, expected_state: &str) -> Option<String> {
    let path = request_line.split_whitespace().nth(1)?;
    let (_, query) = path.split_once('?')?;
    let mut code = None;
    let mut state_ok = false;
    for pair in query.split('&') {
        let (key, value) = pair.split_once('=')?;
        if key == "code" {
            code = Some(value.to_string());
        }
        if key == "state" && value == expected_state {
            state_ok = true;
        }
    }
    if state_ok {
        code
    } else {
        None
    }
}

async fn read_bounded_json<T: for<'de> Deserialize<'de>>(
    response: reqwest::Response,
) -> Result<T, String> {
    let bytes = response
        .bytes()
        .await
        .map_err(|_| "Cognitum response could not be read".to_string())?;
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err("Cognitum response exceeded the size limit".into());
    }
    serde_json::from_slice(&bytes).map_err(|_| "Cognitum returned an invalid response".into())
}

fn http_client() -> Result<Client, String> {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .redirect(Policy::none())
        .build()
        .map_err(|_| "Cognitum HTTP client could not be created".into())
}

async fn fetch_capabilities(client: &Client, token: &str) -> (Option<String>, Vec<String>) {
    let url = format!("{}/v1/capabilities", api_base());
    let response = client.get(url).bearer_auth(token).send().await;
    if let Ok(response) = response {
        if response.status().is_success() {
            if let Ok(parsed) = read_bounded_json::<CapabilitiesResponse>(response).await {
                let capabilities = parsed
                    .capabilities
                    .into_iter()
                    .filter(|capability| KNOWN_CAPABILITIES.contains(&capability.as_str()))
                    .collect();
                return (parsed.account, capabilities);
            }
        }
    }
    // A reachable token without a capabilities endpoint still unlocks the core set.
    (
        None,
        KNOWN_CAPABILITIES
            .iter()
            .map(|value| value.to_string())
            .collect(),
    )
}

#[tauri::command]
pub(crate) fn cognitum_status(provider: State<'_, CognitumProvider>) -> CognitumStatus {
    let state = provider.state.lock().expect("cognitum state");
    let pending = state
        .pending_started_at
        .map(|started| started.elapsed() < Duration::from_secs(CALLBACK_TIMEOUT_SECONDS))
        .unwrap_or(false);
    CognitumStatus {
        signed_in: state.access_token.is_some(),
        pending: pending && state.access_token.is_none(),
        account: state.account.clone(),
        capabilities: state.capabilities.clone(),
        auth_host: Url::parse(&auth_base())
            .ok()
            .and_then(|url| url.host_str().map(str::to_owned))
            .unwrap_or_else(|| "cognitum.one".into()),
        reason: state.last_error.clone(),
    }
}

#[tauri::command]
pub(crate) fn cognitum_sign_out(provider: State<'_, CognitumProvider>) {
    let mut state = provider.state.lock().expect("cognitum state");
    *state = AuthState::default();
}

#[tauri::command]
pub(crate) async fn cognitum_auth_start(
    provider: State<'_, CognitumProvider>,
    app: tauri::AppHandle,
) -> Result<CognitumAuthStart, String> {
    let verifier = random_url_safe(48);
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    let oauth_state = random_url_safe(24);

    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|_| "Could not open the OAuth callback listener".to_string())?;
    let port = listener
        .local_addr()
        .map_err(|_| "Could not read the OAuth callback port".to_string())?
        .port();
    // auth.cognitum.one only admits loopback redirects at exactly /oauth/callback.
    let redirect_uri = format!("http://127.0.0.1:{port}/oauth/callback");

    let auth_url = format!(
        "{}/oauth/authorize?response_type=code&client_id={}&redirect_uri={}&code_challenge={}&code_challenge_method=S256&state={}&scope={}",
        auth_base(),
        urlencoding_encode(&client_id()),
        urlencoding_encode(&redirect_uri),
        challenge,
        oauth_state,
        urlencoding_encode(&oauth_scopes()),
    );
    open_in_system_browser(&auth_url)?;

    {
        let mut state = provider.state.lock().expect("cognitum state");
        state.pending_started_at = Some(Instant::now());
        state.last_error = None;
    }

    let app_handle = app.clone();
    std::thread::spawn(move || {
        listener.set_nonblocking(true).ok();
        let deadline = Instant::now() + Duration::from_secs(CALLBACK_TIMEOUT_SECONDS);
        let accepted = loop {
            match listener.accept() {
                Ok(pair) => break Some(pair),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() > deadline {
                        break None;
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(_) => break None,
            }
        };
        let result = accepted.and_then(|(mut stream, _)| {
            stream.set_nonblocking(false).ok();
            stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
            let mut request_line = String::new();
            BufReader::new(&stream).read_line(&mut request_line).ok()?;
            let code = parse_callback_query(&request_line, &oauth_state);
            let body = if code.is_some() {
                "<html><body style=\"font-family:sans-serif;background:#05070b;color:#9debf6;display:grid;place-items:center;height:100vh\"><div><h2>Musica is connected to Cognitum One</h2><p>You can close this tab and return to the app.</p></div></body></html>"
            } else {
                "<html><body style=\"font-family:sans-serif\"><h2>Sign-in was not completed</h2></body></html>"
            };
            let _ = stream.write_all(
                format!("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).as_bytes(),
            );
            code
        });

        let provider = app_handle.state::<CognitumProvider>();
        let Some(code) = result else {
            let mut state = provider.state.lock().expect("cognitum state");
            state.pending_started_at = None;
            state.last_error = Some("Sign-in timed out or was cancelled".into());
            return;
        };

        let runtime = tauri::async_runtime::handle();
        let exchange = runtime.block_on(async move {
            let client = http_client()?;
            let response = client
                .post(format!("{}/oauth/token", auth_base()))
                .form(&[
                    ("grant_type", "authorization_code"),
                    ("code", code.as_str()),
                    ("redirect_uri", redirect_uri.as_str()),
                    ("client_id", client_id().as_str()),
                    ("code_verifier", verifier.as_str()),
                ])
                .send()
                .await
                .map_err(|_| "Cognitum token exchange failed".to_string())?;
            if !response.status().is_success() {
                return Err("Cognitum rejected the sign-in".to_string());
            }
            let token: TokenResponse = read_bounded_json(response).await?;
            let (account, capabilities) = fetch_capabilities(&client, &token.access_token).await;
            Ok::<_, String>((token, account, capabilities))
        });

        let mut state = provider.state.lock().expect("cognitum state");
        state.pending_started_at = None;
        match exchange {
            Ok((token, account, capabilities)) => {
                store_token_response(&mut state, token, account, capabilities);
            }
            Err(error) => {
                state.last_error = Some(error);
            }
        }
    });

    Ok(CognitumAuthStart { auth_url })
}

fn store_token_response(
    state: &mut AuthState,
    token: TokenResponse,
    account: Option<String>,
    capabilities: Vec<String>,
) {
    state.account = token.account_email.or(account);
    // identity issues 15-minute access tokens; refresh one minute early.
    state.expires_at = token
        .expires_in
        .map(|seconds| Instant::now() + Duration::from_secs(seconds.max(0) as u64))
        .map(|at| at - Duration::from_secs(60));
    state.access_token = Some(token.access_token);
    state.refresh_token = token.refresh_token;
    state.capabilities = capabilities;
    state.last_error = None;
}

/// Returns a currently valid access token, refreshing when expired. The
/// identity service rotates refresh tokens with reuse detection, so the stored
/// refresh token is consumed exactly once and replaced (or the session ends).
async fn fresh_access_token(provider: &State<'_, CognitumProvider>) -> Result<String, String> {
    let (token, refresh) = {
        let mut state = provider.state.lock().expect("cognitum state");
        let token = state
            .access_token
            .clone()
            .ok_or_else(|| "Sign in to Cognitum One first".to_string())?;
        let expired = state
            .expires_at
            .map(|at| Instant::now() >= at)
            .unwrap_or(true);
        if !expired {
            return Ok(token);
        }
        // Take (consume) the refresh token before the network call: a rotated
        // token must never be presented twice.
        (token, state.refresh_token.take())
    };
    let _ = token;
    let Some(refresh) = refresh else {
        let mut state = provider.state.lock().expect("cognitum state");
        state.access_token = None;
        return Err("Cognitum session expired — sign in again".into());
    };
    let client = http_client()?;
    let response = client
        .post(format!("{}/oauth/token", auth_base()))
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh.as_str()),
            ("client_id", client_id().as_str()),
        ])
        .send()
        .await
        .map_err(|_| "Cognitum token refresh failed".to_string())?;
    if !response.status().is_success() {
        let mut state = provider.state.lock().expect("cognitum state");
        *state = AuthState::default();
        state.last_error = Some("Cognitum session expired — sign in again".into());
        return Err("Cognitum session expired — sign in again".into());
    }
    let token: TokenResponse = read_bounded_json(response).await?;
    let access = token.access_token.clone();
    let mut state = provider.state.lock().expect("cognitum state");
    let account = state.account.clone();
    let capabilities = state.capabilities.clone();
    store_token_response(&mut state, token, account, capabilities);
    Ok(access)
}

#[tauri::command]
pub(crate) async fn cognitum_auth_manual_start(
    provider: State<'_, CognitumProvider>,
) -> Result<CognitumAuthStart, String> {
    let verifier = random_url_safe(48);
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    let oauth_state = random_url_safe(24);
    let auth_url = format!(
        "{}/oauth/authorize?response_type=code&client_id={}&redirect_uri={}&code_challenge={}&code_challenge_method=S256&state={}&scope={}",
        auth_base(),
        urlencoding_encode(&client_id()),
        urlencoding_encode(OOB_REDIRECT_URI),
        challenge,
        oauth_state,
        urlencoding_encode(&oauth_scopes()),
    );
    open_in_system_browser(&auth_url)?;
    let mut state = provider.state.lock().expect("cognitum state");
    state.manual_verifier = Some(verifier);
    state.last_error = None;
    Ok(CognitumAuthStart { auth_url })
}

#[tauri::command]
pub(crate) async fn cognitum_auth_manual_complete(
    provider: State<'_, CognitumProvider>,
    code: String,
) -> Result<(), String> {
    let trimmed = code.trim().to_uppercase();
    if trimmed.is_empty() || trimmed.len() > 64 {
        return Err("Paste the CGN- code shown in the browser".into());
    }
    let verifier = provider
        .state
        .lock()
        .expect("cognitum state")
        .manual_verifier
        .clone()
        .ok_or_else(|| "Start the paste-code sign-in first".to_string())?;
    let client = http_client()?;
    #[derive(Serialize)]
    struct CodeExchange<'a> {
        code: &'a str,
        code_verifier: &'a str,
        client_id: String,
    }
    let response = client
        .post(format!("{}/v1/oauth/code-exchange", auth_base()))
        .json(&CodeExchange {
            code: trimmed.as_str(),
            code_verifier: verifier.as_str(),
            client_id: client_id(),
        })
        .send()
        .await
        .map_err(|_| "Cognitum code exchange failed".to_string())?;
    if !response.status().is_success() {
        return Err("Cognitum rejected the pasted code".into());
    }
    let token: TokenResponse = read_bounded_json(response).await?;
    let (account, capabilities) = fetch_capabilities(&client, &token.access_token).await;
    let mut state = provider.state.lock().expect("cognitum state");
    state.manual_verifier = None;
    state.pending_started_at = None;
    store_token_response(&mut state, token, account, capabilities);
    Ok(())
}

#[tauri::command]
pub(crate) async fn cognitum_style_pack(
    provider: State<'_, CognitumProvider>,
    description: String,
) -> Result<StylePack, String> {
    let trimmed = description.trim();
    if trimmed.is_empty() || trimmed.len() > 600 {
        return Err("Describe the style in 1 to 600 characters".into());
    }
    let token = fresh_access_token(&provider).await?;

    let system = "You design Lyria RealTime style packs for a live AI music instrument. Return only JSON with keys label (max 24 chars), description (one sentence), prompts (exactly 4 objects with text max 240 chars and weight; the first three weights are positive between 0.9 and 1.5, the fourth is negative between -1.3 and -1.0 and lists what to avoid), and config with bpm (60-200), density (0-1), brightness (0-1), guidance (3-6), scale (one of C_MAJOR_A_MINOR, D_MAJOR_B_MINOR, E_FLAT_MAJOR_C_MINOR, F_MAJOR_D_MINOR, G_MAJOR_E_MINOR, A_MAJOR_G_FLAT_MINOR, B_FLAT_MAJOR_G_MINOR). Prompt 1 is the sonic identity with tempo, instrumentation and character; prompt 2 blends influences; prompt 3 describes a 32-bar arc and mix; prompt 4 is the negative prompt.".to_string();

    let client = http_client()?;
    let request = ChatRequest {
        model: "meta-llm",
        messages: [
            ChatMessage {
                role: "system",
                content: system,
            },
            ChatMessage {
                role: "user",
                content: trimmed.to_string(),
            },
        ],
        temperature: 0.6,
        max_tokens: 900,
    };
    let response = client
        .post(format!("{}/v1/chat/completions", api_base()))
        .bearer_auth(token)
        .json(&request)
        .send()
        .await
        .map_err(|_| "Cognitum style request failed".to_string())?;
    if !response.status().is_success() {
        return Err("Cognitum style generation was rejected".into());
    }
    let chat: ChatResponse = read_bounded_json(response).await?;
    let content = chat
        .choices
        .first()
        .map(|choice| choice.message.content.trim())
        .ok_or_else(|| "Cognitum returned an empty style".to_string())?;
    let json_start = content
        .find('{')
        .ok_or_else(|| "Cognitum returned an invalid style".to_string())?;
    let json_end = content
        .rfind('}')
        .ok_or_else(|| "Cognitum returned an invalid style".to_string())?;
    let pack: StylePack = serde_json::from_str(&content[json_start..=json_end])
        .map_err(|_| "Cognitum returned an invalid style".to_string())?;
    validate_style_pack(&pack)?;
    Ok(pack)
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SetArcStep {
    at_minute: f32,
    style_id: String,
    visual_scene: String,
    bpm: u16,
    #[serde(default)]
    fx: Option<SetArcFx>,
    note: String,
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SetArcFx {
    #[serde(default)]
    sweep: Option<f32>,
    #[serde(default)]
    reverb: Option<f32>,
    #[serde(default)]
    echo: Option<f32>,
    #[serde(default)]
    flanger: Option<f32>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SetArc {
    title: String,
    duration_minutes: u16,
    steps: Vec<SetArcStep>,
}

#[tauri::command]
pub(crate) async fn cognitum_set_arc(
    provider: State<'_, CognitumProvider>,
    duration_minutes: u16,
    direction: String,
    style_ids: Vec<String>,
    scene_ids: Vec<String>,
) -> Result<SetArc, String> {
    if !(30..=90).contains(&duration_minutes) {
        return Err("Set length must be between 30 and 90 minutes".into());
    }
    let trimmed = direction.trim();
    if trimmed.len() > 600 {
        return Err("Describe the set in at most 600 characters".into());
    }
    if style_ids.is_empty() || scene_ids.is_empty() || style_ids.len() > 64 || scene_ids.len() > 32
    {
        return Err("Style and scene lists are required".into());
    }
    let token = fresh_access_token(&provider).await?;

    let system = format!(
        "You plan complete live AI music performances. Return only JSON with keys title (max 40 chars), durationMinutes (echo the requested length), and steps: 6 to 14 objects, each with atMinute (number, 0 = opening, strictly increasing, all below durationMinutes), styleId (one of: {styles}), visualScene (one of: {scenes}), bpm (60-200), fx (optional object with sweep/reverb/echo/flanger each 0-1, omit for dry), and note (max 90 chars stage direction). Design a professional energy arc: establish, build, peak, breathe, second peak, resolve. Adjacent steps should differ meaningfully.",
        styles = style_ids.join(", "),
        scenes = scene_ids.join(", "),
    );
    let user = if trimmed.is_empty() {
        format!("Plan a {duration_minutes} minute live set.")
    } else {
        format!("Plan a {duration_minutes} minute live set. Direction: {trimmed}")
    };

    let client = http_client()?;
    let request = ChatRequest {
        model: "meta-llm",
        messages: [
            ChatMessage {
                role: "system",
                content: system,
            },
            ChatMessage {
                role: "user",
                content: user,
            },
        ],
        temperature: 0.7,
        max_tokens: 1_400,
    };
    let response = client
        .post(format!("{}/v1/chat/completions", api_base()))
        .bearer_auth(token)
        .json(&request)
        .send()
        .await
        .map_err(|_| "Cognitum set-arc request failed".to_string())?;
    if !response.status().is_success() {
        return Err("Cognitum set-arc planning was rejected".into());
    }
    let chat: ChatResponse = read_bounded_json(response).await?;
    let content = chat
        .choices
        .first()
        .map(|choice| choice.message.content.trim())
        .ok_or_else(|| "Cognitum returned an empty plan".to_string())?;
    let json_start = content
        .find('{')
        .ok_or_else(|| "Cognitum returned an invalid plan".to_string())?;
    let json_end = content
        .rfind('}')
        .ok_or_else(|| "Cognitum returned an invalid plan".to_string())?;
    let arc: SetArc = serde_json::from_str(&content[json_start..=json_end])
        .map_err(|_| "Cognitum returned an invalid plan".to_string())?;
    validate_set_arc(&arc, duration_minutes, &style_ids, &scene_ids)?;
    Ok(arc)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AutoDjBrief {
    brief: String,
    mood: String,
}

#[tauri::command]
pub(crate) async fn cognitum_autodj_brief(
    provider: State<'_, CognitumProvider>,
    style_label: String,
    bpm: u16,
    phrase: u32,
    personalization: String,
    previous_brief: String,
) -> Result<AutoDjBrief, String> {
    if !(60..=200).contains(&bpm) {
        return Err("BPM is out of range".into());
    }
    if personalization.len() > 400 || previous_brief.len() > 500 || style_label.len() > 40 {
        return Err("Auto DJ brief inputs are too long".into());
    }
    let token = fresh_access_token(&provider).await?;

    let system = "You direct one continuous Lyria RealTime main stereo stream, one 32-bar phrase at a time. Return only JSON with keys brief (max 400 chars: a dense production direction covering groove, instrumentation, motif development, energy arc, transition into the phrase, mix character, and exclusions — never change tempo, never multiple songs or streams, no vocals) and mood (max 24 chars: two or three words describing the phrase's emotional color, e.g. 'dark rising tension'). Each phrase must audibly evolve from the previous one while keeping the set coherent.".to_string();
    let user = format!(
        "Style: {style_label}. Master tempo: {bpm} BPM. Phrase number: {phrase}. Set personalization: {personalization}. Previous phrase direction: {previous}.",
        previous = if previous_brief.trim().is_empty() {
            "none — this is the opening phrase"
        } else {
            previous_brief.trim()
        },
    );

    let client = http_client()?;
    let request = ChatRequest {
        model: "meta-llm",
        messages: [
            ChatMessage {
                role: "system",
                content: system,
            },
            ChatMessage {
                role: "user",
                content: user,
            },
        ],
        temperature: 0.75,
        max_tokens: 400,
    };
    let response = client
        .post(format!("{}/v1/chat/completions", api_base()))
        .bearer_auth(token)
        .json(&request)
        .send()
        .await
        .map_err(|_| "Cognitum brief request failed".to_string())?;
    if !response.status().is_success() {
        return Err("Cognitum brief was rejected".into());
    }
    let chat: ChatResponse = read_bounded_json(response).await?;
    let content = chat
        .choices
        .first()
        .map(|choice| choice.message.content.trim())
        .ok_or_else(|| "Cognitum returned an empty brief".to_string())?;
    let json_start = content
        .find('{')
        .ok_or_else(|| "Cognitum returned an invalid brief".to_string())?;
    let json_end = content
        .rfind('}')
        .ok_or_else(|| "Cognitum returned an invalid brief".to_string())?;
    #[derive(Deserialize)]
    struct RawBrief {
        brief: String,
        #[serde(default)]
        mood: String,
    }
    let raw: RawBrief = serde_json::from_str(&content[json_start..=json_end])
        .map_err(|_| "Cognitum returned an invalid brief".to_string())?;
    let brief = raw.brief.trim();
    if brief.is_empty() {
        return Err("Cognitum returned an empty brief".into());
    }
    Ok(AutoDjBrief {
        brief: brief.chars().take(400).collect(),
        mood: raw.mood.trim().chars().take(24).collect(),
    })
}

const FX_EFFECTS: [&str; 7] = [
    "flanger", "phaser", "drive", "crush", "sweep", "reverb", "echo",
];

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FxMove {
    effect: String,
    target: f32,
    at_bar: u16,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FxDirection {
    summary: String,
    moves: Vec<FxMove>,
}

#[tauri::command]
pub(crate) async fn cognitum_fx_direction(
    provider: State<'_, CognitumProvider>,
    mood: String,
    bars: u16,
) -> Result<FxDirection, String> {
    let trimmed = mood.trim();
    if trimmed.is_empty() || trimmed.len() > 300 {
        return Err("Describe the mood in 1 to 300 characters".into());
    }
    if !(4..=128).contains(&bars) {
        return Err("FX direction length must be 4 to 128 bars".into());
    }
    let token = fresh_access_token(&provider).await?;

    let system = format!(
        "You automate a live master effects rack over a fixed number of bars. Return only JSON with keys summary (max 90 chars) and moves: 2 to 12 objects with effect (one of: {effects}), target (0-1, the effect amount to reach), and atBar (integer 0 to the bar budget minus one; 0 means immediately). Design a musical journey: introduce effects gradually, resolve back toward dry (targets at or near 0) by the final bars unless the mood demands otherwise. Never exceed 0.85 for drive or crush.",
        effects = FX_EFFECTS.join(", "),
    );
    let user = format!("Bar budget: {bars}. Mood: {trimmed}");

    let client = http_client()?;
    let request = ChatRequest {
        model: "meta-llm",
        messages: [
            ChatMessage {
                role: "system",
                content: system,
            },
            ChatMessage {
                role: "user",
                content: user,
            },
        ],
        temperature: 0.65,
        max_tokens: 500,
    };
    let response = client
        .post(format!("{}/v1/chat/completions", api_base()))
        .bearer_auth(token)
        .json(&request)
        .send()
        .await
        .map_err(|_| "Cognitum FX direction request failed".to_string())?;
    if !response.status().is_success() {
        return Err("Cognitum FX direction was rejected".into());
    }
    let chat: ChatResponse = read_bounded_json(response).await?;
    let content = chat
        .choices
        .first()
        .map(|choice| choice.message.content.trim())
        .ok_or_else(|| "Cognitum returned an empty FX direction".to_string())?;
    let json_start = content
        .find('{')
        .ok_or_else(|| "Cognitum returned an invalid FX direction".to_string())?;
    let json_end = content
        .rfind('}')
        .ok_or_else(|| "Cognitum returned an invalid FX direction".to_string())?;
    let direction: FxDirection = serde_json::from_str(&content[json_start..=json_end])
        .map_err(|_| "Cognitum returned an invalid FX direction".to_string())?;
    validate_fx_direction(&direction, bars)?;
    Ok(direction)
}

fn validate_fx_direction(direction: &FxDirection, bars: u16) -> Result<(), String> {
    if direction.summary.len() > 90 {
        return Err("FX direction summary is too long".into());
    }
    if !(2..=12).contains(&direction.moves.len()) {
        return Err("FX direction must have 2 to 12 moves".into());
    }
    for fx_move in &direction.moves {
        if !FX_EFFECTS.contains(&fx_move.effect.as_str()) {
            return Err("FX direction references an unknown effect".into());
        }
        if !(0.0..=1.0).contains(&fx_move.target) {
            return Err("FX direction target is out of range".into());
        }
        if fx_move.at_bar >= bars {
            return Err("FX direction move is beyond the bar budget".into());
        }
        if matches!(fx_move.effect.as_str(), "drive" | "crush") && fx_move.target > 0.85 {
            return Err("FX direction pushes distortion beyond the safe ceiling".into());
        }
    }
    Ok(())
}

fn validate_set_arc(
    arc: &SetArc,
    duration_minutes: u16,
    style_ids: &[String],
    scene_ids: &[String],
) -> Result<(), String> {
    if arc.title.trim().is_empty() || arc.title.len() > 40 {
        return Err("Generated plan title is invalid".into());
    }
    if !(6..=14).contains(&arc.steps.len()) {
        return Err("Generated plan must have 6 to 14 steps".into());
    }
    let mut previous = -1.0_f32;
    for step in &arc.steps {
        if !step.at_minute.is_finite() || step.at_minute < 0.0 {
            return Err("Generated plan step time is invalid".into());
        }
        if step.at_minute <= previous || step.at_minute >= f32::from(duration_minutes) {
            return Err("Generated plan step times must strictly increase within the set".into());
        }
        previous = step.at_minute;
        if !style_ids.contains(&step.style_id) {
            return Err("Generated plan references an unknown style".into());
        }
        if !scene_ids.contains(&step.visual_scene) {
            return Err("Generated plan references an unknown scene".into());
        }
        if !(60..=200).contains(&step.bpm) {
            return Err("Generated plan BPM is out of range".into());
        }
        if step.note.len() > 90 {
            return Err("Generated plan note is too long".into());
        }
        if let Some(fx) = &step.fx {
            for value in [fx.sweep, fx.reverb, fx.echo, fx.flanger]
                .into_iter()
                .flatten()
            {
                if !(0.0..=1.0).contains(&value) {
                    return Err("Generated plan FX values are out of range".into());
                }
            }
        }
    }
    Ok(())
}

fn validate_style_pack(pack: &StylePack) -> Result<(), String> {
    if pack.label.trim().is_empty() || pack.label.len() > 24 {
        return Err("Generated style label is invalid".into());
    }
    if pack.prompts.len() != 4 {
        return Err("Generated style must have exactly four prompts".into());
    }
    for (index, prompt) in pack.prompts.iter().enumerate() {
        if prompt.text.trim().is_empty() || prompt.text.len() > 240 {
            return Err("Generated style prompt length is invalid".into());
        }
        let weight_ok = if index == 3 {
            prompt.weight <= -0.5 && prompt.weight >= -2.0
        } else {
            prompt.weight > 0.0 && prompt.weight <= 2.0
        };
        if !weight_ok {
            return Err("Generated style prompt weights are invalid".into());
        }
    }
    let config = &pack.config;
    if !(60..=200).contains(&config.bpm)
        || !(0.0..=1.0).contains(&config.density)
        || !(0.0..=1.0).contains(&config.brightness)
        || !(0.0..=6.0).contains(&config.guidance)
    {
        return Err("Generated style config is out of range".into());
    }
    Ok(())
}

fn urlencoding_encode(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len() * 3);
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                encoded.push('%');
                encoded.push_str(&format!("{byte:02X}"));
            }
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_parsing_requires_matching_state() {
        let line = "GET /callback?code=abc123&state=expected HTTP/1.1";
        assert_eq!(
            parse_callback_query(line, "expected"),
            Some("abc123".into())
        );
        assert_eq!(parse_callback_query(line, "other"), None);
        assert_eq!(
            parse_callback_query("GET /callback HTTP/1.1", "expected"),
            None
        );
    }

    #[test]
    fn style_pack_validation_bounds_prompts_and_config() {
        let valid = StylePack {
            label: "Night Drive".into(),
            description: "Test".into(),
            prompts: vec![
                StylePackPrompt {
                    text: "identity".into(),
                    weight: 1.3,
                },
                StylePackPrompt {
                    text: "blend".into(),
                    weight: 1.1,
                },
                StylePackPrompt {
                    text: "arc".into(),
                    weight: 1.0,
                },
                StylePackPrompt {
                    text: "avoid".into(),
                    weight: -1.1,
                },
            ],
            config: StylePackConfig {
                bpm: 120,
                density: 0.5,
                brightness: 0.5,
                guidance: 5.0,
                scale: "C_MAJOR_A_MINOR".into(),
            },
        };
        assert!(validate_style_pack(&valid).is_ok());

        let mut bad_weight = valid.clone();
        bad_weight.prompts[3].weight = 1.0;
        assert!(validate_style_pack(&bad_weight).is_err());

        let mut bad_bpm = valid.clone();
        bad_bpm.config.bpm = 20;
        assert!(validate_style_pack(&bad_bpm).is_err());
    }

    #[test]
    fn set_arc_validation_enforces_timeline_and_vocabulary() {
        let styles = vec!["rock".to_string(), "techno".to_string()];
        let scenes = vec!["oscilloscope".to_string(), "terrain".to_string()];
        let step = |minute: f32, style: &str| SetArcStep {
            at_minute: minute,
            style_id: style.into(),
            visual_scene: "terrain".into(),
            bpm: 126,
            fx: Some(SetArcFx {
                sweep: Some(0.2),
                ..SetArcFx::default()
            }),
            note: "build".into(),
        };
        let valid = SetArc {
            title: "Night Arc".into(),
            duration_minutes: 60,
            steps: (0..6).map(|i| step(i as f32 * 9.0, "rock")).collect(),
        };
        assert!(validate_set_arc(&valid, 60, &styles, &scenes).is_ok());

        let mut unknown_style = valid.clone();
        unknown_style.steps[2].style_id = "polka".into();
        assert!(validate_set_arc(&unknown_style, 60, &styles, &scenes).is_err());

        let mut out_of_order = valid.clone();
        out_of_order.steps[3].at_minute = 1.0;
        assert!(validate_set_arc(&out_of_order, 60, &styles, &scenes).is_err());

        let mut beyond_end = valid.clone();
        beyond_end.steps[5].at_minute = 75.0;
        assert!(validate_set_arc(&beyond_end, 60, &styles, &scenes).is_err());

        let mut too_few = valid.clone();
        too_few.steps.truncate(3);
        assert!(validate_set_arc(&too_few, 60, &styles, &scenes).is_err());
    }

    #[test]
    fn url_encoding_escapes_reserved_characters() {
        assert_eq!(urlencoding_encode("a b&c"), "a%20b%26c");
        assert_eq!(urlencoding_encode("safe-._~"), "safe-._~");
    }
}
