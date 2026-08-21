use std::{env, sync::Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::State;

use crate::cognitum_provider::{
    fresh_access_token, http_client, read_bounded_json, CognitumProvider,
};

/// Settings broadcast + live follow (ADR-182).
///
/// Rust owns the network hop for the same reason it owns the Lyria broker call
/// (ADR-179): the Cognitum bearer never reaches the webview. The snapshot
/// itself is treated as opaque JSON here — its schema lives in TypeScript
/// (`src/core/broadcast.ts`), which also owns clamping — so this layer enforces
/// only what Rust is actually the right place for: authentication and size.
const BROADCAST_URL_ENV: &str = "MUSICA_BROADCAST_URL";
/// The deployed Cognitum org service. Baked in so a shipped build can broadcast
/// purely from signing in, with no config; override via `MUSICA_BROADCAST_URL`.
const DEFAULT_BROADCAST_URL: &str =
    "https://us-central1-cognitum-20260110.cloudfunctions.net/musicaBroadcast";

/// A snapshot is a few hundred bytes of parameters. This bound is generous
/// enough for the deck-scene list and prompts, and small enough that a
/// malicious or buggy client cannot use the service as storage.
const MAX_SNAPSHOT_BYTES: usize = 16 * 1024;
const MAX_DISPLAY_NAME_CHARS: usize = 32;
const MAX_ID_CHARS: usize = 64;

#[derive(Default)]
pub(crate) struct BroadcastProvider {
    state: Mutex<BroadcastState>,
}

#[derive(Default)]
struct BroadcastState {
    /// The opaque public id the service assigned to this user's broadcast.
    /// Never the OAuth subject — see ADR-182.
    id: Option<String>,
    live: bool,
}

#[derive(Serialize)]
struct PublishRequest<'a> {
    display_name: &'a str,
    snapshot: &'a Value,
}

#[derive(Deserialize)]
struct PublishResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    live: bool,
    #[serde(default)]
    listeners: u32,
}

// camelCase across the Rust -> TypeScript boundary, matching `CognitumStatus`
// and the rest of the provider surface.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BroadcastPublishState {
    live: bool,
    id: Option<String>,
    display_name: Option<String>,
    listeners: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BroadcastAvailability {
    configured: bool,
    live: bool,
    id: Option<String>,
}

fn broadcast_base() -> Option<String> {
    let url = env::var(BROADCAST_URL_ENV)
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_BROADCAST_URL.to_string());
    // The bearer rides on every one of these calls, so TLS is required off-box.
    // Plaintext loopback is allowed for local development only — it never
    // leaves the machine, and it matches how `MUSICA_COGNITUM_API_BASE` already
    // permits a loopback meta-proxy.
    Some(url).filter(|value| value.starts_with("https://") || is_plaintext_loopback(value))
}

/// True only for a plaintext URL whose *host* is loopback. A prefix test alone
/// would accept `http://localhost.evil.example.com`, so the host must end at a
/// port, a path, or the end of the string.
fn is_plaintext_loopback(url: &str) -> bool {
    const HOSTS: [&str; 3] = ["127.0.0.1", "localhost", "[::1]"];
    let Some(rest) = url.strip_prefix("http://") else {
        return false;
    };
    HOSTS.iter().any(|host| {
        rest.strip_prefix(host)
            .is_some_and(|tail| tail.is_empty() || tail.starts_with(':') || tail.starts_with('/'))
    })
}

/// Rejects an id that could escape its path segment. Ids are opaque tokens the
/// service minted, so anything with structure is not one.
fn safe_id(id: &str) -> Result<String, String> {
    let trimmed = id.trim();
    if trimmed.is_empty() || trimmed.chars().count() > MAX_ID_CHARS {
        return Err("Invalid broadcast id".into());
    }
    if !trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err("Invalid broadcast id".into());
    }
    Ok(trimmed.to_string())
}

fn truncate_chars(value: &str, limit: usize) -> String {
    value.trim().chars().take(limit).collect()
}

/// Reports whether broadcasting is reachable at all, so the UI can explain
/// itself rather than failing on click.
#[tauri::command]
pub(crate) fn broadcast_status(provider: State<'_, BroadcastProvider>) -> BroadcastAvailability {
    let state = provider.state.lock().expect("broadcast state");
    BroadcastAvailability {
        configured: broadcast_base().is_some(),
        live: state.live,
        id: state.id.clone(),
    }
}

#[tauri::command]
pub(crate) async fn broadcast_publish(
    cognitum: State<'_, CognitumProvider>,
    provider: State<'_, BroadcastProvider>,
    display_name: String,
    snapshot: Value,
) -> Result<BroadcastPublishState, String> {
    let base = broadcast_base().ok_or_else(|| "No broadcast service configured".to_string())?;
    let encoded = serde_json::to_vec(&snapshot)
        .map_err(|_| "Broadcast snapshot could not be encoded".to_string())?;
    if encoded.len() > MAX_SNAPSHOT_BYTES {
        return Err("Broadcast snapshot is too large".into());
    }

    let token = fresh_access_token(&cognitum).await?;
    let name = truncate_chars(&display_name, MAX_DISPLAY_NAME_CHARS);
    let client = http_client()?;
    let response = client
        .post(format!("{base}/broadcast"))
        .bearer_auth(token)
        .json(&PublishRequest {
            display_name: &name,
            snapshot: &snapshot,
        })
        .send()
        .await
        .map_err(|_| "Broadcast request failed".to_string())?;
    if !response.status().is_success() {
        return Err("Broadcast service rejected the request".into());
    }
    let published: PublishResponse = read_bounded_json(response).await?;

    let mut state = provider.state.lock().expect("broadcast state");
    state.id = published.id.clone();
    state.live = published.live;
    Ok(BroadcastPublishState {
        live: published.live,
        id: published.id,
        display_name: published.display_name,
        listeners: published.listeners,
    })
}

/// Goes offline without deleting the snapshot: a follower who arrives after the
/// broadcaster quits still lands on where the set ended (ADR-182).
#[tauri::command]
pub(crate) async fn broadcast_stop(
    cognitum: State<'_, CognitumProvider>,
    provider: State<'_, BroadcastProvider>,
) -> Result<(), String> {
    let base = broadcast_base().ok_or_else(|| "No broadcast service configured".to_string())?;
    let token = fresh_access_token(&cognitum).await?;
    let client = http_client()?;
    let response = client
        .delete(format!("{base}/broadcast"))
        .bearer_auth(token)
        .send()
        .await
        .map_err(|_| "Broadcast stop request failed".to_string())?;
    // Local state goes offline regardless: a failed stop must not leave the UI
    // claiming to be live.
    {
        let mut state = provider.state.lock().expect("broadcast state");
        state.live = false;
    }
    if !response.status().is_success() {
        return Err("Broadcast service rejected the stop request".into());
    }
    Ok(())
}

#[tauri::command]
pub(crate) async fn broadcast_list(cognitum: State<'_, CognitumProvider>) -> Result<Value, String> {
    let base = broadcast_base().ok_or_else(|| "No broadcast service configured".to_string())?;
    let token = fresh_access_token(&cognitum).await?;
    let client = http_client()?;
    let response = client
        .get(format!("{base}/broadcasts"))
        .bearer_auth(token)
        .send()
        .await
        .map_err(|_| "Broadcast directory request failed".to_string())?;
    if !response.status().is_success() {
        return Err("Broadcast service rejected the directory request".into());
    }
    read_bounded_json(response).await
}

/// One authenticated round-trip that renews the follower's presence and
/// returns the broadcaster's current state. `since` lets the service omit an
/// unchanged snapshot body, so an idle set costs a heartbeat and nothing more.
#[tauri::command]
pub(crate) async fn broadcast_listen(
    cognitum: State<'_, CognitumProvider>,
    id: String,
    since: Option<u64>,
) -> Result<Value, String> {
    let base = broadcast_base().ok_or_else(|| "No broadcast service configured".to_string())?;
    let id = safe_id(&id)?;
    let token = fresh_access_token(&cognitum).await?;
    let client = http_client()?;
    let mut request = client
        .post(format!("{base}/listen/{id}"))
        .bearer_auth(token);
    if let Some(since) = since {
        request = request.query(&[("since", since.to_string())]);
    }
    let response = request
        .send()
        .await
        .map_err(|_| "Broadcast follow request failed".to_string())?;
    if !response.status().is_success() {
        return Err("Broadcast service rejected the follow request".into());
    }
    read_bounded_json(response).await
}

#[tauri::command]
pub(crate) async fn broadcast_leave(
    cognitum: State<'_, CognitumProvider>,
    id: String,
) -> Result<(), String> {
    let base = broadcast_base().ok_or_else(|| "No broadcast service configured".to_string())?;
    let id = safe_id(&id)?;
    let token = fresh_access_token(&cognitum).await?;
    let client = http_client()?;
    // Best-effort: presence correctness rests on expiry, not on this call.
    let _ = client
        .delete(format!("{base}/listen/{id}"))
        .bearer_auth(token)
        .send()
        .await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_ids_that_could_escape_their_path_segment() {
        assert!(safe_id("../../admin").is_err());
        assert!(safe_id("a/b").is_err());
        assert!(safe_id("id?since=1").is_err());
        assert!(safe_id("").is_err());
        assert!(safe_id("   ").is_err());
        assert!(safe_id(&"a".repeat(MAX_ID_CHARS + 1)).is_err());
    }

    #[test]
    fn accepts_an_opaque_service_minted_id() {
        assert_eq!(safe_id(" abc-123_XYZ ").unwrap(), "abc-123_XYZ");
        assert_eq!(
            safe_id(&"a".repeat(MAX_ID_CHARS)).unwrap().len(),
            MAX_ID_CHARS
        );
    }

    #[test]
    fn truncates_display_names_on_char_boundaries() {
        // Byte truncation would split a multi-byte character; this counts chars.
        assert_eq!(
            truncate_chars("  dj nova  ", MAX_DISPLAY_NAME_CHARS),
            "dj nova"
        );
        assert_eq!(
            truncate_chars(&"é".repeat(100), MAX_DISPLAY_NAME_CHARS)
                .chars()
                .count(),
            MAX_DISPLAY_NAME_CHARS
        );
    }

    /// One test, because these all mutate the same process environment and
    /// Rust runs test functions on parallel threads.
    #[test]
    fn accepts_only_tls_or_genuine_loopback_bases() {
        // The bearer rides on every call, so a plaintext remote override is
        // refused rather than silently downgraded.
        std::env::set_var(BROADCAST_URL_ENV, "http://insecure.example.com");
        assert!(broadcast_base().is_none());

        // A host that merely *starts with* a loopback name must not slip past.
        for impostor in [
            "http://localhost.evil.example.com",
            "http://127.0.0.1.evil.example.com",
            "http://localhostile.example.com",
        ] {
            std::env::set_var(BROADCAST_URL_ENV, impostor);
            assert!(broadcast_base().is_none(), "{impostor} must be refused");
        }

        // Plaintext loopback is allowed so the service can be run locally.
        for base in [
            "http://127.0.0.1:8080",
            "http://localhost:8080",
            "http://[::1]:8080",
        ] {
            std::env::set_var(BROADCAST_URL_ENV, base);
            assert_eq!(broadcast_base().unwrap(), base);
        }

        std::env::set_var(BROADCAST_URL_ENV, "https://example.com/svc/");
        assert_eq!(broadcast_base().unwrap(), "https://example.com/svc");
        std::env::remove_var(BROADCAST_URL_ENV);
        assert_eq!(broadcast_base().unwrap(), DEFAULT_BROADCAST_URL);
    }

    /// The webview reads `displayName`; serde's default would emit
    /// `display_name` and the field would silently arrive as undefined.
    #[test]
    fn serializes_to_the_camel_case_shape_the_webview_reads() {
        let json = serde_json::to_value(BroadcastPublishState {
            live: true,
            id: Some("abc".into()),
            display_name: Some("DJ Nova".into()),
            listeners: 3,
        })
        .unwrap();
        assert_eq!(json["displayName"], "DJ Nova");
        assert_eq!(json["listeners"], 3);
        assert!(json.get("display_name").is_none());
    }
}
