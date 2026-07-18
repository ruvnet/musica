use std::{env, time::Duration};

use reqwest::{redirect::Policy, Client, Url};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const ENABLE_ENV: &str = "MUSICA_META_LLM_ENABLED";
const TOKEN_ENV: &str = "MUSICA_META_LLM_API_TOKEN";
const BASE_URL_ENV: &str = "MUSICA_META_LLM_API_BASE";
const MODEL_ENV: &str = "MUSICA_META_LLM_MODEL";
const CHAT_PATH_ENV: &str = "MUSICA_META_LLM_CHAT_PATH";
const DEFAULT_BASE_URL: &str = "https://api.cognitum.one";
const DEFAULT_CHAT_PATH: &str = "/v1/chat/completions";
const DEFAULT_MODEL: &str = "meta-llm";
const COGNITUM_HOST: &str = "api.cognitum.one";
const MAX_GOAL_CHARS: usize = 1_500;
const MAX_RESPONSE_BYTES: usize = 256 * 1024;

#[derive(Clone)]
pub(crate) struct MetaLlmProvider {
    status: AgentStatus,
    client: Option<Client>,
    endpoint: Option<Url>,
    token: Option<String>,
    model: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AgentStatus {
    available: bool,
    provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint_host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct AgentPlanRequest {
    goal: String,
    current_prompt: String,
    bpm: u16,
    scene: String,
    selected_track: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AgentPlan {
    title: String,
    rationale: String,
    prompt: String,
    template_id: String,
    scene: String,
    bpm: u16,
    intensity: f32,
    art_direction: AgentArtDirection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temporal: Option<AgentTemporalControls>,
    arrangement_notes: Vec<String>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AgentArtDirection {
    sculpture: f32,
    motion: f32,
    atmosphere: f32,
    ribbon: f32,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AgentTemporalControls {
    speed: f32,
    strobe: f32,
    trail: f32,
    morph: f32,
    camera: f32,
    phase: f32,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: [ChatMessage<'a>; 2],
    temperature: f32,
    max_tokens: u16,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
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

#[derive(Debug, Error)]
enum AgentError {
    #[error("Meta-LLM agent is unavailable")]
    Unavailable,
    #[error("Meta-LLM agent configuration rejected: {0}")]
    Configuration(&'static str),
    #[error("Meta-LLM agent request is invalid: {0}")]
    Validation(&'static str),
    #[error("Meta-LLM agent request failed")]
    Network,
    #[error("Meta-LLM agent returned an invalid response")]
    MalformedResponse,
    #[error("Meta-LLM agent response exceeded the size limit")]
    ResponseTooLarge,
}

impl MetaLlmProvider {
    pub(crate) fn from_env() -> Self {
        let enabled = env::var(ENABLE_ENV)
            .map(|value| matches!(value.as_str(), "1" | "true"))
            .unwrap_or(false);
        if !enabled {
            return Self::disabled("Set MUSICA_META_LLM_ENABLED=true to enable the agent director");
        }

        match Self::build() {
            Ok(provider) => provider,
            Err(error) => Self::disabled(error.to_string()),
        }
    }

    fn build() -> Result<Self, AgentError> {
        let base = env::var(BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        let path = env::var(CHAT_PATH_ENV).unwrap_or_else(|_| DEFAULT_CHAT_PATH.into());
        let endpoint = validate_endpoint(&base, &path)?;
        let host = endpoint
            .host_str()
            .ok_or(AgentError::Configuration("Meta-LLM API URL has no host"))?
            .to_owned();
        let token = env::var(TOKEN_ENV)
            .map_err(|_| AgentError::Configuration("MUSICA_META_LLM_API_TOKEN is required"))?;
        validate_token(&token)?;
        let model = env::var(MODEL_ENV).unwrap_or_else(|_| DEFAULT_MODEL.into());
        validate_model(&model)?;
        let client = Client::builder()
            .https_only(true)
            .redirect(Policy::none())
            .no_proxy()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(45))
            .pool_idle_timeout(Duration::from_secs(30))
            .user_agent("Musica-VJ/0.1 Meta-LLM")
            .build()
            .map_err(|_| AgentError::Configuration("could not initialize Meta-LLM HTTPS client"))?;
        Ok(Self {
            status: AgentStatus {
                available: true,
                provider: "meta_llm".into(),
                endpoint_host: Some(host),
                model: Some(model.clone()),
                reason: None,
            },
            client: Some(client),
            endpoint: Some(endpoint),
            token: Some(token),
            model,
        })
    }

    fn disabled(reason: impl Into<String>) -> Self {
        Self {
            status: AgentStatus {
                available: false,
                provider: "meta_llm".into(),
                endpoint_host: None,
                model: None,
                reason: Some(reason.into()),
            },
            client: None,
            endpoint: None,
            token: None,
            model: DEFAULT_MODEL.into(),
        }
    }

    pub(crate) fn status(&self) -> AgentStatus {
        self.status.clone()
    }

    pub(crate) async fn plan(&self, request: AgentPlanRequest) -> Result<AgentPlan, String> {
        request.validate().map_err(|error| error.to_string())?;
        let client = self
            .client
            .as_ref()
            .ok_or(AgentError::Unavailable)
            .map_err(|error| error.to_string())?;
        let endpoint = self
            .endpoint
            .as_ref()
            .ok_or(AgentError::Unavailable)
            .map_err(|error| error.to_string())?;
        let token = self
            .token
            .as_ref()
            .ok_or(AgentError::Unavailable)
            .map_err(|error| error.to_string())?;
        let response = client
            .post(endpoint.clone())
            .bearer_auth(token)
            .json(&ChatRequest {
                model: &self.model,
                messages: [
                    ChatMessage {
                        role: "system",
                        content: agent_system_prompt(),
                    },
                    ChatMessage {
                        role: "user",
                        content: agent_user_prompt(&request),
                    },
                ],
                temperature: 0.55,
                max_tokens: 700,
            })
            .send()
            .await
            .map_err(|_| AgentError::Network.to_string())?;
        let plan = parse_chat_response(response)
            .await
            .map_err(|error| error.to_string())?;
        validate_plan(plan).map_err(|error| error.to_string())
    }
}

impl AgentPlanRequest {
    fn validate(&self) -> Result<(), AgentError> {
        let goal = self.goal.trim();
        if goal.is_empty() || goal.chars().count() > MAX_GOAL_CHARS {
            return Err(AgentError::Validation(
                "goal must contain between 1 and 1500 characters",
            ));
        }
        if self
            .goal
            .chars()
            .chain(self.current_prompt.chars())
            .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
        {
            return Err(AgentError::Validation(
                "agent prompt contains a disallowed control character",
            ));
        }
        if !(60..=200).contains(&self.bpm) {
            return Err(AgentError::Validation("bpm must be between 60 and 200"));
        }
        if !valid_identifier(&self.scene) || !valid_identifier(&self.selected_track) {
            return Err(AgentError::Validation("scene or track is invalid"));
        }
        Ok(())
    }
}

async fn parse_chat_response(mut response: reqwest::Response) -> Result<AgentPlan, AgentError> {
    if !response.status().is_success() {
        return Err(AgentError::Network);
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        return Err(AgentError::ResponseTooLarge);
    }
    let mut body = Vec::with_capacity(
        response
            .content_length()
            .unwrap_or(8 * 1024)
            .min(MAX_RESPONSE_BYTES as u64) as usize,
    );
    while let Some(chunk) = response.chunk().await.map_err(|_| AgentError::Network)? {
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(AgentError::ResponseTooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    let chat: ChatResponse =
        serde_json::from_slice(&body).map_err(|_| AgentError::MalformedResponse)?;
    let content = chat
        .choices
        .first()
        .map(|choice| choice.message.content.trim())
        .filter(|content| !content.is_empty())
        .ok_or(AgentError::MalformedResponse)?;
    let json = extract_json(content).ok_or(AgentError::MalformedResponse)?;
    serde_json::from_str(json).map_err(|_| AgentError::MalformedResponse)
}

fn validate_plan(plan: AgentPlan) -> Result<AgentPlan, AgentError> {
    if plan.title.trim().is_empty()
        || plan.title.chars().count() > 80
        || plan.rationale.chars().count() > 400
        || plan.prompt.trim().is_empty()
        || plan.prompt.chars().count() > 1_000
        || !matches!(
            plan.template_id.as_str(),
            "warehouse-techno"
                | "liquid-breaks"
                | "ambient-dub"
                | "synthwave-drive"
                | "footwork-cuts"
                | "cinematic-pulse"
        )
        || !matches!(
            plan.scene.as_str(),
            "tunnel"
                | "bloom"
                | "terrain"
                | "lasergrid"
                | "aurora"
                | "monolith"
                | "pulsefield"
                | "chromawave"
        )
        || !(60..=200).contains(&plan.bpm)
        || !bounded_unit(plan.intensity)
        || !bounded_unit(plan.art_direction.sculpture)
        || !bounded_unit(plan.art_direction.motion)
        || !bounded_unit(plan.art_direction.atmosphere)
        || !bounded_unit(plan.art_direction.ribbon)
        || plan.temporal.as_ref().is_some_and(|temporal| {
            !bounded_unit(temporal.speed)
                || !bounded_unit(temporal.strobe)
                || !bounded_unit(temporal.trail)
                || !bounded_unit(temporal.morph)
                || !bounded_unit(temporal.camera)
                || !bounded_unit(temporal.phase)
        })
        || plan.arrangement_notes.len() > 6
    {
        return Err(AgentError::MalformedResponse);
    }
    Ok(plan)
}

fn agent_system_prompt() -> String {
    "You are Musica VJ's agentic music director. Return only JSON with keys title, rationale, prompt, templateId, scene, bpm, intensity, artDirection, temporal, arrangementNotes. Choose templateId from warehouse-techno, liquid-breaks, ambient-dub, synthwave-drive, footwork-cuts, cinematic-pulse. Choose scene from tunnel, bloom, terrain, lasergrid, aurora, monolith, pulsefield, chromawave. Use numeric unit values for artDirection and temporal. Temporal keys are speed, strobe, trail, morph, camera, phase.".into()
}

fn agent_user_prompt(request: &AgentPlanRequest) -> String {
    format!(
        "Goal: {}\nCurrent prompt: {}\nCurrent BPM: {}\nCurrent visual scene: {}\nSelected track: {}\nReturn a coherent full-performance plan.",
        request.goal.trim(),
        request.current_prompt.trim(),
        request.bpm,
        request.scene,
        request.selected_track,
    )
}

fn extract_json(content: &str) -> Option<&str> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    (start <= end).then_some(&content[start..=end])
}

fn validate_endpoint(base: &str, path: &str) -> Result<Url, AgentError> {
    let mut url = Url::parse(base)
        .map_err(|_| AgentError::Configuration("MUSICA_META_LLM_API_BASE is not a valid URL"))?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some_and(|port| port != 443)
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err(AgentError::Configuration(
            "Meta-LLM API base must be an HTTPS origin",
        ));
    }
    let host = url
        .host_str()
        .ok_or(AgentError::Configuration("Meta-LLM API URL has no host"))?;
    if !host.eq_ignore_ascii_case(COGNITUM_HOST) {
        return Err(AgentError::Configuration(
            "Meta-LLM API host is not allowlisted",
        ));
    }
    if !path.starts_with('/') || path.contains("..") || path.len() > 128 {
        return Err(AgentError::Configuration(
            "MUSICA_META_LLM_CHAT_PATH is invalid",
        ));
    }
    url.set_path(path);
    Ok(url)
}

fn validate_token(token: &str) -> Result<(), AgentError> {
    if token.len() < 24 || token.len() > 4096 || !token.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return Err(AgentError::Configuration(
            "MUSICA_META_LLM_API_TOKEN is invalid",
        ));
    }
    Ok(())
}

fn validate_model(model: &str) -> Result<(), AgentError> {
    if model.is_empty()
        || model.len() > 128
        || !model
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:/-".contains(&byte))
    {
        return Err(AgentError::Configuration(
            "MUSICA_META_LLM_MODEL is invalid",
        ));
    }
    Ok(())
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn bounded_unit(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[tauri::command]
pub(crate) fn meta_llm_status(state: tauri::State<'_, MetaLlmProvider>) -> AgentStatus {
    state.status()
}

#[tauri::command]
pub(crate) async fn meta_llm_plan(
    state: tauri::State<'_, MetaLlmProvider>,
    request: AgentPlanRequest,
) -> Result<AgentPlan, String> {
    state.plan(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_is_pinned_to_cognitum_https_origin() {
        assert!(validate_endpoint("https://api.cognitum.one", "/v1/chat/completions").is_ok());
        assert!(validate_endpoint("http://api.cognitum.one", "/v1/chat/completions").is_err());
        assert!(
            validate_endpoint("https://api.cognitum.one.evil.test", "/v1/chat/completions")
                .is_err()
        );
        assert!(
            validate_endpoint("https://api.cognitum.one:8443", "/v1/chat/completions").is_err()
        );
        assert!(validate_endpoint("https://api.cognitum.one", "../chat").is_err());
    }

    #[test]
    fn validates_agent_plan_contract() {
        let plan = AgentPlan {
            title: "Warehouse Techno".into(),
            rationale: "Matches the request.".into(),
            prompt: "warehouse techno".into(),
            template_id: "warehouse-techno".into(),
            scene: "lasergrid".into(),
            bpm: 132,
            intensity: 0.86,
            art_direction: AgentArtDirection {
                sculpture: 0.7,
                motion: 0.8,
                atmosphere: 0.4,
                ribbon: 0.6,
            },
            temporal: Some(AgentTemporalControls {
                speed: 0.8,
                strobe: 0.2,
                trail: 0.7,
                morph: 0.6,
                camera: 0.8,
                phase: 0.1,
            }),
            arrangement_notes: vec!["Apply full template".into()],
        };
        assert!(validate_plan(plan).is_ok());
    }
}
