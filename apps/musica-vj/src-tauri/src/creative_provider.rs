use std::{env, path::PathBuf, sync::Arc, time::Duration};

use reqwest::{redirect::Policy, Client, Response as HttpResponse, Url};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::ipc::Response as IpcResponse;
use tauri::{AppHandle, State};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use thiserror::Error;
use tokio::{sync::Semaphore, time::timeout};

use crate::lyria_provider::LyriaProvider;

const ENABLE_ENV: &str = "MUSICA_CREATIVE_ENABLED";
const PROVIDER_ENV: &str = "MUSICA_CREATIVE_PROVIDER";
const BASE_URL_ENV: &str = "MUSICA_CREATIVE_API_BASE";
const TOKEN_ENV: &str = "MUSICA_CREATIVE_API_TOKEN";
const MODEL_ENV: &str = "MUSICA_CREATIVE_MODEL";

const DEFAULT_BASE_URL: &str = "https://api.cognitum.one";
const COGNITUM_HOST: &str = "api.cognitum.one";

/// An OEM/partner build may embed additional contractual API hosts at compile
/// time. Runtime environment variables can select a host, but cannot expand
/// this allowlist. Never add reverse-engineered consumer endpoints here.
const BUILD_ALLOWED_HOSTS: Option<&str> = option_env!("MUSICA_CREATIVE_ALLOWED_HOSTS");

const MAX_PROMPT_CHARS: usize = 16_000;
const MIN_DURATION_SECONDS: u16 = 5;
const MAX_DURATION_SECONDS: u16 = 184;
const PARTNER_MAX_DURATION_SECONDS: u16 = 120;
const MAX_TASK_ID_BYTES: usize = 128;
const MAX_RESPONSE_BYTES: usize = 512 * 1024;
const MAX_AUDIO_BYTES: usize = 64 * 1024 * 1024;
const MAX_AUDIO_URL_BYTES: usize = 4_096;
const MAX_TITLE_CHARS: usize = 200;
const MAX_MODEL_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProviderKind {
    AsyncPartner,
    SunoPartner,
    Lyria3Pro,
}

impl ProviderKind {
    fn parse(value: &str) -> Result<Self, ProviderError> {
        match value {
            "async_partner" => Ok(Self::AsyncPartner),
            "suno_partner" => Ok(Self::SunoPartner),
            "lyria_3_pro" => Ok(Self::Lyria3Pro),
            _ => Err(ProviderError::Configuration(
                "provider must be async_partner, suno_partner, or lyria_3_pro",
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::AsyncPartner => "async_partner",
            Self::SunoPartner => "suno_partner",
            Self::Lyria3Pro => "lyria_3_pro",
        }
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProviderStatus {
    available: bool,
    provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint_host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit_cost_usd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_duration_seconds: Option<u16>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(super) enum AudioOutputFormat {
    #[default]
    Mp3,
    Wav,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct TimedSection {
    pub(super) time_seconds: f32,
    pub(super) section: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct ReferenceAsset {
    pub(super) mime_type: String,
    pub(super) storage_uri: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct GenerationRequest {
    pub(super) prompt: String,
    pub(super) duration_seconds: u16,
    pub(super) instrumental: bool,
    #[serde(default)]
    pub(super) seed: Option<u64>,
    #[serde(default)]
    pub(super) language: Option<String>,
    #[serde(default)]
    pub(super) bpm: Option<f32>,
    #[serde(default)]
    pub(super) lyrics: Option<String>,
    #[serde(default)]
    pub(super) structure: Vec<TimedSection>,
    #[serde(default)]
    pub(super) output_format: AudioOutputFormat,
    #[serde(default)]
    pub(super) reference_assets: Vec<ReferenceAsset>,
    #[serde(default)]
    pub(super) max_cost_usd: Option<f64>,
    #[serde(default = "one_candidate")]
    pub(super) candidate_count: u8,
    #[serde(default = "one_attempt")]
    pub(super) max_attempts: u8,
    #[serde(default)]
    pub(super) rights_declared: bool,
    #[serde(default)]
    pub(super) client_request_id: Option<String>,
}

const fn one_candidate() -> u8 {
    1
}

const fn one_attempt() -> u8 {
    1
}

impl GenerationRequest {
    pub(super) fn validate(&self) -> Result<(), ProviderError> {
        let prompt = self.prompt.trim();
        let prompt_chars = prompt.chars().count();
        if prompt_chars == 0 || prompt_chars > MAX_PROMPT_CHARS {
            return Err(ProviderError::Validation(
                "prompt must contain between 1 and 16000 characters",
            ));
        }
        if self
            .prompt
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
        {
            return Err(ProviderError::Validation(
                "prompt contains a disallowed control character",
            ));
        }
        if !(MIN_DURATION_SECONDS..=MAX_DURATION_SECONDS).contains(&self.duration_seconds) {
            return Err(ProviderError::Validation(
                "durationSeconds must be between 5 and 184",
            ));
        }
        if self.candidate_count != 1 {
            return Err(ProviderError::Validation(
                "candidateCount must be 1 in the initial provider contract",
            ));
        }
        if !(1..=2).contains(&self.max_attempts) {
            return Err(ProviderError::Validation("maxAttempts must be 1 or 2"));
        }
        if self
            .max_cost_usd
            .is_some_and(|cost| !cost.is_finite() || cost < 0.0)
        {
            return Err(ProviderError::Validation("maxCostUsd is invalid"));
        }
        if self
            .bpm
            .is_some_and(|bpm| !bpm.is_finite() || !(60.0..=200.0).contains(&bpm))
        {
            return Err(ProviderError::Validation("bpm must be between 60 and 200"));
        }
        if self
            .lyrics
            .as_ref()
            .is_some_and(|lyrics| lyrics.chars().count() > 12_000)
        {
            return Err(ProviderError::Validation("lyrics exceed 12000 characters"));
        }
        if self.lyrics.as_ref().is_some_and(|lyrics| {
            lyrics
                .chars()
                .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
        }) {
            return Err(ProviderError::Validation(
                "lyrics contain a disallowed control character",
            ));
        }
        if self
            .lyrics
            .as_ref()
            .is_some_and(|lyrics| !lyrics.trim().is_empty())
            && !self.rights_declared
        {
            return Err(ProviderError::Validation(
                "rightsDeclared is required for user supplied lyrics",
            ));
        }
        if self.structure.len() > 32 {
            return Err(ProviderError::Validation(
                "structure is limited to 32 sections",
            ));
        }
        if self.reference_assets.len() > 10 {
            return Err(ProviderError::Validation(
                "reference assets are limited to 10",
            ));
        }
        for asset in &self.reference_assets {
            if asset.mime_type.trim().is_empty()
                || asset.storage_uri.trim().is_empty()
                || asset.mime_type.len() > 100
                || asset.storage_uri.len() > 512
                || asset.mime_type.chars().any(char::is_control)
                || asset.storage_uri.chars().any(char::is_control)
            {
                return Err(ProviderError::Validation(
                    "reference asset metadata is invalid",
                ));
            }
        }
        let mut last_time = -1.0_f32;
        for section in &self.structure {
            if !section.time_seconds.is_finite()
                || section.time_seconds < 0.0
                || section.time_seconds >= f32::from(self.duration_seconds)
                || section.time_seconds <= last_time
                || section.section.trim().is_empty()
                || section.section.chars().count() > 80
                || section.section.chars().any(char::is_control)
            {
                return Err(ProviderError::Validation(
                    "structure contains an invalid section",
                ));
            }
            last_time = section.time_seconds;
        }
        Ok(())
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GenerationTask {
    pub(super) id: String,
    pub(super) status: GenerationStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) audio_url: Option<String>,
    pub(super) provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) model: Option<String>,
    pub(super) has_audio: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) audio_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) lyrics: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) structure: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) actual_duration_seconds: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) sample_rate_hz: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) channels: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reserved_cost_usd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) generation_cost_usd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) provenance: Option<GenerationProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error_code: Option<String>,
    pub(super) cancellation_requested: bool,
    pub(super) provider_cancel_confirmed: bool,
    pub(super) completed_after_cancel: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum GenerationStatus {
    Queued,
    Processing,
    Complete,
    Failed,
    Cancelled,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GenerationProvenance {
    pub(super) request_id: String,
    pub(super) prompt_hash: String,
    pub(super) generated_at: String,
    pub(super) model_version: String,
    pub(super) pricing_version: String,
    pub(super) terms_version: Option<String>,
    pub(super) synthid_expected: bool,
    pub(super) c2pa_expected: bool,
    pub(super) c2pa_status: String,
    pub(super) provider_billing_verified: bool,
}

#[derive(Serialize)]
struct UpstreamGenerationRequest<'a> {
    prompt: &'a str,
    duration_seconds: u16,
    instrumental: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
}

#[derive(Deserialize)]
struct UpstreamGenerationTask {
    #[serde(alias = "task_id", alias = "taskId")]
    id: String,
    status: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default, alias = "audioUrl")]
    audio_url: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

struct ProviderConfig {
    kind: ProviderKind,
    base_url: Url,
    endpoint_host: String,
    token: String,
    model: Option<String>,
}

struct PartnerProvider {
    config: ProviderConfig,
    client: Client,
    permits: Arc<Semaphore>,
}

enum ProviderImplementation {
    Partner(PartnerProvider),
    Lyria(LyriaProvider),
}

pub(crate) struct CreativeProvider {
    status: ProviderStatus,
    enabled: Option<ProviderImplementation>,
}

impl CreativeProvider {
    pub(crate) fn from_env(asset_root: PathBuf) -> Self {
        let enabled = env::var(ENABLE_ENV)
            .map(|value| matches!(value.as_str(), "1" | "true"))
            .unwrap_or(false);
        if !enabled {
            return Self::disabled(
                "offline",
                "Creative generation is disabled. Set MUSICA_CREATIVE_ENABLED=true to opt in.",
            );
        }

        let provider_value = env::var(PROVIDER_ENV).unwrap_or_else(|_| "async_partner".into());
        let kind = match ProviderKind::parse(&provider_value) {
            Ok(kind) => kind,
            Err(error) => return Self::disabled("misconfigured", error.to_string()),
        };

        let implementation = match kind {
            ProviderKind::Lyria3Pro => {
                LyriaProvider::from_env(asset_root).map(ProviderImplementation::Lyria)
            }
            ProviderKind::AsyncPartner | ProviderKind::SunoPartner => {
                Self::build_partner(kind).map(ProviderImplementation::Partner)
            }
        };

        match implementation {
            Ok(enabled) => {
                let (endpoint_host, model, unit_cost_usd, max_duration_seconds) = match &enabled {
                    ProviderImplementation::Partner(provider) => (
                        provider.config.endpoint_host.clone(),
                        provider.config.model.clone(),
                        None,
                        Some(PARTNER_MAX_DURATION_SECONDS),
                    ),
                    ProviderImplementation::Lyria(provider) => (
                        provider.endpoint_host().to_owned(),
                        Some(provider.model().to_owned()),
                        Some(provider.unit_cost_usd()),
                        Some(provider.max_duration_seconds()),
                    ),
                };
                Self {
                    status: ProviderStatus {
                        available: true,
                        provider: kind.label().into(),
                        endpoint_host: Some(endpoint_host),
                        reason: None,
                        model,
                        unit_cost_usd,
                        max_duration_seconds,
                    },
                    enabled: Some(enabled),
                }
            }
            Err(error) => Self::disabled(kind.label(), error.to_string()),
        }
    }

    fn disabled(provider: &str, reason: impl Into<String>) -> Self {
        Self {
            status: ProviderStatus {
                available: false,
                provider: provider.into(),
                endpoint_host: None,
                reason: Some(reason.into()),
                model: None,
                unit_cost_usd: None,
                max_duration_seconds: None,
            },
            enabled: None,
        }
    }

    fn build_partner(kind: ProviderKind) -> Result<PartnerProvider, ProviderError> {
        if matches!(kind, ProviderKind::Lyria3Pro) {
            return Err(ProviderError::Configuration(
                "invalid partner provider kind",
            ));
        }
        let base = env::var(BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        let base_url = validate_base_url(&base, kind)?;
        let endpoint_host = base_url
            .host_str()
            .ok_or(ProviderError::Configuration("API base URL has no host"))?
            .to_owned();

        let token = env::var(TOKEN_ENV)
            .map_err(|_| ProviderError::Configuration("MUSICA_CREATIVE_API_TOKEN is required"))?;
        validate_token(&token)?;

        let model = env::var(MODEL_ENV).ok().filter(|value| !value.is_empty());
        if let Some(model) = model.as_deref() {
            validate_model(model)?;
        }

        let client = Client::builder()
            .https_only(true)
            .redirect(Policy::none())
            .no_proxy()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .pool_idle_timeout(Duration::from_secs(30))
            .user_agent("Musica-VJ/0.1")
            .build()
            .map_err(|_| ProviderError::Configuration("could not initialize the HTTPS client"))?;

        Ok(PartnerProvider {
            config: ProviderConfig {
                kind,
                base_url,
                endpoint_host,
                token,
                model,
            },
            client,
            permits: Arc::new(Semaphore::new(2)),
        })
    }

    async fn generate(&self, request: GenerationRequest) -> Result<GenerationTask, ProviderError> {
        request.validate()?;
        match self.enabled.as_ref().ok_or(ProviderError::Unavailable)? {
            ProviderImplementation::Partner(provider) => {
                Self::generate_partner(provider, request).await
            }
            ProviderImplementation::Lyria(provider) => provider.generate(request).await,
        }
    }

    fn requires_native_cost_confirmation(&self) -> bool {
        matches!(
            self.enabled.as_ref(),
            Some(ProviderImplementation::Lyria(_))
        )
    }

    async fn generate_partner(
        provider: &PartnerProvider,
        request: GenerationRequest,
    ) -> Result<GenerationTask, ProviderError> {
        if request.duration_seconds > PARTNER_MAX_DURATION_SECONDS {
            return Err(ProviderError::Validation(
                "durationSeconds exceeds this provider's 120 second limit",
            ));
        }
        if !request.reference_assets.is_empty() {
            return Err(ProviderError::Validation(
                "reference assets are not supported by this provider contract",
            ));
        }
        let _permit = timeout(Duration::from_secs(2), provider.permits.acquire())
            .await
            .map_err(|_| ProviderError::Busy)?
            .map_err(|_| ProviderError::Unavailable)?;

        let body = UpstreamGenerationRequest {
            prompt: request.prompt.trim(),
            duration_seconds: request.duration_seconds,
            instrumental: request.instrumental,
            seed: request.seed,
            model: provider.config.model.as_deref(),
        };
        let endpoint = endpoint(&provider.config.base_url, "/v1/music/generations");
        let response = provider
            .client
            .post(endpoint)
            .bearer_auth(&provider.config.token)
            .json(&body)
            .send()
            .await
            .map_err(|_| ProviderError::Network)?;
        provider.parse_task(response, None).await
    }

    async fn generation_status(&self, task_id: &str) -> Result<GenerationTask, ProviderError> {
        validate_task_id(task_id)?;
        match self.enabled.as_ref().ok_or(ProviderError::Unavailable)? {
            ProviderImplementation::Partner(provider) => {
                Self::partner_generation_status(provider, task_id).await
            }
            ProviderImplementation::Lyria(provider) => provider.generation_status(task_id).await,
        }
    }

    async fn partner_generation_status(
        provider: &PartnerProvider,
        task_id: &str,
    ) -> Result<GenerationTask, ProviderError> {
        let _permit = timeout(Duration::from_secs(2), provider.permits.acquire())
            .await
            .map_err(|_| ProviderError::Busy)?
            .map_err(|_| ProviderError::Unavailable)?;

        let path = format!("/v1/music/generations/{task_id}");
        let response = provider
            .client
            .get(endpoint(&provider.config.base_url, &path))
            .bearer_auth(&provider.config.token)
            .send()
            .await
            .map_err(|_| ProviderError::Network)?;
        provider.parse_task(response, Some(task_id)).await
    }

    async fn download_audio(&self, task_id: &str) -> Result<Vec<u8>, ProviderError> {
        validate_task_id(task_id)?;
        match self.enabled.as_ref().ok_or(ProviderError::Unavailable)? {
            ProviderImplementation::Partner(provider) => {
                Self::partner_download_audio(provider, task_id).await
            }
            ProviderImplementation::Lyria(provider) => provider.download_audio(task_id).await,
        }
    }

    async fn partner_download_audio(
        provider: &PartnerProvider,
        task_id: &str,
    ) -> Result<Vec<u8>, ProviderError> {
        let task = Self::partner_generation_status(provider, task_id).await?;
        if !matches!(task.status, GenerationStatus::Complete) {
            return Err(ProviderError::Validation("generation is not complete"));
        }
        let audio_url = task.audio_url.ok_or(ProviderError::MalformedResponse)?;
        validate_audio_url(&audio_url, provider.config.kind)?;
        let _permit = timeout(Duration::from_secs(2), provider.permits.acquire())
            .await
            .map_err(|_| ProviderError::Busy)?
            .map_err(|_| ProviderError::Unavailable)?;
        let response = provider
            .client
            .get(audio_url)
            .bearer_auth(&provider.config.token)
            .send()
            .await
            .map_err(|_| ProviderError::Network)?;
        bounded_audio(response).await
    }

    async fn cancel(&self, task_id: &str) -> Result<GenerationTask, ProviderError> {
        validate_task_id(task_id)?;
        match self.enabled.as_ref().ok_or(ProviderError::Unavailable)? {
            ProviderImplementation::Partner(_) => Err(ProviderError::Validation(
                "cancellation is not supported by this provider contract",
            )),
            ProviderImplementation::Lyria(provider) => provider.cancel(task_id).await,
        }
    }
}

impl PartnerProvider {
    async fn parse_task(
        &self,
        response: HttpResponse,
        expected_task_id: Option<&str>,
    ) -> Result<GenerationTask, ProviderError> {
        let raw: UpstreamGenerationTask = bounded_json(response).await?;
        validate_task_id(&raw.id)?;
        if expected_task_id.is_some_and(|expected| expected != raw.id) {
            return Err(ProviderError::MalformedResponse);
        }

        let status = match raw.status.as_str() {
            "queued" | "pending" => GenerationStatus::Queued,
            "processing" | "running" => GenerationStatus::Processing,
            "complete" | "completed" | "succeeded" => GenerationStatus::Complete,
            "failed" | "error" | "cancelled" => GenerationStatus::Failed,
            _ => return Err(ProviderError::MalformedResponse),
        };

        if raw.title.as_ref().is_some_and(|title| {
            title.chars().count() > MAX_TITLE_CHARS || title.chars().any(char::is_control)
        }) {
            return Err(ProviderError::MalformedResponse);
        }
        if let Some(model) = raw.model.as_deref() {
            validate_model(model).map_err(|_| ProviderError::MalformedResponse)?;
        }
        if let Some(audio_url) = raw.audio_url.as_deref() {
            validate_audio_url(audio_url, self.config.kind)?;
        }
        if matches!(status, GenerationStatus::Complete) && raw.audio_url.is_none() {
            return Err(ProviderError::MalformedResponse);
        }

        Ok(GenerationTask {
            id: raw.id,
            status,
            title: raw.title,
            has_audio: raw.audio_url.is_some(),
            audio_url: raw.audio_url,
            provider: self.config.kind.label().into(),
            model: raw.model.or_else(|| self.config.model.clone()),
            audio_mime_type: None,
            lyrics: None,
            structure: None,
            actual_duration_seconds: None,
            sample_rate_hz: None,
            channels: None,
            reserved_cost_usd: None,
            generation_cost_usd: None,
            provenance: None,
            error_code: None,
            cancellation_requested: false,
            provider_cancel_confirmed: false,
            completed_after_cancel: false,
        })
    }
}

fn validate_base_url(value: &str, kind: ProviderKind) -> Result<Url, ProviderError> {
    let mut url = Url::parse(value)
        .map_err(|_| ProviderError::Configuration("MUSICA_CREATIVE_API_BASE is not a valid URL"))?;
    if url.scheme() != "https" {
        return Err(ProviderError::Configuration("creative API must use HTTPS"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(ProviderError::Configuration(
            "creative API URL must not contain credentials",
        ));
    }
    if url.port().is_some_and(|port| port != 443) {
        return Err(ProviderError::Configuration(
            "creative API must use HTTPS port 443",
        ));
    }
    if url.query().is_some() || url.fragment().is_some() || !matches!(url.path(), "" | "/") {
        return Err(ProviderError::Configuration(
            "creative API base URL must be an origin",
        ));
    }
    let host = url
        .host_str()
        .ok_or(ProviderError::Configuration("creative API URL has no host"))?;
    if !host_allowed(kind, host) {
        return Err(match kind {
            ProviderKind::SunoPartner => ProviderError::Configuration(
                "no verified official Suno partner host is embedded in this build",
            ),
            ProviderKind::AsyncPartner => {
                ProviderError::Configuration("creative API host is not allowlisted in this build")
            }
            ProviderKind::Lyria3Pro => ProviderError::Configuration(
                "Lyria uses a fixed provider endpoint and does not accept a base URL",
            ),
        });
    }
    url.set_path("/");
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

fn host_allowed(kind: ProviderKind, host: &str) -> bool {
    let built_in = kind == ProviderKind::AsyncPartner && host.eq_ignore_ascii_case(COGNITUM_HOST);
    built_in || build_allowed_hosts().any(|allowed| host.eq_ignore_ascii_case(allowed))
}

fn build_allowed_hosts() -> impl Iterator<Item = &'static str> {
    BUILD_ALLOWED_HOSTS
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|host| is_exact_dns_name(host))
}

fn is_exact_dns_name(host: &str) -> bool {
    !host.is_empty()
        && host.len() <= 253
        && !host.contains('*')
        && host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        })
}

fn endpoint(base: &Url, path: &str) -> Url {
    let mut endpoint = base.clone();
    endpoint.set_path(path);
    endpoint
}

fn validate_token(token: &str) -> Result<(), ProviderError> {
    if token.len() < 16 || token.len() > 4_096 || !token.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return Err(ProviderError::Configuration(
            "creative API token is invalid",
        ));
    }
    Ok(())
}

fn validate_model(model: &str) -> Result<(), ProviderError> {
    if model.is_empty()
        || model.len() > MAX_MODEL_BYTES
        || !model
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:/-".contains(&byte))
    {
        return Err(ProviderError::Configuration(
            "creative model identifier is invalid",
        ));
    }
    Ok(())
}

pub(super) fn validate_task_id(task_id: &str) -> Result<(), ProviderError> {
    if task_id.is_empty()
        || task_id.len() > MAX_TASK_ID_BYTES
        || !task_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(ProviderError::Validation("taskId is invalid"));
    }
    Ok(())
}

fn validate_audio_url(value: &str, kind: ProviderKind) -> Result<(), ProviderError> {
    if value.len() > MAX_AUDIO_URL_BYTES {
        return Err(ProviderError::MalformedResponse);
    }
    let url = Url::parse(value).map_err(|_| ProviderError::MalformedResponse)?;
    if url.scheme() != "https"
        || url.host_str().is_none_or(|host| !host_allowed(kind, host))
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some_and(|port| port != 443)
    {
        return Err(ProviderError::MalformedResponse);
    }
    Ok(())
}

async fn bounded_json<T: DeserializeOwned>(mut response: HttpResponse) -> Result<T, ProviderError> {
    if !response.status().is_success() {
        return Err(ProviderError::Service(response.status().as_u16()));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let media_type = content_type.split(';').next().unwrap_or("").trim();
    if media_type != "application/json" && !media_type.ends_with("+json") {
        return Err(ProviderError::MalformedResponse);
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        return Err(ProviderError::ResponseTooLarge);
    }

    let mut body = Vec::with_capacity(
        response
            .content_length()
            .unwrap_or(8 * 1024)
            .min(MAX_RESPONSE_BYTES as u64) as usize,
    );
    while let Some(chunk) = response.chunk().await.map_err(|_| ProviderError::Network)? {
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(ProviderError::ResponseTooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&body).map_err(|_| ProviderError::MalformedResponse)
}

async fn bounded_audio(mut response: HttpResponse) -> Result<Vec<u8>, ProviderError> {
    if !response.status().is_success() {
        return Err(ProviderError::Service(response.status().as_u16()));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let media_type = content_type.split(';').next().unwrap_or("").trim();
    if !media_type.starts_with("audio/") {
        return Err(ProviderError::UnsupportedMedia);
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_AUDIO_BYTES as u64)
    {
        return Err(ProviderError::ResponseTooLarge);
    }

    let mut body = Vec::with_capacity(
        response
            .content_length()
            .unwrap_or(256 * 1024)
            .min(MAX_AUDIO_BYTES as u64) as usize,
    );
    while let Some(chunk) = response.chunk().await.map_err(|_| ProviderError::Network)? {
        if body.len().saturating_add(chunk.len()) > MAX_AUDIO_BYTES {
            return Err(ProviderError::ResponseTooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    if !has_supported_audio_signature(&body) {
        return Err(ProviderError::UnsupportedMedia);
    }
    Ok(body)
}

pub(super) fn has_supported_audio_signature(bytes: &[u8]) -> bool {
    bytes.starts_with(b"ID3")
        || bytes.starts_with(b"fLaC")
        || bytes.starts_with(b"OggS")
        || (bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE")
        || (bytes.len() >= 8 && &bytes[4..8] == b"ftyp")
        || (bytes.len() >= 2 && bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0)
}

#[derive(Debug, Error)]
pub(super) enum ProviderError {
    #[error("Creative generation is unavailable")]
    Unavailable,
    #[error("Creative provider is busy; try again")]
    Busy,
    #[error("Invalid request: {0}")]
    Validation(&'static str),
    #[error("Provider configuration rejected: {0}")]
    Configuration(&'static str),
    #[error("Creative provider request failed")]
    Network,
    #[error("Creative provider returned HTTP {0}")]
    Service(u16),
    #[error("Creative provider response exceeded the size limit")]
    ResponseTooLarge,
    #[error("Creative provider returned an invalid response")]
    MalformedResponse,
    #[error("Creative provider returned an unsupported audio file")]
    UnsupportedMedia,
}

#[tauri::command]
pub(crate) fn creative_provider_status(state: State<'_, CreativeProvider>) -> ProviderStatus {
    state.status.clone()
}

#[tauri::command]
pub(crate) async fn creative_generate(
    app: AppHandle,
    state: State<'_, CreativeProvider>,
    request: GenerationRequest,
) -> Result<GenerationTask, String> {
    request.validate().map_err(|error| error.to_string())?;
    if state.requires_native_cost_confirmation() {
        let fingerprint =
            generation_request_fingerprint(&request).map_err(|error| error.to_string())?;
        let format = match request.output_format {
            AudioOutputFormat::Mp3 => "MP3",
            AudioOutputFormat::Wav => "WAV",
        };
        let approval = app
            .dialog()
            .message(format!(
                "Approve one Google Lyria 3 Pro request?\n\nMaximum charge: USD 0.08\nDuration: {} seconds\nFormat: {}\nMode: {}\nRequest fingerprint: {}\n\nMusica will make at most one paid POST and will not retry automatically.",
                request.duration_seconds,
                format,
                if request.instrumental { "instrumental" } else { "vocals" },
                &fingerprint[..16],
            ))
            .title("Approve paid music generation")
            .kind(MessageDialogKind::Warning)
            .buttons(MessageDialogButtons::OkCancel)
            .blocking_show();
        if !approval {
            return Err("Paid Lyria generation was not approved in the native dialog".into());
        }
    }
    state
        .generate(request)
        .await
        .map_err(|error| error.to_string())
}

fn generation_request_fingerprint(request: &GenerationRequest) -> Result<String, ProviderError> {
    let encoded = serde_json::to_vec(request)
        .map_err(|_| ProviderError::Validation("generation request could not be fingerprinted"))?;
    let digest = Sha256::digest(encoded);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    Ok(output)
}

#[tauri::command]
pub(crate) async fn creative_generation_status(
    state: State<'_, CreativeProvider>,
    task_id: String,
) -> Result<GenerationTask, String> {
    state
        .generation_status(&task_id)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub(crate) async fn creative_download_audio(
    state: State<'_, CreativeProvider>,
    task_id: String,
) -> Result<IpcResponse, String> {
    state
        .download_audio(&task_id)
        .await
        .map(IpcResponse::new)
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub(crate) async fn creative_cancel_generation(
    state: State<'_, CreativeProvider>,
    task_id: String,
) -> Result<GenerationTask, String> {
    state
        .cancel(&task_id)
        .await
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_request() -> GenerationRequest {
        GenerationRequest {
            prompt: "A crisp electro groove".into(),
            duration_seconds: 30,
            instrumental: true,
            seed: Some(7),
            language: None,
            bpm: None,
            lyrics: None,
            structure: Vec::new(),
            output_format: AudioOutputFormat::Mp3,
            reference_assets: Vec::new(),
            max_cost_usd: None,
            candidate_count: 1,
            max_attempts: 1,
            rights_declared: false,
            client_request_id: None,
        }
    }

    #[test]
    fn validates_generation_request_boundaries() {
        assert!(valid_request().validate().is_ok());

        let mut request = valid_request();
        request.prompt = "  ".into();
        assert!(request.validate().is_err());

        let mut request = valid_request();
        request.duration_seconds = 185;
        assert!(request.validate().is_err());

        let mut request = valid_request();
        request.prompt = "bad\u{0000}control".into();
        assert!(request.validate().is_err());

        let mut request = valid_request();
        request.prompt = "valid\nmultiline prompt".into();
        assert!(request.validate().is_ok());

        let mut request = valid_request();
        request.lyrics = Some("user supplied lyrics".into());
        assert!(request.validate().is_err());
        request.rights_declared = true;
        assert!(request.validate().is_ok());

        let mut request = valid_request();
        request.lyrics = Some("bad\u{0000}lyrics".into());
        request.rights_declared = true;
        assert!(request.validate().is_err());

        let mut request = valid_request();
        request.reference_assets = vec![ReferenceAsset {
            mime_type: String::new(),
            storage_uri: String::new(),
        }];
        assert!(request.validate().is_err());

        let mut request = valid_request();
        request.structure = vec![TimedSection {
            time_seconds: 30.0,
            section: "outro".into(),
        }];
        assert!(request.validate().is_err());
    }

    #[test]
    fn paid_request_fingerprint_binds_the_exact_request_without_plaintext() {
        let first = valid_request();
        let first_hash = generation_request_fingerprint(&first).expect("fingerprint request");
        let mut second = first.clone();
        second.duration_seconds += 1;
        let second_hash = generation_request_fingerprint(&second).expect("fingerprint request");
        assert_eq!(first_hash.len(), 64);
        assert_ne!(first_hash, second_hash);
        assert!(!first_hash.contains("electro"));
    }

    #[tokio::test]
    async fn disabled_provider_never_dispatches_generation() {
        let provider = CreativeProvider::disabled("offline", "test-disabled");
        assert!(!provider.status.available);
        assert!(provider.enabled.is_none());
        assert!(!provider.requires_native_cost_confirmation());
        assert!(matches!(
            provider.generate(valid_request()).await,
            Err(ProviderError::Unavailable)
        ));
    }

    #[test]
    fn task_ids_are_path_segment_safe() {
        assert!(validate_task_id("task_01-Ab").is_ok());
        assert!(validate_task_id("../secrets").is_err());
        assert!(validate_task_id("task?redirect=https://example.test").is_err());
        assert!(validate_task_id("").is_err());
    }

    #[test]
    fn endpoint_requires_https_origin_and_exact_host() {
        assert!(validate_base_url(DEFAULT_BASE_URL, ProviderKind::AsyncPartner).is_ok());
        assert!(validate_base_url("http://api.cognitum.one", ProviderKind::AsyncPartner).is_err());
        assert!(validate_base_url(
            "https://api.cognitum.one.evil.test",
            ProviderKind::AsyncPartner
        )
        .is_err());
        assert!(
            validate_base_url("https://api.cognitum.one/v1", ProviderKind::AsyncPartner).is_err()
        );
        assert!(validate_base_url(
            "https://user:pass@api.cognitum.one",
            ProviderKind::AsyncPartner
        )
        .is_err());
        assert!(
            validate_base_url("https://api.cognitum.one:8443", ProviderKind::AsyncPartner).is_err()
        );
    }

    #[test]
    fn generated_media_is_also_host_pinned() {
        assert!(validate_audio_url(
            "https://api.cognitum.one/v1/music/output/task_1.mp3?signature=test",
            ProviderKind::AsyncPartner,
        )
        .is_ok());
        assert!(
            validate_audio_url("https://127.0.0.1/internal", ProviderKind::AsyncPartner,).is_err()
        );
        assert!(validate_audio_url(
            "http://api.cognitum.one/output.mp3",
            ProviderKind::AsyncPartner,
        )
        .is_err());
    }

    #[test]
    fn recognizes_supported_audio_signatures() {
        assert!(has_supported_audio_signature(b"ID3\x04\x00\x00"));
        assert!(has_supported_audio_signature(
            b"RIFF\x00\x00\x00\x00WAVEfmt "
        ));
        assert!(has_supported_audio_signature(b"\x00\x00\x00\x18ftypM4A "));
        assert!(has_supported_audio_signature(b"\xff\xfb\x90\x64"));
        assert!(!has_supported_audio_signature(b"<html>not audio</html>"));
    }

    #[test]
    fn suno_requires_a_verified_build_time_partner_host() {
        if BUILD_ALLOWED_HOSTS.is_none() {
            assert!(validate_base_url("https://api.suno.com", ProviderKind::SunoPartner).is_err());
        }
    }

    #[test]
    fn build_allowlist_rejects_wildcards() {
        assert!(!is_exact_dns_name("*.example.com"));
        assert!(!is_exact_dns_name("example..com"));
        assert!(is_exact_dns_name("music-api.example.com"));
    }
}
