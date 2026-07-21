use std::{
    collections::HashMap,
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::Duration,
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use rand::{rngs::OsRng, RngCore};
use reqwest::{
    header::{HeaderName, HeaderValue},
    redirect::Policy,
    Client, Response as HttpResponse,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use tokio::sync::{RwLock, Semaphore};

use crate::creative_provider::{
    has_supported_audio_signature, validate_task_id, AudioOutputFormat, GenerationProvenance,
    GenerationRequest, GenerationStatus, GenerationTask, ProviderError,
};

const GEMINI_KEY_ENV: &str = "GEMINI_API_KEY";
const GCP_AUTH_ENV: &str = "MUSICA_GCP_AUTH";
const MAX_BUDGET_ENV: &str = "MUSICA_CREATIVE_MAX_GENERATION_USD";
const RETAIN_PROMPTS_ENV: &str = "MUSICA_CREATIVE_RETAIN_PROMPTS";
const REQUEST_TIMEOUT_ENV: &str = "MUSICA_CREATIVE_REQUEST_TIMEOUT_SECONDS";
const LYRIA_HOST: &str = "generativelanguage.googleapis.com";
const LYRIA_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta/interactions";
const LYRIA_MODEL: &str = "lyria-3-pro-preview";
const PRICING_VERSION: &str = "gemini-developer-api-2026-07-18";
const TERMS_VERSION_ENV: &str = "MUSICA_CREATIVE_TERMS_VERSION";
const UNIT_COST_MICRO_USD: u64 = 80_000;
const DEFAULT_MAX_BUDGET_MICRO_USD: u64 = 320_000;
const UI_MAX_DURATION_SECONDS: u16 = 180;
const HARD_MAX_DURATION_SECONDS: u16 = 184;
const MIN_DURATION_SECONDS: u16 = 31;
const MAX_RESPONSE_BYTES: usize = 96 * 1024 * 1024;
// 184 seconds of stereo 48 kHz 32-bit PCM is about 67.4 MiB. Keep a bounded
// margin above that so a valid provider WAV is not rejected before inspection.
const MAX_AUDIO_BYTES: usize = 72 * 1024 * 1024;
const MAX_TEXT_BYTES: usize = 1024 * 1024;
const MAX_PROVIDER_ID_BYTES: usize = 256;
const MAX_PAID_POST_ATTEMPTS: u8 = 1;
const DEFAULT_REQUEST_TIMEOUT_SECONDS: u64 = 600;
const MIN_REQUEST_TIMEOUT_SECONDS: u64 = 60;
const MAX_REQUEST_TIMEOUT_SECONDS: u64 = 900;

#[derive(Clone)]
pub(super) struct LyriaProvider {
    client: Client,
    auth: LyriaAuth,
    jobs: Arc<RwLock<JobRegistry>>,
    permits: Arc<Semaphore>,
    asset_root: Arc<PathBuf>,
    max_budget_micro_usd: u64,
    retain_prompts: bool,
    terms_version: Option<Arc<str>>,
}

#[derive(Clone)]
enum LyriaAuth {
    ApiKey(Arc<HeaderValue>),
    Gcloud,
}

#[derive(Default)]
struct JobRegistry {
    jobs: HashMap<String, JobRecord>,
    client_requests: HashMap<String, String>,
    reserved_micro_usd: u64,
    charged_micro_usd: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CostState {
    Reserved,
    Charged,
    Released,
}

struct JobRecord {
    task: GenerationTask,
    prompt_hash: String,
    asset_path: Option<PathBuf>,
    asset_sha256: Option<String>,
    abort_handle: Option<tauri::async_runtime::JoinHandle<()>>,
    cancellation_requested: bool,
    dispatched: bool,
    paid_post_attempts: u8,
    cost_state: CostState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CancellationDisposition {
    AlreadyTerminal,
    BeforeDispatch,
    AfterDispatch,
}

impl JobRecord {
    fn request_cancellation(&mut self) -> CancellationDisposition {
        if matches!(
            self.task.status,
            GenerationStatus::Complete | GenerationStatus::Failed | GenerationStatus::Cancelled
        ) {
            return CancellationDisposition::AlreadyTerminal;
        }
        self.cancellation_requested = true;
        self.task.cancellation_requested = true;
        if self.dispatched {
            self.task.error_code = Some("cancellation_requested_provider_unconfirmed".into());
            CancellationDisposition::AfterDispatch
        } else {
            self.task.status = GenerationStatus::Cancelled;
            if let Some(handle) = self.abort_handle.take() {
                handle.abort();
            }
            self.task.error_code = Some("cancelled_before_dispatch".into());
            self.task.provider_cancel_confirmed = true;
            CancellationDisposition::BeforeDispatch
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DispatchDecision {
    Started,
    NotStarted,
}

impl JobRegistry {
    fn existing_task(&self, client_request_id: &str) -> Option<GenerationTask> {
        self.client_requests
            .get(client_request_id)
            .and_then(|task_id| self.jobs.get(task_id))
            .map(|record| record.task.clone())
    }

    fn begin_paid_dispatch(&mut self, task_id: &str) -> DispatchDecision {
        let Some(record) = self.jobs.get_mut(task_id) else {
            return DispatchDecision::NotStarted;
        };
        if record.cancellation_requested
            || record.dispatched
            || record.paid_post_attempts >= MAX_PAID_POST_ATTEMPTS
            || record.task.status != GenerationStatus::Queued
        {
            return DispatchDecision::NotStarted;
        }

        record.dispatched = true;
        record.paid_post_attempts += 1;
        record.abort_handle = None;
        record.task.status = GenerationStatus::Processing;
        record.task.error_code = None;
        DispatchDecision::Started
    }

    fn can_reserve(&self, amount_micro_usd: u64, limit_micro_usd: u64) -> bool {
        self.reserved_micro_usd
            .checked_add(self.charged_micro_usd)
            .and_then(|committed| committed.checked_add(amount_micro_usd))
            .is_some_and(|committed| committed <= limit_micro_usd)
    }

    fn add_reservation(&mut self, amount_micro_usd: u64) {
        self.reserved_micro_usd = self
            .reserved_micro_usd
            .checked_add(amount_micro_usd)
            .expect("validated generation reservation overflowed");
    }

    fn release_reservation(&mut self, task_id: &str) {
        let should_release = self
            .jobs
            .get(task_id)
            .is_some_and(|record| record.cost_state == CostState::Reserved);
        if !should_release {
            return;
        }
        self.reserved_micro_usd = self.reserved_micro_usd.saturating_sub(UNIT_COST_MICRO_USD);
        if let Some(record) = self.jobs.get_mut(task_id) {
            record.cost_state = CostState::Released;
            record.task.reserved_cost_usd = None;
            record.task.generation_cost_usd = None;
        }
    }

    fn charge_reservation(&mut self, task_id: &str) {
        let should_charge = self
            .jobs
            .get(task_id)
            .is_some_and(|record| record.cost_state == CostState::Reserved);
        if !should_charge {
            return;
        }
        self.reserved_micro_usd = self.reserved_micro_usd.saturating_sub(UNIT_COST_MICRO_USD);
        self.charged_micro_usd = self
            .charged_micro_usd
            .checked_add(UNIT_COST_MICRO_USD)
            .expect("validated generation charge overflowed");
        if let Some(record) = self.jobs.get_mut(task_id) {
            record.cost_state = CostState::Charged;
            record.task.generation_cost_usd = Some(micro_to_usd(UNIT_COST_MICRO_USD));
        }
    }
}

#[derive(Serialize)]
struct LyriaInteractionRequest<'a> {
    model: &'static str,
    input: &'a str,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<LyriaResponseFormat>,
}

#[derive(Serialize)]
struct LyriaResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Deserialize)]
struct InteractionResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    steps: Vec<InteractionStep>,
}

#[derive(Deserialize)]
struct InteractionStep {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    content: Vec<InteractionContent>,
}

#[derive(Deserialize)]
struct InteractionContent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default, alias = "mimeType")]
    mime_type: Option<String>,
}

struct ParsedInteraction {
    provider_request_id: String,
    model: String,
    audio: Vec<u8>,
    audio_mime_type: String,
    lyrics: Option<String>,
    structure: Option<String>,
    inspection: AudioInspection,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AudioInspection {
    duration_seconds: f32,
    sample_rate_hz: u32,
    channels: u16,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StoredReceipt<'a> {
    schema_version: u16,
    provider: &'static str,
    model: &'a str,
    provider_request_id: &'a str,
    local_task_id: &'a str,
    submitted_prompt: Option<&'a str>,
    submitted_prompt_sha256: &'a str,
    output_sha256: &'a str,
    provider_response_sha256: &'a str,
    generated_at: &'a str,
    requested_duration_seconds: u16,
    requested_language: Option<&'a str>,
    detected_language: Option<&'a str>,
    requested_output_format: AudioOutputFormat,
    actual_audio: AudioInspection,
    reserved_cost_micro_usd: u64,
    generation_cost_micro_usd: u64,
    pricing_version: &'static str,
    terms_version: Option<&'a str>,
    rights_declared: bool,
    synthid_expected: bool,
    c2pa_expected: bool,
    c2pa_status: &'static str,
    provider_billing_verified: bool,
    input_asset_hashes: Vec<String>,
}

#[derive(Serialize)]
struct StoredFailureReceipt {
    schema_version: u16,
    provider: &'static str,
    model: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_request_id: Option<String>,
    local_task_id: String,
    submitted_prompt_sha256: String,
    failed_at: String,
    error_code: String,
    dispatched: bool,
    cancellation_requested: bool,
    reserved_cost_micro_usd: u64,
    generation_cost_micro_usd: Option<u64>,
    pricing_version: &'static str,
    charge_basis: &'static str,
    provider_billing_verified: bool,
}

struct AssetBundle {
    audio_path: PathBuf,
    output_sha256: String,
    generated_at: String,
}

struct AssetStoreRequest<'a> {
    root: &'a Path,
    task_id: &'a str,
    request: &'a GenerationRequest,
    prompt: &'a str,
    prompt_hash: &'a str,
    retain_prompt: bool,
    raw_response: &'a [u8],
    audio: &'a [u8],
    model: &'a str,
    provider_request_id: &'a str,
    inspection: AudioInspection,
    generated_at: &'a str,
    terms_version: Option<&'a str>,
}

#[derive(Debug)]
enum ProviderFailure {
    Authentication,
    ModelUnavailable,
    Quota,
    RateLimited,
    Safety,
    Service,
    Malformed,
    UnsupportedMedia,
    TooLarge,
    Storage,
    AmbiguousNetwork,
}

struct InteractionReadFailure {
    failure: ProviderFailure,
    provider_request_id: Option<String>,
}

impl ProviderFailure {
    fn code(&self) -> &'static str {
        match self {
            Self::Authentication => "authentication_failed",
            Self::ModelUnavailable => "model_unavailable",
            Self::Quota => "quota_exhausted",
            Self::RateLimited => "rate_limited",
            Self::Safety => "safety_rejected",
            Self::Service => "provider_service_error",
            Self::Malformed => "malformed_provider_response",
            Self::UnsupportedMedia => "unsupported_provider_media",
            Self::TooLarge => "provider_response_too_large",
            Self::Storage => "immutable_asset_storage_failed",
            Self::AmbiguousNetwork => "ambiguous_paid_request_outcome",
        }
    }

    fn potentially_billable(&self) -> bool {
        !matches!(
            self,
            Self::Authentication
                | Self::ModelUnavailable
                | Self::Quota
                | Self::RateLimited
                | Self::Safety
        )
    }
}

impl LyriaProvider {
    pub(super) fn from_env(asset_root: PathBuf) -> Result<Self, ProviderError> {
        let auth = load_auth_from_env()?;
        let max_budget_micro_usd = env::var(MAX_BUDGET_ENV)
            .ok()
            .map(|value| parse_usd_to_micro(&value))
            .transpose()?
            .unwrap_or(DEFAULT_MAX_BUDGET_MICRO_USD);
        if max_budget_micro_usd < UNIT_COST_MICRO_USD {
            return Err(ProviderError::Configuration(
                "MUSICA_CREATIVE_MAX_GENERATION_USD is below one Lyria request",
            ));
        }
        let retain_prompts = env::var(RETAIN_PROMPTS_ENV)
            .ok()
            .map(|value| parse_retain_prompts(&value))
            .transpose()?
            .unwrap_or(true);
        let request_timeout_seconds = env::var(REQUEST_TIMEOUT_ENV)
            .ok()
            .map(|value| parse_request_timeout(&value))
            .transpose()?
            .unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECONDS);
        let terms_version = env::var(TERMS_VERSION_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| {
                validate_terms_version(&value)?;
                Ok::<Arc<str>, ProviderError>(Arc::from(value))
            })
            .transpose()?;

        let client = Client::builder()
            .https_only(true)
            .redirect(Policy::none())
            .retry(reqwest::retry::never())
            .no_proxy()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(request_timeout_seconds))
            .pool_idle_timeout(Duration::from_secs(30))
            .user_agent("Musica-VJ/0.1 Lyria")
            .build()
            .map_err(|_| {
                ProviderError::Configuration("could not initialize the Lyria HTTPS client")
            })?;

        Ok(Self {
            client,
            auth,
            jobs: Arc::new(RwLock::new(JobRegistry::default())),
            permits: Arc::new(Semaphore::new(2)),
            asset_root: Arc::new(asset_root.join("generated")),
            max_budget_micro_usd,
            retain_prompts,
            terms_version,
        })
    }

    pub(super) const fn endpoint_host(&self) -> &'static str {
        LYRIA_HOST
    }

    pub(super) const fn model(&self) -> &'static str {
        LYRIA_MODEL
    }

    pub(super) const fn unit_cost_usd(&self) -> f64 {
        micro_to_usd(UNIT_COST_MICRO_USD)
    }

    pub(super) const fn max_duration_seconds(&self) -> u16 {
        UI_MAX_DURATION_SECONDS
    }

    async fn auth_header(&self) -> Result<(HeaderName, HeaderValue), ProviderFailure> {
        match &self.auth {
            LyriaAuth::ApiKey(value) => Ok((
                HeaderName::from_static("x-goog-api-key"),
                value.as_ref().clone(),
            )),
            LyriaAuth::Gcloud => {
                let token = tauri::async_runtime::spawn_blocking(gcloud_access_token)
                    .await
                    .map_err(|_| ProviderFailure::Authentication)??;
                let mut value = HeaderValue::from_str(&format!("Bearer {token}"))
                    .map_err(|_| ProviderFailure::Authentication)?;
                value.set_sensitive(true);
                Ok((HeaderName::from_static("authorization"), value))
            }
        }
    }

    pub(super) async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationTask, ProviderError> {
        request.validate()?;
        self.validate_request(&request)?;
        let client_request_id = request
            .client_request_id
            .clone()
            .unwrap_or_else(new_task_id);
        validate_client_request_id(&client_request_id)?;

        {
            let registry = self.jobs.read().await;
            if let Some(task) = registry.existing_task(&client_request_id) {
                return Ok(task);
            }
        }

        let task_id = new_task_id();
        let compiled_prompt = compile_prompt(&request);
        let prompt_hash = sha256_hex(compiled_prompt.as_bytes());
        let task = GenerationTask {
            id: task_id.clone(),
            status: GenerationStatus::Queued,
            title: Some("Lyria 3 Pro generation".into()),
            audio_url: None,
            provider: "lyria_3_pro".into(),
            model: Some(LYRIA_MODEL.into()),
            has_audio: false,
            audio_mime_type: None,
            lyrics: None,
            structure: None,
            actual_duration_seconds: None,
            sample_rate_hz: None,
            channels: None,
            reserved_cost_usd: Some(micro_to_usd(UNIT_COST_MICRO_USD)),
            generation_cost_usd: None,
            provenance: None,
            error_code: None,
            cancellation_requested: false,
            provider_cancel_confirmed: false,
            completed_after_cancel: false,
        };

        {
            let mut registry = self.jobs.write().await;
            if let Some(existing) = registry.existing_task(&client_request_id) {
                return Ok(existing);
            }
            if !registry.can_reserve(UNIT_COST_MICRO_USD, self.max_budget_micro_usd) {
                return Err(ProviderError::Validation(
                    "configured Lyria generation budget is exhausted",
                ));
            }
            registry.add_reservation(UNIT_COST_MICRO_USD);
            registry
                .client_requests
                .insert(client_request_id, task_id.clone());
            registry.jobs.insert(
                task_id.clone(),
                JobRecord {
                    task: task.clone(),
                    prompt_hash: prompt_hash.clone(),
                    asset_path: None,
                    asset_sha256: None,
                    abort_handle: None,
                    cancellation_requested: false,
                    dispatched: false,
                    paid_post_attempts: 0,
                    cost_state: CostState::Reserved,
                },
            );
        }

        let worker = self.clone();
        let worker_task_id = task_id.clone();
        let handle = tauri::async_runtime::spawn(async move {
            worker
                .execute(worker_task_id, request, compiled_prompt, prompt_hash)
                .await;
        });
        if let Some(record) = self.jobs.write().await.jobs.get_mut(&task_id) {
            if !record.dispatched && record.task.status == GenerationStatus::Queued {
                record.abort_handle = Some(handle);
            }
        }
        Ok(task)
    }

    pub(super) async fn generation_status(
        &self,
        task_id: &str,
    ) -> Result<GenerationTask, ProviderError> {
        validate_task_id(task_id)?;
        self.jobs
            .read()
            .await
            .jobs
            .get(task_id)
            .map(|record| record.task.clone())
            .ok_or(ProviderError::Validation("generation task was not found"))
    }

    pub(super) async fn download_audio(&self, task_id: &str) -> Result<Vec<u8>, ProviderError> {
        validate_task_id(task_id)?;
        let (path, expected_hash) = {
            let registry = self.jobs.read().await;
            let record = registry
                .jobs
                .get(task_id)
                .ok_or(ProviderError::Validation("generation task was not found"))?;
            if record.task.status != GenerationStatus::Complete || !record.task.has_audio {
                return Err(ProviderError::Validation("generation is not complete"));
            }
            (
                record
                    .asset_path
                    .clone()
                    .ok_or(ProviderError::MalformedResponse)?,
                record
                    .asset_sha256
                    .clone()
                    .ok_or(ProviderError::MalformedResponse)?,
            )
        };
        let bytes = tauri::async_runtime::spawn_blocking(move || fs::read(path))
            .await
            .map_err(|_| ProviderError::Network)?
            .map_err(|_| ProviderError::Network)?;
        if bytes.len() > MAX_AUDIO_BYTES
            || !has_supported_audio_signature(&bytes)
            || sha256_hex(&bytes) != expected_hash
        {
            return Err(ProviderError::UnsupportedMedia);
        }
        Ok(bytes)
    }

    pub(super) async fn cancel(&self, task_id: &str) -> Result<GenerationTask, ProviderError> {
        validate_task_id(task_id)?;
        let mut registry = self.jobs.write().await;
        let disposition = {
            let record = registry
                .jobs
                .get_mut(task_id)
                .ok_or(ProviderError::Validation("generation task was not found"))?;
            let disposition = record.request_cancellation();
            if disposition == CancellationDisposition::AlreadyTerminal {
                return Ok(record.task.clone());
            }
            disposition
        };
        if disposition == CancellationDisposition::BeforeDispatch {
            registry.release_reservation(task_id);
        }
        registry
            .jobs
            .get(task_id)
            .map(|record| record.task.clone())
            .ok_or(ProviderError::Validation("generation task was not found"))
    }

    fn validate_request(&self, request: &GenerationRequest) -> Result<(), ProviderError> {
        if !(MIN_DURATION_SECONDS..=HARD_MAX_DURATION_SECONDS).contains(&request.duration_seconds) {
            return Err(ProviderError::Validation(
                "Lyria 3 Pro durationSeconds must be between 31 and 184",
            ));
        }
        if request.seed.is_some() {
            return Err(ProviderError::Validation(
                "Lyria 3 Pro does not document deterministic seed control",
            ));
        }
        if !request.reference_assets.is_empty() {
            return Err(ProviderError::Validation(
                "reference images are reserved for the V2 capability detected contract; PDFs are unsupported",
            ));
        }
        if request.max_attempts != 1 {
            return Err(ProviderError::Validation(
                "Lyria paid generation permits one automatic attempt because create idempotency is undocumented",
            ));
        }
        let budget = request.max_cost_usd.ok_or(ProviderError::Validation(
            "maxCostUsd is required for paid generation",
        ))?;
        let budget_micro = usd_f64_to_micro(budget)?;
        if budget_micro < UNIT_COST_MICRO_USD {
            return Err(ProviderError::Validation(
                "maxCostUsd does not cover the Lyria 3 Pro reservation",
            ));
        }
        if budget_micro > self.max_budget_micro_usd {
            return Err(ProviderError::Validation(
                "maxCostUsd exceeds the configured installation limit",
            ));
        }
        if let Some(language) = request.language.as_deref() {
            validate_language(language)?;
        }
        Ok(())
    }

    async fn execute(
        &self,
        task_id: String,
        request: GenerationRequest,
        compiled_prompt: String,
        prompt_hash: String,
    ) {
        let _permit = match self.permits.acquire().await {
            Ok(permit) => permit,
            Err(_) => {
                self.fail(&task_id, ProviderFailure::Service, false, None)
                    .await;
                return;
            }
        };
        if self.jobs.write().await.begin_paid_dispatch(&task_id) != DispatchDecision::Started {
            return;
        }

        let response_format = (request.output_format == AudioOutputFormat::Wav)
            .then_some(LyriaResponseFormat { kind: "audio" });
        let body = LyriaInteractionRequest {
            model: LYRIA_MODEL,
            input: &compiled_prompt,
            store: false,
            response_format,
        };
        let (auth_header, auth_value) = match self.auth_header().await {
            Ok(header) => header,
            Err(_) => {
                self.fail(&task_id, ProviderFailure::Authentication, false, None)
                    .await;
                return;
            }
        };
        let response = match self
            .client
            .post(LYRIA_ENDPOINT)
            .header(auth_header, auth_value)
            .json(&body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(_) => {
                self.fail(&task_id, ProviderFailure::AmbiguousNetwork, true, None)
                    .await;
                return;
            }
        };

        let (raw_response, provider_request_header) =
            match read_interaction_response(response).await {
                Ok(result) => result,
                Err(error) => {
                    let potentially_billable = error.failure.potentially_billable();
                    self.fail(
                        &task_id,
                        error.failure,
                        potentially_billable,
                        error.provider_request_id,
                    )
                    .await;
                    return;
                }
            };
        let parsed = match parse_interaction(
            &raw_response,
            provider_request_header.as_deref(),
            request.output_format,
        ) {
            Ok(parsed) => parsed,
            Err(error) => {
                self.fail(&task_id, error, true, provider_request_header)
                    .await;
                return;
            }
        };
        let ParsedInteraction {
            provider_request_id,
            model,
            audio,
            audio_mime_type,
            lyrics,
            structure,
            inspection,
        } = parsed;
        let output_is_short =
            output_shorter_than_requested(inspection.duration_seconds, request.duration_seconds);
        let generated_at = now_rfc3339();
        let terms_version = self.terms_version.as_deref().map(str::to_owned);
        let asset_root = self.asset_root.as_ref().clone();
        let retain_prompts = self.retain_prompts;
        let task_id_for_store = task_id.clone();
        let prompt_for_store = compiled_prompt;
        let hash_for_store = prompt_hash.clone();
        let request_for_store = request;
        let raw_for_store = raw_response;
        let parsed_audio = audio;
        let parsed_model = model.clone();
        let provider_request_id_for_store = provider_request_id.clone();
        let generated_at_for_store = generated_at.clone();
        let terms_for_store = terms_version.clone();
        let stored = tauri::async_runtime::spawn_blocking(move || {
            store_asset_bundle(AssetStoreRequest {
                root: &asset_root,
                task_id: &task_id_for_store,
                request: &request_for_store,
                prompt: &prompt_for_store,
                prompt_hash: &hash_for_store,
                retain_prompt: retain_prompts,
                raw_response: &raw_for_store,
                audio: &parsed_audio,
                model: &parsed_model,
                provider_request_id: &provider_request_id_for_store,
                inspection,
                generated_at: &generated_at_for_store,
                terms_version: terms_for_store.as_deref(),
            })
        })
        .await;
        let asset = match stored {
            Ok(Ok(asset)) => asset,
            _ => {
                self.fail(
                    &task_id,
                    ProviderFailure::Storage,
                    true,
                    Some(provider_request_id),
                )
                .await;
                return;
            }
        };

        let provenance = GenerationProvenance {
            request_id: provider_request_id,
            prompt_hash,
            generated_at: asset.generated_at,
            model_version: model.clone(),
            pricing_version: PRICING_VERSION.into(),
            terms_version,
            synthid_expected: true,
            c2pa_expected: true,
            c2pa_status: "preserved_unverified".into(),
            provider_billing_verified: false,
        };
        let mut registry = self.jobs.write().await;
        registry.charge_reservation(&task_id);
        if let Some(record) = registry.jobs.get_mut(&task_id) {
            let completed_after_cancel = record.cancellation_requested;
            record.asset_path = Some(asset.audio_path);
            record.asset_sha256 = Some(asset.output_sha256);
            record.abort_handle = None;
            record.task.status = GenerationStatus::Complete;
            record.task.has_audio = true;
            record.task.audio_mime_type = Some(audio_mime_type);
            record.task.lyrics = lyrics;
            record.task.structure = structure;
            record.task.actual_duration_seconds = Some(inspection.duration_seconds);
            record.task.sample_rate_hz = Some(inspection.sample_rate_hz);
            record.task.channels = Some(inspection.channels);
            record.task.provenance = Some(provenance);
            record.task.completed_after_cancel = completed_after_cancel;
            record.task.error_code = if completed_after_cancel {
                Some("completed_after_cancel".into())
            } else if output_is_short {
                Some("output_shorter_than_requested".into())
            } else {
                None
            };
        }
    }

    async fn fail(
        &self,
        task_id: &str,
        failure: ProviderFailure,
        potentially_billable: bool,
        provider_request_id: Option<String>,
    ) {
        let code = failure.code();
        let mut registry = self.jobs.write().await;
        if potentially_billable {
            registry.charge_reservation(task_id);
        } else {
            registry.release_reservation(task_id);
        }
        let receipt = if let Some(record) = registry.jobs.get_mut(task_id) {
            record.abort_handle = None;
            Some(StoredFailureReceipt {
                schema_version: 1,
                provider: "google_gemini",
                model: LYRIA_MODEL,
                provider_request_id: provider_request_id.filter(|id| valid_provider_id(id)),
                local_task_id: task_id.to_owned(),
                submitted_prompt_sha256: record.prompt_hash.clone(),
                failed_at: now_rfc3339(),
                error_code: code.into(),
                dispatched: record.dispatched,
                cancellation_requested: record.cancellation_requested,
                reserved_cost_micro_usd: UNIT_COST_MICRO_USD,
                generation_cost_micro_usd: (record.cost_state == CostState::Charged)
                    .then_some(UNIT_COST_MICRO_USD),
                pricing_version: PRICING_VERSION,
                charge_basis: if record.cost_state == CostState::Charged {
                    "conservative_post_dispatch_potential"
                } else {
                    "released_nonbillable_error"
                },
                provider_billing_verified: false,
            })
        } else {
            None
        };
        drop(registry);

        if let Some(receipt) = receipt {
            let root = self.asset_root.as_ref().clone();
            let receipt_task_id = task_id.to_owned();
            let stored = tauri::async_runtime::spawn_blocking(move || {
                store_failure_receipt(&root, &receipt_task_id, &receipt)
            })
            .await;
            let receipt_stored = matches!(stored, Ok(Ok(())));
            if let Some(record) = self.jobs.write().await.jobs.get_mut(task_id) {
                publish_failure(record, code, receipt_stored);
            }
        }
    }
}

fn publish_failure(record: &mut JobRecord, code: &str, receipt_stored: bool) {
    if record.task.status != GenerationStatus::Cancelled {
        record.task.status = GenerationStatus::Failed;
    }
    record.task.error_code = Some(if receipt_stored {
        code.into()
    } else {
        format!("{code}_failure_receipt_storage_failed")
    });
}

async fn read_interaction_response(
    mut response: HttpResponse,
) -> Result<(Vec<u8>, Option<String>), InteractionReadFailure> {
    let status = response.status();
    let request_id = response
        .headers()
        .get("x-goog-request-id")
        .or_else(|| response.headers().get("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .filter(|value| valid_provider_id(value))
        .map(str::to_owned);
    let response_is_json = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|content_type| {
            let media_type = content_type.split(';').next().unwrap_or("").trim();
            media_type == "application/json" || media_type.ends_with("+json")
        })
        .unwrap_or(false);
    let limit = if status.is_success() {
        MAX_RESPONSE_BYTES
    } else {
        512 * 1024
    };
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(InteractionReadFailure {
            failure: ProviderFailure::TooLarge,
            provider_request_id: request_id,
        });
    }
    let mut body = Vec::with_capacity(
        response
            .content_length()
            .unwrap_or(16 * 1024)
            .min(limit as u64) as usize,
    );
    while let Some(chunk) = response.chunk().await.map_err(|_| InteractionReadFailure {
        failure: ProviderFailure::AmbiguousNetwork,
        provider_request_id: request_id.clone(),
    })? {
        if body.len().saturating_add(chunk.len()) > limit {
            return Err(InteractionReadFailure {
                failure: ProviderFailure::TooLarge,
                provider_request_id: request_id,
            });
        }
        body.extend_from_slice(&chunk);
    }
    if !status.is_success() {
        return Err(InteractionReadFailure {
            failure: classify_http_error(status.as_u16(), &body),
            provider_request_id: request_id,
        });
    }
    if !response_is_json {
        return Err(InteractionReadFailure {
            failure: ProviderFailure::Malformed,
            provider_request_id: request_id,
        });
    }
    Ok((body, request_id))
}

fn classify_http_error(status: u16, body: &[u8]) -> ProviderFailure {
    let provider_signal = serde_json::from_slice::<serde_json::Value>(body)
        .map(|value| value.to_string())
        .unwrap_or_default()
        .to_ascii_uppercase();
    let safety = provider_signal.contains("SAFETY")
        || provider_signal.contains("BLOCKED")
        || provider_signal.contains("PROHIBITED_CONTENT");
    match status {
        400 | 403 if safety => ProviderFailure::Safety,
        401 | 403 => ProviderFailure::Authentication,
        404 | 410 => ProviderFailure::ModelUnavailable,
        429 if provider_signal.contains("QUOTA") => ProviderFailure::Quota,
        429 => ProviderFailure::RateLimited,
        500..=599 => ProviderFailure::Service,
        _ => ProviderFailure::Malformed,
    }
}

fn parse_interaction(
    raw: &[u8],
    request_header: Option<&str>,
    requested_format: AudioOutputFormat,
) -> Result<ParsedInteraction, ProviderFailure> {
    let response: InteractionResponse =
        serde_json::from_slice(raw).map_err(|_| ProviderFailure::Malformed)?;
    let response_id = response
        .id
        .as_deref()
        .filter(|value| valid_provider_id(value))
        .or(request_header)
        .ok_or(ProviderFailure::Malformed)?
        .to_owned();
    let model = response.model.unwrap_or_else(|| LYRIA_MODEL.into());
    if model != LYRIA_MODEL || !valid_provider_id(&model) {
        return Err(ProviderFailure::Malformed);
    }

    let mut audio_blocks = Vec::new();
    let mut text_blocks = Vec::new();
    let mut text_bytes = 0_usize;
    for step in response.steps {
        if step.kind != "model_output" {
            continue;
        }
        for content in step.content {
            match content.kind.as_str() {
                "audio" => {
                    let data = content.data.ok_or(ProviderFailure::Malformed)?;
                    let max_encoded = MAX_AUDIO_BYTES.saturating_mul(4) / 3 + 8;
                    if data.len() > max_encoded {
                        return Err(ProviderFailure::TooLarge);
                    }
                    let bytes = BASE64
                        .decode(data.as_bytes())
                        .map_err(|_| ProviderFailure::Malformed)?;
                    if bytes.len() > MAX_AUDIO_BYTES {
                        return Err(ProviderFailure::TooLarge);
                    }
                    audio_blocks.push((bytes, content.mime_type));
                }
                "text" => {
                    let text = content.text.ok_or(ProviderFailure::Malformed)?;
                    text_bytes = text_bytes.saturating_add(text.len());
                    if text_bytes > MAX_TEXT_BYTES {
                        return Err(ProviderFailure::TooLarge);
                    }
                    text_blocks.push(text);
                }
                _ => {}
            }
        }
    }
    if audio_blocks.len() != 1 {
        return Err(ProviderFailure::Malformed);
    }
    let (audio, declared_mime) = audio_blocks.pop().ok_or(ProviderFailure::Malformed)?;
    if !has_supported_audio_signature(&audio) {
        return Err(ProviderFailure::UnsupportedMedia);
    }
    let detected_mime = detect_audio_mime(&audio).ok_or(ProviderFailure::UnsupportedMedia)?;
    if declared_mime
        .as_deref()
        .is_some_and(|mime| normalize_audio_mime(mime) != Some(detected_mime))
    {
        return Err(ProviderFailure::UnsupportedMedia);
    }
    let expected_mime = match requested_format {
        AudioOutputFormat::Mp3 => "audio/mpeg",
        AudioOutputFormat::Wav => "audio/wav",
    };
    if detected_mime != expected_mime {
        return Err(ProviderFailure::UnsupportedMedia);
    }
    let inspection = inspect_audio(&audio, detected_mime)?;
    validate_audio_inspection(inspection)?;
    let (lyrics, structure) = classify_text_blocks(text_blocks);
    Ok(ParsedInteraction {
        provider_request_id: response_id,
        model,
        audio,
        audio_mime_type: detected_mime.into(),
        lyrics,
        structure,
        inspection,
    })
}

fn validate_audio_inspection(inspection: AudioInspection) -> Result<(), ProviderFailure> {
    if inspection.channels != 2
        || inspection.duration_seconds <= 0.0
        || inspection.duration_seconds > f32::from(HARD_MAX_DURATION_SECONDS)
    {
        return Err(ProviderFailure::UnsupportedMedia);
    }
    Ok(())
}

fn output_shorter_than_requested(actual_seconds: f32, requested_seconds: u16) -> bool {
    actual_seconds < (f32::from(requested_seconds) * 0.75).max(30.0)
}

fn classify_text_blocks(blocks: Vec<String>) -> (Option<String>, Option<String>) {
    let mut lyrics = Vec::new();
    let mut structure = Vec::new();
    for block in blocks {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            continue;
        }
        if serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
            || trimmed.to_ascii_lowercase().starts_with("structure:")
        {
            structure.push(trimmed.to_owned());
        } else {
            lyrics.push(trimmed.to_owned());
        }
    }
    let join = |items: Vec<String>| (!items.is_empty()).then(|| items.join("\n\n"));
    (join(lyrics), join(structure))
}

fn inspect_audio(bytes: &[u8], mime: &str) -> Result<AudioInspection, ProviderFailure> {
    match mime {
        "audio/wav" => inspect_wav(bytes),
        "audio/mpeg" => inspect_mp3(bytes),
        _ => Err(ProviderFailure::UnsupportedMedia),
    }
}

fn inspect_wav(bytes: &[u8]) -> Result<AudioInspection, ProviderFailure> {
    if bytes.len() < 12 || !bytes.starts_with(b"RIFF") || &bytes[8..12] != b"WAVE" {
        return Err(ProviderFailure::UnsupportedMedia);
    }
    let riff_size = u32::from_le_bytes(
        bytes[4..8]
            .try_into()
            .map_err(|_| ProviderFailure::UnsupportedMedia)?,
    ) as usize;
    if riff_size.checked_add(8) != Some(bytes.len()) {
        return Err(ProviderFailure::UnsupportedMedia);
    }
    let mut offset = 12_usize;
    let mut format = None;
    let mut data_length = None;
    while offset.saturating_add(8) <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let length = u32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| ProviderFailure::UnsupportedMedia)?,
        ) as usize;
        let start = offset + 8;
        let end = start
            .checked_add(length)
            .ok_or(ProviderFailure::UnsupportedMedia)?;
        if end > bytes.len() {
            return Err(ProviderFailure::UnsupportedMedia);
        }
        if chunk_id == b"fmt " && length >= 16 {
            if format.is_some() {
                return Err(ProviderFailure::UnsupportedMedia);
            }
            let declared_audio_format = u16::from_le_bytes([bytes[start], bytes[start + 1]]);
            let channels = u16::from_le_bytes([bytes[start + 2], bytes[start + 3]]);
            let sample_rate = u32::from_le_bytes(
                bytes[start + 4..start + 8]
                    .try_into()
                    .map_err(|_| ProviderFailure::UnsupportedMedia)?,
            );
            let byte_rate = u32::from_le_bytes(
                bytes[start + 8..start + 12]
                    .try_into()
                    .map_err(|_| ProviderFailure::UnsupportedMedia)?,
            );
            let block_align = u16::from_le_bytes([bytes[start + 12], bytes[start + 13]]);
            let bits_per_sample = u16::from_le_bytes([bytes[start + 14], bytes[start + 15]]);
            let audio_format = if declared_audio_format == 0xfffe {
                if length < 40 || u16::from_le_bytes([bytes[start + 16], bytes[start + 17]]) < 22 {
                    return Err(ProviderFailure::UnsupportedMedia);
                }
                let subformat = &bytes[start + 24..start + 40];
                const PCM_GUID: [u8; 16] =
                    [1, 0, 0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113];
                const FLOAT_GUID: [u8; 16] =
                    [3, 0, 0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113];
                if subformat == PCM_GUID {
                    1
                } else if subformat == FLOAT_GUID {
                    3
                } else {
                    return Err(ProviderFailure::UnsupportedMedia);
                }
            } else {
                declared_audio_format
            };
            let valid_sample_format = matches!(
                (audio_format, bits_per_sample),
                (1, 8 | 16 | 24 | 32) | (3, 32 | 64)
            );
            let expected_block_align = u32::from(channels)
                .checked_mul(u32::from(bits_per_sample))
                .filter(|bits| bits % 8 == 0)
                .map(|bits| bits / 8);
            let expected_byte_rate =
                expected_block_align.and_then(|align| sample_rate.checked_mul(align));
            if !valid_sample_format
                || channels == 0
                || sample_rate == 0
                || !(8_000..=384_000).contains(&sample_rate)
                || block_align == 0
                || expected_block_align != Some(u32::from(block_align))
                || expected_byte_rate != Some(byte_rate)
            {
                return Err(ProviderFailure::UnsupportedMedia);
            }
            format = Some((channels, sample_rate, byte_rate, block_align));
        } else if chunk_id == b"data" {
            if data_length.is_some() {
                return Err(ProviderFailure::UnsupportedMedia);
            }
            data_length = Some(length);
        }
        offset = end
            .checked_add(length & 1)
            .filter(|next| *next <= bytes.len())
            .ok_or(ProviderFailure::UnsupportedMedia)?;
    }
    if offset != bytes.len() {
        return Err(ProviderFailure::UnsupportedMedia);
    }
    let (channels, sample_rate_hz, byte_rate, block_align) =
        format.ok_or(ProviderFailure::UnsupportedMedia)?;
    let data_length = data_length.ok_or(ProviderFailure::UnsupportedMedia)?;
    if data_length == 0 || !data_length.is_multiple_of(usize::from(block_align)) {
        return Err(ProviderFailure::UnsupportedMedia);
    }
    let duration_seconds = data_length as f32 / byte_rate as f32;
    Ok(AudioInspection {
        duration_seconds,
        sample_rate_hz,
        channels,
    })
}

fn inspect_mp3(bytes: &[u8]) -> Result<AudioInspection, ProviderFailure> {
    let mut offset = id3v2_size(bytes).unwrap_or(0);
    let first = parse_mp3_header(
        bytes
            .get(offset..offset + 4)
            .ok_or(ProviderFailure::UnsupportedMedia)?,
    )
    .ok_or(ProviderFailure::UnsupportedMedia)?;
    let sample_rate_hz = first.sample_rate_hz;
    let channels = first.channels;
    let mut frames = 0_u64;
    let mut total_samples = 0_u64;
    while offset + 4 <= bytes.len() {
        let Some(header) = parse_mp3_header(&bytes[offset..offset + 4]) else {
            break;
        };
        if header.sample_rate_hz != sample_rate_hz || header.channels != channels {
            return Err(ProviderFailure::UnsupportedMedia);
        }
        let next = offset
            .checked_add(header.frame_length)
            .ok_or(ProviderFailure::UnsupportedMedia)?;
        if next > bytes.len() {
            return Err(ProviderFailure::UnsupportedMedia);
        }
        frames += 1;
        total_samples += u64::from(header.samples_per_frame);
        offset = next;
    }
    if frames < 2 || !valid_mp3_trailer(&bytes[offset..]) {
        return Err(ProviderFailure::UnsupportedMedia);
    }
    Ok(AudioInspection {
        duration_seconds: total_samples as f32 / sample_rate_hz as f32,
        sample_rate_hz,
        channels,
    })
}

struct Mp3Header {
    sample_rate_hz: u32,
    channels: u16,
    frame_length: usize,
    samples_per_frame: u16,
}

fn parse_mp3_header(bytes: &[u8]) -> Option<Mp3Header> {
    if bytes.len() < 4 || bytes[0] != 0xff || bytes[1] & 0xe0 != 0xe0 {
        return None;
    }
    let version_bits = (bytes[1] >> 3) & 0x03;
    let layer_bits = (bytes[1] >> 1) & 0x03;
    if version_bits == 1 || layer_bits != 1 {
        return None;
    }
    let bitrate_index = (bytes[2] >> 4) & 0x0f;
    let sample_index = (bytes[2] >> 2) & 0x03;
    if bitrate_index == 0 || bitrate_index == 15 || sample_index == 3 {
        return None;
    }
    const MPEG1_LAYER3: [u16; 16] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
    ];
    const MPEG2_LAYER3: [u16; 16] = [
        0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
    ];
    let mpeg1 = version_bits == 3;
    let bitrate_kbps = if mpeg1 {
        MPEG1_LAYER3[bitrate_index as usize]
    } else {
        MPEG2_LAYER3[bitrate_index as usize]
    };
    let base_rate = [44_100_u32, 48_000, 32_000][sample_index as usize];
    let sample_rate_hz = match version_bits {
        3 => base_rate,
        2 => base_rate / 2,
        0 => base_rate / 4,
        _ => return None,
    };
    let padding = usize::from((bytes[2] >> 1) & 1);
    let coefficient = if mpeg1 { 144_000 } else { 72_000 };
    let frame_length = coefficient * usize::from(bitrate_kbps) / sample_rate_hz as usize + padding;
    if frame_length < 24 {
        return None;
    }
    Some(Mp3Header {
        sample_rate_hz,
        channels: if (bytes[3] >> 6) == 3 { 1 } else { 2 },
        frame_length,
        samples_per_frame: if mpeg1 { 1152 } else { 576 },
    })
}

fn id3v2_size(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 10 || !bytes.starts_with(b"ID3") {
        return None;
    }
    if bytes[6..10].iter().any(|byte| byte & 0x80 != 0) {
        return None;
    }
    let size = bytes[6..10]
        .iter()
        .fold(0_usize, |value, byte| (value << 7) | usize::from(*byte));
    let footer_size = usize::from(bytes[5] & 0x10 != 0) * 10;
    10_usize
        .checked_add(size)?
        .checked_add(footer_size)
        .filter(|total| *total <= bytes.len())
}

fn valid_mp3_trailer(bytes: &[u8]) -> bool {
    bytes.is_empty()
        || (bytes.len() == 128 && bytes.starts_with(b"TAG"))
        || (bytes.len() <= 4_096 && bytes.iter().all(|byte| *byte == 0))
}

fn store_asset_bundle(input: AssetStoreRequest<'_>) -> Result<AssetBundle, ProviderFailure> {
    let AssetStoreRequest {
        root,
        task_id,
        request,
        prompt,
        prompt_hash,
        retain_prompt,
        raw_response,
        audio,
        model,
        provider_request_id,
        inspection,
        generated_at,
        terms_version,
    } = input;
    validate_task_id(task_id).map_err(|_| ProviderFailure::Storage)?;
    let directory = root.join(task_id);
    fs::create_dir_all(&directory).map_err(|_| ProviderFailure::Storage)?;
    set_directory_private(&directory).map_err(|_| ProviderFailure::Storage)?;
    let extension = match request.output_format {
        AudioOutputFormat::Mp3 => "mp3",
        AudioOutputFormat::Wav => "wav",
    };
    let output_hash = sha256_hex(audio);
    let response_hash = sha256_hex(raw_response);
    let audio_path = directory.join(format!("{output_hash}.{extension}"));
    write_new_private(&audio_path, audio)?;
    write_new_private(&directory.join("provider-response.json"), raw_response)?;
    let receipt = StoredReceipt {
        schema_version: 1,
        provider: "google_gemini",
        model,
        provider_request_id,
        local_task_id: task_id,
        submitted_prompt: retain_prompt.then_some(prompt),
        submitted_prompt_sha256: prompt_hash,
        output_sha256: &output_hash,
        provider_response_sha256: &response_hash,
        generated_at,
        requested_duration_seconds: request.duration_seconds,
        requested_language: request.language.as_deref(),
        detected_language: None,
        requested_output_format: request.output_format,
        actual_audio: inspection,
        reserved_cost_micro_usd: UNIT_COST_MICRO_USD,
        generation_cost_micro_usd: UNIT_COST_MICRO_USD,
        pricing_version: PRICING_VERSION,
        terms_version,
        rights_declared: request.rights_declared,
        synthid_expected: true,
        c2pa_expected: true,
        c2pa_status: "preserved_unverified",
        provider_billing_verified: false,
        input_asset_hashes: Vec::new(),
    };
    let receipt_bytes =
        serde_json::to_vec_pretty(&receipt).map_err(|_| ProviderFailure::Storage)?;
    write_new_private(&directory.join("receipt.json"), &receipt_bytes)?;
    Ok(AssetBundle {
        audio_path,
        output_sha256: output_hash,
        generated_at: generated_at.into(),
    })
}

fn write_new_private(path: &Path, bytes: &[u8]) -> Result<(), ProviderFailure> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|_| ProviderFailure::Storage)?;
    file.write_all(bytes)
        .map_err(|_| ProviderFailure::Storage)?;
    file.sync_all().map_err(|_| ProviderFailure::Storage)
}

fn store_failure_receipt(
    root: &Path,
    task_id: &str,
    receipt: &StoredFailureReceipt,
) -> Result<(), ProviderFailure> {
    validate_task_id(task_id).map_err(|_| ProviderFailure::Storage)?;
    let directory = root.join(task_id);
    fs::create_dir_all(&directory).map_err(|_| ProviderFailure::Storage)?;
    set_directory_private(&directory).map_err(|_| ProviderFailure::Storage)?;
    let bytes = serde_json::to_vec_pretty(receipt).map_err(|_| ProviderFailure::Storage)?;
    write_new_private(&directory.join("failure-receipt.json"), &bytes)
}

fn set_directory_private(_path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(_path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn compile_prompt(request: &GenerationRequest) -> String {
    let mut lines = vec![request.prompt.trim().to_owned()];
    if request.seamless_loop {
        lines.push("Provider control: render as a seamless DJ-loopable phrase with a strong first downbeat, phrase ending that resolves into bar 1, no long intro, no long outro, and no reverb tail that breaks looping.".into());
    }
    if let Some(key) = request
        .key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        lines.push(format!("Provider control: key {key}."));
    }
    if let Some(center) = request
        .tonal_center
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        lines.push(format!("Provider control: tonal center {center}."));
    }
    if let Some(intensity) = request.production_intensity {
        lines.push(format!(
            "Provider control: production intensity {} percent.",
            (intensity.clamp(0.0, 1.0) * 100.0).round()
        ));
    }
    if let Some(negative) = request
        .negative_prompt
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        lines.push(format!("Provider control: avoid {negative}."));
    }
    lines.join("\n")
}

fn detect_audio_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE" {
        Some("audio/wav")
    } else if bytes.starts_with(b"ID3")
        || (bytes.len() >= 2 && bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0)
    {
        Some("audio/mpeg")
    } else {
        None
    }
}

fn normalize_audio_mime(value: &str) -> Option<&'static str> {
    match value
        .split(';')
        .next()?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "audio/mpeg" | "audio/mp3" => Some("audio/mpeg"),
        "audio/wav" | "audio/wave" | "audio/x-wav" => Some("audio/wav"),
        _ => None,
    }
}

fn load_auth_from_env() -> Result<LyriaAuth, ProviderError> {
    if let Ok(key) = env::var(GEMINI_KEY_ENV) {
        validate_api_key(&key)?;
        return sensitive_api_key(&key).map(|value| LyriaAuth::ApiKey(Arc::new(value)));
    }
    if env::var(GCP_AUTH_ENV)
        .ok()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("gcloud"))
    {
        return Ok(LyriaAuth::Gcloud);
    }
    Err(ProviderError::Configuration(
        "GEMINI_API_KEY is required for Lyria, or set MUSICA_GCP_AUTH=gcloud to use gcloud application-default credentials",
    ))
}

fn gcloud_access_token() -> Result<String, ProviderFailure> {
    let mut command = Command::new("gcloud");
    command.args(["auth", "application-default", "print-access-token"]);
    crate::process_util::hide_console_window(&mut command);
    let output = command
        .output()
        .map_err(|_| ProviderFailure::Authentication)?;
    if !output.status.success() {
        return Err(ProviderFailure::Authentication);
    }
    let token = String::from_utf8(output.stdout).map_err(|_| ProviderFailure::Authentication)?;
    let token = token.trim();
    if token.len() < 20
        || token.len() > 4096
        || !token
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(ProviderFailure::Authentication);
    }
    Ok(token.to_owned())
}

fn validate_api_key(key: &str) -> Result<(), ProviderError> {
    if key.len() < 20 || key.len() > 512 || !key.bytes().all(|byte| byte.is_ascii_graphic()) {
        return Err(ProviderError::Configuration("GEMINI_API_KEY is invalid"));
    }
    Ok(())
}

fn sensitive_api_key(key: &str) -> Result<HeaderValue, ProviderError> {
    let mut value = HeaderValue::from_str(key)
        .map_err(|_| ProviderError::Configuration("GEMINI_API_KEY is invalid"))?;
    value.set_sensitive(true);
    Ok(value)
}

fn validate_client_request_id(value: &str) -> Result<(), ProviderError> {
    if value.len() < 8
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b':'))
    {
        return Err(ProviderError::Validation("clientRequestId is invalid"));
    }
    Ok(())
}

fn valid_provider_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_PROVIDER_ID_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':' | b'/')
        })
}

fn validate_terms_version(value: &str) -> Result<(), ProviderError> {
    if value.len() > 128 || !valid_provider_id(value) {
        return Err(ProviderError::Configuration(
            "MUSICA_CREATIVE_TERMS_VERSION is invalid",
        ));
    }
    Ok(())
}

fn parse_retain_prompts(value: &str) -> Result<bool, ProviderError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" => Ok(true),
        "0" | "false" => Ok(false),
        _ => Err(ProviderError::Configuration(
            "MUSICA_CREATIVE_RETAIN_PROMPTS must be true, false, 1, or 0",
        )),
    }
}

fn parse_request_timeout(value: &str) -> Result<u64, ProviderError> {
    let seconds = value.parse::<u64>().map_err(|_| {
        ProviderError::Configuration(
            "MUSICA_CREATIVE_REQUEST_TIMEOUT_SECONDS must be a whole number",
        )
    })?;
    if !(MIN_REQUEST_TIMEOUT_SECONDS..=MAX_REQUEST_TIMEOUT_SECONDS).contains(&seconds) {
        return Err(ProviderError::Configuration(
            "MUSICA_CREATIVE_REQUEST_TIMEOUT_SECONDS must be between 60 and 900",
        ));
    }
    Ok(seconds)
}

fn validate_language(value: &str) -> Result<(), ProviderError> {
    const LANGUAGES: [&str; 8] = [
        "english",
        "german",
        "spanish",
        "french",
        "hindi",
        "japanese",
        "korean",
        "portuguese",
    ];
    if LANGUAGES
        .iter()
        .any(|language| value.eq_ignore_ascii_case(language))
    {
        Ok(())
    } else {
        Err(ProviderError::Validation(
            "language is not in the documented Lyria 3 vocal language set",
        ))
    }
}

fn parse_usd_to_micro(value: &str) -> Result<u64, ProviderError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| ProviderError::Configuration("creative generation budget is invalid"))?;
    usd_f64_to_micro(parsed)
        .map_err(|_| ProviderError::Configuration("creative generation budget is invalid"))
}

fn usd_f64_to_micro(value: f64) -> Result<u64, ProviderError> {
    if !value.is_finite() || !(0.0..=10_000.0).contains(&value) {
        return Err(ProviderError::Validation("generation budget is invalid"));
    }
    Ok((value * 1_000_000.0).round() as u64)
}

const fn micro_to_usd(value: u64) -> f64 {
    value as f64 / 1_000_000.0
}

fn new_task_id() -> String {
    let mut bytes = [0_u8; 16];
    OsRng.fill_bytes(&mut bytes);
    let mut output = String::with_capacity(32);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> GenerationRequest {
        GenerationRequest {
            prompt: "Cinematic melodic techno with a strong social hook".into(),
            duration_seconds: 150,
            instrumental: true,
            seed: None,
            language: None,
            bpm: Some(128.0),
            lyrics: None,
            structure: vec![
                crate::creative_provider::TimedSection {
                    time_seconds: 0.0,
                    section: "intro".into(),
                },
                crate::creative_provider::TimedSection {
                    time_seconds: 42.0,
                    section: "first drop".into(),
                },
            ],
            output_format: AudioOutputFormat::Mp3,
            reference_assets: Vec::new(),
            seamless_loop: false,
            key: None,
            tonal_center: None,
            negative_prompt: None,
            production_intensity: None,
            max_cost_usd: Some(0.08),
            candidate_count: 1,
            max_attempts: 1,
            rights_declared: false,
            client_request_id: Some("request-0001".into()),
        }
    }

    fn provider(max_budget_micro_usd: u64) -> LyriaProvider {
        LyriaProvider {
            client: Client::builder().build().expect("test HTTPS client"),
            auth: LyriaAuth::ApiKey(Arc::new(
                sensitive_api_key("AIza-test-key-that-is-long-enough")
                    .expect("valid sensitive test header"),
            )),
            jobs: Arc::new(RwLock::new(JobRegistry::default())),
            permits: Arc::new(Semaphore::new(2)),
            asset_root: Arc::new(PathBuf::from("test-generated-assets")),
            max_budget_micro_usd,
            retain_prompts: true,
            terms_version: None,
        }
    }

    fn queued_task(id: &str) -> GenerationTask {
        GenerationTask {
            id: id.into(),
            status: GenerationStatus::Queued,
            title: Some("Lyria 3 Pro generation".into()),
            audio_url: None,
            provider: "lyria_3_pro".into(),
            model: Some(LYRIA_MODEL.into()),
            has_audio: false,
            audio_mime_type: None,
            lyrics: None,
            structure: None,
            actual_duration_seconds: None,
            sample_rate_hz: None,
            channels: None,
            reserved_cost_usd: Some(micro_to_usd(UNIT_COST_MICRO_USD)),
            generation_cost_usd: None,
            provenance: None,
            error_code: None,
            cancellation_requested: false,
            provider_cancel_confirmed: false,
            completed_after_cancel: false,
        }
    }

    fn insert_reserved(registry: &mut JobRegistry, task_id: &str, client_request_id: &str) {
        assert!(registry.can_reserve(UNIT_COST_MICRO_USD, DEFAULT_MAX_BUDGET_MICRO_USD));
        registry.add_reservation(UNIT_COST_MICRO_USD);
        registry
            .client_requests
            .insert(client_request_id.into(), task_id.into());
        registry.jobs.insert(
            task_id.into(),
            JobRecord {
                task: queued_task(task_id),
                prompt_hash: sha256_hex(b"test prompt"),
                asset_path: None,
                asset_sha256: None,
                abort_handle: None,
                cancellation_requested: false,
                dispatched: false,
                paid_post_attempts: 0,
                cost_state: CostState::Reserved,
            },
        );
    }

    fn wav_bytes(data_length: usize) -> Vec<u8> {
        let mut wav = Vec::with_capacity(44 + data_length);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36_u32 + data_length as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&48_000_u32.to_le_bytes());
        wav.extend_from_slice(&192_000_u32.to_le_bytes());
        wav.extend_from_slice(&4_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_length as u32).to_le_bytes());
        wav.resize(wav.len() + data_length, 0);
        wav
    }

    fn mp3_bytes(frame_count: usize) -> Vec<u8> {
        let header = [0xff, 0xfb, 0x90, 0x00];
        let frame_length = parse_mp3_header(&header)
            .expect("valid test MP3 header")
            .frame_length;
        let mut bytes = Vec::with_capacity(frame_length * frame_count);
        for _ in 0..frame_count {
            let start = bytes.len();
            bytes.extend_from_slice(&header);
            bytes.resize(start + frame_length, 0);
        }
        bytes
    }

    #[test]
    fn provider_ready_prompt_is_not_rewritten_or_duplicated() {
        let mut request = request();
        request.prompt = "  Compiled prompt with [0:42] drop and supplied lyrics.  ".into();
        assert_eq!(
            compile_prompt(&request),
            "Compiled prompt with [0:42] drop and supplied lyrics."
        );
    }

    #[test]
    fn loop_and_tone_controls_are_appended_for_provider() {
        let mut request = request();
        request.prompt = "Hardgroove loop".into();
        request.duration_seconds = 32;
        request.seamless_loop = true;
        request.key = Some("F minor".into());
        request.tonal_center = Some("sub bass around F1 with bright chord stabs".into());
        request.production_intensity = Some(0.82);
        request.negative_prompt = Some("muddy low end, weak kick, long ambient intro".into());

        let prompt = compile_prompt(&request);
        assert!(prompt.contains("seamless DJ-loopable phrase"));
        assert!(prompt.contains("key F minor"));
        assert!(prompt.contains("tonal center sub bass around F1"));
        assert!(prompt.contains("production intensity 82 percent"));
        assert!(prompt.contains("avoid muddy low end"));
    }

    #[test]
    fn wav_inspection_uses_encoded_metadata() {
        let wav = wav_bytes(192_000);
        let inspection = inspect_wav(&wav).expect("valid WAV");
        assert_eq!(inspection.sample_rate_hz, 48_000);
        assert_eq!(inspection.channels, 2);
        assert!((inspection.duration_seconds - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn wav_inspection_rejects_inconsistent_container_metadata() {
        let mut wav = wav_bytes(1_920);
        wav[4..8].copy_from_slice(&36_u32.to_le_bytes());
        assert!(matches!(
            inspect_wav(&wav),
            Err(ProviderFailure::UnsupportedMedia)
        ));

        let mut wav = wav_bytes(1_920);
        wav[28..32].copy_from_slice(&1_u32.to_le_bytes());
        assert!(matches!(
            inspect_wav(&wav),
            Err(ProviderFailure::UnsupportedMedia)
        ));
    }

    #[test]
    fn duration_policy_rejects_over_limit_and_warns_on_materially_short_audio() {
        assert!(validate_audio_inspection(AudioInspection {
            duration_seconds: 184.0,
            sample_rate_hz: 48_000,
            channels: 2,
        })
        .is_ok());
        assert!(matches!(
            validate_audio_inspection(AudioInspection {
                duration_seconds: 184.001,
                sample_rate_hz: 48_000,
                channels: 2,
            }),
            Err(ProviderFailure::UnsupportedMedia)
        ));
        assert!(output_shorter_than_requested(89.0, 120));
        assert!(!output_shorter_than_requested(90.0, 120));
        assert!(output_shorter_than_requested(29.9, 31));
    }

    #[test]
    fn mp3_inspection_requires_complete_consecutive_frames() {
        let mp3 = mp3_bytes(10);
        let inspection = inspect_mp3(&mp3).expect("valid MP3 frame sequence");
        assert_eq!(inspection.sample_rate_hz, 44_100);
        assert_eq!(inspection.channels, 2);
        assert!(inspection.duration_seconds > 0.2);

        assert!(matches!(
            inspect_mp3(&mp3_bytes(1)),
            Err(ProviderFailure::UnsupportedMedia)
        ));
        let mut truncated = mp3;
        truncated.pop();
        assert!(matches!(
            inspect_mp3(&truncated),
            Err(ProviderFailure::UnsupportedMedia)
        ));
    }

    #[test]
    fn rejects_mime_signature_mismatch() {
        assert_eq!(detect_audio_mime(b"RIFF\0\0\0\0WAVE"), Some("audio/wav"));
        assert_eq!(
            normalize_audio_mime("audio/mpeg; codecs=mp3"),
            Some("audio/mpeg")
        );
        assert_eq!(normalize_audio_mime("text/html"), None);
    }

    #[test]
    fn interaction_parser_rejects_malformed_base64_and_wrong_signature() {
        let malformed = json!({
            "id": "interactions/test-1",
            "model": LYRIA_MODEL,
            "steps": [{
                "type": "model_output",
                "content": [{"type": "audio", "data": "%%%", "mime_type": "audio/mpeg"}]
            }]
        });
        assert!(matches!(
            parse_interaction(
                &serde_json::to_vec(&malformed).expect("serialize response"),
                None,
                AudioOutputFormat::Mp3,
            ),
            Err(ProviderFailure::Malformed)
        ));

        let wrong_signature = json!({
            "id": "interactions/test-2",
            "model": LYRIA_MODEL,
            "steps": [{
                "type": "model_output",
                "content": [{
                    "type": "audio",
                    "data": BASE64.encode(b"<html>not audio</html>"),
                    "mime_type": "audio/mpeg"
                }]
            }]
        });
        assert!(matches!(
            parse_interaction(
                &serde_json::to_vec(&wrong_signature).expect("serialize response"),
                None,
                AudioOutputFormat::Mp3,
            ),
            Err(ProviderFailure::UnsupportedMedia)
        ));
    }

    #[test]
    fn interaction_parser_requires_exactly_one_matching_audio_block() {
        let wav = BASE64.encode(wav_bytes(1_920));
        let response = json!({
            "id": "interactions/test-3",
            "model": LYRIA_MODEL,
            "steps": [{
                "type": "model_output",
                "content": [
                    {"type": "text", "text": "Structure: intro"},
                    {"type": "audio", "data": wav, "mime_type": "audio/wav"}
                ]
            }]
        });
        let parsed = parse_interaction(
            &serde_json::to_vec(&response).expect("serialize response"),
            None,
            AudioOutputFormat::Wav,
        )
        .expect("one valid WAV block");
        assert_eq!(parsed.audio_mime_type, "audio/wav");
        assert_eq!(parsed.inspection.sample_rate_hz, 48_000);

        let two_audio_blocks = json!({
            "id": "interactions/test-4",
            "model": LYRIA_MODEL,
            "steps": [{
                "type": "model_output",
                "content": [
                    {"type": "audio", "data": BASE64.encode(wav_bytes(1_920))},
                    {"type": "audio", "data": BASE64.encode(wav_bytes(1_920))}
                ]
            }]
        });
        assert!(matches!(
            parse_interaction(
                &serde_json::to_vec(&two_audio_blocks).expect("serialize response"),
                None,
                AudioOutputFormat::Wav,
            ),
            Err(ProviderFailure::Malformed)
        ));
    }

    #[test]
    fn provider_endpoint_and_wav_request_shape_are_fixed() {
        assert_eq!(LYRIA_HOST, "generativelanguage.googleapis.com");
        assert_eq!(
            LYRIA_ENDPOINT,
            "https://generativelanguage.googleapis.com/v1beta/interactions"
        );
        let mp3 = serde_json::to_value(LyriaInteractionRequest {
            model: LYRIA_MODEL,
            input: "test prompt",
            store: false,
            response_format: None,
        })
        .expect("serialize MP3 request");
        assert_eq!(
            mp3,
            json!({"model": LYRIA_MODEL, "input": "test prompt", "store": false})
        );

        let wav = serde_json::to_value(LyriaInteractionRequest {
            model: LYRIA_MODEL,
            input: "test prompt",
            store: false,
            response_format: Some(LyriaResponseFormat { kind: "audio" }),
        })
        .expect("serialize WAV request");
        assert_eq!(
            wav,
            json!({
                "model": LYRIA_MODEL,
                "input": "test prompt",
                "store": false,
                "response_format": {"type": "audio"}
            })
        );
        assert!(!wav.to_string().contains("AIza-test-secret"));
    }

    #[test]
    fn lyria_validation_requires_one_paid_attempt_and_explicit_budget() {
        let provider = provider(DEFAULT_MAX_BUDGET_MICRO_USD);
        assert!(provider.validate_request(&request()).is_ok());

        let mut invalid = request();
        invalid.duration_seconds = 30;
        assert!(provider.validate_request(&invalid).is_err());

        let mut invalid = request();
        invalid.max_attempts = 2;
        assert!(provider.validate_request(&invalid).is_err());

        let mut invalid = request();
        invalid.max_cost_usd = Some(0.079);
        assert!(provider.validate_request(&invalid).is_err());

        let mut invalid = request();
        invalid.language = Some("Italian".into());
        assert!(provider.validate_request(&invalid).is_err());
    }

    #[test]
    fn dedup_and_session_budget_are_atomic_ledger_invariants() {
        let mut registry = JobRegistry::default();
        for index in 0..4 {
            insert_reserved(
                &mut registry,
                &format!("task{index}"),
                &format!("client-request-{index}"),
            );
        }
        assert_eq!(registry.reserved_micro_usd, DEFAULT_MAX_BUDGET_MICRO_USD);
        assert!(!registry.can_reserve(UNIT_COST_MICRO_USD, DEFAULT_MAX_BUDGET_MICRO_USD));
        let duplicate = registry
            .existing_task("client-request-0")
            .expect("deduplicated task");
        assert_eq!(duplicate.id, "task0");
        assert_eq!(registry.reserved_micro_usd, DEFAULT_MAX_BUDGET_MICRO_USD);

        registry.release_reservation("task0");
        assert_eq!(
            registry.reserved_micro_usd,
            DEFAULT_MAX_BUDGET_MICRO_USD - UNIT_COST_MICRO_USD
        );
        assert!(registry.can_reserve(UNIT_COST_MICRO_USD, DEFAULT_MAX_BUDGET_MICRO_USD));
    }

    #[test]
    fn paid_dispatch_can_start_exactly_once() {
        let mut registry = JobRegistry::default();
        insert_reserved(&mut registry, "task-once", "client-request-once");
        assert_eq!(
            registry.begin_paid_dispatch("task-once"),
            DispatchDecision::Started
        );
        assert_eq!(
            registry.begin_paid_dispatch("task-once"),
            DispatchDecision::NotStarted
        );
        let record = registry.jobs.get("task-once").expect("task record");
        assert_eq!(record.paid_post_attempts, 1);
        assert!(record.dispatched);
        assert_eq!(record.task.status, GenerationStatus::Processing);
    }

    #[test]
    fn cancellation_releases_only_before_paid_dispatch() {
        let mut registry = JobRegistry::default();
        insert_reserved(&mut registry, "task-before", "client-before");
        let before = registry
            .jobs
            .get_mut("task-before")
            .expect("queued task")
            .request_cancellation();
        assert_eq!(before, CancellationDisposition::BeforeDispatch);
        assert!(
            registry
                .jobs
                .get("task-before")
                .expect("cancelled task")
                .task
                .provider_cancel_confirmed
        );
        registry.release_reservation("task-before");
        assert_eq!(registry.reserved_micro_usd, 0);
        assert_eq!(
            registry
                .jobs
                .get("task-before")
                .expect("cancelled task")
                .cost_state,
            CostState::Released
        );

        insert_reserved(&mut registry, "task-after", "client-after");
        assert_eq!(
            registry.begin_paid_dispatch("task-after"),
            DispatchDecision::Started
        );
        let after = registry
            .jobs
            .get_mut("task-after")
            .expect("processing task")
            .request_cancellation();
        assert_eq!(after, CancellationDisposition::AfterDispatch);
        assert_eq!(
            registry
                .jobs
                .get("task-after")
                .expect("processing task")
                .task
                .status,
            GenerationStatus::Processing
        );
        assert_eq!(registry.reserved_micro_usd, UNIT_COST_MICRO_USD);
        registry.charge_reservation("task-after");
        assert_eq!(registry.reserved_micro_usd, 0);
        assert_eq!(registry.charged_micro_usd, UNIT_COST_MICRO_USD);
    }

    #[test]
    fn one_song_is_exactly_eight_cents() {
        assert_eq!(UNIT_COST_MICRO_USD, 80_000);
        assert!((micro_to_usd(UNIT_COST_MICRO_USD) - 0.08).abs() < f64::EPSILON);
    }

    #[test]
    fn immutable_asset_bundle_preserves_bytes_hashes_privacy_and_modes() {
        let root = env::temp_dir().join(format!("musica-vj-asset-test-{}", new_task_id()));
        let task_id = new_task_id();
        let mut generation_request = request();
        generation_request.output_format = AudioOutputFormat::Wav;
        let audio = wav_bytes(192_000);
        let raw_response = br#"{"id":"interactions/storage-test"}"#;
        let prompt = "confidential supplied lyrics";
        let prompt_hash = sha256_hex(prompt.as_bytes());
        let inspection = inspect_wav(&audio).expect("inspect fixture");
        let input = || AssetStoreRequest {
            root: &root,
            task_id: &task_id,
            request: &generation_request,
            prompt,
            prompt_hash: &prompt_hash,
            retain_prompt: false,
            raw_response,
            audio: &audio,
            model: LYRIA_MODEL,
            provider_request_id: "interactions/storage-test",
            inspection,
            generated_at: "2026-07-18T00:00:00Z",
            terms_version: Some("terms/2026-07"),
        };

        let bundle = store_asset_bundle(input()).expect("store immutable asset bundle");
        assert_eq!(fs::read(&bundle.audio_path).expect("read audio"), audio);
        let directory = root.join(&task_id);
        assert_eq!(
            fs::read(directory.join("provider-response.json")).expect("read response"),
            raw_response
        );
        let receipt: serde_json::Value = serde_json::from_slice(
            &fs::read(directory.join("receipt.json")).expect("read receipt"),
        )
        .expect("parse receipt");
        assert!(receipt["submittedPrompt"].is_null());
        assert_eq!(
            receipt["submittedPromptSha256"].as_str(),
            Some(prompt_hash.as_str())
        );
        let output_hash = sha256_hex(&audio);
        assert_eq!(receipt["outputSha256"].as_str(), Some(output_hash.as_str()));
        let response_hash = sha256_hex(raw_response);
        assert_eq!(
            receipt["providerResponseSha256"].as_str(),
            Some(response_hash.as_str())
        );
        assert!(matches!(
            store_asset_bundle(input()),
            Err(ProviderFailure::Storage)
        ));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&directory)
                    .expect("asset directory metadata")
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
            for name in [
                bundle.audio_path.file_name().expect("audio file name"),
                std::ffi::OsStr::new("provider-response.json"),
                std::ffi::OsStr::new("receipt.json"),
            ] {
                assert_eq!(
                    fs::metadata(directory.join(name))
                        .expect("asset file metadata")
                        .permissions()
                        .mode()
                        & 0o777,
                    0o600
                );
            }
        }

        let failed_task_id = new_task_id();
        let failure = StoredFailureReceipt {
            schema_version: 1,
            provider: "google_gemini",
            model: LYRIA_MODEL,
            provider_request_id: Some("requests/storage-failure-test".into()),
            local_task_id: failed_task_id.clone(),
            submitted_prompt_sha256: prompt_hash,
            failed_at: "2026-07-18T00:00:01Z".into(),
            error_code: "ambiguous_paid_request_outcome".into(),
            dispatched: true,
            cancellation_requested: false,
            reserved_cost_micro_usd: UNIT_COST_MICRO_USD,
            generation_cost_micro_usd: Some(UNIT_COST_MICRO_USD),
            pricing_version: PRICING_VERSION,
            charge_basis: "conservative_post_dispatch_potential",
            provider_billing_verified: false,
        };
        store_failure_receipt(&root, &failed_task_id, &failure)
            .expect("store immutable failure receipt");
        let stored_failure: serde_json::Value = serde_json::from_slice(
            &fs::read(root.join(&failed_task_id).join("failure-receipt.json"))
                .expect("read failure receipt"),
        )
        .expect("parse failure receipt");
        assert_eq!(
            stored_failure["generation_cost_micro_usd"].as_u64(),
            Some(UNIT_COST_MICRO_USD)
        );
        assert_eq!(
            stored_failure["provider_request_id"].as_str(),
            Some("requests/storage-failure-test")
        );
        assert!(matches!(
            store_failure_receipt(&root, &failed_task_id, &failure),
            Err(ProviderFailure::Storage)
        ));

        fs::remove_dir_all(&root).expect("remove isolated test directory");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failure_receipt_is_durable_before_terminal_publication() {
        let root = env::temp_dir().join(format!("musica-vj-failure-test-{}", new_task_id()));
        let mut provider = provider(DEFAULT_MAX_BUDGET_MICRO_USD);
        provider.asset_root = Arc::new(root.clone());
        let task_id = "failure-sequencing-task";
        {
            let mut registry = provider.jobs.write().await;
            insert_reserved(&mut registry, task_id, "failure-sequencing-request");
            assert_eq!(
                registry.jobs.get(task_id).expect("queued task").task.status,
                GenerationStatus::Queued
            );
        }

        provider
            .fail(
                task_id,
                ProviderFailure::Authentication,
                false,
                Some("requests/http-error-123".into()),
            )
            .await;

        let receipt_path = root.join(task_id).join("failure-receipt.json");
        assert!(receipt_path.is_file());
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&receipt_path).expect("read durable failure receipt"))
                .expect("parse durable failure receipt");
        assert_eq!(
            receipt["provider_request_id"].as_str(),
            Some("requests/http-error-123")
        );
        let registry = provider.jobs.read().await;
        let record = registry.jobs.get(task_id).expect("failed task");
        assert_eq!(record.task.status, GenerationStatus::Failed);
        assert_eq!(
            record.task.error_code.as_deref(),
            Some("authentication_failed")
        );
        assert_eq!(record.cost_state, CostState::Released);
        drop(registry);

        fs::remove_dir_all(&root).expect("remove isolated failure test directory");
    }

    #[test]
    fn failure_publication_preserves_cancelled_status_and_marks_receipt_errors() {
        let mut registry = JobRegistry::default();
        insert_reserved(
            &mut registry,
            "cancelled-failure",
            "cancelled-failure-request",
        );
        let record = registry
            .jobs
            .get_mut("cancelled-failure")
            .expect("cancelled task");
        record.task.status = GenerationStatus::Cancelled;
        publish_failure(record, "provider_service_error", false);
        assert_eq!(record.task.status, GenerationStatus::Cancelled);
        assert_eq!(
            record.task.error_code.as_deref(),
            Some("provider_service_error_failure_receipt_storage_failed")
        );
    }

    #[test]
    fn provider_ids_cannot_inject_control_characters() {
        assert!(valid_provider_id("interactions/abc-123"));
        assert!(!valid_provider_id("abc\nsecret"));
        assert!(!valid_provider_id("abc\"secret"));
    }

    #[test]
    fn credential_and_privacy_settings_are_strictly_validated() {
        assert!(validate_api_key("AIza-test-key-that-is-long-enough").is_ok());
        assert!(validate_api_key("short").is_err());
        assert!(validate_api_key("AIza invalid key with spaces").is_err());
        assert!(!parse_retain_prompts("FALSE").expect("valid false"));
        assert!(parse_retain_prompts("sometimes").is_err());
        assert!(validate_terms_version("google-terms/2026-07").is_ok());
        assert!(validate_terms_version("bad terms value").is_err());
        assert!(sensitive_api_key("AIza-test-key-that-is-long-enough")
            .expect("sensitive key")
            .is_sensitive());
        assert_eq!(parse_request_timeout("600").expect("valid timeout"), 600);
        assert!(parse_request_timeout("59").is_err());
        assert!(parse_request_timeout("901").is_err());
    }

    #[test]
    fn provider_errors_are_classified_without_exposing_raw_details() {
        assert!(matches!(
            classify_http_error(
                429,
                br#"{"error":{"status":"RESOURCE_EXHAUSTED","message":"quota exceeded"}}"#,
            ),
            ProviderFailure::Quota
        ));
        assert!(matches!(
            classify_http_error(
                400,
                br#"{"error":{"status":"INVALID_ARGUMENT","message":"blocked by safety"}}"#,
            ),
            ProviderFailure::Safety
        ));
        assert!(matches!(
            classify_http_error(401, b"not-json"),
            ProviderFailure::Authentication
        ));
    }

    #[test]
    fn output_text_keeps_provider_structure_separate() {
        let (lyrics, structure) = classify_text_blocks(vec![
            "[Verse]\nNeon rain".into(),
            r#"{"sections":[{"type":"intro"}]}"#.into(),
        ]);
        assert!(lyrics.is_some_and(|value| value.contains("Neon rain")));
        assert!(structure.is_some_and(|value| value.contains("sections")));
    }
}
