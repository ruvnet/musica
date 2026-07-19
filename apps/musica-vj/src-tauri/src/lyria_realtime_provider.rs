use std::{env, sync::Mutex};

use serde::{Deserialize, Serialize};

const ENABLE_ENV: &str = "MUSICA_LYRIA_REALTIME_ENABLED";
const API_KEY_ENV: &str = "GEMINI_API_KEY";
const MODEL: &str = "models/lyria-realtime-exp";
const SAMPLE_RATE_HZ: u32 = 48_000;
const CHANNELS: u8 = 2;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LyriaRealtimeStatus {
    available: bool,
    provider: String,
    model: String,
    sample_rate_hz: u32,
    channels: u8,
    audio_format: String,
    instrumental_only: bool,
    reason: Option<String>,
    active_session_id: Option<String>,
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

#[derive(Default)]
pub(crate) struct LyriaRealtimeProvider {
    enabled: bool,
    has_api_key: bool,
    session: Mutex<Option<LyriaRealtimeSession>>,
}

impl LyriaRealtimeProvider {
    pub(crate) fn from_env() -> Self {
        Self {
            enabled: env::var(ENABLE_ENV)
                .map(|value| matches!(value.as_str(), "true" | "1" | "yes"))
                .unwrap_or(false),
            has_api_key: env::var(API_KEY_ENV)
                .map(|value| !value.trim().is_empty())
                .unwrap_or(false),
            session: Mutex::new(None),
        }
    }

    fn status(&self) -> LyriaRealtimeStatus {
        let active_session_id = self
            .session
            .lock()
            .ok()
            .and_then(|session| session.as_ref().map(|active| active.id.clone()));
        let reason = if !self.enabled {
            Some(format!("{ENABLE_ENV}=true is required"))
        } else if !self.has_api_key {
            Some(format!("{API_KEY_ENV} is required"))
        } else {
            None
        };
        LyriaRealtimeStatus {
            available: self.enabled && self.has_api_key,
            provider: "lyria_realtime".into(),
            model: MODEL.into(),
            sample_rate_hz: SAMPLE_RATE_HZ,
            channels: CHANNELS,
            audio_format: "pcm16".into(),
            instrumental_only: true,
            reason,
            active_session_id,
        }
    }

    fn start(&self, request: LyriaRealtimeStartRequest) -> Result<LyriaRealtimeSession, String> {
        if !self.enabled {
            return Err(format!("{ENABLE_ENV}=true is required"));
        }
        if !self.has_api_key {
            return Err(format!("{API_KEY_ENV} is required"));
        }
        validate_request(&request)?;
        let session = LyriaRealtimeSession {
            id: format!("lrt-{}", monotonic_millis()),
            provider: "lyria_realtime".into(),
            model: MODEL.into(),
            state: "control_ready".into(),
            weighted_prompts: request.weighted_prompts,
            config: request.config,
            sample_rate_hz: SAMPLE_RATE_HZ,
            channels: CHANNELS,
            audio_format: "pcm16".into(),
        };
        *self
            .session
            .lock()
            .map_err(|_| "Lyria RealTime session lock failed")? = Some(session.clone());
        Ok(session)
    }

    fn update(&self, request: LyriaRealtimeStartRequest) -> Result<LyriaRealtimeSession, String> {
        validate_request(&request)?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| "Lyria RealTime session lock failed")?;
        let active = session
            .as_mut()
            .ok_or("Lyria RealTime session is not active")?;
        active.weighted_prompts = request.weighted_prompts;
        active.config = request.config;
        Ok(active.clone())
    }

    fn stop(&self) -> Result<(), String> {
        *self
            .session
            .lock()
            .map_err(|_| "Lyria RealTime session lock failed")? = None;
        Ok(())
    }
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
) -> LyriaRealtimeStatus {
    state.status()
}

#[tauri::command]
pub(crate) async fn lyria_realtime_start(
    state: tauri::State<'_, LyriaRealtimeProvider>,
    request: LyriaRealtimeStartRequest,
) -> Result<LyriaRealtimeSession, String> {
    state.start(request)
}

#[tauri::command]
pub(crate) async fn lyria_realtime_update(
    state: tauri::State<'_, LyriaRealtimeProvider>,
    request: LyriaRealtimeStartRequest,
) -> Result<LyriaRealtimeSession, String> {
    state.update(request)
}

#[tauri::command]
pub(crate) async fn lyria_realtime_stop(
    state: tauri::State<'_, LyriaRealtimeProvider>,
) -> Result<(), String> {
    state.stop()
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
}
