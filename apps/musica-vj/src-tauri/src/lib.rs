mod broadcast_provider;
mod cognitum_provider;
mod controller_bridge;
mod creative_provider;
mod lyria_provider;
mod lyria_realtime_provider;
mod meta_llm_provider;
mod process_util;
mod restream_provider;

use broadcast_provider::{
    broadcast_leave, broadcast_list, broadcast_listen, broadcast_publish, broadcast_status,
    broadcast_stop, BroadcastProvider,
};
use cognitum_provider::{
    cognitum_auth_manual_complete, cognitum_auth_manual_start, cognitum_auth_start,
    cognitum_autodj_brief, cognitum_fx_direction, cognitum_lyria_credential, cognitum_set_arc,
    cognitum_sign_out, cognitum_status, cognitum_style_pack, cognitum_visual_direction,
    cognitum_visual_plugin, cognitum_vocal_guidance, CognitumProvider,
};
use creative_provider::{
    creative_cancel_generation, creative_download_audio, creative_generate,
    creative_generation_status, creative_provider_status, CreativeProvider,
};
use lyria_realtime_provider::{
    lyria_realtime_configure_key, lyria_realtime_poll_audio, lyria_realtime_start,
    lyria_realtime_status, lyria_realtime_stop, lyria_realtime_update, LyriaRealtimeProvider,
};
use meta_llm_provider::{meta_llm_plan, meta_llm_status, MetaLlmProvider};
use restream_provider::{
    restream_push_chunk, restream_start, restream_status, restream_stop, transcode_to_mp4,
    RestreamProvider,
};
use tauri::Manager;

const PROVIDER_CONFIG_TEMPLATE: &str = "\
# Musica VJ provider configuration.
# Restart Musica after editing. Lines are KEY=value; # starts a comment.
#
# Google Lyria / Gemini (live AI music + generation). Both lines are required
# for the Start Session button to activate:
# MUSICA_LYRIA_REALTIME_ENABLED=true
# GEMINI_API_KEY=your_gemini_api_key_here
#
# Batch song/loop export (optional):
# MUSICA_CREATIVE_ENABLED=true
# MUSICA_CREATIVE_PROVIDER=lyria_3_pro
#
# Cognitum Meta-LLM env fallback (optional; OAuth sign-in is preferred):
# MUSICA_META_LLM_ENABLED=true
# MUSICA_META_LLM_API_TOKEN=your_token_here
#
# Route all AI text/planning features through a local meta-proxy (or any
# OpenAI-compatible gateway) with no browser sign-in — works on every platform:
# MUSICA_COGNITUM_API_BASE=http://127.0.0.1:8787
# MUSICA_COGNITUM_BEARER=your_meta_proxy_token
# (Lyria audio still needs GEMINI_API_KEY above; meta-proxy is a text-LLM plane.)
#
# Sign in with Cognitum One and get Lyria audio with no GEMINI_API_KEY: point at
# the credential broker and just sign in from the app (ADR-179):
# MUSICA_LYRIA_BROKER_URL=https://your-lyria-broker-url
";

/// Loads `KEY=value` lines from a provider config file into the process
/// environment, without overwriting variables already set by the shell.
/// Only recognized `GEMINI_API_KEY` / `MUSICA_*` / `GOOGLE_*` keys are applied,
/// so the file cannot inject arbitrary environment. Missing file is fine.
/// Returns true if the file existed and was read.
fn load_provider_config(path: &std::path::Path) -> bool {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return false;
    };
    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if !(key == "GEMINI_API_KEY" || key.starts_with("MUSICA_") || key.starts_with("GOOGLE_")) {
            continue;
        }
        let mut value = value.trim();
        if (value.starts_with('"') && value.ends_with('"') && value.len() >= 2)
            || (value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2)
        {
            value = &value[1..value.len() - 1];
        }
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, value);
        }
    }
    true
}

/// Loads provider config from every supported location so the packaged app is
/// configurable without a shell. Precedence (first value for a key wins, and
/// real process env always beats all of them):
///   1. `MUSICA_ENV_FILE` — explicit path override
///   2. `providers.env` next to the executable — drop-in-beside-the-app
///   3. `<app config dir>/providers.env` — the canonical writable location
///
/// On first run, creates the config dir and writes a self-documenting
/// `providers.env.example` there so the user can see exactly where and what to
/// configure.
fn load_all_provider_config(config_dir: Option<std::path::PathBuf>) {
    if let Some(path) = std::env::var_os("MUSICA_ENV_FILE") {
        load_provider_config(std::path::Path::new(&path));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            load_provider_config(&dir.join("providers.env"));
        }
    }
    if let Some(dir) = config_dir {
        let loaded = load_provider_config(&dir.join("providers.env"));
        if !loaded {
            let _ = std::fs::create_dir_all(&dir);
            let example = dir.join("providers.env.example");
            if !example.exists() {
                let _ = std::fs::write(&example, PROVIDER_CONFIG_TEMPLATE);
            }
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            // A packaged app launched from the Start menu / Finder inherits no
            // shell environment, so GEMINI_API_KEY and the MUSICA_* provider
            // flags are absent and every provider reports offline (the "Start
            // Session does nothing" symptom on Windows). Load a dotenv-style
            // config file from the app config dir first so the desktop build is
            // configurable without a terminal. Existing process env always wins,
            // so the dev shell workflow is unchanged.
            load_all_provider_config(app.path().app_config_dir().ok());
            // Stamp the running version into the window title so it's obvious
            // which build is open (helps confirm auto-updates landed).
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_title(&format!("Musica VJ v{}", env!("CARGO_PKG_VERSION")));
            }
            let asset_root = app.path().app_data_dir()?;
            app.manage(BroadcastProvider::default());
            app.manage(CognitumProvider::new());
            app.manage(CreativeProvider::from_env(asset_root));
            app.manage(LyriaRealtimeProvider::from_env());
            app.manage(MetaLlmProvider::from_env());
            app.manage(RestreamProvider::default());
            if let Err(error) = controller_bridge::start(app.handle()) {
                eprintln!("Logitech controller bridge unavailable: {error}");
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            broadcast_status,
            broadcast_publish,
            broadcast_stop,
            broadcast_list,
            broadcast_listen,
            broadcast_leave,
            cognitum_auth_manual_complete,
            cognitum_auth_manual_start,
            cognitum_auth_start,
            cognitum_autodj_brief,
            cognitum_fx_direction,
            cognitum_lyria_credential,
            cognitum_visual_direction,
            cognitum_visual_plugin,
            cognitum_vocal_guidance,
            cognitum_set_arc,
            cognitum_sign_out,
            cognitum_status,
            cognitum_style_pack,
            creative_provider_status,
            creative_generate,
            creative_generation_status,
            creative_download_audio,
            creative_cancel_generation,
            lyria_realtime_status,
            lyria_realtime_configure_key,
            lyria_realtime_start,
            lyria_realtime_update,
            lyria_realtime_stop,
            lyria_realtime_poll_audio,
            meta_llm_status,
            meta_llm_plan,
            restream_status,
            restream_start,
            restream_push_chunk,
            restream_stop,
            transcode_to_mp4
        ])
        .run(tauri::generate_context!())
        .expect("error while running Musica VJ");
}
