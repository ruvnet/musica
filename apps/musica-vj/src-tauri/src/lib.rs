mod cognitum_provider;
mod controller_bridge;
mod creative_provider;
mod lyria_provider;
mod lyria_realtime_provider;
mod meta_llm_provider;
mod restream_provider;

use cognitum_provider::{
    cognitum_auth_manual_complete, cognitum_auth_manual_start, cognitum_auth_start,
    cognitum_autodj_brief, cognitum_fx_direction, cognitum_set_arc, cognitum_sign_out,
    cognitum_status, cognitum_style_pack, cognitum_visual_direction, cognitum_visual_plugin,
    cognitum_vocal_guidance, CognitumProvider,
};
use creative_provider::{
    creative_cancel_generation, creative_download_audio, creative_generate,
    creative_generation_status, creative_provider_status, CreativeProvider,
};
use lyria_realtime_provider::{
    lyria_realtime_poll_audio, lyria_realtime_start, lyria_realtime_status, lyria_realtime_stop,
    lyria_realtime_update, LyriaRealtimeProvider,
};
use meta_llm_provider::{meta_llm_plan, meta_llm_status, MetaLlmProvider};
use restream_provider::{
    restream_push_chunk, restream_start, restream_status, restream_stop, RestreamProvider,
};
use tauri::Manager;

/// Loads `KEY=value` lines from a provider config file into the process
/// environment, without overwriting variables already set by the shell.
/// Only recognized `GEMINI_API_KEY` / `MUSICA_*` / `GOOGLE_*` keys are applied,
/// so the file cannot inject arbitrary environment. Missing file is fine.
fn load_provider_config(path: &std::path::Path) {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
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
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            // A packaged app launched from the Start menu / Finder inherits no
            // shell environment, so GEMINI_API_KEY and the MUSICA_* provider
            // flags are absent and every provider reports offline (the "Start
            // Session does nothing" symptom on Windows). Load a dotenv-style
            // config file from the app config dir first so the desktop build is
            // configurable without a terminal. Existing process env always wins,
            // so the dev shell workflow is unchanged.
            if let Ok(config_dir) = app.path().app_config_dir() {
                load_provider_config(&config_dir.join("providers.env"));
            }
            let asset_root = app.path().app_data_dir()?;
            app.manage(CognitumProvider::default());
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
            cognitum_auth_manual_complete,
            cognitum_auth_manual_start,
            cognitum_auth_start,
            cognitum_autodj_brief,
            cognitum_fx_direction,
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
            lyria_realtime_start,
            lyria_realtime_update,
            lyria_realtime_stop,
            lyria_realtime_poll_audio,
            meta_llm_status,
            meta_llm_plan,
            restream_status,
            restream_start,
            restream_push_chunk,
            restream_stop
        ])
        .run(tauri::generate_context!())
        .expect("error while running Musica VJ");
}
