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
    cognitum_status, cognitum_style_pack, CognitumProvider,
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
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
