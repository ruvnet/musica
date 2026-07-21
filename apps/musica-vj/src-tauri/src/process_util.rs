use std::process::Command;

/// Suppresses the console window Windows otherwise pops up when a GUI-subsystem
/// process (the Tauri app, which owns no console) spawns a console-subsystem
/// child such as `ffmpeg.exe` or `gcloud`. No-op on macOS/Linux, where child
/// processes never allocate a visible window.
pub(crate) fn hide_console_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}
