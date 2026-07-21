//! macOS App Sandbox detection.
//!
//! A Mac App Store build runs inside the App Sandbox, which blocks spawning
//! executables that live outside the app bundle. Features built on external
//! command-line tools therefore cannot work there, and the useful thing to do
//! is say so plainly rather than let the user read a generic authentication
//! failure and go hunting for a credential problem that does not exist.

/// True when the process is running inside the macOS App Sandbox.
///
/// `APP_SANDBOX_CONTAINER_ID` is injected into the environment of every
/// sandboxed process by the container manager, and is absent otherwise. This is
/// a heuristic — there is no public API for the question — but it is the
/// conventional signal and it fails safe: a false negative just restores the
/// previous, less specific error message.
pub(crate) fn is_sandboxed() -> bool {
    cfg!(target_os = "macos") && std::env::var_os("APP_SANDBOX_CONTAINER_ID").is_some()
}

/// Why an external-tool feature is unavailable, or `None` when it should work.
pub(crate) fn external_tool_blocked(tool: &str) -> Option<String> {
    is_sandboxed().then(|| blocked_message(tool))
}

fn blocked_message(tool: &str) -> String {
    format!(
        "{tool} cannot be launched from the sandboxed App Store build; \
         sign in with Cognitum One or set GEMINI_API_KEY instead"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_message_names_the_tool_and_offers_an_alternative() {
        let message = blocked_message("gcloud");
        assert!(message.starts_with("gcloud "), "{message}");
        assert!(message.contains("GEMINI_API_KEY"), "{message}");
    }

    /// Guards against the environment check being inverted: a normal developer
    /// machine and a CI runner are both unsandboxed, so every external-tool
    /// feature must stay enabled here.
    #[test]
    fn a_normal_process_is_not_treated_as_sandboxed() {
        assert!(!is_sandboxed());
        assert!(external_tool_blocked("gcloud").is_none());
    }
}
