/**
 * Webview platform capability checks.
 *
 * The desktop shell runs a different webview per platform — WKWebView on macOS,
 * WebView2 on Windows, WebKitGTK on Linux — and they do not agree on which web
 * APIs exist. Centralize the sniffing here so the call sites stay readable and
 * the reasoning lives in one place.
 */

export function isMacWebview(): boolean {
  return typeof navigator !== "undefined" && /Macintosh|Mac OS X/i.test(navigator.userAgent);
}

/**
 * WKWebView does not implement `getDisplayMedia`. wry's permission handler
 * covers `DisplayCapture` on WebKitGTK only (tauri-apps/wry#1654); the macOS
 * implementation (tauri-apps/wry#1196) is still open. Screen capture therefore
 * has to be hidden on macOS rather than merely failing at click time.
 *
 * This is also why the Mac App Store build needs no screen-recording TCC
 * consent — see docs/mac-app-store-plan.md.
 */
export function supportsDisplayCapture(): boolean {
  if (isMacWebview()) return false;
  return typeof navigator !== "undefined" && typeof navigator.mediaDevices?.getDisplayMedia === "function";
}
