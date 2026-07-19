# Musica VJ for Logitech MX Creative Console

This directory contains the official C# Actions SDK companion for the Musica VJ macOS app. It targets .NET 8, references the `PluginApi.dll` installed by Logitech Options+, and activates with the Tauri bundle identifier `one.cognitum.musica.vj`.

## Actions

The keypad exposes play or pause, recording, previous and next track, mute, solo, six parameterized track triggers, previous and next visual, and tap tempo. The dial exposes master level, visual intensity, tempo, and the Sculpture, Motion, Atmosphere, and Ribbon artist macros as independent assignable adjustments. Touch key images are rendered with the SDK `BitmapBuilder`, so they remain legible at each supported device size without shipping fragile LCD bitmaps. Single-color SVG symbols are packaged for the Options+ action picker.

## Build and test

Prerequisites are macOS, the .NET 8 SDK, current Logitech Options+, and its Logi Plugin Service. Options+ installs the API at:

```text
/Applications/Utilities/LogiPluginService.app/Contents/MonoBundle/PluginApi.dll
```

Run portable core tests on any .NET 8 machine:

```sh
dotnet test tests/MusicaVj.Logitech.Core.Tests/MusicaVj.Logitech.Core.Tests.csproj
```

Build and register the development plugin on the Mac:

```sh
dotnet build src/MusicaVj.Logitech.Plugin/MusicaVj.Logitech.Plugin.csproj -c Debug
```

The build writes `MusicaVj.Logitech.Plugin.link` under `~/Library/Application Support/Logi/LogiPluginService/Plugins/`. Restart Logi Plugin Service from Options+ settings, select the MX Creative Console, and find Musica VJ under Installed Plugins. The app must have run once so the controller token exists.

For a nonstandard SDK location, pass a directory ending in `/`:

```sh
dotnet build src/MusicaVj.Logitech.Plugin/MusicaVj.Logitech.Plugin.csproj -p:PluginApiDir=/absolute/sdk/path/
```

## Package

Install Logitech's supported packaging tool, make a release build without a development link, then pack and verify the build root:

```sh
dotnet tool install --global LogiPluginTool
dotnet build src/MusicaVj.Logitech.Plugin/MusicaVj.Logitech.Plugin.csproj -c Release -p:SkipPluginLink=true
logiplugintool pack src/MusicaVj.Logitech.Plugin/bin/Release/ MusicaVj_1_0_0.lplug4
logiplugintool verify MusicaVj_1_0_0.lplug4
```

Marketplace submission still requires validation on physical MX Creative Keypad and Dialpad hardware and compliance review by Logitech.

## Options+ fallback

If the Actions SDK plugin is unavailable, create an app-specific Options+ profile for Musica VJ and assign F13 through F24 according to `options-plus-fallback.json`. That mapping matches the Tauri global shortcut listener. `Enter` triggers the selected track and `T` taps tempo while the app is focused.

## Security and latency

The app and companion communicate only over a per-user Unix socket. Every NDJSON action carries a 64-character bearer token, a Unix millisecond timestamp, a monotonic sequence number, a finite bounded value, and an allowlisted action. `PROTOCOL.md` defines the receiver invariants.

Hardware callbacks never perform file or socket I/O. They insert into a fixed 64-item queue and return. Socket work has a 100 ms sender deadline and a 150 ms receiver deadline, actions are never retried, and overload is observable through dispatcher counters rather than causing Options+ input lag.
