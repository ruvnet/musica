using MusicaVj.Logitech.Core;

namespace Loupedeck.MusicaVjPlugin;

public sealed class TrackPreviousCommand : MusicaCommand
{
    public TrackPreviousCommand()
        : base("Previous Track", "Selects the previous track", "Tracks", ControlActions.TrackPrevious, "TRK <")
    {
    }
}

public sealed class TrackNextCommand : MusicaCommand
{
    public TrackNextCommand()
        : base("Next Track", "Selects the next track", "Tracks", ControlActions.TrackNext, "TRK >")
    {
    }
}

public sealed class TrackMuteCommand : MusicaCommand
{
    public TrackMuteCommand()
        : base("Mute Track", "Toggles mute on the selected track", "Tracks", ControlActions.TrackMute, "MUTE")
    {
    }
}

public sealed class TrackSoloCommand : MusicaCommand
{
    public TrackSoloCommand()
        : base("Solo Track", "Toggles solo on the selected track", "Tracks", ControlActions.TrackSolo, "SOLO")
    {
    }
}

public sealed class TrackTriggerCommand : PluginDynamicCommand
{
    private static readonly string[] TrackNames = ["Drums", "Bass", "Chords", "Lead", "Voice", "Texture"];

    public TrackTriggerCommand()
        : base()
    {
        for (var index = 0; index < TrackNames.Length; index++)
        {
            AddParameter(index.ToString(), $"Trigger {TrackNames[index]}", "Tracks###Trigger");
        }
    }

    protected override void RunCommand(string actionParameter)
    {
        if (int.TryParse(actionParameter, out var trackIndex) && trackIndex >= 0 && trackIndex < TrackNames.Length)
        {
            ControllerBridge.TrySend(ControlActions.TrackTrigger, trackIndex);
        }
    }

    protected override string? GetCommandDisplayName(string actionParameter, PluginImageSize imageSize) =>
        int.TryParse(actionParameter, out var trackIndex) && trackIndex >= 0 && trackIndex < TrackNames.Length
            ? $"Trigger {TrackNames[trackIndex]}"
            : null;

    protected override BitmapImage GetCommandImage(string actionParameter, PluginImageSize imageSize)
    {
        var label = int.TryParse(actionParameter, out var trackIndex) && trackIndex >= 0 && trackIndex < TrackNames.Length
            ? TrackNames[trackIndex].ToUpperInvariant()
            : "HIT";
        using var bitmapBuilder = new BitmapBuilder(imageSize);
        bitmapBuilder.DrawText(label);
        return bitmapBuilder.ToImage();
    }
}
