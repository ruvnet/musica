using MusicaVj.Logitech.Core;

namespace Loupedeck.MusicaVjPlugin;

public sealed class TransportToggleCommand : MusicaCommand
{
    public TransportToggleCommand()
        : base("Play / Pause", "Toggles Musica VJ playback", "Transport", ControlActions.TransportToggle, "PLAY")
    {
    }
}

public sealed class TransportRecordCommand : MusicaCommand
{
    public TransportRecordCommand()
        : base("Record", "Starts or stops the social video recorder", "Transport", ControlActions.TransportRecord, "REC")
    {
    }
}

public sealed class TapTempoCommand : MusicaCommand
{
    public TapTempoCommand()
        : base("Tap Tempo", "Taps the performance tempo", "Transport", ControlActions.TempoTap, "TAP")
    {
    }
}
