using MusicaVj.Logitech.Core;

namespace Loupedeck.MusicaVjPlugin;

public sealed class VisualPreviousCommand : MusicaCommand
{
    public VisualPreviousCommand()
        : base("Previous Visual", "Selects the previous visual scene", "Visuals", ControlActions.VisualPrevious, "FX <")
    {
    }
}

public sealed class VisualNextCommand : MusicaCommand
{
    public VisualNextCommand()
        : base("Next Visual", "Selects the next visual scene", "Visuals", ControlActions.VisualNext, "FX >")
    {
    }
}
