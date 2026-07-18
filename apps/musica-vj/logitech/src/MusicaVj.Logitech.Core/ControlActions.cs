using System.Collections.Frozen;

namespace MusicaVj.Logitech.Core;

public static class ControlActions
{
    public const string TransportToggle = "transport.toggle";
    public const string TransportRecord = "transport.record";
    public const string TrackPrevious = "track.previous";
    public const string TrackNext = "track.next";
    public const string TrackMute = "track.mute";
    public const string TrackSolo = "track.solo";
    public const string TrackTrigger = "track.trigger";
    public const string VisualPrevious = "visual.previous";
    public const string VisualNext = "visual.next";
    public const string TempoTap = "tempo.tap";
    public const string MasterDelta = "master.delta";
    public const string VisualIntensityDelta = "visual.intensity.delta";
    public const string VisualSculptureDelta = "visual.sculpture.delta";
    public const string VisualMotionDelta = "visual.motion.delta";
    public const string VisualAtmosphereDelta = "visual.atmosphere.delta";
    public const string VisualRibbonDelta = "visual.ribbon.delta";
    public const string TempoDelta = "tempo.delta";

    public static readonly FrozenSet<string> Allowed = new[]
    {
        TransportToggle,
        TransportRecord,
        TrackPrevious,
        TrackNext,
        TrackMute,
        TrackSolo,
        TrackTrigger,
        VisualPrevious,
        VisualNext,
        TempoTap,
        MasterDelta,
        VisualIntensityDelta,
        VisualSculptureDelta,
        VisualMotionDelta,
        VisualAtmosphereDelta,
        VisualRibbonDelta,
        TempoDelta,
    }.ToFrozenSet(StringComparer.Ordinal);

    private static readonly FrozenSet<string> DeltaActions = new[]
    {
        MasterDelta,
        VisualIntensityDelta,
        VisualSculptureDelta,
        VisualMotionDelta,
        VisualAtmosphereDelta,
        VisualRibbonDelta,
        TempoDelta,
    }.ToFrozenSet(StringComparer.Ordinal);

    public static bool IsAllowed(string? action) => action is not null && Allowed.Contains(action);

    public static bool IsValueAllowed(string? action, double value)
    {
        if (!IsAllowed(action) || !double.IsFinite(value))
        {
            return false;
        }

        if (action == TrackTrigger)
        {
            return value is >= 0 and <= 5 && value == Math.Truncate(value);
        }

        return DeltaActions.Contains(action!) ? value is >= -1 and <= 1 : value == 0;
    }
}
