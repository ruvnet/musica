using MusicaVj.Logitech.Core;

namespace Loupedeck.MusicaVjPlugin;

public abstract class MusicaAdjustment : PluginDynamicAdjustment
{
    private readonly string _action;
    private int _accumulatedDelta;

    protected MusicaAdjustment(string displayName, string description, string groupName, string action)
        : base(displayName, description, groupName, hasReset: true)
    {
        if (!ControlActions.IsAllowed(action))
        {
            throw new ArgumentOutOfRangeException(nameof(action));
        }

        _action = action;
    }

    protected override void ApplyAdjustment(string actionParameter, int diff)
    {
        if (diff == 0)
        {
            return;
        }

        var direction = Math.Sign(diff);
        var ticks = Math.Min(Math.Abs((long)diff), 64L);
        var accepted = 0;
        for (var index = 0L; index < ticks; index++)
        {
            if (!ControllerBridge.TrySend(_action, direction))
            {
                break;
            }

            accepted++;
        }

        _accumulatedDelta = Math.Clamp(_accumulatedDelta + (accepted * direction), -999, 999);
        AdjustmentValueChanged();
    }

    protected override void RunCommand(string actionParameter)
    {
        _accumulatedDelta = 0;
        AdjustmentValueChanged();
    }

    protected override string GetAdjustmentValue(string actionParameter) =>
        _accumulatedDelta > 0 ? $"+{_accumulatedDelta}" : _accumulatedDelta.ToString();
}
