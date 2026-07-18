using MusicaVj.Logitech.Core;

namespace MusicaVj.Logitech.Core.Tests;

public sealed class ControlActionsTests
{
    [Fact]
    public void Allowlist_HasOnlyTheProtocolActions()
    {
        Assert.Equal(17, ControlActions.Allowed.Count);
        var allowed = ControlActions.Allowed.ToArray();
        Assert.Contains(ControlActions.TransportToggle, allowed);
        Assert.Contains(ControlActions.TrackTrigger, allowed);
        Assert.Contains(ControlActions.VisualIntensityDelta, allowed);
        Assert.DoesNotContain("transport.stop", allowed);
    }
}
