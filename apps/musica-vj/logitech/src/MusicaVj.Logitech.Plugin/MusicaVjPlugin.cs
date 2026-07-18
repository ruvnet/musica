namespace Loupedeck.MusicaVjPlugin;

public sealed class MusicaVjPlugin : Plugin
{
    public override bool UsesApplicationApiOnly => true;

    public override bool HasNoApplication => false;

    public override void Load() => ControllerBridge.Initialize();

    public override void Unload() => ControllerBridge.Shutdown();
}
