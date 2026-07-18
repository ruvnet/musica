using MusicaVj.Logitech.Core;

namespace Loupedeck.MusicaVjPlugin;

public abstract class MusicaCommand : PluginDynamicCommand
{
    private readonly string _action;
    private readonly string _buttonLabel;

    protected MusicaCommand(
        string displayName,
        string description,
        string groupName,
        string action,
        string buttonLabel)
        : base(displayName, description, groupName)
    {
        if (!ControlActions.IsAllowed(action))
        {
            throw new ArgumentOutOfRangeException(nameof(action));
        }

        _action = action;
        _buttonLabel = buttonLabel;
    }

    protected override void RunCommand(string actionParameter) => ControllerBridge.TrySend(_action);

    protected override BitmapImage GetCommandImage(string actionParameter, PluginImageSize imageSize)
    {
        using var bitmapBuilder = new BitmapBuilder(imageSize);
        bitmapBuilder.DrawText(_buttonLabel);
        return bitmapBuilder.ToImage();
    }
}
