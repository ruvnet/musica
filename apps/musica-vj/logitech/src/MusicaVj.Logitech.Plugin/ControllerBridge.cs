using MusicaVj.Logitech.Core;

namespace Loupedeck.MusicaVjPlugin;

internal static class ControllerBridge
{
    private static readonly object Gate = new();
    private static UnixSocketDispatcher? _dispatcher;

    public static void Initialize()
    {
        lock (Gate)
        {
            _dispatcher ??= new UnixSocketDispatcher();
        }
    }

    public static bool TrySend(string action, double value = 0)
    {
        try
        {
            Initialize();
            return _dispatcher?.TrySend(action, value) == true;
        }
        catch (Exception)
        {
            // Options+ action callbacks must never fail because the app is closed or unready.
            return false;
        }
    }

    public static void Shutdown()
    {
        lock (Gate)
        {
            _dispatcher?.Dispose();
            _dispatcher = null;
        }
    }
}
