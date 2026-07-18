using System.Text.Json;

namespace MusicaVj.Logitech.Core;

public sealed class ControlMessageSerializer
{
    public const int ProtocolVersion = 1;
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = false,
    };

    public string Serialize(
        string action,
        double value,
        string token,
        long sequence,
        long unixTimeMilliseconds)
    {
        if (!ControlActions.IsAllowed(action))
        {
            throw new ArgumentOutOfRangeException(nameof(action), action, "Action is not in the controller allowlist.");
        }

        if (!ControlActions.IsValueAllowed(action, value))
        {
            throw new ArgumentOutOfRangeException(nameof(value), value, "Value is invalid for this action.");
        }

        if (sequence <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequence));
        }

        if (unixTimeMilliseconds < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(unixTimeMilliseconds));
        }

        if (!ControllerToken.IsValid(token))
        {
            throw new ArgumentException("Controller token is invalid.", nameof(token));
        }

        var envelope = new ControlEnvelope
        {
            Version = ProtocolVersion,
            Sequence = sequence,
            UnixTimeMilliseconds = unixTimeMilliseconds,
            Action = action,
            Value = value,
            Token = token,
        };

        return JsonSerializer.Serialize(envelope, Options) + "\n";
    }
}
