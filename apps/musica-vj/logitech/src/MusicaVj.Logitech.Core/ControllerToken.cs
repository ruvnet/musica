namespace MusicaVj.Logitech.Core;

public static class ControllerToken
{
    public const int Length = 64;

    public static bool IsValid(string? token)
    {
        if (token is null || token.Length != Length)
        {
            return false;
        }

        foreach (var character in token)
        {
            if (character is not (>= '0' and <= '9') and not (>= 'a' and <= 'f'))
            {
                return false;
            }
        }

        return true;
    }
}
