using System.Text;

namespace MusicaVj.Logitech.Core;

public sealed class TokenFileProvider
{
    private const long MaximumFileBytes = 1_024;

    public TokenFileProvider(string path)
    {
        Path = path ?? throw new ArgumentNullException(nameof(path));
    }

    public string Path { get; }

    public bool TryRead(out string token)
    {
        token = string.Empty;

        try
        {
            var info = new FileInfo(Path);
            if (!info.Exists || info.Length is <= 0 or > MaximumFileBytes || info.LinkTarget is not null)
            {
                return false;
            }

            if (!HasPrivateUnixPermissions(Path))
            {
                return false;
            }

            using var stream = new FileStream(
                Path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 512,
                FileOptions.SequentialScan);
            using var reader = new StreamReader(
                stream,
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true),
                detectEncodingFromByteOrderMarks: false,
                bufferSize: 512,
                leaveOpen: false);

            token = reader.ReadToEnd().TrimEnd('\r', '\n');
            return ControllerToken.IsValid(token);
        }
        catch (Exception exception) when (
            exception is IOException
            or UnauthorizedAccessException
            or System.Security.SecurityException
            or DecoderFallbackException)
        {
            token = string.Empty;
            return false;
        }
    }

    private static bool HasPrivateUnixPermissions(string path)
    {
        if (OperatingSystem.IsWindows())
        {
            return true;
        }

        try
        {
            const UnixFileMode exposed =
                UnixFileMode.GroupRead
                | UnixFileMode.GroupWrite
                | UnixFileMode.GroupExecute
                | UnixFileMode.OtherRead
                | UnixFileMode.OtherWrite
                | UnixFileMode.OtherExecute;
            return (File.GetUnixFileMode(path) & exposed) == 0;
        }
        catch (PlatformNotSupportedException)
        {
            return true;
        }
    }
}
