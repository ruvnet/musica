using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace MusicaVj.Logitech.Core;

public sealed class UnixSocketDispatcher : IDisposable
{
    public const int DefaultQueueCapacity = 64;
    public const int DefaultOperationTimeoutMilliseconds = 100;

    private readonly BoundedActionQueue<PendingAction> _queue;
    private readonly TokenFileProvider _tokenProvider;
    private readonly ControlMessageSerializer _serializer = new();
    private readonly AutoResetEvent _workAvailable = new(initialState: false);
    private readonly CancellationTokenSource _cancellation = new();
    private readonly Task _worker;
    private readonly int _operationTimeoutMilliseconds;
    private long _sequence;
    private long _sent;
    private long _failed;
    private int _disposed;

    public UnixSocketDispatcher(
        string? socketPath = null,
        string? tokenPath = null,
        int queueCapacity = DefaultQueueCapacity,
        int operationTimeoutMilliseconds = DefaultOperationTimeoutMilliseconds)
    {
        if (operationTimeoutMilliseconds is < 10 or > 2_000)
        {
            throw new ArgumentOutOfRangeException(nameof(operationTimeoutMilliseconds));
        }

        SocketPath = socketPath ?? GetDefaultSocketPath();
        TokenPath = tokenPath ?? GetDefaultTokenPath();
        _operationTimeoutMilliseconds = operationTimeoutMilliseconds;
        _queue = new BoundedActionQueue<PendingAction>(queueCapacity);
        _tokenProvider = new TokenFileProvider(TokenPath);
        _sequence = CreateSequenceBase();
        _worker = Task.Run(ProcessQueueAsync);
    }

    public string SocketPath { get; }

    public string TokenPath { get; }

    public long Sent => Interlocked.Read(ref _sent);

    public long Failed => Interlocked.Read(ref _failed);

    public long Dropped => _queue.Dropped;

    public bool TrySend(string action, double value = 0)
    {
        if (Volatile.Read(ref _disposed) != 0
            || !ControlActions.IsAllowed(action)
            || !ControlActions.IsValueAllowed(action, value))
        {
            return false;
        }

        var accepted = _queue.TryEnqueue(new PendingAction(action, value));
        if (accepted)
        {
            try
            {
                _workAvailable.Set();
            }
            catch (ObjectDisposedException)
            {
                return false;
            }
        }

        return accepted;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _cancellation.Cancel();
        _workAvailable.Set();

        var stopped = false;
        try
        {
            stopped = _worker.Wait(millisecondsTimeout: 250);
        }
        catch (AggregateException)
        {
            // The plugin is unloading. Transport failures are intentionally contained.
        }

        if (stopped)
        {
            _workAvailable.Dispose();
            _cancellation.Dispose();
        }
    }

    // The socket used to live at /tmp/musica-vj-$USER.sock. /tmp is
    // world-traversable and the name was guessable, so it now sits beside the
    // token in the application data directory, which is already per-user.
    public static string GetDefaultSocketPath() =>
        Path.Combine(GetApplicationDataDirectory(), "controller.sock");

    public static string GetDefaultTokenPath() =>
        Path.Combine(GetApplicationDataDirectory(), "controller.token");

    private static string GetApplicationDataDirectory()
    {
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (string.IsNullOrWhiteSpace(home))
        {
            throw new InvalidOperationException("Cannot locate the current user's home directory.");
        }

        return Path.Combine(
            home,
            "Library",
            "Application Support",
            "one.cognitum.musica.vj");
    }

    private static long CreateSequenceBase()
    {
        const long scale = 1_000_000;
        var unixMilliseconds = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        return Math.Min(unixMilliseconds, long.MaxValue / scale) * scale;
    }

    private async Task ProcessQueueAsync()
    {
        var cancellationToken = _cancellation.Token;

        while (!cancellationToken.IsCancellationRequested)
        {
            _workAvailable.WaitOne(millisecondsTimeout: 250);

            while (!cancellationToken.IsCancellationRequested && _queue.TryDequeue(out var pending))
            {
                if (pending is null || !_tokenProvider.TryRead(out var token))
                {
                    Interlocked.Increment(ref _failed);
                    continue;
                }

                try
                {
                    var sequence = Interlocked.Increment(ref _sequence);
                    var line = _serializer.Serialize(
                        pending.Action,
                        pending.Value,
                        token,
                        sequence,
                        DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());
                    await SendLineAsync(line, cancellationToken).ConfigureAwait(false);
                    Interlocked.Increment(ref _sent);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception exception) when (
                    exception is SocketException
                    or IOException
                    or OperationCanceledException
                    or ObjectDisposedException)
                {
                    Interlocked.Increment(ref _failed);
                }
            }
        }
    }

    private async Task SendLineAsync(string line, CancellationToken cancellationToken)
    {
        using var operation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        operation.CancelAfter(_operationTimeoutMilliseconds);

        using var socket = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        await socket.ConnectAsync(new UnixDomainSocketEndPoint(SocketPath), operation.Token).ConfigureAwait(false);

        var bytes = Encoding.UTF8.GetBytes(line);
        var sent = 0;
        while (sent < bytes.Length)
        {
            var count = await socket.SendAsync(
                bytes.AsMemory(sent),
                SocketFlags.None,
                operation.Token).ConfigureAwait(false);
            if (count == 0)
            {
                throw new IOException("Controller socket closed before the message was sent.");
            }

            sent += count;
        }
    }

    private sealed record PendingAction(string Action, double Value);
}
