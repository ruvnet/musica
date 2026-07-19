using System.Collections.Generic;
using System.Threading;

namespace MusicaVj.Logitech.Core;

/// <summary>
/// A bounded FIFO whose producer path never waits for the consumer or for a lock.
/// Contended and full writes are rejected so hardware callbacks remain responsive.
/// </summary>
public sealed class BoundedActionQueue<T>
{
    private readonly object _gate = new();
    private readonly Queue<T> _items;
    private long _dropped;

    public BoundedActionQueue(int capacity)
    {
        if (capacity <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity));
        }

        Capacity = capacity;
        _items = new Queue<T>(capacity);
    }

    public int Capacity { get; }

    public long Dropped => Interlocked.Read(ref _dropped);

    public int Count
    {
        get
        {
            lock (_gate)
            {
                return _items.Count;
            }
        }
    }

    public bool TryEnqueue(T item)
    {
        if (!Monitor.TryEnter(_gate))
        {
            Interlocked.Increment(ref _dropped);
            return false;
        }

        try
        {
            if (_items.Count >= Capacity)
            {
                Interlocked.Increment(ref _dropped);
                return false;
            }

            _items.Enqueue(item);
            return true;
        }
        finally
        {
            Monitor.Exit(_gate);
        }
    }

    public bool TryDequeue(out T? item)
    {
        // Only the hardware callback is latency sensitive. The single
        // background consumer may wait for the producer's very short critical
        // section; returning on contention could strand an item until the
        // worker's next 250 ms wake-up.
        lock (_gate)
        {
            if (_items.Count == 0)
            {
                item = default;
                return false;
            }

            item = _items.Dequeue();
            return true;
        }
    }
}
