using MusicaVj.Logitech.Core;

namespace MusicaVj.Logitech.Core.Tests;

public sealed class BoundedActionQueueTests
{
    [Fact]
    public void FullQueue_RejectsNewItemWithoutReplacingAcceptedActions()
    {
        var queue = new BoundedActionQueue<string>(capacity: 2);

        Assert.True(queue.TryEnqueue("first"));
        Assert.True(queue.TryEnqueue("second"));
        Assert.False(queue.TryEnqueue("third"));
        Assert.Equal(2, queue.Count);
        Assert.Equal(1, queue.Dropped);

        Assert.True(queue.TryDequeue(out var first));
        Assert.True(queue.TryDequeue(out var second));
        Assert.Equal("first", first);
        Assert.Equal("second", second);
        Assert.False(queue.TryDequeue(out _));
    }

    [Fact]
    public void Constructor_RejectsNonPositiveCapacity()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new BoundedActionQueue<int>(0));
    }
}
