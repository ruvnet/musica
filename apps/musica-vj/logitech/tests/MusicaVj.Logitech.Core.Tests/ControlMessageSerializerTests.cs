using System.Text.Json;
using MusicaVj.Logitech.Core;

namespace MusicaVj.Logitech.Core.Tests;

public sealed class ControlMessageSerializerTests
{
    private const string ValidToken = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    [Fact]
    public void Serialize_EmitsStrictNdjsonEnvelope()
    {
        var serializer = new ControlMessageSerializer();

        var line = serializer.Serialize(
            ControlActions.VisualIntensityDelta,
            -1,
            ValidToken,
            sequence: 42,
            unixTimeMilliseconds: 1_721_337_600_123);

        Assert.EndsWith("\n", line);
        Assert.DoesNotContain("\r", line);

        using var document = JsonDocument.Parse(line);
        var root = document.RootElement;
        Assert.Equal(6, root.EnumerateObject().Count());
        Assert.Equal(1, root.GetProperty("v").GetInt32());
        Assert.Equal(42L, root.GetProperty("seq").GetInt64());
        Assert.Equal(1_721_337_600_123, root.GetProperty("ts_ms").GetInt64());
        Assert.Equal(ControlActions.VisualIntensityDelta, root.GetProperty("action").GetString());
        Assert.Equal(-1d, root.GetProperty("value").GetDouble());
        Assert.Equal(ValidToken, root.GetProperty("token").GetString());
    }

    [Theory]
    [InlineData("shell.exec")]
    [InlineData("")]
    [InlineData("TRACK.NEXT")]
    public void Serialize_RejectsActionsOutsideAllowlist(string action)
    {
        var serializer = new ControlMessageSerializer();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            serializer.Serialize(action, 0, ValidToken, 1, 1));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(1.01)]
    [InlineData(-1.01)]
    public void Serialize_RejectsInvalidValues(double value)
    {
        var serializer = new ControlMessageSerializer();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            serializer.Serialize(ControlActions.TempoDelta, value, ValidToken, 1, 1));
    }

    [Theory]
    [InlineData(ControlActions.VisualSculptureDelta)]
    [InlineData(ControlActions.VisualMotionDelta)]
    [InlineData(ControlActions.VisualAtmosphereDelta)]
    [InlineData(ControlActions.VisualRibbonDelta)]
    public void Serialize_AllowsBoundedArtistMacroDeltas(string action)
    {
        var serializer = new ControlMessageSerializer();

        var line = serializer.Serialize(action, 1, ValidToken, 1, 1);

        using var document = JsonDocument.Parse(line);
        Assert.Equal(action, document.RootElement.GetProperty("action").GetString());
        Assert.Equal(1d, document.RootElement.GetProperty("value").GetDouble());
    }

    [Theory]
    [InlineData(6)]
    [InlineData(-1)]
    [InlineData(1.5)]
    public void Serialize_RejectsInvalidTrackIndex(double value)
    {
        var serializer = new ControlMessageSerializer();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            serializer.Serialize(ControlActions.TrackTrigger, value, ValidToken, 1, 1));
    }

    [Fact]
    public void Serialize_RejectsNonzeroValueForDiscreteAction()
    {
        var serializer = new ControlMessageSerializer();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            serializer.Serialize(ControlActions.TransportToggle, 1, ValidToken, 1, 1));
    }

    [Fact]
    public void Serialize_RejectsTokenThatIsNotLowercaseHex64()
    {
        var serializer = new ControlMessageSerializer();

        Assert.Throws<ArgumentException>(() =>
            serializer.Serialize(ControlActions.TempoTap, 0, ValidToken.ToUpperInvariant(), 1, 1));
    }
}
