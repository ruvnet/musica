using System.Text.Json.Serialization;

namespace MusicaVj.Logitech.Core;

public sealed record ControlEnvelope
{
    [JsonPropertyName("v")]
    [JsonPropertyOrder(0)]
    public int Version { get; init; } = 1;

    [JsonPropertyName("seq")]
    [JsonPropertyOrder(1)]
    public required long Sequence { get; init; }

    [JsonPropertyName("ts_ms")]
    [JsonPropertyOrder(2)]
    public required long UnixTimeMilliseconds { get; init; }

    [JsonPropertyName("action")]
    [JsonPropertyOrder(3)]
    public required string Action { get; init; }

    [JsonPropertyName("value")]
    [JsonPropertyOrder(4)]
    public required double Value { get; init; }

    [JsonPropertyName("token")]
    [JsonPropertyOrder(5)]
    public required string Token { get; init; }
}
