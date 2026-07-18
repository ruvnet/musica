using MusicaVj.Logitech.Core;

namespace Loupedeck.MusicaVjPlugin;

public sealed class MasterLevelAdjustment : MusicaAdjustment
{
    public MasterLevelAdjustment()
        : base("Master Level", "Adjusts the master output level", "Mix", ControlActions.MasterDelta)
    {
    }
}

public sealed class VisualIntensityAdjustment : MusicaAdjustment
{
    public VisualIntensityAdjustment()
        : base("Visual Intensity", "Adjusts audio reactive visual intensity", "Visuals", ControlActions.VisualIntensityDelta)
    {
    }
}

public sealed class VisualSculptureAdjustment : MusicaAdjustment
{
    public VisualSculptureAdjustment()
        : base("Sculpture", "Shapes the spectral sculpture and frequency relief", "Artist Macros", ControlActions.VisualSculptureDelta)
    {
    }
}

public sealed class VisualMotionAdjustment : MusicaAdjustment
{
    public VisualMotionAdjustment()
        : base("Motion", "Shapes visual drift and camera movement", "Artist Macros", ControlActions.VisualMotionDelta)
    {
    }
}

public sealed class VisualAtmosphereAdjustment : MusicaAdjustment
{
    public VisualAtmosphereAdjustment()
        : base("Atmosphere", "Shapes particles and ultraviolet haze", "Artist Macros", ControlActions.VisualAtmosphereDelta)
    {
    }
}

public sealed class VisualRibbonAdjustment : MusicaAdjustment
{
    public VisualRibbonAdjustment()
        : base("Ribbon", "Shapes waveform excursion, glow, and reflection", "Artist Macros", ControlActions.VisualRibbonDelta)
    {
    }
}

public sealed class TempoAdjustment : MusicaAdjustment
{
    public TempoAdjustment()
        : base("Tempo", "Adjusts the performance tempo", "Transport", ControlActions.TempoDelta)
    {
    }
}
