import torch

from model_hybrid import (
    ChannelGuidedTemporalAttention,
    StatisticalSLS,
    SwiGLUGate,
    TemporalAttentionPooling,
)


def main():
    batch_size = 2
    num_layers = 24
    time_steps = 201
    feature_dim = 1024
    hidden_dim = 64

    hidden_states = torch.randn(batch_size, num_layers, time_steps, feature_dim)

    sls = StatisticalSLS(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        use_std=True,
    )
    fused, layer_weights = sls(hidden_states)
    assert fused.shape == (batch_size, time_steps, feature_dim)
    assert layer_weights.shape == (batch_size, num_layers)

    swiglu = SwiGLUGate(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
    )
    refined = swiglu(fused)
    assert refined.shape == fused.shape

    temporal_pooling = TemporalAttentionPooling(
        feature_dim=feature_dim,
        attention_dim=hidden_dim,
    )
    pooled, temporal_weights = temporal_pooling(refined)
    assert pooled.shape == (batch_size, feature_dim)
    assert temporal_weights.shape == (batch_size, time_steps)
    assert torch.allclose(
        temporal_weights.sum(dim=1),
        torch.ones(batch_size),
        atol=1e-5,
    )

    cgta = ChannelGuidedTemporalAttention(
        feature_dim=feature_dim,
        attention_dim=hidden_dim,
        use_std=True,
        use_stat_residual=True,
    )
    pooled, temporal_weights, channel_context = cgta(refined)
    assert pooled.shape == (batch_size, feature_dim)
    assert temporal_weights.shape == (batch_size, time_steps)
    assert channel_context.shape == (batch_size, hidden_dim)
    assert torch.allclose(
        temporal_weights.sum(dim=1),
        torch.ones(batch_size),
        atol=1e-5,
    )

    print("All hybrid module tests passed.")


if __name__ == "__main__":
    main()
