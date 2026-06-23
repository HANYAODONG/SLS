import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq


class SSLModel(nn.Module):
    def __init__(self, cp_path, device):
        super(SSLModel, self).__init__()
        if not os.path.isfile(cp_path):
            raise FileNotFoundError(
                "XLS-R checkpoint not found: {}. Download xlsr2_300m.pt and "
                "place it in the project root, or pass --xlsr_checkpoint."
                .format(cp_path)
            )
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        if (
            next(self.model.parameters()).device != input_data.device
            or next(self.model.parameters()).dtype != input_data.dtype
        ):
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        output = self.model(input_tmp, mask=False, features_only=True)
        return output["x"], output["layer_results"]


def get_layer_features(layer_result):
    pooled_layers = []
    full_layers = []
    for layer in layer_result:
        layer_y = layer[0].transpose(0, 1).transpose(1, 2)
        layer_y = F.adaptive_avg_pool1d(layer_y, 1)
        layer_y = layer_y.transpose(1, 2)
        pooled_layers.append(layer_y)

        x = layer[0].transpose(0, 1)
        x = x.view(x.size(0), -1, x.size(1), x.size(2))
        full_layers.append(x)

    pooled = torch.cat(pooled_layers, dim=1)
    full = torch.cat(full_layers, dim=1)
    return pooled, full


def apply_layer_mask(layer_weights, layer_mask):
    """Mask layer weights and preserve each sample's original weight sum."""
    if layer_mask is None:
        return layer_weights
    if layer_mask.dim() == 1:
        layer_mask = layer_mask.unsqueeze(0)
    layer_mask = layer_mask.to(device=layer_weights.device, dtype=layer_weights.dtype)
    if layer_mask.shape[0] == 1 and layer_weights.shape[0] > 1:
        layer_mask = layer_mask.expand(layer_weights.shape[0], -1)
    if layer_mask.shape != layer_weights.shape:
        raise ValueError(
            "layer_mask shape {} does not match layer_weights shape {}".format(
                tuple(layer_mask.shape),
                tuple(layer_weights.shape),
            )
        )
    if torch.any(layer_mask.sum(dim=1) <= 0):
        raise ValueError("layer_mask cannot disable all layers")

    masked_weights = layer_weights * layer_mask
    original_sum = layer_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    masked_sum = masked_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return masked_weights * (original_sum / masked_sum)


def fuse_with_layer_weights(full_features, layer_weights):
    return torch.sum(full_features * layer_weights.unsqueeze(2).unsqueeze(3), dim=1)


class StatisticalSLS(nn.Module):
    """Mean+Std layer weighting for XLS-R hidden states.

    Input shape: [B, L, T, D].
    Output shape: fused [B, T, D], weights [B, L].
    """

    def __init__(self, feature_dim=1024, hidden_dim=128, use_std=True, dropout=0.1):
        super(StatisticalSLS, self).__init__()
        stat_dim = feature_dim * 2 if use_std else feature_dim
        self.use_std = use_std
        self.weight_predictor = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.SELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        mean = torch.mean(x, dim=2)
        if self.use_std:
            variance = torch.var(x, dim=2, unbiased=False)
            std = torch.sqrt(torch.clamp(variance, min=1e-5))
            statistics = torch.cat([mean, std], dim=-1)
        else:
            statistics = mean

        logits = self.weight_predictor(statistics)
        weights = torch.sigmoid(logits)
        fused = torch.sum(x * weights.unsqueeze(2), dim=1)
        return fused, weights.squeeze(-1)


class SwiGLUGate(nn.Module):
    """Feature gate after layer fusion. Input/output shape: [B, T, D]."""

    def __init__(self, input_dim=1024, hidden_dim=128, dropout=0.1):
        super(SwiGLUGate, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        gate = F.silu(self.gate_proj(x))
        value = self.value_proj(x)
        x = self.out_proj(gate * value)
        return residual + self.dropout(x)


class TemporalAttentionPooling(nn.Module):
    """Plain temporal attention pooling. Input [B, T, D], output [B, D]."""

    def __init__(self, feature_dim=1024, attention_dim=128, dropout=0.1):
        super(TemporalAttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled, weights.squeeze(-1)


class ChannelGuidedTemporalAttention(nn.Module):
    """Channel-guided temporal attention from the second implementation note."""

    def __init__(
        self,
        feature_dim=1024,
        attention_dim=128,
        dropout=0.1,
        use_std=True,
        use_stat_residual=True,
    ):
        super(ChannelGuidedTemporalAttention, self).__init__()
        self.use_std = use_std
        self.use_stat_residual = use_stat_residual
        statistic_dim = feature_dim * 2 if use_std else feature_dim

        self.channel_context_net = nn.Sequential(
            nn.Linear(statistic_dim, attention_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.temporal_projection = nn.Linear(feature_dim, attention_dim)
        self.channel_projection = nn.Linear(attention_dim, attention_dim)
        self.score_projection = nn.Linear(attention_dim, 1)
        self.statistics_projection = nn.Sequential(
            nn.Linear(statistic_dim, attention_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, feature_dim),
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        mean = x.mean(dim=1)
        if self.use_std:
            variance = torch.var(x, dim=1, unbiased=False)
            std = torch.sqrt(torch.clamp(variance, min=1e-5))
            statistics = torch.cat([mean, std], dim=-1)
        else:
            statistics = mean

        channel_context = self.channel_context_net(statistics)
        temporal_feature = self.temporal_projection(x)
        channel_feature = self.channel_projection(channel_context).unsqueeze(1)
        joint_feature = torch.tanh(temporal_feature + channel_feature)
        scores = self.score_projection(joint_feature)
        temporal_weights = torch.softmax(scores, dim=1)
        temporal_pooled = torch.sum(temporal_weights * x, dim=1)

        if self.use_stat_residual:
            statistics_feature = self.statistics_projection(statistics)
            pooled = self.output_norm(temporal_pooled + statistics_feature)
        else:
            pooled = self.output_norm(temporal_pooled)

        return pooled, temporal_weights.squeeze(-1), channel_context


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.device = device
        self.ssl_model = SSLModel(args.xlsr_checkpoint, self.device)
        self.use_stat_sls = bool(int(getattr(args, "use_stat_sls", 1)))
        self.stat_sls_use_std = bool(int(getattr(args, "stat_sls_use_std", 1)))
        self.use_swiglu = bool(int(getattr(args, "use_swiglu", 1)))
        self.pooling_type = getattr(args, "pooling_type", "cgta")
        self.cgta_use_std = bool(int(getattr(args, "cgta_use_std", 1)))
        self.cgta_stat_residual = bool(int(getattr(args, "cgta_stat_residual", 1)))
        hybrid_hidden_dim = int(getattr(args, "hybrid_hidden_dim", 128))
        hybrid_dropout = float(getattr(args, "hybrid_dropout", 0.1))

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.stat_sls = StatisticalSLS(
            feature_dim=1024,
            hidden_dim=hybrid_hidden_dim,
            use_std=self.stat_sls_use_std,
            dropout=hybrid_dropout,
        )
        self.swiglu = SwiGLUGate(
            input_dim=1024,
            hidden_dim=hybrid_hidden_dim,
            dropout=hybrid_dropout,
        )
        self.temporal_attention = TemporalAttentionPooling(
            feature_dim=1024,
            attention_dim=hybrid_hidden_dim,
            dropout=hybrid_dropout,
        )
        self.cgta_pooling = ChannelGuidedTemporalAttention(
            feature_dim=1024,
            attention_dim=hybrid_hidden_dim,
            dropout=hybrid_dropout,
            use_std=self.cgta_use_std,
            use_stat_residual=self.cgta_stat_residual,
        )
        self.hybrid_classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.SELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

        self.last_layer_weights = None
        self.last_temporal_weights = None
        self.last_channel_context = None

    def original_sls_fusion(self, pooled_layers, full_features):
        y0 = self.fc0(pooled_layers)
        y0 = self.sig(y0)
        layer_weights = y0.squeeze(2).squeeze(-1)
        fused = fuse_with_layer_weights(full_features, layer_weights)
        return fused, layer_weights

    def original_head(self, fused):
        x = fused.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.selu(x)
        pooled_embedding = x
        x = self.fc3(x)
        x = self.selu(x)
        return self.logsoftmax(x), pooled_embedding

    def attention_head(self, fused):
        if self.pooling_type == "temporal":
            pooled, temporal_weights = self.temporal_attention(fused)
            channel_context = None
        elif self.pooling_type == "cgta":
            pooled, temporal_weights, channel_context = self.cgta_pooling(fused)
        else:
            pooled = torch.max(fused, dim=1)[0]
            temporal_weights = None
            channel_context = None

        logits = self.hybrid_classifier(pooled)
        self.last_temporal_weights = (
            temporal_weights.detach() if temporal_weights is not None else None
        )
        self.last_channel_context = (
            channel_context.detach() if channel_context is not None else None
        )
        return self.logsoftmax(logits), pooled

    def forward(self, x, return_details=False, layer_mask=None):
        _, layer_result = self.ssl_model.extract_feat(x.squeeze(-1))
        pooled_layers, full_features = get_layer_features(layer_result)

        if self.use_stat_sls:
            _, layer_weights = self.stat_sls(full_features)
        else:
            _, layer_weights = self.original_sls_fusion(pooled_layers, full_features)

        original_layer_weights = layer_weights
        effective_layer_weights = apply_layer_mask(layer_weights, layer_mask)
        fused = fuse_with_layer_weights(full_features, effective_layer_weights)

        if self.use_swiglu:
            fused = self.swiglu(fused)

        self.last_layer_weights = original_layer_weights.detach()
        self.last_temporal_weights = None
        self.last_channel_context = None

        if self.pooling_type in ("temporal", "cgta"):
            output, pooled_embedding = self.attention_head(fused)
        elif self.pooling_type == "maxpool":
            output, pooled_embedding = self.original_head(fused)
        else:
            raise ValueError("Unsupported pooling_type: {}".format(self.pooling_type))

        if not return_details:
            return output

        return {
            "logits": output,
            "log_probabilities": output,
            "embedding": pooled_embedding,
            "hidden_states": full_features,
            "fused_sequence": fused,
            "layer_weights": original_layer_weights,
            "effective_layer_weights": effective_layer_weights,
            "layer_mask": layer_mask,
            "temporal_weights": self.last_temporal_weights,
            "channel_context": self.last_channel_context,
        }
