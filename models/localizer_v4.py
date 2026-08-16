"""Generation-aware causal localizers for the v4 calibration protocol."""

import torch
import torch.nn as nn

from models.residual_net import ResidualBlock


class ChannelCalibrationAdapter(nn.Module):
    """A small per-generation diagonal gain/bias calibration layer.

    Gain is parameterized in log space so it is always positive. Bias is
    parameterized in units of the training channel standard deviation; this
    keeps the regularizer dimensionless across channels and generations.
    """

    def __init__(self, generation_count, channel_std):
        super().__init__()
        if generation_count < 1:
            raise ValueError("generation_count must be positive")
        channel_std = torch.as_tensor(channel_std, dtype=torch.float32)
        if channel_std.shape != (generation_count, 9):
            raise ValueError(
                f"channel_std must have shape ({generation_count}, 9), got {tuple(channel_std.shape)}"
            )
        self.log_gain = nn.Parameter(torch.zeros(generation_count, 9))
        self.bias_scaled = nn.Parameter(torch.zeros(generation_count, 9))
        self.register_buffer("channel_std", channel_std.clamp_min(1e-8))

    def forward(self, emf_window, generation_index):
        gain = torch.exp(self.log_gain[generation_index]).unsqueeze(1)
        bias = (
            self.bias_scaled[generation_index]
            * self.channel_std[generation_index]
        ).unsqueeze(1)
        return emf_window * gain + bias

    def regularization(self):
        return torch.mean(self.log_gain ** 2) + torch.mean(self.bias_scaled ** 2)


class ResidualLocalizerV4(nn.Module):
    """Paper-compatible ResNet with causal features built inside the model.

    Packed input layout is ``W*9`` raw EMF values followed by ``W-1`` causal
    delta-time ratios. The internal feature vector is either that raw layout
    or raw + past/current log-ratios + delta-time ratios.
    """

    def __init__(
        self,
        window_size=3,
        feature_mode="combined",
        generation_count=1,
        use_calibration_adapter=False,
        adapter_channel_std=None,
        output_dim=9,
        n_blocks=7,
        hidden=512,
        slope=0.01,
        log_ratio_eps_v=1e-4,
        log_ratio_clip=5.0,
    ):
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if feature_mode not in {"raw_dt", "combined"}:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        self.window_size = int(window_size)
        self.feature_mode = feature_mode
        self.generation_count = int(generation_count)
        self.use_calibration_adapter = bool(use_calibration_adapter)
        self.output_dim = int(output_dim)
        self.n_blocks = int(n_blocks)
        self.hidden = int(hidden)
        self.log_ratio_eps_v = float(log_ratio_eps_v)
        self.log_ratio_clip = float(log_ratio_clip)
        self.packed_input_dim = self.window_size * 9 + self.window_size - 1
        self.feature_dim = (
            self.packed_input_dim
            if feature_mode == "raw_dt"
            else self.window_size * 9 + (self.window_size - 1) * 9 + self.window_size - 1
        )

        if self.use_calibration_adapter:
            if adapter_channel_std is None:
                adapter_channel_std = torch.ones(self.generation_count, 9)
            self.calibration_adapter = ChannelCalibrationAdapter(
                self.generation_count, adapter_channel_std,
            )
        else:
            self.calibration_adapter = None

        self.input_layer = nn.Linear(self.feature_dim, hidden)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, slope) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.Linear(128, output_dim))
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_std", torch.ones(self.feature_dim))
        self.register_buffer("pose_mean", torch.zeros(output_dim))
        self.register_buffer("pose_std", torch.ones(output_dim))

    def split_packed(self, packed):
        expected = self.packed_input_dim
        if packed.ndim != 2 or packed.shape[1] != expected:
            raise ValueError(f"Expected packed input (N, {expected}), got {tuple(packed.shape)}")
        raw_count = self.window_size * 9
        raw = packed[:, :raw_count].reshape(-1, self.window_size, 9)
        dt_ratio = packed[:, raw_count:]
        return raw, dt_ratio

    def build_features(self, packed, generation_index):
        raw, dt_ratio = self.split_packed(packed)
        if self.calibration_adapter is not None:
            raw = self.calibration_adapter(raw, generation_index)
        raw_flat = raw.reshape(raw.shape[0], -1)
        if self.feature_mode == "raw_dt":
            return torch.cat([raw_flat, dt_ratio], dim=1)
        current = raw[:, -1:, :]
        past = raw[:, :-1, :]
        eps = self.log_ratio_eps_v
        log_ratio = torch.log(torch.clamp(past + eps, min=eps)) - torch.log(
            torch.clamp(current + eps, min=eps)
        )
        log_ratio = torch.clamp(log_ratio, -self.log_ratio_clip, self.log_ratio_clip)
        return torch.cat([raw_flat, log_ratio.reshape(raw.shape[0], -1), dt_ratio], dim=1)

    def set_normalization(self, feature_mean, feature_std, pose_mean, pose_std):
        self.feature_mean.copy_(torch.as_tensor(feature_mean, dtype=torch.float32))
        self.feature_std.copy_(torch.as_tensor(feature_std, dtype=torch.float32))
        self.pose_mean.copy_(torch.as_tensor(pose_mean, dtype=torch.float32))
        self.pose_std.copy_(torch.as_tensor(pose_std, dtype=torch.float32))

    def adapter_regularization(self):
        if self.calibration_adapter is None:
            return self.pose_mean.new_zeros(())
        return self.calibration_adapter.regularization()

    def forward(self, packed, generation_index, return_normalized=False):
        features = self.build_features(packed, generation_index)
        x = (features - self.feature_mean) / (self.feature_std + 1e-8)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        if return_normalized:
            return out
        return out * (self.pose_std + 1e-8) + self.pose_mean


class TemporalGRULocalizerV4(nn.Module):
    """Bounded causal GRU fallback candidate operating on a past-only window."""

    def __init__(self, window_size=5, output_dim=9, hidden=128, slope=0.01):
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        self.packed_input_dim = self.window_size * 9 + self.window_size - 1
        self.output_dim = int(output_dim)
        self.hidden = int(hidden)
        self.encoder = nn.Sequential(nn.Linear(9, 128), nn.LeakyReLU(slope))
        self.gru = nn.GRU(input_size=129, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.Linear(128, output_dim))
        self.register_buffer("raw_mean", torch.zeros(9))
        self.register_buffer("raw_std", torch.ones(9))
        self.register_buffer("pose_mean", torch.zeros(output_dim))
        self.register_buffer("pose_std", torch.ones(output_dim))

    def set_normalization(self, feature_mean, feature_std, pose_mean, pose_std):
        window = self.window_size
        feature_mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        feature_std = torch.as_tensor(feature_std, dtype=torch.float32)
        raw_mean = feature_mean[:window * 9].reshape(window, 9).mean(dim=0)
        raw_std = feature_std[:window * 9].reshape(window, 9).mean(dim=0)
        self.raw_mean.copy_(raw_mean)
        self.raw_std.copy_(raw_std)
        self.pose_mean.copy_(torch.as_tensor(pose_mean, dtype=torch.float32))
        self.pose_std.copy_(torch.as_tensor(pose_std, dtype=torch.float32))

    def adapter_regularization(self):
        return self.pose_mean.new_zeros(())

    def forward(self, packed, generation_index, return_normalized=False):
        del generation_index
        raw_count = self.window_size * 9
        raw = packed[:, :raw_count].reshape(-1, self.window_size, 9)
        dt = packed[:, raw_count:]
        first_dt = torch.zeros((len(packed), 1), dtype=packed.dtype, device=packed.device)
        dt_steps = torch.cat([first_dt, dt], dim=1).unsqueeze(2)
        raw = (raw - self.raw_mean) / (self.raw_std + 1e-8)
        encoded = self.encoder(raw)
        sequence = torch.cat([encoded, dt_steps], dim=2)
        output, _ = self.gru(sequence)
        out = self.head(output[:, -1])
        if return_normalized:
            return out
        return out * (self.pose_std + 1e-8) + self.pose_mean
