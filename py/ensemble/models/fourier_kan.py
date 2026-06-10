from typing import List, Sequence

import torch
import torch.nn as nn


def _assert_layers(layers: Sequence[int]) -> List[int]:
    layers_list = list(layers)
    if len(layers_list) < 2:
        raise ValueError(f"layers must have at least 2 elements, got {layers_list}")
    if any(int(x) <= 0 for x in layers_list):
        raise ValueError(f"layers must be positive ints, got {layers_list}")
    return [int(x) for x in layers_list]


class _FourierKANLayer(nn.Module):
    """
    A simple Fourier feature KAN-like layer.
    """

    def __init__(self, in_features: int, out_features: int, num_freqs: int = 8):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_freqs = int(num_freqs)
        if self.num_freqs <= 0:
            raise ValueError("num_freqs must be > 0")

        # For each input dimension, create sin/cos features for frequencies 1..K.
        # Feature dimension per input dim = 2*K.
        self.linear = nn.Linear(
            self.in_features * (2 * self.num_freqs), self.out_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected x shape (B, {self.in_features}), got {tuple(x.shape)}"
            )
        # Build Fourier basis: concat over dims and freqs
        freqs = torch.arange(
            1, self.num_freqs + 1, device=x.device, dtype=x.dtype
        ).view(1, 1, -1)
        x3 = x.unsqueeze(-1)  # (B, in, 1)
        angles = x3 * freqs  # (B, in, K)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, in, 2K)
        feats = feats.reshape(x.size(0), -1)  # (B, in*2K)
        # Some upstream KAN libraries mutate torch default dtype (e.g. to float64).
        # Make this layer robust by aligning activations to the parameter dtype.
        feats = feats.to(self.linear.weight.dtype)
        return self.linear(feats)


class FourierKAN(nn.Module):
    """
    FourierKAN implementation used by thesis scripts.
    """

    def __init__(self, layers: Sequence[int], num_freqs: int = 8, dropout: float = 0.0):
        super().__init__()
        layers_list = _assert_layers(layers)
        if len(layers_list) != 3:
            raise ValueError(
                "FourierKAN currently expects exactly 3 layers: [in, hidden, out]"
            )

        in_dim, hid, out_dim = layers_list
        self.net = nn.Sequential(
            _FourierKANLayer(in_dim, hid, num_freqs=num_freqs),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            _FourierKANLayer(hid, out_dim, num_freqs=num_freqs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
