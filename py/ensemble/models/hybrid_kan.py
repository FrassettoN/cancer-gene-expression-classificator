from typing import List, Optional, Sequence

import torch
import torch.nn as nn


def _assert_layers(layers: Sequence[int]) -> List[int]:
    layers_list = list(layers)
    if len(layers_list) < 2:
        raise ValueError(f"layers must have at least 2 elements, got {layers_list}")
    if any(int(x) <= 0 for x in layers_list):
        raise ValueError(f"layers must be positive ints, got {layers_list}")
    return [int(x) for x in layers_list]


class HybridKAN(nn.Module):
    """
    HybridKAN
    """

    def __init__(
        self,
        layers: Sequence[int],
        fourier_num_freqs: int = 8,
        efficient_kwargs: Optional[dict] = None,
        fourier_dropout: float = 0.0,
    ):
        super().__init__()
        layers_list = _assert_layers(layers)
        self.efficient = EfficientKAN(layers_list, **(efficient_kwargs or {}))
        self.fourier = FourierKAN(
            layers_list, num_freqs=fourier_num_freqs, dropout=fourier_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.efficient(x) + self.fourier(x)) / 2.0
