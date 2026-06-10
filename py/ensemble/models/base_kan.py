from typing import List, Sequence

import torch
import torch.nn as nn

from .efficient_kan import _EfficientKAN

from typing import List, Sequence


def _assert_layers(layers: Sequence[int]) -> List[int]:
    layers_list = list(layers)
    if len(layers_list) < 2:
        raise ValueError(f"layers must have at least 2 elements, got {layers_list}")
    if any(int(x) <= 0 for x in layers_list):
        raise ValueError(f"layers must be positive ints, got {layers_list}")
    return [int(x) for x in layers_list]


class BaseKAN(nn.Module):
    """
    BaseKAN (PyTorch-compatible) implementation used by this thesis codebase.
    """

    def __init__(self, layers: Sequence[int], **kwargs):
        super().__init__()
        layers_list = _assert_layers(layers)

        # Use the same constructor surface as EfficientKAN (layers_hidden + params).
        kwargs = dict(kwargs)
        kwargs.pop("enable_standalone_scale_spline", None)
        self._model = _EfficientKAN(layers_hidden=layers_list, **kwargs)
        for layer in getattr(self._model, "layers", []):
            if hasattr(layer, "enable_standalone_scale_spline"):
                layer.enable_standalone_scale_spline = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            p = next(self._model.parameters())
            x = x.to(p.dtype)
        except StopIteration:
            pass
        return self._model(x)
