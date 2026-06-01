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


class BaseKAN(nn.Module):
    """
    BaseKAN (PyTorch-compatible) implementation used by this thesis codebase.
    """

    def __init__(self, layers: Sequence[int], **kwargs):
        super().__init__()
        layers_list = _assert_layers(layers)
        try:
            _EfficientKAN = _load_local_efficient_kan_class()
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Cannot build BaseKAN because local EfficientKAN implementation is unavailable.\n"
                "Expected to find `fc_kan_repo/models/efficient_kan.py`."
            ) from e

        # Use the same constructor surface as EfficientKAN (layers_hidden + params).
        self._model = _EfficientKAN(layers_hidden=layers_list, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            p = next(self._model.parameters())
            x = x.to(p.dtype)
        except StopIteration:
            pass
        return self._model(x)
