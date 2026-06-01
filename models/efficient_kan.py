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


def _load_local_efficient_kan_class():
    """
    Load `EfficientKAN` class from `fc_kan_repo/models/efficient_kan.py` without importing the package.
    """
    import importlib.util
    from pathlib import Path

    mod_path = (
        Path(__file__).resolve().parent.parent
        / "fc_kan_repo"
        / "models"
        / "efficient_kan.py"
    )
    if not mod_path.exists():
        raise FileNotFoundError(str(mod_path))

    spec = importlib.util.spec_from_file_location("_local_efficient_kan", str(mod_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return getattr(module, "EfficientKAN")


class EfficientKAN(nn.Module):
    """
    EfficientKAN wrapper around the local implementation vendored in `fc_kan_repo/models/efficient_kan.py`.
    """

    def __init__(self, layers: Sequence[int], **kwargs):
        super().__init__()
        layers_list = _assert_layers(layers)

        try:
            _EfficientKAN = _load_local_efficient_kan_class()
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Cannot import local EfficientKAN implementation from `fc_kan_repo/models/efficient_kan.py`.\n"
                "Make sure `fc_kan_repo` exists and is on disk."
            ) from e

        self._model = _EfficientKAN(layers_hidden=layers_list, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The local implementation supports an optional `update_grid` arg.
        try:
            p = next(self._model.parameters())
            x = x.to(p.dtype)
        except StopIteration:
            pass
        return self._model(x)
