"""Shared tensor conversion and optimizer helpers (private API)."""

from typing import Any, Optional
import torch


def _to_tensor(
    value: Any, device: torch.device, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Convert directly into the target dtype, without an intermediate float32.

    In particular, Python float scalars must not lose precision before being
    promoted to float64 (e.g. a kernel bandwidth or an optimizer tolerance).
    """
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype or value.dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _optimizer_display_name(optimizer_class: Any) -> str:
    base = getattr(optimizer_class, "func", optimizer_class)
    return getattr(base, "__name__", base.__class__.__name__)


def _is_lbfgs_optimizer(optimizer_class: Any) -> bool:
    base = getattr(optimizer_class, "func", optimizer_class)
    try:
        return issubclass(base, torch.optim.LBFGS)
    except TypeError:
        return False


def _encode_ids(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    levels, codes = torch.unique(values, sorted=True, return_inverse=True)
    return levels, codes.to(dtype=torch.int64)
