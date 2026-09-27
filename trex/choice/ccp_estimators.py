"""Empirical conditional choice probabilities for discrete state spaces."""

from typing import Optional, Literal
import torch


def estimate_ccps(
    states: torch.Tensor,
    actions: torch.Tensor,
    n_states: int,
    n_choices: int,
    method: Literal["frequency"] = "frequency",
    bandwidth: Optional[float] = None,
) -> torch.Tensor:
    """Frequency CCPs on the input device; unobserved states use uniform rows.

    ``bandwidth`` is retained for compatibility and unused by this estimator.
    Zero empirical probabilities are permitted; inversion uses 0 log 0 = 0.
    """
    if method != "frequency":
        raise NotImplementedError(f"Method {method} not implemented")
    if (
        n_states < 1
        or n_choices < 1
        or states.ndim != 1
        or states.shape != actions.shape
    ):
        raise ValueError(
            "Expected matching index vectors and positive state/choice counts"
        )
    for values, size in ((states, n_states), (actions, n_choices)):
        if (
            not torch.isfinite(values).all()
            or torch.any(values < 0)
            or torch.any(values >= size)
            or (values.is_floating_point() and not torch.equal(values, values.round()))
        ):
            raise ValueError("State/action indices must be integers in range")
    actions = actions.to(device=states.device, dtype=torch.long)
    joint = states.long() * n_choices + actions
    counts = (
        torch.bincount(joint, minlength=n_states * n_choices)
        .reshape(n_states, n_choices)
        .to(torch.get_default_dtype())
    )
    totals = counts.sum(1, keepdim=True)
    return torch.where(
        totals > 0, counts / totals.clamp_min(1), torch.full_like(counts, 1 / n_choices)
    )
