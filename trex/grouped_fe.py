"""
Grouped fixed-effects estimation with KNN-smoothed classification moments.

This module provides a lightweight experimental estimator in the style of the
two-step grouped fixed-effects procedures studied by Bonhomme, Lamadon, and
Manresa. The first step builds unit-level embeddings from panel moments, smooths
them with a K-nearest-neighbor operator, and clusters the smoothed embeddings.
The second step estimates a linear model with group-by-time fixed effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .base import BaseEstimator
from ._utils import _encode_ids
from .linear import LinearRegression

try:
    from flash_kmeans import FlashKMeans

    _HAS_FLASH_KMEANS = True
except ImportError:
    FlashKMeans = None
    _HAS_FLASH_KMEANS = False


def _build_panel_matrix(
    features: torch.Tensor,
    unit_codes: torch.Tensor,
    time_codes: torch.Tensor,
    n_units: int,
    n_times: int,
) -> torch.Tensor:
    """Aggregate observation-level features into a unit-by-time matrix."""
    n_features = features.shape[1]
    panel = torch.zeros(
        n_units,
        n_times,
        n_features,
        device=features.device,
        dtype=features.dtype,
    )
    counts = torch.zeros(
        n_units,
        n_times,
        1,
        device=features.device,
        dtype=features.dtype,
    )

    for feature_idx in range(n_features):
        panel[:, :, feature_idx].index_put_(
            (unit_codes, time_codes),
            features[:, feature_idx],
            accumulate=True,
        )
    counts.index_put_(
        (unit_codes, time_codes, torch.zeros_like(unit_codes)),
        torch.ones(features.shape[0], device=features.device, dtype=features.dtype),
        accumulate=True,
    )
    if torch.any(counts == 0):
        raise ValueError(
            "Panel embeddings require observed cells for every unit/time pair; handle missing cells explicitly"
        )
    return panel / counts


def build_panel_embeddings(
    X: torch.Tensor,
    y: torch.Tensor,
    unit_ids: torch.Tensor,
    time_ids: torch.Tensor,
    include_outcome: bool = True,
    standardize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build unit-level embeddings by stacking panel moments over time.

    Returns embeddings ordered by sorted unique unit IDs, along with the sorted
    unit and time levels used to create that ordering.
    """
    if y.ndim == 1:
        y = y[:, None]

    feature_parts = [X]
    if include_outcome:
        feature_parts = [y] + feature_parts

    obs_features = torch.cat(feature_parts, dim=1)
    unit_levels, unit_codes = _encode_ids(unit_ids.to(dtype=torch.int64))
    time_levels, time_codes = _encode_ids(time_ids.to(dtype=torch.int64))
    panel = _build_panel_matrix(
        features=obs_features,
        unit_codes=unit_codes,
        time_codes=time_codes,
        n_units=int(unit_levels.numel()),
        n_times=int(time_levels.numel()),
    )
    embeddings = panel.reshape(panel.shape[0], -1)

    if standardize:
        mean = embeddings.mean(dim=0, keepdim=True)
        std = embeddings.std(dim=0, keepdim=True, correction=0)
        embeddings = (embeddings - mean) / torch.clamp(std, min=1e-8)

    return embeddings, unit_levels, time_levels


def chunked_knn_indices(
    embeddings: torch.Tensor,
    n_neighbors: int,
    block_size: int = 2048,
    include_self: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exact KNN indices and squared distances in chunks.
    """
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] == 0
        or not torch.isfinite(embeddings).all()
        or block_size < 1
    ):
        raise ValueError(
            "Expected nonempty finite embeddings and a positive block size"
        )
    if include_self and n_neighbors > len(embeddings):
        raise ValueError("n_neighbors cannot exceed the number of units")
    if n_neighbors <= 0:
        raise ValueError("`n_neighbors` must be positive.")

    n_units = embeddings.shape[0]
    if n_neighbors >= n_units and not include_self:
        raise ValueError("`n_neighbors` must be smaller than the number of units.")

    k = n_neighbors
    full_norm = (embeddings**2).sum(dim=1)
    all_indices = []
    all_distances = []

    for start in range(0, n_units, block_size):
        stop = min(start + block_size, n_units)
        block = embeddings[start:stop]
        block_norm = (block**2).sum(dim=1, keepdim=True)
        distances = block_norm + full_norm[None, :] - 2 * (block @ embeddings.T)
        distances = torch.clamp(distances, min=0.0)

        if not include_self:
            row_ids = torch.arange(start, stop, device=embeddings.device)
            distances[torch.arange(stop - start, device=embeddings.device), row_ids] = (
                torch.inf
            )

        values, indices = torch.topk(distances, k=k, largest=False)
        if not include_self:
            values = values[:, :n_neighbors]
            indices = indices[:, :n_neighbors]

        all_distances.append(values)
        all_indices.append(indices)

    return torch.cat(all_indices, dim=0), torch.cat(all_distances, dim=0)


def smooth_embeddings_by_knn(
    embeddings: torch.Tensor,
    neighbor_indices: torch.Tensor,
    include_self: bool = True,
) -> torch.Tensor:
    """Average each embedding with its nearest neighbors."""
    neighbor_values = embeddings[neighbor_indices]
    if include_self:
        stacked = torch.cat([embeddings[:, None, :], neighbor_values], dim=1)
        return stacked.mean(dim=1)
    return neighbor_values.mean(dim=1)


def _torch_kmeans(
    embeddings: torch.Tensor,
    n_groups: int,
    niter: int = 25,
    tol: float = 1e-6,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal torch fallback for K-Means clustering."""
    torch.manual_seed(seed)
    n_units = embeddings.shape[0]
    if n_groups > n_units:
        raise ValueError("`n_groups` cannot exceed the number of units.")

    centers = embeddings[
        torch.randperm(n_units, device=embeddings.device)[:n_groups]
    ].clone()

    for _ in range(niter):
        distances = torch.cdist(embeddings, centers)
        labels = distances.argmin(dim=1)

        updated = centers.clone()
        for group_idx in range(n_groups):
            mask = labels == group_idx
            if mask.any():
                updated[group_idx] = embeddings[mask].mean(dim=0)

        max_shift = torch.norm(updated - centers, dim=1).max()
        centers = updated
        if max_shift < tol:
            break

    final_distances = torch.cdist(embeddings, centers)
    labels = final_distances.argmin(dim=1)
    return labels, centers


def cluster_embeddings(
    embeddings: torch.Tensor,
    n_groups: int,
    niter: int = 25,
    tol: float = 1e-6,
    seed: int = 0,
    use_flash_kmeans: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cluster embeddings with flash-kmeans when available, otherwise torch fallback.
    """
    if use_flash_kmeans and _HAS_FLASH_KMEANS:
        km = FlashKMeans(
            d=embeddings.shape[1],
            k=n_groups,
            niter=niter,
            tol=tol,
            use_triton=embeddings.device.type == "cuda",
            seed=seed,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        labels = km.fit_predict(embeddings)
        centers = km.centroids_b.squeeze(0)
        return labels.to(dtype=torch.int64), centers

    return _torch_kmeans(
        embeddings=embeddings,
        n_groups=n_groups,
        niter=niter,
        tol=tol,
        seed=seed,
    )


@dataclass
class GroupedFEResult:
    """Container for grouped FE output."""

    coef: torch.Tensor
    group_ids: torch.Tensor
    group_time_ids: torch.Tensor
    neighbors: torch.Tensor


class KNNGroupedFixedEffects(BaseEstimator):
    """
    Experimental grouped fixed-effects estimator with KNN-smoothed embeddings.

    Parameters
    ----------
    n_groups : int
        Number of latent groups used in the second-step grouped FE regression.
    n_neighbors : int, default=10
        Number of neighbors for embedding smoothing.
    knn_block_size : int, default=2048
        Query block size for exact KNN evaluation.
    kmeans_niter : int, default=25
        Maximum K-Means iterations.
    seed : int, default=0
        Random seed used by clustering.
    use_flash_kmeans : bool, default=True
        Whether to use flash-kmeans when importable.
    standardize_embeddings : bool, default=True
        Whether to z-score panel embeddings before KNN and K-Means.
    include_outcome : bool, default=True
        Whether to include the outcome in the default panel embeddings.
    """

    def __init__(
        self,
        n_groups: int,
        n_neighbors: int = 10,
        knn_block_size: int = 2048,
        kmeans_niter: int = 25,
        seed: int = 0,
        use_flash_kmeans: bool = True,
        standardize_embeddings: bool = True,
        include_outcome: bool = True,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__(device=device)
        self.n_groups = int(n_groups)
        self.n_neighbors = int(n_neighbors)
        self.knn_block_size = int(knn_block_size)
        self.kmeans_niter = int(kmeans_niter)
        self.seed = int(seed)
        self.use_flash_kmeans = bool(use_flash_kmeans)
        self.standardize_embeddings = bool(standardize_embeddings)
        self.include_outcome = bool(include_outcome)

        self.linear_model_: Optional[LinearRegression] = None
        self.group_ids_: Optional[torch.Tensor] = None
        self.group_time_ids_: Optional[torch.Tensor] = None
        self.neighbor_indices_: Optional[torch.Tensor] = None
        self.embeddings_: Optional[torch.Tensor] = None
        self.smoothed_embeddings_: Optional[torch.Tensor] = None
        self.unit_levels_: Optional[torch.Tensor] = None
        self.time_levels_: Optional[torch.Tensor] = None

    def _coerce_classification_features(
        self,
        classification_features: Optional[torch.Tensor],
        X: torch.Tensor,
        y: torch.Tensor,
        unit_ids: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build or align classification features at the unit level."""
        if classification_features is None:
            return build_panel_embeddings(
                X=X,
                y=y,
                unit_ids=unit_ids,
                time_ids=time_ids,
                include_outcome=self.include_outcome,
                standardize=self.standardize_embeddings,
            )

        unit_levels, unit_codes = _encode_ids(unit_ids.to(dtype=torch.int64))
        time_levels, time_codes = _encode_ids(time_ids.to(dtype=torch.int64))
        features = classification_features.to(self.device)

        if features.ndim != 2:
            raise ValueError("`classification_features` must be a 2D tensor.")

        if features.shape[0] == X.shape[0]:
            panel = _build_panel_matrix(
                features=features,
                unit_codes=unit_codes,
                time_codes=time_codes,
                n_units=int(unit_levels.numel()),
                n_times=int(time_levels.numel()),
            )
            embeddings = panel.reshape(panel.shape[0], -1)
        elif features.shape[0] == unit_levels.numel():
            embeddings = features
        else:
            raise ValueError(
                "`classification_features` must have either `n_obs` or `n_units` rows."
            )

        if self.standardize_embeddings:
            mean = embeddings.mean(dim=0, keepdim=True)
            std = embeddings.std(dim=0, keepdim=True)
            embeddings = (embeddings - mean) / torch.clamp(std, min=1e-8)

        return embeddings, unit_levels, time_levels

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        unit_ids: torch.Tensor,
        time_ids: torch.Tensor,
        classification_features: Optional[torch.Tensor] = None,
        classification_unit_ids: Optional[torch.Tensor] = None,
        se: Optional[str] = None,
    ) -> "KNNGroupedFixedEffects":
        """
        Estimate grouped fixed effects through KNN smoothing and clustered FE OLS.

        If `classification_features` has one row per unit, rows are assumed to be
        ordered by `sorted(unique(unit_ids))` unless `classification_unit_ids` is
        provided. Passing `classification_unit_ids` is safer for shuffled or
        non-contiguous unit identifiers; features are then aligned internally to
        the fitted unit ordering.
        """
        X = X.to(self.device)
        y = y.to(self.device)
        unit_ids = unit_ids.to(self.device)
        time_ids = time_ids.to(self.device)

        embeddings, unit_levels, time_levels = self._coerce_classification_features(
            classification_features=classification_features,
            X=X,
            y=y,
            unit_ids=unit_ids,
            time_ids=time_ids,
        )
        if classification_unit_ids is not None:
            if classification_features is None:
                raise ValueError(
                    "`classification_unit_ids` can only be used with unit-level "
                    "`classification_features`."
                )
            classification_unit_ids = classification_unit_ids.to(self.device).to(
                dtype=torch.int64
            )
            if (
                classification_unit_ids.ndim != 1
                or classification_unit_ids.numel() != embeddings.shape[0]
            ):
                raise ValueError(
                    "`classification_unit_ids` must be a vector with one entry per "
                    "unit-level classification-feature row."
                )
            if embeddings.shape[0] != unit_levels.numel():
                raise ValueError(
                    "`classification_unit_ids` is only valid when `classification_features` "
                    "has one row per unit, not one row per observation."
                )
            sorted_ids, order = torch.sort(classification_unit_ids)
            if torch.any(sorted_ids[1:] == sorted_ids[:-1]):
                raise ValueError(
                    "`classification_unit_ids` must not contain duplicates."
                )
            if sorted_ids.numel() != unit_levels.numel() or not torch.equal(
                sorted_ids, unit_levels
            ):
                raise ValueError(
                    "`classification_unit_ids` must match the sorted unique `unit_ids`."
                )
            embeddings = embeddings[order]

        neighbor_indices, _ = chunked_knn_indices(
            embeddings=embeddings,
            n_neighbors=self.n_neighbors,
            block_size=self.knn_block_size,
            include_self=False,
        )
        smoothed_embeddings = smooth_embeddings_by_knn(
            embeddings=embeddings,
            neighbor_indices=neighbor_indices,
            include_self=True,
        )
        unit_group_ids, _ = cluster_embeddings(
            embeddings=smoothed_embeddings,
            n_groups=self.n_groups,
            niter=self.kmeans_niter,
            seed=self.seed,
            use_flash_kmeans=self.use_flash_kmeans,
        )

        _, unit_codes = _encode_ids(unit_ids.to(dtype=torch.int64))
        _, time_codes = _encode_ids(time_ids.to(dtype=torch.int64))
        obs_group_ids = unit_group_ids[unit_codes]
        group_time_ids = obs_group_ids * int(time_levels.numel()) + time_codes

        linear_model = LinearRegression(device=self.device)
        linear_model.fit(X, y, fe=[group_time_ids], se=se)

        self.linear_model_ = linear_model
        self.group_ids_ = unit_group_ids.detach()
        self.group_time_ids_ = group_time_ids.detach()
        self.neighbor_indices_ = neighbor_indices.detach()
        self.embeddings_ = embeddings.detach()
        self.smoothed_embeddings_ = smoothed_embeddings.detach()
        self.unit_levels_ = unit_levels.detach()
        self.time_levels_ = time_levels.detach()
        self.params = linear_model.params
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Delegate prediction to the second-step linear model."""
        if self.linear_model_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.linear_model_.predict(X)

    def summary(self) -> None:
        """Print the second-step linear regression summary."""
        if self.linear_model_ is None:
            raise ValueError("Model has not been fitted yet.")
        self.linear_model_.summary()
