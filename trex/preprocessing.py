"""Schema-aware numeric tabular transforms used by Trex generators.

Binary rounding and nonnegative clipping are postprocessing, not a mixed-data
likelihood. No missing-data imputation or category encoding is performed.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Optional
import numpy as np


def _column_indices(
    columns: Optional[list[str]],
    selected: Optional[Iterable[int | str]],
) -> list[int]:
    if selected is None:
        return []
    result: list[int] = []
    for item in selected:
        if isinstance(item, str):
            if columns is None:
                raise ValueError(
                    "Column names are required for string column selectors."
                )
            result.append(columns.index(item))
        else:
            result.append(int(item))
    return result


@dataclass
class TabularTransformer:
    """Standardize continuous columns while preserving binary columns.

    Parameters
    ----------
    column_names : list of str, optional
        Names used when accepting or returning pandas data frames.
    binary_columns : iterable of str or int, optional
        Columns treated as Bernoulli indicators. They are left on the original
        0/1 scale during training and rounded after generation.
    nonnegative_columns : iterable of str or int, optional
        Columns clipped at zero after inverse transformation.
    eps : float, default=1e-6
        Added to standard deviations for numerical stability.
    """

    column_names: Optional[list[str]] = None
    binary_columns: Optional[Iterable[int | str]] = None
    nonnegative_columns: Optional[Iterable[int | str]] = None
    eps: float = 1e-6

    def fit(self, data: Any) -> "TabularTransformer":
        values, columns = self._values_and_columns(data)
        if self.eps <= 0 or not np.isfinite(self.eps):
            raise ValueError("eps must be finite and positive")
        if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
            raise ValueError("Training data must be a nonempty finite numeric matrix")
        if self.column_names is None:
            self.column_names = columns
        if self.column_names is not None and (
            len(self.column_names) != values.shape[1]
            or len(set(self.column_names)) != len(self.column_names)
        ):
            raise ValueError("Column names must be unique and match the data width")
        if columns is not None and self.column_names != columns:
            raise ValueError("Dataframe columns do not match column_names in order")
        self.n_features_in_ = values.shape[1]

        self.binary_indices = _column_indices(self.column_names, self.binary_columns)
        self.nonnegative_indices = _column_indices(
            self.column_names,
            self.nonnegative_columns,
        )
        for index in self.binary_indices + self.nonnegative_indices:
            if not 0 <= index < self.n_features_in_:
                raise ValueError("Column selector is out of range")
        if (
            self.binary_indices
            and not np.isin(values[:, self.binary_indices], [0, 1]).all()
        ):
            raise ValueError("Declared binary columns must contain only 0/1")
        self.continuous_indices = [
            j for j in range(values.shape[1]) if j not in set(self.binary_indices)
        ]
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0) + self.eps
        self.mean_[self.binary_indices] = 0.0
        self.std_[self.binary_indices] = 1.0
        return self

    def transform(self, data: Any) -> np.ndarray:
        self._check_is_fitted()
        values, columns = self._values_and_columns(data)
        self._validate_values(values)
        if (
            self.column_names is not None
            and columns is not None
            and columns != self.column_names
        ):
            raise ValueError("Prediction columns must match training order")
        return ((values - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, data: Any) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(
        self,
        values: Any,
        *,
        sample_binary: bool = False,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        values = np.asarray(values, dtype=np.float64)
        self._validate_values(values)
        array = values * self.std_ + self.mean_

        if self.binary_indices:
            probs = np.clip(array[:, self.binary_indices], 0.0, 1.0)
            if sample_binary:
                rng = np.random.default_rng(random_state)
                array[:, self.binary_indices] = rng.binomial(1, probs)
            else:
                array[:, self.binary_indices] = (probs >= 0.5).astype(np.float64)

        if self.nonnegative_indices:
            array[:, self.nonnegative_indices] = np.maximum(
                array[:, self.nonnegative_indices],
                0.0,
            )
        return array

    def transformed_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        lower = np.full_like(self.mean_, -np.inf, dtype=np.float64)
        upper = np.full_like(self.mean_, np.inf, dtype=np.float64)
        if self.binary_indices:
            lower[self.binary_indices] = 0.0
            upper[self.binary_indices] = 1.0
        if self.nonnegative_indices:
            lower[self.nonnegative_indices] = np.maximum(
                lower[self.nonnegative_indices],
                0.0,
            )
        return (
            ((lower - self.mean_) / self.std_).astype(np.float32),
            ((upper - self.mean_) / self.std_).astype(np.float32),
        )

    def _values_and_columns(self, data: Any) -> tuple[np.ndarray, Optional[list[str]]]:
        if hasattr(data, "columns") and hasattr(data, "to_numpy"):
            return data.to_numpy(dtype=np.float64), list(data.columns)
        return np.asarray(data, dtype=np.float64), self.column_names

    def _validate_values(self, values):
        if (
            values.ndim != 2
            or values.shape[1] != self.n_features_in_
            or not np.isfinite(values).all()
        ):
            raise ValueError("Expected finite rows with the fitted number of columns")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "mean_"):
            raise RuntimeError("TabularTransformer must be fitted before use.")
