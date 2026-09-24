from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy import sparse, stats

from .base import BaseEstimator


def _optimizer_display_name(optimizer_class: Any) -> str:
    """Return a readable optimizer name for classes and functools.partial."""
    if hasattr(optimizer_class, "__name__"):
        return optimizer_class.__name__
    if hasattr(optimizer_class, "func") and hasattr(optimizer_class.func, "__name__"):
        return optimizer_class.func.__name__
    return optimizer_class.__class__.__name__


def _is_lbfgs_optimizer(optimizer_class: Any) -> bool:
    """Check whether an optimizer specification resolves to LBFGS."""
    base = getattr(optimizer_class, "func", optimizer_class)
    try:
        return issubclass(base, torch.optim.LBFGS)
    except TypeError:
        return base == torch.optim.LBFGS


def _to_tensor(
    value: Any,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Convert arrays to tensors on the requested device."""
    if isinstance(value, torch.Tensor):
        tensor = value.to(device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    tensor = torch.as_tensor(value, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _safe_inverse(
    matrix: torch.Tensor,
    base_ridge: float = 1e-8,
    max_attempts: int = 6,
) -> tuple[torch.Tensor, float]:
    """
    Invert a matrix with escalating ridge regularization and pseudo-inverse fallback.
    """
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    ridge = 0.0

    for attempt in range(max_attempts):
        adjusted = matrix if ridge == 0.0 else matrix + ridge * eye
        try:
            inv = torch.linalg.inv(adjusted)
            if torch.isfinite(inv).all():
                return inv, ridge
        except RuntimeError:
            pass

        ridge = base_ridge if attempt == 0 else ridge * 10.0

    adjusted = matrix + ridge * eye
    return torch.linalg.pinv(adjusted), ridge


class MaximumLikelihoodEstimator(BaseEstimator):
    """
    Base class for Maximum Likelihood Estimators using PyTorch optimizers.

    This class provides a flexible framework for fitting statistical models via
    maximum likelihood estimation. It supports various PyTorch optimizers and
    automatically computes standard errors using the Fisher information matrix.

    Attributes:
        optimizer_class: PyTorch optimizer class (default: LBFGS).
        maxiter: Maximum number of optimization iterations.
        tol: Convergence tolerance for relative change in loss.
        params: Dictionary containing fitted parameters and diagnostics.
        history: Dictionary tracking optimization history (e.g., loss values).

    Examples:
        >>> class MyModel(MaximumLikelihoodEstimator):
        ...     def _negative_log_likelihood(self, params, X, y):
        ...         return torch.sum((y - X @ params) ** 2)  # OLS example
        ...     def _compute_fisher_information(self, params, X, y):
        ...         return X.T @ X
        >>> model = MyModel()
        >>> model.fit(X, y)
    """

    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        maxiter: int = 5000,
        tol: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)
        self.optimizer_class = optimizer if optimizer is not None else torch.optim.LBFGS
        self.maxiter = maxiter
        self.tol = tol
        self.params: Dict[str, torch.Tensor] = {}
        self.history: Dict[str, list] = {"loss": []}
        self._fitted_X: Optional[torch.Tensor] = None
        self._fitted_y: Optional[torch.Tensor] = None
        self._fit_context: Optional[Dict[str, Any]] = None

    @abstractmethod
    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the negative log-likelihood for the model.
        Must be implemented by subclasses.

        Args:
            params: Model parameters tensor of shape (n_features,).
            X: Design matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).

        Returns:
            Negative log-likelihood as a scalar tensor (maintains gradient information).
        """
        raise NotImplementedError

    def _supports_panel_glm(self) -> bool:
        """Whether the estimator supports the FE-aware GLM fitting path."""
        return False

    def _per_observation_nll_from_eta(
        self,
        linear_predictor: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Per-observation negative log-likelihood used by the FE-GLM path."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement FE-aware likelihoods."
        )

    def _information_weights_from_eta(
        self,
        linear_predictor: torch.Tensor,
    ) -> torch.Tensor:
        """Per-observation Fisher weights used for FE-aware standard errors."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement FE-aware inference."
        )

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        init_params: Optional[torch.Tensor] = None,
        verbose: bool = False,
        fe: Optional[Sequence[Any]] = None,
        fe_design: Optional[Sequence[Any]] = None,
        hdfe_index: Optional[int] = None,
        offset: Optional[Any] = None,
        batch_size: Optional[int] = None,
        stderr: bool = True,
    ) -> "MaximumLikelihoodEstimator":
        """
        Fit the model using maximum likelihood estimation.

        Optimizes the model parameters to minimize the negative log-likelihood
        using the specified PyTorch optimizer. After convergence, computes
        standard errors via the Fisher information matrix.

        Args:
            X: Design matrix of shape (n_samples, n_features).
               Assumes X includes an intercept column if desired.
            y: Target vector of shape (n_samples,).
            init_params: Initial parameter values. For FE-aware GLMs, may be
                a packed parameter vector or a dict with `coef` and `fe_coef`.
            verbose: If True, prints convergence information.
            fe: Optional list of 1D fixed-effect ID vectors.
            fe_design: Optional list of sparse fixed-effect incidence matrices.
            hdfe_index: Optional index of the FE block to use for fast inference.
            offset: Optional offset vector added to the linear predictor.
            batch_size: Optional mini-batch size for first-order optimizers.
            stderr: Whether to compute standard errors.

        Returns:
            Self (fitted estimator) with populated params dictionary.

        Examples:
            >>> model = LogisticRegression()
            >>> model.fit(X_train, y_train, verbose=True)
            >>> print(model.params["coef"])
        """
        if self._supports_panel_glm():
            return self._fit_panel_glm(
                X=X,
                y=y,
                init_params=init_params,
                verbose=verbose,
                fe=fe,
                fe_design=fe_design,
                hdfe_index=hdfe_index,
                offset=offset,
                batch_size=batch_size,
                stderr=stderr,
            )

        if any(
            value is not None
            for value in (fe, fe_design, hdfe_index, offset, batch_size)
        ):
            raise ValueError(
                f"{self.__class__.__name__} does not support FE-aware MLE options."
            )

        # Move data to device
        X = _to_tensor(X, self.device)
        if not X.is_floating_point():
            X = X.to(torch.float64)
        y = _to_tensor(y, self.device, dtype=X.dtype)
        if X.ndim != 2 or y.shape[0] != X.shape[0] or len(X) == 0:
            raise ValueError("X and y must have matching, nonempty rows")
        if not torch.isfinite(X).all() or not torch.isfinite(y).all():
            raise ValueError("X and y must be finite")
        n_features = X.shape[1]
        if init_params is None:
            init_params_val = torch.zeros(n_features, device=self.device, dtype=X.dtype)
        else:
            init_params_val = init_params.to(device=self.device, dtype=X.dtype)

        current_params = init_params_val.detach().clone().requires_grad_(True)

        if _is_lbfgs_optimizer(self.optimizer_class):
            optimizer = self.optimizer_class(
                [current_params],
                max_iter=20,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-9,
                tolerance_change=torch.finfo(X.dtype).eps,
            )
        else:
            optimizer = self.optimizer_class([current_params])

        self.history["loss"] = []
        self._fit_context = None

        for i in range(self.maxiter):

            def closure():
                optimizer.zero_grad()
                loss = self._negative_log_likelihood(current_params, X, y)
                loss.backward()
                return loss

            if _is_lbfgs_optimizer(self.optimizer_class):
                loss_val = optimizer.step(closure)
            else:
                loss_val = closure()
                optimizer.step()

            self.history["loss"].append(loss_val.item())

            if i > 10 and self.tol > 0:
                loss_change = abs(
                    self.history["loss"][-2] - self.history["loss"][-1]
                ) / (abs(self.history["loss"][-2]) + 1e-8)
                if loss_change < self.tol:
                    if verbose:
                        print(f"Convergence tolerance {self.tol} met at iteration {i}.")
                    break

        self.params = {"coef": current_params.detach()}
        self.iterations_run = i + 1

        self._fitted_X = X.detach()
        self._fitted_y = y.detach()

        if stderr:
            self._compute_standard_errors()

        return self

    def _canonicalize_fe_inputs(
        self,
        fe: Optional[Sequence[Any]],
        fe_design: Optional[Sequence[Any]],
        n_obs: int,
        dtype: torch.dtype,
    ) -> list[Dict[str, Any]]:
        """Convert FE inputs into contiguous integer codes with metadata."""
        if fe is not None and fe_design is not None:
            raise ValueError("Provide exactly one of `fe` or `fe_design`, not both.")

        if fe is None and fe_design is None:
            return []

        fe_blocks: list[Dict[str, Any]] = []

        if fe is not None:
            if isinstance(fe, torch.Tensor):
                if fe.ndim == 1:
                    raw_blocks = [fe]
                elif fe.ndim == 2:
                    raw_blocks = [fe[:, j] for j in range(fe.shape[1])]
                else:
                    raise ValueError("`fe` tensor must be 1D or 2D.")
            else:
                raw_blocks = list(fe)

            for block in raw_blocks:
                block_tensor = _to_tensor(block, device=self.device)
                if block_tensor.ndim != 1:
                    raise ValueError("Each fixed-effect ID vector must be 1D.")
                if block_tensor.shape[0] != n_obs:
                    raise ValueError("Fixed-effect ID vectors must match `X` rows.")

                unique_ids, codes = torch.unique(
                    block_tensor, sorted=True, return_inverse=True
                )
                fe_blocks.append(
                    {
                        "mode": "ids",
                        "codes": codes.to(dtype=torch.int64),
                        "n_levels": int(unique_ids.numel()),
                        "levels": unique_ids.to(self.device),
                        "dtype": block_tensor.dtype,
                    }
                )

            return fe_blocks

        for design in list(fe_design):
            block = self._canonicalize_fe_design_block(
                design=design,
                n_obs=n_obs,
                dtype=dtype,
            )
            fe_blocks.append(block)

        return fe_blocks

    def _canonicalize_fe_design_block(
        self,
        design: Any,
        n_obs: int,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """Validate a sparse incidence matrix and reduce it to row-level codes."""
        if sparse.issparse(design):
            csr = design.tocsr()
            if csr.shape[0] != n_obs:
                raise ValueError("Fixed-effect design matrices must match `X` rows.")
            row_counts = np.diff(csr.indptr)
            if not np.all(row_counts == 1):
                raise ValueError(
                    "Each FE design row must contain exactly one active level."
                )
            if not np.allclose(csr.data, 1.0):
                raise ValueError(
                    "FE design matrices must be one-hot incidence matrices."
                )

            codes = torch.as_tensor(csr.indices, device=self.device, dtype=torch.int64)
            n_levels = csr.shape[1]
            return {
                "mode": "design",
                "codes": codes,
                "n_levels": int(n_levels),
                "levels": torch.arange(n_levels, device=self.device, dtype=torch.int64),
                "dtype": dtype,
            }

        if isinstance(design, torch.Tensor) and design.layout == torch.sparse_csr:
            if design.shape[0] != n_obs:
                raise ValueError("Fixed-effect design matrices must match `X` rows.")

            crow = design.crow_indices().to(device=self.device)
            col = design.col_indices().to(device=self.device)
            values = design.values().to(device=self.device)
            row_counts = crow[1:] - crow[:-1]
            if not torch.all(row_counts == 1):
                raise ValueError(
                    "Each FE design row must contain exactly one active level."
                )
            if not torch.allclose(values, torch.ones_like(values)):
                raise ValueError(
                    "FE design matrices must be one-hot incidence matrices."
                )

            return {
                "mode": "design",
                "codes": col.to(dtype=torch.int64),
                "n_levels": int(design.shape[1]),
                "levels": torch.arange(
                    design.shape[1], device=self.device, dtype=torch.int64
                ),
                "dtype": dtype,
            }

        if isinstance(design, torch.Tensor) and design.layout == torch.sparse_coo:
            coalesced = design.coalesce().to(self.device)
            if coalesced.shape[0] != n_obs:
                raise ValueError("Fixed-effect design matrices must match `X` rows.")

            indices = coalesced.indices()
            values = coalesced.values()
            row = indices[0]
            col = indices[1]
            row_counts = torch.bincount(row, minlength=n_obs)
            if not torch.all(row_counts == 1):
                raise ValueError(
                    "Each FE design row must contain exactly one active level."
                )
            if not torch.allclose(values, torch.ones_like(values)):
                raise ValueError(
                    "FE design matrices must be one-hot incidence matrices."
                )

            codes = torch.empty(n_obs, device=self.device, dtype=torch.int64)
            codes.scatter_(0, row, col.to(dtype=torch.int64))
            return {
                "mode": "design",
                "codes": codes,
                "n_levels": int(design.shape[1]),
                "levels": torch.arange(
                    design.shape[1], device=self.device, dtype=torch.int64
                ),
                "dtype": dtype,
            }

        dense = _to_tensor(design, device=self.device, dtype=dtype)
        if dense.ndim != 2 or dense.shape[0] != n_obs:
            raise ValueError("Dense FE design matrices must be 2D with `n_obs` rows.")
        if not torch.allclose(
            dense.sum(dim=1), torch.ones(n_obs, device=self.device, dtype=dtype)
        ):
            raise ValueError(
                "Dense FE design matrices must be one-hot incidence matrices."
            )

        codes = torch.argmax(dense, dim=1).to(dtype=torch.int64)
        return {
            "mode": "design",
            "codes": codes,
            "n_levels": int(dense.shape[1]),
            "levels": torch.arange(
                dense.shape[1], device=self.device, dtype=torch.int64
            ),
            "dtype": dtype,
        }

    def _initialize_panel_parameters(
        self,
        n_features: int,
        fe_blocks: Sequence[Dict[str, Any]],
        dtype: torch.dtype,
        init_params: Optional[Any],
    ) -> torch.Tensor:
        """Construct an initial packed parameter vector."""
        total_params = n_features + sum(
            max(block["n_levels"] - 1, 0) for block in fe_blocks
        )

        if init_params is None:
            return torch.zeros(total_params, device=self.device, dtype=dtype)

        if isinstance(init_params, dict):
            packed = torch.zeros(total_params, device=self.device, dtype=dtype)
            beta = _to_tensor(init_params["coef"], self.device, dtype=dtype).flatten()
            if beta.numel() != n_features:
                raise ValueError("`init_params['coef']` has the wrong length.")
            packed[:n_features] = beta

            fe_init = init_params.get("fe_coef", [])
            if len(fe_init) != len(fe_blocks):
                raise ValueError(
                    "`init_params['fe_coef']` must match the FE block count."
                )

            cursor = n_features
            for block, values in zip(fe_blocks, fe_init):
                reduced = max(block["n_levels"] - 1, 0)
                if reduced == 0:
                    continue
                full_values = _to_tensor(values, self.device, dtype=dtype).flatten()
                if full_values.numel() == block["n_levels"]:
                    packed[cursor : cursor + reduced] = full_values[1:]
                elif full_values.numel() == reduced:
                    packed[cursor : cursor + reduced] = full_values
                else:
                    raise ValueError("Initial FE coefficients have the wrong length.")
                cursor += reduced

            return packed

        init_tensor = _to_tensor(init_params, self.device, dtype=dtype).flatten()
        if init_tensor.numel() == total_params:
            return init_tensor
        if init_tensor.numel() == n_features:
            packed = torch.zeros(total_params, device=self.device, dtype=dtype)
            packed[:n_features] = init_tensor
            return packed

        raise ValueError("Initial parameter vector has the wrong length.")

    def _unpack_panel_parameters(
        self,
        params: torch.Tensor,
        n_features: int,
        fe_blocks: Sequence[Dict[str, Any]],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Split the packed parameter vector into beta and FE blocks."""
        beta = params[:n_features]
        cursor = n_features
        fe_params: list[torch.Tensor] = []

        for block in fe_blocks:
            reduced = max(block["n_levels"] - 1, 0)
            full = torch.zeros(
                block["n_levels"],
                device=params.device,
                dtype=params.dtype,
            )
            if reduced > 0:
                full[1:] = params[cursor : cursor + reduced]
                cursor += reduced
            fe_params.append(full)

        return beta, fe_params

    def _panel_linear_predictor(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        fe_blocks: Sequence[Dict[str, Any]],
        offset: Optional[torch.Tensor],
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the FE-aware linear predictor for a full sample or batch."""
        beta, fe_params = self._unpack_panel_parameters(params, X.shape[1], fe_blocks)

        X_view = X if indices is None else X[indices]
        linear_predictor = X_view @ beta

        if offset is not None:
            offset_view = offset if indices is None else offset[indices]
            linear_predictor = linear_predictor + offset_view

        for block, values in zip(fe_blocks, fe_params):
            codes = block["codes"] if indices is None else block["codes"][indices]
            linear_predictor = linear_predictor + values[codes]

        return linear_predictor

    def _panel_objective(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        fe_blocks: Sequence[Dict[str, Any]],
        offset: Optional[torch.Tensor],
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean negative log-likelihood for dense or batched FE-aware GLMs."""
        y_view = y if indices is None else y[indices]
        eta = self._panel_linear_predictor(
            params=params,
            X=X,
            fe_blocks=fe_blocks,
            offset=offset,
            indices=indices,
        )
        return self._per_observation_nll_from_eta(eta, y_view).mean()

    def _dense_fe_block(
        self,
        codes: torch.Tensor,
        n_levels: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Construct a dense dummy matrix with the reference level dropped."""
        reduced = max(n_levels - 1, 0)
        design = torch.zeros(
            codes.shape[0],
            reduced,
            device=codes.device,
            dtype=dtype,
        )
        if reduced == 0:
            return design

        mask = codes > 0
        if mask.any():
            rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
            cols = (codes[mask] - 1).to(dtype=torch.int64)
            design[rows, cols] = 1.0

        return design

    def _prepare_prediction_fe_blocks(
        self,
        fe: Optional[Sequence[Any]],
        fe_design: Optional[Sequence[Any]],
        n_obs: int,
    ) -> list[Dict[str, Any]]:
        """Map prediction-time FE inputs into the fitted level space."""
        if self._fit_context is None:
            return []

        fitted_blocks = self._fit_context["fe_blocks"]
        if not fitted_blocks:
            return []

        if fe is None and fe_design is None:
            if self._fitted_X is not None and n_obs == self._fitted_X.shape[0]:
                return fitted_blocks
            raise ValueError(
                "Prediction for FE models requires `fe` or `fe_design` for new data."
            )

        if fe is not None and fe_design is not None:
            raise ValueError("Provide exactly one of `fe` or `fe_design`, not both.")

        if fe is not None:
            if isinstance(fe, torch.Tensor):
                if fe.ndim == 1:
                    raw_blocks = [fe]
                elif fe.ndim == 2:
                    raw_blocks = [fe[:, j] for j in range(fe.shape[1])]
                else:
                    raise ValueError("`fe` tensor must be 1D or 2D.")
            else:
                raw_blocks = list(fe)

            if len(raw_blocks) != len(fitted_blocks):
                raise ValueError("Prediction FE blocks must match the fitted model.")

            pred_blocks: list[Dict[str, Any]] = []
            for raw_block, fitted_block in zip(raw_blocks, fitted_blocks):
                block_tensor = _to_tensor(raw_block, self.device)
                if block_tensor.ndim != 1 or block_tensor.shape[0] != n_obs:
                    raise ValueError(
                        "Prediction FE vectors must be 1D and match `X` rows."
                    )

                if fitted_block["mode"] == "design":
                    raise ValueError(
                        "This model was fitted with `fe_design`; pass `fe_design` at prediction time."
                    )

                levels = fitted_block["levels"]
                codes = torch.searchsorted(levels, block_tensor)
                valid = codes < levels.numel()
                valid = valid & (
                    levels[codes.clamp_max(levels.numel() - 1)] == block_tensor
                )
                if not torch.all(valid):
                    raise ValueError("Prediction FE contains unseen levels.")

                pred_blocks.append(
                    {
                        "mode": "ids",
                        "codes": codes.to(dtype=torch.int64),
                        "n_levels": fitted_block["n_levels"],
                        "levels": levels,
                        "dtype": block_tensor.dtype,
                    }
                )

            return pred_blocks

        if len(fe_design) != len(fitted_blocks):
            raise ValueError("Prediction FE blocks must match the fitted model.")

        pred_blocks = []
        for raw_block, fitted_block in zip(list(fe_design), fitted_blocks):
            block = self._canonicalize_fe_design_block(
                design=raw_block,
                n_obs=n_obs,
                dtype=self.params["coef"].dtype,
            )
            if block["n_levels"] != fitted_block["n_levels"]:
                raise ValueError(
                    "Prediction FE design columns do not match the fitted model."
                )
            pred_blocks.append(block)

        return pred_blocks

    def _predict_linear_predictor(
        self,
        X: torch.Tensor,
        fe: Optional[Sequence[Any]] = None,
        fe_design: Optional[Sequence[Any]] = None,
        offset: Optional[Any] = None,
    ) -> torch.Tensor:
        """Compute the fitted model's linear predictor for dense or FE-aware GLMs."""
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        X = _to_tensor(X, self.device, dtype=self.params["coef"].dtype)
        linear_predictor = X @ self.params["coef"]

        if self._fit_context is None:
            if offset is not None:
                linear_predictor = linear_predictor + _to_tensor(
                    offset, self.device, dtype=X.dtype
                )
            return linear_predictor

        fitted_offset = self._fit_context["offset"]
        if offset is not None:
            linear_predictor = linear_predictor + _to_tensor(
                offset, self.device, dtype=X.dtype
            )
        elif fitted_offset is not None:
            if self._fitted_X is not None and X.shape[0] == self._fitted_X.shape[0]:
                linear_predictor = linear_predictor + fitted_offset
            else:
                raise ValueError(
                    "Prediction for offset models requires an explicit `offset` for new data."
                )

        fe_blocks = self._prepare_prediction_fe_blocks(
            fe=fe,
            fe_design=fe_design,
            n_obs=X.shape[0],
        )
        fe_params = self.params.get("fe_coef", [])
        for block, values in zip(fe_blocks, fe_params):
            linear_predictor = linear_predictor + values[block["codes"]]

        return linear_predictor

    def _compute_panel_standard_errors(self) -> None:
        """Compute FE-aware Fisher standard errors for supported GLMs."""
        if self._fit_context is None:
            return

        X = self._fit_context["X"]
        y = self._fit_context["y"]
        fe_blocks = self._fit_context["fe_blocks"]
        offset = self._fit_context["offset"]
        hdfe_index = self._fit_context["hdfe_index"]
        dtype = X.dtype

        eta = self._predict_linear_predictor(
            X,
            fe=None,
            fe_design=None,
            offset=offset,
        )
        weights = self._information_weights_from_eta(eta).to(dtype=dtype)
        weights = torch.clamp(weights, min=1e-12)

        n_features = X.shape[1]
        n_blocks = len(fe_blocks)

        if hdfe_index is not None and (hdfe_index < 0 or hdfe_index >= n_blocks):
            raise ValueError("`hdfe_index` must refer to a valid FE block.")

        if hdfe_index is None:
            design_parts = [X]
            for block in fe_blocks:
                design_parts.append(
                    self._dense_fe_block(
                        codes=block["codes"],
                        n_levels=block["n_levels"],
                        dtype=dtype,
                    )
                )

            design = torch.cat(design_parts, dim=1)
            weighted_design = design * torch.sqrt(weights).unsqueeze(1)
            fisher = weighted_design.T @ weighted_design
            fisher_inv, ridge = _safe_inverse(fisher)

            self.params["se"] = torch.sqrt(
                torch.clamp(torch.diag(fisher_inv[:n_features, :n_features]), min=0.0)
            )
            self.params["vcov"] = fisher_inv[:n_features, :n_features]
            self.params["inference_ridge"] = ridge

            if fe_blocks:
                cursor = n_features
                fe_se: list[torch.Tensor] = []
                for block in fe_blocks:
                    reduced = max(block["n_levels"] - 1, 0)
                    full = torch.zeros(
                        block["n_levels"],
                        device=self.device,
                        dtype=dtype,
                    )
                    if reduced > 0:
                        block_vcov = fisher_inv[
                            cursor : cursor + reduced, cursor : cursor + reduced
                        ]
                        full[1:] = torch.sqrt(
                            torch.clamp(torch.diag(block_vcov), min=0.0)
                        )
                        cursor += reduced
                    fe_se.append(full)
                self.params["fe_se"] = fe_se

            return

        theta_parts = [X]
        theta_layout: list[tuple[int, slice]] = []
        cursor = n_features
        theta_width = n_features

        for block_index, block in enumerate(fe_blocks):
            if block_index == hdfe_index:
                continue
            dense_block = self._dense_fe_block(
                codes=block["codes"],
                n_levels=block["n_levels"],
                dtype=dtype,
            )
            theta_parts.append(dense_block)
            width = dense_block.shape[1]
            theta_layout.append((block_index, slice(theta_width, theta_width + width)))
            theta_width += width

        theta_design = torch.cat(theta_parts, dim=1)
        weighted_theta = theta_design * torch.sqrt(weights).unsqueeze(1)
        fisher_theta = weighted_theta.T @ weighted_theta

        hdfe_block = fe_blocks[hdfe_index]
        reduced_levels = max(hdfe_block["n_levels"] - 1, 0)

        if reduced_levels == 0:
            schur_inv, ridge = _safe_inverse(fisher_theta)
            self.params["se"] = torch.sqrt(
                torch.clamp(torch.diag(schur_inv[:n_features, :n_features]), min=0.0)
            )
            self.params["vcov"] = schur_inv[:n_features, :n_features]
            self.params["fe_se_diag"] = torch.zeros(
                hdfe_block["n_levels"],
                device=self.device,
                dtype=dtype,
            )
            self.params["inference_ridge"] = ridge
            return

        mask = hdfe_block["codes"] > 0
        reduced_codes = hdfe_block["codes"][mask] - 1

        B_t = torch.zeros(
            reduced_levels,
            theta_design.shape[1],
            device=self.device,
            dtype=dtype,
        )
        weighted_rows = theta_design[mask] * weights[mask].unsqueeze(1)
        for col_idx in range(theta_design.shape[1]):
            B_t[:, col_idx].scatter_add_(0, reduced_codes, weighted_rows[:, col_idx])
        B = B_t.T

        D = torch.zeros(reduced_levels, device=self.device, dtype=dtype)
        D.scatter_add_(0, reduced_codes, weights[mask])
        D = torch.clamp(D, min=1e-12)

        schur = fisher_theta - (B / D.unsqueeze(0)) @ B.T
        schur_inv, ridge = _safe_inverse(schur)

        self.params["se"] = torch.sqrt(
            torch.clamp(torch.diag(schur_inv[:n_features, :n_features]), min=0.0)
        )
        self.params["vcov"] = schur_inv[:n_features, :n_features]
        self.params["inference_ridge"] = ridge

        fe_se: list[Optional[torch.Tensor]] = [None] * n_blocks
        for block_index, block_slice in theta_layout:
            block = fe_blocks[block_index]
            full = torch.zeros(
                block["n_levels"],
                device=self.device,
                dtype=dtype,
            )
            if block_slice.stop > block_slice.start:
                block_vcov = schur_inv[block_slice, block_slice]
                full[1:] = torch.sqrt(torch.clamp(torch.diag(block_vcov), min=0.0))
            fe_se[block_index] = full

        projected = schur_inv @ B
        hdfe_diag = (1.0 / D) + torch.sum(B * projected, dim=0) / (D**2)
        hdfe_full = torch.zeros(
            hdfe_block["n_levels"],
            device=self.device,
            dtype=dtype,
        )
        hdfe_full[1:] = torch.sqrt(torch.clamp(hdfe_diag, min=0.0))
        self.params["fe_se"] = fe_se
        self.params["fe_se_diag"] = hdfe_full

    def _fit_panel_glm(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        init_params: Optional[Any],
        verbose: bool,
        fe: Optional[Sequence[Any]],
        fe_design: Optional[Sequence[Any]],
        hdfe_index: Optional[int],
        offset: Optional[Any],
        batch_size: Optional[int],
        stderr: bool,
    ) -> "MaximumLikelihoodEstimator":
        """Shared FE-aware GLM fitting path for logistic and Poisson models."""
        X = _to_tensor(X, self.device)
        y = _to_tensor(y, self.device, dtype=X.dtype)

        n_obs, n_features = X.shape
        if batch_size is not None and batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if batch_size is not None and batch_size > n_obs:
            batch_size = n_obs

        offset_tensor = None
        if offset is not None:
            offset_tensor = _to_tensor(offset, self.device, dtype=X.dtype).flatten()
            if offset_tensor.shape[0] != n_obs:
                raise ValueError("`offset` must have one entry per observation.")

        fe_blocks = self._canonicalize_fe_inputs(
            fe=fe,
            fe_design=fe_design,
            n_obs=n_obs,
            dtype=X.dtype,
        )
        current_params = (
            self._initialize_panel_parameters(
                n_features=n_features,
                fe_blocks=fe_blocks,
                dtype=X.dtype,
                init_params=init_params,
            )
            .clone()
            .requires_grad_(True)
        )

        if _is_lbfgs_optimizer(self.optimizer_class):
            if batch_size is not None and batch_size < n_obs:
                raise ValueError("LBFGS only supports full-batch FE-GLM fitting.")
            optimizer = self.optimizer_class(
                [current_params],
                max_iter=20,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-9,
                tolerance_change=torch.finfo(X.dtype).eps,
            )
        else:
            optimizer = self.optimizer_class([current_params])

        self.history["loss"] = []

        for i in range(self.maxiter):
            if _is_lbfgs_optimizer(self.optimizer_class):

                def closure():
                    optimizer.zero_grad()
                    loss = self._panel_objective(
                        params=current_params,
                        X=X,
                        y=y,
                        fe_blocks=fe_blocks,
                        offset=offset_tensor,
                    )
                    loss.backward()
                    return loss

                loss_val = optimizer.step(closure)
                epoch_loss = float(loss_val.item())
            else:
                if batch_size is not None and batch_size < n_obs:
                    permutation = torch.randperm(n_obs, device=self.device)
                    epoch_loss = 0.0
                    total_count = 0
                    for start in range(0, n_obs, batch_size):
                        batch_index = permutation[start : start + batch_size]
                        optimizer.zero_grad()
                        loss = self._panel_objective(
                            params=current_params,
                            X=X,
                            y=y,
                            fe_blocks=fe_blocks,
                            offset=offset_tensor,
                            indices=batch_index,
                        )
                        loss.backward()
                        optimizer.step()

                        batch_n = int(batch_index.numel())
                        epoch_loss += float(loss.item()) * batch_n
                        total_count += batch_n
                    epoch_loss /= max(total_count, 1)
                else:
                    optimizer.zero_grad()
                    loss = self._panel_objective(
                        params=current_params,
                        X=X,
                        y=y,
                        fe_blocks=fe_blocks,
                        offset=offset_tensor,
                    )
                    loss.backward()
                    optimizer.step()
                    epoch_loss = float(loss.item())

            self.history["loss"].append(epoch_loss)

            if i > 10 and self.tol > 0:
                loss_change = abs(
                    self.history["loss"][-2] - self.history["loss"][-1]
                ) / (abs(self.history["loss"][-2]) + 1e-8)
                if loss_change < self.tol:
                    if verbose:
                        print(f"Convergence tolerance {self.tol} met at iteration {i}.")
                    break

        coef, fe_coef = self._unpack_panel_parameters(
            current_params.detach(), n_features, fe_blocks
        )
        self.params = {"coef": coef}
        if fe_blocks:
            self.params["fe_coef"] = [values.detach() for values in fe_coef]

        self.iterations_run = i + 1
        self._fitted_X = X.detach()
        self._fitted_y = y.detach()
        self._fit_context = {
            "X": X.detach(),
            "y": y.detach(),
            "fe_blocks": fe_blocks,
            "offset": None if offset_tensor is None else offset_tensor.detach(),
            "hdfe_index": hdfe_index,
        }

        if stderr:
            self._compute_panel_standard_errors()

        return self

    def to(self, device):
        """Move model parameters and fitted data to device."""
        super().to(device)
        if self.params:
            moved_params = {}
            for key, value in self.params.items():
                if isinstance(value, torch.Tensor):
                    moved_params[key] = value.to(self.device)
                elif isinstance(value, list):
                    moved_params[key] = [
                        item.to(self.device) if isinstance(item, torch.Tensor) else item
                        for item in value
                    ]
                else:
                    moved_params[key] = value
            self.params = moved_params
        if self._fitted_X is not None:
            self._fitted_X = self._fitted_X.to(device)
        if self._fitted_y is not None:
            self._fitted_y = self._fitted_y.to(device)

        if self._fit_context is not None:
            moved_context = dict(self._fit_context)
            moved_context["X"] = moved_context["X"].to(device)
            moved_context["y"] = moved_context["y"].to(device)
            moved_context["fe_blocks"] = [
                {
                    **block,
                    "codes": block["codes"].to(device),
                    "levels": block["levels"].to(device),
                }
                for block in moved_context["fe_blocks"]
            ]
            if moved_context["offset"] is not None:
                moved_context["offset"] = moved_context["offset"].to(device)
            self._fit_context = moved_context

        return self

    def _compute_standard_errors(self) -> None:
        """
        Compute standard errors using the Fisher information matrix.

        Calculates the variance-covariance matrix as the inverse of the Fisher
        information matrix, then extracts standard errors as the square root
        of the diagonal elements.

        Updates self.params with:
            - se: Standard errors (vector)
            - vcov: Variance-covariance matrix

        If computation fails (e.g., singular Fisher matrix), sets both to None.
        """
        if self._fitted_X is None or self._fitted_y is None:
            return

        try:
            fisher_info = self._compute_fisher_information(
                self.params["coef"], self._fitted_X, self._fitted_y
            )
            fisher_inv, ridge = _safe_inverse(fisher_info)
            self.params["se"] = torch.sqrt(torch.clamp(torch.diag(fisher_inv), min=0.0))
            self.params["vcov"] = fisher_inv
            self.params["inference_ridge"] = ridge

        except Exception as exc:
            print(f"Warning: Could not compute standard errors: {exc}")
            self.params["se"] = None
            self.params["vcov"] = None

    @abstractmethod
    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Fisher information matrix for standard error calculation.

        The Fisher information matrix is the expected value of the Hessian
        of the negative log-likelihood, and its inverse gives the variance-
        covariance matrix of the maximum likelihood estimators.

        Args:
            params: Model parameters of shape (n_features,).
            X: Design matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).

        Returns:
            Fisher information matrix of shape (n_features, n_features).
        """
        raise NotImplementedError

    def summary(self, alpha: float = 0.05) -> None:
        """
        Print a formatted summary of model estimation results.

        Displays a comprehensive table including parameter estimates, standard
        errors, t-statistics, p-values, and confidence intervals (if standard
        errors are available).

        Args:
            alpha: Significance level for confidence intervals (default: 0.05
                   for 95% confidence intervals).

        Examples:
            >>> model.fit(X, y)
            >>> model.summary(alpha=0.01)  # 99% confidence intervals
        """
        if not self.params or "coef" not in self.params:
            print("Model has not been fitted yet.")
            return

        print(f"{self.__class__.__name__} Results")
        print("=" * 50)
        print(f"Optimizer: {_optimizer_display_name(self.optimizer_class)}")
        if hasattr(self, "iterations_run"):
            print(f"Optimization: {self.iterations_run}/{self.maxiter} iterations")
        if self.history["loss"]:
            print(f"Final Mean Log-Likelihood: {-self.history['loss'][-1]:.4f}")
        print(f"Device: {self.device}")

        n_obs = self._fitted_X.shape[0] if self._fitted_X is not None else "Unknown"
        print(f"No. Observations: {n_obs}")

        if self._fit_context is not None and self._fit_context["fe_blocks"]:
            levels = [block["n_levels"] for block in self._fit_context["fe_blocks"]]
            print(f"Fixed Effects Blocks: {len(levels)}")
            print(f"FE Levels: {levels}")
            print(
                "Inference: "
                + (
                    "Fisher block path (hdfe)"
                    if self._fit_context["hdfe_index"] is not None
                    else "Full Fisher information"
                )
            )

        print("\n" + "=" * 50)

        coef = self.params["coef"].detach().cpu().numpy().ravel()

        if self.params.get("se") is not None:
            se = self.params["se"].detach().cpu().numpy()
            t_stats = coef / se
            p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

            critical_val = stats.norm.ppf(1 - alpha / 2)
            ci_lower = coef - critical_val * se
            ci_upper = coef + critical_val * se

            print(
                f"{'Variable':<12} {'Coef.':<10} {'Std.Err.':<10} {'t':<8} {'P>|t|':<8} {'[{:.1f}%'.format((1 - alpha) * 100):<8} {'Conf. Interval]':<8}"
            )
            print("-" * 70)

            for i in range(len(coef)):
                var_name = f"x{i}" if i > 0 else "const" if i == 0 else f"x{i}"
                print(
                    f"{var_name:<12} {coef[i]:<10.4f} {se[i]:<10.4f} {t_stats[i]:<8.3f} {p_values[i]:<8.3f} {ci_lower[i]:<8.3f} {ci_upper[i]:<8.3f}"
                )
        else:
            print(f"{'Variable':<12} {'Coef.':<10}")
            print("-" * 22)
            for i in range(len(coef)):
                var_name = f"x{i}" if i > 0 else "const" if i == 0 else f"x{i}"
                print(f"{var_name:<12} {coef[i]:<10.4f}")
            print("\nNote: Standard errors could not be computed.")

        print("=" * 50)


class LogisticRegression(MaximumLikelihoodEstimator):
    """
    Logistic Regression for binary classification with MLE estimation.

    Estimates the relationship between a binary outcome and covariates using
    the logistic (sigmoid) link function. Provides asymptotically efficient
    maximum likelihood estimates with proper standard errors.

    The model assumes: P(y=1|X) = σ(X'β) where σ is the sigmoid function.

    Examples:
        >>> model = LogisticRegression(optimizer=torch.optim.LBFGS, maxiter=100)
        >>> X = torch.randn(1000, 5)
        >>> y = (torch.randn(1000) > 0).float()
        >>> model.fit(X, y)
        >>> model.summary()
    """

    def _supports_panel_glm(self) -> bool:
        return True

    def _per_observation_nll_from_eta(
        self,
        linear_predictor: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return -(
            y * torch.nn.functional.logsigmoid(linear_predictor)
            + (1 - y) * torch.nn.functional.logsigmoid(-linear_predictor)
        )

    def _information_weights_from_eta(
        self,
        linear_predictor: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(linear_predictor)
        return probs * (1 - probs)

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        logits = X @ params
        return torch.sum(self._per_observation_nll_from_eta(logits, y))

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        logits = X @ params
        weights = self._information_weights_from_eta(logits)
        weighted_X = X * weights.unsqueeze(1)
        return weighted_X.T @ X

    def predict_proba(
        self,
        X: torch.Tensor,
        fe: Optional[Sequence[Any]] = None,
        fe_design: Optional[Sequence[Any]] = None,
        offset: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Predict class probabilities P(y=1|X).

        Args:
            X: Design matrix of shape (n_samples, n_features).
            fe: Optional FE inputs for FE-aware models.
            fe_design: Optional sparse FE design inputs for FE-aware models.
            offset: Optional offset vector for FE-aware models.

        Returns:
            Predicted probabilities of shape (n_samples,).
        """
        logits = self._predict_linear_predictor(
            X=X,
            fe=fe,
            fe_design=fe_design,
            offset=offset,
        )
        return torch.sigmoid(logits)

    def predict(
        self,
        X: torch.Tensor,
        threshold: float = 0.5,
        fe: Optional[Sequence[Any]] = None,
        fe_design: Optional[Sequence[Any]] = None,
        offset: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Predict binary class labels.

        Args:
            X: Design matrix of shape (n_samples, n_features).
            threshold: Probability threshold for assigning class 1.
            fe: Optional FE inputs for FE-aware models.
            fe_design: Optional sparse FE design inputs for FE-aware models.
            offset: Optional offset vector for FE-aware models.

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        probas = self.predict_proba(
            X=X,
            fe=fe,
            fe_design=fe_design,
            offset=offset,
        )
        return (probas >= threshold).to(torch.int32)


class PoissonRegression(MaximumLikelihoodEstimator):
    """
    Poisson Regression for count data with MLE estimation.

    Estimates the relationship between count outcomes and covariates using
    the log link function. Assumes the conditional mean equals the conditional
    variance: E[y|X] = Var[y|X] = exp(X'β).

    Examples:
        >>> model = PoissonRegression(maxiter=1000, tol=1e-6)
        >>> X = torch.randn(500, 4)
        >>> true_beta = torch.tensor([1.0, -0.5, 0.8, 0.3])
        >>> y = torch.poisson(torch.exp(X @ true_beta))
        >>> model.fit(X, y)
        >>> model.summary()
    """

    def _supports_panel_glm(self) -> bool:
        return True

    def _per_observation_nll_from_eta(
        self,
        linear_predictor: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        mean = torch.exp(linear_predictor)
        return mean - y * linear_predictor

    def _information_weights_from_eta(
        self,
        linear_predictor: torch.Tensor,
    ) -> torch.Tensor:
        return torch.exp(linear_predictor)

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        linear_predictor = X @ params
        return torch.sum(self._per_observation_nll_from_eta(linear_predictor, y))

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        linear_predictor = X @ params
        lambda_i = self._information_weights_from_eta(linear_predictor)
        weighted_X = X * lambda_i.unsqueeze(1)
        return weighted_X.T @ X

    def predict(
        self,
        X: torch.Tensor,
        fe: Optional[Sequence[Any]] = None,
        fe_design: Optional[Sequence[Any]] = None,
        offset: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Predict expected counts E[y|X] = exp(X'β).

        Args:
            X: Design matrix of shape (n_samples, n_features).
            fe: Optional FE inputs for FE-aware models.
            fe_design: Optional sparse FE design inputs for FE-aware models.
            offset: Optional offset vector for FE-aware models.

        Returns:
            Predicted expected counts of shape (n_samples,).
        """
        linear_predictor = self._predict_linear_predictor(
            X=X,
            fe=fe,
            fe_design=fe_design,
            offset=offset,
        )
        return torch.exp(linear_predictor)
