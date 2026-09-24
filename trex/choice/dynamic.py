"""
Dynamic discrete choice models with forward-looking agents.

This module implements structural models where agents solve sequential decision
problems under uncertainty, including the Rust (1987) nested fixed point method
and Hotz-Miller (1993) CCP inversion approach.

References:
    Rust (1987): "Optimal replacement of GMC bus engines"
    Hotz & Miller (1993): "Conditional choice probabilities and estimation"
    Aguirregabiria & Mira (2002): "Nested pseudo-likelihood"
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Literal, Union

import torch
import torch.nn as nn

from .base import ChoiceModel
from .ccp_estimators import estimate_ccps


@dataclass
class DynamicChoiceData:
    """
    Container for dynamic choice panel data.

    Attributes:
        states: (n_obs,) observed states x_t
        actions: (n_obs,) observed actions a_t
        next_states: (n_obs,) observed next states x_{t+1}
        individual_ids: (n_obs,) panel identifier
        time_periods: (n_obs,) time index
        covariates: Optional (n_obs, n_features) additional state variables
    """

    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    individual_ids: torch.Tensor
    time_periods: torch.Tensor
    covariates: Optional[torch.Tensor] = None

    def validate(self) -> None:
        """
        Check data integrity and stationarity.

        Raises:
            ValueError: If data shapes are inconsistent or contain invalid values
        """
        n_obs = len(self.states)
        if n_obs == 0:
            raise ValueError("Choice data must not be empty")
        for name in (
            "states",
            "actions",
            "next_states",
            "individual_ids",
            "time_periods",
        ):
            value = getattr(self, name)
            if (
                value.ndim != 1
                or not torch.isfinite(value).all()
                or (value.is_floating_point() and not torch.equal(value, value.round()))
            ):
                raise ValueError(f"{name} must be a finite integer vector")
        if torch.any(self.next_states < 0):
            raise ValueError("next_states must be nonnegative")
        if not all(
            len(x) == n_obs
            for x in [
                self.actions,
                self.next_states,
                self.individual_ids,
                self.time_periods,
            ]
        ):
            raise ValueError("All tensors must have the same length")

        if self.covariates is not None and len(self.covariates) != n_obs:
            raise ValueError(f"Covariates length {len(self.covariates)} != {n_obs}")

        # Check for negative values in states/actions
        if torch.any(self.states < 0) or torch.any(self.actions < 0):
            raise ValueError("States and actions must be non-negative integers")

    def to_dict(self) -> dict:
        """Convert to dict for compatibility with existing code."""
        result = {
            "states": self.states,
            "actions": self.actions,
            "next_states": self.next_states,
            "individual_ids": self.individual_ids,
            "time_periods": self.time_periods,
        }
        if self.covariates is not None:
            result["covariates"] = self.covariates
        return result


class DynamicChoiceModel(ChoiceModel):
    """
    Base class for dynamic discrete choice models.

    Implements common infrastructure for:
    - State space management
    - Transition probability specification
    - Value function iteration
    - Choice probability computation

    Subclasses implement specific estimation methods (NFP, CCP, NPL).
    """

    def __init__(
        self,
        n_states: int,
        n_choices: int,
        discount_factor: float,
        transition_type: Literal["parametric", "nonparametric"] = "parametric",
        optimizer: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        maxiter: int = 1000,
        tol: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize dynamic choice model.

        Args:
            n_states: Number of discrete states in X
            n_choices: Number of discrete actions in J
            discount_factor: β ∈ (0,1), typically 0.95-0.9999
            transition_type: How to model P(x'|x,a)
            optimizer: PyTorch optimizer class
            maxiter: Maximum iterations for value iteration
            tol: Convergence tolerance
            device: Device for computations (auto-detects if None)
        """
        # Initialize these first to prevent AttributeError during super().__init__ calls
        # that might try to access self.utility_fn via self.to()
        self.utility_fn: Optional[nn.Module] = None
        self.utility_params: Optional[dict] = None

        super().__init__(optimizer=optimizer, maxiter=maxiter, tol=tol, device=device)
        if n_states < 1 or n_choices < 2 or not 0 <= discount_factor < 1:
            raise ValueError(
                "Need positive state count, >=2 choices, and 0 <= discount_factor < 1"
            )
        if maxiter < 1 or tol <= 0:
            raise ValueError("maxiter and tol must be positive")
        self.n_states = n_states
        self.n_choices = n_choices
        self.discount_factor = discount_factor
        self.transition_type = transition_type

        # Will be set by set_transition_probabilities()
        self.transition_matrix: Optional[torch.Tensor] = None
        self.transition_params: Optional[dict] = None

    def to(self, device: Union[torch.device, str]) -> "DynamicChoiceModel":
        """Move model and transition matrix to specified device."""
        super().to(device)
        if getattr(self, "transition_matrix", None) is not None:
            self.transition_matrix = self.transition_matrix.to(self.device)
        if self.utility_fn is not None and isinstance(self.utility_fn, nn.Module):
            self.utility_fn.to(self.device)
        for name in (
            "all_states_features",
            "ccp_hat",
            "inv_matrix",
            "entropy_term",
            "inversion_matrix",
        ):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(self.device))
        return self

    def fit(
        self,
        data: Union[DynamicChoiceData, dict],
        init_params: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> "DynamicChoiceModel":
        """
        Fit the model to the data.
        Overriden to handle DynamicChoiceData.
        """
        if isinstance(data, DynamicChoiceData):
            data.validate()
            data_dict = data.to_dict()
        else:
            data_dict = dict(data)

        if "states" not in data_dict or "actions" not in data_dict:
            raise ValueError("Data must contain 'states' and 'actions'")

        # Move data tensors to device
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(self.device)

        if self.utility_fn is None or self.transition_matrix is None:
            raise ValueError(
                "Set flow utility and transition probabilities before fitting"
            )
        for name, upper in (("states", self.n_states), ("actions", self.n_choices)):
            values = data_dict[name]
            if (
                values.ndim != 1
                or not len(values)
                or not torch.isfinite(values).all()
                or torch.any(values < 0)
                or torch.any(values >= upper)
                or (
                    values.is_floating_point()
                    and not torch.equal(values, values.round())
                )
            ):
                raise ValueError(f"{name} must contain valid integer indices")
            data_dict[name] = values.long()
        if len(data_dict["states"]) != len(data_dict["actions"]):
            raise ValueError("states and actions must have the same length")
        if "all_states_features" in data_dict:
            self.all_states_features = (
                data_dict["all_states_features"].detach().clone().to(self.device)
            )

        # Gather initial params from utility_fn if not provided
        if init_params is None:
            if self.utility_fn is None:
                raise ValueError("utility_fn must be set before fitting.")

            flat_params = []
            for p in self.utility_fn.parameters():
                flat_params.append(p.view(-1))

            if not flat_params:
                # Utility might not have parameters (e.g. fixed)
                init_params_val = torch.tensor([], device=self.device)
            else:
                init_params_val = torch.cat(flat_params)
        else:
            init_params_val = init_params.to(self.device)

        if init_params_val.numel() == 0:
            print("No parameters to optimize.")
            return self

        current_params = init_params_val.detach().clone().requires_grad_(True)

        # Optimizer
        if self.optimizer_class == torch.optim.LBFGS:
            optimizer = self.optimizer_class(
                [current_params],
                max_iter=20,
                line_search_fn="strong_wolfe",
                tolerance_grad=min(self.tol, 1e-8),
                tolerance_change=1e-12,
            )
        else:
            optimizer = self.optimizer_class([current_params])

        self.history["loss"] = []

        for i in range(self.maxiter):

            def closure():
                optimizer.zero_grad()
                loss = self._negative_log_likelihood(current_params, data_dict) / len(
                    data_dict["states"]
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Dynamic-choice objective is not finite")
                loss.backward()
                return loss

            if self.optimizer_class == torch.optim.LBFGS:
                loss_val = optimizer.step(closure)
            else:
                loss_val = closure()
                optimizer.step()

            self.history["loss"].append(loss_val.item())

            # Convergence check
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

        # Update utility_fn parameters
        idx = 0
        for p in self.utility_fn.parameters():
            numel = p.numel()
            p.data.copy_(self.params["coef"][idx : idx + numel].view_as(p))
            idx += numel

        # We don't have standard X, y for _compute_standard_errors in the base class way.
        # Subclasses will need to handle SE computation or we adapt it later.

        return self

    def set_transition_probabilities(
        self,
        transition_matrix: torch.Tensor,
        transition_params: Optional[dict] = None,
    ) -> None:
        """
        Specify state transition mechanism P(x_{t+1} | x_t, a_t).

        Args:
            transition_matrix: (n_states, n_choices, n_states) tensor
                              P[x, a, x'] = Pr(x_{t+1}=x' | x_t=x, a_t=a)
            transition_params: Parameters φ if transitions are parametric

        Raises:
            ValueError: If transition matrix shape is incorrect
        """
        expected_shape = (self.n_states, self.n_choices, self.n_states)
        if transition_matrix.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {transition_matrix.shape}"
            )

        if not torch.isfinite(transition_matrix).all() or torch.any(
            transition_matrix < 0
        ):
            raise ValueError("Transition probabilities must be finite and nonnegative")
        # Verify probabilities sum to 1
        row_sums = transition_matrix.sum(dim=2)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            raise ValueError("Transition probabilities must sum to 1 for each (x,a)")

        self.transition_matrix = transition_matrix.to(self.device)
        self.transition_params = transition_params

    def set_flow_utility(
        self,
        utility_fn: nn.Module,
        utility_params: Optional[dict] = None,
    ) -> None:
        """
        Specify per-period utility ū(x, a; θ).

        Args:
            utility_fn: Function computing u(x, a; θ)
            utility_params: Initial values for θ
        """
        self.utility_fn = utility_fn.to(self.device)
        self.utility_params = utility_params

    def _bellman_operator(
        self,
        v_bar: torch.Tensor,
        flow_utility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Bellman operator T to value functions.

        Computes:
            T v̄_j(x) = ū_j(x) + β Σ_{x'} P(x'|x,a=j) log[Σ_k exp(v̄_k(x'))]

        Args:
            v_bar: (n_states, n_choices) current value functions
            flow_utility: (n_states, n_choices) per-period utilities ū(x,a)

        Returns:
            T_v_bar: (n_states, n_choices) updated value functions
        """
        if self.transition_matrix is None:
            raise ValueError("Must call set_transition_probabilities() first")

        # Compute log-sum-exp for continuation value
        # emax(x') = log[Σ_k exp(v̄_k(x'))]
        emax = torch.logsumexp(v_bar, dim=1)  # (n_states,)

        # Expected continuation value for each (x, a)
        # E[emax(x') | x, a] = Σ_{x'} P(x'|x,a) emax(x')
        # Shape: (n_states, n_choices, n_states) @ (n_states,) -> (n_states, n_choices)
        continuation_value = torch.einsum(
            "xay,y->xa", self.transition_matrix.to(dtype=emax.dtype), emax
        )

        # Bellman update
        T_v_bar = flow_utility + self.discount_factor * continuation_value

        return T_v_bar

    def solve_value_functions(
        self,
        flow_utility: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
    ) -> torch.Tensor:
        """
        Solve value function fixed point via successive approximations.

        Iterates: v̄^{k+1} = T(v̄^k) until convergence, where T is
        the Bellman operator.

        Args:
            flow_utility: (n_states, n_choices) per-period utilities ū(x,a)
            tol: Convergence tolerance for sup-norm
            max_iter: Maximum iterations

        Returns:
            v_bar: (n_states, n_choices) converged value functions

        Raises:
            RuntimeError: If iteration does not converge
        """
        v_bar = torch.zeros(
            self.n_states, self.n_choices, device=self.device, dtype=flow_utility.dtype
        )

        for iteration in range(max_iter):
            v_bar_new = self._bellman_operator(v_bar, flow_utility)
            error = torch.max(torch.abs(v_bar_new - v_bar)).item()

            if error < tol:
                return v_bar_new

            v_bar = v_bar_new

        raise RuntimeError(
            f"Value iteration did not converge after {max_iter} iterations. "
            f"Final error: {error:.6e}"
        )

    def _compute_choice_probs(
        self,
        v_bar: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute conditional choice probabilities from value functions.

        Under Type I extreme value errors:
            P(a|x) = exp(v̄_a(x)) / Σ_j exp(v̄_j(x))

        Args:
            v_bar: (n_states, n_choices) value functions
            states: (n_obs,) state indices

        Returns:
            probs: (n_obs, n_choices) choice probabilities
        """
        # Follow the supplied value tensor's device, including standalone use.
        states = states.to(v_bar.device)
        if (
            states.ndim != 1
            or not torch.isfinite(states).all()
            or torch.any(states < 0)
            or torch.any(states >= self.n_states)
            or (states.is_floating_point() and not torch.equal(states, states.round()))
        ):
            raise ValueError("states must be valid integer state indices")
        v_obs = v_bar[states.long()]  # (n_obs, n_choices)

        # Softmax to get probabilities
        probs = torch.softmax(v_obs, dim=1)

        return probs

    def _unpack_params(
        self, params: torch.Tensor
    ) -> tuple[dict, Optional[torch.Tensor]]:
        """
        Unpack concatenated parameter tensor into utility (theta) and transition (phi) parameters.
        """
        if isinstance(self.utility_fn, ReplacementUtility):
            # Assuming params is [theta_maintenance, theta_replacement_cost]
            theta_dict = {
                "theta_maintenance": params[0],
                "theta_replacement_cost": params[1],
            }
            phi = None
        elif isinstance(self.utility_fn, LinearFlowUtility):
            theta_dict = {"theta": params.reshape(self.utility_fn.theta.shape)}
            phi = None
        else:
            raise NotImplementedError("Unsupported flow utility parameterization")

        return theta_dict, phi

    def _get_coef_params(self) -> torch.Tensor:
        """
        Return fitted coefficient parameters in tensor form.

        Supports both the canonical internal representation
        (``self.params == {"coef": tensor}``) and a legacy direct tensor assignment
        (``self.params == tensor``), which appears in older tests/examples.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before accessing parameters")

        if isinstance(self.params, dict):
            if "coef" not in self.params:
                raise ValueError("Fitted parameters are missing the 'coef' entry")
            coef = self.params["coef"]
        elif isinstance(self.params, torch.Tensor):
            coef = self.params
        else:
            raise TypeError(
                "Expected fitted parameters to be a dict with key 'coef' "
                f"or a torch.Tensor, got {type(self.params).__name__}"
            )

        if not isinstance(coef, torch.Tensor):
            raise TypeError(
                f"Expected coefficient parameters to be a torch.Tensor, got {type(coef).__name__}"
            )

        return coef.to(self.device)

    @abstractmethod
    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        data: dict,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _compute_fisher_information(self) -> torch.Tensor:
        raise NotImplementedError

    def _flow_utility(self, params, data=None, policy_change=None):
        theta, _ = self._unpack_params(params)
        change = policy_change or {}
        if set(change) - set(theta):
            raise ValueError(f"Unknown policy parameters: {set(change) - set(theta)}")
        theta = {
            k: torch.as_tensor(change.get(k, v), dtype=params.dtype, device=self.device)
            for k, v in theta.items()
        }
        if isinstance(self.utility_fn, ReplacementUtility):
            return self.utility_fn(
                torch.arange(self.n_states, device=self.device), **theta
            )
        if isinstance(self.utility_fn, LinearFlowUtility):
            features = (data or {}).get(
                "all_states_features", getattr(self, "all_states_features", None)
            )
            if features is None:
                raise ValueError("LinearFlowUtility requires all_states_features")
            if features.shape != (self.n_states, self.utility_fn.theta.shape[0]):
                raise ValueError("all_states_features has incompatible shape")
            return self.utility_fn(
                features.to(device=self.device, dtype=params.dtype), **theta
            )
        raise NotImplementedError(
            "Supported utilities are ReplacementUtility and LinearFlowUtility"
        )

    def predict_proba(self, states):
        """Structural/equilibrium probabilities using fitted utility coefficients.

        For Hotz-Miller this re-solves the Bellman equation; it does not return
        the empirical first-stage CCPs or freeze them under policy changes.
        """
        flow = self._flow_utility(self._get_coef_params())
        values = self.solve_value_functions(flow, tol=self.tol)
        return self._compute_choice_probs(values, states.to(self.device))

    def _simulate_values(self, initial_states, n_periods, values, rng=None):
        if not isinstance(n_periods, int) or n_periods < 1:
            raise ValueError("n_periods must be a positive integer")
        initial_states = initial_states.to(self.device)
        if (
            initial_states.ndim != 1
            or torch.any(initial_states < 0)
            or torch.any(initial_states >= self.n_states)
        ):
            raise ValueError("initial_states must be valid state indices")
        if initial_states.is_floating_point() and not torch.equal(
            initial_states, initial_states.round()
        ):
            raise ValueError("initial_states must be integer indices")
        current = initial_states.long()
        paths = torch.empty(
            (len(current), n_periods + 1), dtype=torch.long, device=self.device
        )
        actions = torch.empty(
            (len(current), n_periods), dtype=torch.long, device=self.device
        )
        paths[:, 0] = current
        probs = torch.softmax(values, dim=1)
        for period in range(n_periods):
            chosen = torch.multinomial(probs[current], 1, generator=rng).squeeze(1)
            current = torch.multinomial(
                self.transition_matrix[current, chosen], 1, generator=rng
            ).squeeze(1)
            actions[:, period] = chosen
            paths[:, period + 1] = current
        return paths, actions

    def simulate(self, initial_states, n_periods, rng=None):
        values = self.solve_value_functions(
            self._flow_utility(self._get_coef_params()), tol=self.tol
        )
        return self._simulate_values(initial_states, n_periods, values, rng)

    def counterfactual(self, data, policy_change, rng=None, *, n_periods=None):
        """Re-solve and simulate altered utilities without mutating fitted state.

        Starts with each individual's earliest observation. Default horizon is
        the largest number of observed periods per individual, not max(time ID).
        Pass n_periods explicitly for unbalanced-panel policy simulations.
        """
        data.validate()
        values = self.solve_value_functions(
            self._flow_utility(self._get_coef_params(), policy_change=policy_change),
            tol=self.tol,
        )
        initial, counts = [], []
        for identifier in torch.unique(data.individual_ids):
            indices = torch.where(data.individual_ids == identifier)[0]
            initial.append(data.states[indices[data.time_periods[indices].argmin()]])
            counts.append(len(torch.unique(data.time_periods[indices])))
        initial_states = torch.stack(initial).to(self.device)
        horizon = max(counts) if n_periods is None else n_periods
        states, actions = self._simulate_values(initial_states, horizon, values, rng)
        return {
            "simulated_states": states,
            "simulated_actions": actions,
            "counterfactual_v_bar": values,
            "counterfactual_probs": torch.softmax(values, dim=1),
        }


class RustNFP(DynamicChoiceModel):
    """
    Rust (1987) full maximum likelihood via nested fixed point (NFP).
    """

    def __init__(
        self,
        n_states: int,
        n_choices: int,
        discount_factor: float,
        transition_type: Literal["parametric", "nonparametric"] = "nonparametric",
        optimizer: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        maxiter: int = 1000,
        tol: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
        estimate_transitions: bool = False,
    ):
        super().__init__(
            n_states=n_states,
            n_choices=n_choices,
            discount_factor=discount_factor,
            transition_type=transition_type,
            optimizer=optimizer,
            maxiter=maxiter,
            tol=tol,
            device=device,
        )
        self.estimate_transitions = estimate_transitions

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        data: dict,
    ) -> torch.Tensor:
        if self.estimate_transitions:
            raise NotImplementedError("Joint transition estimation is not implemented")
        flow = self._flow_utility(params, data)
        values = self.solve_value_functions(flow, tol=min(self.tol, 1e-8))
        log_probs = torch.log_softmax(values[data["states"]], dim=1)
        return -log_probs.gather(
            1, data["actions"].to(device=self.device, dtype=torch.long)[:, None]
        ).sum()

    def _compute_fisher_information(self):
        raise NotImplementedError("RustNFP inference is not implemented")


class HotzMillerCCP(DynamicChoiceModel):
    """
    Hotz & Miller (1993) CCP inversion estimator.

    Two-stage procedure:
    1. Estimate conditional choice probabilities P̂(a|x) nonparametrically
    2. Invert CCPs to recover value function differences
    3. Estimate structural parameters via matching/MLE
    """

    def __init__(
        self,
        n_states: int,
        n_choices: int,
        discount_factor: float,
        transition_type: Literal["parametric", "nonparametric"] = "nonparametric",
        optimizer: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        maxiter: int = 1000,
        tol: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__(
            n_states=n_states,
            n_choices=n_choices,
            discount_factor=discount_factor,
            transition_type=transition_type,
            optimizer=optimizer,
            maxiter=maxiter,
            tol=tol,
            device=device,
        )
        self.ccp_hat: Optional[torch.Tensor] = None
        self.inv_matrix: Optional[torch.Tensor] = None
        self.entropy_term: Optional[torch.Tensor] = None

    def estimate_ccps(self, data: DynamicChoiceData) -> None:
        """Estimate CCPs from data."""
        data.validate()
        self.ccp_hat = estimate_ccps(
            data.states, data.actions, self.n_states, self.n_choices
        ).to(self.device)
        self._automatic_ccps = True

    def _precompute_inversion_matrices(self) -> None:
        """
        Precompute constant matrices for CCP inversion:
        (I - beta * M)^-1 and entropy term.
        """
        if self.ccp_hat is None:
            raise ValueError("CCPs must be estimated first")
        if self.transition_matrix is None:
            raise ValueError("Transition matrix must be set")

        # M(x, x') = sum_a P_hat(a|x) * P(x'|x,a)
        # ccp_hat: (n_states, n_choices)
        # transition: (n_states, n_choices, n_states)
        self.ccp_hat = self.ccp_hat.to(
            device=self.device, dtype=self.transition_matrix.dtype
        )
        if (
            self.ccp_hat.shape != (self.n_states, self.n_choices)
            or not torch.isfinite(self.ccp_hat).all()
            or torch.any(self.ccp_hat < 0)
            or not torch.allclose(
                self.ccp_hat.sum(1),
                torch.ones(self.n_states, device=self.device, dtype=self.ccp_hat.dtype),
            )
        ):
            raise ValueError(
                "CCPs must be finite probability rows with shape (n_states,n_choices)"
            )
        M = torch.einsum("xa,xay->xy", self.ccp_hat, self.transition_matrix)

        # I - beta * M
        I = torch.eye(self.n_states, device=self.device, dtype=M.dtype)
        A = I - self.discount_factor * M

        # Invert
        self.inversion_matrix = A
        self.inv_matrix = (
            None  # retained attribute; solve the system instead of inverting
        )

        # Entropy term: sum_a P_hat(a|x) * ln P_hat(a|x)
        # Avoid log(0)
        self.entropy_term = torch.special.xlogy(self.ccp_hat, self.ccp_hat).sum(dim=1)

    def fit(
        self,
        data: Union[DynamicChoiceData, dict],
        init_params: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> "HotzMillerCCP":
        if isinstance(data, DynamicChoiceData):
            # Honor explicitly supplied CCPs; refresh automatically estimated ones.
            if self.ccp_hat is None or getattr(self, "_automatic_ccps", False):
                self.estimate_ccps(data)
        elif isinstance(data, dict) and "ccp_hat" in data:
            self.ccp_hat = data["ccp_hat"].to(self.device)
            self._automatic_ccps = False

        if self.ccp_hat is None or (
            isinstance(data, dict)
            and "ccp_hat" not in data
            and getattr(self, "_automatic_ccps", False)
        ):
            # Fallback: if data is dict but missing ccp_hat, try to estimate from states/actions
            if isinstance(data, dict) and "states" in data and "actions" in data:
                self.ccp_hat = estimate_ccps(
                    data["states"], data["actions"], self.n_states, self.n_choices
                ).to(self.device)
                self._automatic_ccps = True
            else:
                raise ValueError("CCP estimates required for HotzMillerCCP")

        # Precompute matrices
        self._precompute_inversion_matrices()

        return super().fit(data, init_params, verbose)

    def invert_ccps(self, flow_utility: torch.Tensor) -> torch.Tensor:
        """
        Invert CCPs to recover integrated value function V_bar.

        Args:
            flow_utility: (n_states, n_choices) flow utilities

        Returns:
            V_bar_integrated: (n_states,)
        """
        # 2. Compute expected utility term: sum_a P_hat(a|x) * u(x,a)
        exp_utility = torch.sum(self.ccp_hat * flow_utility, dim=1)

        # 3. RHS = exp_utility - entropy_term
        rhs = exp_utility - self.entropy_term

        # 4. Invert to get integrated value function V_bar
        V_bar_integrated = torch.linalg.solve(
            self.inversion_matrix.to(dtype=rhs.dtype), rhs
        )

        return V_bar_integrated

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        data: dict,
    ) -> torch.Tensor:
        """
        Compute pseudo-likelihood using inverted CCPs.
        """
        flow_utility = self._flow_utility(params, data)

        # Invert to get integrated value function V_bar
        V_bar_integrated = self.invert_ccps(flow_utility)

        # 5. Compute conditional value functions v_j(x)
        # v_j(x) = u_j(x) + beta * sum_x' P(x'|x,j) * V_bar(x')
        # P(x'|x,j) * V_bar(x') -> (n_states, n_choices, n_states) @ (n_states,) -> (n_states, n_choices)

        continuation_value = torch.einsum(
            "xay,y->xa",
            self.transition_matrix.to(dtype=V_bar_integrated.dtype),
            V_bar_integrated,
        )

        v_bar_implied = flow_utility + self.discount_factor * continuation_value

        # 6. Pseudo-likelihood
        # Compute choice probabilities from implied v_bar
        # P_model(a|x) = exp(v_bar_implied) / sum exp

        log_probs = torch.log_softmax(v_bar_implied[data["states"]], dim=1)
        return -log_probs.gather(
            1, data["actions"].to(device=self.device, dtype=torch.long)[:, None]
        ).sum()

    def _compute_fisher_information(self) -> torch.Tensor:
        raise NotImplementedError(
            "Fisher information for HotzMillerCCP not yet implemented."
        )


# Utility Specifications
class LinearFlowUtility(nn.Module):
    """
    Linear-in-features flow utility specification.

    The model parameterizes period utility as:

        u(x, a) = f(x)^T theta[:, a]

    where `f(x)` is a feature vector for state `x`.
    """

    def __init__(
        self,
        n_features: int,
        n_choices: int,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize linear utility coefficients.

        Args:
            n_features: Number of features in state representation.
            n_choices: Number of discrete actions.
            device: Torch device for parameter storage.
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if isinstance(device, str) else device
        self.theta = nn.Parameter(
            torch.randn(n_features, n_choices, device=self.device, dtype=torch.float64)
            * 0.01
        )

    def forward(
        self,
        states: torch.Tensor,
        choice: Optional[int] = None,
        theta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate flow utility for states and actions.

        Args:
            states: State feature matrix of shape `(n_obs, n_features)`.
            choice: If provided, returns utility for a single action index.
            theta: Optional coefficient override, used during optimization.

        Returns:
            Utility tensor of shape `(n_obs,)` for single-choice mode or
            `(n_obs, n_choices)` when `choice is None`.
        """
        _theta = theta if theta is not None else self.theta
        states = states.to(device=_theta.device, dtype=_theta.dtype)
        if choice is not None:
            return states @ _theta[:, choice]
        else:
            return states @ _theta


class ReplacementUtility(nn.Module):
    """
    Rust-style bus replacement flow utility.

    Two-action specification:

    - `a=0` (maintain): `u(x,0) = -theta_maintenance * x`
    - `a=1` (replace): `u(x,1) = -theta_replacement_cost`
    """

    def __init__(
        self,
        theta_maintenance: float = 0.001,
        theta_replacement_cost: float = 10.0,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize replacement utility parameters.

        Args:
            theta_maintenance: Marginal maintenance cost per state unit.
            theta_replacement_cost: Fixed utility cost of replacement.
            device: Torch device for parameter storage.
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if isinstance(device, str) else device
        self.theta_maintenance = nn.Parameter(
            torch.tensor(theta_maintenance, device=self.device, dtype=torch.float64)
        )
        self.theta_replacement_cost = nn.Parameter(
            torch.tensor(
                theta_replacement_cost, device=self.device, dtype=torch.float64
            )
        )

    def forward(
        self,
        state: torch.Tensor,
        choice: Optional[int] = None,
        theta_maintenance: Optional[torch.Tensor] = None,
        theta_replacement_cost: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate replacement-model flow utility.

        Args:
            state: Discrete state index tensor.
            choice: Optional action index (`0` maintain, `1` replace).
            theta_maintenance: Optional maintenance parameter override.
            theta_replacement_cost: Optional replacement-cost override.

        Returns:
            Utility tensor for requested action or stacked `(n_obs, 2)` utility.
        """
        _theta_maintenance = (
            theta_maintenance
            if theta_maintenance is not None
            else self.theta_maintenance
        )
        _theta_replacement_cost = (
            theta_replacement_cost
            if theta_replacement_cost is not None
            else self.theta_replacement_cost
        )
        maintain_utility = -_theta_maintenance * state
        replace_utility = -_theta_replacement_cost + torch.zeros_like(state)
        if choice == 0:
            return maintain_utility
        elif choice == 1:
            return replace_utility
        else:
            return torch.stack([maintain_utility, replace_utility], dim=1)

    def get_params(self) -> dict:
        """Return scalar utility parameters as a Python dictionary."""
        return {
            "theta_maintenance": self.theta_maintenance.item(),
            "theta_replacement_cost": self.theta_replacement_cost.item(),
        }

    def set_params(self, theta_maintenance, theta_replacement_cost):
        """Set utility parameters from Python scalars or tensors."""
        if isinstance(theta_maintenance, torch.Tensor):
            self.theta_maintenance.data = theta_maintenance.to(self.device).to(
                torch.float64
            )
        else:
            self.theta_maintenance.data = torch.tensor(
                theta_maintenance, device=self.device, dtype=torch.float64
            )
        if isinstance(theta_replacement_cost, torch.Tensor):
            self.theta_replacement_cost.data = theta_replacement_cost.to(
                self.device
            ).to(torch.float64)
        else:
            self.theta_replacement_cost.data = torch.tensor(
                theta_replacement_cost, device=self.device, dtype=torch.float64
            )
