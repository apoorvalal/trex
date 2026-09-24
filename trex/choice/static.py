import torch

from .base import ChoiceModel


class BinaryLogit(ChoiceModel):
    """
    Binary Logit model for discrete choice analysis.

    This model estimates utility-based binary choice using the logistic
    (sigmoid) link function. Commonly used in structural estimation for
    modeling binary decisions such as product purchase, participation, etc.

    Examples:
        >>> model = BinaryLogit()
        >>> X = torch.randn(100, 3)  # 100 observations, 3 features
        >>> y = torch.randint(0, 2, (100,))
        >>> model.fit(X, y)
        >>> probs = model.predict_proba(X)
    """

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood for binary logit model.

        Args:
            params: Coefficient vector of shape (n_features,).
            X: Design matrix of shape (n_samples, n_features).
            y: Binary choice vector of shape (n_samples,).

        Returns:
            Negative log-likelihood as a scalar tensor.
        """
        logits = X @ params
        return -torch.sum(
            y * torch.nn.functional.logsigmoid(logits)
            + (1 - y) * torch.nn.functional.logsigmoid(-logits)
        )

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Fisher information matrix for binary logit.

        Args:
            params: Coefficient vector of shape (n_features,).
            X: Design matrix of shape (n_samples, n_features).
            y: Binary choice vector (unused but kept for interface consistency).

        Returns:
            Fisher information matrix of shape (n_features, n_features).
        """
        logits = X @ params
        probs = torch.sigmoid(logits)
        weights = probs * (1 - probs)
        weighted_X = X * weights.unsqueeze(1)
        return weighted_X.T @ X

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict choice probabilities for binary logit model.

        Args:
            X: Design matrix of shape (n_samples, n_features).

        Returns:
            Predicted probabilities of choosing option 1, shape (n_samples,).

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")
        logits = X @ self.params["coef"]
        return torch.sigmoid(logits)

    def simulate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate binary choices from the fitted model.

        Args:
            X: Design matrix of shape (n_samples, n_features).

        Returns:
            Simulated binary choices of shape (n_samples,).
        """
        probas = self.predict_proba(X)
        return (torch.rand_like(probas) < probas).to(torch.int32)

    def counterfactual(self, X_new: torch.Tensor) -> dict:
        """
        Compute change in market share for a counterfactual scenario.

        Compares predicted market share (average choice probability) between
        the original fitted data and a counterfactual scenario with different
        covariates.

        Args:
            X_new: Counterfactual design matrix of shape (n_samples, n_features).

        Returns:
            Dictionary containing:
                - market_share_original: Original market share (scalar tensor).
                - market_share_counterfactual: Counterfactual market share.
                - change_in_market_share: Difference in market shares.

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if self._fitted_X is None:
            raise ValueError("Model must be fitted before running counterfactuals.")

        # Original market share
        p_old = self.predict_proba(self._fitted_X)
        market_share_old = torch.mean(p_old)

        # Counterfactual market share
        p_new = self.predict_proba(X_new)
        market_share_new = torch.mean(p_new)

        return {
            "market_share_original": market_share_old,
            "market_share_counterfactual": market_share_new,
            "change_in_market_share": market_share_new - market_share_old,
        }


class BinaryProbit(ChoiceModel):
    """
    Binary Probit model for discrete choice analysis.

    This model estimates utility-based binary choice using the normal
    (probit) link function, assuming normally distributed errors. Often
    used when the choice depends on a latent continuous variable.

    Examples:
        >>> model = BinaryProbit()
        >>> X = torch.randn(100, 3)
        >>> y = torch.randint(0, 2, (100,))
        >>> model.fit(X, y)
        >>> probs = model.predict_proba(X)
    """

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood for binary probit model.

        Uses the standard normal CDF (Φ) for link function with numerical
        stability via the log normal CDF.

        Args:
            params: Coefficient vector of shape (n_features,).
            X: Design matrix of shape (n_samples, n_features).
            y: Binary choice vector of shape (n_samples,).

        Returns:
            Negative log-likelihood as a scalar tensor.
        """
        logits = X @ params
        # log_ndtr retains rare-event gradients even when Phi underflows.
        signed_logits = torch.where(y == 1, logits, -logits)
        return -torch.special.log_ndtr(signed_logits).sum()

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Fisher information matrix for binary probit.

        Args:
            params: Coefficient vector of shape (n_features,).
            X: Design matrix of shape (n_samples, n_features).
            y: Binary choice vector (unused but kept for interface consistency).

        Returns:
            Fisher information matrix of shape (n_features, n_features).
        """
        logits = X @ params
        log_pdf = -0.5 * logits.square() - 0.5 * torch.log(
            logits.new_tensor(2 * torch.pi)
        )
        weights = torch.exp(
            2 * log_pdf
            - torch.special.log_ndtr(logits)
            - torch.special.log_ndtr(-logits)
        )
        weighted_X = X * weights.unsqueeze(1)
        return weighted_X.T @ X

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict choice probabilities for binary probit model.

        Args:
            X: Design matrix of shape (n_samples, n_features).

        Returns:
            Predicted probabilities of choosing option 1, shape (n_samples,).

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")
        logits = X @ self.params["coef"]
        norm_dist = torch.distributions.Normal(0, 1)
        return norm_dist.cdf(logits)

    def simulate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate binary choices from the fitted model.

        Args:
            X: Design matrix of shape (n_samples, n_features).

        Returns:
            Simulated binary choices of shape (n_samples,).
        """
        probas = self.predict_proba(X)
        return (torch.rand_like(probas) < probas).to(torch.int32)

    def counterfactual(self, X_new: torch.Tensor) -> dict:
        """
        Compute change in market share for a counterfactual scenario.

        Args:
            X_new: Counterfactual design matrix of shape (n_samples, n_features).

        Returns:
            Dictionary containing market share comparisons.

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if self._fitted_X is None:
            raise ValueError("Model must be fitted before running counterfactuals.")

        # Original market share
        p_old = self.predict_proba(self._fitted_X)
        market_share_old = torch.mean(p_old)

        # Counterfactual market share
        p_new = self.predict_proba(X_new)
        market_share_new = torch.mean(p_new)

        return {
            "market_share_original": market_share_old,
            "market_share_counterfactual": market_share_new,
            "change_in_market_share": market_share_new - market_share_old,
        }


class MultinomialLogit(ChoiceModel):
    """
    Multinomial Logit (MNL) model for discrete choice with multiple alternatives.

    This model generalizes binary logit to J > 2 alternatives using the
    softmax (multinomial logit) link function. One alternative's parameters
    are normalized to zero for identification.

    Examples:
        >>> model = MultinomialLogit()
        >>> X = torch.randn(100, 3)
        >>> y = torch.zeros(100, 4)  # 4 alternatives
        >>> y[range(100), torch.randint(0, 4, (100,))] = 1  # One-hot encode
        >>> model.fit(X, y)
        >>> probs = model.predict_proba(X)
    """

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        init_params: torch.Tensor = None,
        verbose: bool = False,
    ) -> "MultinomialLogit":
        """
        Fit multinomial logit model with proper param initialization.

        Args:
            X: Design matrix of shape (n_samples, n_features).
            y: One-hot encoded choices of shape (n_samples, n_choices).
            init_params: Initial parameter matrix of shape (n_features, n_choices - 1).
            verbose: If True, prints convergence information.

        Returns:
            Fitted model.
        """
        n_features = X.shape[1]
        n_choices = y.shape[1]

        # Reshape init_params for multinomial case
        if init_params is not None:
            if init_params.ndim == 1:
                init_params = init_params.reshape(n_features, n_choices - 1)
        else:
            # Initialize as matrix, then flatten for optimizer
            init_params = (
                torch.randn(
                    n_features, n_choices - 1, device=self.device, dtype=X.dtype
                )
                * 0.01
            )

        # Flatten for optimizer
        init_params_flat = init_params.flatten()

        # Call parent fit with flattened params
        super().fit(X, y, init_params=init_params_flat, verbose=verbose)

        # Reshape stored params back to matrix form
        self.params["coef"] = self.params["coef"].reshape(n_features, n_choices - 1)

        return self

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood for multinomial logit model.

        Uses cross-entropy loss with one alternative's parameters fixed to
        zero for identification.

        Args:
            params: Flattened coefficient vector (reshaped to (n_features, n_choices - 1) internally).
            X: Design matrix of shape (n_samples, n_features).
            y: One-hot encoded choices of shape (n_samples, n_choices).

        Returns:
            Negative log-likelihood as a scalar tensor.
        """
        n_features = X.shape[1]
        n_choices = y.shape[1]
        # Reshape flattened params during optimization
        params = params.reshape(n_features, n_choices - 1)
        # We fix one choice's params to 0 for identification
        params_full = torch.cat(
            [params, torch.zeros(X.shape[1], 1, device=self.device, dtype=X.dtype)],
            dim=1,
        )
        logits = X @ params_full
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        return -torch.sum(y * log_probs)

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Expected information in flattened (feature, non-base class) order."""
        p, k = X.shape[1], y.shape[1] - 1
        coefficients = params.reshape(p, k)
        logits = X @ torch.cat([coefficients, coefficients.new_zeros(p, 1)], dim=1)
        probabilities = torch.softmax(logits, dim=1)[:, :k]
        categorical_cov = (
            torch.diag_embed(probabilities)
            - probabilities[:, :, None] * probabilities[:, None, :]
        )
        return torch.einsum("ni,nj,nab->iajb", X, X, categorical_cov).reshape(
            p * k, p * k
        )

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict choice probabilities for each alternative.

        Args:
            X: Design matrix of shape (n_samples, n_features).

        Returns:
            Predicted probabilities of shape (n_samples, n_choices).

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")
        X = X.to(self.device)
        params_full = torch.cat(
            [
                self.params["coef"],
                torch.zeros(X.shape[1], 1, device=self.device, dtype=X.dtype),
            ],
            dim=1,
        )
        logits = X @ params_full
        return torch.nn.functional.softmax(logits, dim=1)

    def simulate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate choices from the fitted multinomial logit model.

        Args:
            X: Design matrix of shape (n_samples, n_features).

        Returns:
            Simulated choice indices of shape (n_samples,).
        """
        probs = self.predict_proba(X)
        return torch.multinomial(probs, 1).squeeze(1)

    def counterfactual(self, X_new: torch.Tensor) -> dict:
        """
        Compute change in market shares across alternatives for counterfactual.

        Args:
            X_new: Counterfactual design matrix of shape (n_samples, n_features).

        Returns:
            Dictionary containing:
                - market_share_original: Market shares per alternative (n_choices,).
                - market_share_counterfactual: Counterfactual market shares.
                - change_in_market_share: Difference in market shares.

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if self._fitted_X is None:
            raise ValueError("Model must be fitted before running counterfactuals.")

        # Original market share
        p_old = self.predict_proba(self._fitted_X)
        market_share_old = torch.mean(p_old, dim=0)

        # Counterfactual market share
        p_new = self.predict_proba(X_new)
        market_share_new = torch.mean(p_new, dim=0)

        return {
            "market_share_original": market_share_old,
            "market_share_counterfactual": market_share_new,
            "change_in_market_share": market_share_new - market_share_old,
        }


class LowRankLogit(ChoiceModel):
    """
    Low-Rank Logit model for matrix completion style choice problems.

    This model handles varying choice sets (assortments) by learning a low-rank
    factorization Θ = AB^T of the user-item utility matrix, where users choose
    from presented assortments according to multinomial logit probabilities.

    Based on: Kallus & Udell (2016), "Revealed Preference at Scale: Learning
    Personalized Preferences from Assortment Choices"
    """

    def __init__(
        self,
        rank: int,
        n_users: int,
        n_items: int,
        lam: float = 0.0,
        optimizer=torch.optim.LBFGS,
        maxiter=20000,
        tol=1e-4,
        device=None,
    ):
        super().__init__(optimizer, maxiter, tol, device=device)
        self.rank = rank
        self.n_users = n_users
        self.n_items = n_items
        self.lam = lam  # Regularization parameter

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        assortments: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood for observations with varying choice sets.

        Args:
            params: Flattened parameters [A; B] where A is n_users x rank, B is n_items x rank.
            X: User indices (n_obs,).
            y: Chosen item indices (n_obs,).
            assortments: Binary mask (n_obs, n_items) indicating available items per observation.
                        If None, assumes all items are available (full choice set).

        Returns:
            Negative log-likelihood as a scalar tensor.
        """
        user_indices = X.long()
        item_indices = y.long()

        A = params[: self.n_users * self.rank].reshape(self.n_users, self.rank)
        B = params[self.n_users * self.rank :].reshape(self.n_items, self.rank)

        theta = A @ B.T

        # Enforce zero-sum constraint (Θe = 0) per user for identification
        theta = theta - theta.mean(dim=1, keepdim=True)

        # Select utilities for observed users
        utilities = theta[user_indices]  # (n_obs, n_items)

        # Mask unavailable items by setting their utilities to -inf
        if assortments is not None:
            utilities = torch.where(
                assortments.bool(),
                utilities,
                torch.tensor(float("-inf"), device=self.device),
            )

        # Compute log-softmax over available items
        log_probs = torch.nn.functional.log_softmax(utilities, dim=1)
        nll = -torch.sum(log_probs[torch.arange(len(item_indices)), item_indices])

        # Add regularization: λ/2(||A||_F^2 + ||B||_F^2) approximates λ||Θ||_*
        if self.lam > 0:
            reg = 0.5 * self.lam * (torch.sum(A**2) + torch.sum(B**2))
            return nll + reg

        return nll

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        assortments: torch.Tensor = None,
        init_params: torch.Tensor = None,
        verbose: bool = False,
    ) -> "LowRankLogit":
        """
        Fit the low-rank logit model.

        Args:
            X: User indices (n_obs,)
            y: Chosen item indices (n_obs,)
            assortments: Binary mask (n_obs, n_items) where 1 indicates item was available.
                        If None, assumes all items are always available.
            init_params: Initial parameter values (optional)
            verbose: Print convergence information

        Returns:
            self
        """
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        if assortments is not None:
            assortments = assortments.to(self.device)

        if init_params is None:
            A_init = torch.randn(self.n_users, self.rank, device=self.device) * 0.1
            B_init = torch.randn(self.n_items, self.rank, device=self.device) * 0.1
            init_params = torch.cat([A_init.flatten(), B_init.flatten()])
        else:
            init_params = init_params.to(self.device)

        current_params = init_params.clone().requires_grad_(True)

        if self.optimizer_class == torch.optim.LBFGS:
            optimizer = self.optimizer_class([current_params], max_iter=20)
        else:
            optimizer = self.optimizer_class([current_params])

        self.history["loss"] = []

        for i in range(self.maxiter):

            def closure():
                optimizer.zero_grad()
                loss = self._negative_log_likelihood(current_params, X, y, assortments)
                loss.backward()
                return loss

            if self.optimizer_class == torch.optim.LBFGS:
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
        self._compute_standard_errors()

        # Unpack and store final parameters with zero-sum constraint
        A = self.params["coef"][: self.n_users * self.rank].reshape(
            self.n_users, self.rank
        )
        B = self.params["coef"][self.n_users * self.rank :].reshape(
            self.n_items, self.rank
        )
        theta = A @ B.T
        theta = theta - theta.mean(dim=1, keepdim=True)

        self.params["A"] = A
        self.params["B"] = B
        self.params["theta"] = theta
        return self

    def predict_proba(
        self, X: torch.Tensor, assortments: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predict choice probabilities.

        Args:
            X: User indices (n_obs,)
            assortments: Binary mask (n_obs, n_items) indicating available items.
                        If None, assumes all items are available.

        Returns:
            Choice probabilities (n_obs, n_items)
        """
        if not self.params or "theta" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        X = X.to(self.device)
        if assortments is not None:
            assortments = assortments.to(self.device)

        user_indices = X.long()
        utilities = self.params["theta"][user_indices]

        # Mask unavailable items
        if assortments is not None:
            utilities = torch.where(
                assortments.bool(),
                utilities,
                torch.tensor(float("-inf"), device=self.device),
            )

        return torch.nn.functional.softmax(utilities, dim=1)

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Fisher information matrix for low-rank logit model.

        Note: This is a placeholder returning identity matrix. Full implementation
        requires computation of expected Hessian of the low-rank factorized utility.

        Args:
            params: Flattened [A; B] parameters.
            X: User indices (unused but kept for interface consistency).
            y: Chosen item indices (unused but kept for interface consistency).

        Returns:
            Placeholder identity matrix of shape (n_params, n_params).
        """
        # This is a placeholder and should be implemented properly.
        return torch.eye(params.shape[0], device=self.device)

    def simulate(
        self, X: torch.Tensor, assortments: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Simulate choices from the fitted model.

        Args:
            X: User indices (n_obs,)
            assortments: Binary mask (n_obs, n_items) indicating available items.
                        If None, assumes all items are available.

        Returns:
            Simulated item choices (n_obs,)
        """
        probs = self.predict_proba(X, assortments)
        return torch.multinomial(probs, 1).squeeze(1)

    def counterfactual(
        self,
        user_indices: torch.Tensor,
        baseline_assortments: torch.Tensor,
        counterfactual_assortments: torch.Tensor,
        item_revenues: torch.Tensor = None,
    ) -> dict:
        """
        Perform counterfactual analysis comparing two assortment scenarios.

        This method evaluates the impact of changing the choice set (assortment)
        on market shares, choice probabilities, and optionally expected revenues.
        The low-rank structure allows heterogeneous user preferences, avoiding
        the IIA problem of standard multinomial logit.

        Args:
            user_indices: User indices to evaluate (n_obs,)
            baseline_assortments: Binary mask (n_obs, n_items) for baseline scenario
            counterfactual_assortments: Binary mask (n_obs, n_items) for counterfactual
            item_revenues: Optional revenue per item (n_items,). If provided, computes
                          expected revenues under each scenario.

        Returns:
            dict containing:
                - baseline_probs: Choice probabilities under baseline (n_obs, n_items)
                - counterfactual_probs: Choice probabilities under counterfactual
                - baseline_market_share: Average choice probability per item (n_items,)
                - counterfactual_market_share: Average choice probability per item
                - market_share_change: Change in market share (n_items,)
                - baseline_expected_revenue: Expected revenue under baseline (if revenues provided)
                - counterfactual_expected_revenue: Expected revenue under counterfactual
                - revenue_change: Change in expected revenue
                - baseline_choices: Most likely choice per user under baseline (n_obs,)
                - counterfactual_choices: Most likely choice per user under counterfactual

        Example:
            >>> # Evaluate impact of adding a new product (item 5) to assortments
            >>> baseline = torch.ones(n_obs, n_items)
            >>> baseline[:, 5] = 0  # Product 5 not available in baseline
            >>> counterfactual = torch.ones(n_obs, n_items)  # All products available
            >>> results = model.counterfactual(users, baseline, counterfactual)
            >>> print(f"Adding product 5 changes market share by {results['market_share_change'][5]:.2%}")
        """
        if not self.params or "theta" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        # Move inputs to device
        if item_revenues is not None:
            item_revenues = item_revenues.to(self.device)

        # Compute choice probabilities under both scenarios
        baseline_probs = self.predict_proba(user_indices, baseline_assortments)
        counterfactual_probs = self.predict_proba(
            user_indices, counterfactual_assortments
        )

        # Compute market shares (average probability of choosing each item)
        baseline_market_share = baseline_probs.mean(dim=0)
        counterfactual_market_share = counterfactual_probs.mean(dim=0)
        market_share_change = counterfactual_market_share - baseline_market_share

        # Most likely choice per user (argmax)
        baseline_choices = torch.argmax(baseline_probs, dim=1)
        counterfactual_choices = torch.argmax(counterfactual_probs, dim=1)

        results = {
            "baseline_probs": baseline_probs,
            "counterfactual_probs": counterfactual_probs,
            "baseline_market_share": baseline_market_share,
            "counterfactual_market_share": counterfactual_market_share,
            "market_share_change": market_share_change,
            "baseline_choices": baseline_choices,
            "counterfactual_choices": counterfactual_choices,
        }

        # Compute expected revenues if provided
        if item_revenues is not None:
            # Expected revenue = sum over items of (choice probability * revenue)
            baseline_revenue = torch.sum(baseline_probs * item_revenues)
            counterfactual_revenue = torch.sum(counterfactual_probs * item_revenues)

            results["baseline_expected_revenue"] = baseline_revenue
            results["counterfactual_expected_revenue"] = counterfactual_revenue
            results["revenue_change"] = counterfactual_revenue - baseline_revenue
            results["revenue_change_pct"] = (
                counterfactual_revenue - baseline_revenue
            ) / baseline_revenue

        return results
