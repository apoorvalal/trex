"""Experimental density splats: a normalized GMM with split/refine training.

This is not gsplat's alpha-composited image model. Every component integrates
into its mixture weight. Fixed eigenvalue floors prevent likelihood collapse.
The class is a research prototype, deliberately outside Trex's public API.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn


class GaussianSplats(nn.Module):
    def __init__(self, *, variance_floor=0.0025, seed=0, device="cpu",
                 dtype=torch.float64):
        super().__init__()
        if not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("variance_floor must be finite and positive")
        self.variance_floor = float(variance_floor)
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.history = []

    def _rows(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        if x.ndim != 2 or len(x) == 0 or not torch.isfinite(x).all():
            raise ValueError("Expected nonempty, finite two-dimensional rows")
        if hasattr(self, "means") and x.shape[1] != self.means.shape[1]:
            raise ValueError("Wrong number of columns")
        return x

    def _set_components(self, weights, means, covariances):
        d = means.shape[1]
        # Subtract the fixed floor; project only numerical/splitting roundoff.
        eye = torch.eye(d, device=self.device, dtype=self.dtype)
        vals, vecs = torch.linalg.eigh(covariances - self.variance_floor * eye)
        base = (vecs * vals.clamp_min(1e-8).unsqueeze(-2)) @ vecs.transpose(-1, -2)
        chol = torch.linalg.cholesky(base)
        packed = torch.tril(chol, diagonal=-1) + torch.diag_embed(chol.diagonal(dim1=-2, dim2=-1).log())
        self.logits = nn.Parameter(weights.log().detach())
        self.means = nn.Parameter(means.detach().clone())
        self.raw_chol = nn.Parameter(packed.detach())

    def initialize(self, x):
        x = self._rows(x)
        if len(x) < 2:
            raise ValueError("Need at least two training rows")
        mean = x.mean(0, keepdim=True)
        centered = x - mean
        cov = centered.T @ centered / len(x)
        cov += self.variance_floor * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        self._set_components(torch.ones(1, device=x.device, dtype=x.dtype), mean, cov[None])
        self.history = []
        return self

    @property
    def n_components(self):
        return len(self.means)

    @property
    def weights(self):
        return self.logits.softmax(0)

    @property
    def covariances(self):
        diag = self.raw_chol.diagonal(dim1=-2, dim2=-1).exp()
        lower = torch.tril(self.raw_chol, diagonal=-1) + torch.diag_embed(diag)
        eye = torch.eye(self.means.shape[1], device=self.device, dtype=self.dtype)
        return lower @ lower.transpose(-1, -2) + self.variance_floor * eye

    def log_prob(self, x):
        if not hasattr(self, "means"):
            raise RuntimeError("Initialize or fit before evaluating")
        x = self._rows(x)
        components = torch.distributions.MultivariateNormal(self.means, covariance_matrix=self.covariances)
        return torch.logsumexp(components.log_prob(x[:, None, :]) + self.logits.log_softmax(0), dim=1)

    def refine(self, x, *, validation=None, steps=300, lr=0.025, batch_size=512):
        x = self._rows(x)
        if steps < 1 or batch_size < 1 or lr <= 0:
            raise ValueError("steps, batch_size and lr must be positive")
        valid = None if validation is None else self._rows(validation)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        generator = torch.Generator(device=self.device).manual_seed(self.seed + self.n_components)
        best = {k: v.detach().clone() for k, v in self.state_dict().items()}
        with torch.no_grad():
            best_loss = float(-self.log_prob(x if valid is None else valid).mean())
        history = []
        for step in range(steps):
            ids = torch.randint(len(x), (min(batch_size, len(x)),), generator=generator, device=x.device)
            loss = -self.log_prob(x[ids]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 25 == 24 or step == steps - 1:
                with torch.no_grad():
                    check_loss = float(-self.log_prob(x if valid is None else valid).mean())
                history.append({"step": step + 1, "selection_nll": check_loss})
                if check_loss < best_loss:
                    best_loss = check_loss
                    best = {k: v.detach().clone() for k, v in self.state_dict().items()}
        self.load_state_dict(best)
        self.history.append({"n_components": self.n_components, "history": history})
        return self

    @torch.no_grad()
    def split(self):
        """Double capacity by splitting each component along its longest axis.

        Symmetric means and reduced child covariance preserve the parent's
        mean and covariance before subsequent gradient refinement.
        """
        cov = self.covariances
        vals, vecs = torch.linalg.eigh(cov)
        available = (vals[:, -1] - self.variance_floor).clamp_min(0)
        delta = 0.5 * available.sqrt()[:, None] * vecs[:, :, -1]
        child_cov = cov - delta[:, :, None] * delta[:, None, :]
        self._set_components(self.weights.repeat_interleave(2) / 2,
                             torch.stack([self.means - delta, self.means + delta], 1).flatten(0, 1),
                             child_cov.repeat_interleave(2, 0))
        return self

    @torch.no_grad()
    def sample(self, n, *, seed=None):
        if n < 0:
            raise ValueError("n must be nonnegative")
        g = torch.Generator(device=self.device).manual_seed(self.seed if seed is None else seed)
        d = self.means.shape[1]
        if n == 0:
            return torch.empty((0, d), dtype=self.dtype)
        labels = torch.multinomial(self.weights, n, replacement=True, generator=g)
        noise = torch.randn(n, d, device=self.device, dtype=self.dtype, generator=g)
        chol = torch.linalg.cholesky(self.covariances)[labels]
        return (self.means[labels] + torch.bmm(chol, noise[:, :, None]).squeeze(-1)).cpu()

    def snapshot(self):
        return {key: value.detach().cpu().numpy() for key, value in
                {"weights": self.weights, "means": self.means, "covariances": self.covariances}.items()}
