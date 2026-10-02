"""
Generalized method of moments (GMM) and GEL estimators.

The module exports factory-backed GMM estimators and generalized empirical
likelihood implementations used for moment-based econometric inference.
"""

from .gmm import GMMEstimator
from .gel import GELEstimator, rho_exponential, rho_cue, rho_el
from .cue import ScalarIVCUE, PolynomialCUEResult

__all__ = [
    "GMMEstimator",
    "GELEstimator",
    "rho_exponential",
    "rho_cue",
    "rho_el",
    "ScalarIVCUE",
    "PolynomialCUEResult",
]
