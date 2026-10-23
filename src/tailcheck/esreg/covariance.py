"""Asymptotic sandwich covariance of the joint estimator and its nuisance estimators."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tailcheck.esreg.estimator import ESRegFit

Sparsity = Literal["iid", "nid"]
SigmaEstimator = Literal["ind", "scl_N", "scl_sp"]
Bandwidth = Literal["hall-sheather", "bofinger", "chamberlain"]


def vcov_asymptotic(
    fit: ESRegFit,
    sparsity: Sparsity = "nid",
    sigma_est: SigmaEstimator = "scl_sp",
    misspec: bool = True,
    bandwidth: Bandwidth = "hall-sheather",
) -> NDArray[np.float64]:
    """``Lambda^{-1} Sigma Lambda^{-1} / n`` for the stacked coefficients ``(bq, be)``."""
    raise NotImplementedError("M4")


def bandwidth_sj(x: ArrayLike) -> float:
    """Sheather-Jones (1991) solve-the-equation bandwidth for a Gaussian KDE."""
    raise NotImplementedError("M4")
