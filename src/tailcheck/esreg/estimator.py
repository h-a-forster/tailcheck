"""M-estimator for the joint linear quantile / ES regression with iterated local search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tailcheck.esreg.gfuns import G1, G2


@dataclass(frozen=True)
class ESRegFit:
    """Fitted joint regression. Coefficients are on the original ``y`` scale."""

    coef_q: NDArray[np.float64]
    coef_e: NDArray[np.float64]
    loss: float
    alpha: float
    g1: G1
    g2: G2
    y: NDArray[np.float64]
    xq: NDArray[np.float64]
    xe: NDArray[np.float64]
    n_local_searches: int

    @property
    def coef(self) -> NDArray[np.float64]:
        return np.concatenate([self.coef_q, self.coef_e])


def esreg(
    y: ArrayLike,
    xq: ArrayLike | None,
    xe: ArrayLike | None,
    alpha: float,
    g1: G1 = G1.ZERO,
    g2: G2 = G2.NEG_INV,
    early_stopping: int = 10,
    seed: int | None = None,
) -> ESRegFit:
    """Fit ``Q_alpha(y|xq) = xq'bq`` and ``ES_alpha(y|xe) = xe'be`` jointly.

    ``xq``/``xe`` are covariates without intercept (an intercept is always added);
    ``None`` means intercept only.
    """
    raise NotImplementedError("M4")
