"""Multiple-testing adjustments: Holm (FWER) and Benjamini-Hochberg (FDR)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def holm(pvalues: ArrayLike) -> NDArray[np.float64]:
    """Holm (1979) step-down adjusted p-values; NaN entries stay NaN and are not counted."""
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv[order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def benjamini_hochberg(pvalues: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg (1995) q-values; NaN entries stay NaN and are not counted."""
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out
