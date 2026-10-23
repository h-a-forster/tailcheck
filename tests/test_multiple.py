"""Holm and BH adjustments against statsmodels."""

from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests

from tailcheck.multiple import benjamini_hochberg, holm


def test_against_statsmodels() -> None:
    p = np.random.default_rng(0).uniform(0, 0.2, 50)
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1])
    np.testing.assert_allclose(benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])


def test_nan_ignored() -> None:
    p = np.array([0.01, np.nan, 0.04])
    np.testing.assert_allclose(holm(p), [0.02, np.nan, 0.04])
    np.testing.assert_allclose(benjamini_hochberg(p), [0.02, np.nan, 0.04])
