"""Joint quantile and expected shortfall regression (Dimitriadis and Bayer, 2019).

A Python implementation written from the published estimator; the R package ``esreg``
(GPL-3) is used only as a black-box numerical oracle (``scripts/r_fixtures/``), never as a
source to translate. Specification: ``docs/DESIGN.md`` section 4.3. Milestone M4.
"""

from __future__ import annotations

from tailcheck.esreg.covariance import vcov_asymptotic
from tailcheck.esreg.estimator import ESRegFit, esreg
from tailcheck.esreg.gfuns import G1, G2
from tailcheck.esreg.loss import esr_loss

__all__ = ["G1", "G2", "ESRegFit", "esr_loss", "esreg", "vcov_asymptotic"]
