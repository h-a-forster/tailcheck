"""Write examples/garch_t5.csv: 1000 days of GARCH(1,1)-t5 returns with the true 2.5% VaR/ES.

Run from the repository root: ``uv run python scripts/make_example.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tailcheck.montecarlo.dgp import simulate


def main() -> None:
    s = simulate("garch-t5", 1000, 0.025, np.random.default_rng(20261023))
    pd.DataFrame({"r": s.r, "q": s.q, "e": s.e, "sigma": s.sigma}).to_csv(
        "examples/garch_t5.csv", index=False, float_format="%.8f"
    )


if __name__ == "__main__":
    main()
