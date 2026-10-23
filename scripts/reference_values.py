"""Recompute the closed-form and simulated reference values quoted in docs/DESIGN.md.

Independent of the tailcheck package on purpose (only numpy/scipy), so the values can serve
as oracles for it. Run: ``uv run python scripts/reference_values.py`` (about 1 minute).
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * float(np.log(y))


def lr_pof(x: int, t: int, a: float) -> float:
    return -2 * (xlogy(t - x, 1 - a) + xlogy(x, a) - xlogy(t - x, 1 - x / t) - xlogy(x, x / t))


def lr_ind(h: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
    a, b = h[:-1], h[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / (len(h) - 1)
    l0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    l1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    return (n00, n01, n10, n11), -2 * (l0 - l1)


def std_t_var_es(nu: float, a: float) -> tuple[float, float]:
    ta = stats.t.ppf(a, nu)
    s = np.sqrt((nu - 2) / nu)
    return ta * s, -s * (nu + ta**2) / (nu - 1) * stats.t.pdf(ta, nu) / a


def hall_sheather(n: int, a: float, tau: float = 0.05) -> float:
    z = stats.norm.ppf(a)
    core = (1.5 * stats.norm.pdf(z) ** 2) / (2 * z**2 + 1)
    return n ** (-1 / 3) * stats.norm.ppf(1 - tau / 2) ** (2 / 3) * core ** (1 / 3)


def main() -> None:
    print("Kupiec 5% non-rejection regions")
    for a in (0.01, 0.025, 0.05):
        for t in (250, 510, 1000):
            acc = [x for x in range(t + 1) if stats.chi2.sf(lr_pof(x, t, a), 1) > 0.05]
            print(f"  alpha={a} T={t}: {min(acc)}..{max(acc)}")

    c, lr = lr_ind(np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0]))
    print(f"Christoffersen 0110011100: counts={c} LR_ind={lr:.10f} p={stats.chi2.sf(lr, 1):.10f}")
    h = np.zeros(250, dtype=int)
    h[[10, 11, 100, 180, 181, 182]] = 1
    c, li = lr_ind(h)
    lp = lr_pof(int(h.sum()), 250, 0.01)
    print(
        f"Christoffersen T=250: counts={c} LR_pof={lp:.10f} (p {stats.chi2.sf(lp, 1):.10f}) "
        f"LR_ind={li:.10f} (p {stats.chi2.sf(li, 1):.6g}) "
        f"LR_cc={lp + li:.10f} (p {stats.chi2.sf(lp + li, 2):.6g})"
    )

    b = stats.binom(250, 0.01)
    print("Basel cdf x=0..10:", [round(100 * b.cdf(k), 2) for k in range(11)])
    print("Basel cdf exact x=4,5,9,10:", [f"{b.cdf(k):.6f}" for k in (4, 5, 9, 10)])

    for nu in (3, 4, 5, 10):
        for a in (0.01, 0.025):
            q, e = std_t_var_es(nu, a)
            print(f"std t nu={nu} alpha={a}: q={q:.10f} es={e:.10f}")
    for a in (0.01, 0.025):
        z = stats.norm.ppf(a)
        print(f"normal alpha={a}: q={z:.10f} es={-stats.norm.pdf(z) / a:.10f}")

    for n, a in ((500, 0.01), (1000, 0.025), (2000, 0.025), (250, 0.01)):
        h_bw = hall_sheather(n, a)
        print(f"Hall-Sheather n={n} alpha={a}: h={h_bw:.8f} alpha-h={a - h_bw:.6f}")

    for a in (0.01, 0.025):
        e = -stats.norm.pdf(stats.norm.ppf(a)) / a
        at = optimize.brentq(lambda x, e=e: stats.norm.ppf(x) - e, 1e-12, a)
        print(f"alpha_tilde({a}) = {at:.10f}")

    rng = np.random.default_rng(1)
    a, t, m = 0.025, 250, 200_000
    for nu in (None, 3, 5, 10):
        if nu is None:
            x = rng.standard_normal((m, t))
            q = stats.norm.ppf(a)
            e = -stats.norm.pdf(q) / a
        else:
            x = rng.standard_t(nu, (m, t)) * np.sqrt((nu - 2) / nu)
            q, e = std_t_var_es(nu, a)
        z2 = 1 - (x * (x <= q)).sum(axis=1) / (t * a * e)
        print(
            f"AS Z2 null nu={nu}: mean={z2.mean():.4f} q05={np.quantile(z2, 0.05):.3f} "
            f"q0001={np.quantile(z2, 0.0001):.3f}"
        )


if __name__ == "__main__":
    main()
