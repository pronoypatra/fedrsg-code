"""
Phase-2 supporting study: solver efficiency.

The structure-exploiting solver's rigorous win is on the CLIENT best response: the
inner problem is concave, so a 1-D golden-section search finds (a*,eps*) to additive
accuracy xi in O(log 1/xi) utility evaluations, versus a naive 2-D grid over (a,eps)
which needs O(1/xi^2).  We measure both -- utility-evaluation count and wall-clock --
as xi shrinks, on the MEASURED FEMNIST oracle, and confirm the golden-section result
matches the grid optimum.

Run (LOCAL):  python phase2_solver_efficiency.py
Outputs: prints a table; writes results_phase2/solver_efficiency.json +
         figures/phase2_solver_efficiency.{pdf,png}
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mechanism_layer.costs import ClientCosts
from mechanism_layer import payments as pay
from mechanism_layer.measured import MeasuredComp, MeasuredAccuracy

DS = "femnist"
A_CURR = 0.5


class Counter:
    """Wraps a utility function to count evaluations."""
    def __init__(self, f):
        self.f = f; self.n = 0
    def __call__(self, *a):
        self.n += 1
        return self.f(*a)


def make_util(client, A, P, r):
    def u(a, eps):
        return P(r, A(a, eps)) - client.total(a, eps)
    return u


def golden_br(util, a_lo, a_hi, eps_max, xi):
    """1-D-reduced golden-section: for each a, inner golden-section over eps, then
    outer golden-section over a. Returns (a*, eps*, u*)."""
    g = (np.sqrt(5) - 1) / 2

    def gs(f, lo, hi):
        a, b = lo, hi
        c = b - g * (b - a); d = a + g * (b - a)
        fc, fd = f(c), f(d)
        while b - a > xi:
            if fc > fd:
                b, d, fd = d, c, fc; c = b - g * (b - a); fc = f(c)
            else:
                a, c, fc = c, d, fd; d = a + g * (b - a); fd = f(d)
        return (a + b) / 2

    def best_eps(a):
        e = gs(lambda ee: util(a, ee), 0.0, eps_max)
        cands = [(0.0, util(a, 0.0)), (eps_max, util(a, eps_max)), (e, util(a, e))]
        return max(cands, key=lambda t: t[1])

    a_star = gs(lambda a: best_eps(a)[1], a_lo, a_hi)
    e_star, u_star = best_eps(a_star)
    return a_star, e_star, u_star


def grid_br(util, a_lo, a_hi, eps_max, xi):
    """Naive 2-D grid over (a,eps) at resolution ~xi in each axis: O(1/xi^2)."""
    na = max(4, int(np.ceil((a_hi - a_lo) / xi)))
    ne = max(4, int(np.ceil(eps_max / xi)))
    A = np.linspace(a_lo, a_hi, na); E = np.linspace(0.0, eps_max, ne)
    best = (-np.inf, a_lo, 0.0)
    for a in A:
        for e in E:
            u = util(a, e)
            if u > best[0]:
                best = (u, a, e)
    return best[1], best[2], best[0]


def main():
    mc = MeasuredComp(DS, a_curr=A_CURR); A = MeasuredAccuracy(DS)
    client = ClientCosts(gamma=1.5, delta=1.5, rho=0.0, a_curr=A_CURR,
                         comp="measured", comp_fn=mc.cost)
    P = pay.convex_payment(3, A_CURR); r = 2.0
    eps_max = 16.0; a_lo, a_hi = A_CURR, 1 - 1e-3

    xis = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    rows = []
    print(f"### solver efficiency ({DS} measured oracle) ###")
    print(f"{'xi':>8}{'gold_evals':>12}{'grid_evals':>12}{'speedup':>10}"
          f"{'|du|':>10}{'gold_ms':>10}{'grid_ms':>10}")
    for xi in xis:
        gu = Counter(make_util(client, A, P, r))
        t0 = time.time(); ag, eg, ug = golden_br(gu, a_lo, a_hi, eps_max, xi)
        tg = (time.time() - t0) * 1e3
        ru = Counter(make_util(client, A, P, r))
        t0 = time.time(); ar, er, ur = grid_br(ru, a_lo, a_hi, eps_max, xi)
        tr = (time.time() - t0) * 1e3
        speed = ru.n / max(gu.n, 1)
        du = abs(ug - ur)
        rows.append({"xi": xi, "gold_evals": gu.n, "grid_evals": ru.n,
                     "speedup": speed, "util_gap": du, "gold_ms": tg, "grid_ms": tr})
        print(f"{xi:>8}{gu.n:>12}{ru.n:>12}{speed:>10.1f}{du:>10.4f}{tg:>10.1f}{tr:>10.1f}")

    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase2")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "solver_efficiency.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    _plot(rows)
    print("\nwrote results_phase2/solver_efficiency.json + figures/phase2_solver_efficiency.*")


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    xi = [r["xi"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(xi, [r["grid_evals"] for r in rows], "-s", label="naive 2-D grid $O(1/\\xi^2)$")
    ax.plot(xi, [r["gold_evals"] for r in rows], "-o", label="golden-section $O(\\log 1/\\xi)$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"additive accuracy $\xi$"); ax.set_ylabel("utility evaluations")
    ax.set_title(f"{DS}: client best-response cost")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(figdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "phase2_solver_efficiency.pdf"))
    fig.savefig(os.path.join(figdir, "phase2_solver_efficiency.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
