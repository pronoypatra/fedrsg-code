"""
Regression + benchmark for the Phase-1.5 solver speedup.

Validates that the adaptive (DIRECT-style) server search returns the SAME
equilibrium as the dense grid, at a fraction of the evaluations.  If this passes,
Phase-2 runs can use method="adaptive" safely.

Run:  code/.venv/bin/python code/test_solver_speedup.py
"""
from __future__ import annotations
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mechanism_layer.costs import ClientCosts
from mechanism_layer import payments as pay
from mechanism_layer import metrics
from mechanism_layer.mechanisms import fedrsg


def make_clients(seed, n, a_curr, hetero=True):
    rng = np.random.default_rng(seed)
    if hetero:
        g = rng.uniform(0.5, 3.0, n); d = rng.uniform(0.5, 3.0, n)
    else:
        g = np.full(n, 1.5); d = np.full(n, 1.5)
    return [ClientCosts(gamma=float(x), delta=float(y), rho=0.0, a_curr=a_curr,
                        comp="log", priv="linear") for x, y in zip(g, d)]


def main():
    a_curr = 0.5
    cfg = fedrsg.FedRSGConfig(beta=50.0, a_curr=a_curr, r_min=0.0, r_max=50.0, r_grid=300)
    n_bad = 0
    print(f"{'config':28s}{'r*_grid':>9}{'r*_adap':>9}{'U^s_grid':>10}{'U^s_adap':>10}{'|dU|':>8}")
    for seed in (0, 1, 2):
        for m in (1, 2, 3, 4, 5):
            for hetero in (True, False):
                clients = make_clients(seed, 10, a_curr, hetero)
                P = pay.convex_payment(m, a_curr)

                t0 = time.time()
                rg, og = fedrsg.solve(P, clients, cfg, method="grid")
                tg = time.time() - t0
                t0 = time.time()
                ra, oa = fedrsg.solve(P, clients, cfg, method="adaptive")
                ta = time.time() - t0

                ug = metrics.server_utility(og, a_curr)
                ua = metrics.server_utility(oa, a_curr)
                dU = abs(ug - ua)
                # accept if server utility matches within a small tolerance
                # (r* itself may differ slightly on flat regions; U^s is what matters)
                ok = dU <= 0.5 + 0.01 * abs(ug)
                if not ok:
                    n_bad += 1
                tag = f"s{seed} m{m} {'het' if hetero else 'hom'}"
                flag = "" if ok else "  <-- MISMATCH"
                print(f"{tag:28s}{rg:>9.2f}{ra:>9.2f}{ug:>10.2f}{ua:>10.2f}{dU:>8.3f}{flag}")
    # timing on one representative config
    clients = make_clients(0, 10, a_curr, True); P = pay.convex_payment(3, a_curr)
    t0 = time.time(); [fedrsg.solve(P, clients, cfg, method="grid") for _ in range(3)]
    tg = (time.time() - t0) / 3
    t0 = time.time(); [fedrsg.solve(P, clients, cfg, method="adaptive") for _ in range(3)]
    ta = (time.time() - t0) / 3
    print(f"\ntiming (hetero n=10, m=3): grid={tg*1000:.0f}ms  adaptive={ta*1000:.0f}ms  "
          f"speedup={tg/ta:.1f}x")
    print(f"\n{'PASS' if n_bad == 0 else f'FAIL ({n_bad} mismatches)'}: "
          f"adaptive matches grid U^s across 30 configs")


if __name__ == "__main__":
    main()
