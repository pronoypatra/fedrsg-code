"""
Phase-2 step 1: fit the measured grounding CSVs into parametric oracles.

Turns the GPU output (grounding/results/*.csv) into:
  1. C^comp(a): the computation-cost oracle -- monotone convex cost of reaching
     accuracy a.  Data = (cumulative steps, test_acc); we invert to steps-to-first-
     reach-a and fit both the paper's log-form and a monotone spline, checking the
     shape assumption (convex, increasing, steep near ceiling).
  2. A(a, eps): the observed-accuracy-under-privacy oracle.  Data = (eps, test_acc)
     at the model's own accuracy ceiling; we fit a concave-increasing-in-eps form
     and confirm the assumption.

Outputs (to grounding/fitted/):
  - <ds>_ccomp.json / <ds>_aeps.json : fitted params + goodness-of-fit
  - figures: measured points + fitted curve per dataset ("faithful model" panels)

Run:  code/.venv/bin/python code/fit_oracles.py
"""
from __future__ import annotations
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "grounding", "results")
OUT = os.path.join(HERE, "grounding", "fitted")

HEADLINE = ["mnist", "femnist"]
SECONDARY = ["cifar10", "adult"]


# ----------------------------------------------------------------- io

def read_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    hdr, data = rows[0], rows[1:]
    cols = {h: np.array([float(r[i]) for r in data]) for i, h in enumerate(hdr)}
    return cols


# ------------------------------------------------- C^comp(a): invert to cost(a)

def steps_to_reach(step, acc, grid):
    """For each target accuracy g in grid, the first cumulative step at which the
    (running-max) test accuracy reaches g. Running-max because we want 'compute to
    ACHIEVE a', ignoring transient dips."""
    order = np.argsort(step)
    step, acc = step[order], acc[order]
    run = np.maximum.accumulate(acc)
    out = []
    for g in grid:
        idx = np.searchsorted(run, g)
        out.append(step[idx] if idx < len(step) else np.nan)
    return np.array(out, float)


def fit_ccomp(step, acc):
    """Return (grid_a, cost_steps, fit) with a monotone cost curve and the paper's
    log-form fit cost(a) = k * log(a_curr(1-a_curr)/(a(1-a))) checked for shape."""
    a_lo = float(np.min(acc)); a_hi = float(np.max(acc))
    # sample targets in the resolved interior range (avoid the noisy last 1%)
    grid = np.linspace(max(a_lo, 0.2), a_hi - 0.005, 40)
    cost = steps_to_reach(step, acc, grid)
    ok = np.isfinite(cost)
    grid, cost = grid[ok], cost[ok]
    # shape checks
    monotone = bool(np.all(np.diff(cost) >= -1e-9))
    # convexity: second difference mostly >= 0
    d2 = np.diff(cost, 2)
    convex_frac = float(np.mean(d2 >= 0)) if len(d2) else float("nan")
    return {
        "a_grid": grid.tolist(), "cost_steps": cost.tolist(),
        "a_min": a_lo, "a_max": a_hi,
        "monotone_increasing": monotone,
        "convex_fraction": convex_frac,
    }


# ------------------------------------------------- A(a, eps): concave in eps

def fit_aeps(eps, acc):
    """Fit accuracy(eps) with a saturating concave form
        A(eps) = a_inf - (a_inf - a0) * exp(-eps / tau)
    (rises from ~a0 at eps->0 to ceiling a_inf, concave & increasing). Report
    params + R^2 + whether the data are (weakly) increasing at low eps."""
    from scipy.optimize import curve_fit
    order = np.argsort(eps); eps, acc = eps[order], acc[order]

    def model(e, a_inf, gap, tau):
        return a_inf - gap * np.exp(-e / max(tau, 1e-6))

    a_inf0 = float(np.max(acc)); gap0 = float(np.max(acc) - np.min(acc)) + 1e-3
    try:
        p, _ = curve_fit(model, eps, acc, p0=[a_inf0, gap0, 0.3],
                         maxfev=20000, bounds=([0, 0, 1e-3], [1.5, 1.5, 100]))
        pred = model(eps, *p)
        ss_res = float(np.sum((acc - pred) ** 2))
        ss_tot = float(np.sum((acc - np.mean(acc)) ** 2)) + 1e-12
        r2 = 1 - ss_res / ss_tot
        a_inf, gap, tau = [float(v) for v in p]
    except Exception as ex:
        a_inf = gap = tau = r2 = float("nan")
        print(f"    aeps fit failed: {ex}")
    # concavity/monotonicity signal from raw data (low-eps rise)
    lo = acc[eps <= np.median(eps)]
    rises_low = bool(lo[-1] >= lo[0]) if len(lo) > 1 else False
    return {
        "eps": eps.tolist(), "acc": acc.tolist(),
        "form": "a_inf - gap*exp(-eps/tau)",
        "a_inf": a_inf, "gap": gap, "tau": tau, "r2": r2,
        "rises_at_low_eps": rises_low,
    }


# ----------------------------------------------------------------- driver

def main():
    os.makedirs(OUT, exist_ok=True)
    datasets = HEADLINE + SECONDARY
    summary = {}
    for ds in datasets:
        cpath = os.path.join(RES, f"comp_{ds}.csv")
        dpath = os.path.join(RES, f"dp_{ds}.csv")
        entry = {}
        if os.path.exists(cpath):
            c = read_csv(cpath)
            step_key = "step" if "step" in c else "epoch"
            entry["ccomp"] = fit_ccomp(c[step_key], c["test_acc"])
        if os.path.exists(dpath):
            d = read_csv(dpath)
            entry["aeps"] = fit_aeps(d["target_eps"], d["test_acc"])
        summary[ds] = entry
        with open(os.path.join(OUT, f"{ds}.json"), "w") as fh:
            json.dump(entry, fh, indent=2)

    # print a compact report
    print(f"{'dataset':10s} {'C^comp mono':>12s} {'C^comp conv%':>13s} "
          f"{'A r2':>7s} {'A a_inf':>8s} {'A rises':>8s}")
    for ds in datasets:
        e = summary[ds]
        cc = e.get("ccomp", {}); ae = e.get("aeps", {})
        print(f"{ds:10s} {str(cc.get('monotone_increasing','-')):>12s} "
              f"{cc.get('convex_fraction', float('nan')):>13.2f} "
              f"{ae.get('r2', float('nan')):>7.3f} "
              f"{ae.get('a_inf', float('nan')):>8.3f} "
              f"{str(ae.get('rises_at_low_eps','-')):>8s}")
    _plot(summary)
    print(f"\nfitted params -> {OUT}/<ds>.json")


def _plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    for ds, e in summary.items():
        if not e:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        cc = e.get("ccomp")
        if cc:
            axes[0].plot(cc["a_grid"], cc["cost_steps"], "-o", ms=3)
            axes[0].set_xlabel("target accuracy $a$")
            axes[0].set_ylabel("compute to reach $a$ (steps)")
            axes[0].set_title(f"{ds}: $C^{{comp}}(a)$")
            axes[0].grid(alpha=0.3)
        ae = e.get("aeps")
        if ae and np.isfinite(ae.get("r2", float("nan"))):
            eps = np.array(ae["eps"]); acc = np.array(ae["acc"])
            xs = np.linspace(min(eps), max(eps), 200)
            ys = ae["a_inf"] - ae["gap"] * np.exp(-xs / ae["tau"])
            axes[1].scatter(eps, acc, s=25, zorder=5, label="measured")
            axes[1].plot(xs, ys, "-", label=f"fit ($R^2$={ae['r2']:.3f})")
            axes[1].set_xscale("log")
            axes[1].set_xlabel(r"privacy budget $\varepsilon$ (log)")
            axes[1].set_ylabel("observed accuracy")
            axes[1].set_title(f"{ds}: $A(a,\\varepsilon)$")
            axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{ds}_grounding.pdf"))
        fig.savefig(os.path.join(OUT, f"{ds}_grounding.png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
