"""
Phase-2: the headline experiments, run on the MEASURED oracles (grounding fits),
not the synthetic closed forms.  Produces the numbers/figures the paper reports.

For each headline dataset (MNIST, FEMNIST) we build C^comp(a) and A(a,eps) from
grounding/fitted/<ds>.json and, on that measured substrate, compute:

  (A) the DESIGN-LEVER FRONTIER: sweep payment convexity m in {1..5}; report the
      (client welfare, server utility, participation) trajectory -- the tunable
      lever that is the paper's thesis.
  (B) the RSG-vs-FORWARD-Stackelberg comparison: RSG payment schedule vs a single
      fixed (p, a_target) contract, on identical clients -- the "what commitment to
      a function buys" result (reach: welfare + participation at comparable U^s).
  (C) the iFedCrowd point: linear payment (m=1) as the subsumed baseline.

Rigor: N_SEEDS heterogeneous client draws per config; report mean +/- 95% CI.
All on method="grid" (bounded additive-error, the paper's stated guarantee).

Run (LOCAL, no GPU):  python phase2_measured.py
Outputs: prints tables; writes figures/phase2_<ds>_frontier.{pdf,png} and
         results_phase2/<ds>.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mechanism_layer.costs import ClientCosts
from mechanism_layer import payments as pay
from mechanism_layer import metrics
from mechanism_layer.mechanisms import fedrsg
from mechanism_layer.measured import MeasuredComp, MeasuredAccuracy
from phase0_gap import forward_stackelberg   # reuse the correct forward baseline

HEADLINE = ["mnist", "femnist"]
N_SEEDS = 5
A_CURR = 0.5


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(np.mean(x)) if len(x) else float("nan"), 0.0)
    return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))


def make_clients(ds, seed, n, comp_fn, a_curr):
    """Heterogeneous clients: per-client cost weights; measured C^comp via comp_fn."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.5, 3.0, n); d = rng.uniform(0.5, 3.0, n)
    return [ClientCosts(gamma=float(gi), delta=float(di), rho=0.0, a_curr=a_curr,
                        comp="measured", comp_fn=comp_fn) for gi, di in zip(g, d)]


def run_dataset(ds, n=10):
    mc = MeasuredComp(ds, a_curr=A_CURR)
    A = MeasuredAccuracy(ds)
    comp_fn = mc.cost
    # server weight: measured comp costs are normalized to [0,1]; beta scales the
    # accuracy benefit into the same units. Use a value that makes participation
    # worthwhile across seeds (calibrated once, reported).
    cfg = fedrsg.FedRSGConfig(beta=5.0, a_curr=A_CURR, r_min=0.0, r_max=5.0,
                              r_grid=200, eps_max=16.0)

    # (A) frontier over m, averaged across seeds
    frontier = {m: {"Us": [], "W": [], "part": [], "acc": []} for m in (1, 2, 3, 4, 5)}
    fwd = {"Us": [], "W": [], "part": [], "acc": []}
    for seed in range(N_SEEDS):
        clients = make_clients(ds, seed, n, comp_fn, A_CURR)
        for m in (1, 2, 3, 4, 5):
            _, out = fedrsg.solve(pay.convex_payment(m, A_CURR), clients, cfg, A,
                                  method="grid")
            s = metrics.summary(out)
            frontier[m]["Us"].append(s["server_utility"])
            frontier[m]["W"].append(s["welfare_avg"])
            frontier[m]["part"].append(s["participation"])
            frontier[m]["acc"].append(s["mean_obs_acc"])
        f = forward_stackelberg(clients, cfg, A)
        fwd["Us"].append(f["server_utility"]); fwd["W"].append(f["welfare_avg"])
        fwd["part"].append(f["participation"]); fwd["acc"].append(f["mean_obs_acc"])

    def agg(d):
        return {k: ci95(v) for k, v in d.items()}
    result = {"dataset": ds, "n": n, "seeds": N_SEEDS,
              "frontier": {m: agg(frontier[m]) for m in frontier},
              "forward": agg(fwd)}
    return result, frontier, fwd


def report(result):
    ds = result["dataset"]
    print(f"\n################  {ds.upper()}  (measured oracle, n={result['n']}, "
          f"{result['seeds']} seeds, mean +/- 95%CI)  ################")
    # calibration guard: warn if the frontier is degenerate (all-abstain or
    # all-saturated across m), which means beta/r_max need retuning.
    parts = [result["frontier"][m]["part"][0] for m in (1, 2, 3, 4, 5)]
    if max(parts) < 0.05:
        print("  [!] participation ~0 across all m -> beta too LOW (raise beta / r_max).")
    elif min(parts) > 0.95 and (max(result["frontier"][m]["acc"][0] for m in (1,2,3,4,5))
                                - min(result["frontier"][m]["acc"][0] for m in (1,2,3,4,5))) < 0.02:
        print("  [!] participation ~1 and accuracy flat across m -> beta too HIGH "
              "(lower beta) so the shape lever separates.")
    print(f"{'config':16s}{'U^s':>16s}{'welfare':>16s}{'particip.':>14s}{'acc':>14s}")
    for m in (1, 2, 3, 4, 5):
        f = result["frontier"][m]
        tag = f"RSG m={m}" + ("  (=iFedCrowd)" if m == 1 else "")
        _row(tag, f)
    _row("forward-Stackel.", result["forward"])


def _row(tag, f):
    def fmt(pair): return f"{pair[0]:.3f}+/-{pair[1]:.3f}"
    print(f"{tag:16s}{fmt(f['Us']):>16s}{fmt(f['W']):>16s}"
          f"{fmt(f['part']):>14s}{fmt(f['acc']):>14s}")


def plot(result, frontier, fwd, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ms = (1, 2, 3, 4, 5)
    wx = [np.mean(frontier[m]["W"]) for m in ms]
    uy = [np.mean(frontier[m]["Us"]) for m in ms]
    ax.plot(wx, uy, "-o", label="FedRSG (payment shape $m$=1..5)")
    for m, x, y in zip(ms, wx, uy):
        ax.annotate(f"m={m}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.scatter([np.mean(frontier[1]["W"])], [np.mean(frontier[1]["Us"])],
               marker="s", s=120, facecolors="none", edgecolors="C1",
               label="iFedCrowd (m=1)", zorder=6)
    ax.scatter([np.mean(fwd["W"])], [np.mean(fwd["Us"])], marker="D", s=90,
               color="C3", label="forward-Stackelberg", zorder=6)
    ax.set_xlabel("client welfare (mean net utility)")
    ax.set_ylabel("server utility $U^s$")
    ax.set_title(f"{result['dataset']}: measured-oracle frontier")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    base = os.path.splitext(path)[0]
    fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=140)
    plt.close(fig)


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase2")
    os.makedirs(outdir, exist_ok=True)
    for ds in HEADLINE:
        result, frontier, fwd = run_dataset(ds)
        report(result)
        with open(os.path.join(outdir, f"{ds}.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        plot(result, frontier, fwd,
             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "figures", f"phase2_{ds}_frontier.png"))
    print(f"\nwrote results_phase2/*.json and figures/phase2_*_frontier.{{pdf,png}}")


if __name__ == "__main__":
    main()
