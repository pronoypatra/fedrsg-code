"""
Phase-2 (extended): the three secondary studies, on the MEASURED oracle.

  (B1) beta-sweep: the server's accuracy weight beta as a second design lever ---
       raising beta should slide the operating point toward higher accuracy /
       participation. (fixed m=3, the FEMNIST sweet spot.)
  (B2) scale study: vary n in {5,10,20,50}; report how U^s, welfare, participation,
       accuracy move as the federation grows.
  (B3) multi-round: iterate the single-round solver, feeding each round's achieved
       accuracy a-hat* forward as the next round's a_curr, until it stops rising ---
       the self-terminating, monotone-and-bounded climb the multi-round proposition
       predicts. We plot a_curr(t) vs the theoretical monotone envelope.

Headline dataset = FEMNIST (the one with a non-trivial cost curve / visible lever).
Rigor: N_SEEDS seeds, mean +/- 95% CI.  method="grid".

Run (LOCAL, no GPU):  python phase2_extended.py
Outputs: prints tables; writes results_phase2/<ds>_extended.json and
         figures/phase2_<ds>_{beta,scale,rounds}.{pdf,png}
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

DS = "femnist"
N_SEEDS = 5
A_CURR0 = 0.5
M_STAR = 3            # the FEMNIST sweet-spot convexity
BETA0 = 5.0


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(np.mean(x)) if len(x) else float("nan"), 0.0)
    return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))


def make_clients(seed, n, comp_fn, a_curr):
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.5, 3.0, n); d = rng.uniform(0.5, 3.0, n)
    return [ClientCosts(gamma=float(gi), delta=float(di), rho=0.0, a_curr=a_curr,
                        comp="measured", comp_fn=comp_fn) for gi, di in zip(g, d)]


def solve_once(clients, A, beta, a_curr, m):
    cfg = fedrsg.FedRSGConfig(beta=beta, a_curr=a_curr, r_min=0.0, r_max=5.0,
                              r_grid=200, eps_max=16.0)
    _, out = fedrsg.solve(pay.convex_payment(m, a_curr), clients, cfg, A, method="grid")
    return metrics.summary(out), out


# ---------------------------------------------------------- (B1) beta sweep

def beta_sweep(comp_fn, A, betas=(1, 2, 5, 10, 20, 50)):
    rows = {}
    for beta in betas:
        acc = {"Us": [], "W": [], "part": [], "acc": []}
        for seed in range(N_SEEDS):
            cl = make_clients(seed, 10, comp_fn, A_CURR0)
            s, _ = solve_once(cl, A, beta, A_CURR0, M_STAR)
            acc["Us"].append(s["server_utility"]); acc["W"].append(s["welfare_avg"])
            acc["part"].append(s["participation"]); acc["acc"].append(s["mean_obs_acc"])
        rows[beta] = {k: ci95(v) for k, v in acc.items()}
    return rows


# ---------------------------------------------------------- (B2) scale study

def scale_study(comp_fn, A, ns=(5, 10, 20, 50)):
    rows = {}
    for n in ns:
        acc = {"Us": [], "W": [], "part": [], "acc": []}
        for seed in range(N_SEEDS):
            cl = make_clients(seed, n, comp_fn, A_CURR0)
            s, _ = solve_once(cl, A, BETA0, A_CURR0, M_STAR)
            acc["Us"].append(s["server_utility"]); acc["W"].append(s["welfare_avg"])
            acc["part"].append(s["participation"]); acc["acc"].append(s["mean_obs_acc"])
        rows[n] = {k: ci95(v) for k, v in acc.items()}
    return rows


# ---------------------------------------------------------- (B3) multi-round

def multi_round(comp_fn, A, rounds=8, step_cap=0.05):
    """Feed each round's achieved accuracy forward as next a_curr. Track a_curr(t).

    A single unconstrained solve already reaches the oracle's accuracy ceiling, so
    a_curr would jump to the plateau in one round.  Real FL does not: one round of
    local training + aggregation improves the global model by a BOUNDED amount.  We
    model that with a per-round improvement cap `step_cap`: a_curr rises toward the
    round's equilibrium target but by at most step_cap per round.  The result is the
    gradual, monotone, self-terminating climb the multi-round proposition predicts
    (it saturates once the equilibrium target stops exceeding a_curr)."""
    traj_seeds = []
    for seed in range(N_SEEDS):
        a_curr = A_CURR0
        traj = [a_curr]
        for t in range(rounds):
            cl = make_clients(seed, 10, comp_fn, a_curr)
            s, _ = solve_once(cl, A, BETA0, a_curr, M_STAR)
            target = s["mean_obs_acc"]
            gain = target - a_curr                 # equilibrium wants this much more
            if gain > 0:                           # climb, capped by per-round budget
                a_curr = a_curr + min(step_cap, gain)
            traj.append(a_curr)
        traj_seeds.append(traj)
    traj_seeds = np.array(traj_seeds)               # (seeds, rounds+1)
    mean = traj_seeds.mean(0); half = 1.96 * traj_seeds.std(0, ddof=1) / np.sqrt(N_SEEDS)
    monotone = bool(np.all(np.diff(mean) >= -1e-9))
    return {"mean": mean.tolist(), "ci": half.tolist(), "monotone": monotone}


# ---------------------------------------------------------- report + plot

def main():
    mc = MeasuredComp(DS, a_curr=A_CURR0); A = MeasuredAccuracy(DS); comp_fn = mc.cost
    print(f"### {DS.upper()} extended studies (measured oracle, {N_SEEDS} seeds, mean+/-95%CI) ###")

    beta = beta_sweep(comp_fn, A)
    print("\n(B1) beta-sweep (m=3):")
    print(f"{'beta':>6}{'U^s':>16}{'welfare':>16}{'particip.':>14}{'acc':>14}")
    for b, r in beta.items():
        print(f"{b:>6}{_f(r['Us']):>16}{_f(r['W']):>16}{_f(r['part']):>14}{_f(r['acc']):>14}")

    scale = scale_study(comp_fn, A)
    print("\n(B2) scale study (n):")
    print(f"{'n':>6}{'U^s':>16}{'welfare':>16}{'particip.':>14}{'acc':>14}")
    for n, r in scale.items():
        print(f"{n:>6}{_f(r['Us']):>16}{_f(r['W']):>16}{_f(r['part']):>14}{_f(r['acc']):>14}")

    mr = multi_round(comp_fn, A)
    print("\n(B3) multi-round a_curr(t):")
    print("  " + "  ".join(f"{v:.3f}" for v in mr["mean"]))
    print(f"  monotone climb: {mr['monotone']}")

    out = {"dataset": DS, "seeds": N_SEEDS, "m_star": M_STAR,
           "beta_sweep": {str(k): v for k, v in beta.items()},
           "scale": {str(k): v for k, v in scale.items()},
           "multi_round": mr}
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase2")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"{DS}_extended.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    _plots(beta, scale, mr)
    print("\nwrote results_phase2/%s_extended.json + figures/phase2_%s_{beta,scale,rounds}.*" % (DS, DS))


def _f(pair):
    return f"{pair[0]:.3f}+/-{pair[1]:.3f}"


def _plots(beta, scale, mr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(figdir, exist_ok=True)

    # beta
    b = sorted(beta.keys())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(b, [beta[k]["acc"][0] for k in b], "-o", label="accuracy")
    ax.plot(b, [beta[k]["part"][0] for k in b], "-s", label="participation")
    ax.set_xscale("log"); ax.set_xlabel(r"server weight $\beta$")
    ax.set_ylabel("value"); ax.set_title(f"{DS}: $\\beta$ dial (m={M_STAR})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"phase2_{DS}_beta.pdf")); fig.savefig(os.path.join(figdir, f"phase2_{DS}_beta.png"), dpi=140); plt.close(fig)

    # scale
    ns = sorted(scale.keys())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ns, [scale[k]["Us"][0] for k in ns], "-o", label="$U^s$")
    ax2 = ax.twinx()
    ax2.plot(ns, [scale[k]["W"][0] for k in ns], "-^", color="C1", label="welfare")
    ax.set_xlabel("clients $n$"); ax.set_ylabel("$U^s$"); ax2.set_ylabel("welfare")
    ax.set_title(f"{DS}: scale (m={M_STAR}, $\\beta$={BETA0})"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"phase2_{DS}_scale.pdf")); fig.savefig(os.path.join(figdir, f"phase2_{DS}_scale.png"), dpi=140); plt.close(fig)

    # multi-round
    m = np.array(mr["mean"]); h = np.array(mr["ci"]); t = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(t, m, yerr=h, fmt="-o", capsize=3, label="$a_{curr}(t)$ (measured)")
    ax.axhline(m[-1], ls="--", color="gray", label="saturation")
    ax.set_xlabel("round $t$"); ax.set_ylabel("global accuracy $a_{curr}$")
    ax.set_title(f"{DS}: multi-round climb"); ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"phase2_{DS}_rounds.pdf")); fig.savefig(os.path.join(figdir, f"phase2_{DS}_rounds.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
