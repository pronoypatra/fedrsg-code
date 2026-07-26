"""
FedRSG solver: the structure-exploiting equilibrium computation.

Client (inner): utility U_k(r, a, eps) = P(r, A(a,eps)) - C_k(a,eps) is concave
  (paper Prop.1).  We reduce to 1-D by solving eps for each a, then golden-section
  search over a.  O(log 1/xi).
Server (outer): Phi(r) = U^s(r, omega*(r)) is Lipschitz but generally non-concave,
  so we use bounded 1-D global search (scipy DIRECT if available, else a dense
  Lipschitz grid fallback) over r in [r_min, r_max].

The payment P is passed in as a callable P(r, hat_a), so the SAME solver runs the
full FedRSG, the linear (m=1) ablation, and the constant-payment ablation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import minimize_scalar

from ..costs import ClientCosts
from ..oracle import observed_accuracy
from ..metrics import Outcome


GOLDEN = (np.sqrt(5) - 1) / 2  # 0.618...


def _two_stage_argmin(f: Callable[[float], float], lo: float, hi: float,
                      coarse: int, refine: int, k_brackets: int = 3) -> float:
    """Minimize a continuous but NON-CONCAVE (multi-modal) f on [lo,hi].

    The server objective Phi(r) is continuous (proved: Lemma continuity-of-best-
    response + Weierstrass) but non-concave -- it can have several local optima, so
    a golden-section / unimodal search on the outer loop is unsafe.  Rather than a
    Lipschitz global optimizer, we mirror the dense grid's guarantee at lower cost:
      1. a COARSE uniform scan locates the competitive basins (cannot miss a basin
         wider than the coarse step);
      2. around each of the top-k coarse points we run a DENSE local scan, so the
         final resolution near the optimum matches a dense global grid.
    Deterministic and reproducible; converges to the dense-grid optimum as
    coarse*refine -> (dense grid size).  Returns argmin.
    """
    xs = np.linspace(lo, hi, coarse)
    fs = np.array([f(float(x)) for x in xs])
    step = (hi - lo) / max(coarse - 1, 1)
    order = np.argsort(fs)[:max(1, k_brackets)]
    best_x, best_f = float(xs[order[0]]), float(fs[order[0]])
    for i in order:
        a = max(lo, xs[i] - step)
        b = min(hi, xs[i] + step)
        for x in np.linspace(a, b, refine):
            fx = f(float(x))
            if fx < best_f:
                best_f, best_x = fx, float(x)
    return best_x


def _golden_section(f: Callable[[float], float], lo: float, hi: float,
                    tol: float = 1e-5, maxit: int = 200) -> float:
    """Maximize a unimodal (concave) f on [lo,hi]. Returns argmax."""
    a, b = lo, hi
    c = b - GOLDEN * (b - a)
    d = a + GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(maxit):
        if b - a < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = f(d)
    return (a + b) / 2


@dataclass
class FedRSGConfig:
    eps_max: float = 10.0
    a_curr: float = 0.5
    eta: float = 1e-3          # a in [a_curr, 1-eta]
    beta: float = 1.0          # server weight on observed accuracy
    alpha: float = 0.0
    r_min: float = 0.0
    r_max: float = 50.0
    r_grid: int = 300          # server line-search resolution
    tol: float = 1e-4


def client_best_response(P: Callable[[float, float], float],
                         costs: ClientCosts, r: float, cfg: FedRSGConfig,
                         A: Callable[[float, float], float] = observed_accuracy):
    """Return (a*, eps*, hat_a*, u*) maximizing U_k(r, a, eps)."""
    def util(a: float, eps: float) -> float:
        hat_a = A(a, eps)
        return P(r, hat_a) - costs.total(a, eps)

    # inner: best eps for a fixed a (1-D concave in eps on [0, eps_max])
    _cache: dict = {}

    def best_eps(a: float):
        key = round(a, 7)
        if key in _cache:
            return _cache[key]
        g = lambda e: util(a, e)
        e = _golden_section(g, 0.0, cfg.eps_max, cfg.tol)
        # compare against boundaries (optimum may sit at 0 or eps_max)
        cands = (0.0, cfg.eps_max, e)
        eb = max(cands, key=g)
        res = (eb, g(eb))
        _cache[key] = res
        return res

    # outer: 1-D concave in a on [a_curr, 1-eta]
    def g_of_a(a: float) -> float:
        return best_eps(a)[1]

    a_star = _golden_section(g_of_a, cfg.a_curr, 1 - cfg.eta, cfg.tol)
    eps_star, u_star = best_eps(a_star)

    # Individual-rationality floor: the trivial action (a=a_curr, eps=0) yields
    # hat_a=A(a_curr,0). With a centered payment P(.,a_curr)=0 and zero comp cost
    # at a_curr, its utility is 0 (minus a possible privacy cost at eps=0, =0).
    # A rational client never accepts a negative-utility optimum -- it abstains.
    # Also guards against the solver returning a slightly-suboptimal interior point.
    trivial_hat = A(cfg.a_curr, 0.0)
    trivial_u = P(r, trivial_hat) - costs.total(cfg.a_curr, 0.0)
    if trivial_u >= u_star:
        return cfg.a_curr, 0.0, trivial_hat, trivial_u
    return a_star, eps_star, A(a_star, eps_star), u_star


def solve(P: Callable[[float, float], float],
          clients: list[ClientCosts], cfg: FedRSGConfig,
          A: Callable[[float, float], float] = observed_accuracy,
          global_acc_fn: Callable[[np.ndarray], float] | None = None,
          method: str = "grid") -> tuple[float, Outcome]:
    """Compute the FedRSG equilibrium.

    Returns (r*, Outcome). global_acc_fn maps the vector of hat_a to a global
    accuracy (default: mean of observed accuracies).

    The client (inner) best response is solved exactly by a provably-concave
    1-D reduction + golden section -- that is the rigorous algorithmic content.
    The server (outer) objective Phi(r) is continuous (proved via Berge +
    Weierstrass) but NON-CONCAVE (multi-modal), so a unimodal outer search is
    unsafe; we search r by discretization instead -- exactly the "discretization-
    based additive-error approximation" the paper describes, with a bounded,
    reproducible guarantee.

    method:
      - "grid" (default): dense uniform scan of cfg.r_grid points; the additive
        error is controlled by the grid step.  Simple, deterministic, and the
        method the theory's guarantee is stated for.
      - "twostage": a coarse scan to locate competitive basins + dense local
        refine around the best few -- same near-optimum resolution as a dense grid
        at a fraction of the evaluations, and (unlike a Lipschitz global optimizer)
        robust to Phi's jumps.  Validate against "grid" before relying on it.
    """
    def responses(r: float):
        out = [client_best_response(P, c, r, cfg, A) for c in clients]
        a = np.array([o[0] for o in out])
        eps = np.array([o[1] for o in out])
        hat = np.array([o[2] for o in out])
        return a, eps, hat

    def neg_server_util(r: float) -> float:
        _, _, hat = responses(r)
        pay = np.array([P(r, h) for h in hat])
        # sum (not mean) of accuracy improvement over a_curr; matches metrics.server_utility
        return -(cfg.beta * np.sum(hat - cfg.a_curr) - np.sum(pay))

    # --- outer search over r ---
    if method == "grid":
        grid = np.linspace(cfg.r_min, cfg.r_max, cfg.r_grid)
        vals = [neg_server_util(float(r)) for r in grid]
        r_star = float(grid[int(np.argmin(vals))])
    elif method == "twostage":
        # coarse*refine ~ evaluations; pick so the local step matches the dense grid
        coarse = max(8, int(np.sqrt(cfg.r_grid)) * 2)
        refine = max(8, cfg.r_grid // coarse + 2)
        r_star = _two_stage_argmin(neg_server_util, cfg.r_min, cfg.r_max,
                                   coarse=coarse, refine=refine)
    else:
        raise ValueError(f"unknown method {method!r}")

    a, eps, hat = responses(r_star)
    pay = np.array([P(r_star, h) for h in hat])
    costs_val = np.array([c.total(ai, ei) for c, ai, ei in zip(clients, a, eps)])
    gacc = float(np.mean(hat)) if global_acc_fn is None else global_acc_fn(hat)
    out = Outcome(obs_acc=hat, payments=pay, costs=costs_val,
                  global_acc=gacc, alpha=cfg.alpha, beta=cfg.beta)
    return r_star, out
