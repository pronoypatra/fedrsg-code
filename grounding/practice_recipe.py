"""
Phase-3: translate an equilibrium target a* into real training -- APPROXIMATELY,
and report the residual honestly.

Framing (deliberate).  Hitting a specific accuracy exactly is genuinely hard:
accuracy rises very fast early (MNIST/CIFAR reach high accuracy within an epoch)
and then flattens near the ceiling, so a target in the steep region is easy to
overshoot and a target near the plateau is hard to stop below.  We therefore do
NOT claim exact control.  We show two things instead:
  1. a client can get CLOSE to a* with a simple recipe, and we report the achieved
     accuracy and the residual |achieved - a*| honestly (never hidden);
  2. the mechanism is ROBUST to that residual: because a* maximizes the client's
     utility, dU/da = 0 there, so a small miss Delta_a costs only O(Delta_a^2) in
     utility -- an approximate landing is near-optimal on the equilibrium.
The comp_cost.py epochs->accuracy curve is the primary "realizable in practice"
evidence; this script quantifies how close the recipe lands and the utility gap.

Recipe (best-effort landing, tuned for the steep regime):
  Phase 1 (coarse): high LR, evaluate HELD-OUT accuracy every `eval_every` batches
    (SUB-EPOCH, so we do not blow past a low target within one epoch); stop once
    held-out accuracy reaches ema_frac * target.
  Phase 2 (fine): drop LR; keep evaluating sub-epoch, checkpoint the closest-to-
    target model on HELD-OUT data (not the optimistic training signal), stop within
    +/- tol; restore that best checkpoint at the end.

Optionally applies DP-SGD at a target eps* (private regime).

Output CSV schema:  target_acc,achieved_acc,abs_err,phase1_steps,phase2_steps,realized_eps

Run (on GPU server):
  python practice_recipe.py --dataset mnist --targets 0.90 0.95 0.97 --out results/practice_mnist.csv
  python practice_recipe.py --dataset cifar10 --targets 0.60 0.70 0.80 --eps 4 --out results/practice_cifar_dp.csv
"""
from __future__ import annotations
import argparse, os, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import get_dataset, build_model, test_acc, device_str


import copy


def train_to_target(dataset, target, root, max_epochs, batch, lr1, lr2,
                    eval_every, ema_frac, tol, eps, delta, max_grad_norm):
    """Best-effort landing at `target`, deciding on HELD-OUT accuracy, sub-epoch.

    Returns (achieved_acc, phase1_steps, phase2_steps, realized_eps) where
    achieved is the held-out accuracy of the closest checkpoint to `target`.
    """
    device = device_str()
    tr, te, meta = get_dataset(dataset, root)
    tel = DataLoader(te, batch_size=512, shuffle=False, num_workers=2)
    model = build_model(meta).to(device)

    engine = None
    opt = torch.optim.SGD(model.parameters(), lr=lr1, momentum=0.9)
    trl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2)
    if eps is not None:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        model = ModuleValidator.fix(model)
        opt = torch.optim.SGD(model.parameters(), lr=lr1, momentum=0.9)
        engine = PrivacyEngine()
        model, opt, trl = engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=trl, target_epsilon=eps,
            target_delta=delta, epochs=max_epochs, max_grad_norm=max_grad_norm)

    # track the checkpoint whose HELD-OUT accuracy is closest to target
    best = {"err": float("inf"), "state": None, "acc": 0.0}
    def _consider():
        acc = test_acc(model, tel, device)
        if abs(acc - target) < best["err"]:
            best.update(err=abs(acc - target), acc=acc,
                        state=copy.deepcopy(model.state_dict()))
        return acc

    def run_phase(lr, reach, budget):
        """Train until held-out accuracy first reaches `reach` (or budget spent),
        checking every `eval_every` optimizer steps. Returns steps used."""
        for g in opt.param_groups:
            g["lr"] = lr
        steps = 0
        for _ in range(budget):
            model.train()
            for x, y in trl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
                steps += 1
                if steps % eval_every == 0:
                    acc = _consider()
                    if acc >= reach:
                        return steps
        _consider()
        return steps

    # Phase 1 (coarse): reach ema_frac * target quickly
    p1 = run_phase(lr1, ema_frac * target, max_epochs)
    # Phase 2 (fine): low LR, land within tol of target
    p2 = run_phase(lr2, target - tol, max_epochs)

    if best["state"] is not None:                 # restore closest-to-target model
        model.load_state_dict(best["state"])
    achieved = best["acc"] if best["state"] is not None else test_acc(model, tel, device)
    realized = engine.get_epsilon(delta) if engine is not None else float("inf")
    return achieved, p1, p2, realized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--targets", type=float, nargs="+", required=True,
                    help="target accuracies a* from the equilibrium solver")
    ap.add_argument("--eps", type=float, default=None, help="optional DP target eps*")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr1", type=float, default=0.1)
    ap.add_argument("--lr2", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=50,
                    help="evaluate held-out accuracy every N optimizer steps (sub-epoch)")
    ap.add_argument("--ema-frac", type=float, default=0.9,
                    help="phase-1 stops at ema_frac*target on held-out")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target_acc", "achieved_acc", "abs_err", "phase1_steps", "phase2_steps", "realized_eps"])
        for t in args.targets:
            ach, p1, p2, realized = train_to_target(
                args.dataset, t, args.root, args.max_epochs, args.batch,
                args.lr1, args.lr2, args.eval_every, args.ema_frac, args.tol,
                args.eps, args.delta, args.max_grad_norm)
            w.writerow([t, ach, abs(ach - t), p1, p2, realized]); fh.flush()
            print(f"[{args.dataset}] target={t:.3f}  achieved={ach:.3f}  "
                  f"|err|={abs(ach-t):.3f}  p1={p1} p2={p2} eps={realized:.2f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
