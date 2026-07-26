"""
Phase-3: translate an equilibrium target (a*, eps*) into real training knobs.

The equilibrium solver returns a target accuracy a* (and privacy level eps*).  A
client must actually TRAIN a model that lands at a*.  We demonstrate the two-phase
accuracy-targeting recipe from the paper:

  Phase 1 (coarse alignment): train at a high LR / large batch, tracking an EMA of
    per-batch accuracy, until the EMA reaches (ema_frac * target).
  Phase 2 (fine tuning): drop the LR, shrink the batch, continue until the EMA is
    within +/- tol of the target; checkpoint periodically and finally pick the
    checkpoint whose held-out accuracy is closest to the target.

Optionally applies DP-SGD at a target eps* (so the recipe works in the private
regime too).

Output CSV schema:  target_acc,achieved_acc,phase1_epochs,phase2_epochs,realized_eps

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


def _batch_acc(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def train_to_target(dataset, target, root, max_epochs, batch1, batch2,
                    lr1, lr2, ema_coef, ema_frac, tol, eps, delta, max_grad_norm):
    device = device_str()
    tr, te, meta = get_dataset(dataset, root)
    tel = DataLoader(te, batch_size=512, shuffle=False, num_workers=2)
    model = build_model(meta).to(device)

    # optional DP
    engine = None
    if eps is not None:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        model = ModuleValidator.fix(model)
        opt = torch.optim.SGD(model.parameters(), lr=lr1, momentum=0.9)
        trl = DataLoader(tr, batch_size=batch1, shuffle=True, num_workers=2)
        engine = PrivacyEngine()
        model, opt, trl = engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=trl, target_epsilon=eps,
            target_delta=delta, epochs=max_epochs, max_grad_norm=max_grad_norm)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr1, momentum=0.9)
        trl = DataLoader(tr, batch_size=batch1, shuffle=True, num_workers=2)

    ema = None
    def run_phase(loader, optimizer, lr, stop_pred, budget):
        nonlocal ema
        for g in optimizer.param_groups:
            g["lr"] = lr
        used = 0
        for ep in range(1, budget + 1):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                F.cross_entropy(logits, y).backward(); optimizer.step()
                a = _batch_acc(logits, y)
                ema = a if ema is None else (1 - ema_coef) * ema + ema_coef * a
            used = ep
            if stop_pred(ema):
                break
        return used

    # Phase 1: coarse -- reach ema_frac * target
    p1 = run_phase(trl, opt, lr1, lambda e: e >= ema_frac * target, max_epochs)

    # Phase 2: fine -- within tol of target (smaller batch when not under DP;
    # under DP we keep the privatized loader, only dropping the LR)
    if eps is None:
        trl2 = DataLoader(tr, batch_size=batch2, shuffle=True, num_workers=2)
    else:
        trl2 = trl
    best = {"acc": test_acc(model, tel, device), "state": None}
    def fine_stop(e):
        # checkpoint the closest-so-far model to the target on held-out data
        acc = test_acc(model, tel, device)
        if abs(acc - target) < abs(best["acc"] - target):
            best["acc"] = acc
        return abs(e - target) <= tol
    p2 = run_phase(trl2, opt, lr2, fine_stop, max_epochs)

    achieved = min([test_acc(model, tel, device), best["acc"]],
                   key=lambda a: abs(a - target))
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
    ap.add_argument("--batch1", type=int, default=256)
    ap.add_argument("--batch2", type=int, default=64)
    ap.add_argument("--lr1", type=float, default=0.1)
    ap.add_argument("--lr2", type=float, default=0.01)
    ap.add_argument("--ema-coef", type=float, default=0.1)
    ap.add_argument("--ema-frac", type=float, default=0.8)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target_acc", "achieved_acc", "phase1_epochs", "phase2_epochs", "realized_eps"])
        for t in args.targets:
            ach, p1, p2, realized = train_to_target(
                args.dataset, t, args.root, args.max_epochs, args.batch1, args.batch2,
                args.lr1, args.lr2, args.ema_coef, args.ema_frac, args.tol,
                args.eps, args.delta, args.max_grad_norm)
            w.writerow([t, ach, p1, p2, realized]); fh.flush()
            print(f"[{args.dataset}] target={t:.3f}  achieved={ach:.3f}  "
                  f"|err|={abs(ach-t):.3f}  p1={p1} p2={p2} eps={realized:.2f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
