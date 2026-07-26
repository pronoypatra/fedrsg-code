"""
Phase-1 grounding: measure the COMPUTATION-COST oracle C^comp(a).

Logs (cumulative_optimizer_steps, test_accuracy) at SUB-EPOCH resolution.  This is
essential: on easy datasets (MNIST/FEMNIST) accuracy jumps from ~random to ~0.98
WITHIN THE FIRST EPOCH, so per-epoch logging yields only ~2 useful points (floor
and ceiling) and NOTHING in the 0.3-0.9 range the mechanism actually optimizes
over.  We instead evaluate every `--eval-every` steps, so the curve is resolved
across the whole accuracy range.  Locally we invert to "compute (steps) needed to
first reach accuracy a" = C^comp(a) -- convex, increasing, steep near the ceiling.

Output CSV schema:  step,test_acc     (step = cumulative optimizer steps)

Run (on GPU server):
  python comp_cost.py --dataset mnist   --epochs 30 --eval-every 20  --out results/comp_mnist.csv
  python comp_cost.py --dataset cifar10 --epochs 80 --eval-every 100 --out results/comp_cifar10.csv
  python comp_cost.py --dataset femnist --epochs 40 --eval-every 50  --out results/comp_femnist.csv
  python comp_cost.py --dataset adult   --epochs 40 --eval-every 20  --out results/comp_adult.csv
"""
from __future__ import annotations
import argparse, os, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import get_dataset, build_model, test_acc, device_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=5e-4,
                    help="L2 regularization; helps CIFAR avoid late overfitting")
    ap.add_argument("--eval-every", type=int, default=20,
                    help="evaluate test accuracy every N optimizer steps (SUB-EPOCH); "
                         "essential to resolve the 0.3-0.9 range on easy datasets")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers; 0 avoids multiprocessing hangs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = device_str()
    print(f"[{args.dataset}] device={device}, loading data...", flush=True)
    tr, te, meta = get_dataset(args.dataset, args.root)
    print(f"[{args.dataset}] data ready ({len(tr)} train), building model...", flush=True)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    tel = DataLoader(te, batch_size=512, shuffle=False, num_workers=args.workers)

    model = build_model(meta).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    # cosine LR decay to ~0 over training: makes accuracy climb to a stable plateau
    # instead of diverging late (the CIFAR peak-then-decline we saw). Monotone
    # C^comp(a) requires this.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["step", "test_acc"])
        step = 0
        w.writerow([0, test_acc(model, tel, device)]); fh.flush()
        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y in trl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
                step += 1
                if step % args.eval_every == 0:            # SUB-EPOCH sampling
                    a = test_acc(model, tel, device)
                    w.writerow([step, a]); fh.flush()
                    model.train()
            sched.step()
            a = test_acc(model, tel, device)
            w.writerow([step, a]); fh.flush()
            print(f"[{args.dataset}] epoch {ep:3d}  step {step}  test_acc {a:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.4f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
