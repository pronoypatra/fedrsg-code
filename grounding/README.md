# GPU runs (Phase 1 grounding + Phase 3 practice)

**Runs on your GPU server, NOT in the local `.venv`.** These scripts measure the
oracle functions the paper assumes and demonstrate the equilibrium-to-training
recipe, from *real* training. Together they answer reviewers' "assumptions
unjustified," "DP never actually implemented," and "no path from theory to
practice."

## Datasets (all four supported out of the box)

| name      | source                                   | shape            | model |
|-----------|------------------------------------------|------------------|-------|
| `mnist`   | torchvision MNIST                        | 1×28×28, 10 cls  | CNN   |
| `cifar10` | torchvision CIFAR-10                     | 3×32×32, 10 cls  | CNN   |
| `femnist` | torchvision EMNIST `balanced` (stand-in) | 1×28×28, 47 cls  | CNN   |
| `adult`   | UCI Adult via OpenML (non-vision)        | 108-d tabular    | MLP   |

All models are BatchNorm-free with concrete (non-Lazy) Linear layers, so Opacus
can wrap them for DP-SGD.

## Setup (on the GPU box)

```
python -m venv .venv-gpu && source .venv-gpu/bin/activate
pip install -r requirements-gpu.txt
```

## Phase 1 — grounding (do this first; feeds Phase 2)

**1. `C^comp(a)` — computation cost** (`comp_cost.py`) logs `(epoch, test_acc)`:

```
python comp_cost.py --dataset mnist   --epochs 30 --out results/comp_mnist.csv
python comp_cost.py --dataset cifar10 --epochs 60 --out results/comp_cifar10.csv
python comp_cost.py --dataset femnist --epochs 40 --out results/comp_femnist.csv
python comp_cost.py --dataset adult   --epochs 40 --out results/comp_adult.csv
```

**2. `A(a, ε)` — observed accuracy under real DP-SGD** (`dp_accuracy.py`) logs
`(target_eps, realized_eps, test_acc)`:

```
python dp_accuracy.py --dataset mnist   --epsilons 0.5 1 2 4 8 --out results/dp_mnist.csv
python dp_accuracy.py --dataset cifar10 --epsilons 0.5 1 2 4 8 --out results/dp_cifar10.csv
python dp_accuracy.py --dataset femnist --epsilons 0.5 1 2 4 8 --out results/dp_femnist.csv
python dp_accuracy.py --dataset adult   --epsilons 0.5 1 2 4 8 --out results/dp_adult.csv
```

## Phase 3 — practice recipe (after we compute equilibria; can also demo now)

`practice_recipe.py` trains a model to a target accuracy `a*` via the two-phase
recipe (coarse alignment → fine tuning), optionally under DP at `ε*`. Logs
`(target_acc, achieved_acc, phase1_epochs, phase2_epochs, realized_eps)`:

```
python practice_recipe.py --dataset mnist   --targets 0.90 0.95 0.97       --out results/practice_mnist.csv
python practice_recipe.py --dataset cifar10 --targets 0.60 0.70 0.80 --eps 4 --out results/practice_cifar_dp.csv
```

(The `--targets` will eventually be the `a*` values our solver returns; any
reasonable targets work for a standalone demo now.)

## Send back

Just the CSVs in `results/`. The fitting step (curve → parametric oracle) and all
figures are done locally — hand me the CSVs and I produce the fitted `A(a,ε)` and
`C^comp(a)` used in Phase 2, plus the "faithful model" and practice figures.

CSV schemas (all the local side needs):
- `comp_*.csv`     : `epoch,test_acc`
- `dp_*.csv`       : `target_eps,realized_eps,test_acc`
- `practice_*.csv` : `target_acc,achieved_acc,phase1_epochs,phase2_epochs,realized_eps`
