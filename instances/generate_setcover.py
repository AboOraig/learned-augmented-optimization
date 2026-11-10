"""Random set covering instance generator (Balas & Ho style).

Writes CPLEX .lp files (plain text, no solver needed to generate them).

    min  sum_j c_j x_j
    s.t. sum_{j in S_i} x_j >= 1    for every row i
         x_j in {0, 1}

Usage:
    python instances/generate_setcover.py --split train --n 300 --rows 500 --cols 1000 --seed 0
    python instances/generate_setcover.py --split test  --n 50  --rows 500 --cols 1000 --seed 1000
    python instances/generate_setcover.py --split transfer --n 30 --rows 750 --cols 1000 --seed 2000
"""
import argparse
from pathlib import Path

import numpy as np


def generate(nrows, ncols, density, max_cost, rng):
    """Return (A, c): boolean coverage matrix (nrows x ncols) and integer costs."""
    nnz = max(int(nrows * ncols * density), 2 * nrows, ncols)
    A = np.zeros((nrows, ncols), dtype=bool)

    # every row is covered by at least 2 columns
    for i in range(nrows):
        A[i, rng.choice(ncols, size=2, replace=False)] = True
    # every column covers at least 1 row
    for j in np.flatnonzero(~A.any(axis=0)):
        A[rng.integers(nrows), j] = True
    # fill up to the target density
    while A.sum() < nnz:
        A[rng.integers(nrows), rng.integers(ncols)] = True

    c = rng.integers(1, max_cost + 1, size=ncols)
    return A, c


def write_lp(path, A, c):
    nrows, ncols = A.shape
    lines = ["Minimize", " obj: " + " + ".join(f"{int(c[j])} x{j}" for j in range(ncols)),
             "Subject To"]
    for i in range(nrows):
        cols = np.flatnonzero(A[i])
        lines.append(f" c{i}: " + " + ".join(f"x{j}" for j in cols) + " >= 1")
    lines.append("Binary")
    lines.append(" " + " ".join(f"x{j}" for j in range(ncols)))
    lines.append("End")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, help="train | test | transfer (output subfolder)")
    p.add_argument("--n", type=int, required=True, help="number of instances")
    p.add_argument("--rows", type=int, default=500)
    p.add_argument("--cols", type=int, default=1000)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--max-cost", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(__file__).parent / "setcover" / args.split
    out.mkdir(parents=True, exist_ok=True)
    for k in range(args.n):
        rng = np.random.default_rng(args.seed + k)
        A, c = generate(args.rows, args.cols, args.density, args.max_cost, rng)
        write_lp(out / f"sc_{args.rows}r_{args.cols}c_{args.seed + k}.lp", A, c)
    print(f"wrote {args.n} instances to {out}")


if __name__ == "__main__":
    main()
