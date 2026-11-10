"""Benchmark classical branching rules in SCIP on a folder of .lp instances.

Usage:
    python solver/baseline.py --instances instances/setcover/test --rules relpscost mostinf fullstrong \
        --time-limit 300 --out analysis/baseline_test.csv

Rules are SCIP branching rule names:
    relpscost  reliability pseudocost branching (SCIP default, strong baseline)
    pscost     pseudocost branching
    mostinf    most infeasible (= most fractional) branching
    fullstrong full strong branching (the expert we will imitate later)

Solver settings live in solver/common.py (shared with the data collector and the learned rule).
"""
import argparse
import csv
from pathlib import Path

from pyscipopt import Model

from solver.common import apply_base_settings


def configure(model, rule, time_limit, seed):
    apply_base_settings(model, time_limit, seed)
    # make the chosen rule the one SCIP calls first
    model.setIntParam(f"branching/{rule}/priority", 536870911)


def run(path, rule, time_limit, seed):
    model = Model()
    model.readProblem(str(path))
    configure(model, rule, time_limit, seed)
    model.optimize()
    return {
        "instance": Path(path).name,
        "rule": rule,
        "status": model.getStatus(),
        "nodes": model.getNNodes(),
        "time": round(model.getSolvingTime(), 3),
        "gap": model.getGap(),
        "primal": model.getPrimalbound(),
        "dual": model.getDualbound(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True)
    p.add_argument("--rules", nargs="+", default=["relpscost", "mostinf"])
    p.add_argument("--time-limit", type=float, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="only the first N instances (quick test)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = sorted(Path(args.instances).glob("*.lp"))[: args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        for rule in args.rules:
            r = run(f, rule, args.time_limit, args.seed)
            rows.append(r)
            print(f"{r['instance']:32s} {rule:11s} {r['status']:10s} "
                  f"nodes={r['nodes']:7d} time={r['time']:8.2f}s gap={r['gap']:.4f}")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
