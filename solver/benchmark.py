"""Benchmark branching rules on a folder of .lp instances (replaces baseline.py).

Rules: any SCIP branching rule name, or `learned` (needs --model):
    relpscost  reliability pseudocost branching (SCIP default, the strong baseline)
    pscost     pseudocost branching
    mostinf    most infeasible (= most fractional) branching
    fullstrong full strong branching (too slow for a full benchmark; use a few instances)
    learned    our trained model, plugged in as a branching rule

Usage:
    python solver/benchmark.py --instances instances/setcover/test --limit 3 \
        --rules relpscost mostinf learned --model train/model --time-limit 60 --out analysis/smoke.csv

Every rule runs under the same SCIP settings (solver/common.py). Recorded per run: status, nodes,
wall-clock time (perf_counter around optimize), SCIP's own time, optimality gap, bounds, and for
the learned rule the number of decisions and the time spent inside the rule (overhead).

--workers N runs several solves in parallel. Parallel runs compete for CPU and memory, so times
get noisier; for the final numbers use few workers (or 1) and a machine doing nothing else.
"""
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pyscipopt import Model

from common import apply_base_settings

PRIORITY = 536870911


def run(path, rule, seed, time_limit, model_dir):
    model = Model()
    model.readProblem(str(path))
    apply_base_settings(model, time_limit, seed)
    learned = None
    if rule == "learned":
        from learned_branching import LearnedBranchrule
        learned = LearnedBranchrule(model_dir)
        model.includeBranchrule(learned, "learned", "learned branching policy",
                                priority=PRIORITY, maxdepth=-1, maxbounddist=1.0)
    else:
        model.setIntParam(f"branching/{rule}/priority", PRIORITY)
    t0 = time.perf_counter()
    model.optimize()
    wall = time.perf_counter() - t0
    return {
        "instance": Path(path).name, "rule": rule, "seed": seed,
        "status": model.getStatus(), "nodes": model.getNNodes(),
        "time_wall": round(wall, 3), "time_scip": round(model.getSolvingTime(), 3),
        "gap": model.getGap(), "primal": model.getPrimalbound(), "dual": model.getDualbound(),
        "decisions": learned.n_decisions if learned else "",
        "decision_time": round(learned.decision_time, 3) if learned else "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True)
    p.add_argument("--rules", nargs="+", default=["relpscost", "mostinf", "learned"])
    p.add_argument("--model", default="train/model", help="folder with model.txt and meta.json")
    p.add_argument("--time-limit", type=float, default=300)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--limit", type=int, default=None, help="only the first N instances (quick test)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = sorted(Path(args.instances).glob("*.lp"))[: args.limit]
    jobs = [(f, r, s) for f in files for s in args.seeds for r in args.rules]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, f, r, s, args.time_limit, args.model) for f, r, s in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            extra = f" decision_time={r['decision_time']}s" if r["rule"] == "learned" else ""
            print(f"[{k}/{len(jobs)}] {r['instance']:28s} {r['rule']:10s} seed={r['seed']} "
                  f"{r['status']:10s} nodes={r['nodes']:7d} time={r['time_wall']:8.2f}s "
                  f"gap={r['gap']:.4f}{extra}")
    rows.sort(key=lambda r: (r["instance"], r["seed"], r["rule"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
