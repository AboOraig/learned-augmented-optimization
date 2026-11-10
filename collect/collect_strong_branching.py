"""Collect strong-branching expert data inside SCIP's branch-and-bound.

A custom branching rule (highest priority) is called at every node. At a random sample of
nodes (always the root, then each node with probability --sample-prob) it:
  1. extracts features for each LP branching candidate (a random subset of at most --max-cands),
  2. runs strong branching on those candidates to get the expert score of each one,
  3. records one row per candidate, with the expert's choice flagged (is_best),
  4. branches on the expert's choice (unless --no-follow), so deeper states come from a good tree.
At all other nodes the rule declines and SCIP's default reliability pseudocost branching decides.

Output: one <instance>.csv.gz per instance in --out (long format, one row per candidate,
grouped by sample_id). Instances already present in --out are skipped, so the script can be
interrupted and restarted.

Usage:
    python collect/collect_strong_branching.py --instances instances/setcover/train \
        --out data/samples/train --workers 4
    # quick check on one instance first:
    python collect/collect_strong_branching.py --instances instances/setcover/train \
        --out data/samples/check --limit 1 --time-limit 30
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from pyscipopt import Branchrule, Model, SCIP_RESULT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solver"))
from solver.common import apply_base_settings  # noqa: E402
from solver.features import FAILED, candidate_columns, column_degrees, node_features, safe  # noqa: E402

EPS = 1e-6


class StrongBranchingCollector(Branchrule):
    def __init__(self, sample_prob, max_samples, max_cands, itlim, seed, follow_expert):
        self.sample_prob = sample_prob
        self.max_samples = max_samples
        self.max_cands = max_cands
        self.itlim = itlim
        self.follow_expert = follow_expert
        self.rng = random.Random(seed)
        self.root_lpobj = None
        self.col_deg = None  # variable name -> number of LP rows it appears in (computed once)
        self.n_samples = 0
        self.sb_time = 0.0
        self.rows = []

    # ------------------------------------------------------------------ branching callback
    def branchexeclp(self, allowaddcons):
        m = self.model
        cands, cand_vals, _, ncands, _, _ = m.getLPBranchCands()
        if self.root_lpobj is None:
            self.root_lpobj = m.getLPObjVal()
        if self.col_deg is None:
            self.col_deg = column_degrees(m)

        is_root = m.getCurrentNode().getDepth() == 0
        take = is_root or self.rng.random() < self.sample_prob
        if ncands < 2 or self.n_samples >= self.max_samples or not take:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        idx = list(range(ncands))
        if ncands > self.max_cands:
            idx = sorted(self.rng.sample(idx, self.max_cands))

        node_feat = node_features(m, ncands, self.root_lpobj)
        node_feat["n_cands_sampled"] = len(idx)
        cols = candidate_columns(m, [cands[i] for i in idx], [cand_vals[i] for i in idx],
                                 self.col_deg)

        # ---- strong branching on the chosen candidates
        lpobj = m.getLPObjVal()
        cutoff = safe("cutoff", lambda: m.getCutoffbound(), float("inf"))
        t0 = time.perf_counter()
        results = []  # (position in idx, gain_down, gain_up, down_infeasible, up_infeasible)
        m.startStrongbranch()
        try:
            for pos, i in enumerate(idx):
                res = m.getVarStrongbranch(cands[i], self.itlim)
                down, up, lperror = res[0], res[1], res[-1]
                if lperror:
                    continue
                down_inf = (len(res) == 9 and bool(res[4])) or down >= cutoff
                up_inf = (len(res) == 9 and bool(res[5])) or up >= cutoff
                results.append((pos, max(down - lpobj, 0.0), max(up - lpobj, 0.0),
                                down_inf, up_inf))
        finally:
            m.endStrongbranch()
        self.sb_time += time.perf_counter() - t0

        if not results:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # an infeasible child is the best possible outcome: give it a gain larger than any finite one
        finite = [g for _, gd, gu, di, ui in results for g, bad in ((gd, di), (gu, ui)) if not bad]
        big = 10.0 * max(finite + [1.0])
        scored = []
        for pos, gd, gu, di, ui in results:
            gd, gu = (big if di else gd), (big if ui else gu)
            scored.append((pos, gd, gu, max(gd, EPS) * max(gu, EPS)))
        best_pos = max(scored, key=lambda s: s[3])[0]

        sample_id = self.n_samples
        for pos, gd, gu, score in scored:
            row = {"sample_id": sample_id}
            row.update(node_feat)
            row.update({name: float(values[pos]) for name, values in cols.items()})
            row.update({"var_name": cands[idx[pos]].name, "sb_gain_down": gd, "sb_gain_up": gu,
                        "sb_score": score, "is_best": int(pos == best_pos)})
            self.rows.append(row)
        self.n_samples += 1

        if self.follow_expert:
            m.branchVar(cands[idx[best_pos]])
            return {"result": SCIP_RESULT.BRANCHED}
        return {"result": SCIP_RESULT.DIDNOTRUN}


def finalize(rows):
    """Long-format DataFrame; adds the expert's rank within each node (the within-node ranks of
    the features were already computed by features.py)."""
    df = pd.DataFrame(rows)
    df["sb_rank"] = df.groupby("sample_id")["sb_score"].rank(ascending=False, method="first")
    return df


def collect_instance(path, out_dir, opts):
    out = Path(out_dir) / (Path(path).stem + ".csv.gz")
    if out.exists():
        return {"instance": Path(path).name, "status": "skipped"}
    FAILED.clear()
    model = Model()
    model.readProblem(str(path))
    apply_base_settings(model, opts["time_limit"], opts["seed"])
    rule = StrongBranchingCollector(opts["sample_prob"], opts["max_samples"], opts["max_cands"],
                                    opts["itlim"], opts["seed"], opts["follow_expert"])
    model.includeBranchrule(rule, "sbcollector", "strong branching data collection",
                            priority=536870911, maxdepth=-1, maxbounddist=1.0)
    model.optimize()
    n_rows = 0
    if rule.rows:
        df = finalize(rule.rows)
        df.insert(0, "instance", Path(path).name)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        n_rows = len(df)
    return {"instance": Path(path).name, "status": model.getStatus(), "nodes": model.getNNodes(),
            "time": round(model.getSolvingTime(), 1), "samples": rule.n_samples, "rows": n_rows,
            "sb_time": round(rule.sb_time, 1), "unavailable": dict(FAILED)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--time-limit", type=float, default=60, help="per instance, seconds")
    p.add_argument("--sample-prob", type=float, default=0.1, help="probability of sampling a node")
    p.add_argument("--max-samples", type=int, default=30, help="max sampled nodes per instance")
    p.add_argument("--max-cands", type=int, default=60, help="max candidates strong-branched per node")
    p.add_argument("--itlim", type=int, default=100, help="LP iteration limit per strong-branching probe")
    p.add_argument("--no-follow", action="store_true", help="do not branch on the expert's choice")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    opts = dict(time_limit=args.time_limit, sample_prob=args.sample_prob, max_samples=args.max_samples,
                max_cands=args.max_cands, itlim=args.itlim, follow_expert=not args.no_follow,
                seed=args.seed)
    files = sorted(Path(args.instances).glob("*.lp"))[: args.limit]
    Path(args.out).mkdir(parents=True, exist_ok=True)

    summaries, unavailable = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect_instance, f, args.out, opts) for f in files]
        for k, fut in enumerate(as_completed(futures), 1):
            s = fut.result()
            summaries.append(s)
            for name, count in s.get("unavailable", {}).items():
                unavailable[name] = unavailable.get(name, 0) + count
            if s["status"] == "skipped":
                print(f"[{k}/{len(files)}] {s['instance']}: skipped (already collected)")
            else:
                print(f"[{k}/{len(files)}] {s['instance']}: {s['status']}, {s['nodes']} nodes, "
                      f"{s['time']}s, {s['samples']} samples ({s['rows']} rows), "
                      f"strong branching {s['sb_time']}s")

    done = [s for s in summaries if s["status"] != "skipped"]
    if done:
        pd.DataFrame(done).drop(columns=["unavailable"]).to_csv(Path(args.out) / "_summary.csv", index=False)
        print(f"\nTotal: {sum(s['samples'] for s in done)} samples, {sum(s['rows'] for s in done)} rows "
              f"from {len(done)} instances")
    if unavailable:
        print("\nWARNING: these features could not be computed with your PySCIPOpt version and are "
              f"filled with 0 (tell me, I will adapt them): {unavailable}")


if __name__ == "__main__":
    main()
