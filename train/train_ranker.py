"""Train a model that imitates strong branching, and evaluate it offline on validation nodes.

Each sampled node is a group of candidate variables; the model gives every candidate a score
and the branching decision is the highest score. Trained with LightGBM's LambdaRank objective
(or plain binary classification with --objective binary).

Offline metrics, per validation node (group):
  top1   the model's pick is exactly the strong-branching pick
  top3   the model's pick is among the strong-branching top 3
  ratio  strong-branching score of the model's pick / best strong-branching score (1.0 = as good
         as the expert; more forgiving than top1 when several candidates are near-ties)
Reference rules computed on the same nodes: most fractional, SCIP's pseudocost score, random.

NOTE: the validation set is used for early stopping, so these numbers are slightly optimistic.
The real test is solving unseen instances with the model inside SCIP (steps 4 and 5).

Usage:
    python train/train_ranker.py --train data/samples/train --val data/samples/val --out train/model
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# Never used as features: identifiers, labels, and quantities that do not transfer between
# instances or between training and solving:
#  - lpobj: absolute scale, instance dependent (lpobj_rel_root is kept)
#  - n_cands_sampled: capped at --max-cands when collecting, but the solver sees ALL candidates
NON_FEATURES = {"instance", "sample_id", "group", "var_name", "sb_gain_down", "sb_gain_up",
                "sb_score", "is_best", "sb_rank", "lpobj", "n_cands_sampled"}


def load(folder):
    files = sorted(Path(folder).glob("*.csv.gz"))
    if not files:
        sys.exit(f"No .csv.gz files in {folder}. Run the collection step first.")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.sort_values(["instance", "sample_id"], kind="stable").reset_index(drop=True)
    df["group"] = df.groupby(["instance", "sample_id"], sort=False).ngroup()
    return df


def group_sizes(df):
    return df.groupby("group", sort=False).size().to_numpy()


def relevance(sb_rank):
    """Graded relevance from the expert's ranking: 4 for its pick, down to 0 beyond rank 20."""
    r = np.asarray(sb_rank)
    return np.select([r == 1, r <= 3, r <= 8, r <= 20], [4, 3, 2, 1], default=0)


def evaluate(df, scores):
    """Offline decision quality of a scoring (higher score = preferred candidate)."""
    d = df[["group", "sb_score", "sb_rank"]].copy()
    d["s"] = np.asarray(scores)
    picked = d.loc[d.groupby("group", sort=False)["s"].idxmax().to_numpy()]
    best = d.groupby("group", sort=False)["sb_score"].max().to_numpy()
    return {"top1": float((picked["sb_rank"] == 1).mean()),
            "top3": float((picked["sb_rank"] <= 3).mean()),
            "ratio": float((picked["sb_score"].to_numpy() / best).mean())}


def evaluate_random(df):
    """Expected metrics of picking a candidate uniformly at random (closed form)."""
    n = df.groupby("group", sort=False).size().to_numpy()
    d = df[["group", "sb_score"]].copy()
    d["r"] = d["sb_score"] / d.groupby("group")["sb_score"].transform("max")
    return {"top1": float((1.0 / n).mean()), "top3": float((np.minimum(3, n) / n).mean()),
            "ratio": float(d.groupby("group", sort=False)["r"].mean().mean())}


def fit(tr, va, features, args):
    stop = lgb.early_stopping(30, first_metric_only=True, verbose=False)
    common = dict(n_estimators=args.n_estimators, learning_rate=args.lr, num_leaves=args.leaves,
                  min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=args.seed, verbose=-1)
    if args.objective == "lambdarank":
        model = lgb.LGBMRanker(objective="lambdarank", **common)
        model.fit(tr[features], relevance(tr["sb_rank"]), group=group_sizes(tr),
                  eval_set=[(va[features], relevance(va["sb_rank"]))], eval_group=[group_sizes(va)],
                  eval_at=[1, 3], callbacks=[stop])
        return model, lambda X: model.predict(X)
    model = lgb.LGBMClassifier(objective="binary", **common)
    model.fit(tr[features], tr["is_best"], eval_set=[(va[features], va["is_best"])],
              eval_metric="binary_logloss", callbacks=[stop])
    return model, lambda X: model.predict_proba(X)[:, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/samples/train")
    p.add_argument("--val", default="data/samples/val")
    p.add_argument("--out", default="train/model")
    p.add_argument("--objective", choices=["lambdarank", "binary"], default="lambdarank")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tr, va = load(args.train), load(args.val)
    features = [c for c in tr.columns if c not in NON_FEATURES]
    constant = [c for c in features if tr[c].nunique() <= 1]
    features = [c for c in features if c not in constant]
    print(f"train: {tr['group'].nunique()} nodes, {len(tr)} rows, {tr['instance'].nunique()} instances")
    print(f"val:   {va['group'].nunique()} nodes, {len(va)} rows, {va['instance'].nunique()} instances")
    print(f"{len(features)} features; dropped constant: {constant}")

    model, predict = fit(tr, va, features, args)
    results = {
        "learned (" + args.objective + ")": evaluate(va, predict(va[features])),
        "pseudocost score": evaluate(va, va["pc_score"]),
        "most fractional": evaluate(va, va["frac_dist"]),
        "random": evaluate_random(va),
    }
    print("\nValidation, offline decision quality:")
    print(pd.DataFrame(results).T[["top1", "top3", "ratio"]].round(3).to_string())

    gain = pd.Series(model.booster_.feature_importance("gain"), index=features)
    print("\nTop features by gain (share of total):")
    print((gain / gain.sum()).sort_values(ascending=False).head(10).round(3).to_string())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / "model.txt"))
    (out / "meta.json").write_text(json.dumps(
        {"features": features, "objective": args.objective,
         "val_metrics": results["learned (" + args.objective + ")"]}, indent=2))
    print(f"\nsaved model and feature list to {out}/")


if __name__ == "__main__":
    main()
