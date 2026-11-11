"""Summarise a benchmark CSV (from solver/benchmark.py): tables + one figure.

Usage:
    python analysis/analyze.py --results analysis/test.csv --out-dir analysis/report
    python analysis/analyze.py --results analysis/transfer.csv --out-dir analysis/report

Outputs (named after the CSV): <name>_summary.csv / .md (table view) and <name>_profile.png.

Conventions (standard in the learning-to-branch literature, state them in the write-up):
  * a run is "solved" if SCIP proved optimality within the time limit; a timeout counts with its
    time (= the limit) in the time means;
  * shifted geometric mean (SGM): exp(mean(log(x + shift))) - shift, shift 1 s for time and 100 for
    nodes, so that very easy instances do not dominate;
  * a "problem" is an (instance, seed) pair; only problems where every rule has a run are used;
  * times below --min-time are raised to it before comparing rules (sub-second timings are noise);
  * "fastest" = solved and within 5% of the best solved time on that problem.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One fixed colour per rule (colour follows the entity, never its rank). These are slots 1-5 of a
# validated colour-blind-safe categorical palette; aqua/yellow are low-contrast on a light surface,
# which is why every line is also labelled and the table (summary.md) carries the same numbers.
COLORS = {"learned": "#2a78d6", "relpscost": "#eb6834", "pscost": "#1baf7a",
          "mostinf": "#eda100", "fullstrong": "#e87ba4"}
NAMES = {"learned": "Learned (ours)", "relpscost": "Reliability pseudocost (SCIP default)",
         "pscost": "Pseudocost", "mostinf": "Most fractional", "fullstrong": "Full strong branching"}
SHORT = {"learned": "Learned", "relpscost": "Reliability", "pscost": "Pseudocost",
         "mostinf": "Most fractional", "fullstrong": "Full strong"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def sgm(values, shift):
    x = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(x + shift))) - shift)


def load(path):
    df = pd.read_csv(path)
    df["solved"] = df["status"].eq("optimal")
    df["problem"] = df["instance"] + "|" + df["seed"].astype(str)
    rules = [r for r in COLORS if r in set(df["rule"])] + sorted(set(df["rule"]) - set(COLORS))
    complete = df.groupby("problem")["rule"].nunique() == len(rules)
    dropped = int((~complete).sum())
    df = df[df["problem"].isin(complete[complete].index)]
    return df, rules, dropped


def tables(df, rules, min_time):
    def wide(col):
        return df.pivot_table(index="problem", columns="rule", values=col, aggfunc="first")[rules]

    t_raw, nodes = wide("time_wall"), wide("nodes")
    solved = wide("solved").fillna(False).astype(bool)
    t_eff = t_raw.clip(lower=min_time)
    t_solved = t_eff.where(solved)
    best = t_solved.min(axis=1)
    common = solved.all(axis=1)

    rows = []
    for r in rules:
        d = df[df["rule"] == r]
        gaps = d.loc[~d["solved"], "gap"].replace([np.inf, -np.inf], np.nan).dropna()
        # gap over ALL runs: solved = 0, no primal bound yet (inf) = 100%, capped at 100%
        gap_all = d["gap"].where(~d["solved"], 0.0).replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(upper=1.0)
        rows.append({
            "rule": r,
            "solved": f"{int(solved[r].sum())}/{len(solved)}",
            "SGM time (s)": round(sgm(t_raw[r], 1.0), 2),
            "SGM nodes": round(sgm(nodes[r], 100.0)),
            f"SGM time, commonly solved (n={int(common.sum())})":
                round(sgm(t_raw.loc[common, r], 1.0), 2) if common.any() else np.nan,
            "SGM nodes, commonly solved":
                round(sgm(nodes.loc[common, r], 100.0)) if common.any() else np.nan,
            "fastest (within 5%)": int((t_solved[r] <= 1.05 * best).sum()),
            "avg gap on unsolved (%)": round(100 * gaps.mean(), 2) if len(gaps) else np.nan,
            "avg gap, all runs (%)": round(100 * gap_all.mean(), 2),
        })
    return pd.DataFrame(rows), t_eff, t_solved, best, solved, nodes, common


def paired_lines(rules, ref, t_eff, nodes, common):
    out = []
    if "learned" not in rules or ref not in rules or ref == "learned":
        return out
    ratio = t_eff["learned"] / t_eff[ref]
    faster, slower = int((ratio < 0.95).sum()), int((ratio > 1.05).sum())
    out.append(f"Learned vs {ref}, per problem (n={len(ratio)}): learned faster by >5% on {faster}, "
               f"within 5% on {len(ratio) - faster - slower}, slower by >5% on {slower}.")
    # Seeds of the same instance are not independent, so resample INSTANCES (mean log-ratio over
    # that instance's seeds), 10,000 bootstrap draws, 95% percentile interval.
    per_instance = np.log(ratio).groupby(ratio.index.str.split("|").str[0]).mean().to_numpy()
    draws = np.random.default_rng(0).integers(0, len(per_instance), (10_000, len(per_instance)))
    lo, hi = np.exp(np.percentile(per_instance[draws].mean(axis=1), [2.5, 97.5]))
    verdict = ("learned faster" if hi < 1 else f"{ref} faster" if lo > 1
               else "no clear difference at this sample size")
    out.append(f"Geometric-mean time ratio learned/{ref}: {np.exp(per_instance.mean()):.2f}, 95% CI "
               f"[{lo:.2f}, {hi:.2f}] (bootstrap over {len(per_instance)} instances): {verdict}.")
    if common.any():
        nr = (nodes.loc[common, "learned"] + 1) / (nodes.loc[common, ref] + 1)
        out.append(f"On commonly solved problems (n={int(common.sum())}) learned explores "
                   f"{np.exp(np.log(nr).mean()):.2f}x the nodes of {ref} (geometric mean).")
    return out


def overhead_line(df):
    d = df[df["rule"] == "learned"]
    if d.empty or d["decisions"].isna().all():
        return None
    total, dec = d["time_wall"].sum(), d["decisions"].sum()
    if "feature_time" in d and d["feature_time"].notna().any():
        f, m = d["feature_time"].sum(), d["predict_time"].sum()
        rest = d["decision_time"].sum() - f - m
        return (f"Learned-rule overhead (share of its total solve time): features {100 * f / total:.1f}%, "
                f"model {100 * m / total:.1f}% = {100 * (f + m) / total:.1f}% extra cost vs a classical rule; "
                f"{1000 * f / dec:.2f} ms features + {1000 * m / dec:.2f} ms model per decision. "
                f"(Rest of the callback, e.g. SCIP's branchVar, {100 * rest / total:.1f}%: every rule pays it.)")
    share = d["decision_time"].sum() / total
    return (f"Learned-rule callback: {100 * share:.1f}% of its total solve time is spent inside the rule "
            f"(feature extraction + model + SCIP's branchVar), {1000 * d['decision_time'].sum() / dec:.2f} ms "
            f"per decision. Re-run the learned rule with the current benchmark.py for the split.")


def markdown(table):
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join("-" if (isinstance(v, float) and np.isnan(v)) else str(v)
                                        for v in row) + " |")
    return "\n".join(lines)


def place_labels(ax, ends, min_gap=0.075):
    """Direct labels at the right edge of the axes, pushed apart so they never overlap."""
    ends = sorted(ends, key=lambda e: e[0])
    ys = []
    for y, *_ in ends:
        ys.append(max(y, ys[-1] + min_gap) if ys else y)
    shift = max(0.0, ys[-1] - 1.0)
    for (y0, name, color), y in zip(ends, ys):
        y -= shift
        ax.plot([1.02], [y], marker="s", ms=6, color=color, transform=ax.transAxes, clip_on=False,
                linestyle="none")
        ax.text(1.045, y, name, transform=ax.transAxes, va="center", ha="left", fontsize=8.5, color=INK2)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)


def figure(df, rules, t_eff, t_solved, best, min_time, path):
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ("Segoe UI", "Arial", "DejaVu Sans") if f in installed),
                                       "sans-serif")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.24, top=0.88, wspace=0.55)
    valid = best.notna()
    n_valid = int(valid.sum())

    # (a) performance profile: share of problems each rule solves within tau x the best time
    ratios = {r: (t_solved[r] / best)[valid].fillna(np.inf).to_numpy() for r in rules}
    finite = np.concatenate([x[np.isfinite(x)] for x in ratios.values()])
    tau_max = float(np.clip(finite.max() if finite.size else 2.0, 2.0, 100.0))
    grid = np.logspace(0, np.log10(tau_max), 300)
    for r in rules:
        a.plot(grid, [(ratios[r] <= g).mean() for g in grid], color=COLORS.get(r, MUTED), lw=2.0)
    a.set_xscale("log")
    a.set_xlim(1, tau_max)
    a.set_ylim(0, 1.02)
    a.set_xlabel("time relative to the fastest rule on each problem (log scale)", color=INK2, fontsize=9)
    a.set_ylabel("share of problems", color=INK2, fontsize=9)
    a.set_title(f"Closer to the top-left is better (n={n_valid} problems)", loc="left", color=INK,
                fontsize=10, pad=8)
    style(a)

    # (b) problems solved within a time budget
    n_all = len(t_eff)
    xmax = float(df["time_wall"].max()) * 1.05
    ends = []
    for r in rules:
        ts = np.sort(t_solved[r].dropna().to_numpy())
        b.step(np.r_[min_time, ts, xmax], np.r_[0, np.arange(1, len(ts) + 1), len(ts)],
               where="post", color=COLORS.get(r, MUTED), lw=2.0)
        ends.append((len(ts) / n_all, SHORT.get(r, r), COLORS.get(r, MUTED)))
    b.set_xscale("log")
    b.set_xlim(min_time, xmax)
    b.set_ylim(0, n_all)
    b.set_xlabel("time budget in seconds (log scale)", color=INK2, fontsize=9)
    b.set_ylabel("problems solved", color=INK2, fontsize=9)
    b.set_title(f"Further up is better (n={n_all} problems)", loc="left", color=INK, fontsize=10, pad=8)
    style(b)
    place_labels(b, ends)

    handles = [plt.Line2D([0], [0], color=COLORS.get(r, MUTED), lw=2.0) for r in rules]
    fig.legend(handles, [NAMES.get(r, r) for r in rules], loc="lower center", ncol=len(rules),
               frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.46, 0.005))
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="CSV written by solver/benchmark.py")
    p.add_argument("--out-dir", default="analysis/report")
    p.add_argument("--ref", nargs="+", default=["relpscost", "pscost"],
                   help="reference rule(s) for the paired comparison with the learned rule")
    p.add_argument("--min-time", type=float, default=0.5, help="floor for times when comparing rules")
    args = p.parse_args()

    df, rules, dropped = load(args.results)
    if df.empty:
        raise SystemExit("No problem has a run for every rule; nothing to compare.")
    table, t_eff, t_solved, best, solved, nodes, common = tables(df, rules, args.min_time)
    notes = []
    if dropped:
        notes.append(f"{dropped} runs were dropped because some rule has no run on the same problem.")
    for ref in args.ref:
        notes += paired_lines(rules, ref, t_eff, nodes, common)
    if (line := overhead_line(df)):
        notes.append(line)
    n_unsolved_by_all = int((~solved.any(axis=1)).sum())
    if n_unsolved_by_all:
        notes.append(f"{n_unsolved_by_all} problems were solved by no rule (left out of the profile).")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.results).stem
    table.to_csv(out / f"{stem}_summary.csv", index=False)
    (out / f"{stem}_summary.md").write_text(markdown(table) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    figure(df, rules, t_eff, t_solved, best, args.min_time, out / f"{stem}_profile.png")

    print(table.to_string(index=False))
    print()
    for n in notes:
        print("-", n)
    print(f"\nsaved {stem}_summary.csv, {stem}_summary.md and {stem}_profile.png to {out}/")


if __name__ == "__main__":
    main()