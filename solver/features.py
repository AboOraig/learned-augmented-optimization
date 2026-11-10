"""Feature extraction shared by data collection and the learned branching rule.

The SAME code computes the features when we collect training data and when the trained model
branches inside SCIP, so the two cannot drift apart. To change or add a feature: edit it here,
re-collect the data, retrain.

Features are split in node-level (same for all candidates of a node) and candidate-level, plus
within-node percentile ranks of some candidate features (these transfer better across instances
than raw values).

`needed` (a set of feature names, from the trained model) lets the solver compute only what the
model uses, which keeps the per-node overhead down. needed=None computes everything.
"""
import math

import numpy as np
from scipy.stats import rankdata

EPS = 1e-6
BASIS_CODE = {"lower": 0, "basic": 1, "upper": 2, "zero": 3}
RANKED = ["frac_dist", "obj", "redcost", "pc_score", "col_nnz"]
NODE_FEATURES = ["depth", "n_nodes", "lpobj", "lpobj_rel_root", "has_incumbent", "gap",
                 "dual_rel_lp", "frac_fixed", "n_cands"]
CANDIDATE_FEATURES = ["frac", "frac_dist", "obj", "redcost", "is_basic", "at_upper", "col_nnz",
                      "lb_loc", "ub_loc", "lb_glob", "ub_glob", "pc_down", "pc_up", "pc_score"]
ALL_FEATURES = NODE_FEATURES + CANDIDATE_FEATURES + [f"{c}_rank" for c in RANKED]

FAILED = {}  # feature name -> number of times extraction raised (API missing in this version)


def safe(name, fn, default=0.0):
    """Call fn(); on any error return default and remember that this feature is unavailable."""
    try:
        value = fn()
        return default if value is None else value
    except Exception:
        FAILED[name] = FAILED.get(name, 0) + 1
        return default


def rank_pct(values):
    """Percentile rank in (0, 1], ties share the average rank."""
    return rankdata(values, method="average") / len(values)


def column_degrees(model):
    """Number of LP rows each variable appears in. Computed once at the root: cuts are off, so
    the LP rows are the problem's constraints and stay the same at every node."""
    deg = {}
    try:
        for row in model.getLPRowsData():
            for col in row.getCols():
                name = col.getVar().name
                deg[name] = deg.get(name, 0) + 1
    except Exception:
        deg = {}
    if not deg:
        FAILED["col_nnz"] = FAILED.get("col_nnz", 0) + 1
    return deg


def node_features(model, ncands, root_lpobj, needed=None):
    def want(name):
        return needed is None or name in needed

    m = model
    lpobj = m.getLPObjVal()
    f = {"n_cands": ncands, "lpobj": lpobj,
         "lpobj_rel_root": (lpobj - root_lpobj) / (abs(root_lpobj) + EPS)}
    if want("depth"):
        f["depth"] = m.getCurrentNode().getDepth()
    if want("n_nodes"):
        f["n_nodes"] = m.getNNodes()
    if want("has_incumbent") or want("gap"):
        has_inc = m.getPrimalbound() < 1e19
        gap = m.getGap() if has_inc else 1.0
        if not math.isfinite(gap) or gap > 1.0:
            gap = 1.0
        f["has_incumbent"] = int(has_inc)
        f["gap"] = gap
    if want("dual_rel_lp"):
        f["dual_rel_lp"] = (lpobj - m.getDualbound()) / (abs(lpobj) + EPS)
    if want("frac_fixed"):  # the most expensive node feature: loops over all variables
        variables = m.getVars()
        n_fixed = sum(1 for v in variables if v.getLbLocal() >= v.getUbLocal() - EPS)
        f["frac_fixed"] = n_fixed / max(1, len(variables))
    return f


def candidate_columns(model, cands, vals, col_deg, needed=None):
    """Candidate features as {name: numpy array of length len(cands)}, ranks included."""
    def want(*names):
        return needed is None or any(n in needed for n in names)

    m = model
    cols = {}
    frac = np.array([v - math.floor(v) for v in vals], dtype=float)
    cols["frac"] = frac
    cols["frac_dist"] = np.minimum(frac, 1.0 - frac)
    if want("obj", "obj_rank"):
        cols["obj"] = np.array([v.getObj() for v in cands], dtype=float)
    if want("redcost", "redcost_rank"):
        cols["redcost"] = np.array([safe("redcost", lambda v=v: m.getVarRedcost(v)) for v in cands],
                                   dtype=float)
    if want("is_basic", "at_upper"):
        codes = np.array([safe("basis_status",
                               lambda v=v: BASIS_CODE.get(v.getCol().getBasisStatus(), 3), 3)
                          for v in cands])
        cols["is_basic"] = (codes == 1).astype(float)
        cols["at_upper"] = (codes == 2).astype(float)
    if want("col_nnz", "col_nnz_rank"):
        cols["col_nnz"] = np.array([col_deg.get(v.name, 0) for v in cands], dtype=float)
    if want("lb_loc", "ub_loc", "lb_glob", "ub_glob"):
        cols["lb_loc"] = np.array([v.getLbLocal() for v in cands], dtype=float)
        cols["ub_loc"] = np.array([v.getUbLocal() for v in cands], dtype=float)
        cols["lb_glob"] = np.array([v.getLbGlobal() for v in cands], dtype=float)
        cols["ub_glob"] = np.array([v.getUbGlobal() for v in cands], dtype=float)
    if want("pc_down"):
        cols["pc_down"] = np.array([safe("pc_down", lambda v=v: m.getVarPseudocost(v, 0))
                                    for v in cands], dtype=float)
    if want("pc_up"):
        cols["pc_up"] = np.array([safe("pc_up", lambda v=v: m.getVarPseudocost(v, 1))
                                  for v in cands], dtype=float)
    if want("pc_score", "pc_score_rank"):
        cols["pc_score"] = np.array([safe("pc_score", lambda v=v, x=x: m.getVarPseudocostScore(v, x))
                                     for v, x in zip(cands, vals)], dtype=float)
    for c in RANKED:
        if c in cols and (needed is None or f"{c}_rank" in needed):
            cols[f"{c}_rank"] = rank_pct(cols[c])
    return cols
