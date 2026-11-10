"""SCIP settings shared by the baseline runner, the data collector and the learned rule.

Every experiment must use exactly these settings, otherwise node counts and times are not
comparable. Edit them here, never in the individual scripts.

Setup follows the usual learning-to-branch protocol (Gasse et al., 2019): no presolve, no
restarts, no cuts, so that the branching rule is what drives the size of the tree.
Primal heuristics are left at their SCIP defaults, so that a primal bound (and hence an
optimality gap) is available when a run hits the time limit.
"""
from pyscipopt import SCIP_PARAMSETTING


def apply_base_settings(model, time_limit, seed=0):
    model.hideOutput()
    # NOTE: the argument is the SCIP_PARAMSETTING enum. In SCIP, DEFAULT != OFF, so passing the
    # integer 0 does NOT switch anything off. Always use the named values.
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setIntParam("presolving/maxrounds", 0)
    model.setIntParam("presolving/maxrestarts", 0)
    model.setIntParam("separating/maxrounds", 0)       # no cuts at non-root nodes
    model.setIntParam("separating/maxroundsroot", 0)   # no cuts at the root
    model.setRealParam("limits/time", time_limit)
    model.setIntParam("randomization/randomseedshift", seed)
