"""The learned branching rule: a trained LightGBM model used as a SCIP branching rule.

At every node where SCIP calls it, the rule
  1. computes the features the model was trained on (same code as data collection: features.py),
  2. scores all LP branching candidates with the model,
  3. branches on the highest-scoring candidate.
The time spent in steps 1-3 is measured (decision_time) so the computational overhead of the
learned policy can be reported separately from SCIP's own time.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from pyscipopt import Branchrule, SCIP_RESULT

from features import ALL_FEATURES, candidate_columns, column_degrees, node_features


class LearnedBranchrule(Branchrule):
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        self.features = meta["features"]
        unknown = [f for f in self.features if f not in ALL_FEATURES]
        if unknown:
            raise ValueError(f"model uses features that features.py does not provide: {unknown}")
        self.needed = set(self.features)
        self.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        self.root_lpobj = None
        self.col_deg = None
        self.n_decisions = 0
        self.decision_time = 0.0

    def branchexeclp(self, allowaddcons):
        t0 = time.perf_counter()
        m = self.model
        cands, vals, _, ncands, _, _ = m.getLPBranchCands()
        cands, vals = cands[:ncands], vals[:ncands]
        if self.root_lpobj is None:  # the rule runs at the root first
            self.root_lpobj = m.getLPObjVal()
        if self.col_deg is None:
            self.col_deg = column_degrees(m)

        if ncands == 1:
            best = 0
        else:
            node = node_features(m, ncands, self.root_lpobj, self.needed)
            cols = candidate_columns(m, cands, vals, self.col_deg, self.needed)
            X = np.empty((ncands, len(self.features)))
            for j, name in enumerate(self.features):
                X[:, j] = cols[name] if name in cols else node[name]
            best = int(np.argmax(self.booster.predict(X, num_threads=1)))

        m.branchVar(cands[best])
        self.n_decisions += 1
        self.decision_time += time.perf_counter() - t0
        return {"result": SCIP_RESULT.BRANCHED}
