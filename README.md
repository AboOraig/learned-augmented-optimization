# Learning-Augmented Branch-and-Bound for Mixed-Integer Optimization

Learn a branching policy that imitates strong branching using cheap features from the
current B&B state, plug it into SCIP (PySCIPOpt), and compare it with classical rules
(most fractional, pseudocost, reliability pseudocost) on unseen instances.

## Layout
```
instances/   instance generators (+ generated .lp files)
collect/     strong-branching data collection        (step 2)
data/        collected samples (generated)
train/       model training                          (step 3)
solver/      SCIP settings, shared features, benchmark runner, learned branching rule (steps 1 and 4)
analysis/    result tables, plots                    (step 5)
```

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: instances + baselines
```
python instances/generate_setcover.py --split train    --n 300 --rows 500 --cols 1000 --seed 0
python instances/generate_setcover.py --split test     --n 50  --rows 500 --cols 1000 --seed 1000
python instances/generate_setcover.py --split transfer --n 30  --rows 750 --cols 1000 --seed 2000

# quick smoke test first
python solver/benchmark.py --instances instances/setcover/test --limit 3 \
    --rules relpscost mostinf fullstrong --time-limit 60 --out analysis/smoke.csv

# full baseline
python solver/benchmark.py --instances instances/setcover/test \
    --rules relpscost pscost mostinf fullstrong --time-limit 300 --out analysis/baseline_test.csv
```

All scripts take their SCIP settings from `solver/common.py` (no presolve, restarts or cuts;
primal heuristics at SCIP defaults). Never change settings inside one script only: results are
comparable only if every experiment uses the same ones.

## Step 2: collect strong-branching data
Quick check on ONE instance first (look at the output, then scale up):
```
python collect/collect_strong_branching.py --instances instances/setcover/train --out data/samples/check --limit 1 --time-limit 30
```
Then the real collection (resumable: instances already in the output folder are skipped):
```
python collect/collect_strong_branching.py --instances instances/setcover/train --out data/samples/train --workers 4
# separate validation instances (never the test set)
python instances/generate_setcover.py --split val --n 30 --rows 500 --cols 1000 --seed 500
python collect/collect_strong_branching.py --instances instances/setcover/val --out data/samples/val --workers 4
```
Choices to know about (they are part of the method):
- Nodes are sampled at random (root always, then 10% of nodes, at most 30 per instance).
- At a sampled node strong branching runs on at most 60 randomly chosen LP candidates, not all of them.
- The label is the candidate with the best product score of the two LP bound gains.
- At sampled nodes the tree follows the expert's choice; elsewhere SCIP's reliability branching decides.

## Step 3: train the model
```
python train/train_ranker.py --train data/samples/train --val data/samples/val --out train/model
python train/train_ranker.py --objective binary --out train/model_binary   # optional comparison
```
Prints offline decision quality on validation nodes (top1 / top3 / ratio) for the learned model, SCIP's
pseudocost score, most-fractional and random, then saves `train/model/model.txt` and `meta.json`
(the exact feature list the SCIP branching rule must reproduce in step 4).
Offline metrics are only a proxy: validation is used for early stopping, and a good imitation of
strong branching does not guarantee smaller trees or less time. Step 5 is the real test.

## Step 4: the learned rule inside SCIP
`solver/features.py` computes the features for BOTH data collection and the learned rule, so the model
sees at solve time exactly what it saw in training. `solver/learned_branching.py` is the SCIP branching
rule (scores all LP candidates with the trained model, branches on the best, times itself).
`solver/benchmark.py` runs any SCIP rule or `learned` under identical settings.

Smoke test on VALIDATION instances (keep the test set untouched until the final benchmark):
```
python solver/benchmark.py --instances instances/setcover/val --limit 5 --rules relpscost mostinf learned --model train/model --time-limit 60 --out analysis/smoke_learned.csv
```
Columns to look at: nodes, time_wall, gap, and for `learned` decisions / decision_time
(decision_time / decisions = overhead per branching decision).
If you change a feature in features.py you must re-collect the data and retrain.