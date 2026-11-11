# Learning-Augmented Branch-and-Bound for Mixed-Integer Optimization

Learn a branching policy that imitates strong branching using cheap features from the
current B&B state, plug it into SCIP (PySCIPOpt), and compare it with classical rules
(most fractional, pseudocost, reliability pseudocost) on unseen instances.

**Results:** [Learned_Augmented_Optimization.pdf](paper/Learned_Augmented_Optimization.pdf) (source: [`paper/main.tex`](paper/main.tex)).

## Layout
```
instances/   instance generators (+ generated .lp files)
collect/     strong-branching data collection        (step 2)
data/        collected samples (generated)
train/       model training                          (step 3)
solver/      SCIP settings, shared features, benchmark runner, learned branching rule (steps 1 and 4)
analysis/    result tables, plots                    (step 5)
paper/       short technical report on this study.
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

## Step 5: full benchmark and analysis
Run on the TEST instances (never used for training or early stopping). Close other heavy programs, since
parallel runs share the CPU. Try `--limit 5` first to estimate the duration.
```
python solver/benchmark.py --instances instances/setcover/test --rules relpscost pscost mostinf learned --model train/model --time-limit 120 --seeds 0 1 --workers 2 --out analysis/test.csv
python analysis/analyze.py --results analysis/test.csv --out-dir analysis/report
```
The second command prints and saves `analysis/report/test_summary.csv|md` (the table view) and
`test_profile.png` (performance profile + problems solved vs time budget). Conventions used (shifted
geometric means, what counts as solved or fastest) are listed at the top of `analysis/analyze.py`; quote
them in the write-up. Repeat with `instances/setcover/transfer` (larger instances) to study generalisation.

---

# Results

All numbers below come from the CSV files in `analysis/` (produced by `solver/benchmark.py`, summarised by
`analysis/analyze.py`; figures in `analysis/report/*_profile.png`).

**Setup.** Random set-covering instances (500 rows x 1000 columns). 300 for training, 30 for validation
(early stopping only), 50 for testing, plus 30 larger unseen instances (750 x 1000) to test generalisation.
SCIP via PySCIPOpt with presolve, cuts and restarts off, primal heuristics on (see `solver/common.py`).
Model: LightGBM LambdaRank trained to imitate strong branching on 3,869 sampled nodes (strong branching on
at most 60 random candidates per node), 19 features. SGM = shifted geometric mean (1 s for time, 100 for nodes).
Intervals are 95% bootstrap intervals resampling instances.

**Offline imitation quality (30 validation instances, 395 nodes).** Picks the strong-branching choice in
19.2% of nodes (top-3: 40.3%), against 15.7% (34.7%) for SCIP's pseudocost score, 4.8% for most-fractional and
1.8% for random. The model mostly re-learns pseudocost information (top features: pseudocost rank,
objective-coefficient rank, fractionality).

**Test instances (50 instances x 2 seeds = 100 problems, 120 s limit).**

| Rule | Solved | SGM time (s) | SGM nodes | Fastest |
|---|---|---|---|---|
| Learned | 100/100 | 2.38 | 297 | 46 |
| Reliability pseudocost (SCIP default rule) | 100/100 | 4.41 | 132 | 3 |
| Pseudocost | 100/100 | **2.21** | 388 | **78** |
| Most fractional | 90/100 | 9.13 | 2104 | 6 |

Learned / reliability: time ratio 0.52 [0.48, 0.57] (learned faster), 3.7x the nodes.
Learned / pseudocost: time ratio 1.09 [1.04, 1.14] (pseudocost faster), 0.77x the nodes.

**Larger unseen instances (30 instances x 2 seeds = 60 problems, 120 s limit).**

| Rule | Solved | SGM time (s) | SGM nodes | Fastest |
|---|---|---|---|---|
| Learned | 60/60 | 10.05 | 1138 | 36 |
| Reliability pseudocost | 60/60 | 15.93 | 634 | 5 |
| Pseudocost | 60/60 | 9.85 | 1420 | 41 |

Learned / reliability: 0.61 [0.54, 0.68], 2.1x the nodes. Learned / pseudocost: 1.03 [0.99, 1.08]
(no clear difference), 0.79x the nodes.

**Tight time budget (larger instances, 10 s, seed 0; optimality gap is only informative here).**

| Rule | Solved | Avg gap, all runs | Avg gap, unsolved runs |
|---|---|---|---|
| Learned | 16/30 | 2.83% | 6.06% |
| Reliability pseudocost | 9/30 | 6.15% | 8.78% |
| Pseudocost | 16/30 | 2.98% | 6.40% |

The gap difference between learned and pseudocost (2.83% vs 2.98%) was not tested and should not be read as a win.

**Computational overhead of the learned rule.** On the test set it spends 11.7% of its solve time on
extra work: 9.9% extracting features (1.29 ms per decision) and 1.8% scoring with the model (0.23 ms).
Feature extraction, done through many small SCIP calls from Python, is about 85% of the overhead.
The overhead (about 8 to 12% across experiments) does not by itself explain why a ~20% node reduction
versus pseudocost is not a time advantage: on the test set the learned rule's nodes cost roughly 40% more
time (1.09 / 0.77), of which only a factor ~1.13 is the measured overhead; the rest is undiagnosed (LP
iterations were not recorded). Removing the overhead entirely would, by this rough estimate, give about
0.96 of pseudocost's time, i.e. at most a few percent faster.

**Conclusions.** (1) The learned rule is clearly better than most-fractional and than reliability
pseudocost branching under this protocol, mainly because reliability branching pays for expensive
strong-branching probes. (2) Compared with plain pseudocost it builds smaller trees (about 20% fewer nodes,
also on larger unseen instances) but is not faster: it is slower on the test set and indistinguishable on the
larger instances. (3) The model generalises from 500-row to 750-row instances without retraining.

**Limitations.** One problem class (set covering). The protocol disables cuts, presolve and restarts, so the
results say nothing about SCIP with default settings. Only 2 seeds; runs used 2 parallel workers, so timings
carry some noise. Validation instances were used for early stopping. Training labels come from strong
branching on a random subset of at most 60 candidates per node. The learned rule is implemented in Python;
a native implementation would lower the per-decision cost, but by the estimate above that alone would not make it clearly faster than pseudocost.

---