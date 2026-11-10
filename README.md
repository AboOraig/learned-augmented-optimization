# Learning-Augmented Branch-and-Bound for Mixed-Integer Optimization

Learn a branching policy that imitates strong branching using cheap features from the
current B&B state, plug it into SCIP (PySCIPOpt), and compare it with classical rules
(most fractional, pseudocost, reliability pseudocost) on unseen instances.

## Layout
```
instances/   instance generators (+ generated .lp files)
collect/     strong-branching data collection        (step 2)
train/       model training                          (step 3)
solver/      baseline runner, learned branching rule (steps 1 and 4)
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
python solver/baseline.py --instances instances/setcover/test --limit 3 \
    --rules relpscost mostinf fullstrong --time-limit 60 --out analysis/smoke.csv

# full baseline
python solver/baseline.py --instances instances/setcover/test \
    --rules relpscost pscost mostinf fullstrong --time-limit 300 --out analysis/baseline_test.csv
```