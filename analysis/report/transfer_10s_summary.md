| rule | solved | SGM time (s) | SGM nodes | SGM time, commonly solved (n=9) | SGM nodes, commonly solved | fastest (within 5%) | avg gap on unsolved (%) | avg gap, all runs (%) |
|---|---|---|---|---|---|---|---|---|
| learned | 16/30 | 6.71 | 548 | 3.15 | 258 | 6 | 6.06 | 2.83 |
| relpscost | 9/30 | 8.73 | 56 | 6.86 | 79 | 0 | 8.78 | 6.15 |
| pscost | 16/30 | 6.28 | 674 | 2.77 | 336 | 13 | 6.4 | 2.98 |

- Learned vs relpscost, per problem (n=30): learned faster by >5% on 14, within 5% on 14, slower by >5% on 2.
- Geometric-mean time ratio learned/relpscost: 0.75, 95% CI [0.66, 0.86] (bootstrap over 30 instances): learned faster.
- On commonly solved problems (n=9) learned explores 3.71x the nodes of relpscost (geometric mean).
- Learned vs pscost, per problem (n=30): learned faster by >5% on 3, within 5% on 12, slower by >5% on 15.
- Geometric-mean time ratio learned/pscost: 1.08, 95% CI [1.02, 1.14] (bootstrap over 30 instances): pscost faster.
- On commonly solved problems (n=9) learned explores 0.76x the nodes of pscost (geometric mean).
- Learned-rule overhead (share of its total solve time): features 6.8%, model 1.2% = 8.0% extra cost vs a classical rule; 1.42 ms features + 0.26 ms model per decision. (Rest of the callback, e.g. SCIP's branchVar, 0.4%: every rule pays it.)
- 13 problems were solved by no rule (left out of the profile).
