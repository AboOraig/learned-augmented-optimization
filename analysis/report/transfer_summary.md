| rule | solved | SGM time (s) | SGM nodes | SGM time, commonly solved (n=60) | SGM nodes, commonly solved | fastest (within 5%) | avg gap on unsolved (%) |
|---|---|---|---|---|---|---|---|
| learned | 60/60 | 10.05 | 1138 | 10.05 | 1138 | 36 | - |
| relpscost | 60/60 | 15.93 | 634 | 15.93 | 634 | 5 | - |
| pscost | 60/60 | 9.85 | 1420 | 9.85 | 1420 | 41 | - |

- Learned vs relpscost, per problem (n=60): learned faster by >5% on 55, within 5% on 4, slower by >5% on 1.
- Geometric-mean time ratio learned/relpscost: 0.61, 95% CI [0.54, 0.68] (bootstrap over 30 instances): learned faster.
- On commonly solved problems (n=60) learned explores 2.12x the nodes of relpscost (geometric mean).
- Learned vs pscost, per problem (n=60): learned faster by >5% on 19, within 5% on 18, slower by >5% on 23.
- Geometric-mean time ratio learned/pscost: 1.03, 95% CI [0.99, 1.08] (bootstrap over 30 instances): no clear difference at this sample size.
- On commonly solved problems (n=60) learned explores 0.79x the nodes of pscost (geometric mean).
- Learned-rule overhead: 9.7% of its total solve time is spent inside the rule (feature extraction + model), 1.44 ms per branching decision.
