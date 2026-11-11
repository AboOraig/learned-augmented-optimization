| rule | solved | SGM time (s) | SGM nodes | SGM time, commonly solved (n=90) | SGM nodes, commonly solved | fastest (within 5%) | avg gap on unsolved (%) |
|---|---|---|---|---|---|---|---|
| learned | 100/100 | 2.38 | 297 | 1.8 | 222 | 46 | - |
| relpscost | 100/100 | 4.41 | 132 | 3.61 | 85 | 3 | - |
| pscost | 100/100 | 2.21 | 388 | 1.62 | 289 | 78 | - |
| mostinf | 90/100 | 9.13 | 2104 | 6.69 | 1571 | 6 | 3.66 |

- Learned vs relpscost, per problem (n=100): learned faster by >5% on 97, within 5% on 3, slower by >5% on 0.
- Geometric-mean time ratio learned/relpscost: 0.52, 95% CI [0.48, 0.57] (bootstrap over 50 instances): learned faster.
- On commonly solved problems (n=90) learned explores 3.69x the nodes of relpscost (geometric mean).
- Learned vs pscost, per problem (n=100): learned faster by >5% on 22, within 5% on 24, slower by >5% on 54.
- Geometric-mean time ratio learned/pscost: 1.09, 95% CI [1.04, 1.14] (bootstrap over 50 instances): pscost faster.
- On commonly solved problems (n=90) learned explores 0.77x the nodes of pscost (geometric mean).
- Learned-rule overhead: 12.2% of its total solve time is spent inside the rule (feature extraction + model), 1.74 ms per branching decision.
