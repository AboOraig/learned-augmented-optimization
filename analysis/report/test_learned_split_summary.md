| rule | solved | SGM time (s) | SGM nodes | SGM time, commonly solved (n=50) | SGM nodes, commonly solved | fastest (within 5%) | avg gap on unsolved (%) | avg gap, all runs (%) |
|---|---|---|---|---|---|---|---|---|
| learned | 50/50 | 2.31 | 298 | 2.31 | 298 | 50 | - | 0.0 |

- Learned-rule overhead (share of its total solve time): features 9.9%, model 1.8% = 11.7% extra cost vs a classical rule; 1.29 ms features + 0.23 ms model per decision. (Rest of the callback, e.g. SCIP's branchVar, 0.5%: every rule pays it.)
