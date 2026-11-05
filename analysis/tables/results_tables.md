### Table 1. Main results (all mazes, 30 seeds)

| Maze | Method | Final path optimal | Mean path gap | Runs reaching & keeping optimum | Real steps to keep optimum (mean [95% CI]) | Total updates (mean) | Time, s (mean) | Sweeps (VI) | Hazard hits per run |
|---|---|---|---|---|---|---|---|---|---|
| small | Value Iteration | 30/30 | 0.00 | - | 0 (model given) | 3,600 | 0.02 | 25 | - |
| small | Q-learning | 30/30 | 0.00 | 30/30 | 21,042 [20,673, 21,412] | 68,510 | 0.39 | - | - |
| small | Dyna-Q (n=5) | 30/30 | 0.00 | 30/30 | 4,708 [4,364, 5,051] | 385,088 | 2.03 | - | - |
| small | Dyna-Q (n=20) | 30/30 | 0.00 | 30/30 | 2,815 [2,419, 3,211] | 1,347,393 | 7.65 | - | - |
| small | Dyna-Q (n=50) | 30/30 | 0.00 | 30/30 | 2,513 [2,091, 2,935] | 3,273,593 | 18.54 | - | - |
| large | Value Iteration | 30/30 | 0.00 | - | 0 (model given) | 39,668 | 0.13 | 47 | - |
| large | Q-learning | 30/30 | 0.00 | 30/30 | 120,157 [119,486, 120,828] | 180,520 | 1.03 | - | - |
| large | Dyna-Q (n=5) | 30/30 | 0.00 | 30/30 | 36,269 [34,296, 38,242] | 712,627 | 4.58 | - | - |
| large | Dyna-Q (n=20) | 30/30 | 0.00 | 30/30 | 16,887 [15,213, 18,561] | 2,363,051 | 13.94 | - | - |
| large | Dyna-Q (n=50) | 30/30 | 0.00 | 30/30 | 13,099 [11,893, 14,305] | 5,686,774 | 36.33 | - | - |
| cliff | Value Iteration | 30/30 | 0.00 | - | 0 (model given) | 2,880 | 0.01 | 15 | - |
| cliff | Q-learning | 7/30 | 1.60 | 7/30 | 12,940 [11,878, 14,002] | 59,385 | 0.33 | - | 2,338 |
| cliff | Dyna-Q (n=5) | 30/30 | 0.00 | 30/30 | 7,459 [6,134, 8,785] | 446,962 | 3.17 | - | 4,487 |
| cliff | Dyna-Q (n=20) | 30/30 | 0.00 | 30/30 | 6,615 [5,088, 8,142] | 1,655,623 | 10.62 | - | 5,114 |
| cliff | Dyna-Q (n=50) | 30/30 | 0.00 | 30/30 | 6,528 [4,901, 8,154] | 4,026,884 | 24.17 | - | 5,149 |
| change | Value Iteration | 30/30 | 0.00 | - | 0 (model given) | 3,772 | 0.02 | 23 | - |
| change | Q-learning | 1/30 | 17.40 | 30/30 | 18,035 [17,801, 18,269] | 62,553 | 0.39 | - | - |
| change | Dyna-Q (n=5) | 27/30 | 1.80 | 30/30 | 3,374 [3,082, 3,666] | 192,595 | 1.44 | - | - |
| change | Dyna-Q (n=20) | 30/30 | 0.00 | 30/30 | 2,204 [1,848, 2,561] | 598,296 | 3.78 | - | - |
| change | Dyna-Q (n=50) | 26/30 | 2.40 | 30/30 | 1,952 [1,548, 2,356] | 1,692,410 | 10.21 | - | - |

### Table 2. Q-learning vs Dyna-Q: steps to keep the optimal path (runs that never did are counted at their run length)

| Maze | Comparison | Median steps, Q-learning | Median steps, Dyna-Q | Effect size (negative = Dyna-Q needs fewer steps) | p (Mann-Whitney U) | p (Holm-adjusted) | Significant (adj. p < 0.05) |
|---|---|---|---|---|---|---|---|
| small | Q-learning vs Dyna-Q (n=5) | 21,132 | 4,690 | -1.0 | <0.001 | <0.001 | yes |
| small | Q-learning vs Dyna-Q (n=20) | 21,132 | 2,764 | -1.0 | <0.001 | <0.001 | yes |
| small | Q-learning vs Dyna-Q (n=50) | 21,132 | 2,200 | -1.0 | <0.001 | <0.001 | yes |
| large | Q-learning vs Dyna-Q (n=5) | 119,900 | 37,730 | -1.0 | <0.001 | <0.001 | yes |
| large | Q-learning vs Dyna-Q (n=20) | 119,900 | 16,419 | -1.0 | <0.001 | <0.001 | yes |
| large | Q-learning vs Dyna-Q (n=50) | 119,900 | 13,311 | -1.0 | <0.001 | <0.001 | yes |
| cliff | Q-learning vs Dyna-Q (n=5) | 57,394 | 7,528 | -0.96 | <0.001 | <0.001 | yes |
| cliff | Q-learning vs Dyna-Q (n=20) | 57,394 | 5,516 | -0.96 | <0.001 | <0.001 | yes |
| cliff | Q-learning vs Dyna-Q (n=50) | 57,394 | 5,280 | -0.95 | <0.001 | <0.001 | yes |
| change | Q-learning vs Dyna-Q (n=5) | 18,036 | 3,302 | -1.0 | <0.001 | <0.001 | yes |
| change | Q-learning vs Dyna-Q (n=20) | 18,036 | 1,936 | -1.0 | <0.001 | <0.001 | yes |
| change | Q-learning vs Dyna-Q (n=50) | 18,036 | 1,745 | -1.0 | <0.001 | <0.001 | yes |

### Table 3. Changing maze

| Method | Optimal path (22) found before change | Ever found the shortcut (4) | Using shortcut at the end | Median real steps to find it (after change) |
|---|---|---|---|---|
| Q-learning | 30/30 | 1/30 | 1/30 | 1,578 |
| Dyna-Q (n=5) | 30/30 | 27/30 | 27/30 | 1,165 |
| Dyna-Q (n=20) | 30/30 | 30/30 | 30/30 | 483 |
| Dyna-Q (n=50) | 30/30 | 26/30 | 26/30 | 402 |

### Table 4. Planning-steps ablation (Dyna-Q)

| Maze | n | Reached & kept optimum | Real steps (mean) | Total updates (mean) | Time, s (mean) | Model entries (mean) |
|---|---|---|---|---|---|---|
| small | 0 | 30/30 | 21,042 | 68,510 | 0.54 | 140 |
| small | 5 | 30/30 | 4,708 | 385,088 | 2.03 | 140 |
| small | 20 | 30/30 | 2,815 | 1,347,393 | 7.65 | 140 |
| small | 50 | 30/30 | 2,513 | 3,273,593 | 18.54 | 140 |
| large | 0 | 30/30 | 120,157 | 180,520 | 2.03 | 840 |
| large | 5 | 30/30 | 36,269 | 712,627 | 4.58 | 814 |
| large | 20 | 30/30 | 16,887 | 2,363,051 | 13.94 | 811 |
| large | 50 | 30/30 | 13,099 | 5,686,774 | 36.33 | 793 |
| cliff | 0 | 7/30 | 12,940 | 59,385 | 0.70 | 148 |
| cliff | 5 | 30/30 | 7,459 | 446,962 | 3.17 | 148 |
| cliff | 20 | 30/30 | 6,615 | 1,655,623 | 10.62 | 148 |
| cliff | 50 | 30/30 | 6,528 | 4,026,884 | 24.17 | 148 |
| change | 0 | 30/30 | 18,035 | 62,553 | 0.78 | 162 |
| change | 5 | 30/30 | 3,374 | 192,595 | 1.44 | 164 |
| change | 20 | 30/30 | 2,204 | 598,296 | 3.78 | 164 |
| change | 50 | 30/30 | 1,952 | 1,692,410 | 10.21 | 164 |
