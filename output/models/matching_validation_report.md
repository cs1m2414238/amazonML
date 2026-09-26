# LightGBM Business Entity Matcher ? Validation Report

> [!IMPORTANT]
> **Official Scoring**: Evaluated strictly on held-out validation entities using real retrieved candidates. Macro-{0.5}$ weights precision over recall ({0.5} = rac{5 TP}{5 TP + 4 FP + FN}$ for non-singletons, .0$ for correctly empty singletons, .0$ for false-positive singletons).

## Summary Metrics

- **Held-Out Macro-0.5$**: **87.54%**
- **Optimal Decision Threshold**: **0.7**
- **Pair Precision**: 95.44%
- **Pair Recall**: 78.03%
- **Singleton Accuracy**: 80.65%
- **Candidate Recall Ceiling (Blocking)**: 94.55%
- **Average Predicted Matches / Query**: 2.85
- **Candidate Subset Violations**: 0

## Performance Across Diagnostic Strata

| Stratum | Entities | Macro-F0.5 | Pair Precision | Pair Recall | Singleton Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| both_sources | 396 | 89.38% | 96.72% | 77.63% | 100.00% |
| country_india | 179 | 82.27% | 93.32% | 69.73% | 90.91% |
| country_us | 321 | 90.48% | 96.46% | 82.62% | 75.00% |
| multi_2_4 | 312 | 89.02% | 95.78% | 78.51% | 100.00% |
| multi_5_plus | 128 | 90.56% | 97.26% | 77.60% | 100.00% |
| one_match | 29 | 65.69% | 75.00% | 72.41% | 100.00% |
| s2_only | 21 | 73.77% | 86.11% | 86.11% | 100.00% |
| s3_only | 52 | 83.20% | 90.00% | 81.08% | 100.00% |
| singleton | 31 | 80.65% | 0.00% | 100.00% | 80.65% |

## Threshold Optimization Curve

| Threshold | Macro-F0.5 | Pair Precision | Pair Recall | Singleton Acc |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 61.75% | 56.42% | 93.00% | 0.00% |
| 0.10 | 71.17% | 68.89% | 91.97% | 9.68% |
| 0.20 | 80.09% | 79.45% | 90.48% | 35.48% |
| 0.30 | 83.73% | 84.69% | 88.87% | 48.39% |
| 0.40 | 86.65% | 88.95% | 86.40% | 61.29% |
| 0.50 | 87.11% | 91.48% | 83.82% | 67.74% |
| 0.60 | 87.46% | 93.64% | 81.12% | 70.97% |
| 0.70 **(Optimal)** | 87.54% | 95.44% | 78.03% | 80.65% |
| 0.75 | 86.90% | 96.09% | 76.19% | 83.87% |
| 0.80 | 85.95% | 96.42% | 74.07% | 83.87% |
| 0.85 | 85.12% | 97.21% | 72.00% | 83.87% |
| 0.90 | 84.11% | 98.05% | 69.36% | 87.10% |
| 0.92 | 83.72% | 98.65% | 66.95% | 93.55% |
| 0.95 | 81.46% | 98.99% | 62.13% | 96.77% |
| 0.97 | 75.46% | 99.26% | 53.70% | 96.77% |
| 0.98 | 67.66% | 99.61% | 43.72% | 100.00% |
| 0.99 | 6.59% | 100.00% | 0.17% | 100.00% |