# LightGBM Business Entity Matcher ? Validation Report

> [!IMPORTANT]
> **Official Scoring**: Evaluated strictly on held-out validation entities using real retrieved candidates. Macro-{0.5}$ weights precision over recall ({0.5} = rac{5 TP}{5 TP + 4 FP + FN}$ for non-singletons, .0$ for correctly empty singletons, .0$ for false-positive singletons).

## Summary Metrics

- **Held-Out Macro-0.5$**: **90.33%**
- **Optimal Decision Threshold**: **0.6**
- **Pair Precision**: 94.67%
- **Pair Recall**: 84.57%
- **Singleton Accuracy**: 83.87%
- **Candidate Recall Ceiling (Blocking)**: 94.55%
- **Average Predicted Matches / Query**: 3.11
- **Candidate Subset Violations**: 0

## Performance Across Diagnostic Strata

| Stratum | Entities | Macro-F0.5 | Pair Precision | Pair Recall | Singleton Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| both_sources | 396 | 91.62% | 95.66% | 84.21% | 100.00% |
| country_india | 179 | 85.48% | 92.06% | 78.42% | 90.91% |
| country_us | 321 | 93.04% | 96.01% | 87.97% | 80.00% |
| multi_2_4 | 312 | 90.73% | 93.91% | 84.73% | 100.00% |
| multi_5_plus | 128 | 93.48% | 97.17% | 84.43% | 100.00% |
| one_match | 29 | 79.10% | 82.76% | 82.76% | 100.00% |
| s2_only | 21 | 84.13% | 96.97% | 88.89% | 100.00% |
| s3_only | 52 | 86.92% | 86.73% | 88.29% | 100.00% |
| singleton | 31 | 83.87% | 0.00% | 100.00% | 83.87% |

## Threshold Optimization Curve

| Threshold | Macro-F0.5 | Pair Precision | Pair Recall | Singleton Acc |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 71.44% | 68.13% | 93.69% | 12.90% |
| 0.10 | 78.17% | 76.50% | 93.17% | 25.81% |
| 0.20 | 83.91% | 83.59% | 92.03% | 45.16% |
| 0.30 | 86.99% | 88.01% | 90.13% | 54.84% |
| 0.40 | 88.27% | 90.98% | 88.01% | 58.06% |
| 0.50 | 89.44% | 92.98% | 86.57% | 70.97% |
| 0.60 **(Optimal)** | 90.33% | 94.67% | 84.57% | 83.87% |
| 0.70 | 89.53% | 95.66% | 82.21% | 87.10% |
| 0.75 | 89.41% | 96.13% | 81.18% | 87.10% |
| 0.80 | 89.27% | 96.68% | 80.09% | 90.32% |
| 0.85 | 88.78% | 97.56% | 77.85% | 96.77% |
| 0.90 | 87.49% | 98.32% | 74.01% | 100.00% |
| 0.92 | 86.64% | 98.44% | 72.46% | 100.00% |
| 0.95 | 84.42% | 98.84% | 68.33% | 100.00% |
| 0.97 | 81.34% | 99.10% | 63.11% | 100.00% |
| 0.98 | 78.94% | 99.71% | 59.55% | 100.00% |
| 0.99 | 73.89% | 99.78% | 52.84% | 100.00% |