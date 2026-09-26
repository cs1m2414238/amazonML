# Retrieval Benchmark Comparison Report

> [!IMPORTANT]
> **Methodological safeguard**: Candidate recall measures search space coverage and blocking efficiency. It is **NOT** a measurement of final matching accuracy or macro-F0.5. True macro-F0.5 can only be evaluated once a matching model (e.g. LightGBM) produces final predicted match sets.

## Evaluation Metadata

- **Validation Sample**: validation_smoke_500
- **Sample Size**: 20 S1 entities
- **Sample Entities SHA-256**: 456257146f58e284a799374c9b2ba1fdab5ea7380f6031683ba369b724329f3b
- **Platform**: Windows-11-10.0.26200-SP0
- **Python**: 3.13.14
- **CPU**: Intel64 Family 6 Model 142 Stepping 10, GenuineIntel (8 cores)
- **Index Path**: H:\Amazon_ML_Work\output\step5\indices.db

## Recall vs. Candidate Count vs. Speed Comparison

| Budget | Metric | optimized |
| ---: | --- | ---: |
| B=16 | Pair Recall | 97.14% |
| B=16 | Complete Match Recall | 89.47% |
| B=16 | Source 2 Recall | 97.06% |
| B=16 | Source 3 Recall | 97.22% |
| B=16 | Average Candidates | 16.0 |
| B=16 | P95 Candidates | 16 |
| B=16 | Reduction Ratio | 0.999998 |
| B=32 | Pair Recall | 100.00% |
| B=32 | Complete Match Recall | 100.00% |
| B=32 | Source 2 Recall | 100.00% |
| B=32 | Source 3 Recall | 100.00% |
| B=32 | Average Candidates | 32.0 |
| B=32 | P95 Candidates | 32 |
| B=32 | Reduction Ratio | 0.999997 |
| B=64 | Pair Recall | 100.00% |
| B=64 | Complete Match Recall | 100.00% |
| B=64 | Source 2 Recall | 100.00% |
| B=64 | Source 3 Recall | 100.00% |
| B=64 | Average Candidates | 64.0 |
| B=64 | P95 Candidates | 64 |
| B=64 | Reduction Ratio | 0.999994 |
| B=128 | Pair Recall | 100.00% |
| B=128 | Complete Match Recall | 100.00% |
| B=128 | Source 2 Recall | 100.00% |
| B=128 | Source 3 Recall | 100.00% |
| B=128 | Average Candidates | 128.0 |
| B=128 | P95 Candidates | 128 |
| B=128 | Reduction Ratio | 0.999988 |
| B=256 | Pair Recall | 100.00% |
| B=256 | Complete Match Recall | 100.00% |
| B=256 | Source 2 Recall | 100.00% |
| B=256 | Source 3 Recall | 100.00% |
| B=256 | Average Candidates | 256.0 |
| B=256 | P95 Candidates | 256 |
| B=256 | Reduction Ratio | 0.999975 |

## Runtime Efficiency and Resource Utilization

| Retriever | Total Runtime (s) | QPS | Median Latency (ms) | P95 Latency (ms) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| optimized | 16.36 | 1.22 | 719.9 | 1962.0 | 579.4 |

## Diagnostic Strata Breakdown (optimized @ B=64)

> [!NOTE]
> Diagnostic strata are overlapping categories designed to test difficult subsets. These metrics are reported separately from the population-representative results.

| Stratum | Entities | True Pairs | Pair Recall | Complete Match Recall | Avg Candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| both_sources | 13 | 59 | 100.00% | 100.00% | 64.0 |
| country_india | 5 | 15 | 100.00% | 100.00% | 64.0 |
| country_us | 15 | 55 | 100.00% | 100.00% | 64.0 |
| multi_2_4 | 12 | 35 | 100.00% | 100.00% | 64.0 |
| multi_5_plus | 6 | 34 | 100.00% | 100.00% | 64.0 |
| one_match | 1 | 1 | 100.00% | 100.00% | 64.0 |
| s2_only | 2 | 3 | 100.00% | 100.00% | 64.0 |
| s3_only | 4 | 8 | 100.00% | 100.00% | 64.0 |
| singleton | 1 | 0 | 100.00% | 100.00% | 64.0 |