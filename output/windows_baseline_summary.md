# Windows blocking baseline

## Environment

- CPU: Intel i5-8250U, 4 cores / 8 threads
- RAM: 20 GB DDR4
- Active storage: H: SSD
- Python: 3.13.14
- Dataset: `student_resource` on C:
- Generated artifacts: `H:\Amazon_ML_Work\output`

## Validation

- Full dataset validation completed successfully.
- Peak memory: 1,132.54 MB.
- All reported source counts, countries, and ground-truth consistency checks match
  the existing report.

## Combined training target index

- Records indexed: 10,320,219
- Build time: 2,965.15 seconds
- Throughput: 3,480.50 records/second
- Peak memory: 1,433.25 MB
- SQLite size: 3,311.42 MiB
- SQLite integrity check: `ok`
- Token and n-gram counts exactly match the M1 report.

The teammate's M1 build took 773.06 seconds. This Windows build was 3.84 times
slower using the unchanged single-threaded algorithm.

## Retrieval benchmark

The benchmark used the same first 2,000 S1 records and retrieved once at B=256,
then sliced the ranked lists for smaller budgets.

| Budget | Pair recall | Complete-match-set recall | Average candidates |
| ---: | ---: | ---: | ---: |
| 16 | 0.9148 | 0.7616 | 16 |
| 32 | 0.9283 | 0.7971 | 32 |
| 64 | 0.9400 | 0.8295 | 64 |
| 128 | 0.9483 | 0.8508 | 128 |
| 256 | 0.9571 | 0.8720 | 256 |

- Runtime: 1,798.80 seconds
- Throughput: 1.11 queries/second
- Peak memory: 1,920.90 MB
- Recall values exactly match the M1 report.
- M1 throughput: 2.82 queries/second; Windows baseline is 2.54 times slower.
- Straight-line full-test estimate: about 433.6 hours, or 18.1 days.

## Main finding

The current implementation is CPU-bound, single-threaded, and has severe query
tail latency caused by expensive posting-list and n-gram work. RAM and SSD capacity
are not limiting factors. The next implementation should bound high-frequency
channel work, retrieve S2 and S3 independently, batch posting lookup, cache decoded
postings, and record per-query latency percentiles.

## Optimized retrieval benchmark

The optimized implementation was measured on the identical first 2,000 S1 records
against the same index. It uses batched SQLite reads, a byte-bounded decoded-posting
LRU cache, zero-copy NumPy posting views, vectorized high-frequency intersections,
vectorized country filtering, and bounded top-B selection. The original generator
remains available as the reference implementation.

- Ordered candidate lists matched the reference exactly on 100 real queries at
  B=256 before timing.
- All recall and candidate-count metrics match the baseline exactly at B=16, 32,
  64, 128, and 256.
- Runtime: 281.31 seconds, down from 1,798.80 seconds.
- Throughput: 7.11 queries/second, a 6.41x improvement over 1.11 QPS.
- Peak memory: 2,211.16 MB, compared with 1,920.90 MB for the baseline.
- Latency: p50 0.0977 s, p95 0.4080 s, p99 0.7090 s, maximum 2.1270 s.
- Decoded-posting cache used 436.54 MB of its 1,024 MB limit.
- Straight-line full-test estimate: about 67.7 hours, or 2.8 days, before
  multiprocessing.

The 5-10 QPS milestone is met without a recall regression. Remaining single-query
time is concentrated in high-DF scoring (40.5%), low-DF scoring (28.5%), top-B
ranking (16.8%), and country filtering (10.6%). Parallel worker testing should be
performed only after the stratified quality benchmark is fixed.
