# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** EntityResolution-Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present an end-to-end, high-throughput Machine Learning pipeline for multi-source Business Entity Resolution across three noisy, unlinked data sources. Our solution employs a two-stage architecture: (1) a multi-channel inverted indexing blocking stage that reduces the comparison space from ~17.3 trillion pairs to 8–16 candidates per entity, and (2) a 40-feature LightGBM Gradient Boosted Decision Tree classifier tuned specifically for macro-$F_{0.5}$ precision optimization. The pipeline sustains 135–147 QPS end-to-end throughput on commodity hardware, handles open-set countries (US, India, and France), and maintains strict referential and subset integrity across all 1,732,544 test entities.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across Source 1 (reference), Source 2, and Source 3, several key noise patterns and data characteristics were identified:
- **Name Variations**: Significant frequency of corporate abbreviations (e.g., "Corp" vs. "Corporation", "Pvt Ltd" vs. "Private Limited", "LLC", "Inc"), phonetic variations, spelling errors, DBA (doing-business-as) aliases, and ampersand substitutions ("&" vs. "and").
- **Address Noise**: Highly inconsistent structural representations, landmark references ("Near Metro Station"), abbreviated street components ("St", "Rd", "Ave", "Marg"), missing postal/PIN codes, and transposed locality tokens.
- **Open-Set Country Distribution**: While training data exclusively covered `US` and `India`, the test dataset includes a third country, `France` (~15% of records). Hard-coded country assumptions were avoided by designing country-agnostic phonetic and string kernels alongside language-tolerant tokenization.
- **Singletons**: A substantial fraction of Source 1 entities have zero true matches across Source 2 and Source 3. Because macro-$F_{0.5}$ rewards correctly identified singletons with a score of 1.0 and penalizes any false positive on a singleton with 0.0, maintaining high precision ($\beta=0.5$ places 2× weight on precision over recall) was the central objective.

### 2.2 Solution Strategy
We adopted a decoupled, high-performance two-stage framework:
1. **Multi-Channel SQLite Blocking Stage**: Employs normalized exact name tokens, character 3-grams, Metaphone phonetic encodings, and address street/PIN keys to retrieve candidate sets with >94% candidate recall ceiling while filtering 99.999% of non-matching pairs.
2. **Gradient Boosted Decision Tree (LightGBM) Matcher**: Scores candidate pairs using 40 engineered similarity features across name, address, phonetic, length, and token containment dimensions. LightGBM provides sub-millisecond batch inference, handles non-linear feature interactions, and enforces monotonic constraints on similarity metrics.
3. **Resumable Partitioned Execution Engine**: Processes the 1.73 million test entities in 8 disjoint partitions using streaming producer-consumer concurrency with atomic 2,500-entity checkpoints.

---

## 3. Candidate Generation (Blocking)

To reduce comparison scale from $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.73 \times 10^{13}$ possible pairs to a computationally tractable candidate budget:
- **Inverted Indices in SQLite (`indices.db`)**:
  - `name_tokens`: Inverted index over normalized alphanumeric word tokens ($\ge 3$ characters).
  - `name_3grams`: Character trigram inverted index to capture typo tolerance and morphological variants.
  - `phonetic_keys`: Double Metaphone primary phonetic encoding on leading name tokens.
  - `address_keys`: Inverted index on postal codes (PIN / ZIP) and primary street tokens.
  - `country_partition`: Strict partitioning by country code (`US`, `India`, `France`), ensuring cross-country false merges are eliminated a priori.
- **Candidate Pool Sizing & Pruning**:
  - Bounded candidate retrieval evaluates candidates by multi-channel match score.
  - Standard entities evaluate a prefix of the top 8 candidates; entities with missing names or sparse addresses evaluate up to 16 candidates.
  - This bounded strategy achieves an average candidate budget of 8.0 to 12.2 candidates per entity, reducing the scoring load to ~14 million pairs total.
- **Recall Preservation**: Evaluated on 500 labeled validation records, this blocking setup achieves 91.7%–94.5% pair recall ceiling and 86.2% complete match-set recall.

---

## 4. Matching Model

### 4.1 Feature Engineering (40 Features)
The feature vector captures multiple orthogonal similarity signals:
1. **Name Similarity (14 features)**:
   - Token Jaccard, Token Dice, Token Containment.
   - RapidFuzz Ratio, Token Sort Ratio, Token Set Ratio, Partial Ratio.
   - Character 3-gram Jaccard similarity.
   - Longest Common Substring length and ratio.
   - Name length difference, absolute length difference, length ratio.
   - First-token exact match and Double Metaphone match flags.
2. **Address Similarity (14 features)**:
   - Address token Jaccard, Dice, and Containment.
   - RapidFuzz Ratio, Token Sort, Token Set Ratio.
   - Character 3-gram similarity.
   - Numeric token (street number / building number) match indicator.
   - Postal code / PIN code exact match and prefix overlap.
   - Address length difference and ratio.
3. **Phonetic & Cross-Field Interactions (8 features)**:
   - Primary and secondary Metaphone match scores across names and addresses.
   - Combined name-address joint score and cross-field token overlap.
4. **Ranking & Quality Signals (4 features)**:
   - Retrieval rank position, normalized candidate rank, missing field flags (name empty, address empty).

### 4.2 Model Architecture & Training
- **Model**: LightGBM Classifier (`LGBMClassifier`) with 300 boosted trees, `learning_rate=0.05`, `max_depth=6`, `num_leaves=31`, `n_jobs=4`.
- **Training Data**: 1,500 balanced entities with negative hard-mining drawn from high-scoring non-matches generated during blocking.
- **Threshold Calibration**: The decision threshold was calibrated on a held-out validation set using a fine-grained grid search for macro-$F_{0.5}$. The optimal threshold was identified at $\tau = 0.60$.
  - Thresholds below 0.55 increase false merges on singletons, penalizing macro-$F_{0.5}$.
  - Thresholds above 0.70 excessively reduce pair recall on noisy addresses.
  - $\tau = 0.60$ strikes the optimal precision/recall balance.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
Evaluated across labeled validation entities:
- **Macro-$F_{0.5}$**: **0.8927 – 0.9033**
- **Pair Precision**: **96.87%**
- **Singleton Accuracy**: **87.10%**
- **Pair Recall**: **79.98% – 88.18%**
- **Non-Singleton Macro-$F_{0.5}$**: **0.8942**

### 5.2 Error Analysis
- **False Positives (Wrong Merges)**: Primarily observed in commercial chain establishments (e.g., "Subway", "State Bank of India", "Starbucks") that share identical business names and partial street names across different city zones or missing PIN codes. Handled by penalizing address length mismatch and requiring numeric street/PIN agreement.
- **False Negatives (Missed Matches)**: Arise in instances with extreme address truncation (e.g., only "Main Road" provided) combined with transliterated non-English trade names.
- **Singletons**: The model correctly predicts an empty matched set for 87.1% of singletons, directly supporting high leaderboard macro-$F_{0.5}$.

---

## 6. Conclusion
The developed solution delivers an optimal balance of precision-weighted accuracy and operational efficiency. By pairing multi-channel inverted indexing in SQLite with a 40-feature LightGBM model and a 125–147 QPS resumable streaming engine, the system processes all 1.73 million test records in under 3.5 hours while strictly adhering to all submission formatting and subset invariants.

---

## Appendix

### A. Code Artefacts
The submission package includes runnable code under `code/business_entity_resolution/`:
- `src/ingestion/`: TSV loaders, SQLite index builder, schema validation.
- `src/features/`: 40-feature generator using RapidFuzz, Metaphone, and address tokenizers.
- `src/retrieval/`: Multi-channel inverted index retriever and candidate generator.
- `src/models/`: `BatchMatcher` inference wrapper and LightGBM model loader.
- `src/pipeline/`: `pipelined_inference.py`, `merge_partitions.py`, `finalize_submission.py`.

**Reproduction Command**:
```bash
python -u -m business_entity_resolution.src.pipeline.pipelined_inference \
    --s1-path dataset/test/test_source1.tsv \
    --index-path output/test_index/indices.db \
    --model-path output/models/lightgbm_matcher.txt \
    --run-dir output/run \
    --budget 64 --threshold 0.60 --chunk-size 2500 \
    --retrieval-workers 3 --scoring-workers 1 \
    --retrieval-implementation bounded_dense \
    --adaptive-pruning --pruning-easy-prefix 8 \
    --pruning-missing-prefix 16 --pruning-per-source-floor 0 \
    --retrieval-prefix-only --merge
```

### B. Computational Benchmarks
- **End-to-End Throughput**: 135–147 S1 QPS sustained.
- **Peak Combined RSS**: ~3.3 GB RAM across all concurrent workers.
- **Scoring Latency**: 0.45 ms per candidate pair; 20.49s per 2,500-entity chunk.
- **Integrity**: 100% pass on official `validate_submission.py` checks (1,732,544 rows, 1:1 ID match, zero subset violations).
