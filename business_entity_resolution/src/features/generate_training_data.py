"""Generate leak-free training pairs with retrieval-based hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.features.pair_features import PairFeatureGenerator
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.retrieval.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def load_validation_exclusion_set(validation_ids_path: str | Path) -> set[str]:
    """Load Agent 2's validation S1 IDs to guarantee 100% leak-free training."""
    path = Path(validation_ids_path)
    if not path.exists():
        raise FileNotFoundError(f"Validation IDs file not found at: {path}")

    logging.info(f"Loading validation exclusion set from {path}...")
    val_df = pd.read_csv(path, sep="\t", dtype="string")
    val_ids = set(val_df["entity_id"].dropna().str.strip())
    logging.info(f"Loaded {len(val_ids)} validation S1 entities for strict exclusion.")
    return val_ids


def load_ground_truth(gt_path: str | Path) -> dict[str, set[str]]:
    """Load ground truth mapping from S1 ID to set of matching target IDs."""
    gt_path = Path(gt_path)
    logging.info(f"Loading ground truth from {gt_path}...")
    gt_map: dict[str, set[str]] = defaultdict(set)
    df = pd.read_csv(gt_path, sep="\t", dtype="string", keep_default_na=False)

    for row in df.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        matched = str(row.matched_entity_ids).strip() if pd.notna(row.matched_entity_ids) else ""
        if matched:
            targets = {t.strip() for t in matched.split(",") if t.strip()}
            gt_map[s1_id] = targets
    logging.info(f"Loaded ground truth for {len(gt_map)} S1 entities.")
    return gt_map


def hydrate_target_records(
    target_ids: set[str],
    s2_path: str | Path,
    s3_path: str | Path,
    cache_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    """Stream Source 2 and Source 3 to hydrate candidate entity records."""
    target_records: dict[str, dict[str, str]] = {}

    # Check cache
    if cache_path and Path(cache_path).exists():
        logging.info(f"Loading cached target records from {cache_path}...")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Check if all needed target_ids are cached
            if target_ids.issubset(cached.keys()):
                logging.info(f"All {len(target_ids)} target records loaded from cache.")
                return {tid: cached[tid] for tid in target_ids}
        except Exception as e:
            logging.warning(f"Failed to read cache {cache_path}: {e}")

    s2_needed = {tid for tid in target_ids if tid.startswith("S2-")}
    s3_needed = {tid for tid in target_ids if tid.startswith("S3-")}

    logging.info(f"Hydrating {len(s2_needed)} S2 records from {s2_path}...")
    for chunk in read_tsv_chunks(s2_path, "source2", chunk_size=200_000):
        for row in chunk.itertuples(index=False):
            eid = str(row.entity_id).strip()
            if eid in s2_needed:
                target_records[eid] = {
                    "id": eid,
                    "name": str(row.business_name).strip() if pd.notna(row.business_name) else "",
                    "addr": str(row.business_address).strip() if pd.notna(row.business_address) else "",
                    "country": str(row.country).strip() if pd.notna(row.country) else "",
                }
                if len(target_records) == len(s2_needed):
                    break
        if len(target_records) == len(s2_needed):
            break

    logging.info(f"Hydrating {len(s3_needed)} S3 records from {s3_path}...")
    s3_found = 0
    for chunk in read_tsv_chunks(s3_path, "source3", chunk_size=200_000):
        for row in chunk.itertuples(index=False):
            eid = str(row.entity_id).strip()
            if eid in s3_needed:
                target_records[eid] = {
                    "id": eid,
                    "name": str(row.business_name).strip() if pd.notna(row.business_name) else "",
                    "addr": str(row.business_address).strip() if pd.notna(row.business_address) else "",
                    "country": str(row.country).strip() if pd.notna(row.country) else "",
                }
                s3_found += 1
                if s3_found == len(s3_needed):
                    break
        if s3_found == len(s3_needed):
            break

    logging.info(f"Hydrated total {len(target_records)} target records.")

    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(target_records, f)
            logging.info(f"Saved hydrated records cache to {cache_path}")
        except Exception as e:
            logging.warning(f"Could not save cache to {cache_path}: {e}")

    return target_records


def generate_training_dataset(
    s1_sample_size: int = 1000,
    retrieval_budget: int = 64,
    max_hard_negatives: int = 8,
    seed: int = 42,
    output_dir: str | Path = "output/models",
    db_path: str | Path | None = None,
    validation_ids_path: str | Path = "output/validation/validation_quality_10000_ids.tsv",
) -> Path:
    """Generate leak-free training dataset containing positive pairs and hard negatives."""
    config = load_config()
    data_root = Path(config["data"]["root"])
    s1_path = data_root / "dataset/train/train_source1.tsv"
    s2_path = data_root / "dataset/train/train_source2.tsv"
    s3_path = data_root / "dataset/train/train_source3.tsv"
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"

    if db_path is None:
        db_path = Path(config["output"]["directory"]) / "step5/indices.db"
        if not db_path.exists():
            db_path = Path(r"H:/Amazon_ML_Work/output/step5/indices.db")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    val_exclusion_set = load_validation_exclusion_set(validation_ids_path)
    gt_map = load_ground_truth(gt_path)

    # Sample S1 records strictly excluding validation set
    logging.info(f"Sampling {s1_sample_size} training S1 entities (seed={seed})...")
    rng = np.random.default_rng(seed)

    s1_pool: list[dict[str, Any]] = []
    # Collect candidate training entities from S1
    for chunk in read_tsv_chunks(s1_path, "source1", chunk_size=100_000):
        for row in chunk.itertuples(index=False):
            s1_id = str(row.entity_id).strip()
            if s1_id in val_exclusion_set:
                continue  # STRICT EXCLUSION

            s1_pool.append({
                "id": s1_id,
                "name": str(row.business_name).strip() if pd.notna(row.business_name) else "",
                "addr": str(row.business_address).strip() if pd.notna(row.business_address) else "",
                "country": str(row.country).strip() if pd.notna(row.country) else "",
            })
            if len(s1_pool) >= s1_sample_size * 2:
                break
        if len(s1_pool) >= s1_sample_size * 2:
            break

    # Randomly shuffle and take exact sample size
    sampled_indices = rng.choice(len(s1_pool), size=min(s1_sample_size, len(s1_pool)), replace=False)
    sampled_s1 = [s1_pool[i] for i in sampled_indices]

    # Save sampled training IDs
    train_ids_path = output_dir / "training_s1_ids.tsv"
    with open(train_ids_path, "w", encoding="utf-8") as f:
        f.write("entity_id\n")
        for s in sampled_s1:
            f.write(f"{s['id']}\n")
    logging.info(f"Saved {len(sampled_s1)} training S1 IDs to {train_ids_path}")

    # Initialize retriever
    high_df = config.get("retrieval", {}).get("high_df_threshold", 50000)
    retriever = OptimizedCandidateGenerator(
        db_path=db_path,
        high_df_threshold=high_df,
        posting_cache_mb=1024,
        sqlite_cache_mb=512,
        mmap_mb=8192,
    )

    logging.info(f"Retrieving candidate pools (B={retrieval_budget}) for training queries...")
    s1_candidate_pairs: list[dict[str, Any]] = []
    all_target_ids: set[str] = set()

    for idx, s1 in enumerate(sampled_s1):
        s1_id = s1["id"]
        true_targets = gt_map.get(s1_id, set())

        cands = retriever.get_candidates(
            s1["name"], s1["addr"], s1["country"], budget=retrieval_budget
        )

        cand_set = set(cands)
        # Positives in retrieval
        for rank_idx, cand_id in enumerate(cands, start=1):
            if cand_id in true_targets:
                s1_candidate_pairs.append({
                    "s1_id": s1_id,
                    "target_id": cand_id,
                    "label": 1,
                    "retrieval_rank": rank_idx,
                })
                all_target_ids.add(cand_id)

        # Stratified negative sampling across candidate pool (top, middle, tail)
        non_matching_cands = [
            (rank_idx, cand_id)
            for rank_idx, cand_id in enumerate(cands, start=1)
            if cand_id not in true_targets
        ]

        # Top 4 hardest negatives
        top_negs = non_matching_cands[:4]
        # Middle / tail negatives
        rem_negs = non_matching_cands[4:]
        if len(rem_negs) > 8:
            sub_indices = rng.choice(len(rem_negs), size=8, replace=False)
            sampled_rem = [rem_negs[i] for i in sub_indices]
        else:
            sampled_rem = rem_negs

        for rank_idx, cand_id in (top_negs + sampled_rem):
            s1_candidate_pairs.append({
                "s1_id": s1_id,
                "target_id": cand_id,
                "label": 0,
                "retrieval_rank": rank_idx,
            })
            all_target_ids.add(cand_id)

        if (idx + 1) % 200 == 0 or (idx + 1) == len(sampled_s1):
            logging.info(f"Retrieved for {idx + 1}/{len(sampled_s1)} training S1 queries...")

    retriever.close()

    logging.info(f"Generated {len(s1_candidate_pairs)} candidate pairs. Target entities to hydrate: {len(all_target_ids)}")

    # Hydrate target records
    cache_path = output_dir / "cached_train_target_records.json"
    target_records = hydrate_target_records(
        target_ids=all_target_ids,
        s2_path=s2_path,
        s3_path=s3_path,
        cache_path=cache_path,
    )

    # Compute features
    logging.info("Computing pairwise features...")
    feat_gen = PairFeatureGenerator()
    s1_dict_map = {s["id"]: s for s in sampled_s1}

    dataset_rows: list[dict[str, Any]] = []
    for pair in s1_candidate_pairs:
        s1_id = pair["s1_id"]
        t_id = pair["target_id"]
        label = pair["label"]
        rank = pair["retrieval_rank"]

        s1_info = s1_dict_map.get(s1_id, {})
        t_info = target_records.get(t_id, {"id": t_id, "name": "", "addr": "", "country": ""})

        t_src = "source3" if t_id.startswith("S3-") else "source2"
        feats = feat_gen.compute_features(
            s1_dict=s1_info,
            s2_dict=t_info,
            retrieval_rank=rank,
            target_source=t_src,
        )

        feats["s1_id"] = s1_id
        feats["target_id"] = t_id
        feats["label"] = label
        dataset_rows.append(feats)

    df_pairs = pd.DataFrame(dataset_rows)
    pos_count = (df_pairs["label"] == 1).sum()
    neg_count = (df_pairs["label"] == 0).sum()
    logging.info(f"Final training dataset shape: {df_pairs.shape} (Positives: {pos_count}, Negatives: {neg_count})")

    out_csv = output_dir / "train_pairs.csv"
    df_pairs.to_csv(out_csv, index=False)
    logging.info(f"Saved training pairs CSV to {out_csv}")

    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Generate leak-free training dataset for business entity resolution")
    parser.add_argument("--s1-sample-size", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="output/models")
    args = parser.parse_args()

    generate_training_dataset(
        s1_sample_size=args.s1_sample_size,
        retrieval_budget=args.budget,
        max_hard_negatives=args.max_negatives,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
