import logging
import sqlite3
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time
import numpy as np

from business_entity_resolution.src.retrieval.candidate_generator import CandidateGenerator
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def verify_slicing(generator, s1_records):
    logging.info("Verifying B-slicing optimization on 50 records...")
    match_count = 0
    for idx, (entity_id, name, addr, country) in enumerate(s1_records[:50]):
        c256 = generator.get_candidates(name, addr, country, budget=256)
        for b in [16, 32, 64, 128]:
            cb = generator.get_candidates(name, addr, country, budget=b)
            if cb != c256[:b]:
                logging.error(f"Mismatch at B={b} for {entity_id}")
                return False
        match_count += 1
    logging.info(f"Successfully verified B-slicing on {match_count} records. Candidate sets are IDENTICAL.")
    return True

def main():
    config = load_config()
    data_root = Path(config["data"]["root"])
    output_root = Path(config["output"]["directory"])
    db_path = output_root / "step5/indices.db"
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    s1_path = data_root / "dataset/train/train_source1.tsv"
    
    generator = CandidateGenerator(db_path, high_df_threshold=50000)
    
    # Load GT
    gt_map = defaultdict(set)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype="string")
    for row in df_gt.itertuples(index=False):
        if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            targets = str(row.matched_entity_ids).split(",")
            for t in targets:
                gt_map[row.source1_entity_id].add(t.strip())
                
    # Load 2000 S1 records
    s1_records = []
    for chunk in read_tsv_chunks(s1_path, "source1", 2000):
        for row in chunk.itertuples(index=False):
            s1_records.append((row.entity_id, row.business_name, row.business_address, row.country))
            if len(s1_records) >= 2000:
                break
        if len(s1_records) >= 2000:
            break
            
    # Task 1: Verify Slicing
    verify_slicing(generator, s1_records)
    
    # Task 2 & 3: Run B=256 and Analyze Failures
    logging.info("Running B=256 across 2000 queries for failure analysis...")
    
    total_true_pairs = 0
    missed_pairs = []
    s1_miss_records = []
    
    # SQLite connection to check target countries
    conn = sqlite3.connect(db_path)
    
    for idx, (entity_id, name, addr, country) in enumerate(s1_records):
        if idx > 0 and idx % 200 == 0:
            logging.info(f"Processed {idx}/2000 queries...")
            
        gt_set = gt_map.get(entity_id, set())
        if not gt_set:
            continue
            
        total_true_pairs += len(gt_set)
        
        candidates = generator.get_candidates(name, addr, country, budget=256)
        cand_set = set(candidates)
        
        misses = gt_set - cand_set
        if misses:
            s1_miss_records.append({
                "s1_id": entity_id,
                "name": name,
                "addr": addr,
                "country": country,
                "gt_len": len(gt_set),
                "miss_count": len(misses)
            })
            for miss in misses:
                missed_pairs.append({
                    "s1_id": entity_id,
                    "target_id": miss,
                    "s1_country": str(country).lower().strip() if pd.notna(country) else "missing",
                    "s1_name": name,
                    "s1_addr": addr
                })

    logging.info(f"Total True Pairs: {total_true_pairs}")
    logging.info(f"Missed Pairs at B=256: {len(missed_pairs)}")
    
    # Categorize missed pairs
    categories = {
        "candidate_budget_truncation": 0,
        "country_filter_failure": 0,
        "block_failure_no_overlap": 0,
        "missing_fields": 0
    }
    
    logging.info("Deep querying missed pairs to categorize failures...")
    for idx, mp in enumerate(missed_pairs):
        if idx > 0 and idx % 50 == 0:
            logging.info(f"Analyzed {idx}/{len(missed_pairs)} misses...")
            
        # Check Country
        cursor = conn.execute("SELECT country FROM documents WHERE entity_id = ?", (mp["target_id"],))
        target_row = cursor.fetchone()
        target_country = target_row[0] if target_row else "unknown"
        target_country = "missing" if not target_country else target_country.lower().strip()
        
        s1_c = mp["s1_country"]
        if s1_c != "missing" and target_country != "missing" and s1_c != target_country:
            categories["country_filter_failure"] += 1
            continue
            
        # Check if missing fields in S1 caused it
        if pd.isna(mp["s1_name"]) or not str(mp["s1_name"]).strip():
            categories["missing_fields"] += 1
            continue
            
        # Check if it was truncated (Deep Query B=10000)
        deep_cands = generator.get_candidates(mp["s1_name"], mp["s1_addr"], mp["s1_country"], budget=10000)
        if mp["target_id"] in deep_cands:
            categories["candidate_budget_truncation"] += 1
        else:
            categories["block_failure_no_overlap"] += 1

    # Analyze S1 records that failed full-match recall
    s1_fail_analysis = {
        "total_failed_s1": len(s1_miss_records),
        "multi_match_records": sum(1 for r in s1_miss_records if r["gt_len"] > 1),
        "one_match_records": sum(1 for r in s1_miss_records if r["gt_len"] == 1),
        "missing_address_in_s1": sum(1 for r in s1_miss_records if pd.isna(r["addr"]) or not str(r["addr"]).strip()),
        "missing_name_in_s1": sum(1 for r in s1_miss_records if pd.isna(r["name"]) or not str(r["name"]).strip()),
    }
    
    logging.info("=== FAILURE ANALYSIS REPORT ===")
    logging.info(f"Missed Pairs: {len(missed_pairs)} / {total_true_pairs} ({(len(missed_pairs)/total_true_pairs)*100:.2f}%)")
    for cat, count in categories.items():
        pct = (count / len(missed_pairs) * 100) if missed_pairs else 0
        logging.info(f" - {cat}: {count} ({pct:.2f}%)")
        
    logging.info(f"S1 Records failing full-match: {len(s1_miss_records)}")
    for cat, count in s1_fail_analysis.items():
        if cat != "total_failed_s1":
            pct = (count / len(s1_miss_records) * 100) if s1_miss_records else 0
            logging.info(f" - {cat}: {count} ({pct:.2f}%)")
            
if __name__ == "__main__":
    main()
