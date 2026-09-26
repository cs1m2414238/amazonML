import pandas as pd
import sqlite3
import logging
from collections import defaultdict
import numpy as np
from pathlib import Path

from business_entity_resolution.src.retrieval.candidate_generator import CandidateGenerator
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    config = load_config()
    data_root = Path(config["data"]["root"])
    output_root = Path(config["output"]["directory"])
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    s1_path = data_root / "dataset/train/train_source1.tsv"
    db_path = output_root / "step5/indices.db"
    
    logging.info("1. Parsing Ground Truth for strata definition...")
    gt_map = defaultdict(set)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype="string")
    for row in df_gt.itertuples(index=False):
        if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            targets = str(row.matched_entity_ids).split(",")
            for t in targets:
                gt_map[row.source1_entity_id].add(t.strip())
                
    max_match_len = max([len(v) for v in gt_map.values()] or [0])
    logging.info(f"Max matches per S1: {max_match_len}")
    
    # 2. Collect Stratified Sample from the Full S1 dataset
    logging.info("2. Streaming full S1 dataset to build stress groups...")
    
    strata = {
        "1_match": [],
        "2_to_4_match": [],
        "5_plus_match": [],
        "max_match": [],
        "missing_address": [],
        "short_common_name": []
    }
    
    target_samples_per_stratum = 30
    
    for chunk in read_tsv_chunks(s1_path, "source1"):
        for row in chunk.itertuples(index=False):
            s1_id = row.entity_id
            name = str(row.business_name) if pd.notna(row.business_name) else ""
            addr = str(row.business_address) if pd.notna(row.business_address) else ""
            country = str(row.country) if pd.notna(row.country) else ""
            
            gt_set = gt_map.get(s1_id, set())
            gt_len = len(gt_set)
            
            if gt_len == 0:
                continue
                
            record = (s1_id, name, addr, country)
            
            # Categorize
            if gt_len == 1 and len(strata["1_match"]) < target_samples_per_stratum:
                strata["1_match"].append(record)
            elif 2 <= gt_len <= 4 and len(strata["2_to_4_match"]) < target_samples_per_stratum:
                strata["2_to_4_match"].append(record)
            elif gt_len >= 5 and gt_len < max_match_len and len(strata["5_plus_match"]) < target_samples_per_stratum:
                strata["5_plus_match"].append(record)
            elif gt_len == max_match_len and len(strata["max_match"]) < target_samples_per_stratum:
                strata["max_match"].append(record)
                
            if not addr.strip() and len(strata["missing_address"]) < target_samples_per_stratum:
                strata["missing_address"].append(record)
                
            if len(name.split()) == 1 and len(strata["short_common_name"]) < target_samples_per_stratum:
                strata["short_common_name"].append(record)
                
        # Stop if all strata are full
        if all(len(lst) >= target_samples_per_stratum for k, lst in strata.items() if k != "max_match"):
            # max_match might have < 200 total in the whole dataset, so we just collect what we can.
            break

    # Combine unique stress records
    stress_records_dict = {}
    for lst in strata.values():
        for r in lst:
            stress_records_dict[r[0]] = r
            
    stress_records = list(stress_records_dict.values())
    logging.info(f"Total stress records selected: {len(stress_records)}")
    
    # 3. Evaluate B=256, 512, 1024, 2048
    generator = CandidateGenerator(db_path, high_df_threshold=50000)
    
    budgets = [256, 512, 1024, 2048]
    max_b = max(budgets)
    deep_b = 10000 # For diagnosing misses
    
    master_results = {}
    logging.info(f"3. Running retrieval at Deep Budget B={deep_b} for stress records...")
    
    for idx, (s1_id, name, addr, country) in enumerate(stress_records):
        if idx > 0 and idx % 100 == 0:
            logging.info(f"Processed {idx}/{len(stress_records)} stress queries...")
        # Deep query to allow slicing later
        master_results[s1_id] = generator.get_candidates(name, addr, country, budget=deep_b)

    logging.info("4. Evaluating budgets and causes...")
    
    for b in budgets:
        logging.info(f"\n--- Results for B={b} ---")
        found_pairs = 0
        total_pairs = 0
        s1_full_matches = 0
        
        # Group tracking
        group_metrics = {k: {"found": 0, "total": 0} for k in strata.keys()}
        
        for s1_id, name, addr, country in stress_records:
            gt_set = gt_map.get(s1_id, set())
            cands = set(master_results[s1_id][:b])
            
            total_pairs += len(gt_set)
            found = len(gt_set & cands)
            found_pairs += found
            
            if found == len(gt_set):
                s1_full_matches += 1
                
            # Update group metrics
            for k, lst in strata.items():
                if any(r[0] == s1_id for r in lst):
                    group_metrics[k]["total"] += len(gt_set)
                    group_metrics[k]["found"] += found

        logging.info(f"Overall Pair Recall: {(found_pairs/total_pairs)*100:.2f}%")
        logging.info(f"Overall S1 Full-Match Recall: {(s1_full_matches/len(stress_records))*100:.2f}%")
        for k, v in group_metrics.items():
            if v["total"] > 0:
                logging.info(f"  {k} Recall: {(v['found']/v['total'])*100:.2f}%")
                
    # 5. Diagnostic Analysis of what happens at B=2048 vs B=10000
    logging.info("\n--- Diagnostic Analysis of Misses at B=2048 ---")
    missed_at_2048 = 0
    rescued_by_10000 = 0
    block_failure = 0
    
    for s1_id, name, addr, country in stress_records:
        gt_set = gt_map.get(s1_id, set())
        cands_2048 = set(master_results[s1_id][:2048])
        cands_10000 = set(master_results[s1_id])
        
        misses = gt_set - cands_2048
        missed_at_2048 += len(misses)
        
        for miss in misses:
            if miss in cands_10000:
                rescued_by_10000 += 1
            else:
                block_failure += 1
                
    logging.info(f"Total True Pairs in Stress Set: {sum([len(gt_map[r[0]]) for r in stress_records])}")
    logging.info(f"Missed at B=2048: {missed_at_2048}")
    if missed_at_2048 > 0:
        logging.info(f" - Lost due to tight budget (found in 10000): {rescued_by_10000} ({(rescued_by_10000/missed_at_2048)*100:.2f}%)")
        logging.info(f" - Lost due to block failure / extreme ranking drop: {block_failure} ({(block_failure/missed_at_2048)*100:.2f}%)")

if __name__ == "__main__":
    main()
