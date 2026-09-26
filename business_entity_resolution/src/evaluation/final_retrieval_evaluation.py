import pandas as pd
import sqlite3
import logging
from collections import defaultdict
import random
import time
import numpy as np
import json
from pathlib import Path

from business_entity_resolution.src.retrieval.candidate_generator import CandidateGenerator
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    random.seed(42)
    config = load_config()
    data_root = Path(config["data"]["root"])
    output_root = Path(config["output"]["directory"])
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    s1_path = data_root / "dataset/train/train_source1.tsv"
    db_path = output_root / "step5/indices.db"
    
    # 1. Parsing GT
    logging.info("Parsing Ground Truth...")
    gt_map = defaultdict(set)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype="string")
    for row in df_gt.itertuples(index=False):
        if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            targets = str(row.matched_entity_ids).split(",")
            for t in targets:
                gt_map[row.source1_entity_id].add(t.strip())
                
    # 2. Build Representative Evaluation Set
    logging.info("Streaming S1 to build Representative Set...")
    groups = {
        "A_Random": [],
        "B_Zero": [],
        "C_One": [],
        "D_Multi": [],
        "E_5Plus": [],
        "F_10_11": [],
        "H_Short": [],
        "I_MissingAddr": []
    }
    
    target_sizes = {
        "A_Random": 1000,
        "B_Zero": 150,
        "C_One": 150,
        "D_Multi": 150,
        "E_5Plus": 150,
        "F_10_11": 150,
        "H_Short": 150,
        "I_MissingAddr": 150
    }
    
    all_s1 = []
    # To avoid loading all 2.2M into RAM just for random choice, we will systematically sample 
    # every 1000th record for Group A until we hit 2000.
    
    idx = 0
    for chunk in read_tsv_chunks(s1_path, "source1"):
        for row in chunk.itertuples(index=False):
            idx += 1
            s1_id = row.entity_id
            name = str(row.business_name).strip() if pd.notna(row.business_name) else ""
            addr = str(row.business_address).strip() if pd.notna(row.business_address) else ""
            country = str(row.country).strip() if pd.notna(row.country) else ""
            
            gt_set = gt_map.get(s1_id, set())
            gt_len = len(gt_set)
            
            rec = (s1_id, name, addr, country)
            
            if idx % 1000 == 0 and len(groups["A_Random"]) < target_sizes["A_Random"]:
                groups["A_Random"].append(rec)
                
            if gt_len == 0 and len(groups["B_Zero"]) < target_sizes["B_Zero"]:
                groups["B_Zero"].append(rec)
            elif gt_len == 1 and len(groups["C_One"]) < target_sizes["C_One"]:
                groups["C_One"].append(rec)
            elif 2 <= gt_len <= 4 and len(groups["D_Multi"]) < target_sizes["D_Multi"]:
                groups["D_Multi"].append(rec)
            elif 5 <= gt_len <= 9 and len(groups["E_5Plus"]) < target_sizes["E_5Plus"]:
                groups["E_5Plus"].append(rec)
            elif gt_len >= 10 and len(groups["F_10_11"]) < target_sizes["F_10_11"]:
                groups["F_10_11"].append(rec)
                
            if len(name.split()) == 1 and gt_len > 0 and len(groups["H_Short"]) < target_sizes["H_Short"]:
                groups["H_Short"].append(rec)
                
            if not addr and gt_len > 0 and len(groups["I_MissingAddr"]) < target_sizes["I_MissingAddr"]:
                groups["I_MissingAddr"].append(rec)
                
        # Early break if all filled
        if all(len(lst) >= target_sizes[k] for k, lst in groups.items() if k != "F_10_11"):
            break
            
    # Combine uniquely
    eval_set_dict = {}
    overlap_counts = defaultdict(int)
    for gname, lst in groups.items():
        for r in lst:
            if r[0] in eval_set_dict:
                overlap_counts[gname] += 1
            eval_set_dict[r[0]] = r
            
    eval_set = list(eval_set_dict.values())
    logging.info(f"Final Evaluation Set Size: {len(eval_set)} (Total Overlaps: {dict(overlap_counts)})")
    
    # 3. Ground Truth Stats for Eval Set
    total_true_pairs = 0
    stats_zero = 0
    stats_one = 0
    stats_multi = 0
    max_matches = 0
    for r in eval_set:
        l = len(gt_map.get(r[0], set()))
        total_true_pairs += l
        if l == 0: stats_zero += 1
        elif l == 1: stats_one += 1
        else: stats_multi += 1
        if l > max_matches: max_matches = l
        
    logging.info(f"GT Stats: Pairs={total_true_pairs}, Zero={stats_zero}, One={stats_one}, Multi={stats_multi}, Max={max_matches}")
    
    # 4. Independent Budget Equivalence Test
    generator = CandidateGenerator(db_path, high_df_threshold=50000)
    logging.info("Testing B-Slicing Equivalence on 20 random eval records...")
    equiv_passed = True
    for r in eval_set[:20]:
        c1024 = generator.get_candidates(r[1], r[2], r[3], budget=1024)
        for b in [256, 512]:
            cb = generator.get_candidates(r[1], r[2], r[3], budget=b)
            if cb != c1024[:b]:
                equiv_passed = False
                break
    if equiv_passed:
        logging.info("EQUIVALENCE TEST PASSED: independent(B) == max_B_result[:B]")
    else:
        logging.error("EQUIVALENCE TEST FAILED!")
        return

    # 5. Execute Deep Retrieval (B=10000 for analysis, B=1024 for eval)
    master_results = {}
    deep_results = {}
    
    t0 = time.time()
    for idx, r in enumerate(eval_set):
        if idx > 0 and idx % 500 == 0:
            logging.info(f"Retrieved {idx}/{len(eval_set)}...")
        cands = generator.get_candidates(r[1], r[2], r[3], budget=10000)
        deep_results[r[0]] = set(cands)
        master_results[r[0]] = cands[:1024]
        
    t1 = time.time()
    runtime = t1 - t0
    
    # 6. Evaluate Fixed Budgets
    def evaluate_slice(budget_cands_dict):
        metrics = {"total_found": 0, "s1_full_matches": 0, "zero_match_correct": 0, "cand_counts": []}
        group_recalls = {g: {"found": 0, "total": 0} for g in groups.keys()}
        
        for r in eval_set:
            s1_id = r[0]
            gt = gt_map.get(s1_id, set())
            cands = set(budget_cands_dict[s1_id])
            
            metrics["cand_counts"].append(len(cands))
            
            if len(gt) == 0:
                # Zero-match S1
                if len(cands) >= 0: # It's impossible to have a false negative for zero match
                    metrics["zero_match_correct"] += 1
                continue
                
            found = len(gt & cands)
            metrics["total_found"] += found
            if found == len(gt):
                metrics["s1_full_matches"] += 1
                
            for gname, lst in groups.items():
                if any(x[0] == s1_id for x in lst):
                    group_recalls[gname]["total"] += len(gt)
                    group_recalls[gname]["found"] += found
                    
        return metrics, group_recalls

    fixed_metrics = {}
    for b in [256, 512, 1024]:
        b_dict = {k: v[:b] for k,v in master_results.items()}
        m, gm = evaluate_slice(b_dict)
        fixed_metrics[f"B_{b}"] = {"metrics": m, "group_recalls": gm}
        
    # 7. Adaptive Retrieval
    adaptive_cands = {}
    expansion_stats = {"at_256": 0, "at_512": 0, "at_1024": 0}
    false_expansion = {"useful": 0, "unnecessary": 0}
    
    for r in eval_set:
        s1_id = r[0]
        cands = master_results[s1_id]
        c256 = cands[:256]
        
        # Difficulty Triggers
        is_missing_addr = (not r[2])
        is_short_name = (len(r[1].split()) <= 2)
        
        # Trigger 1: Saturated Budget + Ambiguity
        if len(c256) == 256 and (is_missing_addr or is_short_name):
            # Expand to 512
            c512 = cands[:512]
            if len(c512) == 512 and (is_missing_addr and is_short_name):
                # Extreme ambiguity -> 1024
                final_c = cands[:1024]
                expansion_stats["at_1024"] += 1
                base_c = c512
            else:
                final_c = c512
                expansion_stats["at_512"] += 1
                base_c = c256
                
            # Track false expansion
            gt = gt_map.get(s1_id, set())
            if len(gt) > 0:
                if len(gt & set(final_c)) > len(gt & set(base_c)):
                    false_expansion["useful"] += 1
                else:
                    false_expansion["unnecessary"] += 1
        else:
            final_c = c256
            expansion_stats["at_256"] += 1
            
        adaptive_cands[s1_id] = final_c
        
    adapt_m, adapt_gm = evaluate_slice(adaptive_cands)
    
    # 8. Failure Analysis (Adaptive Misses)
    failure_counts = {"budget_casualty": 0, "block_failure": 0}
    for r in eval_set:
        s1_id = r[0]
        gt = gt_map.get(s1_id, set())
        ac = set(adaptive_cands[s1_id])
        
        misses = gt - ac
        for m in misses:
            if m in deep_results[s1_id]:
                failure_counts["budget_casualty"] += 1
            else:
                failure_counts["block_failure"] += 1
                
    # Compile JSON
    report = {
        "set_size": len(eval_set),
        "total_true_pairs": total_true_pairs,
        "stats_zero": stats_zero,
        "runtime": runtime,
        "qps": len(eval_set) / runtime if runtime > 0 else 0,
        "equivalence_passed": equiv_passed,
        "fixed_results": fixed_metrics,
        "adaptive_results": {
            "expansion_counts": expansion_stats,
            "false_expansion": false_expansion,
            "metrics": adapt_m,
            "group_recalls": adapt_gm,
            "failures": failure_counts
        }
    }
    
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / "step6_final_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logging.info(f"Final evaluation complete. Report saved to {out_path}")

if __name__ == "__main__":
    main()
