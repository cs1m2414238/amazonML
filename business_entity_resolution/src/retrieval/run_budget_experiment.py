import time
import logging
import os
import platform
from pathlib import Path
import json
import numpy as np

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.retrieval.candidate_generator import CandidateGenerator
from business_entity_resolution.src.evaluation.recall_evaluator import RecallEvaluator
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    config = load_config()
        
    data_root = Path(config["data"]["root"])
    budgets = config.get("retrieval", {}).get("budgets", [16, 32, 64, 128, 256])
    high_df_threshold = config.get("retrieval", {}).get("high_df_threshold", 50000)
    sample_size = config.get("retrieval", {}).get("experiment_sample_size", 10000)
    
    db_path = Path(config["output"]["directory"]) / "step5/indices.db"
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    s1_path = data_root / "dataset/train/train_source1.tsv"
    
    output_dir = Path(config["output"]["directory"]) / "step6"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logging.info("Initializing Generator and Evaluator...")
    generator = CandidateGenerator(str(db_path), high_df_threshold)
    evaluator = RecallEvaluator(str(gt_path))
    
    # Read sample of S1
    logging.info(f"Reading {sample_size} records from S1 for experiment...")
    s1_records = []
    for chunk in read_tsv_chunks(s1_path, "source1", sample_size):
        for row in chunk.itertuples(index=False):
            s1_records.append((row.entity_id, row.business_name, row.business_address, row.country))
            if len(s1_records) >= sample_size:
                break
        if len(s1_records) >= sample_size:
            break
            
    experiment_results = {
        "_metadata": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "sample_size": len(s1_records),
            "sample_first_id": s1_records[0][0] if s1_records else None,
            "sample_last_id": s1_records[-1][0] if s1_records else None,
            "index_path": str(db_path.resolve()),
        }
    }
    
    max_budget = max(budgets)
    master_results = {}
    
    t0 = time.time()
    for idx, (entity_id, name, addr, country) in enumerate(s1_records):
        if idx > 0 and idx % 200 == 0:
            logging.info(f"Processed {idx}/{sample_size} queries...")
            
        candidates = generator.get_candidates(name, addr, country, budget=max_budget)
        master_results[entity_id] = candidates
        
    t1 = time.time()
    runtime_secs = t1 - t0
    
    for B in budgets:
        logging.info(f"=== Evaluating experiment for Budget B = {B} ===")
        retrieved_results = {}
        cand_counts = []
        
        for entity_id, cands in master_results.items():
            sliced = cands[:B]
            retrieved_results[entity_id] = sliced
            cand_counts.append(len(sliced))
            
        metrics = evaluator.evaluate(retrieved_results)
        
        metrics["average_candidates"] = round(np.mean(cand_counts), 2)
        metrics["p95_candidates"] = int(np.percentile(cand_counts, 95))
        metrics["p99_candidates"] = int(np.percentile(cand_counts, 99))
        metrics["max_candidates"] = int(np.max(cand_counts))
        metrics["runtime_seconds"] = round(runtime_secs, 2)
        metrics["queries_per_second"] = round(len(s1_records) / runtime_secs, 2) if runtime_secs > 0 else 0
        metrics["peak_rss_mb"] = round(get_peak_memory_mb(), 2)
        
        experiment_results[f"B_{B}"] = metrics
        
        logging.info(f"B={B} | Pair Recall: {metrics['pair_level_recall']} | Avg Cands: {metrics['average_candidates']}")
        
    report_path = output_dir / "budget_experiment_results.json"
    with open(report_path, "w") as f:
        json.dump(experiment_results, f, indent=4)
        
    logging.info(f"Experiment complete. Results saved to {report_path}")

if __name__ == "__main__":
    main()
