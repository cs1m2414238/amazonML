import pandas as pd
import logging
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
import gc
from pathlib import Path

from business_entity_resolution.src.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    config = load_config()
    data_root = Path(config["data"]["root"])
    s2_path = data_root / "dataset/train/train_source2.tsv"
    s3_path = data_root / "dataset/train/train_source3.tsv"
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"

    logging.info("Building S2/S3 ID set...")
    valid_targets = set()
    for path, source in [(s2_path, "source2"), (s3_path, "source3")]:
        for chunk in read_tsv_chunks(path, source):
            valid_targets.update(chunk['entity_id'].tolist())
            
    logging.info(f"Loaded {len(valid_targets)} valid target IDs from S2 and S3.")
    
    logging.info("Checking Ground Truth against valid targets...")
    df_gt = pd.read_csv(gt_path, sep="\t", dtype="string")
    
    missing_targets = []
    
    for row in df_gt.itertuples(index=False):
        if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            targets = str(row.matched_entity_ids).split(",")
            for t in targets:
                t = t.strip()
                if t not in valid_targets:
                    missing_targets.append((row.source1_entity_id, t))
                    
    logging.info(f"Found {len(missing_targets)} GT targets that DO NOT exist in S2/S3!")
    
    if missing_targets:
        for i in range(min(10, len(missing_targets))):
            logging.info(f"Missing GT target: S1={missing_targets[i][0]} -> S2/S3={missing_targets[i][1]}")

if __name__ == "__main__":
    main()
