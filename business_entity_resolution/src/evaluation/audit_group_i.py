import pandas as pd
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
import logging
from pathlib import Path

from business_entity_resolution.src.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    config = load_config()
    data_root = Path(config["data"]["root"])
    s1_path = data_root / "dataset/train/train_source1.tsv"
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    
    logging.info("Parsing Ground Truth...")
    gt_map = {}
    df_gt = pd.read_csv(gt_path, sep="\t", dtype="string")
    for row in df_gt.itertuples(index=False):
        if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            targets = str(row.matched_entity_ids).split(",")
            gt_map[row.source1_entity_id] = len(targets)
        else:
            gt_map[row.source1_entity_id] = 0

    logging.info("Streaming S1 for Missing Address audit...")
    missing_addr_count = 0
    missing_addr_with_gt = 0
    missing_addr_zero_gt = 0
    gt_dist = {}
    
    for chunk in read_tsv_chunks(s1_path, "source1"):
        for row in chunk.itertuples(index=False):
            addr = str(row.business_address).strip() if pd.notna(row.business_address) else ""
            if not addr:
                missing_addr_count += 1
                gt_len = gt_map.get(row.entity_id, 0)
                if gt_len > 0:
                    missing_addr_with_gt += 1
                else:
                    missing_addr_zero_gt += 1
                gt_dist[gt_len] = gt_dist.get(gt_len, 0) + 1
                
    logging.info(f"Total Missing Address records in S1: {missing_addr_count}")
    logging.info(f"With >=1 GT match: {missing_addr_with_gt}")
    logging.info(f"With 0 GT matches: {missing_addr_zero_gt}")
    logging.info(f"GT Distribution: {gt_dist}")

if __name__ == "__main__":
    main()
