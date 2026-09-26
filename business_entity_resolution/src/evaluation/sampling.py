"""Deterministic entity-level validation sampling with diagnostic strata classification."""

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
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

STRATA_KEYS = [
    "singleton",
    "one_match",
    "multi_2_4",
    "multi_5_plus",
    "missing_address",
    "country_us",
    "country_india",
    "s2_only",
    "s3_only",
    "both_sources",
]


def classify_strata(
    name: str,
    address: str,
    country: str,
    gt_targets: list[str],
) -> list[str]:
    """Classify an S1 record into overlapping diagnostic strata."""
    strata: list[str] = []
    num_matches = len(gt_targets)

    # Match cardinality strata
    if num_matches == 0:
        strata.append("singleton")
    elif num_matches == 1:
        strata.append("one_match")
    elif 2 <= num_matches <= 4:
        strata.append("multi_2_4")
    else:  # >= 5
        strata.append("multi_5_plus")

    # Missing address
    addr_clean = address.strip() if address else ""
    if not addr_clean or addr_clean.lower() in ("nan", "none", "null", "na", "-", "."):
        strata.append("missing_address")

    # Country strata
    country_clean = country.strip().upper() if country else ""
    if country_clean == "US":
        strata.append("country_us")
    elif country_clean in ("INDIA", "IN"):
        strata.append("country_india")

    # Target source distribution (only for non-singletons)
    if num_matches > 0:
        has_s2 = any(t.startswith("S2-") for t in gt_targets)
        has_s3 = any(t.startswith("S3-") for t in gt_targets)

        if has_s2 and not has_s3:
            strata.append("s2_only")
        elif has_s3 and not has_s2:
            strata.append("s3_only")
        elif has_s2 and has_s3:
            strata.append("both_sources")

    return strata


def load_ground_truth(gt_path: str | Path) -> dict[str, list[str]]:
    """Load ground truth mapping from S1 ID to list of target S2/S3 IDs."""
    gt_path = Path(gt_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    logging.info(f"Loading ground truth from {gt_path}...")
    gt_map: dict[str, list[str]] = {}
    df = pd.read_csv(gt_path, sep="\t", dtype="string", keep_default_na=False)

    for row in df.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        matched = str(row.matched_entity_ids).strip() if pd.notna(row.matched_entity_ids) else ""
        if matched:
            targets = [t.strip() for t in matched.split(",") if t.strip()]
            gt_map[s1_id] = targets
        else:
            gt_map[s1_id] = []

    logging.info(f"Loaded ground truth for {len(gt_map)} S1 entities.")
    return gt_map


def generate_samples(
    gt_path: str | Path,
    s1_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    smoke_size: int = 500,
    quality_size: int = 10000,
) -> dict[str, Any]:
    """Generate deterministic validation samples with diagnostic strata."""
    gt_path = Path(gt_path).resolve()
    s1_path = Path(s1_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_map = load_ground_truth(gt_path)

    total_population = 2_206_821
    if quality_size > total_population:
        raise ValueError(f"Quality sample size {quality_size} exceeds total population {total_population}")
    if smoke_size > quality_size:
        raise ValueError(f"Smoke size {smoke_size} cannot exceed quality size {quality_size}")

    rng = np.random.default_rng(seed)
    all_sample_indices = rng.choice(total_population, size=quality_size, replace=False)
    quality_indices_set = set(int(i) for i in all_sample_indices)
    smoke_indices_set = set(int(i) for i in all_sample_indices[:smoke_size])

    index_to_quality_order = {int(idx): rank for rank, idx in enumerate(all_sample_indices)}

    logging.info(f"Streaming S1 dataset ({s1_path}) to collect {quality_size} sample entities...")
    quality_records_dict: dict[int, dict[str, Any]] = {}

    current_idx = 0
    for chunk in read_tsv_chunks(s1_path, "source1", chunk_size=100_000):
        for row in chunk.itertuples(index=False):
            if current_idx in quality_indices_set:
                s1_id = str(row.entity_id).strip()
                name = str(row.business_name).strip() if pd.notna(row.business_name) else ""
                addr = str(row.business_address).strip() if pd.notna(row.business_address) else ""
                country = str(row.country).strip() if pd.notna(row.country) else ""

                if country.lower() == "france":
                    raise ValueError(f"Disallowed country 'France' encountered in training S1 record {s1_id}")

                targets = gt_map.get(s1_id, [])
                strata = classify_strata(name, addr, country, targets)

                order = index_to_quality_order[current_idx]
                is_smoke = current_idx in smoke_indices_set

                quality_records_dict[order] = {
                    "entity_id": s1_id,
                    "business_name": name,
                    "business_address": addr,
                    "country": country,
                    "ground_truth_matches": targets,
                    "num_matches": len(targets),
                    "strata": strata,
                    "is_smoke": is_smoke,
                    "population_index": current_idx,
                }

                if len(quality_records_dict) == quality_size:
                    break

            current_idx += 1
        if len(quality_records_dict) == quality_size:
            break

    quality_records = [quality_records_dict[i] for i in range(quality_size)]
    smoke_records = [r for r in quality_records if r["is_smoke"]]

    def save_sample_artifacts(records: list[dict[str, Any]], sample_name: str):
        json_path = output_dir / f"{sample_name}.json"
        tsv_path = output_dir / f"{sample_name}_ids.tsv"

        id_list = [r["entity_id"] for r in records]
        id_hash = hashlib.sha256("\n".join(id_list).encode("utf-8")).hexdigest()

        strata_distribution: dict[str, int] = defaultdict(int)
        for r in records:
            for s in r["strata"]:
                strata_distribution[s] += 1

        payload = {
            "metadata": {
                "sample_name": sample_name,
                "sample_size": len(records),
                "seed": seed,
                "entity_ids_sha256": id_hash,
                "source1_path": str(s1_path),
                "ground_truth_path": str(gt_path),
            },
            "strata_counts": dict(strata_distribution),
            "strata_percentages": {
                k: round(v / len(records) * 100, 2) for k, v in strata_distribution.items()
            },
            "records": records,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tnum_matches\tstrata\n")
            for r in records:
                s_str = ",".join(r["strata"])
                f.write(f"{r['entity_id']}\t{r['num_matches']}\t{s_str}\n")

        logging.info(f"Saved {sample_name} to {json_path} and {tsv_path}")
        return payload

    smoke_payload = save_sample_artifacts(smoke_records, f"validation_smoke_{smoke_size}")
    quality_payload = save_sample_artifacts(quality_records, f"validation_quality_{quality_size}")

    manifest = {
        "seed": seed,
        "population_size": total_population,
        "smoke_sample": {
            "size": len(smoke_records),
            "file": f"validation_smoke_{smoke_size}.json",
            "sha256": smoke_payload["metadata"]["entity_ids_sha256"],
            "strata_distribution": smoke_payload["strata_counts"],
        },
        "quality_sample": {
            "size": len(quality_records),
            "file": f"validation_quality_{quality_size}.json",
            "sha256": quality_payload["metadata"]["entity_ids_sha256"],
            "strata_distribution": quality_payload["strata_counts"],
        },
        "training_exclusion_policy": (
            "Any future matching model (e.g., LightGBM / Hard-Negative Mining) MUST exclude "
            "all S1 entities present in these validation samples to prevent data leakage."
        ),
    }

    manifest_path = output_dir / "validation_split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logging.info(f"Validation split manifest written to {manifest_path}")
    return manifest


def load_validation_sample(sample_path: str | Path) -> dict[str, Any]:
    """Load a persisted validation sample JSON."""
    sample_path = Path(sample_path).resolve()
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    with open(sample_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Deterministic entity-level validation sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-size", type=int, default=500)
    parser.add_argument("--quality-size", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="output/validation")
    args = parser.parse_args()

    config = load_config()
    data_root = Path(config["data"]["root"])
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    s1_path = data_root / "dataset/train/train_source1.tsv"

    generate_samples(
        gt_path=gt_path,
        s1_path=s1_path,
        output_dir=args.output_dir,
        seed=args.seed,
        smoke_size=args.smoke_size,
        quality_size=args.quality_size,
    )


if __name__ == "__main__":
    main()
