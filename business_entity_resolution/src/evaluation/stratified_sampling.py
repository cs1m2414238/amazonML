"""Quota-controlled, deterministic validation sampling for blocking evaluation.

The quality sample deliberately over-represents rare singleton and one-match
entities.  Overall figures from this sample are therefore diagnostic sample
metrics, not population-weighted estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.evaluation.sampling import classify_strata
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

MATCH_BANDS = ("singleton", "one_match", "multi_2_4", "multi_5_plus")
COUNTRIES = ("US", "India")

# Deliberately diagnostic rather than population-proportional.  Rare
# singleton and one-match entities each receive 10% of the 10,000 sample.
DEFAULT_QUOTAS_10K: dict[str, int] = {
    "US|singleton": 600,
    "US|one_match": 600,
    "US|multi_2_4": 3_000,
    "US|multi_5_plus": 1_800,
    "India|singleton": 400,
    "India|one_match": 400,
    "India|multi_2_4": 2_000,
    "India|multi_5_plus": 1_200,
}


def is_missing_text(value: Any) -> bool:
    """Return whether a source text value is semantically missing."""
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "na", "-", "."}


def match_band(match_count: int) -> str:
    """Map a ground-truth cardinality to its exclusive sampling band."""
    if match_count == 0:
        return "singleton"
    if match_count == 1:
        return "one_match"
    if match_count <= 4:
        return "multi_2_4"
    return "multi_5_plus"


def primary_stratum(country: str, match_count: int) -> str:
    """Build the exclusive country-by-cardinality sampling stratum."""
    normalized = str(country).strip().lower()
    if normalized == "us":
        country_label = "US"
    elif normalized in {"india", "in"}:
        country_label = "India"
    else:
        raise ValueError(f"Unexpected training country: {country!r}")
    return f"{country_label}|{match_band(match_count)}"


def scaled_quotas(sample_size: int) -> dict[str, int]:
    """Scale the canonical 10k quotas using deterministic largest remainders."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    canonical_total = sum(DEFAULT_QUOTAS_10K.values())
    exact = {
        key: sample_size * quota / canonical_total
        for key, quota in DEFAULT_QUOTAS_10K.items()
    }
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = sample_size - sum(quotas.values())
    remainder_order = sorted(
        exact,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in remainder_order[:remaining]:
        quotas[key] += 1
    return quotas


def stable_priority(seed: int, entity_id: str, namespace: str = "select") -> int:
    """Return a stable 128-bit priority independent of input row order."""
    payload = f"{namespace}|{seed}|{entity_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _load_ground_truth_strings(gt_path: Path) -> dict[str, str]:
    """Load the compact S1 -> comma-separated ground truth representation."""
    logging.info("Loading compact ground-truth lookup from %s", gt_path)
    result: dict[str, str] = {}
    for chunk in pd.read_csv(
        gt_path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        chunksize=250_000,
        usecols=["source1_entity_id", "matched_entity_ids"],
    ):
        result.update(
            zip(
                chunk["source1_entity_id"].astype(str),
                chunk["matched_entity_ids"].astype(str),
            )
        )
    logging.info("Loaded ground truth for %s Source 1 entities", f"{len(result):,}")
    return result


def _match_targets(raw_matches: str) -> list[str]:
    return [item.strip() for item in raw_matches.split(",") if item.strip()]


def _sample_hash(records: Iterable[dict[str, Any]]) -> str:
    ids = [str(record["entity_id"]) for record in records]
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def select_stratified_records(
    gt_path: str | Path,
    s1_path: str | Path,
    quotas: dict[str, int],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the smallest stable hashes within each exclusive stratum.

    Keeping the smallest hashes makes the result reproducible even if the TSV
    row order changes.  The heap bounds retained records to the requested
    sample size while the full population is streamed once.
    """
    gt_path = Path(gt_path).resolve()
    s1_path = Path(s1_path).resolve()
    unknown_keys = set(quotas) - set(DEFAULT_QUOTAS_10K)
    if unknown_keys:
        raise ValueError(f"Unknown quota strata: {sorted(unknown_keys)}")
    if any(value < 0 for value in quotas.values()):
        raise ValueError("Stratum quotas cannot be negative")

    gt_lookup = _load_ground_truth_strings(gt_path)
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
        key: [] for key in quotas
    }
    population_counts: Counter[str] = Counter()
    missing_address_by_country: Counter[str] = Counter()
    source_pattern_counts: Counter[str] = Counter()
    population_rows = 0

    logging.info("Streaming Source 1 and applying deterministic stratum quotas")
    for chunk in read_tsv_chunks(s1_path, "source1", chunk_size=100_000):
        for row in chunk.itertuples(index=False):
            population_index = population_rows
            population_rows += 1
            entity_id = str(row.entity_id).strip()
            country = str(row.country).strip()
            raw_matches = gt_lookup.get(entity_id)
            if raw_matches is None:
                raise ValueError(f"Ground truth is missing Source 1 entity {entity_id}")
            targets = _match_targets(raw_matches)
            stratum = primary_stratum(country, len(targets))
            population_counts[stratum] += 1

            address = "" if is_missing_text(row.business_address) else str(row.business_address).strip()
            if not address:
                missing_address_by_country[country] += 1

            if targets:
                has_s2 = any(target.startswith("S2-") for target in targets)
                has_s3 = any(target.startswith("S3-") for target in targets)
                if has_s2 and has_s3:
                    source_pattern_counts["both_sources"] += 1
                elif has_s2:
                    source_pattern_counts["s2_only"] += 1
                elif has_s3:
                    source_pattern_counts["s3_only"] += 1

            quota = quotas.get(stratum, 0)
            if quota <= 0:
                continue

            name = "" if is_missing_text(row.business_name) else str(row.business_name).strip()
            record = {
                "entity_id": entity_id,
                "business_name": name,
                "business_address": address,
                "country": country,
                "ground_truth_matches": targets,
                "num_matches": len(targets),
                "strata": classify_strata(name, address, country, targets),
                "sampling_stratum": stratum,
                "population_index": population_index,
            }
            priority = stable_priority(seed, entity_id)
            item = (-priority, entity_id, record)
            heap = heaps[stratum]
            if len(heap) < quota:
                heapq.heappush(heap, item)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, item)

    if len(gt_lookup) != population_rows:
        raise ValueError(
            "Source 1 and ground truth row counts differ: "
            f"{population_rows:,} vs {len(gt_lookup):,}"
        )

    shortfalls = {
        key: {"requested": quota, "available": population_counts.get(key, 0)}
        for key, quota in quotas.items()
        if len(heaps[key]) != quota
    }
    if shortfalls:
        raise ValueError(f"Insufficient population for requested quotas: {shortfalls}")

    selected = [item[2] for heap in heaps.values() for item in heap]
    selected.sort(
        key=lambda record: (
            stable_priority(seed, record["entity_id"], namespace="order"),
            record["entity_id"],
        )
    )

    sample_strata = Counter(record["sampling_stratum"] for record in selected)
    diagnostic_strata: Counter[str] = Counter()
    for record in selected:
        diagnostic_strata.update(record["strata"])

    missing_population = sum(missing_address_by_country.values())
    metadata = {
        "seed": seed,
        "selection_method": "smallest_sha256_priority_per_country_cardinality_stratum_v1",
        "population_size": population_rows,
        "population_primary_strata": dict(sorted(population_counts.items())),
        "population_source_patterns": dict(sorted(source_pattern_counts.items())),
        "requested_quotas": dict(quotas),
        "realized_primary_strata": dict(sorted(sample_strata.items())),
        "diagnostic_strata_counts": dict(sorted(diagnostic_strata.items())),
        "missing_address_population": missing_population,
        "missing_address_population_by_country": dict(sorted(missing_address_by_country.items())),
        "missing_address_coverage": (
            "covered" if diagnostic_strata.get("missing_address", 0) else "unavailable_in_training_source1"
        ),
        "sample_size": len(selected),
        "entity_ids_sha256": _sample_hash(selected),
        "source1_path": str(s1_path),
        "ground_truth_path": str(gt_path),
    }
    return selected, metadata


def write_stratified_artifacts(
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Persist the fixed sample, compact ID list, and split manifest."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_size = len(records)
    sample_name = f"validation_stratified_{sample_size}"
    sample_path = output_dir / f"{sample_name}.json"
    ids_path = output_dir / f"{sample_name}_ids.tsv"
    manifest_path = output_dir / "validation_stratified_manifest.json"

    payload = {
        "metadata": {"sample_name": sample_name, **metadata},
        "records": records,
    }
    with sample_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)

    with ids_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("entity_id\tcountry\tnum_matches\tsampling_stratum\tstrata\n")
        for record in records:
            stream.write(
                f"{record['entity_id']}\t{record['country']}\t{record['num_matches']}\t"
                f"{record['sampling_stratum']}\t{','.join(record['strata'])}\n"
            )

    manifest = {
        **metadata,
        "sample_file": str(sample_path),
        "id_file": str(ids_path),
        "training_exclusion_policy": (
            "Exclude every listed Source 1 entity from matcher fitting and hard-negative "
            "mining. Use them only for validation and threshold selection."
        ),
        "metric_interpretation": (
            "The quotas intentionally oversample rare cardinalities, so aggregate values are "
            "diagnostic stratified-sample metrics rather than population-weighted estimates."
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    logging.info("Saved sample to %s", sample_path)
    logging.info("Saved manifest to %s", manifest_path)
    return {
        "sample": str(sample_path),
        "ids": str(ids_path),
        "manifest": str(manifest_path),
    }


def generate_stratified_sample(
    gt_path: str | Path,
    s1_path: str | Path,
    output_dir: str | Path,
    sample_size: int = 10_000,
    seed: int = 42,
) -> dict[str, str]:
    quotas = scaled_quotas(sample_size)
    records, metadata = select_stratified_records(
        gt_path=gt_path,
        s1_path=s1_path,
        quotas=quotas,
        seed=seed,
    )
    return write_stratified_artifacts(records, metadata, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic stratified validation sample")
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()

    config = load_config()
    data_root = Path(config["data"]["root"])
    output_root = Path(config["output"]["directory"])
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "validation_stratified"
    generate_stratified_sample(
        gt_path=data_root / "dataset/train/train_ground_truth.tsv",
        s1_path=data_root / "dataset/train/train_source1.tsv",
        output_dir=output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
