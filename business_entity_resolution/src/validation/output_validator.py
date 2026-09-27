"""Output validator for business entity resolution challenge submissions.

Validates the two submission TSV files:
1. candidate_pairs.tsv:
   - Header: source1_entity_id\tcandidate_entity_ids
   - Every S1 entity present exactly once
   - No duplicate candidates within a row
   - Candidate order preserved from retrieval
2. matching_results.tsv:
   - Header: source1_entity_id\tmatched_entity_ids
   - Every S1 entity present exactly once
   - No duplicate matches within a row
   - Strict subset invariant: every matched entity ID MUST appear in candidate_pairs for that S1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def validate_tsv_files(
    candidate_tsv_path: str | Path,
    matching_tsv_path: str | Path,
    expected_s1_ids: Iterable[str] | None = None,
    expected_count: int | None = 1_732_544,
) -> dict[str, Any]:
    """Validate both candidate_pairs.tsv and matching_results.tsv against all competition rules."""
    cand_p = Path(candidate_tsv_path).resolve()
    match_p = Path(matching_tsv_path).resolve()

    if not cand_p.exists():
        raise FileNotFoundError(f"Candidate TSV not found at {cand_p}")
    if not match_p.exists():
        raise FileNotFoundError(f"Matching TSV not found at {match_p}")

    expected_id_set = set(expected_s1_ids) if expected_s1_ids is not None else None
    if expected_id_set is not None and expected_count is not None:
        if len(expected_id_set) != expected_count:
            logging.warning(
                f"expected_s1_ids length ({len(expected_id_set)}) differs from expected_count ({expected_count})"
            )

    logging.info(f"Validating candidate TSV: {cand_p}")
    logging.info(f"Validating matching TSV:  {match_p}")

    cand_rows = 0
    match_rows = 0
    seen_cand_s1: set[str] = set()
    seen_match_s1: set[str] = set()

    total_candidates = 0
    total_matches = 0
    empty_candidates_count = 0
    empty_matches_count = 0
    max_candidates_per_s1 = 0
    max_matches_per_s1 = 0

    with open(cand_p, "r", encoding="utf-8") as f_cand, open(match_p, "r", encoding="utf-8") as f_match:
        # Validate headers
        cand_header = f_cand.readline()
        match_header = f_match.readline()

        cand_header_clean = cand_header.rstrip("\r\n")
        match_header_clean = match_header.rstrip("\r\n")

        expected_cand_header = "source1_entity_id\tcandidate_entity_ids"
        expected_match_header = "source1_entity_id\tmatched_entity_ids"

        if cand_header_clean != expected_cand_header:
            raise ValueError(f"Invalid candidate TSV header: {cand_header_clean!r}. Expected: {expected_cand_header!r}")
        if match_header_clean != expected_match_header:
            raise ValueError(f"Invalid matching TSV header: {match_header_clean!r}. Expected: {expected_match_header!r}")

        # Stream row by row
        line_num = 1
        for cand_line in f_cand:
            line_num += 1
            match_line = f_match.readline()
            if not match_line:
                raise ValueError(f"Matching TSV has fewer lines than candidate TSV at line {line_num}")

            cand_parts = cand_line.rstrip("\r\n").split("\t")
            match_parts = match_line.rstrip("\r\n").split("\t")

            if len(cand_parts) == 1:
                cand_s1 = cand_parts[0]
                cand_str = ""
            elif len(cand_parts) == 2:
                cand_s1, cand_str = cand_parts
            else:
                raise ValueError(f"Candidate line {line_num} has {len(cand_parts)} tab-separated parts (expected 2)")

            if len(match_parts) == 1:
                match_s1 = match_parts[0]
                match_str = ""
            elif len(match_parts) == 2:
                match_s1, match_str = match_parts
            else:
                raise ValueError(f"Matching line {line_num} has {len(match_parts)} tab-separated parts (expected 2)")

            cand_s1 = cand_s1.strip()
            match_s1 = match_s1.strip()

            if cand_s1 != match_s1:
                raise ValueError(f"Row {line_num}: S1 ID mismatch! candidate='{cand_s1}', matching='{match_s1}'")

            # Check uniqueness
            if cand_s1 in seen_cand_s1:
                raise ValueError(f"Row {line_num}: Duplicate S1 ID '{cand_s1}' in candidate TSV")
            if match_s1 in seen_match_s1:
                raise ValueError(f"Row {line_num}: Duplicate S1 ID '{match_s1}' in matching TSV")

            seen_cand_s1.add(cand_s1)
            seen_match_s1.add(match_s1)

            # Parse candidate list
            cands = [c.strip() for c in cand_str.split(",") if c.strip()] if cand_str else []
            cand_set = set(cands)
            if len(cand_set) != len(cands):
                raise ValueError(f"Row {line_num} ({cand_s1}): Duplicate candidates found in candidate_pairs.tsv!")

            # Parse matches list
            matches = [m.strip() for m in match_str.split(",") if m.strip()] if match_str else []
            match_set = set(matches)
            if len(match_set) != len(matches):
                raise ValueError(f"Row {line_num} ({match_s1}): Duplicate matches found in matching_results.tsv!")

            # Strict subset invariant: matches must be subset of candidates
            if not match_set.issubset(cand_set):
                invalid_matches = match_set - cand_set
                raise ValueError(
                    f"Row {line_num} ({cand_s1}): Violation of strict subset invariant! "
                    f"Matches {invalid_matches} are not in candidate set {cand_set}."
                )

            cand_rows += 1
            match_rows += 1
            total_candidates += len(cands)
            total_matches += len(matches)
            if not cands:
                empty_candidates_count += 1
            if not matches:
                empty_matches_count += 1
            max_candidates_per_s1 = max(max_candidates_per_s1, len(cands))
            max_matches_per_s1 = max(max_matches_per_s1, len(matches))

        # Check if matching TSV has extra lines
        extra_match_line = f_match.readline()
        if extra_match_line:
            raise ValueError("Matching TSV has more rows than candidate TSV!")

    if expected_count is not None and cand_rows != expected_count:
        raise ValueError(f"Row count mismatch! Found {cand_rows} rows, expected {expected_count}")

    if expected_id_set is not None:
        missing_ids = expected_id_set - seen_cand_s1
        if missing_ids:
            raise ValueError(f"{len(missing_ids)} expected S1 entities missing from outputs (e.g. {next(iter(missing_ids))})")
        unexpected_ids = seen_cand_s1 - expected_id_set
        if unexpected_ids:
            raise ValueError(f"{len(unexpected_ids)} unexpected S1 entities in outputs (e.g. {next(iter(unexpected_ids))})")

    report = {
        "status": "PASS",
        "rows_validated": cand_rows,
        "total_candidates": total_candidates,
        "avg_candidates_per_s1": round(total_candidates / cand_rows, 2) if cand_rows else 0.0,
        "max_candidates_per_s1": max_candidates_per_s1,
        "empty_candidates_count": empty_candidates_count,
        "total_matches": total_matches,
        "avg_matches_per_s1": round(total_matches / cand_rows, 2) if cand_rows else 0.0,
        "max_matches_per_s1": max_matches_per_s1,
        "empty_matches_count (singletons)": empty_matches_count,
        "singleton_rate": round(empty_matches_count / cand_rows, 4) if cand_rows else 0.0,
        "candidate_file": str(cand_p),
        "matching_file": str(match_p),
    }

    logging.info(f"VALIDATION PASSED! Summary: {report}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate candidate_pairs.tsv and matching_results.tsv")
    parser.add_argument("--candidate-tsv", type=str, default="output/candidate_pairs.tsv")
    parser.add_argument("--matching-tsv", type=str, default="output/matching_results.tsv")
    parser.add_argument("--expected-count", type=int, default=1_732_544)
    args = parser.parse_args()

    validate_tsv_files(
        candidate_tsv_path=args.candidate_tsv,
        matching_tsv_path=args.matching_tsv,
        expected_count=args.expected_count,
    )


if __name__ == "__main__":
    main()
