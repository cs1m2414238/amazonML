"""Windows-safe parallel retrieval with atomic, resumable output parts.

Each worker owns an independent read-only ``OptimizedCandidateGenerator`` and
writes one JSONL part per input batch. A sidecar metadata file is written only
after the data file is atomically installed, so an interrupted run can safely
skip completed batches on restart.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from business_entity_resolution.src.retrieval.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb


Record = tuple[str, str, str, str]
RecordBatch = tuple[int, list[Record]]

_WORKER_GENERATOR: OptimizedCandidateGenerator | None = None


@dataclass(frozen=True)
class WorkerSettings:
    db_path: str
    high_df_threshold: int = 50_000
    posting_cache_mb: int = 1024
    sqlite_cache_mb: int = 512
    mmap_mb: int = 8192


@dataclass(frozen=True)
class RunSettings:
    budget: int
    workers: int
    batch_size: int
    dataset_signature: str
    total_records: int
    total_batches: int
    worker: WorkerSettings


def make_record_batches(
    records: Sequence[Record], batch_size: int
) -> Iterator[RecordBatch]:
    """Yield stable, zero-based batches from an in-memory record sequence."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(records), batch_size):
        yield start // batch_size, list(records[start : start + batch_size])


def _close_worker() -> None:
    global _WORKER_GENERATOR
    if _WORKER_GENERATOR is not None:
        _WORKER_GENERATOR.close()
        _WORKER_GENERATOR = None


def _init_worker(settings: dict, ready_queue) -> None:
    """Create exactly one read-only SQLite connection in each worker."""
    global _WORKER_GENERATOR
    try:
        worker_settings = WorkerSettings(**settings)
        _WORKER_GENERATOR = OptimizedCandidateGenerator(
            worker_settings.db_path,
            high_df_threshold=worker_settings.high_df_threshold,
            posting_cache_mb=worker_settings.posting_cache_mb,
            sqlite_cache_mb=worker_settings.sqlite_cache_mb,
            mmap_mb=worker_settings.mmap_mb,
        )
        atexit.register(_close_worker)
        ready_queue.put({"ok": True, "pid": os.getpid()})
    except BaseException:
        ready_queue.put(
            {
                "ok": False,
                "pid": os.getpid(),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _process_batch(task: dict) -> dict:
    """Retrieve and atomically persist one batch inside a worker process."""
    if _WORKER_GENERATOR is None:
        raise RuntimeError("Retrieval worker was not initialized")

    batch_id = int(task["batch_id"])
    records: list[Record] = task["records"]
    budget = int(task["budget"])
    run_signature = task["run_signature"]
    part_path = Path(task["part_path"])
    meta_path = Path(task["meta_path"])

    lines: list[str] = []
    latencies: list[float] = []
    candidate_total = 0
    started = time.perf_counter()
    for entity_id, name, address, country in records:
        query_started = time.perf_counter()
        candidates = _WORKER_GENERATOR.get_candidates(
            name, address, country, budget=budget
        )
        latencies.append(time.perf_counter() - query_started)
        candidate_total += len(candidates)
        lines.append(
            json.dumps(
                {"source1_entity_id": entity_id, "candidates": candidates},
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
    elapsed = time.perf_counter() - started

    payload = "\n".join(lines) + ("\n" if lines else "")
    payload_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    _atomic_write_text(part_path, payload)

    metadata = {
        "run_signature": run_signature,
        "batch_id": batch_id,
        "record_count": len(records),
        "first_entity_id": records[0][0] if records else None,
        "last_entity_id": records[-1][0] if records else None,
        "candidate_total": candidate_total,
        "payload_sha256": payload_sha256,
    }
    _atomic_write_text(
        meta_path,
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
    )

    return {
        **metadata,
        "elapsed_seconds": elapsed,
        "latency_sum_seconds": float(sum(latencies)),
        "latency_min_seconds": min(latencies) if latencies else 0.0,
        "latency_max_seconds": max(latencies) if latencies else 0.0,
        "latencies_seconds": latencies,
        "pid": os.getpid(),
        "worker_peak_rss_mb": get_peak_memory_mb(),
    }


class ParallelResumableRetriever:
    """Orchestrate deterministic parallel retrieval into resumable parts."""

    MANIFEST_VERSION = 1

    def __init__(self, run_directory: str | Path):
        self.run_directory = Path(run_directory).resolve()
        self.parts_directory = self.run_directory / "parts"
        self.manifest_path = self.run_directory / "manifest.json"

    @staticmethod
    def _run_signature(settings: RunSettings) -> str:
        encoded = json.dumps(
            asdict(settings), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _part_paths(self, batch_id: int) -> tuple[Path, Path]:
        stem = f"part-{batch_id:06d}"
        return (
            self.parts_directory / f"{stem}.jsonl",
            self.parts_directory / f"{stem}.meta.json",
        )

    def _read_part_meta(self, batch_id: int, run_signature: str) -> dict | None:
        part_path, meta_path = self._part_paths(batch_id)
        if not part_path.exists() or not meta_path.exists():
            return None
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            metadata.get("run_signature") != run_signature
            or metadata.get("batch_id") != batch_id
        ):
            return None
        return metadata

    def _write_manifest(self, manifest: dict) -> None:
        _atomic_write_text(
            self.manifest_path,
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        )

    def _prepare_manifest(self, settings: RunSettings, run_signature: str) -> tuple[dict, str]:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.parts_directory.mkdir(parents=True, exist_ok=True)
        expected = {
            "manifest_version": self.MANIFEST_VERSION,
            "run_signature": run_signature,
            "settings": asdict(settings),
        }
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("run_signature") != run_signature:
                # Check if data settings match (everything except workers)
                existing_settings = existing.get("settings", {})
                current_settings = asdict(settings)
                data_keys = ("batch_size", "budget", "dataset_signature", "total_batches", "total_records", "worker")
                data_matches = all(existing_settings.get(k) == current_settings.get(k) for k in data_keys)
                if data_matches:
                    # Safe resumption: preserve existing run_signature so that all existing parts remain valid
                    run_signature = existing["run_signature"]
                    existing["settings"]["workers"] = settings.workers
                    self._write_manifest(existing)
                    return existing, run_signature
                else:
                    raise ValueError(
                        "Existing run directory belongs to different retrieval settings: "
                        f"{self.run_directory}"
                    )
            return existing, run_signature
        expected.update(
            {
                "status": "running",
                "completed_batches": 0,
                "completed_records": 0,
            }
        )
        self._write_manifest(expected)
        return expected, run_signature

    def run(
        self,
        batches: Iterable[RecordBatch],
        settings: RunSettings,
        startup_timeout_seconds: float = 300.0,
        collect_latencies: bool = True,
    ) -> dict:
        """Run pending batches and return speed/memory metadata.

        Completed parts with matching atomic sidecars are skipped. The returned
        processing QPS covers newly processed records only; startup is reported
        separately so worker-count comparisons remain interpretable.
        """
        if settings.workers <= 0:
            raise ValueError("workers must be positive")
        if settings.budget <= 0:
            raise ValueError("budget must be positive")

        run_signature = self._run_signature(settings)
        manifest, run_signature = self._prepare_manifest(settings, run_signature)
        completed_meta = {
            batch_id: metadata
            for batch_id in range(settings.total_batches)
            if (metadata := self._read_part_meta(batch_id, run_signature)) is not None
        }

        def pending_tasks() -> Iterator[dict]:
            for batch_id, records in batches:
                if batch_id in completed_meta:
                    continue
                part_path, meta_path = self._part_paths(batch_id)
                yield {
                    "batch_id": batch_id,
                    "records": records,
                    "budget": settings.budget,
                    "run_signature": run_signature,
                    "part_path": str(part_path),
                    "meta_path": str(meta_path),
                }

        task_iterator = iter(pending_tasks())
        first_task = next(task_iterator, None)
        if first_task is None:
            manifest.update(
                {
                    "status": "complete",
                    "completed_batches": len(completed_meta),
                    "completed_records": sum(
                        int(meta["record_count"]) for meta in completed_meta.values()
                    ),
                }
            )
            self._write_manifest(manifest)
            return {
                "run_signature": run_signature,
                "resumed_batches": len(completed_meta),
                "processed_batches": 0,
                "processed_records": 0,
                "startup_seconds": 0.0,
                "processing_seconds": 0.0,
                "end_to_end_seconds": 0.0,
                "processing_qps": 0.0,
                "worker_peak_rss_mb_by_pid": {},
                "worker_peak_rss_sum_mb": 0.0,
                "parent_peak_rss_mb": get_peak_memory_mb(),
                "aggregate_peak_upper_bound_mb": get_peak_memory_mb(),
                "latencies_seconds": [],
            }
        tasks = chain((first_task,), task_iterator)

        context = mp.get_context("spawn")
        ready_queue = context.Queue()
        overall_started = time.perf_counter()
        pool = context.Pool(
            processes=settings.workers,
            initializer=_init_worker,
            initargs=(asdict(settings.worker), ready_queue),
        )
        worker_pids: set[int] = set()
        try:
            deadline = time.monotonic() + startup_timeout_seconds
            while len(worker_pids) < settings.workers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out while initializing retrieval workers")
                try:
                    status = ready_queue.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError(
                        "Timed out while initializing retrieval workers"
                    ) from exc
                if not status.get("ok"):
                    raise RuntimeError(status.get("traceback", "Worker startup failed"))
                worker_pids.add(int(status["pid"]))
            startup_seconds = time.perf_counter() - overall_started

            processing_started = time.perf_counter()
            processed_records = 0
            processed_batches = 0
            latencies: list[float] = []
            worker_peaks: dict[int, float] = {}
            batch_results = pool.imap_unordered(_process_batch, tasks, chunksize=1)
            for result in batch_results:
                batch_id = int(result["batch_id"])
                completed_meta[batch_id] = result
                processed_batches += 1
                processed_records += int(result["record_count"])
                if collect_latencies:
                    latencies.extend(result["latencies_seconds"])
                pid = int(result["pid"])
                worker_peaks[pid] = max(
                    worker_peaks.get(pid, 0.0),
                    float(result["worker_peak_rss_mb"]),
                )
                manifest.update(
                    {
                        "status": "running",
                        "completed_batches": len(completed_meta),
                        "completed_records": sum(
                            int(meta["record_count"])
                            for meta in completed_meta.values()
                        ),
                    }
                )
                self._write_manifest(manifest)

            processing_seconds = time.perf_counter() - processing_started
            pool.close()
            pool.join()
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        finally:
            ready_queue.close()

        end_to_end_seconds = time.perf_counter() - overall_started
        parent_peak = get_peak_memory_mb()
        worker_peak_sum = sum(worker_peaks.values())
        if len(completed_meta) != settings.total_batches:
            raise RuntimeError(
                f"Expected {settings.total_batches} completed batches, found "
                f"{len(completed_meta)}. Check total_records and the input iterator."
            )
        manifest.update(
            {
                "status": "complete",
                "completed_batches": len(completed_meta),
                "completed_records": sum(
                    int(meta["record_count"]) for meta in completed_meta.values()
                ),
            }
        )
        self._write_manifest(manifest)
        return {
            "run_signature": run_signature,
            "resumed_batches": settings.total_batches - processed_batches,
            "processed_batches": processed_batches,
            "processed_records": processed_records,
            "startup_seconds": startup_seconds,
            "processing_seconds": processing_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            "processing_qps": (
                processed_records / processing_seconds if processing_seconds else 0.0
            ),
            "worker_peak_rss_mb_by_pid": {
                str(pid): value for pid, value in sorted(worker_peaks.items())
            },
            "worker_peak_rss_sum_mb": worker_peak_sum,
            "parent_peak_rss_mb": parent_peak,
            "aggregate_peak_upper_bound_mb": parent_peak + worker_peak_sum,
            "latencies_seconds": latencies,
        }

    def iter_results(self) -> Iterator[dict]:
        """Stream completed result records in deterministic input-batch order."""
        for path in sorted(self.parts_directory.glob("part-*.jsonl")):
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)

    def result_digest(self) -> str:
        """Hash ordered entity IDs and candidates for equivalence checks."""
        digest = hashlib.sha256()
        for result in self.iter_results():
            digest.update(result["source1_entity_id"].encode("utf-8"))
            digest.update(b"\0")
            for candidate in result["candidates"]:
                digest.update(candidate.encode("utf-8"))
                digest.update(b"\0")
            digest.update(b"\n")
        return digest.hexdigest()

    def export_candidate_pairs_tsv(self, output_path: str | Path) -> dict:
        """Export the exact checkpointed candidate set in competition format.

        This is a streaming, atomic conversion: every JSONL checkpoint record
        becomes exactly one TSV row, including records with an empty candidate
        list. Candidate order is preserved and duplicates are rejected rather
        than silently changing the set supplied to the matching model.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        row_count = 0
        empty_count = 0
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("source1_entity_id\tcandidate_entity_ids\n")
            for result in self.iter_results():
                source1_id = result["source1_entity_id"]
                candidates = result["candidates"]
                if len(candidates) != len(set(candidates)):
                    raise ValueError(
                        f"Duplicate candidate IDs for {source1_id}; refusing to "
                        "change the model input during TSV export"
                    )
                if "\t" in source1_id or "\n" in source1_id:
                    raise ValueError(f"Invalid Source 1 ID for TSV: {source1_id!r}")
                if any("\t" in value or "\n" in value or "," in value for value in candidates):
                    raise ValueError(f"Invalid candidate ID for {source1_id}")
                stream.write(f"{source1_id}\t{','.join(candidates)}\n")
                row_count += 1
                empty_count += not candidates
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
        return {
            "path": str(output_path),
            "rows": row_count,
            "empty_candidate_rows": empty_count,
        }


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(max(values)),
    }
