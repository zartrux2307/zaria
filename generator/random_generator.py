from __future__ import annotations
import os
import time
import csv
import math  # <-- IMPORTACIÓN PATCH OK
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore

from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.proxy.randomx_validator import RandomXValidator
from iazar.generator.NonceCSVWriter import NonceCSVWriter

LOGGER_NAME = "generator.random"
logger = logging.getLogger(LOGGER_NAME)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

UINT32_MASK = 0xFFFFFFFF
CSV_FIELDS = ["nonce", "entropy", "uniqueness", "zero_density", "pattern_score", "is_valid", "block_height"]

class RandomGenerator(BaseNonceGenerator):
    DEFAULTS = {
        "batch_candidate_factor": 4,
        "max_validation_workers": 8,
        "recent_nonces_size": 500,
        "max_attempt_loops": 8,
        "max_candidate_factor": 32,
        "min_entropy_threshold": 0.0,
        "pattern_score_min": 0.5,
        "write_direct": True
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, validator: Optional[RandomXValidator] = None):
        merged = {**self.DEFAULTS, **(config or {}).get("random_generator", {})}
        super().__init__("random", {"random_generator": merged})
        self.config = merged
        self._recent_lock = threading.RLock()
        self.validator = validator or RandomXValidator((config or {}).get("randomx", {}))
        self.use_validator = True if self.validator else False
        self._recent: List[int] = []
        self._recent_size = int(self.config["recent_nonces_size"])
        self._executor = None
        self._executor_lock = threading.Lock()
        base_dir = Path(os.environ.get("IAZAR_BASE", "C:/zarturxia/src/iazar"))
        self.data_dir = base_dir / "data"
        self.writer = NonceCSVWriter(self.data_dir / "nonces_exitosos.csv") if self.config.get("write_direct", False) else None
        logger.info("[RandomGenerator] Initialized (workers=%d write_direct=%s)",
                    int(self.config["max_validation_workers"]), bool(self.writer))

    # ---------------- Public API ----------------
    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        t0 = time.perf_counter()
        batch_factor = int(self.config["batch_candidate_factor"])
        max_factor = int(self.config["max_candidate_factor"])
        max_loops = int(self.config["max_attempt_loops"])

        accepted: List[dict] = []
        attempts = 0
        while len(accepted) < batch_size and attempts < max_loops:
            candidate_count = batch_size * batch_factor
            candidates = self._generate_candidates(candidate_count)
            metrics = self._compute_metrics(candidates)
            mask = [
                (m["entropy"] >= self.config["min_entropy_threshold"]) and
                (m["pattern_score"] >= self.config["pattern_score_min"])
                for m in metrics
            ]
            filtered = [c for c, ok in zip(candidates, mask) if ok]
            filtered_metrics = [m for m, ok in zip(metrics, mask) if ok]
            if not filtered:
                batch_factor = min(batch_factor * 2, max_factor)
                attempts += 1
                continue
            val_flags, hashes = self._validate_vector(filtered, block_data)
            for i, ok in enumerate(val_flags):
                if not ok:
                    self.record_accept(False)
                    continue
                rec = {
                    "nonce": int(filtered[i]) & UINT32_MASK,
                    "entropy": round(filtered_metrics[i]["entropy"], 5),
                    "uniqueness": round(filtered_metrics[i]["uniqueness"], 5),
                    "zero_density": round(filtered_metrics[i]["zero_density"], 5),
                    "pattern_score": round(filtered_metrics[i]["pattern_score"], 5),
                    "is_valid": True,
                    "block_height": block_height
                }
                if hashes[i]:
                    rec["hash"] = hashes[i].hex()
                accepted.append(rec)
                self._append_recent(rec["nonce"])
                self.record_accept(True)
                if len(accepted) >= batch_size:
                    break
            if len(accepted) < batch_size:
                batch_factor = min(batch_factor * 2, max_factor)
            attempts += 1

        if accepted and self.writer:
            self.writer.write_many(accepted)

        elapsed = time.perf_counter() - t0
        logger.info("[RandomGenerator] block=%s accepted=%d/%d attempts=%d elapsed=%.3fs",
                    block_height, len(accepted), batch_size, attempts, elapsed)
        return accepted

    # ---------------- Candidate Generation ----------------
    def _generate_candidates(self, count: int):
        if _HAS_NUMPY:
            raw64 = np.random.randint(0, 2**64, size=int(count * 1.2), dtype=np.uint64)
            c32 = (raw64 & 0xFFFFFFFF).astype(np.uint32)
            uniq = np.unique(c32)
            if uniq.size < count:
                while uniq.size < count:
                    extra = (np.random.randint(0, 2**64, size=count - uniq.size, dtype=np.uint64) & 0xFFFFFFFF).astype(np.uint32)
                    uniq = np.unique(np.concatenate([uniq, extra]))
            return uniq[:count].tolist()
        # Fallback puro Python
        import secrets
        seen = set()
        out = []
        while len(out) < count:
            n = int.from_bytes(secrets.token_bytes(4), 'little')
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    # ---------------- Metrics ----------------
    def _compute_metrics(self, nonces: List[int]):
        metrics: List[Dict[str, float]] = []
        recent = self._recent_snapshot()
        for n in nonces:
            b = f"{n:032b}"
            zeros = b.count('0')
            p0 = zeros / 32.0
            p1 = 1.0 - p0
            if p0 in (0.0, 1.0):
                ent = 0.0
            else:
                ent = -(p0 * math.log2(p0) + p1 * math.log2(p1))
            max_run0 = max((len(r) for r in b.split('1')), default=0)
            max_run1 = max((len(r) for r in b.split('0')), default=0)
            run_penalty = min(0.5, max(max_run0, max_run1) / 16)
            pattern = max(0.5, 1.0 - run_penalty)
            uniq = self._uniqueness(n, recent)
            metrics.append({
                "entropy": ent,
                "zero_density": p0,
                "pattern_score": pattern,
                "uniqueness": uniq
            })
        return metrics

    def _uniqueness(self, n: int, recent: List[int]) -> float:
        if not recent:
            return 1.0
        subset = recent[-min(len(recent), 64):]
        diffs = 0
        for r in subset:
            diffs += bin((r ^ n) & UINT32_MASK).count("1")
        avg = diffs / (len(subset) * 32.0)
        return max(0.8, min(0.99, avg))

    # ---------------- Validation ----------------
    def _validate_vector(self, nonces: List[int], block_data: dict):
        if not self.use_validator or not self.validator:
            return [False]*len(nonces), [None]*len(nonces)
        from concurrent.futures import ThreadPoolExecutor
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=int(self.config["max_validation_workers"]),
                                                    thread_name_prefix="rng-val")
        futures = []
        for n in nonces:
            futures.append(self._executor.submit(self.validator.validate, int(n), block_data, True))
        results = []
        hashes: List[Optional[bytes]] = []
        for f in futures:
            try:
                res = f.result()
                if isinstance(res, tuple) and len(res) == 2:
                    ok, h = res
                    results.append(bool(ok))
                    hashes.append(h if (ok and isinstance(h, (bytes, bytearray))) else None)
                else:
                    results.append(bool(res))
                    hashes.append(None)
            except Exception:
                results.append(False)
                hashes.append(None)
        return results, hashes

    # ---------------- Recent Cache ----------------
    def _append_recent(self, nonce: int):
        with self._recent_lock:
            self._recent.append(nonce)
            if len(self._recent) > self._recent_size:
                self._recent.pop(0)

    def _recent_snapshot(self):
        with self._recent_lock:
            return list(self._recent)

    def close(self):
        with self._executor_lock:
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

def create_generator(config: Optional[Dict[str, Any]] = None,
                     validator: Optional[RandomXValidator] = None) -> RandomGenerator:
    return RandomGenerator(config=config, validator=validator)
