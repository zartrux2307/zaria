from __future__ import annotations
"""
range_based_generator.py
------------------------
Generador basado en rangos “calientes” históricos para nonces RandomX.

Estrategia:
- Mantiene rangos preferentes (ventanas) con mayor ratio histórico de éxito.
- Roulette wheel ponderada por (success+1)/(attempts+2) para exploración suave (Laplace smoothing).
- Sobre-muestreo (oversampling) adaptativo para compensar rechazos.
- Validación RandomX paralela (return_hash=True) → cada registro válido incluye hash real (en memoria).
- Métricas bitwise (entropy, zero_density, pattern_score, uniqueness) vectorizadas con NumPy cuando posible.
- Lectura incremental de nonces válidos recientes desde su propio CSV (opcional) o desde el CSV central.
- Preparado para delegar persistencia al orquestador (por defecto `write_direct=False`).

Salida estándar (lista de dicts) por ciclo:
  ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height","hash?"]

IMPORTANTE:
El CSV “oficial” (`nonces_exitosos.csv`) mantiene **solo 7 columnas** (sin hash) según tu formato global.
El campo "hash" queda disponible en memoria para el orquestador / proxy.

Config esperada (sección `range_generator` en global_config.json opcional):
{
  "range_success_windows": [
      [1000000, 2000000],
      [5000000, 6000000]
  ],
  "range_over_sampling": 3.5,
  "write_direct": false
}
"""

import os
import csv
import time
import math
import random
import logging
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Iterable, Tuple, Any

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None      # type: ignore
    _HAS_NUMPY = False

from concurrent.futures import ThreadPoolExecutor, as_completed

from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.proxy.randomx_validator import RandomXValidator
from iazar.generator.NonceCSVWriter import NonceCSVWriter

__all__ = ["RangeBasedGenerator"]

CSV_FIELDS = [
    "nonce", "entropy", "uniqueness", "zero_density", "pattern_score", "is_valid", "block_height"
]

LOGGER_NAME = "generator.range"
logger = logging.getLogger(LOGGER_NAME)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class RangeBasedGenerator(BaseNonceGenerator):
    MAX_NONCE = 0xFFFFFFFF
    RECENT_NONCES_TARGET = 500
    DEFAULT_RANGES: List[Tuple[int, int]] = [
        (1_000_000, 2_000_000),
        (5_000_000, 6_000_000),
        (8_000_000, 9_000_000),
    ]
    MAX_VALIDATION_WORKERS = max(2, min(16, os.cpu_count() or 4))
    MIN_OVER_SAMPLING = 2.0
    MAX_OVER_SAMPLING = 6.0

    def __init__(self,
                 config: Optional[Dict] = None,
                 *,
                 data_dir: Optional[Path] = None,
                 validator: Optional[RandomXValidator] = None,
                 executor: Optional[ThreadPoolExecutor] = None):

        super().__init__("range", {"range_generator": (config or {}).get("range_generator", {})})
        cfg_section = (config or {}).get("range_generator", {})

        base_dir = Path(os.environ.get("IAZAR_BASE", "C:/zarturxia/src/iazar"))
        self.data_dir = data_dir or (base_dir / "data")

        # CSV local (solo si write_direct=True; normalmente False para delegar al orquestador)
        self.output_path = self.data_dir / "range_generated_nonces.csv"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.validator = validator or RandomXValidator((config or {}).get("randomx", {}))
        self._ext_executor = executor
        self._lock = threading.RLock()
        self._rng = random.SystemRandom()

        # Rango inicial
        self._ranges: List[Tuple[int, int]] = self._load_ranges_from_config(cfg_section)
        if not self._ranges:
            self._ranges = self.DEFAULT_RANGES.copy()
        self._range_stats = {r: {"attempts": 0, "success": 0} for r in self._ranges}

        # Oversampling
        raw_os = cfg_section.get("range_over_sampling", 3.0)
        self._over_sampling = float(min(self.MAX_OVER_SAMPLING, max(self.MIN_OVER_SAMPLING, raw_os)))

        # Persistencia directa opcional
        self.write_direct = bool(cfg_section.get("write_direct", False))
        self.writer = None
        if self.write_direct:
            self.writer = NonceCSVWriter(self.data_dir / "nonces_exitosos.csv")

        logger.info("RangeBasedGenerator inicializado | ranges=%s oversampling=%.2f write_direct=%s",
                    self._ranges, self._over_sampling, self.write_direct)

    # -------- Config / Ranges --------
    def _load_ranges_from_config(self, section_cfg: Dict) -> List[Tuple[int, int]]:
        ranges_cfg = section_cfg.get("range_success_windows")
        parsed: List[Tuple[int, int]] = []
        if not ranges_cfg:
            return parsed
        for item in ranges_cfg:
            try:
                a, b = int(item[0]), int(item[1])
                if 0 <= a <= b <= self.MAX_NONCE:
                    parsed.append((a, b))
            except Exception:
                continue
        return parsed

    # -------- Public API --------
    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        t0 = time.perf_counter()

        # Cargar recientes válidos (solo si se usa write_direct; si no, esto podría reconfigurarse a lectura global)
        with self._lock:
            recent_valid = self._load_recent_valid(limit=self.RECENT_NONCES_TARGET)

        # Generar candidatos con oversampling
        target_candidates = int(batch_size * self._over_sampling)
        candidates = self._generate_candidates(target_candidates)

        # Deduplicación intra-lote
        if candidates:
            if _HAS_NUMPY:
                arr = np.array(candidates, dtype=np.uint32)
                uniq, idx_map = np.unique(arr, return_index=True)
                candidates = [int(x) for x in uniq]
            else:
                seen = set()
                dedup = []
                for n in candidates:
                    if n not in seen:
                        seen.add(n)
                        dedup.append(n)
                candidates = dedup

        validation = self._validate_parallel(candidates, block_data)  # List[(ok, hash_bytes)]

        accepted: List[dict] = []
        if candidates:
            # Métricas vectorizadas
            metrics_all = self._compute_metrics_vectorized(candidates, recent_valid)

            for i, (ok, h) in enumerate(validation):
                if not ok:
                    continue
                nonce = candidates[i]
                m = metrics_all[i]
                record = {
                    "nonce": nonce,
                    "entropy": m["entropy"],
                    "uniqueness": m["uniqueness"],
                    "zero_density": m["zero_density"],
                    "pattern_score": m["pattern_score"],
                    "is_valid": True,
                    "block_height": block_height,
                    "hash": h.hex() if h and len(h) == 32 else None
                }
                accepted.append(record)
                recent_valid.add(nonce)
                if len(accepted) >= batch_size:
                    break

        # Persistencia local (solo si modo directo)
        if self.write_direct and accepted:
            if self.writer:
                # Escribe SOLO las 7 columnas estándar
                plain_rows = [{
                    "nonce": r["nonce"],
                    "entropy": r["entropy"],
                    "uniqueness": r["uniqueness"],
                    "zero_density": r["zero_density"],
                    "pattern_score": r["pattern_score"],
                    "is_valid": r["is_valid"],
                    "block_height": r["block_height"]
                } for r in accepted]
                self.writer.write_many(plain_rows)
            else:
                self._persist_local_csv(accepted)

        elapsed = time.perf_counter() - t0
        rate = (len(accepted) / elapsed) if elapsed else 0.0
        self._tune_over_sampling(len(accepted), batch_size, elapsed)

        logger.info("[RangeBasedGenerator] block=%s accepted=%d target=%d elapsed=%.3fs rate=%.1f/s oversampling=%.2f",
                    block_height, len(accepted), batch_size, elapsed, rate, self._over_sampling)
        # METRIC: range_generator_batch_latency.observe(elapsed)
        return accepted

    # -------- Candidate Generation --------
    def _pick_range_weighted(self) -> Tuple[int, int]:
        weights = []
        for r in self._ranges:
            stats = self._range_stats[r]
            w = (stats["success"] + 1) / (stats["attempts"] + 2)
            weights.append(w)
        total = sum(weights) or 1.0
        rpick = self._rng.random() * total
        acc = 0.0
        for r, w in zip(self._ranges, weights):
            acc += w
            if rpick <= acc:
                return r
        return self._ranges[-1]

    def _generate_candidates(self, n: int) -> List[int]:
        out: List[int] = []
        for _ in range(n):
            a, b = self._pick_range_weighted()
            val = self._rng.randint(a, b) & self.MAX_NONCE
            out.append(val)
            self._range_stats[(a, b)]["attempts"] += 1
        return out

    # -------- Validation --------
    def _validate_parallel(self, nonces: List[int], block_data: dict) -> List[Tuple[bool, bytes]]:
        if not nonces:
            return []
        executor = self._ext_executor or ThreadPoolExecutor(
            max_workers=self.MAX_VALIDATION_WORKERS,
            thread_name_prefix="range-val"
        )
        results: List[Tuple[bool, bytes]] = [(False, b"")] * len(nonces)
        try:
            futures = {executor.submit(self.validator.validate, n, block_data, True): i
                       for i, n in enumerate(nonces)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                    if isinstance(res, tuple) and len(res) == 2:
                        ok, h = res
                        if not isinstance(h, (bytes, bytearray)):
                            h = b""
                        results[idx] = (bool(ok), bytes(h))
                    else:
                        results[idx] = (bool(res), b"")
                except Exception as e:
                    logger.debug("Validation error nonce=%d err=%s", nonces[idx], e)
                    results[idx] = (False, b"")
                    continue
                if results[idx][0]:
                    n = nonces[idx]
                    for r in self._ranges:
                        if r[0] <= n <= r[1]:
                            self._range_stats[r]["success"] += 1
                            break
        finally:
            if self._ext_executor is None:
                executor.shutdown(wait=False, cancel_futures=True)
        return results

    # -------- Metrics (vectorized) --------
    def _compute_metrics_vectorized(self, nonces: List[int], recent: Set[int]) -> List[Dict[str, float]]:
        if not nonces:
            return []
        if _HAS_NUMPY:
            arr = np.array(nonces, dtype=np.uint32)
            bits = np.unpackbits(arr.view(np.uint8)).reshape(-1, 32)
            ones = bits.sum(axis=1)
            zeros = 32 - ones
            p0 = zeros / 32.0
            p1 = ones / 32.0
            with np.errstate(divide='ignore', invalid='ignore'):
                entropy = -(p0 * np.log2(p0, where=(p0 > 0)) + p1 * np.log2(p1, where=(p1 > 0)))
                entropy = np.nan_to_num(entropy)

            # Longest run
            longest = []
            for row in bits:
                max_run = 1
                cur = 1
                for i in range(1, 32):
                    if row[i] == row[i-1]:
                        cur += 1
                        if cur > max_run:
                            max_run = cur
                    else:
                        cur = 1
                longest.append(max_run)
            longest = np.array(longest, dtype=np.int16)
            run_pen = np.minimum(0.3, longest / 32.0)
            transitions = np.count_nonzero(np.diff(bits, axis=1), axis=1)
            trans_bonus = np.minimum(0.2, transitions / 31.0)
            pattern_score = np.maximum(0.5, 1.0 - run_pen + trans_bonus)

            # Uniqueness
            uniq = np.ones(arr.shape[0], dtype=np.float32)
            if recent:
                # Convertimos solo una vez
                recent_arr = np.fromiter(recent, dtype=np.uint32)
                xor_mat = arr[:, None] ^ recent_arr[None, :]
                popcnt = np.unpackbits(xor.view(np.uint8), axis=-1).sum(axis=-1)
                uniq = np.clip(pop.mean(axis=1) / 32.0, 0.8, 0.99)

            out: List[Dict[str, float]] = []
            for i in range(arr.shape[0]):
                out.append({
                    "entropy": round(float(entropy[i]), 5),
                    "uniqueness": round(float(uniq[i]), 5),
                    "zero_density": round(float(p0[i]), 5),
                    "pattern_score": round(float(pattern_score[i]), 5)
                })
            return out

        # Fallback Python
        recent_list = list(recent)[-256:] if recent else []
        out: List[Dict[str, float]] = []
        for n in nonces:
            b = f"{n:032b}"
            zeros = b.count('0')
            p0 = zeros / 32.0
            p1 = 1.0 - p0
            ent = 0.0 if p0 in (0.0, 1.0) else -(p0 * math.log2(p0) + p1 * math.log2(p1))
            # Longest run
            max_run = 1
            cur = 1
            for i in range(1, 32):
                if b[i] == b[i-1]:
                    cur += 1
                    if cur > max_run:
                        max_run = cur
                else:
                    cur = 1
            run_pen = min(0.3, max_run / 32.0)
            transitions = sum(1 for i in range(1, 32) if b[i] != b[i-1])
            trans_bonus = min(0.2, transitions / 31.0)
            pattern = max(0.5, 1.0 - run_pen + trans_bonus)
            # uniqueness
            if recent_list:
                diffs = 0
                for r in recent_list:
                    diffs += bin((r ^ n) & self.MAX_NONCE).count("1")
                avg = diffs / (len(recent_list) * 32.0)
                uq = min(0.99, max(0.8, avg))
            else:
                uq = 1.0
            out.append({
                "entropy": round(ent, 5),
                "uniqueness": round(uq, 5),
                "zero_density": round(p0, 5),
                "pattern_score": round(pattern, 5)
            })
        return out

    # -------- Persistence (local fallback) --------
    def _persist_local_csv(self, rows: List[Dict[str, Any]]):
        if not rows:
            return
        new_file = not self.output_path.exists()
        try:
            with self.output_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if new_file:
                    writer.writeheader()
                safe_rows = []
                for r in rows:
                    safe = {k: r.get(k) for k in CSV_FIELDS}
                    safe_rows.append(safe)
                writer.writerows(safe_rows)
        except Exception as e:
            logger.error("Persist error path=%s err=%s", self.output_path, e)

    # -------- Load Recent Valid (tail) --------
    def _load_recent_valid(self, limit: int) -> Set[int]:
        """
        Lee nonces válidos recientes desde:
          - CSV local (si write_direct) O
          - (Extensible: leer 'nonces_exitosos.csv' global).
        """
        recent: Set[int] = set()
        path = self.writer.path if (self.write_direct and self.writer) else self.output_path
        if not path.exists():
            return recent
        try:
            with path.open("rb") as fb:
                fb.seek(0, os.SEEK_END)
                pos = fb.tell()
                buffer = b""
                block_size = 4096
                while pos > 0 and len(recent) < limit:
                    read_size = min(block_size, pos)
                    pos -= read_size
                    fb.seek(pos)
                    chunk = fb.read(read_size)
                    buffer = chunk + buffer
                    parts = buffer.split(b"\n")
                    buffer = parts[0]
                    for ln in reversed(parts[1:]):
                        if len(recent) >= limit:
                            break
                        if not ln:
                            continue
                        try:
                            s = ln.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            continue
                        if not s or s.startswith("nonce"):
                            continue
                        cols = s.split(",")
                        if len(cols) < 7:
                            continue
                        is_valid = cols[5].lower() == "true"
                        if not is_valid:
                            continue
                        try:
                            nonce_val = int(cols[0]) & self.MAX_NONCE
                        except Exception:
                            continue
                        recent.add(nonce_val)
                # Procesar buffer restante (primera línea potencial)
                if buffer and len(recent) < limit:
                    try:
                        s = buffer.decode("utf-8").strip()
                        if s and not s.startswith("nonce"):
                            cols = s.split(",")
                            if len(cols) >= 7 and cols[5].lower() == "true":
                                nonce_val = int(cols[0]) & self.MAX_NONCE
                                recent.add(nonce_val)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("Recent load error err=%s", e)
        return recent

    # -------- Oversampling Adaptation --------
    def _tune_over_sampling(self, produced: int, target: int, elapsed: float):
        if target <= 0:
            return
        ratio = produced / target
        if ratio < 0.8:
            self._over_sampling = min(self._over_sampling * 1.12, self.MAX_OVER_SAMPLING)
        elif ratio > 1.1:
            self._over_sampling = max(self._over_sampling * 0.90, self.MIN_OVER_SAMPLING)
        if elapsed > 1.0:
            self._over_sampling = max(self.MIN_OVER_SAMPLING, self._over_sampling * 0.92)
        # METRIC: range_generator_oversampling.set(self._over_sampling)

    # -------- Hot Reload --------
    def update_ranges(self, new_ranges: Iterable[Tuple[int, int]]):
        with self._lock:
            valid = []
            for a, b in new_ranges:
                a, b = int(a), int(b)
                if 0 <= a <= b <= self.MAX_NONCE:
                    valid.append((a, b))
            if valid:
                self._ranges = valid
                self._range_stats = {r: {"attempts": 0, "success": 0} for r in self._ranges}
                logger.info("[RangeBasedGenerator] ranges updated=%s", self._ranges)

    # -------- Shutdown --------
    def close(self):
        pass


def create_generator(config: Optional[Dict] = None,
                     validator: Optional[RandomXValidator] = None) -> RangeBasedGenerator:
    return RangeBasedGenerator(config=config, validator=validator)
