from __future__ import annotations
"""
sequence_based_generator.py (patched)
------------------------------------
Generador basado en secuencias pseudo‑aleatorias (LCG / XorShift / PCG) con
bandit de pesos adaptativos y soporte de validación RandomX (hash real).

Mejoras en este patch:
- Añadido recent_accept_ratio() para compatibilidad con orquestador.
- Uso de record_accept(True/False) en cada validación.
- Protección contra divisiones por cero en normalización de pesos.
- Factor interno calculado correctamente dentro del método (sin usar self fuera).
- Devuelve registros con campo "hash" si validator retorna bytes.
"""

import os
import time
import math
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore

from concurrent.futures import ThreadPoolExecutor, as_completed

from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.proxy.randomx_validator import RandomXValidator

LOGGER_NAME = "generator.sequence"
logger = logging.getLogger(LOGGER_NAME)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

UINT32_MASK = 0xFFFFFFFF
CSV_FIELDS = ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height"]

class SequenceBasedGenerator(BaseNonceGenerator):
    NAME = "sequence"

    DEFAULTS = {
        "initial_weights": {"lcg": 0.4, "xorshift": 0.3, "pcg": 0.3},
        "min_entropy_batch": 4.8,
        "degeneration_threshold": 4.2,
        "weight_adjust_alpha": 0.06,
        "internal_factor": 3,
        "max_internal_factor": 6,
        "recent_uniqueness_window": 1500,
        "use_validator": True,
        "validator_workers": 8
    }

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 base_dir: Optional[Path] = None,
                 validator: Optional[RandomXValidator] = None):
        merged = {**self.DEFAULTS, **(config or {}).get("sequence_generator", {})}
        super().__init__("sequence", {"sequence_generator": merged})
        self.cfg = merged
        self.base_dir = Path(base_dir or os.environ.get("IAZAR_BASE", "C:/zarturxia/src/iazar"))
        self.data_dir = self.base_dir / "data"
        self.use_validator = bool(self.cfg.get("use_validator", True))
        self.validator = validator or (RandomXValidator((config or {}).get("randomx", {}))
                                       if self.use_validator else None)
        self.use_validator = bool(self.use_validator and self.validator)

        wcfg = self.cfg.get("initial_weights", {})
        self.weights = {
            "lcg": float(wcfg.get("lcg", 0.4)),
            "xorshift": float(wcfg.get("xorshift", 0.3)),
            "pcg": float(wcfg.get("pcg", 0.3))
        }
        # Normaliza
        self._normalize_weights()
        self.alpha = float(self.cfg["weight_adjust_alpha"])
        self.recent_valid: List[int] = []
        self.recent_window = int(self.cfg["recent_uniqueness_window"])
        self._val_pool: Optional[ThreadPoolExecutor] = None
        self._val_pool_lock = threading.Lock()

        logger.info("[sequence] Initialized (validator=%s) weights=%s",
                    "ON" if self.use_validator else "OFF", self.weights)

    # --------------- Internal sequences ---------------
    def _seq_lcg(self, seed: int, count: int) -> List[int]:
        a = 1664525
        c = 1013904223
        m = 2**32
        x = seed & UINT32_MASK
        out = []
        for _ in range(count):
            x = (a * x + c) % m
            out.append(x)
        return out

    def _seq_xorshift(self, seed: int, count: int) -> List[int]:
        x = (seed | 1) & UINT32_MASK
        out = []
        for _ in range(count):
            x ^= (x << 13) & UINT32_MASK
            x ^= (x >> 17) & UINT32_MASK
            x ^= (x << 5) & UINT32_MASK
            out.append(x & UINT32_MASK)
        return out

    def _seq_pcg(self, seed: int, count: int) -> List[int]:
        state = seed & UINT32_MASK
        inc = 0x5851F42D
        out = []
        for _ in range(count):
            old = state
            state = (old * 747796405 + inc) & UINT32_MASK
            xorshifted = (((old >> 18) ^ old) >> 27) & UINT32_MASK
            rot = old >> 27
            val = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & UINT32_MASK
            out.append(val)
        return out

    def _normalize_weights(self):
        s = sum(self.weights.values())
        if s <= 0:
            self.weights = {k: 1.0 / len(self.weights) for k in self.weights}
        else:
            for k in self.weights:
                self.weights[k] = self.weights[k] / s

    # --------------- Metrics ---------------
    @staticmethod
    def _entropy(n: int) -> float:
        b = f"{n:032b}"
        ones = b.count("1")
        p = ones / 32.0
        if p in (0.0, 1.0):
            return 0.0
        return -(p*math.log2(p) + (1-p)*math.log2(1-p))

    @staticmethod
    def _zero_density(n: int) -> float:
        b = f"{n:032b}"
        return b.count("0") / 32.0

    @staticmethod
    def _pattern_score(n: int) -> float:
        b = f"{n:032b}"
        max_run = 1
        cur = 1
        for i in range(1, 32):
            if b[i] == b[i-1]:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 1
        penalty = min(0.5, max_run / 32.0)
        return max(0.4, 1.0 - penalty)

    def _uniqueness(self, n: int) -> float:
        if not self.recent_valid:
            return 1.0
        sample = self.recent_valid[-min(len(self.recent_valid), 256):]
        diffs = 0
        for r in sample:
            diffs += bin((r ^ n) & UINT32_MASK).count("1")
        avg = diffs / (len(sample) * 32.0)
        return min(0.99, max(0.7, avg))

    # --------------- Validation ---------------
    def _ensure_pool(self):
        if not self.use_validator:
            return
        with self._val_pool_lock:
            if self._val_pool is None:
                workers = min(int(self.cfg["validator_workers"]), os.cpu_count() or 4)
                self._val_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="seq-val")

    def _validate_batch(self, nonces: List[int], block_data: Dict[str, Any]) -> Tuple[List[bool], List[Optional[bytes]]]:
        if not self.use_validator or not self.validator:
            return ([False] * len(nonces)), [None] * len(nonces)
        self._ensure_pool()
        assert self._val_pool is not None
        futures = {self._val_pool.submit(self.validator.validate, int(n), block_data, True): i
                   for i, n in enumerate(nonces)}
        results = [False] * len(nonces)
        hashes: List[Optional[bytes]] = [None] * len(nonces)
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
                if isinstance(res, tuple) and len(res) == 2:
                    ok, h = res
                    results[idx] = bool(ok)
                    if ok and isinstance(h, (bytes, bytearray)):
                        hashes[idx] = bytes(h)
                else:
                    results[idx] = bool(res)
            except Exception as e:  # pragma: no cover
                logger.debug("[sequence] validate error idx=%d err=%s", idx, e)
        return results, hashes

    # --------------- Core Generation ---------------
    def run_generation(self, block_height: int, block_data: Dict[str, Any], batch_size: int = 500) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()

        base_internal = int(self.cfg.get("internal_factor", 3))
        max_internal = int(self.cfg.get("max_internal_factor", 6))
        internal_factor = max(1, min(base_internal, max_internal))

        # Distribuir candidatos entre secuencias según pesos
        self._normalize_weights()
        total_weight = sum(self.weights.values()) or 1.0
        seq_counts = {}
        remaining = batch_size * internal_factor
        for name, w in self.weights.items():
            c = int((w / total_weight) * batch_size * internal_factor)
            seq_counts[name] = c
            remaining -= c
        # repartir restante
        keys = list(self.weights.keys())
        ki = 0
        while remaining > 0:
            k = keys[ki % len(keys)]
            seq_counts[k] += 1
            remaining -= 1
            ki += 1

        seed_base = int(time.time_ns() & UINT32_MASK)
        seq_outputs: List[int] = []
        seq_origin: List[str] = []

        # Generar
        for name, count in seq_counts.items():
            if count <= 0:
                continue
            seed = (seed_base ^ hash(name)) & UINT32_MASK
            if name == "lcg":
                seq = self._seq_lcg(seed, count)
            elif name == "xorshift":
                seq = self._seq_xorshift(seed, count)
            else:
                seq = self._seq_pcg(seed, count)
            seq_outputs.extend(seq)
            seq_origin.extend([name] * len(seq))

        # Métricas y filtrado ligero
        entropies = [self._entropy(n) for n in seq_outputs]
        zdens = [self._zero_density(n) for n in seq_outputs]
        patterns = [self._pattern_score(n) for n in seq_outputs]
        uniqs = [self._uniqueness(n) for n in seq_outputs]

        # Degeneración: si media de entropía demasiado baja subir internal_factor (siguiente lote)
        avg_ent = sum(entropies) / len(entropies) if entropies else 0.0
        if avg_ent < float(self.cfg["degeneration_threshold"]):
            self.cfg["internal_factor"] = min(max_internal, base_internal + 1)

        # Selección top por entropía + pattern
        scored_idx = list(range(len(seq_outputs)))
        scored_idx.sort(key=lambda i: (entropies[i] + patterns[i]), reverse=True)
        picked_idx = scored_idx[:batch_size]

        chosen_nonces = [seq_outputs[i] for i in picked_idx]
        chosen_origin = [seq_origin[i] for i in picked_idx]
        chosen_entropy = [entropies[i] for i in picked_idx]
        chosen_zd = [zdens[i] for i in picked_idx]
        chosen_pattern = [patterns[i] for i in picked_idx]
        chosen_uniq = [uniqs[i] for i in picked_idx]

        val_flags, hashes = self._validate_batch(chosen_nonces, block_data)

        records: List[Dict[str, Any]] = []
        for i, n in enumerate(chosen_nonces):
            ok = bool(val_flags[i])
            rec = {
                "nonce": int(n) & UINT32_MASK,
                "entropy": round(chosen_entropy[i], 5),
                "uniqueness": round(chosen_uniq[i], 5),
                "zero_density": round(chosen_zd[i], 5),
                "pattern_score": round(chosen_pattern[i], 5),
                "is_valid": ok,
                "block_height": block_height
            }
            if ok and hashes[i]:
                rec["hash"] = hashes[i].hex()
                self.recent_valid.append(rec["nonce"])
            # Bandit update
            self._update_weight(chosen_origin[i], ok)
            self.record_accept(ok)
            records.append(rec)

        # Trim recent
        if len(self.recent_valid) > self.recent_window * 1.3:
            self.recent_valid = self.recent_valid[-self.recent_window:]

        elapsed = time.perf_counter() - t0
        logger.debug("[sequence] block=%s out=%d entropy_avg=%.3f latency=%.3fs",
                     block_height, len(records), avg_ent, elapsed)
        return records

    # --------------- Bandit Weight Update ---------------
    def _update_weight(self, name: str, accepted: bool):
        # Pequeña actualización tipo gradient / EMA
        delta = self.alpha if accepted else -self.alpha
        self.weights[name] = max(0.01, self.weights[name] + delta)
        self._normalize_weights()

    # --------------- Compatibility Hook ---------------
    def recent_accept_ratio(self) -> float:
        # Asegura compatibilidad con orquestador
        return super().recent_accept_ratio()

    # --------------- Shutdown ---------------
    def close(self):
        if self._val_pool:
            self._val_pool.shutdown(wait=False, cancel_futures=True)

def create_generator(config: Optional[Dict[str, Any]] = None,
                     base_dir: Optional[Path] = None,
                     validator: Optional[RandomXValidator] = None) -> SequenceBasedGenerator:
    return SequenceBasedGenerator(config=config, base_dir=base_dir, validator=validator)
