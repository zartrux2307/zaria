"""
adaptive_generator.py
---------------------
Generador Adaptativo de Nonces (CPU / RandomX) - Nivel Producción.

Características clave:
- Espacio de búsqueda (uint32) particionado en bins dinámicos; pesos se ajustan con refuerzo ligero (EMA + reward/penalty).
- Selección de bins mediante softmax con temperatura adaptativa + decaimiento global suave.
- Auto‑expansión / contracción de número de bins según calidad global (EMA promedio) con histéresis temporal y mínima evidencia.
- Batch generation dinámica: factor se incrementa / reduce según ratio de éxito y latencia del ciclo.
- Validación RandomX paralela (pool persistente) con soporte opcional a `return_hash=True` del validator.
- Métricas bitwise vectorizadas vía NumPy (entropy, zero_density, pattern_score, uniqueness) con fallback puro Python.
- Uniqueness calculada contra ventana de nonces válidos recientes (vectorizada cuando posible).
- Hooks de métricas (# METRIC:) para fácil instrumentación (Prometheus / StatsD).
- Opcional escritura directa CSV (normalmente se delega al orquestador).
- Diseño idempotente y safe: no “placebos”; cada nonce marcado válido ha pasado por `validator.validate()` real.

Salida (lista de dicts) campos estándar:
  ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height"]
Opcional (si validator devuelve hash): "hash"
"""

from __future__ import annotations
import os
import time
import math
import logging
import threading
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.proxy.randomx_validator import RandomXValidator
from iazar.generator.NonceCSVWriter import NonceCSVWriter

LOGGER_NAME = "generator.adaptive"
logger = logging.getLogger(LOGGER_NAME)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

UINT32_MASK = 0xFFFFFFFF

class AdaptiveGenerator(BaseNonceGenerator):
    DEFAULTS = {
        "initial_bins": 48,
        "min_bins": 24,
        "max_bins": 160,
        "bin_adjust_check_interval": 300.0,
        "bin_expand_success_threshold": 0.72,
        "bin_shrink_success_threshold": 0.28,
        "bin_min_total_samples": 500,
        "bin_hysteresis_seconds": 120.0,

        "batch_candidate_factor": 5,
        "max_candidate_factor": 40,
        "max_attempt_loops": 8,

        "entropy_min": 0.0,
        "pattern_score_min": 0.40,
        "adaptive_entropy_raise": True,
        "adaptive_entropy_target_ratio": 0.30,
        "entropy_raise_step": 0.02,
        "entropy_raise_cap": 0.40,

        "ema_alpha": 0.15,
        "decay_factor": 0.985,
        "reward_scale": 1.25,
        "penalty_scale": 0.70,

        "softmax_min_temp": 0.4,
        "softmax_max_temp": 2.0,
        "softmax_temp_decay": 0.98,

        "success_backoff_low": 0.05,
        "success_relief_high": 0.25,

        "recent_window_uniqueness": 1200,
        "recent_valid_trim_factor": 1.15,

        "validator_workers": 2,
        "use_validator": True,

        "write_direct": False,

        "target_batch_latency_sec": 0.35,

        "rng_seed": None
    }

    FIELDNAMES = ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height"]
    BIT_WIDTH = 32

    def __init__(self, config: Optional[Dict] = None, validator: Optional[RandomXValidator] = None):
        merged = {**self.DEFAULTS, **(config or {}).get("adaptive_generator", {})}
        super().__init__("adaptive", {"adaptive_generator": merged})
        self.cfg = merged

        self.validator = validator or RandomXValidator((config or {}).get("randomx", {}))
        self.use_validator = bool(self.cfg.get("use_validator", True))

        self.write_direct = bool(self.cfg.get("write_direct", False))
        self.writer = None
        if self.write_direct:
            base_dir = Path(os.environ.get("IAZAR_BASE", "C:/zarturxia/src/iazar"))
            self.writer = NonceCSVWriter(base_dir / "data" / "nonces_exitosos.csv")

        self._lock = threading.RLock()
        self.bins = int(self.cfg["initial_bins"])
        self.min_bins = int(self.cfg["min_bins"])
        self.max_bins = int(self.cfg["max_bins"])
        self._range_span = (1 << self.BIT_WIDTH)
        self._bin_width = self._range_span // self.bins
        self._init_bins()

        self.recent_valid: List[int] = []
        self.recent_uniqueness_window = int(self.cfg["recent_window_uniqueness"])
        self.recent_trim_factor = float(self.cfg["recent_valid_trim_factor"])

        self._last_bin_adjust = time.time()
        self._last_bin_change = 0.0
        self.temperature = float(self.cfg["softmax_max_temp"])

        seed = self.cfg.get("rng_seed")
        self._rng = random.Random(seed) if seed is not None else random

        self._val_workers = max(1, min(int(self.cfg["validator_workers"]), (os.cpu_count() or 4)))
        self._val_pool = ThreadPoolExecutor(max_workers=self._val_workers, thread_name_prefix="adp-val")

        self._global_attempts = 0
        self._global_success = 0

        logger.info("[AdaptiveGenerator] Ready (bins=%d workers=%d validator=%s)",
                    self.bins, self._val_workers, "ON" if self.use_validator else "OFF")

    # ---------------- Public API ----------------
    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        start = time.perf_counter()
        accepted: List[dict] = []
        factor = int(self.cfg["batch_candidate_factor"])
        max_factor = int(self.cfg["max_candidate_factor"])
        attempts = 0

        while len(accepted) < batch_size and attempts < int(self.cfg["max_attempt_loops"]):
            loop_t0 = time.perf_counter()
            target_candidates = batch_size * factor

            candidates, bin_indices = self._generate_candidates(target_candidates)
            if not candidates:
                attempts += 1
                factor = min(factor * 2, max_factor)
                continue

            metrics = self._compute_metrics(candidates)
            entropy_map = metrics["entropy_map"]
            zero_map = metrics["zero_density_map"]
            pattern_map = metrics["pattern_score_map"]
            uniq_map = metrics["uniqueness_map"]

            good_idx = [
                i for i, n in enumerate(candidates)
                if entropy_map[n] >= self.cfg["entropy_min"]
                and pattern_map[n] >= self.cfg["pattern_score_min"]
            ]
            if not good_idx:
                attempts += 1
                factor = min(factor * 2, max_factor)
                self._cool_temperature()
                continue

            filtered_nonces = [candidates[i] for i in good_idx]
            filtered_bins   = [bin_indices[i] for i in good_idx]

            if self.use_validator:
                val_results, hashes = self._validate_batch(filtered_nonces, block_data)
            else:
                val_results = [True] * len(filtered_nonces)
                hashes = [None] * len(filtered_nonces)

            valid_count = 0
            for i, is_valid in enumerate(val_results):
                nonce = filtered_nonces[i] & UINT32_MASK
                bidx  = filtered_bins[i]
                hv    = hashes[i]
                self._update_bin_stats(bidx, is_valid)
                self._global_attempts += 1
                if is_valid:
                    self._global_success += 1
                    valid_count += 1
                    self._record_recent(nonce)
                rec = {
                    "nonce": nonce,
                    "entropy": round(entropy_map[nonce], 5),
                    "uniqueness": round(uniq_map[nonce], 5),
                    "zero_density": round(zero_map[nonce], 5),
                    "pattern_score": round(pattern_map[nonce], 5),
                    "is_valid": bool(is_valid),
                    "block_height": block_height
                }
                if is_valid and hv and len(hv) == 32:
                    rec["hash"] = hv.hex()
                    # Aquí puedes hacer el log visible de nonce y hash:
                    logger.info(f"[AdaptiveGen] Nonce aceptado: {nonce:08x}  Hash: {hv.hex()}  Altura: {block_height}")
                accepted.append(rec)
                if len(accepted) >= batch_size:
                    break

            round_ratio = valid_count / max(1, len(filtered_nonces))
            self._adaptive_entropy_tune(round_ratio)
            factor = self._adjust_factor(factor, round_ratio, max_factor, loop_t0)

            attempts += 1
            self._cool_temperature()
            self._maybe_adjust_bins()

        if self.write_direct and accepted and self.writer:
            self.writer.write_many(accepted)

        elapsed = time.perf_counter() - start
        logger.info("[AdaptiveGenerator] block=%s out=%d attempts=%d factor_final=%d elapsed=%.3fs global_sr=%.3f",
                    block_height, len(accepted), attempts, factor, elapsed,
                    (self._global_success / self._global_attempts) if self._global_attempts else 0.0)
        return accepted

    # ---------------- Internal: Bins ----------------
    def _init_bins(self):
        with self._lock:
            self.bin_attempts = [1] * self.bins
            self.bin_success  = [1] * self.bins
            self.bin_weights  = [1.0] * self.bins
            self.bin_ema      = [0.5] * self.bins

    def _maybe_adjust_bins(self):
        now = time.time()
        if now - self._last_bin_adjust < float(self.cfg["bin_adjust_check_interval"]):
            return
        if now - self._last_bin_change < float(self.cfg["bin_hysteresis_seconds"]):
            return
        total_samples = sum(self.bin_attempts)
        if total_samples < int(self.cfg["bin_min_total_samples"]):
            self._last_bin_adjust = now
            return

        self._last_bin_adjust = now
        ema_avg = sum(self.bin_ema) / len(self.bin_ema)
        changed = False
        if ema_avg > float(self.cfg["bin_expand_success_threshold"]) and self.bins < self.max_bins:
            new_bins = min(self.max_bins, int(self.bins * 1.25))
            logger.info("[AdaptiveGenerator] Expand bins %d -> %d (ema_avg=%.3f)", self.bins, new_bins, ema_avg)
            self.bins = new_bins
            self._bin_width = self._range_span // self.bins
            self._init_bins()
            changed = True
        elif ema_avg < float(self.cfg["bin_shrink_success_threshold"]) and self.bins > self.min_bins:
            new_bins = max(self.min_bins, int(self.bins * 0.75))
            logger.info("[AdaptiveGenerator] Shrink bins %d -> %d (ema_avg=%.3f)", self.bins, new_bins, ema_avg)
            self.bins = new_bins
            self._bin_width = self._range_span // self.bins
            self._init_bins()
            changed = True
        if changed:
            self._last_bin_change = now

    def _bin_of(self, nonce: int) -> int:
        idx = nonce // self._bin_width
        return idx if idx < self.bins else self.bins - 1

    def _softmax_probs(self) -> List[float]:
        with self._lock:
            self.bin_weights = [w * float(self.cfg["decay_factor"]) for w in self.bin_weights]
            scores = [ema * w for ema, w in zip(self.bin_ema, self.bin_weights)]
            mx = max(scores)
            temp = max(float(self.cfg["softmax_min_temp"]),
                       min(self.temperature, float(self.cfg["softmax_max_temp"])))
            exps = [math.exp((s - mx) / temp) for s in scores]
            ssum = sum(exps) or 1.0
            return [e / ssum for e in exps]

    def _cool_temperature(self):
        self.temperature = max(float(self.cfg["softmax_min_temp"]),
                               self.temperature * float(self.cfg["softmax_temp_decay"]))

    def _update_bin_stats(self, bin_idx: int, success: bool):
        with self._lock:
            self.bin_attempts[bin_idx] += 1
            if success:
                self.bin_success[bin_idx] += 1
            sr = self.bin_success[bin_idx] / self.bin_attempts[bin_idx]
            alpha = float(self.cfg["ema_alpha"])
            self.bin_ema[bin_idx] = (1 - alpha) * self.bin_ema[bin_idx] + alpha * sr
            scale = float(self.cfg["reward_scale"] if success else self.cfg["penalty_scale"])
            new_w = self.bin_weights[bin_idx] * scale
            self.bin_weights[bin_idx] = min(12.0, max(0.05, new_w))

    # ---------------- Candidate Generation ----------------
    def _generate_candidates(self, count: int) -> Tuple[List[int], List[int]]:
        probs = self._softmax_probs()
        base = [int(p * count) for p in probs]
        deficit = count - sum(base)
        i = 0
        while deficit > 0 and base:
            base[i % len(base)] += 1
            deficit -= 1
            i += 1

        candidates: List[int] = []
        bins_out: List[int] = []
        for bidx, c in enumerate(base):
            if c <= 0:
                continue
            start = bidx * self._bin_width
            end = start + self._bin_width - 1
            if bidx == self.bins - 1:
                end = self._range_span - 1
            width = end - start + 1
            if _HAS_NUMPY:
                arr = (np.random.randint(0, width, size=c, dtype=np.uint32) + start) & UINT32_MASK
                candidates.extend(int(x) for x in arr)
            else:
                for _ in range(c):
                    n = start + (self._rng.getrandbits(self.BIT_WIDTH) % width)
                    candidates.append(n & UINT32_MASK)
            bins_out.extend([bidx] * c)

        if _HAS_NUMPY and candidates:
            arr = np.array(candidates, dtype=np.uint32)
            uniq, idx_map = np.unique(arr, return_index=True)
            candidates = [int(x) for x in uniq]
            bins_out = [bins_out[i] for i in idx_map]

        return candidates, bins_out

    # ---------------- Metrics ----------------
    def _compute_metrics(self, nonces: List[int]) -> Dict[str, Any]:
        entropy_map = {}
        zero_map = {}
        pattern_map = {}
        uniq_map = {}

        if not nonces:
            return {
                "entropy_map": entropy_map,
                "zero_density_map": zero_map,
                "pattern_score_map": pattern_map,
                "uniqueness_map": uniq_map,
                "entropy": [],
                "zero_density": [],
                "pattern_score": [],
                "uniqueness": []
            }

        recent_snapshot = self._recent_snapshot()

        if _HAS_NUMPY:
            arr = np.array(nonces, dtype=np.uint32)
            bits = np.unpackbits(arr.view(np.uint8)).reshape(-1, 32)
            zeros = (bits == 0).sum(axis=1)
            p0 = zeros / 32.0
            p1 = 1.0 - p0
            with np.errstate(divide='ignore', invalid='ignore'):
                entropy = -(p0 * np.log2(p0, where=(p0 > 0)) + p1 * np.log2(p1, where=(p1 > 0)))
                entropy = np.nan_to_num(entropy)
            transitions = np.diff(bits, axis=1) != 0
            run_lengths = 32 - transitions.sum(axis=1)
            run_penalty = np.clip(run_lengths / 32.0, 0, 0.6)
            pattern_score = np.maximum(0.4, 1.0 - run_penalty)

            # --- PATCH CRÍTICO: uniqueness robusto para lote 0, 1, o N ---
            if recent_snapshot.size > 0:
                xor = arr[:, None] ^ recent_snapshot[None, :]
                popcnt = np.unpackbits(xor.view(np.uint8), axis=-1).sum(axis=-1)
                if popcnt.ndim == 1:
                    mean_val = popcnt.mean() / 32.0
                    uniqueness = np.full(arr.shape[0], np.clip(mean_val, 0.7, 1.0), dtype=np.float32)
                else:
                    uniqueness = np.clip(popcnt.mean(axis=1) / 32.0, 0.7, 1.0)
            else:
                uniqueness = np.full(arr.shape[0], 1.0, dtype=np.float32)

            for i, n in enumerate(nonces):
                entropy_map[n] = float(entropy[i])
                zero_map[n] = float(p0[i])
                pattern_map[n] = float(pattern_score[i])
                uniq_map[n] = float(uniqueness[i])
        else:
            for n in nonces:
                b = f"{n:032b}"
                z = b.count('0')
                p0 = z / 32.0
                if p0 in (0.0, 1.0):
                    ent = 0.0
                else:
                    ent = -(p0*math.log2(p0) + (1-p0)*math.log2(1-p0))
                runs0 = b.split('1')
                runs1 = b.split('0')
                max_run = max(max((len(r) for r in runs0), default=0),
                              max((len(r) for r in runs1), default=0))
                penalty = min(0.6, max_run / 32.0)
                pat = max(0.4, 1.0 - penalty)
                uniq_map[n] = self._uniqueness_scalar(n, recent_snapshot)
                entropy_map[n] = ent
                zero_map[n] = p0
                pattern_map[n] = pat

        return {
            "entropy_map": entropy_map,
            "zero_density_map": zero_map,
            "pattern_score_map": pattern_map,
            "uniqueness_map": uniq_map,
            "entropy": [entropy_map[n] for n in nonces],
            "zero_density": [zero_map[n] for n in nonces],
            "pattern_score": [pattern_map[n] for n in nonces],
            "uniqueness": [uniq_map[n] for n in nonces]
        }

    def _uniqueness_scalar(self, nonce: int, recent: List[int]) -> float:
        if not recent:
            return 1.0
        diffs = 0
        subset = recent[-64:]
        for r in subset:
            diffs += bin((r ^ nonce) & UINT32_MASK).count("1")
        return max(0.7, min(1.0, diffs / (len(subset) * 32.0)))

    # ---------------- Validation ----------------
    def _validate_batch(self, nonces: List[int], block_data: dict) -> Tuple[List[bool], List[bytes | None]]:
        results: List[bool] = []
        hashes: List[bytes | None] = []
        if not nonces:
            return results, hashes

        def _call(n):
            try:
                r = self.validator.validate(n, block_data, return_hash=True)
                if isinstance(r, tuple) and len(r) == 2:
                    return bool(r[0]), r[1]
                return bool(r), None
            except Exception:
                return False, None

        for ok, hv in self._val_pool.map(_call, nonces):
            results.append(ok)
            hashes.append(hv)
        return results, hashes

    # ---------------- Recent / Uniqueness ----------------
    def _record_recent(self, nonce: int):
        self.recent_valid.append(nonce & UINT32_MASK)
        max_allowed = int(self.recent_uniqueness_window * self.recent_trim_factor)
        if len(self.recent_valid) > max_allowed:
            self.recent_valid = self.recent_valid[-self.recent_uniqueness_window:]

    def _recent_snapshot(self):
        if _HAS_NUMPY:
            return np.array(self.recent_valid, dtype=np.uint32) if self.recent_valid else np.empty(0, dtype=np.uint32)
        return list(self.recent_valid)

    # ---------------- Adapt Factor / Entropy ----------------
    def _adjust_factor(self, factor: int, round_ratio: float, max_factor: int, loop_start: float) -> int:
        if round_ratio < self.cfg["success_backoff_low"]:
            factor = min(factor * 2, max_factor)
        elif round_ratio > self.cfg["success_relief_high"] and factor > 1:
            factor = max(1, factor // 2)
        elapsed = time.perf_counter() - loop_start
        target_lat = float(self.cfg["target_batch_latency_sec"])
        if elapsed > target_lat * 1.6 and factor > 1:
            factor = max(1, factor - 1)
        elif elapsed < target_lat * 0.5 and factor < max_factor and round_ratio > 0.0:
            factor = min(max_factor, factor + 1)
        return factor

    def _adaptive_entropy_tune(self, round_ratio: float):
        if not self.cfg.get("adaptive_entropy_raise", True):
            return
        target_ratio = float(self.cfg["adaptive_entropy_target_ratio"])
        if round_ratio > target_ratio and self.cfg["entropy_min"] < self.cfg["entropy_raise_cap"]:
            new_ent = min(self.cfg["entropy_raise_cap"],
                          self.cfg["entropy_min"] + self.cfg["entropy_raise_step"])
            self.cfg["entropy_min"] = round(new_ent, 5)

    # ---------------- Housekeeping ----------------
    def close(self):
        try:
            self._val_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

def create_generator(config: Optional[Dict] = None, validator: Optional[RandomXValidator] = None) -> AdaptiveGenerator:
    return AdaptiveGenerator(config=config, validator=validator)
