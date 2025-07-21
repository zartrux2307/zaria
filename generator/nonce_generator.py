from __future__ import annotations
"""
nonce_generator.py
------------------
BaseNonceGenerator + GroupedTopNonceGenerator (bandit adaptativo) **versión producción**.

Características clave:
- Clase base con utilidades bitwise, unicidad y ratio aceptación deslizante.
- Generador 'grouped' que:
    * Divide el espacio uint32 en G intervalos.
    * Mantiene contadores éxito/intent (Laplace smoothing).
    * Selecciona candidatos vía softmax con temperatura y decaimiento.
    * Vectoriza métricas (NumPy) si disponible; fallback puro Python.
    * Aplica pre‑filtro métricas (entropía / pattern).
    * Valida contra objeto externo `validator` (interfaz flexible):
         validate(nonce:int, block_data:dict) -> bool
       o bien:
         validate(...) -> (bool, hash_bytes)
      En este segundo caso, si hash_bytes es de 32 bytes se añade `hash` al registro.
- Preparado para instrumentación (# METRIC comments).

Registro estándar por nonce:
    {
      "nonce": int,
      "entropy": float,
      "uniqueness": float,
      "zero_density": float,
      "pattern_score": float,
      "is_valid": bool,
      "block_height": int,
      "hash": (hex str opcional)
    }

Configuración (sección `grouped_generator` en global_config.json):
{
  "groups": 16,
  "softmax_temperature": 1.5,
  "temperature_min": 0.5,
  "temperature_decay": 0.98,
  "batch_candidate_factor": 4,
  "max_candidate_factor": 32,
  "max_attempt_loops": 6,
  "min_entropy_threshold": 0.0,
  "pattern_score_min": 0.40,
  "recent_window": 2048,
  "use_validator": true,
  "history_window": 500
}
"""

import os
import json
import math
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, List, Iterable, Any, Tuple, Union, Callable

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

# ----------------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------------
UINT32_MASK = 0xFFFFFFFF
BIT_WIDTH = 32
CSV_FIELDS = ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height","hash"]
LOGGER_BASE = "generator"


# ----------------------------------------------------------------------------------
# Config Loader Cache Ligero
# ----------------------------------------------------------------------------------
class _ConfigCache:
    _lock = threading.Lock()
    _cache: Dict[str, Dict] = {}
    _mtimes: Dict[str, float] = {}

    @classmethod
    def load(cls, path: Path, force: bool = False) -> Dict:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return {}
        key = str(path)
        with cls._lock:
            if (not force) and key in cls._cache and cls._mtimes.get(key) == mtime:
                return cls._cache[key]
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                cls._cache[key] = data
                cls._mtimes[key] = mtime
                return data
            except Exception:
                return cls._cache.get(key, {})


# ----------------------------------------------------------------------------------
# Clase Base
# ----------------------------------------------------------------------------------
class BaseNonceGenerator:
    name: str
    DEFAULTS: Dict[str, Any] = {}

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, rng: Optional[Any] = None):
        self.name = name
        self.logger = logging.getLogger(f"{LOGGER_BASE}.{name}")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            self.logger.addHandler(h)
        self.logger.propagate = True

        base_dir = Path(os.environ.get("IAZAR_BASE", "C:/zarturxia/src/iazar"))
        self.base_dir = base_dir
        self.config_path = base_dir / "config" / "global_config.json"
        global_cfg = _ConfigCache.load(self.config_path)

        section_key = f"{self.name}_generator"
        section_cfg = (config or {}).get(section_key, {}) if config else global_cfg.get(section_key, {})
        merged = {**self.DEFAULTS, **section_cfg}
        self.config = merged

        self._accept_history: List[int] = []
        self._history_window = int(self.config.get("history_window", 500))
        self._lock = threading.RLock()
        # RNG local (para reproducibilidad si se desea)
        self._rng = rng
        if self._rng is None and _HAS_NUMPY:
            self._rng = np.random.default_rng()

    # --- API principal ---
    def run_generation(self, block_height: int, block_data: Dict[str, Any],
                       batch_size: int = 500) -> Iterable[Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    # --- Utilidades métricas ---
    @staticmethod
    def sanitize_uint32(v: int) -> int:
        return int(v) & UINT32_MASK

    @staticmethod
    def bit_entropy(n: int) -> float:
        b = f"{n:032b}"
        z = b.count('0') / BIT_WIDTH
        if z == 0.0 or z == 1.0:
            return 0.0
        o = 1.0 - z
        return -(z * math.log2(z) + o * math.log2(o))

    @staticmethod
    def zero_density(n: int) -> float:
        return f"{n:032b}".count('0') / BIT_WIDTH

    @staticmethod
    def pattern_score(n: int) -> float:
        b = f"{n:032b}"
        max_run = 1
        cur = 1
        for i in range(1, BIT_WIDTH):
            if b[i] == b[i-1]:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 1
        penalty = min(0.6, max_run / BIT_WIDTH)
        return max(0.4, 1.0 - penalty)

    def uniqueness(self, n: int, recent: List[int]) -> float:
        if not recent:
            return 1.0
        subset = recent[-32:]
        total = 0
        for r in subset:
            total += bin((r ^ n) & UINT32_MASK).count("1")
        return max(0.6, min(1.0, total / (BIT_WIDTH * len(subset))))

    # --- Accept ratio tracking ---
    def record_accept(self, accepted: bool):
        self._accept_history.append(1 if accepted else 0)
        if len(self._accept_history) > self._history_window:
            self._accept_history = self._accept_history[-self._history_window:]

    def recent_accept_ratio(self) -> float:
        if not self._accept_history:
            return 0.0
        return sum(self._accept_history) / len(self._accept_history)

    # --- Vectorizado de métricas base ---
    @staticmethod
    def _compute_bit_metrics_vectorized(nonces: List[int]) -> Dict[str, List[float]]:
        if not nonces:
            return {"entropy": [], "zero_density": [], "pattern_score": []}
        if _HAS_NUMPY:
            arr = np.array(nonces, dtype=np.uint32)
            bits = np.unpackbits(arr.view(np.uint8)).reshape(-1, BIT_WIDTH)
            zeros = (bits == 0).sum(axis=1)
            p0 = zeros / BIT_WIDTH
            p1 = 1.0 - p0
            with np.errstate(divide='ignore', invalid='ignore'):
                entropy = -(p0 * np.log2(p0, where=(p0 > 0)) + p1 * np.log2(p1, where=(p1 > 0)))
                entropy = np.nan_to_num(entropy)
            transitions = np.diff(bits, axis=1) != 0
            run_lengths = BIT_WIDTH - transitions.sum(axis=1)
            run_penalty = np.clip(run_lengths / BIT_WIDTH, 0, 0.6)
            pattern_score = np.maximum(0.4, 1.0 - run_penalty)
            return {
                "entropy": entropy.tolist(),
                "zero_density": p0.tolist(),
                "pattern_score": pattern_score.tolist()
            }
        # Fallback puro
        ent, zd, pat = [], [], []
        for n in nonces:
            b = f"{n:032b}"
            z = b.count('0') / BIT_WIDTH
            if z in (0.0, 1.0):
                e = 0.0
            else:
                o = 1.0 - z
                e = -(z * math.log2(z) + o * math.log2(o))
            # pattern simplificado
            max_run = 1
            cur = 1
            for i in range(1, BIT_WIDTH):
                if b[i] == b[i-1]:
                    cur += 1
                    if cur > max_run:
                        max_run = cur
                else:
                    cur = 1
            penalty = min(0.6, max_run / BIT_WIDTH)
            pat.append(max(0.4, 1.0 - penalty))
            ent.append(e)
            zd.append(z)
        return {"entropy": ent, "zero_density": zd, "pattern_score": pat}

    def _post_validation_record(self, record: Dict[str, Any], validation_result: Any):  # pragma: no cover
        if isinstance(validation_result, tuple) and len(validation_result) == 2:
            ok, h = validation_result
            if ok and isinstance(h, (bytes, bytearray)) and len(h) == 32:
                record["hash"] = h.hex()

    def close(self):  # pragma: no cover
        pass


# ----------------------------------------------------------------------------------
# GroupedTopNonceGenerator
# ----------------------------------------------------------------------------------
class GroupedTopNonceGenerator(BaseNonceGenerator):
    DEFAULTS = {
        "groups": 16,
        "softmax_temperature": 1.5,
        "temperature_min": 0.5,
        "temperature_decay": 0.98,
        "batch_candidate_factor": 4,
        "max_candidate_factor": 32,
        "max_attempt_loops": 6,
        "min_entropy_threshold": 0.0,
        "pattern_score_min": 0.40,
        "recent_window": 2048,
        "use_validator": True
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, validator: Optional[Any] = None, rng: Optional[Any] = None):
        super().__init__("grouped", config, rng=rng)
        self.validator = validator
        self.groups = max(2, int(self.config.get("groups", 16)))
        span = (UINT32_MASK + 1) // self.groups
        self._ranges: List[Tuple[int, int]] = []
        start = 0
        for g in range(self.groups):
            end = start + span - 1 if g < self.groups - 1 else UINT32_MASK
            self._ranges.append((start, end))
            start = end + 1

        self.accept_counters = [1] * self.groups
        self.total_counters = [1] * self.groups
        self.temperature = float(self.config.get("softmax_temperature", 1.5))

        self._recent_valid: List[int] = []
        self._recent_lock = threading.Lock()
        self._recent_limit = max(256, min(8192, int(self.config.get("recent_window", 2048))))

    # --- Public Generation ---
    def run_generation(self, block_height: int, block_data: Dict[str, Any], batch_size: int = 500) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        out: List[Dict[str, Any]] = []
        factor = int(self.config.get("batch_candidate_factor", 4))
        max_factor = int(self.config.get("max_candidate_factor", 32))
        attempts = 0
        use_validator = bool(self.config.get("use_validator", True) and self.validator)

        while len(out) < batch_size and attempts < int(self.config.get("max_attempt_loops", 6)):
            target = batch_size * factor
            probs = self._softmax_weights()
            counts = self._allocate_counts(target, probs)
            candidates, groups_meta = self._generate_candidates(counts)
            if not candidates:
                attempts += 1
                factor = min(factor * 2, max_factor)
                self._decay_temperature()
                continue

            metrics = self._compute_metrics(candidates)
            idxs = self._prefilter(metrics)

            if not idxs:
                attempts += 1
                factor = min(factor * 2, max_factor)
                self._decay_temperature()
                continue

            for idx in idxs:
                nonce = candidates[idx]
                grp = groups_meta[idx]
                validation_res: Union[bool, Tuple[bool, bytes]]
                if use_validator:
                    try:
                        validation_res = self.validator.validate(nonce, block_data)
                    except Exception:
                        validation_res = False
                else:
                    validation_res = True

                if isinstance(validation_res, tuple):
                    is_valid = bool(validation_res[0])
                else:
                    is_valid = bool(validation_res)

                self.total_counters[grp] += 1
                if is_valid:
                    self.accept_counters[grp] += 1
                    self.record_accept(True)
                else:
                    self.record_accept(False)

                rec = {
                    "nonce": nonce,
                    "entropy": round(metrics["entropy"][idx], 5),
                    "uniqueness": round(metrics["uniqueness"][idx], 5),
                    "zero_density": round(metrics["zero_density"][idx], 5),
                    "pattern_score": round(metrics["pattern_score"][idx], 5),
                    "is_valid": is_valid,
                    "block_height": block_height
                }
                self._post_validation_record(rec, validation_res)
                if is_valid:
                    self._record_recent(nonce)
                out.append(rec)
                if len(out) >= batch_size:
                    break

            attempts += 1
            if len(out) < batch_size:
                factor = min(factor * 2, max_factor)
            self._decay_temperature()

        elapsed = time.perf_counter() - t0
        self.logger.info(
            "[GroupedGenerator] block=%s produced=%d attempts=%d factor_final=%d elapsed=%.3fs accept_ratio=%.4f",
            block_height, len(out), attempts, factor, elapsed, self.recent_accept_ratio()
        )
        # METRIC: grouped_generator_batch_latency.observe(elapsed)
        return out

    # --- Internals ---
    def _softmax_weights(self) -> List[float]:
        ratios = [(a / t) if t else 0.0 for a, t in zip(self.accept_counters, self.total_counters)]
        mx = max(ratios) if ratios else 1.0
        temp_min = float(self.config.get("temperature_min", 0.5))
        temp = max(temp_min, self.temperature)
        exps = [math.exp((r - mx) / temp) for r in ratios]
        ssum = sum(exps)
        if not ssum:
            return [1.0 / self.groups] * self.groups
        return [e / ssum for e in exps]

    def _allocate_counts(self, total: int, probs: List[float]) -> List[int]:
        counts = [int(p * total) for p in probs]
        diff = total - sum(counts)
        i = 0
        while diff > 0 and counts:
            counts[i % len(counts)] += 1
            diff -= 1
            i += 1
        return counts

    def _generate_candidates(self, counts: List[int]) -> Tuple[List[int], List[int]]:
        cands: List[int] = []
        meta: List[int] = []
        for gi, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            lo, hi = self._ranges[gi]
            width = hi - lo + 1
            if _HAS_NUMPY:
                # RNG local si disponible
                if hasattr(self._rng, "integers"):
                    arr = (self._rng.integers(0, width, size=cnt, dtype=np.uint32) + lo) & UINT32_MASK
                else:
                    arr = (np.random.randint(0, width, size=cnt, dtype=np.uint32) + lo) & UINT32_MASK
                cands.extend(int(x) for x in arr)
            else:
                import secrets
                for _ in range(cnt):
                    n = lo + (secrets.randbits(32) % width)
                    cands.append(n & UINT32_MASK)
            meta.extend([gi] * cnt)
        if _HAS_NUMPY and cands:
            arr = np.array(cands, dtype=np.uint32)
            uniq, idx_map = np.unique(arr, return_index=True)
            cands = [int(x) for x in uniq]
            meta = [meta[i] for i in idx_map]
        return cands, meta

    def _compute_metrics(self, nonces: List[int]) -> Dict[str, List[float]]:
        base = self._compute_bit_metrics_vectorized(nonces)
        recent = self._recent_snapshot()
        if _HAS_NUMPY and nonces:
            arr = np.array(nonces, dtype=np.uint32)
            if recent:
                recent_arr = np.array(recent[-32:], dtype=np.uint32)
                xor = arr[:, None] ^ recent_arr[None, :]
                pop = np.unpackbits(xor.view(np.uint8), axis=2).sum(axis=2)
                uniq = np.clip(pop.mean(axis=1) / BIT_WIDTH, 0.6, 1.0).tolist()
            else:
                uniq = [1.0] * len(arr)
        else:
            uniq = [self.uniqueness(n, recent) for n in nonces]
        base["uniqueness"] = uniq
        return base

    def _prefilter(self, metrics: Dict[str, List[float]]) -> List[int]:
        ent_min = float(self.config.get("min_entropy_threshold", 0.0))
        pat_min = float(self.config.get("pattern_score_min", 0.40))
        entropy = metrics["entropy"]
        pattern = metrics["pattern_score"]
        return [i for i in range(len(entropy)) if entropy[i] >= ent_min and pattern[i] >= pat_min]

    def _decay_temperature(self):
        decay = float(self.config.get("temperature_decay", 0.98))
        self.temperature = max(float(self.config.get("temperature_min", 0.5)), self.temperature * decay)

    def _record_recent(self, nonce: int):
        with self._recent_lock:
            self._recent_valid.append(nonce & UINT32_MASK)
            if len(self._recent_valid) > self._recent_limit:
                self._recent_valid = self._recent_valid[-self._recent_limit:]

    def _recent_snapshot(self) -> List[int]:
        with self._recent_lock:
            return list(self._recent_valid)

    def reset_stats(self):
        self.accept_counters = [1] * self.groups
        self.total_counters = [1] * self.groups
        self._accept_history.clear()
        with self._recent_lock:
            self._recent_valid.clear()
        self.temperature = float(self.config.get("softmax_temperature", 1.5))

    def stats(self) -> Dict[str, Any]:
        ratios = [(a / t) if t else 0.0 for a, t in zip(self.accept_counters, self.total_counters)]
        return {
            "groups": self.groups,
            "temperature": round(self.temperature, 4),
            "ratios": ratios,
            "recent_accept_ratio": round(self.recent_accept_ratio(), 4),
            "recent_cache": len(self._recent_valid)
        }

    def close(self):
        pass


# ----------------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------------
def create_grouped_generator(config: Optional[Dict[str, Any]] = None, validator: Optional[Any] = None) -> GroupedTopNonceGenerator:
    return GroupedTopNonceGenerator(config=config, validator=validator)

# Alias genérico para autodescubrimiento
def create_generator(config: Optional[Dict[str, Any]] = None, validator: Optional[Any] = None) -> GroupedTopNonceGenerator:
    return create_grouped_generator(config=config, validator=validator)
