from __future__ import annotations
"""
ml_based_generator.py
---------------------
Generador de nonces basado en modelo ML incremental (feature engineering bitwise + scoring adaptativo).

Flujo:
 1. Sobre-genera candidatos (factor dinámico).
 2. Calcula features bitwise (entropy, zero_density, pattern_score, uniqueness).
 3. Escala + aplica modelo ML (o heurística fallback).
 4. Selecciona Top-K (reduce coste de validación).
 5. Valida paralelamente con RandomX (retorna hash si ok).
 6. Registra aceptados (solo return; persistencia delegada al orquestador).
 7. Alimenta buffer de entrenamiento (positivos + muestra de negativos).
 8. Reentrena cuando se alcanza umbral de muestras nuevas.

Salida estándar (lista de dicts):
  ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height","hash?"]

El orquestador usará únicamente las 7 columnas para CSV. "hash" se utiliza para proxy/submit.

Modelos soportados:
  - "gbrt" (GradientBoostingRegressor)
  - "rf"   (RandomForestRegressor)
  - "mlp"  (MLPRegressor)
  - "internal" (heurística sin sklearn)

Checkpoints versionados (model, scaler, meta) bajo: <base>/data/models/ml_based/
"""

import os
import time
import json
import math
import logging
import threading
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.proxy.randomx_validator import RandomXValidator

LOGGER_NAME = "generator.ml"
logger = logging.getLogger(LOGGER_NAME)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _binary_entropy_from_p(p0: float) -> float:
    if p0 <= 0.0 or p0 >= 1.0:
        return 0.0
    p1 = 1.0 - p0
    return -(p0 * math.log2(p0) + p1 * math.log2(p1))

MODEL_VERSION = 1
MODEL_FILENAME = f"model_v{MODEL_VERSION}.pkl"
SCALER_FILENAME = f"scaler_v{MODEL_VERSION}.pkl"
META_FILENAME = f"meta_v{MODEL_VERSION}.json"

class OnlineScaler:
    """Normalizador incremental (mean / std) por feature (Welford). Thread-safe."""
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.count = 0
        self.mean = [0.0] * n_features
        self.M2 = [0.0] * n_features
        self._lock = threading.Lock()

    def partial_fit(self, X: List[List[float]]):
        with self._lock:
            for row in X:
                self.count += 1
                for i, val in enumerate(row):
                    delta = val - self.mean[i]
                    self.mean[i] += delta / self.count
                    delta2 = val - self.mean[i]
                    self.M2[i] += delta * delta2

    def transform(self, X: List[List[float]]):
        if self.count < 2:
            return X
        std = [math.sqrt(m2 / (self.count - 1)) if (self.count > 1 and m2 > 0) else 1.0
               for m2 in self.M2]
        out: List[List[float]] = []
        for row in X:
            out.append([(val - self.mean[i]) / std[i] for i, val in enumerate(row)])
        return out

    def save(self, path: Path):
        with path.open('wb') as f:
            pickle.dump({
                "count": self.count,
                "mean": self.mean,
                "M2": self.M2,
                "n_features": self.n_features
            }, f)

    @staticmethod
    def load(path: Path) -> "OnlineScaler":
        with path.open('rb') as f:
            d = pickle.load(f)
        obj = OnlineScaler(d['n_features'])
        obj.count = d['count']
        obj.mean = d['mean']
        obj.M2 = d['M2']
        return obj

# ---------------------------------------------------------------------------
# Generador ML
# ---------------------------------------------------------------------------
class MLBasedGenerator(BaseNonceGenerator):
    """Generador con scoring ML incremental."""

    DEFAULTS = {
        "batch_candidate_factor": 6,
        "max_candidate_factor": 40,
        "max_attempt_loops": 8,
        "recent_training_window": 8000,
        "min_train_samples": 400,
        "retrain_interval_samples": 500,
        "feature_entropy_bins": 4,   # (reservado para features extendidas futuras)
        "topk_ratio": 0.6,
        "min_entropy_threshold": 0.0,
        "pattern_score_min": 0.45,
        "model_type": "gbrt",        # gbrt | rf | mlp | internal
        "mlp_hidden": [64, 32],
        "random_state": 1337,
        "save_checkpoints": True,
        "validation_workers": 8
    }

    FIELDNAMES = ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**self.DEFAULTS, **(config or {}).get("ml_based_generator", {})}
        super().__init__("ml", merged)
        base_dir = Path(os.environ.get("IAZAR_BASE", "C:/zarturxia/src/iazar"))
        self.data_dir = base_dir / "data"
        self.model_dir = self.data_dir / "models" / "ml_based"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.validator = RandomXValidator((config or {}).get("randomx", {}))
        self._validation_workers = min(int(self.config["validation_workers"]), os.cpu_count() or 4)

        self._model_lock = threading.RLock()
        self._train_lock = threading.RLock()

        self._model = None
        self._scaler: Optional[OnlineScaler] = None
        self._train_buffer: List[Tuple[List[float], float]] = []
        self._samples_since_retrain = 0
        self._loaded_checkpoint = False

        self._rng = np.random.default_rng(self.config["random_state"]) if _HAS_NUMPY else None
        self._last_block_height: Optional[int] = None
        self._recent_valid_nonces: List[int] = []
        self._recent_valid_size = 1000

        self._executor: Optional["ThreadPoolExecutor"] = None
        self._executor_lock = threading.Lock()

        self._load_checkpoint_if_available()
        logger.info("[MLBasedGenerator] Initialized (model=%s, loaded=%s, workers=%d)",
                    self.config['model_type'], self._loaded_checkpoint, self._validation_workers)

    # ---------------- Public API ----------------
    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        t0 = time.perf_counter()
        self._refresh_recent(block_height)

        factor = int(self.config["batch_candidate_factor"])
        max_factor = int(self.config["max_candidate_factor"])
        attempts = 0
        accepted: List[dict] = []

        while len(accepted) < batch_size and attempts < int(self.config["max_attempt_loops"]):
            candidate_count = batch_size * factor
            nonces = self._generate_candidates(candidate_count)

            feats, metrics = self._build_features_and_metrics(nonces)
            scores = self._score(feats)

            topk = max(1, int(len(scores) * float(self.config["topk_ratio"])))
            if _HAS_NUMPY:
                idx_sorted = np.argsort(scores)[::-1][:topk]
            else:
                idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]

            selected_nonces = nonces[idx_sorted] if _HAS_NUMPY else [nonces[i] for i in idx_sorted]
            selected_feats = [feats[i] for i in idx_sorted]
            selected_metrics = {k: (metrics[k][idx_sorted] if _HAS_NUMPY else [metrics[k][i] for i in idx_sorted])
                                for k in metrics.keys()}

            val_hash_pairs = self._validate_parallel(selected_nonces, block_data)

            for i, (ok, h) in enumerate(val_hash_pairs):
                if not ok:
                    # Negativo para entrenamiento equilibrado (muestreo parcial)
                    if i < len(selected_feats):
                        self._push_train_example(selected_feats[i], 0.0)
                    continue
                nonce = int(selected_nonces[i])
                rec = {
                    "nonce": nonce,
                    "entropy": round(float(selected_metrics['entropy'][i]), 5),
                    "uniqueness": round(float(selected_metrics['uniqueness'][i]), 5),
                    "zero_density": round(float(selected_metrics['zero_density'][i]), 5),
                    "pattern_score": round(float(selected_metrics['pattern_score'][i]), 5),
                    "is_valid": True,
                    "block_height": block_height,
                    "hash": h.hex() if h and len(h) == 32 else None
                }
                accepted.append(rec)
                self._update_recent_valid(nonce)
                self._push_train_example(selected_feats[i], 1.0)
                if len(accepted) >= batch_size:
                    break

            if len(accepted) < batch_size:
                factor = min(factor * 2, max_factor)
            attempts += 1

        # No escritura aquí → orquestador persistirá accepted (sin campo hash)
        self._maybe_retrain()

        elapsed = time.perf_counter() - t0
        logger.info("[MLBasedGenerator] block=%s accepted=%d/%d attempts=%d elapsed=%.3fs",
                    block_height, len(accepted), batch_size, attempts, elapsed)
        # METRIC: ml_generator_batch_latency_seconds.observe(elapsed)
        return accepted

    # ---------------- Candidate Generation ----------------
    def _generate_candidates(self, count: int):
        if _HAS_NUMPY:
            raw64 = self._rng.integers(0, 2**64, size=int(count * 1.15), dtype=np.uint64)
            c32 = (raw64 & 0xFFFFFFFF).astype(np.uint32)
            uniq = np.unique(c32)
            # Completar hasta count si faltan
            while uniq.size < count:
                extra = (self._rng.integers(0, 2**64, size=count - uniq.size, dtype=np.uint64) & 0xFFFFFFFF).astype(np.uint32)
                uniq = np.unique(np.concatenate([uniq, extra]))
            return uniq[:count]
        else:  # pragma: no cover
            out = set()
            import secrets
            while len(out) < count:
                out.add(int.from_bytes(secrets.token_bytes(4), "little"))
            return list(out)

    # ---------------- Features & Metrics ----------------
    def _build_features_and_metrics(self, nonces):
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

            recent = self._recent_valid_snapshot()
            if recent.size == 0:
                uniqueness = np.full(arr.shape[0], 1.0, dtype=np.float32)
            else:
                xor = arr[:, None] ^ recent[None, :]
                popcnt = np.unpackbits(xor.view(np.uint8), axis=-1).sum(axis=-1)
                uniqueness = np.clip(popcnt.mean(axis=1) / 32.0, 0.7, 1.0)

            zero_density = p0
            feats = [[float(entropy[i]),
                      float(zero_density[i]),
                      float(pattern_score[i]),
                      float(uniqueness[i])]
                     for i in range(arr.shape[0])]
            metrics = {
                "entropy": entropy.astype(np.float32),
                "zero_density": zero_density.astype(np.float32),
                "pattern_score": pattern_score.astype(np.float32),
                "uniqueness": uniqueness.astype(np.float32)
            }
            return feats, metrics

        # Fallback puro Python
        feats: List[List[float]] = []
        metrics_arrays = {k: [] for k in ["entropy","zero_density","pattern_score","uniqueness"]}
        recent = self._recent_valid_nonces
        for n in (nonces.tolist() if hasattr(nonces, "tolist") else nonces):
            b = f"{int(n)&0xFFFFFFFF:032b}"
            zeros = b.count('0')
            p0 = zeros / 32.0
            ent = _binary_entropy_from_p(p0)
            runs0 = b.split('1')
            runs1 = b.split('0')
            max_run0 = max((len(r) for r in runs0), default=0)
            max_run1 = max((len(r) for r in runs1), default=0)
            run_penalty = min(0.6, max(max_run0, max_run1)/32.0)
            pat = max(0.4, 1.0 - run_penalty)
            if recent:
                diffs = []
                for rv in recent[-64:]:
                    x = (rv ^ int(n)) & 0xFFFFFFFF
                    diffs.append(bin(x).count("1"))
                uniq = max(0.7, min(1.0, sum(diffs)/(len(diffs)*32.0)))
            else:
                uniq = 1.0
            feats.append([ent, p0, pat, uniq])
            metrics_arrays['entropy'].append(ent)
            metrics_arrays['zero_density'].append(p0)
            metrics_arrays['pattern_score'].append(pat)
            metrics_arrays['uniqueness'].append(uniq)
        metrics = metrics_arrays
        return feats, metrics

    # ---------------- Scoring ----------------
    def _score(self, feats: List[List[float]]):
        scaled = self._apply_scaler(feats)
        with self._model_lock:
            if self._model is None:
                # Heurística fallback
                scores = []
                for f in scaled:
                    ent, zd, pat, uniq = f
                    balance = 1.0 - abs(zd - 0.5)
                    score = 0.35*ent + 0.25*pat + 0.2*uniq + 0.2*balance
                    scores.append(max(0.0, min(1.0, score)))
                return np.array(scores, dtype=np.float32) if _HAS_NUMPY else scores
            try:
                pred = self._model.predict(scaled)
                if _HAS_NUMPY:
                    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
                else:
                    pred = [max(0.0, min(1.0, float(x))) for x in pred]
                return pred
            except Exception as e:  # pragma: no cover
                logger.warning("Model prediction failed (%s) -> fallback heuristic", e)
                self._model = None
                return self._score(feats)

    def _apply_scaler(self, feats: List[List[float]]):
        if not feats:
            return feats
        if self._scaler is None:
            self._scaler = OnlineScaler(len(feats[0]))
        self._scaler.partial_fit(feats)
        return self._scaler.transform(feats)

    # ---------------- Training Buffer / Retrain ----------------
    def _push_train_example(self, features: List[float], label: float):
        with self._train_lock:
            self._train_buffer.append((features, label))
            if len(self._train_buffer) > int(self.config["recent_training_window"]):
                overflow = len(self._train_buffer) - int(self.config["recent_training_window"])
                if overflow > 0:
                    self._train_buffer = self._train_buffer[overflow:]
            self._samples_since_retrain += 1

    def _maybe_retrain(self):
        with self._train_lock:
            if self._samples_since_retrain < int(self.config["retrain_interval_samples"]):
                return
            if len(self._train_buffer) < int(self.config["min_train_samples"]):
                self._samples_since_retrain = 0
                return
            X = [f for f, _ in self._train_buffer]
            y = [lbl for _, lbl in self._train_buffer]

        model_type = self.config["model_type"]
        if model_type == "internal":
            with self._train_lock:
                self._samples_since_retrain = 0
            return

        if not _HAS_SKLEARN:
            logger.warning("sklearn no disponible – usando heurística interna permanentemente.")
            return

        start = time.perf_counter()
        scaler_snapshot = self._scaler
        try:
            with self._model_lock:
                if model_type == "gbrt":
                    model = GradientBoostingRegressor(random_state=self.config["random_state"],
                                                      n_estimators=80, max_depth=5)
                elif model_type == "rf":
                    model = RandomForestRegressor(random_state=self.config["random_state"],
                                                  n_estimators=80, max_depth=8, n_jobs=-1)
                elif model_type == "mlp":
                    hidden = self.config.get("mlp_hidden", [64, 32])
                    model = MLPRegressor(hidden_layer_sizes=tuple(hidden),
                                         random_state=self.config["random_state"],
                                         max_iter=300)
                else:
                    logger.info("Model type '%s' desconocido → heurística interna.", model_type)
                    self._model = None
                    with self._train_lock:
                        self._samples_since_retrain = 0
                    return

                model.fit(scaler_snapshot.transform(X), y)
                self._model = model
                with self._train_lock:
                    self._samples_since_retrain = 0

                if bool(self.config.get("save_checkpoints", True)):
                    self._save_checkpoint(model, scaler_snapshot)

            dur = time.perf_counter() - start
            logger.info("[MLBasedGenerator] Retrained model=%s samples=%d duration=%.2fs",
                        model_type, len(X), dur)
            # METRIC: ml_generator_retrain_seconds.observe(dur)
        except Exception as e:  # pragma: no cover
            logger.warning("Retrain failed: %s", e)

    # ---------------- Validation (Parallel) ----------------
    def _ensure_executor(self):
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            with self._executor_lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=self._validation_workers,
                                                        thread_name_prefix="ml-val")

    def _validate_parallel(self, nonces, block_data: dict) -> List[Tuple[bool, bytes]]:
        self._ensure_executor()
        from concurrent.futures import as_completed
        assert self._executor is not None
        results: List[Tuple[bool, bytes]] = [(False, b"")] * len(nonces)
        futures = {self._executor.submit(self.validator.validate, int(n), block_data, True): i
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
            except Exception as e:  # pragma: no cover
                logger.debug("Validation error nonce=%s err=%s", nonces[idx], e)
                results[idx] = (False, b"")
        return results

    # ---------------- Recent Accepted Tracking ----------------
    def _update_recent_valid(self, nonce: int):
        self._recent_valid_nonces.append(int(nonce) & 0xFFFFFFFF)
        if len(self._recent_valid_nonces) > self._recent_valid_size:
            self._recent_valid_nonces = self._recent_valid_nonces[-self._recent_valid_size:]

    def _recent_valid_snapshot(self):
        if not _HAS_NUMPY:
            import numpy as _np
            return _np.array(self._recent_valid_nonces, dtype=_np.uint32)
        return np.array(self._recent_valid_nonces, dtype=np.uint32) if self._recent_valid_nonces else np.empty(0, dtype=np.uint32)

    def _refresh_recent(self, block_height: int):
        if self._last_block_height == block_height:
            return
        self._last_block_height = block_height
        # (Opcional) se podría cargar un tail de nonces válidos históricos aquí.

    # ---------------- Checkpointing ----------------
    def _checkpoint_paths(self):
        return (self.model_dir / MODEL_FILENAME,
                self.model_dir / SCALER_FILENAME,
                self.model_dir / META_FILENAME)

    def _save_checkpoint(self, model, scaler: OnlineScaler):
        model_path, scaler_path, meta_path = self._checkpoint_paths()
        tmp_model = model_path.with_suffix(".pkl.tmp")
        tmp_scaler = scaler_path.with_suffix(".pkl.tmp")
        tmp_meta = meta_path.with_suffix(".json.tmp")
        try:
            with tmp_model.open('wb') as f:
                pickle.dump(model, f)
            scaler.save(tmp_scaler)
            with tmp_meta.open('w', encoding='utf-8') as f:
                json.dump({
                    "version": MODEL_VERSION,
                    "model_type": self.config["model_type"],
                    "timestamp": time.time()
                }, f)
            tmp_model.replace(model_path)
            tmp_scaler.replace(scaler_path)
            tmp_meta.replace(meta_path)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to save checkpoint: %s", e)

    def _load_checkpoint_if_available(self):
        model_path, scaler_path, meta_path = self._checkpoint_paths()
        if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
            return
        try:
            with meta_path.open('r', encoding='utf-8') as f:
                meta = json.load(f)
            if meta.get("version") != MODEL_VERSION:
                logger.info("Checkpoint version mismatch (found %s) ignoring.", meta.get("version"))
                return
            with model_path.open('rb') as f:
                model = pickle.load(f)
            scaler = OnlineScaler.load(scaler_path)
            with self._model_lock:
                self._model = model
            self._scaler = scaler
            self._loaded_checkpoint = True
            logger.info("Loaded ML model checkpoint (version=%s type=%s)",
                        meta.get("version"), meta.get("model_type"))
        except Exception as e:
            logger.warning("Failed loading checkpoint: %s", e)

    # ---------------- Close ----------------
    def close(self):
        if self._executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_generator(config: Optional[Dict[str, Any]] = None) -> MLBasedGenerator:
    return MLBasedGenerator(config=config)
