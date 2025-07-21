from __future__ import annotations
"""
entropy_based_generator.py
--------------------------
Generador guiado por perfiles de entropía y densidad de ceros.
"""

import os
import math
import time
import logging
import threading
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    _HAS_PANDAS = False

from iazar.proxy.randomx_validator import RandomXValidator

LOGGER_NAME = "generator.entropy"
logger = logging.getLogger(LOGGER_NAME)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

UINT32_MASK = 0xFFFFFFFF
CSV_FIELDS = ["nonce", "entropy", "uniqueness", "zero_density", "pattern_score", "is_valid", "block_height"]

DEFAULT_TARGET_ENTROPY = (5.5, 7.8)
TRAIN_SAMPLE_LIMIT = 200_000

class _EntropyHistogramModel:
    def __init__(self, bins: int = 32):
        self.bins = bins
        self.hist = np.zeros(bins, dtype=np.float64) if _HAS_NUMPY else [0.0] * bins
        self.count = 0

    def update(self, values: Sequence[float]):
        if not values:
            return
        if _HAS_NUMPY:
            hist, _ = np.histogram(values, bins=self.bins, range=(0.0, 1.0), density=False)
            self.hist += hist
            self.count += len(values)
        else:
            # Simple binning fallback
            for v in values:
                idx = min(self.bins - 1, int(v * self.bins))
                self.hist[idx] += 1
                self.count += 1

    def normalize(self):
        if self.count == 0:
            return
        if _HAS_NUMPY:
            self.hist = self.hist / self.count
        else:
            self.hist = [h / self.count for h in self.hist]

    def get(self):
        return self.hist

class EntropyBasedGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None, base_dir: Optional[Path] = None, validator: Optional[RandomXValidator] = None):
        self.config = config or {}
        self.base_dir = base_dir or Path(os.getcwd())
        self.validator = validator
        self.recent_valid = []
        self.recent_window = 128
        self.model_entropy = _EntropyHistogramModel(bins=32)
        self.model_zero_density = _EntropyHistogramModel(bins=32)
        self.lock = threading.Lock()

    def train(self, data: List[Dict[str, Any]]):
        entropies = [row["entropy"] for row in data if "entropy" in row]
        zeros = [row["zero_density"] for row in data if "zero_density" in row]
        self.model_entropy.update(entropies)
        self.model_zero_density.update(zeros)
        self.model_entropy.normalize()
        self.model_zero_density.normalize()

    def generate_nonce(self, blob: bytes, target: int, block_height: int = 0) -> int:
        # Ejemplo: genera un nonce aleatorio, puedes reemplazar por tu heurística IA real.
        nonce = random.getrandbits(32)
        return nonce

    def validate_nonce(self, nonce: int, blob: bytes, target: int) -> bool:
        if self.validator is not None:
            return self.validator.is_valid_nonce(blob, nonce, target)
        # fallback simple
        return True

    def _entropy(self, data: bytes) -> float:
        """Calcula entropía Shannon del bloque de datos."""
        if not data:
            return 0.0
        freq = [0] * 256
        for b in data:
            freq[b] += 1
        probs = [f / len(data) for f in freq if f]
        return -sum(p * math.log2(p) for p in probs)

    def _zero_density(self, data: bytes) -> float:
        if not data:
            return 0.0
        zeros = sum(1 for b in data if b == 0)
        return zeros / len(data)

    def update_recent(self, nonce: int):
        self.recent_valid.append(nonce)
        if len(self.recent_valid) > self.recent_window:
            self.recent_valid = self.recent_valid[-self.recent_window:]

    def score_pattern(self, nonce: int) -> float:
        # Ejemplo dummy: puntuación basada en la cantidad de unos en el nonce
        return bin(nonce).count('1') / 32.0

    # --- PATCH PRO: método robusto contra AxisError ---
    def _uniqueness(self, values: list) -> list:
        if not self.recent_valid:
            return [1.0]*len(values)
        recent = self.recent_valid[-self.recent_window:]
        if not _HAS_NUMPY:
            out=[]
            for v in values:
                diffs = sum(bin((r ^ v) & UINT32_MASK).count('1') for r in recent[-256:])
                avg = diffs/(min(len(recent), 256)*32.0)
                out.append(min(0.99, max(0.7, avg)))
            return out
        arr = np.asarray(values, dtype=np.uint32)
        if arr.ndim == 0 or arr.size == 0:
            return []
        if arr.ndim == 1 and arr.size == 1:
            arr = arr.reshape(1)
        rec = np.asarray(recent[-512:], dtype=np.uint32)
        if rec.ndim == 0 or rec.size == 0:
            return [1.0]*arr.shape[0]
        xor = arr[:, None] ^ rec[None, :]
        pop = np.unpackbits(xor.view(np.uint8), axis=-1).sum(axis=-1)
        # --- PATCH: Manejo robusto para lote de 1 o shape 1D ---
        if pop.ndim == 1:  # Si solo hay un elemento, mean sobre toda la fila
            mean_val = pop.mean() / 32.0
            return [float(np.clip(mean_val, 0.7, 0.99))]
        return np.clip((pop.mean(axis=1) / 32.0), 0.7, 0.99).tolist()

def create_generator(config: Optional[Dict[str, Any]] = None,
                     base_dir: Optional[Path] = None,
                     validator: Optional[RandomXValidator] = None) -> EntropyBasedGenerator:
    return EntropyBasedGenerator(config=config, base_dir=base_dir, validator=validator)
