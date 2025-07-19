import os
import csv
import time
import random
import logging
import threading
from typing import Optional, Dict, List, Set

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.generator.config_loader import config_loader
from iazar.generator.randomx_validator import RandomXValidator

# Directorio base para datos
DATA_DIR = './data'

class SequenceBasedGenerator(BaseNonceGenerator):
    """
    Sequence-Based Nonce Generator for Monero/RandomX Mining.
    - Learns step/gap patterns from historical data (incremental, decremental, variable)
    - Safe, concurrent, batch and industrial grade for enterprise mining.
    """

    DATA_REFRESH_INTERVAL = 300  # seconds
    RECENT_NONCES_SIZE = 500
    MAX_VALIDATION_WORKERS = max(2, min(16, os.cpu_count() or 4))
    SEQUENCE_WINDOW = 1024

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("sequence_based", config)
        self.lock = threading.RLock()
        self.last_refresh = 0
        self.validator = RandomXValidator(self.config)
        self.seq_patterns: List[int] = []
        self._initialize_data()
        self._log("Initialized.")
    
    def _get_data_path(self, key: str) -> str:
        """Obtener ruta de datos profesional"""
        return os.path.join(DATA_DIR, f"{self.generator_name}_{key}.csv")
    
    def _load_training_data(self) -> list:
        """Carga profesional de datos de entrenamiento"""
        path = self._get_data_path('training')
        if not os.path.exists(path):
            self._log(f"No training data found at {path}", level="warning")
            return []
        
        try:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            self._log(f"Loaded {len(data)} training samples")
            return data
        except Exception as e:
            self._log(f"Error loading training data: {e}", level="error")
            return []

    def _initialize_data(self):
        with self.lock:
            self.training_data = self._load_training_data()
            self.seq_patterns = self._extract_sequence_patterns()
            self.last_refresh = time.time()
            self._log(f"Patterns: {self.seq_patterns}")

    def _should_refresh(self) -> bool:
        return (time.time() - self.last_refresh) > self.DATA_REFRESH_INTERVAL

    def _extract_sequence_patterns(self) -> List[int]:
        """Detects frequent step/gap sizes from successful historical nonces."""
        nonces = [int(row['nonce']) for row in self.training_data
                  if row.get('is_valid', 'False').lower() == 'true' and 'nonce' in row]
        if len(nonces) < 20:
            return [1, 4, 8, 32, 128, 1024]
        nonces = sorted(nonces)
        gaps = np.diff(nonces)
        if len(gaps) == 0:
            return [1, 4, 8, 32]
        hist, bin_edges = np.histogram(gaps, bins=np.logspace(0, 16, num=17, base=2))
        top_indices = hist.argsort()[-8:][::-1]
        steps = sorted(set(int(bin_edges[i]) for i in top_indices if hist[i] > 0))
        return steps or [1, 4, 8, 16, 32, 128]

    def _generate_sequence_batch(self, block_height: int, batch_size: int) -> List[int]:
        """Batch of sequence-based nonces using learned patterns."""
        nonces = set()
        max_start = 2 ** 64 - self.SEQUENCE_WINDOW
        rng = random.SystemRandom()
        for pattern in self.seq_patterns:
            if len(nonces) >= batch_size:
                break
            direction = rng.choice([1, -1])
            start = rng.randint(0, max_start)
            for i in range(self.SEQUENCE_WINDOW // pattern):
                if len(nonces) >= batch_size:
                    break
                candidate = start + direction * pattern * i
                if 0 <= candidate < 2**64:
                    nonces.add(candidate)
        # Fill with random nonces if needed
        while len(nonces) < batch_size:
            nonces.add(rng.getrandbits(64))
        return list(nonces)[:batch_size]

    def _get_recent_nonces(self) -> Set[int]:
        """Returns most recent valid nonces."""
        recent = set()
        for row in reversed(self.training_data):
            if len(recent) >= self.RECENT_NONCES_SIZE:
                break
            try:
                if row.get('is_valid', 'False').lower() == 'true':
                    recent.add(int(row['nonce']))
            except (ValueError, KeyError):
                continue
        return recent

    def _calculate_metrics(self, nonce: int, recent_set: Set[int]) -> dict:
        bin_repr = bin(nonce)[2:].zfill(64)
        arr = np.array([int(b) for b in bin_repr], dtype=np.uint8)
        p0 = np.count_nonzero(arr == 0) / 64.0
        p1 = 1 - p0
        entropy = - (p0 * np.log2(p0 + 1e-12) + p1 * np.log2(p1 + 1e-12))
        zero_runs = np.split(arr, np.where(np.diff(arr) != 0)[0] + 1)
        max_zero_run = max((len(run) for run in zero_runs if run[0] == 0), default=0)
        run_lengths = np.diff(np.where(np.concatenate(([0], arr[:-1] != arr[1:], [0])))[0])
        max_run = np.max(run_lengths) if run_lengths.size > 0 else 0
        transitions = np.sum(arr[:-1] != arr[1:])
        uniqueness = self._calculate_uniqueness(nonce, recent_set)
        pattern_score = max(0.5, 1.0 - min(0.3, max_run / 32) + min(0.2, transitions / 63))
        return {
            "entropy": round(entropy, 5),
            "uniqueness": round(uniqueness, 5),
            "zero_density": round(p0, 5),
            "pattern_score": round(pattern_score, 5)
        }

    def _calculate_uniqueness(self, nonce: int, recent_set: Set[int]) -> float:
        """Promedio de Hamming contra nonces recientes (vectorizado)."""
        if not recent_set:
            return 1.0
        nonce_arr = np.full(len(recent_set), nonce, dtype=np.uint64)
        recent_arr = np.array(list(recent_set), dtype=np.uint64)
        xor_arr = np.bitwise_xor(nonce_arr, recent_arr)
        popcnts = np.vectorize(lambda x: bin(x).count('1'))(xor_arr)
        avg_distance = np.mean(popcnts)
        return max(0.8, min(0.99, avg_distance / 64))

    def _validate_batch(self, nonces: List[int], block_data: dict) -> List[bool]:
        """Valida los nonces en paralelo usando RandomX."""
        if not nonces:
            return []
        results = [False] * len(nonces)
        with ThreadPoolExecutor(max_workers=self.MAX_VALIDATION_WORKERS) as executor:
            future_map = {executor.submit(self.validator.validate, n, block_data): i for i, n in enumerate(nonces)}
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = bool(future.result())
                except Exception as e:
                    self._log(f"Validation failed for idx {idx}: {e}", level="error")
        return results

    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        t0 = time.time()
        if self._should_refresh():
            self._initialize_data()
        with self.lock:
            recent_nonces = self._get_recent_nonces()
            candidates = self._generate_sequence_batch(block_height, batch_size * 4)
            validation = self._validate_batch(candidates, block_data)
            valid_nonces = []
            for nonce, is_valid in zip(candidates, validation):
                if len(valid_nonces) >= batch_size:
                    break
                if is_valid:
                    metrics = self._calculate_metrics(nonce, recent_nonces)
                    valid_nonces.append({
                        "nonce": nonce,
                        "block_height": block_height,
                        "generator": self.generator_name,
                        "is_valid": True,
                        **metrics
                    })
                    recent_nonces.add(nonce)
            self._save_nonces_batch(valid_nonces)
            elapsed = time.time() - t0
            self._log(
                f"Block {block_height}: {len(valid_nonces)}/{batch_size} valid, "
                f"{len(candidates)} tried, {len(valid_nonces)/len(candidates):.2%} ok, "
                f"{elapsed:.2f}s, {len(valid_nonces)/elapsed if elapsed else 0:.1f} nonces/sec"
            )
            return valid_nonces

    def _save_nonces_batch(self, nonces: List[dict]):
        if not nonces:
            return
        output_path = self._get_data_path('generated_nonces')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = self.FIELDNAMES
        file_exists = os.path.exists(output_path)
        try:
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows([{k: v for k, v in item.items() if k in fieldnames} for item in nonces])
        except Exception as e:
            self._log(f"Failed to save nonces: {e}", level="error")

    def _log(self, msg: str, level="info"):
        logger = logging.getLogger("SequenceBasedGenerator")
        if not logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)
        getattr(logger, level, logger.info)(msg)