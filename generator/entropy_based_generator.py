import os
import csv
import time
import random
import logging
import threading
import concurrent.futures
from typing import Optional, Dict, List, Set, Any

import numpy as np

from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.generator.config_loader import config_loader
from iazar.generator.randomx_validator import RandomXValidator

# --- Logging Industrial ---
logger = logging.getLogger("EntropyBasedGenerator")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

# Directorio base para datos
DATA_DIR = './data'

class EntropyBasedGenerator(BaseNonceGenerator):
    """
    Generador industrial de nonces basado en métricas de entropía, preparado para minería Monero RandomX.
    Cumple los requisitos de integración multi-hilo, batch y escalabilidad.
    """

    DATA_REFRESH_INTERVAL = 300
    RECENT_NONCES_SIZE = 500
    ENTROPY_BINS = 16
    MAX_VALIDATION_WORKERS = max(2, min(16, os.cpu_count() or 4))
    BATCH_ENTROPY_SAMPLES = 10000

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("entropy_based", config)
        self.lock = threading.RLock()
        self.last_refresh = 0
        self.validator = RandomXValidator(self.config)
        self._initialize_data()
        logger.info("[EntropyBasedGenerator] Vectorized generator initialized.")
    
    def _get_data_path(self, key: str) -> str:
        """Obtener ruta de datos profesional"""
        return os.path.join(DATA_DIR, f"{self.generator_name}_{key}.csv")
    
    def _load_training_data(self) -> list:
        """Carga profesional de datos de entrenamiento"""
        path = self._get_data_path('training')
        if not os.path.exists(path):
            logger.warning(f"[EntropyBasedGenerator] No training data found at {path}")
            return []
        
        try:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            logger.info(f"[EntropyBasedGenerator] Loaded {len(data)} training samples")
            return data
        except Exception as e:
            logger.error(f"[EntropyBasedGenerator] Error loading training data: {e}")
            return []

    def _initialize_data(self):
        with self.lock:
            self.training_data = self._load_training_data()
            self.entropy_distribution = self._calculate_entropy_distribution()
            self.popcount_lut = self._create_popcount_lut()
            self.last_refresh = time.time()

    def _create_popcount_lut(self) -> np.ndarray:
        # Lookup table for population count (popcount) for fast Hamming metrics
        return np.array([bin(x).count('1') for x in range(256)], dtype=np.uint8)

    def _should_refresh(self) -> bool:
        return time.time() - self.last_refresh > self.DATA_REFRESH_INTERVAL

    def _calculate_entropy_distribution(self) -> np.ndarray:
        # Cálculo vectorizado de la distribución de entropía objetivo
        try:
            valid_entropies = [
                float(row['entropy']) for row in self.training_data
                if row.get('is_valid', 'false').lower() == 'true'
            ]
            valid_entropies = [e for e in valid_entropies if 0.7 <= e <= 1.0]
            if not valid_entropies:
                return np.array([0.85, 0.90, 0.95])
            hist, _ = np.histogram(
                valid_entropies,
                bins=self.ENTROPY_BINS,
                range=(0.7, 1.0),
                density=True
            )
            return np.cumsum(hist) / np.sum(hist)
        except Exception as e:
            logger.error(f"[EntropyBasedGenerator] Entropy distribution error: {e}")
            return np.array([0.85, 0.90, 0.95])

    def _sample_target_entropy(self) -> float:
        if isinstance(self.entropy_distribution, np.ndarray):
            u = random.uniform(0, 1)
            idx = np.searchsorted(self.entropy_distribution, u)
            bin_low = 0.7 + (idx / self.ENTROPY_BINS) * 0.3
            bin_high = 0.7 + ((idx + 1) / self.ENTROPY_BINS) * 0.3
            return random.uniform(bin_low, bin_high)
        return random.uniform(0.85, 0.95)

    def _generate_candidate_batch(self, batch_size: int, seen_batch: Set[int]) -> List[int]:
        # Vectorized generation + batch filtering by entropy
        nonces_arr = np.random.randint(0, 2**64, size=batch_size, dtype=np.uint64)
        nonces_arr = np.array([n for n in nonces_arr if n not in seen_batch], dtype=np.uint64)
        if nonces_arr.size == 0:
            return []
        counts = np.zeros(nonces_arr.size, dtype=np.uint8)
        for shift in range(0, 64, 8):
            byte_vals = (nonces_arr >> shift) & 0xFF
            counts += self.popcount_lut[byte_vals]
        p1 = counts / 64.0
        p0 = 1.0 - p1
        
        # CORRECCIÓN DEFINITIVA: Expresión simplificada y corregida
        log_p0 = np.where(p0 > 0, np.log2(p0), 0)
        log_p1 = np.where(p1 > 0, np.log2(p1), 0)
        entropy = - (p0 * log_p0 + p1 * log_p1)
        
        target_entropies = np.array([self._sample_target_entropy() for _ in range(nonces_arr.size)])
        mask = np.abs(entropy - target_entropies) < 0.03
        return nonces_arr[mask].tolist()

    def _validate_batch(self, nonces: List[int], block_data: dict) -> List[bool]:
        # Validación paralela multi-hilo
        if not nonces:
            return []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_VALIDATION_WORKERS) as executor:
            futures = {executor.submit(self.validator.validate, nonce, block_data): nonce for nonce in nonces}
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"[EntropyBasedGenerator] Validation failed: {e}")
                    results.append(False)
        return results

    def _calculate_uniqueness(self, nonce: int, recent_set: Set[int]) -> float:
        if not recent_set:
            return 1.0
        nonce_arr = np.full(len(recent_set), nonce, dtype=np.uint64)
        recent_arr = np.array(list(recent_set), dtype=np.uint64)
        xor_arr = np.bitwise_xor(nonce_arr, recent_arr)
        popcnts = np.vectorize(lambda x: bin(x).count('1'))(xor_arr)
        avg_distance = np.mean(popcnts)
        return max(0.8, min(0.99, avg_distance / 64))

    def _calculate_metrics(self, nonce: int, recent_set: Set[int]) -> dict:
        bin_repr = bin(nonce)[2:].zfill(64)
        arr = np.array([int(b) for b in bin_repr])
        p0 = np.count_nonzero(arr == 0) / 64.0
        p1 = 1 - p0
        entropy = - (p0 * np.log2(p0 + 1e-10) + p1 * np.log2(p1 + 1e-10))
        zero_runs = np.split(arr, np.where(np.diff(arr) != 0)[0] + 1)
        max_zero_run = max(len(run) for run in zero_runs if run[0] == 0) if any(arr == 0) else 0
        diff = np.diff(arr, prepend=arr[0]-1, append=arr[-1]-1)
        run_starts = np.where(diff != 0)[0]
        run_lengths = np.diff(run_starts)
        max_run = np.max(run_lengths) if run_lengths.size > 0 else 0
        transitions = np.sum(arr[:-1] != arr[1:])
        uniqueness = self._calculate_uniqueness(nonce, recent_set)
        pattern_score = max(0.5, 1.0 - min(0.3, max_run/32) + min(0.2, transitions/63))
        return {
            "entropy": round(entropy, 5),
            "uniqueness": round(uniqueness, 5),
            "zero_density": round(p0, 5),
            "pattern_score": round(pattern_score, 5)
        }

    def _get_recent_nonces(self) -> Set[int]:
        recent_nonces = set()
        count = 0
        for row in reversed(self.training_data):
            if count >= self.RECENT_NONCES_SIZE:
                break
            if row.get('is_valid', 'false').lower() == 'true':
                try:
                    recent_nonces.add(int(row['nonce']))
                    count += 1
                except (ValueError, KeyError):
                    continue
        return recent_nonces

    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        start_time = time.time()
        if self._should_refresh():
            self._initialize_data()
        with self.lock:
            recent_nonces = self._get_recent_nonces()
            valid_nonces = []
            total_tries = 0
            max_batch_tries = 4
            batch_factor = 4
            for _ in range(max_batch_tries):
                seen_batch = set()
                candidates = self._generate_candidate_batch(batch_size * batch_factor, seen_batch)
                candidates = [c for c in candidates if c not in seen_batch and not seen_batch.add(c)]
                total_tries += len(candidates)
                validation_results = self._validate_batch(candidates, block_data)
                for nonce, is_valid in zip(candidates, validation_results):
                    if len(valid_nonces) >= batch_size:
                        break
                    if is_valid:
                        metrics = self._calculate_metrics(nonce, recent_nonces)
                        nonce_data = {
                            "nonce": nonce,
                            "block_height": block_height,
                            "generator": self.generator_name,
                            "is_valid": True,
                            **metrics
                        }
                        valid_nonces.append(nonce_data)
                        recent_nonces.add(nonce)
                if len(valid_nonces) >= batch_size:
                    break
                batch_factor *= 2
            if valid_nonces:
                self._save_nonces_batch(valid_nonces)
            elapsed = time.time() - start_time
            success_rate = len(valid_nonces) / total_tries if total_tries else 0
            logger.info(
                f"[EntropyBasedGenerator] Block {block_height}: "
                f"{len(valid_nonces)}/{batch_size} valid, "
                f"{total_tries} tries, "
                f"{success_rate:.2%} success, "
                f"{elapsed:.2f}s elapsed, "
                f"{len(valid_nonces)/elapsed if elapsed else 0:.1f} nonces/sec"
            )
            return valid_nonces

    def _save_nonces_batch(self, nonces: List[dict]):
        if not nonces:
            return
        output_path = self._get_data_path('generated_nonces')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = self.FIELDNAMES
        try:
            file_exists = os.path.exists(output_path)
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(
                    [{k: v for k, v in item.items() if k in fieldnames}
                     for item in nonces]
                )
        except IOError as e:
            logger.error(f"[EntropyBasedGenerator] Failed to save nonces: {e}")