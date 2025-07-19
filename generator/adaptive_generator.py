import numpy as np
import random
import threading
import os
import csv
import time
import logging
from typing import Optional, Dict, List, Set, Tuple
from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.generator.config_loader import config_loader
from iazar.generator.randomx_validator import RandomXValidator

# Directorio base para datos
DATA_DIR = './data'

logger = logging.getLogger("AdaptiveGenerator")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

class AdaptiveGenerator(BaseNonceGenerator):
    """
    Adaptive Nonce Generator (Enterprise Grade)
    - Refuerzo online adaptativo (RL)
    - Bins auto-ajustables, batch concurrente y métricas avanzadas
    - Compatible con orquestador y RandomX
    """
    DEFAULT_BINS = 50
    MAX_BINS = 200
    MIN_BINS = 20
    DECAY_RATE = 0.995
    ADAPTATION_INTERVAL = 500
    PENALTY_FACTOR = 0.7
    REWARD_FACTOR = 1.3
    DATA_REFRESH_INTERVAL = 300
    RECENT_NONCES_SIZE = 500
    MAX_VALIDATION_WORKERS = max(2, min(16, os.cpu_count() or 4))

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("adaptive", config)
        self.lock = threading.RLock()
        self.bins = self.config.get("adaptive_params", {}).get("bins", self.DEFAULT_BINS)
        self.last_refresh = 0
        self.last_adaptation = 0
        self.validator = RandomXValidator(self.config)
        self._initialize_data()
        logger.info("[AdaptiveGenerator] Ready.")
    
    def _get_data_path(self, key: str) -> str:
        """Obtener ruta de datos profesional"""
        return os.path.join(DATA_DIR, f"{self.generator_name}_{key}.csv")
    
    def _load_training_data(self) -> list:
        """Carga profesional de datos de entrenamiento"""
        path = self._get_data_path('training')
        if not os.path.exists(path):
            logger.warning(f"[AdaptiveGenerator] No training data found at {path}")
            return []
        
        try:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            logger.info(f"[AdaptiveGenerator] Loaded {len(data)} training samples")
            return data
        except Exception as e:
            logger.error(f"[AdaptiveGenerator] Error loading training data: {e}")
            return []

    def _initialize_data(self):
        with self.lock:
            self.training_data = self._load_training_data()
            self.min_nonce = 0
            self.max_nonce = self.config.get('performance_settings', {}).get('max_nonce', 2**64-1)
            self.bin_weights = np.ones(self.bins)
            self.bin_success = np.zeros(self.bins)
            self.bin_attempts = np.zeros(self.bins)
            self.last_refresh = time.time()
            self._load_bin_stats_from_history()

    def _should_refresh(self) -> bool:
        return time.time() - self.last_refresh > self.DATA_REFRESH_INTERVAL

    def _load_bin_stats_from_history(self):
        if not self.training_data:
            return
        for row in self.training_data:
            try:
                nonce = int(row['nonce'])
                is_valid = str(row.get('is_valid', 'False')).lower() in ('true', '1', 'yes')
                bin_idx = self.nonce_to_bin(nonce)
                self.bin_attempts[bin_idx] += 1
                if is_valid:
                    self.bin_success[bin_idx] += 1
            except Exception:
                continue
        # Inicializar pesos tras la estadística histórica
        for i in range(self.bins):
            if self.bin_attempts[i] > 0:
                sr = self.bin_success[i] / self.bin_attempts[i]
                self.bin_weights[i] = max(0.1, sr * self.REWARD_FACTOR)

    def nonce_to_bin(self, nonce: int) -> int:
        bin_size = (self.max_nonce - self.min_nonce + 1) / self.bins
        idx = int((nonce - self.min_nonce) / bin_size)
        return max(0, min(self.bins - 1, idx))

    def bin_to_range(self, bin_idx: int) -> Tuple[int, int]:
        bin_size = (self.max_nonce - self.min_nonce + 1) / self.bins
        start = int(self.min_nonce + bin_idx * bin_size)
        end = int(min(self.max_nonce, start + bin_size - 1))
        return (start, end)

    def update_bin_weights(self, bin_idx: int, success: bool):
        with self.lock:
            self.bin_weights *= self.DECAY_RATE
            self.bin_attempts[bin_idx] += 1
            if success:
                self.bin_success[bin_idx] += 1
            if self.bin_attempts[bin_idx] > 0:
                sr = self.bin_success[bin_idx] / self.bin_attempts[bin_idx]
                adj = self.REWARD_FACTOR if success else self.PENALTY_FACTOR
                new_weight = sr * adj
                self.bin_weights[bin_idx] = 0.7 * self.bin_weights[bin_idx] + 0.3 * new_weight

    def adapt_bin_count(self):
        if np.sum(self.bin_attempts) < 1000:
            return
        avg_success = np.mean(self.bin_success / (self.bin_attempts + 1e-10))
        if avg_success < 0.3 and self.bins > self.MIN_BINS:
            new_bins = max(self.MIN_BINS, int(self.bins * 0.8))
            logger.info(f"[AdaptiveGenerator] Decreasing bins from {self.bins} to {new_bins} (low success)")
            self.bins = new_bins
            self._initialize_data()
        elif avg_success > 0.7 and self.bins < self.MAX_BINS:
            new_bins = min(self.MAX_BINS, int(self.bins * 1.2))
            logger.info(f"[AdaptiveGenerator] Increasing bins from {self.bins} to {new_bins} (high success)")
            self.bins = new_bins
            self._initialize_data()

    def select_bin(self) -> int:
        with self.lock:
            total_weight = np.sum(self.bin_weights)
            if total_weight <= 0:
                return random.randint(0, self.bins - 1)
            probs = self.bin_weights / total_weight
            return np.random.choice(self.bins, p=probs)

    def _generate_adaptive_batch(self, batch_size: int) -> List[int]:
        nonces = set()
        for _ in range(batch_size * 4):
            bin_idx = self.select_bin()
            start, end = self.bin_to_range(bin_idx)
            center = (start + end) // 2
            std_dev = max(1, (end - start) // 6)
            nonce = int(np.random.normal(center, std_dev))
            nonce = max(start, min(end, nonce))
            nonces.add(nonce)
            if len(nonces) >= batch_size * 4:
                break
        return list(nonces)

    def _get_recent_nonces(self) -> Set[int]:
        recent_nonces = set()
        count = 0
        for row in reversed(self.training_data):
            if count >= self.RECENT_NONCES_SIZE:
                break
            if str(row.get('is_valid', 'False')).lower() == 'true':
                try:
                    recent_nonces.add(int(row['nonce']))
                    count += 1
                except Exception:
                    continue
        return recent_nonces

    def _calculate_metrics(self, nonce: int, recent_set: Set[int]) -> dict:
        bin_repr = bin(nonce)[2:].zfill(64)
        ones = bin_repr.count('1')
        zeros = bin_repr.count('0')
        p1 = ones / 64
        p0 = zeros / 64
        entropy = - (p0 * np.log2(p0 + 1e-10) + p1 * np.log2(p1 + 1e-10))
        zero_density, max_zero_run = self.calculate_penalized_zero_density(bin_repr)
        pattern_score = self.randomx_pattern_score(bin_repr, max_zero_run)
        uniqueness = self.historical_uniqueness(nonce, recent_set)
        return {
            "entropy": max(0.7, min(0.99, entropy)),
            "uniqueness": uniqueness,
            "zero_density": zero_density,
            "pattern_score": pattern_score
        }

    def calculate_penalized_zero_density(self, binary_str: str) -> Tuple[float, int]:
        zero_count = binary_str.count('0')
        zero_density = zero_count / 64
        max_run = 0
        current_run = 0
        for char in binary_str:
            if char == '0':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        if max_run > 6:
            penalty = min(0.2, (max_run - 6) * 0.05)
            zero_density -= penalty
        return max(0.01, zero_density), max_run

    def randomx_pattern_score(self, binary_str: str, max_zero_run: int) -> float:
        score = 1.0
        if max_zero_run > 6:
            score -= min(0.3, (max_zero_run - 6) * 0.05)
        transitions = sum(binary_str[i] != binary_str[i-1] for i in range(1, len(binary_str)))
        transition_ratio = transitions / (len(binary_str) - 1)
        if transition_ratio > 0.5:
            bonus = min(0.15, (transition_ratio - 0.5) * 0.5)
            score += bonus
        for period in [2, 4, 8, 16]:
            pattern = binary_str[:period]
            if binary_str == pattern * (64 // period):
                score -= 0.25
                break
        return max(0.65, min(1.0, score))

    def historical_uniqueness(self, nonce: int, recent_set: Set[int]) -> float:
        if not recent_set:
            return random.uniform(0.85, 0.95)
        nonce_arr = np.full(len(recent_set), nonce, dtype=np.uint64)
        recent_arr = np.array(list(recent_set), dtype=np.uint64)
        xor_arr = np.bitwise_xor(nonce_arr, recent_arr)
        popcnts = np.vectorize(lambda x: bin(x).count('1'))(xor_arr)
        avg_distance = np.mean(popcnts)
        return max(0.8, min(0.99, avg_distance / 64))

    def _validate_batch(self, nonces: List[int], block_data: dict) -> List[bool]:
        if not nonces:
            return []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=self.MAX_VALIDATION_WORKERS) as executor:
            futures = {executor.submit(self.validator.validate, nonce, block_data): nonce for nonce in nonces}
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Validation failed: {e}")
                    results.append(False)
        return results

    def _save_nonces(self, nonces: List[dict], path_key='generated_nonces'):
        if not nonces:
            return
        output_path = self._get_data_path(path_key)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = self.FIELDNAMES
        try:
            file_exists = os.path.exists(output_path)
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows([{k: v for k, v in item.items() if k in fieldnames} for item in nonces])
        except Exception as e:
            logger.error(f"[AdaptiveGenerator] Failed to save nonces: {e}")

    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        start_time = time.time()
        if self._should_refresh():
            self._initialize_data()
        with self.lock:
            if time.time() - self.last_adaptation > 60:
                self.adapt_bin_count()
                self.last_adaptation = time.time()
            recent_nonces = self._get_recent_nonces()
            candidates = self._generate_adaptive_batch(batch_size)
            validation_results = self._validate_batch(candidates, block_data)
            valid_nonces = []
            for nonce, is_valid in zip(candidates, validation_results):
                if len(valid_nonces) >= batch_size:
                    break
                metrics = self._calculate_metrics(nonce, recent_nonces)
                nonce_data = {
                    "nonce": nonce,
                    "block_height": block_height,
                    "generator": self.generator_name,
                    "is_valid": bool(is_valid),
                    **metrics
                }
                valid_nonces.append(nonce_data)
                self.update_bin_weights(self.nonce_to_bin(nonce), bool(is_valid))
                recent_nonces.add(nonce)
            if valid_nonces:
                self._save_nonces(valid_nonces, path_key='generated_nonces')
            elapsed = time.time() - start_time
            success_rate = len([n for n in valid_nonces if n["is_valid"]]) / len(candidates) if candidates else 0
            logger.info(
                f"[AdaptiveGenerator] Block {block_height}: "
                f"{len(valid_nonces)}/{batch_size} valid, "
                f"{len(candidates)} tried, "
                f"{success_rate:.2%} success, "
                f"{elapsed:.2f}s elapsed, "
                f"{len(valid_nonces)/elapsed if elapsed else 0:.1f} nonces/sec"
            )
            return valid_nonces