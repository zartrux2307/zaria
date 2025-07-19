import numpy as np
import threading
import csv
import os
import time
import logging
from typing import Optional, Dict, List

from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.generator.config_loader import config_loader
from iazar.generator.randomx_validator import RandomXValidator

logger = logging.getLogger("RandomGenerator")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

class RandomGenerator(BaseNonceGenerator):
    """
    Industrial-Grade Random Nonce Generator for Monero/RandomX Mining.
    - Uniform 64-bit nonces, vectorized, no repetition in batch.
    - Parallel RandomX validation.
    - Full metrics: entropy, uniqueness (bitwise, vectorized), zero_density, pattern_score.
    - Thread-safe. Batch-prepared. Mainnet ready.
    """
    BATCH_CANDIDATE_FACTOR = 4
    MAX_VALIDATION_WORKERS = min(os.cpu_count() or 4, 16)
    RECENT_NONCES_SIZE = 500

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("random", config)
        self.lock = threading.RLock()
        self.validator = RandomXValidator(self.config)
        self._last_block_height = None
        self._recent_arr_cache = None
        logger.info("[RandomGenerator] Ready (optimized, mainnet-proof)")

    def _generate_candidate_nonces(self, count: int) -> np.ndarray:
        """Generate unique random 64-bit nonces using vectorization."""
        nonces = np.random.randint(0, 2**64, size=int(count * 1.2), dtype=np.uint64)
        unique = np.unique(nonces)
        if unique.size < count:
            needed = count - unique.size
            extra = np.random.randint(0, 2**64, size=needed, dtype=np.uint64)
            unique = np.unique(np.concatenate([unique, extra]))
        return unique[:count]

    def _calc_metrics(self, nonce: int, recent_arr: np.ndarray) -> dict:
        """
        Enterprise metrics for nonce quality assessment.
        """
        bin_str = bin(nonce)[2:].zfill(64)
        p0 = bin_str.count('0') / 64
        p1 = 1.0 - p0
        entropy = 0.0
        if 0 < p0 < 1:
            entropy = - (p0 * np.log2(p0) + p1 * np.log2(p1))
        uniqueness = self._calc_uniqueness_vec(nonce, recent_arr)
        max_run_0 = max((len(run) for run in bin_str.split('1')), default=0)
        max_run_1 = max((len(run) for run in bin_str.split('0')), default=0)
        run_penalty = min(0.5, max(max_run_0, max_run_1) / 32)
        period_penalty = 0
        for p in [2, 4, 8, 16]:
            pattern = bin_str[:p]
            if bin_str == pattern * (64 // p):
                period_penalty += 0.15
        pattern_score = max(0.5, 1.0 - run_penalty - period_penalty)
        return {
            "entropy": round(entropy, 5),
            "uniqueness": round(uniqueness, 5),
            "zero_density": round(p0, 5),
            "pattern_score": round(pattern_score, 5)
        }

    def _calc_uniqueness_vec(self, nonce: int, recent_arr: Optional[np.ndarray]) -> float:
        """Mean Hamming distance between nonce and recent nonces, vectorized."""
        if recent_arr is None or recent_arr.size == 0:
            return 1.0
        xor = np.bitwise_xor(recent_arr, nonce)
        hamming = np.unpackbits(xor.view(np.uint8)).reshape(-1, 64).sum(axis=1)
        return max(0.8, min(0.99, hamming.mean() / 64))

    def _get_recent_nonces_arr(self, block_height: int) -> np.ndarray:
        """
        Returns last RECENT_NONCES_SIZE valid nonces from training_data (cached per block).
        """
        if self._last_block_height == block_height and self._recent_arr_cache is not None:
            return self._recent_arr_cache
        recent = []
        count = 0
        for row in reversed(self.training_data):
            if count >= self.RECENT_NONCES_SIZE:
                break
            if row.get('is_valid', 'false').lower() == 'true':
                try:
                    recent.append(int(row['nonce']))
                    count += 1
                except Exception:
                    continue
        arr = np.array(recent, dtype=np.uint64) if recent else np.array([], dtype=np.uint64)
        self._recent_arr_cache = arr
        self._last_block_height = block_height
        return arr

    def _validate_batch(self, nonces: np.ndarray, block_data: dict) -> List[bool]:
        """
        Validates batch of nonces via RandomXValidator (concurrent).
        """
        from concurrent.futures import ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=self.MAX_VALIDATION_WORKERS) as executor:
            futs = [executor.submit(self.validator.validate, int(n), block_data) for n in nonces]
            for f in futs:
                try:
                    results.append(f.result())
                except Exception as e:
                    logger.error(f"[RandomGenerator] Validation error: {e}")
                    results.append(False)
        return results

    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        """
        Uniform random batch generation, vectorized and thread-safe.
        """
        t0 = time.time()
        with self.lock:
            self.training_data = self._load_training_data()
            recent_arr = self._get_recent_nonces_arr(block_height)
            valid_nonces = []
            tries = 0
            batch_factor = self.BATCH_CANDIDATE_FACTOR

            while len(valid_nonces) < batch_size and tries < 8:
                candidates = self._generate_candidate_nonces(batch_size * batch_factor)
                metrics_list = [self._calc_metrics(n, recent_arr) for n in candidates]
                valids = self._validate_batch(candidates, block_data)
                for i, is_valid in enumerate(valids):
                    if is_valid and len(valid_nonces) < batch_size:
                        n = candidates[i]
                        nonce_data = {
                            "nonce": int(n),
                            "block_height": block_height,
                            "generator": self.generator_name,
                            "is_valid": True,
                            **metrics_list[i]
                        }
                        valid_nonces.append(nonce_data)
                        if recent_arr.size < self.RECENT_NONCES_SIZE:
                            recent_arr = np.append(recent_arr, n)
                tries += 1
                batch_factor = min(batch_factor * 2, 32)

            if valid_nonces:
                self._save_nonces_batch(valid_nonces)
            elapsed = time.time() - t0
            logger.info(
                f"[RandomGenerator] Block {block_height}: "
                f"{len(valid_nonces)}/{batch_size} valid, "
                f"{elapsed:.2f}s elapsed"
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
                writer.writerows([{k: v for k, v in item.items() if k in fieldnames} for item in nonces])
        except IOError as e:
            logger.error(f"[RandomGenerator] Save error: {e}")

# Uncomment for stress testing
# if __name__ == "__main__":
#     from iazar.generator.config_loader import config_loader
#     generator = RandomGenerator(config_loader.load_config())
#     fake_block = {...}
#     result = generator.run_generation(fake_block["height"], fake_block, batch_size=2000)
#     print(f"Nonces generados: {len(result)}")
