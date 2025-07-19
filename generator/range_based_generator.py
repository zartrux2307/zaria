import os
import csv
import time
import logging
import threading
from typing import List, Tuple, Dict, Optional, Set

import numpy as np

from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.generator.config_loader import config_loader
from iazar.generator.randomx_validator import RandomXValidator

try:
    from pybloom_live import BloomFilter
    BLOOM_AVAILABLE = True
except ImportError:
    BLOOM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class RangeBasedGenerator(BaseNonceGenerator):
    """
    Enterprise-grade Range-Based Nonce Generator for Monero/RandomX Mining.
    Optimized for throughput, robustness, and monitoring in industrial mining environments.
    """

    MIN_RANGE_SIZE = 100_000
    DEFAULT_RANGE = [(0, 2**64 - 1)]
    RECENT_NONCES_SIZE = 500
    DATA_REFRESH_INTERVAL = 300  # seconds
    DISK_WRITE_BUFFER_SIZE = 1000
    FIELDNAMES = ["nonce", "entropy", "uniqueness", "zero_density", "pattern_score", "is_valid", "block_height"]

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("range_based", config)
        self.lock = threading.RLock()
        self.last_refresh = 0
        self.validator = RandomXValidator(self.config)
        self.disk_write_buffer = []
        self.bloom = None
        self.total_generated = 0
        self.total_saved = 0
        self.last_stats_time = time.time()
        if BLOOM_AVAILABLE:
            self.bloom = BloomFilter(capacity=10_000_000, error_rate=0.01)
        self.training_data = []
        self.optimal_ranges = self.DEFAULT_RANGE
        self.recent_nonces = np.array([], dtype=np.uint64)
        self._initialize_data()
        logging.info(f"[RangeBasedGenerator] Initialized with {len(self.optimal_ranges)} optimal ranges")

    def _initialize_data(self):
        with self.lock:
            self.training_data = self._load_training_data()
            self.optimal_ranges = self._calculate_optimal_ranges()
            self.recent_nonces = self._get_recent_nonces()
            self.last_refresh = time.time()

    def _load_training_data(self) -> List[dict]:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nonce_training_data.csv")
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [row for row in reader]

    def _should_refresh(self):
        return time.time() - self.last_refresh > self.DATA_REFRESH_INTERVAL

    def _get_recent_nonces(self) -> np.ndarray:
        if not self.training_data:
            return np.array([], dtype=np.uint64)
        nonces = []
        for row in reversed(self.training_data):
            if str(row.get('is_valid', 'False')).lower() == 'true' and 'nonce' in row:
                try:
                    nonces.append(int(row['nonce']))
                    if len(nonces) >= self.RECENT_NONCES_SIZE:
                        break
                except ValueError:
                    continue
        return np.array(nonces, dtype=np.uint64)

    def _calculate_optimal_ranges(self) -> List[Tuple[int, int]]:
        if not self.training_data:
            return self.DEFAULT_RANGE
        valid_nonces = [
            int(row['nonce']) for row in self.training_data
            if str(row.get('is_valid', 'False')).lower() == 'true' and 'nonce' in row
        ]
        if len(valid_nonces) < 50:
            return self.DEFAULT_RANGE
        nonce_array = np.array(valid_nonces, dtype=np.uint64)
        sorted_nonces = np.sort(nonce_array)
        density_ranges = []
        window_size = max(100_000, len(sorted_nonces) // 20)
        for i in range(0, len(sorted_nonces), window_size // 2):
            if i + window_size >= len(sorted_nonces):
                break
            window = sorted_nonces[i:i+window_size]
            if window[-1] == window[0]:
                continue
            density = len(window) / (window[-1] - window[0] + 1)
            if density > 1e-8:
                density_ranges.append((window[0], window[-1]))
        # Merge overlapping
        merged = []
        for start, end in sorted(density_ranges):
            if not merged:
                merged.append((start, end))
            else:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    merged[-1] = (last_start, max(last_end, end))
                else:
                    merged.append((start, end))
        return [
            (max(0, s), min(2**64-1, e))
            for s, e in merged
            if e - s >= self.MIN_RANGE_SIZE
        ] or self.DEFAULT_RANGE

    def _generate_nonces_vectorized(self, block_height: int, count: int) -> List[dict]:
        if not self.optimal_ranges:
            return []
        range_sizes = [e - s for s, e in self.optimal_ranges]
        total_size = sum(range_sizes)
        range_probs = [size / total_size for size in range_sizes]
        range_counts = np.random.multinomial(count, range_probs)
        all_nonces = []
        for (start, end), n in zip(self.optimal_ranges, range_counts):
            if n == 0:
                continue
            if end - start < 10_000:
                nonces = np.random.randint(start, end+1, size=n, dtype=np.uint64)
            else:
                centers = np.random.randint(start, end, size=n)
                std_devs = max((end - start) / 6, 1)
                nonces = np.random.normal(centers, std_devs).astype(np.uint64)
                nonces = np.clip(nonces, start, end)
            all_nonces.extend(nonces.tolist())
        all_nonces = list(set(all_nonces))
        return self._calculate_metrics_batch(block_height, all_nonces)

    def _calculate_metrics_batch(self, block_height: int, nonces: List[int]) -> List[dict]:
        results = []
        recent_set = set(self.recent_nonces.tolist()) if self.recent_nonces.size > 0 else set()
        for nonce in nonces:
            bin_repr = bin(nonce)[2:].zfill(64)
            entropy = self._shannon_entropy(bin_repr)
            uniqueness = self._hamming_uniqueness(nonce, recent_set)
            zero_density = bin_repr.count('0') / 64
            pattern_score = self._pattern_score(bin_repr)
            results.append({
                "nonce": nonce,
                "block_height": block_height,
                "entropy": round(entropy, 5),
                "uniqueness": round(uniqueness, 5),
                "zero_density": round(zero_density, 5),
                "pattern_score": round(pattern_score, 5),
                "is_valid": True
            })
        return results

    def _shannon_entropy(self, bin_repr: str) -> float:
        counts = np.bincount([int(b) for b in bin_repr])
        probabilities = counts / len(bin_repr)
        return -np.sum(probabilities * np.log2(probabilities + 1e-12))

    def _hamming_uniqueness(self, nonce: int, recent_set: Set[int]) -> float:
        if not recent_set:
            return 1.0
        nonce_bin = np.array([int(b) for b in bin(nonce)[2:].zfill(64)])
        total_distance = 0
        for other in recent_set:
            other_bin = np.array([int(b) for b in bin(other)[2:].zfill(64)])
            total_distance += np.sum(nonce_bin != other_bin)
        return total_distance / (64 * len(recent_set))

    def _pattern_score(self, bin_str: str) -> float:
        arr = np.array([int(b) for b in bin_str])
        diff = np.diff(arr, prepend=arr[0]-1, append=arr[-1]-1)
        run_starts = np.where(diff != 0)[0]
        run_lengths = np.diff(run_starts)
        max_run = np.max(run_lengths) if run_lengths.size > 0 else 0
        run_penalty = min(0.5, max_run / 32)
        autocorr_penalty = 0
        for shift in [1, 2, 4, 8]:
            shifted = np.roll(arr, shift)
            correlation = np.sum(arr == shifted) / len(arr)
            if correlation > 0.7:
                autocorr_penalty += 0.1 * (correlation - 0.7) / 0.3
        return max(0.5, 1.0 - run_penalty - autocorr_penalty)

    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        if self._should_refresh():
            self._initialize_data()
        with self.lock:
            nonce_data_list = self._generate_nonces_vectorized(block_height, batch_size * 2)
            valid_nonces = []
            for data in nonce_data_list:
                if BLOOM_AVAILABLE and self.bloom:
                    if data["nonce"] in self.bloom:
                        continue
                try:
                    is_valid = self.validator.validate(
                        nonce=data["nonce"],
                        block_data=block_data
                    )
                    if is_valid:
                        data["is_valid"] = True
                        valid_nonces.append(data)
                        if BLOOM_AVAILABLE and self.bloom:
                            self.bloom.add(data["nonce"])
                except Exception as e:
                    logging.error(f"Validation error: {e}")
                    continue
                if len(valid_nonces) >= batch_size:
                    break
            if valid_nonces:
                self._save_nonces_buffered(valid_nonces)
            self.total_generated += len(nonce_data_list)
            self.total_saved += len(valid_nonces)
            self._log_stats()
            return valid_nonces

    def _save_nonces_buffered(self, nonces: List[dict]):
        self.disk_write_buffer.extend(nonces)
        if len(self.disk_write_buffer) >= self.DISK_WRITE_BUFFER_SIZE:
            output_path = self._get_data_path('generated_nonces')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with self.lock:
                file_exists = os.path.exists(output_path)
                with open(output_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerows(self.disk_write_buffer)
            logging.info(f"Batch write: {len(self.disk_write_buffer)} nonces saved to {output_path}")
            self.disk_write_buffer = []

    def flush(self):
        if not self.disk_write_buffer:
            return
        output_path = self._get_data_path('generated_nonces')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with self.lock:
            file_exists = os.path.exists(output_path)
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self.disk_write_buffer)
        logging.info(f"Final flush: {len(self.disk_write_buffer)} nonces saved to {output_path}")
        self.disk_write_buffer = []

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_generated": self.total_generated,
            "total_saved": self.total_saved,
            "buffer_size": len(self.disk_write_buffer),
            "ranges": len(self.optimal_ranges)
        }

    def _get_data_path(self, name: str) -> str:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{name}.csv")

    def _log_stats(self):
        if time.time() - self.last_stats_time > 60:
            stats = self.get_stats()
            logging.info(f"Stats: {stats}")
            self.last_stats_time = time.time()
