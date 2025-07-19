import csv
import os
import threading
import logging
import random
import math
import secrets
from collections import defaultdict
from datetime import datetime
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, conint
from typing import Optional, Dict, List, Set, Any
from iazar.generator.config_loader import config_loader
from iazar.generator.randomx_validator import RandomXValidator
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class GeneratorConfig(BaseModel):
    num_groups: conint(ge=2, le=256) = 16
    top_groups: conint(ge=1, le=16) = 4
    batch_size: conint(ge=1, le=100_000) = 500
    max_attempts: conint(ge=1, le=10_000_000) = 10_000
    data_paths: Dict[str, str] = {
        "training_data": "C:/zarturxia/src/iazar/data/nonce_training_data.csv",
        "generated_nonces": "C:/zarturxia/src/iazar/data/nonces_exitosos.csv"
    }

    @model_validator(mode='after')
    def check_groups(self) -> 'GeneratorConfig':
        if self.top_groups > self.num_groups:
            raise ValueError("top_groups cannot be greater than num_groups")
        return self

    model_config = ConfigDict(extra="allow")

class BaseNonceGenerator(ABC):
    """Clase base abstracta para generadores de nonces."""
    def __init__(self, generator_name: str, config: dict = None):
        self.generator_name = generator_name
        self.config = config or {}

    @abstractmethod
    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> list:
        """Debe implementar la lógica de generación de nonces."""
        pass

class GroupedTopNonceGenerator:
    FIELDNAMES = [
        "nonce", "entropy", "uniqueness", "zero_density",
        "pattern_score", "is_valid", "block_height", "generator"
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        user_config = config or config_loader.load_config()
        merged_config = {**GeneratorConfig().model_dump(), **user_config}
        self.config = GeneratorConfig(**merged_config)
        self.lock = threading.Lock()
        self._last_mtime = 0
        self.training_data = self._load_training_data()
        self.validator = RandomXValidator(self.config.model_dump())
        logging.info(f"Loaded {len(self.training_data)} training nonces.")

    def _get_data_path(self, key: str) -> str:
        return self.config.data_paths.get(key, "")

    def _maybe_reload_training_data(self):
        """Reload training data only if the file changed (for long-running miners)."""
        path = self._get_data_path('training_data')
        if not path or not os.path.exists(path):
            return
        current_mtime = os.path.getmtime(path)
        if current_mtime > self._last_mtime:
            self.training_data = self._load_training_data()
            self._last_mtime = current_mtime
            logging.info("Training data automatically reloaded.")

    def _load_training_data(self) -> List[dict]:
        path = self._get_data_path('training_data')
        if not path or not os.path.exists(path):
            logging.warning(f"Training data file does not exist: {path}")
            return []
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return [row for row in reader if row.get('nonce')]
        except Exception as e:
            logging.error(f"Error loading training data: {e}")
            return []

    def _analyze_group_performance(self) -> Dict[int, float]:
        group_success = defaultdict(int)
        group_total = defaultdict(int)
        for row in self.training_data:
            try:
                nonce = int(row['nonce'])
                group_id = nonce % self.config.num_groups
                group_total[group_id] += 1
                if row.get('is_valid', 'False').lower() == 'true':
                    group_success[group_id] += 1
            except Exception as e:
                logging.debug(f"Skipping invalid row: {e}")
        success_rates = {gid: (group_success[gid] / group_total[gid]) if group_total[gid] > 0 else 0.0 for gid in group_total}
        logging.info(f"Group performance analysis: {success_rates}")
        return success_rates

    def _select_top_groups(self) -> List[int]:
        success_rates = self._analyze_group_performance()
        available_groups = list(success_rates.keys())
        top_n = min(self.config.top_groups, len(available_groups))
        if not available_groups:
            logging.warning("No groups available for selection")
            return []
        sorted_groups = sorted(
            available_groups,
            key=lambda g: (success_rates[g], g),
            reverse=True
        )
        top_selected = sorted_groups[:top_n]
        logging.info(f"Selected top groups: {top_selected}")
        return top_selected

    def _generate_nonce_for_group(self, group_id: int, use_csrandom=True) -> int:
        """Generate new nonce using CSPRNG for mining. Recommended for production."""
        base_nonce = secrets.randbits(64) if use_csrandom else random.getrandbits(64)
        adjustment = (group_id - (base_nonce % self.config.num_groups)) % self.config.num_groups
        target_nonce = (base_nonce + adjustment) % (1 << 64)
        return target_nonce

    def _calc_shannon_entropy(self, binary: str) -> float:
        if not binary:
            return 0.0
        char_counts = {}
        for char in binary:
            char_counts[char] = char_counts.get(char, 0) + 1
        entropy = 0.0
        total = len(binary)
        for count in char_counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def _calc_zero_density(self, binary: str) -> float:
        return binary.count('0') / len(binary) if binary else 0.0

    def _calc_pattern_score(self, binary: str) -> float:
        if len(binary) < 4:
            return 1.0
        max_run = 0
        current_run = 1
        last_char = binary[0]
        for char in binary[1:]:
            if char == last_char:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
            last_char = char
        return 1.0 - (max_run / len(binary))

    def _calc_uniqueness(self, nonce: int, seen_set: Set[int]) -> float:
        return 1.0 if nonce not in seen_set else 0.0

    def _validate_block_data(self, block_data: dict):
        required_keys = {'block_hash', 'difficulty', 'height'}
        if not required_keys.issubset(block_data):
            raise ValueError(f"Block data missing required keys: {required_keys - block_data.keys()}")

    def _buffered_csv_writer(self, output_path: str, valid_nonces: List[dict], buffer_size=1000):
        """Buffered writer for high throughput (batching to reduce disk IO)."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        file_exists = os.path.exists(output_path)
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            buffer = []
            for nonce_data in valid_nonces:
                buffer.append(nonce_data)
                if len(buffer) >= buffer_size:
                    writer.writerows(buffer)
                    buffer.clear()
            if buffer:
                writer.writerows(buffer)

    def generate_batch(self, block_height: int, block_data: dict) -> List[dict]:
        """Generate batch of nonces targeting top-performing groups"""
        self._maybe_reload_training_data()
        self._validate_block_data(block_data)
        top_groups = self._select_top_groups()
        if not top_groups:
            logging.error("No groups available for generation")
            return []
        batch_size = self.config.batch_size
        nonces = []
        attempts = 0
        max_attempts = self.config.max_attempts
        seen: Set[int] = set()
        logging.info(f"Generating {batch_size} nonces for block {block_height} targeting groups: {top_groups}")
        while len(nonces) < batch_size and attempts < max_attempts:
            group_id = random.choice(top_groups)
            nonce = self._generate_nonce_for_group(group_id)
            if nonce in seen:
                attempts += 1
                continue
            bin_str = bin(nonce)[2:].zfill(64)
            metrics = {
                "entropy": self._calc_shannon_entropy(bin_str),
                "uniqueness": self._calc_uniqueness(nonce, seen),
                "zero_density": self._calc_zero_density(bin_str),
                "pattern_score": self._calc_pattern_score(bin_str),
                "is_valid": False
            }
            nonce_data = {
                "nonce": nonce,
                "block_height": block_height,
                "generator": "GroupedTopNonceGenerator",
                **metrics
            }
            try:
                nonce_data["is_valid"] = self.validator.validate(nonce=nonce, block_data=block_data)
            except Exception as e:
                logging.error(f"Validation error: {e}")
                nonce_data["is_valid"] = False
            seen.add(nonce)
            nonces.append(nonce_data)
            attempts += 1
        valid_count = sum(1 for n in nonces if n["is_valid"])
        logging.info(f"Generated {len(nonces)} nonces ({valid_count} valid) in {attempts} attempts")
        return nonces

    def save_nonces(self, nonces: List[dict]):
        """Save valid nonces to output file, thread-safe, with buffering and deduplication"""
        if not nonces:
            return
        output_path = self._get_data_path('generated_nonces')
        if not output_path:
            logging.error("Output path not configured")
            return
        valid_nonces = [
            {k: v for k, v in n.items() if k in self.FIELDNAMES}
            for n in nonces if n.get("is_valid")
        ]
        if not valid_nonces:
            logging.info("No valid nonces to save")
            return
        with self.lock:
            self._buffered_csv_writer(output_path, valid_nonces, buffer_size=2000)
        logging.info(f"Saved {len(valid_nonces)} valid nonces to {output_path}")

    def run_generation(self, block_height: int, block_data: dict):
        """Public interface for batch generation (used by orchestrator)."""
        try:
            nonces = self.generate_batch(block_height, block_data)
            self.save_nonces(nonces)
        except Exception as e:
            logging.critical(f"Batch generation failed: {e}", exc_info=True)
