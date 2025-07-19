import os
import sys
import csv
import time
import json
import queue
import signal
import logging
import threading
import traceback
import secrets
from pathlib import Path
from typing import Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importar Generadores y Evaluadores
from iazar.generator.range_based_generator import RangeBasedGenerator
from iazar.generator.ml_based_generator import MLBasedGenerator
from iazar.generator.entropy_based_generator import EntropyBasedGenerator
from iazar.generator.sequence_based_generator import SequenceBasedGenerator
from iazar.generator.adaptive_generator import AdaptiveGenerator
from iazar.generator.hybrid_generator import HybridGenerator
from iazar.generator.random_generator import RandomGenerator

from iazar.evaluation import (
    NonceCalculator,
    CorrelationAnalyzer,
    DistributionAnalyzer,
    EntropyAnalyzer,
    NonceQualityFilter,
    SurvivalAnalyzer
)

# Logging industrial
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("orchestrator.log", encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger("NonceOrchestrator")

# Paths y Constantes
CONFIG_PATH = "C:/zarturxia/src/iazar/config/global_config.json"
DATA_DIR = "C:/zarturxia/src/iazar/data"
PROXY_OUT_PATH = os.path.join(DATA_DIR, "proxy_outbox.csv")
NONCES_EXITOSOS_PATH = os.path.join(DATA_DIR, "nonces_exitosos.csv")
CSV_FIELDNAMES = ["nonce", "entropy", "uniqueness", "zero_density", "pattern_score", "is_valid", "block_height"]

GENERATOR_CLASSES = [
    RangeBasedGenerator, MLBasedGenerator, EntropyBasedGenerator,
    SequenceBasedGenerator, AdaptiveGenerator, HybridGenerator, RandomGenerator
]
EVALUATION_CLASSES = [
   NonceCalculator, CorrelationAnalyzer, DistributionAnalyzer,
    EntropyAnalyzer, NonceQualityFilter, SurvivalAnalyzer
]

# CSV Writer con buffer
class CSVWriter:
    def __init__(self, path: str, buffer_size: int = 1000):
        self.path = path
        self.buffer = []
        self.buffer_size = buffer_size
        self.lock = threading.RLock()
        self.header_written = os.path.exists(path) and os.path.getsize(path) > 0

    def write(self, row: dict):
        with self.lock:
            self.buffer.append(row)
            if len(self.buffer) >= self.buffer_size:
                self.flush()

    def flush(self):
        with self.lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            file_exists = os.path.exists(self.path)
            with open(self.path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                if not self.header_written:
                    writer.writeheader()
                    self.header_written = True
                writer.writerows(self.buffer)
            self.buffer = []

    def close(self):
        self.flush()

# Validación de nonces
def validate_nonce_fields(nonce_dict: dict) -> bool:
    try:
        for k in CSV_FIELDNAMES:
            if k not in nonce_dict:
                logger.error(f"Missing {k} in {nonce_dict}")
                return False
        entropy = float(nonce_dict["entropy"])
        uniqueness = float(nonce_dict["uniqueness"])
        zero_density = float(nonce_dict["zero_density"])
        pattern_score = float(nonce_dict["pattern_score"])
        block_height = int(nonce_dict["block_height"])
        if not (0.0 <= entropy <= 8.0 and 0.0 <= uniqueness <= 1.0 and 0.0 <= zero_density <= 1.0):
            logger.error(f"Invalid nonce field values: {nonce_dict}")
            return False
        return True
    except Exception as e:
        logger.error(f"Exception in validate_nonce_fields: {e} ({nonce_dict})")
        return False

# BlockDataProvider (Observer pattern)
class BlockDataProvider:
    def __init__(self):
        self._subscribers: List[Callable] = []
        self._lock = threading.Lock()
        self._current_block = {"block_height": 0, "block_data": {}}

    def register(self, callback: Callable):
        with self._lock:
            self._subscribers.append(callback)

    def update(self, block_height: int, block_data: dict):
        with self._lock:
            self._current_block = {"block_height": block_height, "block_data": block_data}
            for sub in self._subscribers:
                sub(self._current_block)

    def get(self):
        with self._lock:
            return self._current_block["block_height"], self._current_block["block_data"]

# Safe put (reintento, backoff)
def safe_put(q, item, max_retries=5):
    for attempt in range(max_retries):
        try:
            q.put(item, timeout=2**attempt)
            return True
        except queue.Full:
            logger.warning(f"Proxy queue full (attempt {attempt+1})")
    logger.error("Failed to enqueue nonce after max retries")
    return False

# Scheduler para evaluadores
class EvaluatorSchedule:
    def __init__(self, schedule_dict: dict):
        self.schedule = schedule_dict
        self.last_run = {k: 0 for k in schedule_dict.keys()}

    def should_run(self, name: str, now: float) -> bool:
        return now - self.last_run.get(name, 0) >= self.schedule.get(name, 60)

    def record_run(self, name: str, now: float):
        self.last_run[name] = now

# Orquestador principal
class NonceOrchestrator:
    BATCH_SIZE = 500
    GENERATOR_POOL = 6
    QUEUE_MAXSIZE = 5000

    def __init__(self):
        self.config = self._load_config()
        self.generator_priorities = self.config.get("generator_priorities", {})
        self.generators = self._init_generators()
        self.evaluators = self._init_evaluators()
        self.proxy_out_queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self.block_provider = BlockDataProvider()
        self.nonces_writer = CSVWriter(NONCES_EXITOSOS_PATH, buffer_size=200)
        self.proxy_writer = CSVWriter(PROXY_OUT_PATH, buffer_size=200)
        self.running = False
        self.lock = threading.RLock()
        self.eval_schedule = EvaluatorSchedule({
            "NonceCalculator": 60, "CorrelationAnalyzer": 300, "DistributionAnalyzer": 600,
            "EntropyAnalyzer": 600, "NonceQualityFilter": 180, "SurvivalAnalyzer": 3600
        })
        logger.info("NonceOrchestrator initialized with enterprise-grade reliability")

    def _load_config(self):
        default_config = {
            "generator_priorities": {
                "RangeBasedGenerator": 10,
                "MLBasedGenerator": 9,
                "EntropyBasedGenerator": 8,
                "SequenceBasedGenerator": 7,
                "AdaptiveGenerator": 6,
                "HybridGenerator": 8,
                "RandomGenerator": 5
            },
            "ia": {
                "min_entropy": 0.85,
                "min_uniqueness": 0.8,
                "max_zero_density": 0.15,
                "min_pattern_score": 0.8,
                "quality_filter": True,
                "target_entropy": 0.92
            },
            "performance_settings": {
                "analysis_interval": 300,
                "generation_interval": 1,
                "entropy_window": 100000
            }
        }
        
        try:
            if not os.path.exists(CONFIG_PATH):
                logger.warning("Config file not found, creating default")
                os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
                with open(CONFIG_PATH, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
            
            with open(CONFIG_PATH, encoding='utf-8') as f:
                config = json.load(f)
                logger.info("Configuration loaded successfully")
                return config
        except Exception as e:
            logger.critical(f"Cannot load config: {e}. Using default config.")
            return default_config

    def _init_generators(self):
        priorities = self.generator_priorities
        classes = sorted(GENERATOR_CLASSES, key=lambda x: priorities.get(x.__name__, 5), reverse=True)
        return {cls.__name__: cls(self.config) for cls in classes}

    def _init_evaluators(self):
        evaluators = {}
        for cls in EVALUATION_CLASSES:
            try:
                evaluators[cls.__name__] = cls(self.config)
                logger.info(f"Initialized {cls.__name__} with config")
            except TypeError as e:
                if "takes no arguments" in str(e) or "positional arguments" in str(e):
                    evaluators[cls.__name__] = cls()
                    logger.info(f"Initialized {cls.__name__} without config")
                else:
                    logger.error(f"Error initializing {cls.__name__}: {e}")
                    raise
            except Exception as e:
                logger.error(f"Error initializing {cls.__name__}: {e}")
                raise
        return evaluators

    def handle_signal(self, sig, frame):
        logger.warning(f"Signal received: {sig}. Shutting down orchestrator...")
        self.running = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Shutting down, flushing writers and closing generators.")
        self.nonces_writer.close()
        self.proxy_writer.close()
        for gen in self.generators.values():
            if hasattr(gen, "close"):
                gen.close()
        logger.info("All resources closed cleanly.")

    def _run_generator(self, name, generator):
        t0 = time.time()
        block_height, block_data = self.block_provider.get()
        try:
            batch = generator.run_generation(block_height, block_data, self.BATCH_SIZE)
            valid_batch = [n for n in batch if validate_nonce_fields(n)]
            latency = time.time() - t0
            logger.info(f"{name}: {len(valid_batch)} valid nonces in {latency:.2f}s")
            
            # CORRECCIÓN: Manejo de generadores sin FIELDNAMES
            if hasattr(generator, 'FIELDNAMES'):
                fieldnames = generator.FIELDNAMES
            else:
                fieldnames = CSV_FIELDNAMES
                
            self._save_generated_nonces(valid_batch, fieldnames)
            return valid_batch
        except Exception as e:
            logger.error(f"Generator {name} failed: {traceback.format_exc()}")
            return []
            
    def _save_generated_nonces(self, nonces, fieldnames):
        """Guarda nonces usando los campos especificados"""
        for nonce in nonces:
            record = {field: nonce.get(field, '') for field in fieldnames}
            self.nonces_writer.write(record)

    def _send_to_proxy(self):
        to_send = []
        while len(to_send) < 500 and not self.proxy_out_queue.empty():
            try:
                to_send.append(self.proxy_out_queue.get_nowait())
            except queue.Empty:
                break
        if to_send:
            for nonce in to_send:
                self.proxy_writer.write(nonce)
            logger.info(f"Sent {len(to_send)} nonces to proxy")

    def main_loop(self):
        self.running = True
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

        logger.info("Nonce Orchestrator main loop started.")
        last_block = 0

        while self.running:
            t0 = time.time()
            # (1) Simula o integra actualización de bloque
            block_height = int(time.time()) % 2**32
            block_data = {
                "blob": secrets.token_bytes(76),
                "seed": secrets.token_hex(32),
                "target": 2**256 - 1,
                "dummy": "data"
            }
            if block_height != last_block:
                self.block_provider.update(block_height, block_data)
                last_block = block_height

            # (2) Generadores en paralelo con ThreadPoolExecutor
            results = []
            with ThreadPoolExecutor(max_workers=self.GENERATOR_POOL) as executor:
                futures = {executor.submit(self._run_generator, name, gen): name
                           for name, gen in self.generators.items()}
                for future in as_completed(futures):
                    try:
                        results.extend(future.result())
                    except Exception as e:
                        logger.error(f"Generator {futures[future]} failed: {traceback.format_exc()}")

            # (3) Encolar nonces para proxy
            for nonce in results:
                safe_put(self.proxy_out_queue, nonce)
            self.nonces_writer.flush()

            # (4) Enviar nonces a proxy en lotes
            self._send_to_proxy()
            self.proxy_writer.flush()

            # (5) Ejecución programada de evaluadores
            now = time.time()
            for name, evaluator in self.evaluators.items():
                try:
                    if self.eval_schedule.should_run(name, now):
                        if hasattr(evaluator, 'run'):
                            evaluator.run()
                        elif hasattr(evaluator, 'run_analysis'):
                            evaluator.run_analysis()
                        self.eval_schedule.record_run(name, now)
                        logger.info(f"Evaluator {name} executed.")
                except Exception as e:
                    logger.error(f"Evaluator error in {name}: {traceback.format_exc()}")

            logger.info(f"Orchestrator main cycle finished in {time.time()-t0:.2f}s.")
            time.sleep(max(1, 1.0 - (time.time() - t0)))

        logger.info("Nonce Orchestrator main loop exited.")

# Lanzador principal
if __name__ == "__main__":
    with NonceOrchestrator() as orchestrator:
        orchestrator.main_loop()