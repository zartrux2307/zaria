from __future__ import annotations
"""
nonce_orchestrator.py (ZMQ Version)
--------------------
Orquestador principal con comunicación ZMQ
"""

import os
import sys
import time
import json
import math
import signal
import logging
import threading
import queue
import argparse
import zmq
import platform  
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass

# ---------------- Logging base ----------------
logger = logging.getLogger("orchestrator")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------- Config Cache ----------------
class _ConfigCache:
    _lock = threading.Lock()
    _cache: Dict[str, Dict] = {}
    _mtimes: Dict[str, float] = {}

    @classmethod
    def load(cls, path: Path) -> Dict:
        try:
            m = path.stat().st_mtime
        except FileNotFoundError:
            return {}
        p = path.as_posix()
        with cls._lock:
            if p in cls._cache and cls._mtimes.get(p) == m:
                return cls._cache[p]
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                cls._cache[p] = data
                cls._mtimes[p] = m
                return data
            except Exception:
                logger.warning("Failed loading config %s", path, exc_info=True)
                return cls._cache.get(p, {})

# ---------------- Dependencias ----------------
from iazar.proxy.randomx_validator import RandomXValidator
from iazar.generator.NonceCSVWriter import NonceCSVWriter

# Generadores
from iazar.generator.range_based_generator import RangeBasedGenerator
from iazar.generator.entropy_based_generator import EntropyBasedGenerator
from iazar.generator.sequence_based_generator import SequenceBasedGenerator
from iazar.generator.adaptive_generator import AdaptiveGenerator
from iazar.generator.random_generator import RandomGenerator
from iazar.generator.ml_based_generator import MLBasedGenerator
from iazar.generator.hybrid_generator import HybridGenerator
from iazar.generator.nonce_generator import GroupedTopNonceGenerator as GroupedGenerator

# Evaluadores opcionales
try:
    from iazar.evaluation.correlation_analysis import CorrelationAnalyzer
    from iazar.evaluation.distribution_analyzer import DistributionAnalyzer
    from iazar.evaluation.entropy_analysis import EntropyAnalyzer
    from iazar.evaluation.nonce_quality_filter import NonceQualityFilter
    from iazar.evaluation.survival_analyzer import SurvivalAnalyzer
    from iazar.evaluation.calculatenonce import NonceCalculator
except Exception:
    CorrelationAnalyzer = DistributionAnalyzer = EntropyAnalyzer = NonceQualityFilter = SurvivalAnalyzer = NonceCalculator = None
    logger.info("Evaluators not fully available; proceeding without them.")

# ---------------- Utilidades ----------------
def difficulty_to_target(difficulty: int) -> int:
    if not isinstance(difficulty, int) or difficulty <= 0:
        return (1 << 256) - 1
    return ((1 << 256) - 1) // difficulty

# ---------------- Rate Control ----------------
class TokenBucket:
    def __init__(self, rate_per_sec: int, burst: Optional[int] = None):
        self.capacity = burst or rate_per_sec
        self.rate = rate_per_sec
        self.tokens = float(self.capacity)
        self.timestamp = time.perf_counter()
        self._lock = threading.Lock()

    def consume(self, n: int) -> bool:
        with self._lock:
            now = time.perf_counter()
            delta = now - self.timestamp
            self.timestamp = now
            self.tokens = min(self.capacity, self.tokens + delta * self.rate)
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def refill_now(self):
        with self._lock:
            self.tokens = float(self.capacity)
            self.timestamp = time.perf_counter()

    def update_rate(self, new_rate: int):
        with self._lock:
            self.rate = new_rate
            self.capacity = new_rate
            self.tokens = min(self.tokens, self.capacity)

# ---------------- Registry Generadores ----------------
class GeneratorRegistry:
    def __init__(self):
        self._classes: Dict[str, Callable[..., Any]] = {}
    def register(self, key: str, cls):
        self._classes[key] = cls
    def create(self, key: str, config: Dict, validator: RandomXValidator):
        cls = self._classes.get(key)
        if not cls:
            raise KeyError(f"Generator '{key}' not registered.")
        try:
            return cls(config=config, validator=validator)
        except TypeError:
            return cls(config=config)

# ---------------- ZMQ Components ----------------
class ZmqJobSubscriber:
    """Recibe trabajos mineros a través de ZMQ SUB"""
    def __init__(self, address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.current_job = None
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        logger.info("Job Subscriber conectado a %s", address)

    def start(self):
        self.thread.start()

    def _receive_loop(self):
        error_count = 0
        MAX_ERRORS_LOG = 5
        while self.running:
            try:
                msg = self.socket.recv(flags=zmq.NOBLOCK)
                if not msg or not msg.strip():
                    continue
                try:
                    job = json.loads(msg.decode("utf-8"))
                    # FORMATO: debe contener "blob", "job_id", "height", "target", "header_without_nonce", "seed", "nonce_offset", "pool_id"
                    # Asegura presencia de claves para el orquestador y el validador
                    if isinstance(job, dict) and all(k in job for k in ("blob", "job_id", "height", "target", "header_without_nonce")):
                        self.current_job = job
                except Exception as err:
                    error_count += 1
                    if error_count <= MAX_ERRORS_LOG:
                        logger.warning("Job ZMQ corrupto o vacío descartado: %s", err)
                    if error_count == MAX_ERRORS_LOG:
                        logger.warning("Demasiados errores JSON en jobs, suprimiré más logs.")
            except zmq.Again:
                time.sleep(0.01)
            except Exception as e:
                logger.error("Error recibiendo trabajo: %s", e)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.socket.close()
        self.context.term()

class ZmqSolutionPublisher:
    """Envía soluciones mineras a través de ZMQ PUSH"""
    def __init__(self, address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(address)
        logger.info("Solution Publisher conectado a %s", address)

    def try_submit(self, job_id: str, nonce: int, hash_bytes: bytes, is_valid: bool, pool_id: str) -> bool:
        if not is_valid:
            return False
        try:
            solution = {
                "job_id": job_id,
                "nonce": nonce,
                "hash_hex": hash_bytes.hex(),
                "pool_id": pool_id
            }
            self.socket.send_json(solution, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False
        except Exception as e:
            logger.error("Error enviando solución: %s", e)
            return False

    def close(self):
        self.socket.close()
        self.context.term()

# ---------------- Orquestador ----------------
class NonceOrchestrator:
    CSV_FIELDS = ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height"]

    def __init__(self, config: Optional[Dict] = None, disable_large_pages: bool = False):
        base_dir = Path(os.environ.get("IAZAR_BASE", "C:/zarturxia/src/iazar"))
        self.base_dir = base_dir
        self.config_path = base_dir / "config" / "global_config.json"
        self.global_config = _ConfigCache.load(self.config_path)

        orch_cfg = (config or {}).get("orchestrator", {}) or self.global_config.get("orchestrator", {})
        self.target_rate = int(orch_cfg.get("target_rate", 500))
        self.loop_interval = float(orch_cfg.get("loop_interval_sec", 1.0))
        self.max_queue = int(orch_cfg.get("max_queue", 10000))
        self.merge_max_per_generator = int(orch_cfg.get("merge_max_per_generator", 500))
        self.dedup_window = int(orch_cfg.get("deduplicate_recent_window", 20000))
        self.backpressure_policy = orch_cfg.get("backpressure_policy", "drop_oldest")
        self.enable_evaluators = bool(orch_cfg.get("enable_evaluators", True))
        self.generator_pool_size = int(orch_cfg.get("generator_pool", 6))
        self.share_submit_retry = int(orch_cfg.get("share_submit_retry", 3))
        
        # Configuración ZMQ
        zmq_cfg = orch_cfg.get("zmq", {})
        self.zmq_job_addr = zmq_cfg.get("job_addr", "tcp://localhost:5557")
        self.zmq_solution_addr = zmq_cfg.get("solution_addr", "tcp://localhost:55560")

        # Configuración RandomX con flags optimizados
        rx_config = self.global_config.get("randomx", {})
        if disable_large_pages or platform.system() == "Windows":
            # Flags optimizados para Windows
            rx_config["flags"] = ["HARD_AES", "JIT", "FULL_MEM", "SECURE"]
            logger.info("RandomX: flags optimizados para Windows")

        # Validator principal (compartido)
        self.validator = RandomXValidator(rx_config)

        # Componentes ZMQ
        self.job_subscriber = ZmqJobSubscriber(self.zmq_job_addr)
        self.job_subscriber.start()
        self.solution_publisher = ZmqSolutionPublisher(self.zmq_solution_addr)

        # Registry + Generators
        self.registry = GeneratorRegistry()
        self._register_generators()
        weight_cfg = (config or {}).get("generator_weights", {}) or self.global_config.get("generator_weights", {})
        self.generator_weights = weight_cfg
        self.generators = self._instantiate_generators(config or {})

        # Writer CSV central
        self.writer = NonceCSVWriter(
            self.base_dir / "data" / "nonces_exitosos.csv",
            batch_size=500,
            rotate_daily=True
        )

        # Cola interna y pacing
        self.output_queue: "queue.Queue[dict]" = queue.Queue(maxsize=self.max_queue)
        self.rate_bucket = TokenBucket(rate_per_sec=self.target_rate, burst=self.target_rate)

        # Deduplicación
        self._recent_deque: deque[int] = deque(maxlen=self.dedup_window)
        self._recent_set: set[int] = set()
        self._recent_lock = threading.Lock()

        # Evaluadores
        self.evaluators = self._init_evaluators()

        # Pools de ejecución
        self.pool = ThreadPoolExecutor(max_workers=self.generator_pool_size, thread_name_prefix="gen")
        self.analysis_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="eval")

        # Control
        self.running = threading.Event()
        self.running.set()
        self._last_config_check = time.time()
        self._config_reload_interval = 5.0
        self._last_stats_log = time.time()
        self._stats_interval = 15.0
        self._accepted_shares = 0
        self._last_rate_timestamp = time.time()
        self._dispatched_in_window = 0

        self._setup_signals()

        logger.info("NonceOrchestrator (ZMQ) inicializado")
        logger.info("  Target rate: %d/s", self.target_rate)
        logger.info("  Generadores activos: %s", list(self.generators.keys()))
        logger.info("  Dirección ZMQ jobs: %s", self.zmq_job_addr)
        logger.info("  Dirección ZMQ solutions: %s", self.zmq_solution_addr)

    # ----- Registro de Generadores -----
    def _register_generators(self):
        self.registry.register("range", RangeBasedGenerator)
        self.registry.register("entropy", EntropyBasedGenerator)
        self.registry.register("sequence", SequenceBasedGenerator)
        self.registry.register("adaptive", AdaptiveGenerator)
        self.registry.register("random", RandomGenerator)
        self.registry.register("ml", MLBasedGenerator)
        self.registry.register("hybrid", HybridGenerator)
        self.registry.register("grouped", GroupedGenerator)

    def _instantiate_generators(self, config: Dict):
        gens = {}
        for key in self.registry._classes.keys():
            try:
                if self.generator_weights.get(key, 0) <= 0:
                    continue
                inst = self.registry.create(key, config, self.validator)
                gens[key] = inst
                logger.info("Generator '%s' activado (%s)", key, type(inst).__name__)
            except Exception:
                logger.exception("Error inicializando generador '%s'", key)
        if not gens:
            logger.warning("No se inicializaron generadores (¿todos los pesos son cero?)")
        return gens

    # ----- Evaluadores -----
    def _init_evaluators(self):
        if not self.enable_evaluators:
            return {}
        ev = {}
        try:
            if NonceQualityFilter:
                ev["quality"] = NonceQualityFilter(self.global_config.get("quality_filter", {}))
            if CorrelationAnalyzer:
                ev["correlation"] = CorrelationAnalyzer()
            if DistributionAnalyzer:
                ev["distribution"] = DistributionAnalyzer()
            if EntropyAnalyzer:
                ev["entropy"] = EntropyAnalyzer()
            if SurvivalAnalyzer:
                ev["survival"] = SurvivalAnalyzer()
            if NonceCalculator:
                ev["calculator"] = NonceCalculator()
        except Exception:
            logger.exception("Error inicializando evaluadores")
        return ev

    # ----- Main Loop -----
    def start(self):
        logger.info(">>> Ciclo principal del orquestador INICIADO")
        iteration = 0
        try:
            while self.running.is_set():
                iteration += 1
                loop_start = time.perf_counter()

                block_meta = self._current_job_blockdata()
                
                # Saltar ciclo si no hay trabajo válido
                if block_meta["height"] == 0:
                    logger.warning("No hay trabajo válido. Saltando ciclo de generación.")
                    time.sleep(1.0)
                    self._maybe_reload_config()
                    continue

                if not self.generators:
                    logger.warning("No hay generadores activos. En espera.")
                    time.sleep(1.0)
                    self._maybe_reload_config()
                    continue

                futures: List[Future] = []
                for key, gen in self.generators.items():
                    futures.append(self.pool.submit(
                        self._run_generator_safe,
                        key, gen,
                        block_meta["height"], block_meta
                    ))

                all_records: List[dict] = []
                for fut in futures:
                    try:
                        res = fut.result()
                        if res:
                            if len(res) > self.merge_max_per_generator:
                                res = res[:self.merge_max_per_generator]
                            all_records.extend(res)
                    except Exception:
                        logger.exception("Error en futuro del generador")

                if all_records:
                    filtered = self._post_process_and_enqueue(all_records, block_meta)
                    if self.enable_evaluators and filtered:
                        self._schedule_evaluators(filtered, block_meta)

                self._drain_to_proxy(block_meta)

                self._maybe_reload_config()
                self._maybe_log_stats()

                elapsed = time.perf_counter() - loop_start
                remain = self.loop_interval - elapsed
                if remain > 0:
                    time.sleep(remain)
        finally:
            logger.info(">>> Ciclo del orquestador FINALIZADO")
            self.shutdown()

    # ----- Obtención job → block_data -----
    def _current_job_blockdata(self) -> Dict[str, Any]:
        job = self.job_subscriber.current_job
        if not job:
            return {
                "height": 0,
                "seed": "00" * 32,
                "blob": b"\x00" * 84,
                "target": (1 << 256) - 1,
                "nonce_offset": 39,
                "job_id": "none",
                "pool_id": "default"
            }
        
        try:
            blob = bytes.fromhex(job["blob"]) if isinstance(job["blob"], str) else job["blob"]
            seed = bytes.fromhex(job["seed"]) if isinstance(job["seed"], str) else job["seed"]
            target = job["target"]
            if isinstance(target, str):
                target = bytes.fromhex(target)
            if isinstance(target, (bytes, bytearray)):
                target_val = int.from_bytes(target, "little")
            else:
                target_val = int(target)

            header_wn = job.get("header_without_nonce", "")
            if isinstance(header_wn, str):
                header_without_nonce = bytes.fromhex(header_wn)
            else:
                header_without_nonce = header_wn

            return {
                "height": int(job.get("height", 0)),
                "seed": seed.hex(),
                "blob": blob,
                "target": target_val,
                "nonce_offset": int(job.get("nonce_offset", 39)),
                "header_without_nonce": header_without_nonce,
                "job_id": job.get("job_id", "none"),
                "pool_id": job.get("pool_id", "default")
            }
        except Exception as e:
            logger.error("Error procesando trabajo: %s", e)
            return {
                "height": 0,
                "seed": "00" * 32,
                "blob": b"\x00" * 84,
                "target": (1 << 256) - 1,
                "nonce_offset": 39,
                "job_id": "none",
                "pool_id": "default"
            }

    # ----- Ejecutar Generador -----
    def _run_generator_safe(self, key: str, gen, height: int, block_data: dict):
        try:
            return gen.run_generation(height, block_data, batch_size=500)
        except Exception as e:
            logger.error("Error en generador %s: %s", key, e)
            return []

    # ----- Post-proceso + cola -----
    def _post_process_and_enqueue(self, records: List[dict], block_meta: dict) -> List[dict]:
        cleaned: List[dict] = []
        csv_rows: List[dict] = []

        with self._recent_lock:
            for r in records:
                try:
                    nonce = int(r["nonce"]) & 0xFFFFFFFF
                except Exception:
                    continue
                if nonce in self._recent_set:
                    continue
                self._recent_deque.append(nonce)
                self._recent_set.add(nonce)
                cleaned.append(r)
            # Re-sincronizar set si deque recortó
            if len(self._recent_set) > len(self._recent_deque):
                self._recent_set = set(self._recent_deque)

        for r in cleaned:
            norm = {
                "nonce": int(r.get("nonce", 0)) & 0xFFFFFFFF,
                "entropy": float(r.get("entropy", 0.0)),
                "uniqueness": float(r.get("uniqueness", 1.0)),
                "zero_density": float(r.get("zero_density", 0.0)),
                "pattern_score": float(r.get("pattern_score", 0.0)),
                "is_valid": bool(r.get("is_valid", False)),
                "block_height": int(r.get("block_height", 0))
            }

            # Validar altura del bloque
            if norm["block_height"] != block_meta["height"]:
                logger.warning("Nonce con altura inconsistente: %d vs %d. Invalidando.", 
                              norm["block_height"], block_meta["height"])
                norm["is_valid"] = False

            # Validación secundaria si marca válido sin hash
            need_secondary = norm["is_valid"] and not r.get("hash")
            if need_secondary:
                ok, h = self.validator.validate(norm["nonce"], block_meta, return_hash=True)
                if ok and isinstance(h, (bytes, bytearray)) and len(h) == 32:
                    r["hash"] = h.hex()
                else:
                    norm["is_valid"] = False  # invalidar share dudoso

            csv_rows.append(norm)
            self._enqueue_record({**norm, "hash": r.get("hash"), "pool_id": block_meta.get("pool_id")})

        if csv_rows:
            self.writer.write_many(csv_rows)
        return cleaned

    def _enqueue_record(self, r: dict):
        if self._offer_queue(r):
            return
        # backpressure
        if self.backpressure_policy == "drop_newest":
            logger.debug("Backpressure: drop_newest nonce=%s", r.get("nonce"))
            return
        elif self.backpressure_policy == "drop_oldest":
            try:
                _ = self.output_queue.get_nowait()
            except queue.Empty:
                return
            self._offer_queue(r)
        elif self.backpressure_policy == "block":
            self.output_queue.put(r)
        else:
            return

    def _offer_queue(self, item: dict) -> bool:
        try:
            self.output_queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    # ----- Drain hacia Proxy / Submit Shares -----
    def _drain_to_proxy(self, block_meta: dict):
        dispatched = 0
        while not self.output_queue.empty() and self.rate_bucket.consume(1):
            try:
                item = self.output_queue.get_nowait()
            except queue.Empty:
                break
            dispatched += 1
            self._dispatched_in_window += 1

            if item.get("is_valid"):
                hash_hex = item.get("hash")
                if hash_hex and isinstance(hash_hex, str) and len(hash_hex) == 64:
                    hash_bytes = bytes.fromhex(hash_hex)
                else:
                    ok, h = self.validator.validate(item["nonce"], block_meta, return_hash=True)
                    if not ok:
                        continue
                    hash_bytes = h

                pool_id = item.get("pool_id", block_meta.get("pool_id", "default"))
                submitted = False
                for _ in range(self.share_submit_retry):
                    if self.solution_publisher.try_submit(
                        block_meta.get("job_id", ""), 
                        item["nonce"], 
                        hash_bytes, 
                        True,
                        pool_id
                    ):
                        submitted = True
                        self._accepted_shares += 1
                        break
                    time.sleep(0.001)
                if not submitted:
                    logger.debug("Buffer de envío ocupado; share descartado nonce=%s", item["nonce"])

        if dispatched:
            logger.debug("Nonces enviados: %d (shares válidos: %d)", dispatched, self._accepted_shares)

        # Ajuste dinámico del pacing
        now = time.time()
        if now - self._last_rate_timestamp >= 1.0:
            actual_rate = self._dispatched_in_window / (now - self._last_rate_timestamp)
            if actual_rate < self.target_rate * 0.85:
                self.rate_bucket.refill_now()
            self._dispatched_in_window = 0
            self._last_rate_timestamp = now

    # ----- Evaluadores -----
    def _schedule_evaluators(self, records: List[dict], block_meta: dict):
        for name, ev in self.evaluators.items():
            self.analysis_pool.submit(self._run_evaluator, name, ev, records, block_meta)

    def _run_evaluator(self, name: str, evaluator, records: List[dict], block_meta: dict):
        try:
            evaluator.evaluate(records, block_meta)
        except Exception:
            logger.debug("Evaluador %s falló", name, exc_info=True)

    # ----- Hot Reload -----
    def _maybe_reload_config(self):
        now = time.time()
        if now - self._last_config_check < self._config_reload_interval:
            return
        self._last_config_check = now
        new_cfg = _ConfigCache.load(self.config_path)
        if not new_cfg:
            return

        # Pesos generadores
        new_weights = new_cfg.get("generator_weights", {})
        if new_weights and new_weights != self.generator_weights:
            logger.info("Recargando pesos de generadores: %s", new_weights)
            self.generator_weights = new_weights
            # Activar nuevos
            for k, w in new_weights.items():
                if w > 0 and k not in self.generators:
                    try:
                        inst = self.registry.create(k, {}, self.validator)
                        self.generators[k] = inst
                        logger.info("Generador '%s' activado tras recarga", k)
                    except Exception:
                        logger.exception("No se pudo activar generador %s", k)
            # Desactivar peso 0
            to_remove = [k for k, w in new_weights.items() if w == 0 and k in self.generators]
            for k in to_remove:
                try:
                    self.generators[k].close()
                except Exception:
                    pass
                del self.generators[k]
                logger.info("Generador '%s' desactivado", k)

        orch = new_cfg.get("orchestrator", {})
        updated_rate = orch.get("target_rate")
        if updated_rate and int(updated_rate) != self.target_rate:
            self.target_rate = int(updated_rate)
            self.rate_bucket.update_rate(self.target_rate)
            logger.info("Tasa objetivo actualizada: %d", self.target_rate)

        li = orch.get("loop_interval_sec")
        if li and float(li) != self.loop_interval:
            self.loop_interval = float(li)
            logger.info("Intervalo de ciclo actualizado: %.3f", self.loop_interval)

        mq = orch.get("max_queue")
        if mq and int(mq) != self.max_queue:
            self.max_queue = int(mq)
            logger.info("Cola máxima actualizada: %d", self.max_queue)

    # ----- Stats -----
    def _maybe_log_stats(self):
        now = time.time()
        if now - self._last_stats_log >= self._stats_interval:
            in_queue = self.output_queue.qsize()
            logger.info("[ESTADÍSTICAS] cola=%d shares_aceptados=%d generadores_activos=%d tasa_objetivo=%d",
                        in_queue, self._accepted_shares, len(self.generators), self.target_rate)
            self._last_stats_log = now

    # ----- Signals & Shutdown -----
    def _setup_signals(self):
        if os.name == "nt":
            return
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            logger.warning("Manejadores de señales no establecidos.", exc_info=True)

    def _signal_handler(self, signum, frame):
        logger.info("Señal %s recibida -> apagando", signum)
        self.running.clear()

    def shutdown(self):
        self.running.clear()
        logger.info("Apagando orquestador...")
        try:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.analysis_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            NonceCSVWriter.close_all()
        except Exception:
            pass
        for g in self.generators.values():
            close_fn = getattr(g, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        try:
            self.validator.close()
        except Exception:
            pass
        
        # Cerrar componentes ZMQ
        self.job_subscriber.stop()
        self.solution_publisher.close()
        
        logger.info("Orquestador detenido.")

# ---------------- CLI ----------------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="Habilitar logging DEBUG")
    ap.add_argument("--zmq-job-addr", default="tcp://localhost:5557", help="Dirección ZMQ para trabajos")
    ap.add_argument("--zmq-solution-addr", default="tcp://localhost:55560", help="Dirección ZMQ para soluciones")
    ap.add_argument("--disable-large-pages", action="store_true", help="Deshabilitar Large Pages en Windows")
    return ap.parse_args()

def main():
    args = _parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for lg in ["orchestrator", "generator.range", "generator.adaptive",
                   "generator.random", "generator.ml", "generator.sequence",
                   "generator.hybrid", "generator.entropy"]:
            logging.getLogger(lg).setLevel(logging.DEBUG)
    
    # Pasar argumentos al orquestador
    config = {
        "orchestrator": {
            "zmq": {
                "job_addr": args.zmq_job_addr,
                "solution_addr": args.zmq_solution_addr
            }
        }
    }
    
    # Crear orquestador con configuración optimizada
    orch = NonceOrchestrator(
        config, 
        disable_large_pages=args.disable_large_pages or platform.system() == "Windows"
    )
    
    try:
        orch.start()
    except KeyboardInterrupt:
        pass
    finally:
        orch.shutdown()

if __name__ == "__main__":
    main()
