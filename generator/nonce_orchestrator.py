from __future__ import annotations
"""
nonce_orchestrator.py - Sistema profesional de minería Monero
"""

import os
import time
import json
import logging
import threading
import queue
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future

# ---------------- Logging profesional ----------------
logger = logging.getLogger("mining.orchestrator")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

# ---------------- Configuración central ----------------
CONFIG_PATH = "C:/zarturxia/src/iazar/config/global_config.json"

def load_config() -> Dict:
    """Carga la configuración con manejo robusto de errores"""
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error cargando configuración: %s", e)
        return {
            "orchestrator": {
                "target_rate": 500,
                "dedup_window": 20000
            },
            "randomx": {
                "dll_path": "C:/zarturxia/src/libs/randomx.dll",
                "flags": ["LARGE_PAGES", "HARD_AES", "JIT", "FULL_MEM"],
                "validation_mode": "light"
            },
            "generator_weights": {
                "range": 0.4,
                "ml": 0.25,
                "hybrid": 0.2,
                "adaptive": 0.15
            }
        }

# ---------------- Dependencias esenciales ----------------
from iazar.proxy.randomx_validator import RandomXValidator
from iazar.generator.NonceCSVWriter import NonceCSVWriter
from iazar.proxy.pool_job_provider import SharedMemoryJobProvider
from iazar.generator.shm_solution_writer import SharedMemorySolutionWriter

# Import dinámico de generadores
GENERATOR_MAP = {
    "range": "iazar.generator.range_based_generator.RangeBasedGenerator",
    "ml": "iazar.generator.ml_based_generator.MLBasedGenerator",
    "hybrid": "iazar.generator.hybrid_generator.HybridGenerator",
    "adaptive": "iazar.generator.adaptive_generator.AdaptiveGenerator"
}

def import_generator(gen_name: str):
    """Importa dinámicamente un generador"""
    try:
        module_name, class_name = GENERATOR_MAP[gen_name].rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, KeyError) as e:
        logger.error("Error importando generador %s: %s", gen_name, e)
        return None

# ---------------- Control de tasa ----------------
class RateController:
    """Control preciso de tasa de generación"""
    def __init__(self, rate_per_sec: int):
        self.rate = rate_per_sec
        self.min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.01
        self.last_time = time.perf_counter()
        
    def wait_next(self):
        """Espera para mantener la tasa constante"""
        elapsed = time.perf_counter() - self.last_time
        wait_time = max(0, self.min_interval - elapsed)
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_time = time.perf_counter()

# ---------------- Núcleo del Orquestador ----------------
class NonceOrchestrator:
    def __init__(self):
        self.config = load_config()
        self.running = threading.Event()
        self.running.set()
        
        # Configuración central
        orch_cfg = self.config.get("orchestrator", {})
        self.target_rate = int(orch_cfg.get("target_rate", 500))
        self.job_prefix = orch_cfg.get("job_shm_prefix", "5555")
        self.dedup_window = int(orch_cfg.get("deduplicate_recent_window", 20000))
        
        # Validación profesional
        rx_cfg = self.config.get("randomx", {})
        self.validator = RandomXValidator(rx_cfg)
        
        # Comunicación SHM
        self.job_provider = SharedMemoryJobProvider(prefix=self.job_prefix)
        self.solution_writer = SharedMemorySolutionWriter(prefix=self.job_prefix)
        
        # Inicialización de generadores
        self.generators = self._init_generators()
        
        # Gestión de datos
        self.writer = NonceCSVWriter("C:/zarturxia/src/iazar/data/nonces_exitosos.csv")
        
        # Control de estado
        self._recent_nonces = deque(maxlen=self.dedup_window)
        self._recent_set = set()
        self._lock = threading.Lock()
        self.rate_controller = RateController(self.target_rate)
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.accepted_shares = 0
        
        logger.info("Sistema de minería profesional iniciado")
        logger.info("Tasa objetivo: %d nonces/seg | Generadores activos: %s", 
                   self.target_rate, list(self.generators.keys()))

    def _init_generators(self) -> Dict[str, Any]:
        """Inicializa generadores basados en configuración"""
        generators = {}
        weights = self.config.get("generator_weights", {})
        
        for gen_name, weight in weights.items():
            if weight <= 0:
                continue
                
            gen_class = import_generator(gen_name)
            if not gen_class:
                continue
                
            try:
                generator = gen_class(config=self.config, validator=self.validator)
                generators[gen_name] = generator
                logger.info("Generador '%s' activado (peso: %.2f)", gen_name, weight)
            except Exception as e:
                logger.error("Error iniciando generador %s: %s", gen_name, e)
                
        return generators

    def run(self):
        """Ciclo principal de minería profesional"""
        logger.info("Iniciando ciclo de minería...")
        last_job_id = None
        
        try:
            while self.running.is_set():
                start_time = time.perf_counter()
                
                # Obtener trabajo actual
                job = self.job_provider.current_job()
                if not job or not job.get("blob"):
                    logger.warning("Esperando trabajo válido...")
                    time.sleep(1)
                    continue
                    
                # Saltar si es el mismo trabajo
                if job.get("job_id") == last_job_id:
                    time.sleep(0.05)
                    continue
                    
                last_job_id = job.get("job_id")
                logger.info("Nuevo trabajo recibido | Altura: %d", job.get("height", 0))
                
                # Ejecutar generadores en paralelo
                futures = []
                for gen_name, generator in self.generators.items():
                    futures.append(self.pool.submit(
                        generator.generate_batch,
                        job,
                        batch_size=500
                    ))
                
                # Procesar resultados
                for future in futures:
                    try:
                        records = future.result()
                        self._process_records(records, job)
                    except Exception as e:
                        logger.error("Error en generación: %s", e)
                
                # Control de tasa
                self.rate_controller.wait_next()
                
                # Log de rendimiento
                elapsed = time.perf_counter() - start_time
                logger.debug("Ciclo completado en %.4f seg", elapsed)
                
        except Exception as e:
            logger.critical("Error crítico: %s", e, exc_info=True)
        finally:
            self.shutdown()

    def _process_records(self, records: List[dict], job: dict):
        """Procesamiento profesional de nonces generados"""
        if not records:
            return
            
        valid_records = []
        height = job.get("height", 0)
        
        with self._lock:
            for record in records:
                # Validación y normalización
                nonce = self._normalize_nonce(record.get("nonce"))
                if nonce is None or nonce in self._recent_set:
                    continue
                    
                # Validación profesional
                is_valid = False
                if record.get("is_valid", False):
                    is_valid, hash_val = self.validator.validate(
                        nonce,
                        job["blob"],
                        job.get("nonce_offset", 39)
                    )
                
                # Crear registro estándar
                norm_record = {
                    "nonce": nonce,
                    "entropy": float(record.get("entropy", 0.0)),
                    "uniqueness": float(record.get("uniqueness", 1.0)),
                    "zero_density": float(record.get("zero_density", 0.0)),
                    "pattern_score": float(record.get("pattern_score", 0.0)),
                    "is_valid": is_valid,
                    "block_height": height
                }
                
                # Solo procesar válidos
                if is_valid:
                    valid_records.append(norm_record)
                    self._recent_nonces.append(nonce)
                    self._recent_set.add(nonce)
                    self._submit_solution(job, nonce, hash_val)
        
        # Almacenamiento profesional
        if valid_records:
            self.writer.write_many(valid_records)
            logger.info("%d nonces válidos procesados", len(valid_records))

    def _normalize_nonce(self, nonce) -> Optional[int]:
        """Garantiza nonces de 32 bits válidos"""
        try:
            return int(nonce) & 0xFFFFFFFF
        except (TypeError, ValueError):
            return None

    def _submit_solution(self, job: dict, nonce: int, hash_val: bytes):
        """Envía solución al proxy de minería"""
        solution = {
            "job_id": job.get("job_id", ""),
            "nonce": nonce,
            "result": hash_val.hex(),
            "valid": True
        }
        
        if self.solution_writer.submit_solution(solution):
            self.accepted_shares += 1
            logger.debug("Solución enviada: nonce=%d", nonce)
        else:
            logger.warning("Error enviando solución: nonce=%d", nonce)

    def shutdown(self):
        """Cierre profesional del sistema"""
        self.running.clear()
        logger.info("Iniciando apagado seguro...")
        
        # Detener generadores
        for generator in self.generators.values():
            if hasattr(generator, "close"):
                try:
                    generator.close()
                except Exception:
                    pass
        
        # Liberar recursos
        self.pool.shutdown(wait=False)
        self.validator.close()
        
        logger.info("Sistema detenido. Total shares aceptados: %d", self.accepted_shares)

# ---------------- Punto de entrada ----------------
def main():
    parser = argparse.ArgumentParser(description="Sistema profesional de minería Monero")
    parser.add_argument("--verbose", action="store_true", help="Habilitar modo detallado")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Modo detallado activado")
    
    try:
        orchestrator = NonceOrchestrator()
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Interrupción por usuario")
    except Exception as e:
        logger.critical("Error fatal: %s", e, exc_info=True)

if __name__ == "__main__":
    main()