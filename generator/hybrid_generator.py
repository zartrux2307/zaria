import random
import json
import time
import threading
import logging
from typing import Optional, Dict, List
import numpy as np

from iazar.generator.range_based_generator import RangeBasedGenerator
from iazar.generator.ml_based_generator import MLBasedGenerator
from iazar.generator.entropy_based_generator import EntropyBasedGenerator
from iazar.generator.sequence_based_generator import SequenceBasedGenerator
from iazar.generator.adaptive_generator import AdaptiveGenerator
from iazar.generator.random_generator import RandomGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s [HybridGenerator] %(message)s"
)

class HybridGenerator:
    """
    Generador Híbrido Profesional Enterprise para Monero/RandomX.
    Orquestación con Multi-Armed Bandit adaptativo, circuit breaker, batch numpy-aware
    y rebalanceo en caliente. Listo para minería intensiva y autoajuste en producción.
    """

    REBALANCE_INTERVAL = 300      # segundos
    PERFORMANCE_WINDOW = 1000
    CIRCUIT_BREAKER_THRESHOLD = 4   # Fallos seguidos antes de aislar generador
    CIRCUIT_RECOVERY_TIME = 600     # s para recuperar generador automáticamente

    def __init__(self, config: Optional[Dict] = None):
        self.generator_name = "hybrid"
        self.config = config or {}
        self.generators = self._initialize_generators()
        self.weights = self._initial_weights()
        self.performance = {name: {'success': 0, 'total': 0} for name in self.generators}
        self.generator_health = {name: 0 for name in self.generators}  # 0=ok, +1 por fallo
        self.circuit_last_failure = {name: 0 for name in self.generators}
        self.lock = threading.RLock()
        self.last_rebalance = time.time()
        logging.info(f"Initialized with weights: {json.dumps(self.weights, indent=2)}")

    def _initialize_generators(self) -> dict:
        """Instancia todos los sub-generadores enterprise con config unificada."""
        return {
            "range": RangeBasedGenerator(config=self.config),
            "ml": MLBasedGenerator(config=self.config),
            "entropy": EntropyBasedGenerator(config=self.config),
            "sequence": SequenceBasedGenerator(config=self.config),
            "adaptive": AdaptiveGenerator(config=self.config),
            "random": RandomGenerator(config=self.config)
        }

    def _initial_weights(self) -> dict:
        # Estos pesos iniciales se pueden ajustar por config externa si lo deseas
        return {
            "range": 0.30,
            "ml": 0.25,
            "entropy": 0.15,
            "sequence": 0.10,
            "adaptive": 0.15,
            "random": 0.05
        }

    def update_performance(self, generator_name: str, success: bool):
        with self.lock:
            self.performance[generator_name]['total'] += 1
            if success:
                self.performance[generator_name]['success'] += 1
                self.generator_health[generator_name] = 0
            else:
                self.generator_health[generator_name] += 1
                if self.generator_health[generator_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self.circuit_last_failure[generator_name] = time.time()
                    logging.warning(f"Circuit breaker: {generator_name} aislado por fallos consecutivos.")

            # Rebalanceo periódico (autoajuste de pesos)
            if time.time() - self.last_rebalance > self.REBALANCE_INTERVAL:
                self.rebalance_weights()
                self.last_rebalance = time.time()

    def calculate_success_rate(self, generator_name: str) -> float:
        stats = self.performance[generator_name]
        if stats['total'] == 0:
            return self.weights[generator_name]
        # Laplace smoothing para evitar division by zero
        return (stats['success'] + 1) / (stats['total'] + 2)

    def rebalance_weights(self):
        rates = {name: self.calculate_success_rate(name) for name in self.generators}
        total_rate = sum(rates.values())
        # Ajusta pesos proporcionalmente y asegura mínimos/máximos
        new_weights = {
            name: max(0.05, min(0.40, rate / total_rate))
            for name, rate in rates.items()
        }
        # Normaliza
        total = sum(new_weights.values())
        for name in new_weights:
            new_weights[name] /= total
        self.weights = new_weights
        # Reset stats para nueva ventana de evaluación
        for name in self.performance:
            self.performance[name] = {'success': 0, 'total': 0}
        logging.info(f"Weights rebalanced: {json.dumps(self.weights, indent=2)}")

    def select_generators_batch(self, batch_size: int) -> List[str]:
        """Muestreo eficiente (numpy) para selección de generadores activos."""
        with self.lock:
            available = [name for name in self.generators if not self.is_circuit_open(name)]
            if not available:
                logging.critical("All circuits open! Forzando random fallback.")
                available = list(self.generators.keys())
            weights = np.array([self.weights[n] for n in available])
            weights /= weights.sum()
            return np.random.choice(available, size=batch_size, p=weights).tolist()

    def is_circuit_open(self, generator_name: str) -> bool:
        """Verifica si el generador está aislado por circuit breaker."""
        fail_count = self.generator_health[generator_name]
        if fail_count < self.CIRCUIT_BREAKER_THRESHOLD:
            return False
        last_fail = self.circuit_last_failure[generator_name]
        # Recuperación automática tras timeout
        if time.time() - last_fail > self.CIRCUIT_RECOVERY_TIME:
            self.generator_health[generator_name] = 0
            return False
        return True

    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        """
        Generación batch robusta, selección multi-generador, integración con monitorización.
        Retorna una lista de dicts de nonces completos (con metadatos).
        """
        selected_generators = self.select_generators_batch(batch_size)
        nonce_data_list = []
        generator_counts = {name: 0 for name in self.generators}
        for gen_name in selected_generators:
            generator_counts[gen_name] += 1

        for gen_name, count in generator_counts.items():
            if count == 0 or self.is_circuit_open(gen_name):
                continue
            subgen = self.generators[gen_name]
            try:
                # Validación estricta de interfaz
                if not hasattr(subgen, "run_generation"):
                    raise NotImplementedError(
                        f"Generator {gen_name} must implement run_generation."
                    )
                nonces = subgen.run_generation(block_height, block_data, batch_size=count)
                for n in nonces:
                    n['hybrid_generator'] = gen_name
                    self.update_performance(gen_name, n.get('is_valid', True))
                nonce_data_list.extend(nonces)
            except Exception as e:
                self.update_performance(gen_name, False)
                logging.error(f"Error in {gen_name}: {e}", exc_info=True)

        random.shuffle(nonce_data_list)
        return nonce_data_list[:batch_size]

    def report_success(self, nonce_data: dict, success: bool):
        generator_name = nonce_data.get('hybrid_generator')
        if generator_name in self.generators:
            self.update_performance(generator_name, success)

    def get_mixture_report(self) -> Dict[str, float]:
        """Devuelve la mezcla actual de pesos para dashboards externos."""
        return self.weights.copy()

    def get_health_report(self) -> Dict[str, int]:
        """Reporte de health/circuit breaker status por generador."""
        return self.generator_health.copy()

    # Block-aware weighting: para perfiles avanzados por tipo de bloque
    def set_block_profile(self, block_type: str, weights: Dict[str, float]):
        """
        Permite definir perfiles de peso por tipo de bloque.
        NOTA: requiere implementación avanzada de perfiles si lo usas.
        """
        pass

# Logs y métricas en inglés para auditoría profesional
