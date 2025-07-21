from __future__ import annotations
"""
hybrid_generator.py
-------------------
Generador híbrido (meta‑orquestador) que coordina múltiples sub‑generadores
(range, ml, entropy, sequence, adaptive, random) con:

- Multi-armed bandit adaptativo (pesos por tasa de éxito suavizada).
- Circuit breaker por generador con recuperación diferida.
- Rebalanceo periódico de pesos (clamp min/max).
- Agrupación de peticiones: se llama a cada subgenerador sólo una vez por batch.
- Preserva campo 'hash' que puedan aportar subgeneradores, pero para persistencia
  sólo se escriben las 7 columnas estándar.
- Opción de desactivar escritura directa (write_direct=False) y delegar al
  orquestador global. Si write_direct=True usa NonceCSVWriter.
- Hooks de extensión para métricas externas o generadores adicionales.

Formato estándar de registros (para CSV):
  ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height"]
'hash' (si existe) queda en memoria para envío a proxy / submit.

Este módulo NO realiza validación RandomX adicional: depende de cada subgenerador.
"""

import os
import time
import json
import random
import logging
import threading
from typing import Optional, Dict, List, Any

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None       # type: ignore
    _HAS_NUMPY = False

from iazar.generator.range_based_generator import RangeBasedGenerator
from iazar.generator.ml_based_generator import MLBasedGenerator
from iazar.generator.entropy_based_generator import EntropyBasedGenerator
from iazar.generator.sequence_based_generator import SequenceBasedGenerator
from iazar.generator.adaptive_generator import AdaptiveGenerator
from iazar.generator.random_generator import RandomGenerator
from iazar.generator.NonceCSVWriter import NonceCSVWriter

LOGGER_NAME = "generator.hybrid"
logger = logging.getLogger(LOGGER_NAME)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Columnas estándar (sin hash) para persistencia CSV
CSV_FIELDS = ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height"]


class HybridGenerator:
    """Meta‑orquestador de sub‑generadores."""

    DEFAULTS = {
        "rebalance_interval": 300,
        "performance_window": 1000,
        "min_weight": 0.05,
        "max_weight": 0.40,
        "circuit_breaker_threshold": 4,
        "circuit_recovery_time": 600,
        "write_direct": False,
        "csv_path": "C:/zarturxia/src/iazar/data/nonces_exitosos.csv"
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg_root = config or {}
        hybrid_cfg = {**self.DEFAULTS, **cfg_root.get("hybrid_generator", {})}
        self.config = hybrid_cfg

        # Sub‑generadores (instanciación lazy posible; aquí directa)
        self.generators: Dict[str, Any] = self._initialize_generators(cfg_root)

        # Pesos iniciales: usa config.global "generator_weights" si disponible
        self.weights = self._initial_weights(cfg_root)

        # Estadísticas de rendimiento por generador
        self.performance: Dict[str, Dict[str, int]] = {name: {'success': 0, 'total': 0}
                                                       for name in self.generators}
        # Salud (conteo de fallos consecutivos)
        self.generator_health: Dict[str, int] = {name: 0 for name in self.generators}
        self.circuit_last_failure: Dict[str, float] = {name: 0.0 for name in self.generators}

        # Sincronización
        self._lock = threading.RLock()
        self._last_rebalance = time.time()

        # Persistencia directa opcional
        self.write_direct = bool(hybrid_cfg["write_direct"])
        self.writer = NonceCSVWriter(hybrid_cfg["csv_path"]) if self.write_direct else None

        logger.info("[HybridGenerator] Initialized weights=%s write_direct=%s",
                    json.dumps(self.weights), self.write_direct)

    # ------------------------------------------------------------------
    # Inicialización de subgeneradores
    # ------------------------------------------------------------------
    def _initialize_generators(self, full_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Instancia todos los subgeneradores con el mismo bloque de configuración raíz.
        Se asume que cada subgenerador internamente extrae su sección específica.
        """
        gens = {
            "range": RangeBasedGenerator(config=full_config),
            "ml": MLBasedGenerator(config=full_config),
            "entropy": EntropyBasedGenerator(config=full_config),
            "sequence": SequenceBasedGenerator(config=full_config),
            "adaptive": AdaptiveGenerator(config=full_config),
            "random": RandomGenerator(config=full_config)
        }
        return gens

    def register_external_generator(self, name: str, generator_obj: Any, weight: float = 0.05):
        """Permite añadir un generador nuevo en caliente."""
        with self._lock:
            if name in self.generators:
                logger.warning("Generator '%s' already exists. Skipping.", name)
                return
            self.generators[name] = generator_obj
            self.performance[name] = {'success': 0, 'total': 0}
            self.generator_health[name] = 0
            self.circuit_last_failure[name] = 0.0
            self.weights[name] = float(weight)
            logger.info("[HybridGenerator] Registered external generator '%s' weight=%.4f", name, weight)
            self._normalize_weights()

    # ------------------------------------------------------------------
    # Pesos adaptativos
    # ------------------------------------------------------------------
    def _initial_weights(self, full_config: Dict[str, Any]) -> Dict[str, float]:
        # Intenta leer pesos globales si existen
        gw = full_config.get("generator_weights")
        if isinstance(gw, dict):
            usable = {k: float(v) for k, v in gw.items() if k in self.generators}
            if usable:
                total = sum(usable.values()) or 1.0
                return {k: v / total for k, v in usable.items()}

        # Fallback por defecto
        defaults = {
            "range": 0.30,
            "ml": 0.25,
            "entropy": 0.15,
            "sequence": 0.10,
            "adaptive": 0.15,
            "random": 0.05
        }
        # Filtra solo generadores instanciados
        usable = {k: defaults[k] for k in self.generators if k in defaults}
        total = sum(usable.values()) or 1.0
        return {k: v / total for k, v in usable.items()}

    def _normalize_weights(self):
        total = sum(self.weights.values()) or 1.0
        for k in self.weights:
            self.weights[k] /= total

    def _calculate_success_rate(self, name: str) -> float:
        stats = self.performance[name]
        # Laplace smoothing
        return (stats['success'] + 1) / (stats['total'] + 2)

    def _rebalance_weights_if_needed(self):
        now = time.time()
        if now - self._last_rebalance < float(self.config["rebalance_interval"]):
            return
        self._rebalance_weights()
        self._last_rebalance = now

    def _rebalance_weights(self):
        min_w = float(self.config["min_weight"])
        max_w = float(self.config["max_weight"])
        rates = {name: self._calculate_success_rate(name) for name in self.generators}
        total_rate = sum(rates.values()) or 1.0
        new_w = {name: max(min_w, min(max_w, rates[name] / total_rate)) for name in rates}
        # Renormaliza después de clamping
        total = sum(new_w.values()) or 1.0
        for n in new_w:
            new_w[n] /= total
        self.weights = new_w
        # Resetea ventana de rendimiento
        for name in self.performance:
            self.performance[name] = {'success': 0, 'total': 0}
        logger.info("[HybridGenerator] Weights rebalanced -> %s", json.dumps(self.weights))

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------
    def _is_circuit_open(self, name: str) -> bool:
        threshold = int(self.config["circuit_breaker_threshold"])
        if self.generator_health[name] < threshold:
            return False
        recovery = float(self.config["circuit_recovery_time"])
        elapsed = time.time() - self.circuit_last_failure[name]
        if elapsed >= recovery:
            # Reset tras ventana de recuperación
            self.generator_health[name] = 0
            return False
        return True

    def _mark_failure(self, name: str):
        self.generator_health[name] += 1
        threshold = int(self.config["circuit_breaker_threshold"])
        if self.generator_health[name] == threshold:
            self.circuit_last_failure[name] = time.time()
            logger.warning("[HybridGenerator] Circuit breaker OPEN for '%s'", name)

    def _mark_success(self, name: str):
        self.generator_health[name] = 0

    # ------------------------------------------------------------------
    # Selección de generadores para el batch
    # ------------------------------------------------------------------
    def _allocate_batch(self, batch_size: int) -> Dict[str, int]:
        """Devuelve cuántos nonces pediremos a cada subgenerador."""
        with self._lock:
            enabled = [g for g in self.generators if not self._is_circuit_open(g)]
            if not enabled:
                # Todos bloqueados -> forzar habilitación
                enabled = list(self.generators.keys())
                logger.critical("[HybridGenerator] All generators in circuit breaker; forcing usage.")

            if not _HAS_NUMPY:
                # Reparto proporcional simple
                alloc = {g: 0 for g in enabled}
                total_w = sum(self.weights[g] for g in enabled) or 1.0
                rem = batch_size
                for g in enabled:
                    cnt = int(round(batch_size * self.weights[g] / total_w))
                    cnt = min(cnt, rem)
                    alloc[g] = cnt
                    rem -= cnt
                # Ajuste restos
                i = 0
                keys = list(alloc.keys())
                while rem > 0 and keys:
                    alloc[keys[i % len(keys)]] += 1
                    rem -= 1
                    i += 1
                return alloc

            # Numpy-based multinomial
            w = np.array([self.weights[g] for g in enabled], dtype=np.float64)
            w = w / w.sum()
            draws = np.random.multinomial(batch_size, w)
            alloc = {g: int(draws[i]) for i, g in enumerate(enabled)}
            return alloc

    # ------------------------------------------------------------------
    # Métricas / Rendimiento
    # ------------------------------------------------------------------
    def _update_performance(self, name: str, success: bool):
        perf = self.performance[name]
        perf['total'] += 1
        if success:
            perf['success'] += 1
            self._mark_success(name)
        else:
            self._mark_failure(name)

        # Mantener ventana máxima (performance_window)
        window = int(self.config["performance_window"])
        # No almacenamos histórico detallado para reducir memoria; solo contadores.
        self._rebalance_weights_if_needed()

    # ------------------------------------------------------------------
    # Ciclo de generación
    # ------------------------------------------------------------------
    def run_generation(self, block_height: int, block_data: Dict[str, Any], batch_size: int = 500) -> List[Dict[str, Any]]:
        """
        Produce un batch de nonces combinando subgeneradores según pesos y salud.
        Devuelve lista de registros (incluye 'hash' si subgenerador lo añadió).
        """
        t0 = time.perf_counter()
        allocation = self._allocate_batch(batch_size)

        produced: List[Dict[str, Any]] = []
        for name, count in allocation.items():
            if count <= 0:
                continue
            gen = self.generators.get(name)
            if gen is None:
                continue
            try:
                if not hasattr(gen, "run_generation"):
                    raise RuntimeError(f"Subgenerator '{name}' lacks run_generation()")
                sub_batch = gen.run_generation(block_height, block_data, batch_size=count)
                # Actualizar rendimiento con resultados
                for rec in sub_batch:
                    self._update_performance(name, bool(rec.get("is_valid")))
                    rec["source"] = name  # Anotación de procedencia
                produced.extend(sub_batch)
            except Exception as e:
                logger.error("Error in subgenerator '%s': %s", name, e, exc_info=True)
                self._update_performance(name, False)

        # Mezcla final aleatoria (para no sesgar orden por generador)
        random.shuffle(produced)

        # Recorte (si algún generador sobreprodujo)
        if len(produced) > batch_size:
            produced = produced[:batch_size]

        # Opcional: persistir (solo columnas estándar)
        if self.write_direct and produced and self.writer:
            rows_std = []
            for r in produced:
                rows_std.append({k: r.get(k) for k in CSV_FIELDS})
            self.writer.write_many(rows_std)

        elapsed = time.perf_counter() - t0
        valid_count = sum(1 for r in produced if r.get("is_valid"))
        logger.info("[HybridGenerator] block=%s batch=%d valid=%d latency=%.3fs weights=%s",
                    block_height, len(produced), valid_count, elapsed, json.dumps(self.weights))
        # METRIC: hybrid_generator_batch_latency_seconds.observe(elapsed)
        # METRIC: hybrid_generator_valid_ratio.observe(valid_count / (len(produced) or 1))

        return produced

    # ------------------------------------------------------------------
    # Reports / Info
    # ------------------------------------------------------------------
    def get_weight_report(self) -> Dict[str, float]:
        with self._lock:
            return dict(self.weights)

    def get_health_report(self) -> Dict[str, int]:
        with self._lock:
            return dict(self.generator_health)

    def get_performance_snapshot(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {k: v.copy() for k, v in self.performance.items()}

    # ------------------------------------------------------------------
    # Hooks (extensibles)
    # ------------------------------------------------------------------
    def _pre_cycle(self):
        pass

    def _post_cycle(self):
        pass

    # ------------------------------------------------------------------
    # Cierre
    # ------------------------------------------------------------------
    def close(self):
        if self.writer:
            self.writer.close()
        # Cerrar subgeneradores que tengan close()
        for g in self.generators.values():
            close_fn = getattr(g, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        logger.info("[HybridGenerator] Closed.")

# Factory para descubrimiento dinámico
def create_generator(config: Optional[Dict[str, Any]] = None) -> HybridGenerator:
    return HybridGenerator(config=config)
