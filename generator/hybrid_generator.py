"""
Técnica 6: Generador Híbrido Profesional para Monero/RandomX
Combina estratégicamente múltiples técnicas de generación con balance dinámico
"""

import random
import json
import os
import time
import threading

class HybridGenerator:
    """
    Sistema híbrido profesional para minería Monero.
    Combina dinámicamente múltiples técnicas de generación basado en rendimiento histórico.
    """
    
    REBALANCE_INTERVAL = 300  # Segundos entre rebalanceos
    PERFORMANCE_WINDOW = 1000  # Nonces para calcular rendimiento
    
    def __init__(self, config=None):
        self.generator_name = "hybrid"
        self.config = config or {}
        self.generators = self.initialize_generators()
        self.weights = self.initial_weights()
        self.performance = {name: {'success': 0, 'total': 0} for name in self.generators}
        self.lock = threading.Lock()
        self.last_rebalance = time.time()
        self.additional_metadata = {}
        
    def initialize_generators(self) -> dict:
        """Inicializa todos los generadores con configuración profesional"""
        return {
            "range": RangeBasedGenerator(config=self.config),
            "ml": MLBasedGenerator(config=self.config),
            "entropy": EntropyBasedGenerator(config=self.config),
            "sequence": SequenceBasedGenerator(config=self.config),
            "adaptive": AdaptiveGenerator(config=self.config),
            "random": RandomGenerator(config=self.config)
        }
    
    def initial_weights(self) -> dict:
        """Pesos iniciales basados en configuración"""
        return {
            "range": 0.30,
            "ml": 0.25,
            "entropy": 0.15,
            "sequence": 0.10,
            "adaptive": 0.15,
            "random": 0.05
        }
    
    def update_performance(self, generator_name: str, success: bool):
        """Actualiza estadísticas de rendimiento para un generador"""
        with self.lock:
            self.performance[generator_name]['total'] += 1
            if success:
                self.performance[generator_name]['success'] += 1
                
            # Rebalancear periódicamente
            if time.time() - self.last_rebalance > self.REBALANCE_INTERVAL:
                self.rebalance_weights()
                self.last_rebalance = time.time()
    
    def calculate_success_rate(self, generator_name: str) -> float:
        """Calcula tasa de éxito con suavizado de Laplace"""
        stats = self.performance[generator_name]
        if stats['total'] == 0:
            return self.weights[generator_name]  # Mantener peso inicial
        
        # Suavizado de Laplace para evitar divisiones por cero
        return (stats['success'] + 1) / (stats['total'] + 2)
    
    def rebalance_weights(self):
        """Reajusta pesos dinámicamente basado en rendimiento"""
        print("Rebalancing hybrid weights...")
        total_success = sum(self.calculate_success_rate(name) for name in self.generators)
        
        new_weights = {}
        for name in self.generators:
            success_rate = self.calculate_success_rate(name)
            new_weights[name] = max(0.05, min(0.40, success_rate / total_success))
        
        # Normalizar
        total = sum(new_weights.values())
        for name in new_weights:
            new_weights[name] /= total
            
        self.weights = new_weights
        print("New weights:", json.dumps(self.weights, indent=2))
        
        # Resetear estadísticas
        for name in self.performance:
            self.performance[name] = {'success': 0, 'total': 0}
    
    def select_generator(self) -> str:
        """Selecciona un generador basado en pesos actuales"""
        names, weights = zip(*self.weights.items())
        return random.choices(names, weights=weights, k=1)[0]
    
    def generate_nonce(self, block_height: int) -> dict:
        """Genera un nonce usando estrategia híbrida optimizada"""
        # Seleccionar generador
        generator_name = self.select_generator()
        generator = self.generators[generator_name]
        
        # Generar nonce
        nonce_data = generator.generate_nonce(block_height)
        
        # Añadir metadatos híbridos
        nonce_data['hybrid_generator'] = generator_name
        self.additional_metadata = nonce_data  # Guardar metadatos para reporte
        
        # La validación real la hará el proxy, esto es solo para seguimiento
        is_valid = nonce_data.get('is_valid', True)
        
        # Actualizar rendimiento
        self.update_performance(generator_name, is_valid)
        
        return nonce_data
    
    def report_success(self, nonce: int, success: bool):
        """Reporta éxito real después de validación por proxy"""
        # Encuentra qué generador creó este nonce
        if 'hybrid_generator' in self.additional_metadata:
            generator_name = self.additional_metadata['hybrid_generator']
            self.update_performance(generator_name, success)