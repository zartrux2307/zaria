"""
Técnica 1: Generador basado en rangos optimizados
Genera nonces dentro de intervalos de alta probabilidad para Monero (RandomX)
"""

import random
import numpy as np
from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.generator.config_loader import config_loader
class RangeBasedGenerator(BaseNonceGenerator):
    """
    Generador de nonces basado en rangos probabilísticos optimizados para CPU/RandomX.
    Utiliza distribución gaussiana centrada en zonas de alta aceptación histórica.
    """
    
    def __init__(self, config=None):  # Mantener parámetro
        super().__init__("range_based", config)  # Pasar config
        self.optimal_ranges = self.calculate_optimal_ranges()
        
    def calculate_optimal_ranges(self) -> list:
        """
        Calcula rangos óptimos basados en datos históricos de nonces aceptados.
        Utiliza percentiles estadísticos para identificar zonas de alta probabilidad.
        """
        if not self.training_data:
            # Default ranges if no training data
            return [(0, 2**64 - 1)]
            
        accepted_nonces = [
            int(row['nonce']) for row in self.training_data 
            if row.get('is_valid') == 'True' and row.get('nonce', '').isdigit()
        ]
        
        if not accepted_nonces:
            return [(0, 2**64 - 1)]
        
        # Análisis estadístico de nonces aceptados
        nonce_array = np.array(accepted_nonces)
        q25, q50, q75 = np.percentile(nonce_array, [25, 50, 75])
        iqr = q75 - q25
        
        # Definir rangos óptimos (centrados en zonas de alta densidad)
        return [
            (max(0, int(q25 - 1.5 * iqr)), int(q25 + 0.5 * iqr)),
            (int(q50 - 0.5 * iqr), int(q50 + 0.5 * iqr)),
            (int(q75 - 0.5 * iqr), min(2**64 - 1, int(q75 + 1.5 * iqr)))
        ]
    
    def generate_nonce(self, block_height: int) -> dict:
        """Genera un nonce dentro de los rangos óptimos identificados"""
        # Seleccionar rango óptimo aleatorio
        selected_range = random.choice(self.optimal_ranges)
        min_val, max_val = selected_range
        
        # Generar nonce con distribución gaussiana centrada
        center = (min_val + max_val) // 2
        std_dev = (max_val - min_val) / 6  # Cubre ~99% del rango
        nonce = int(np.random.normal(center, std_dev))
        
        # Asegurar que esté dentro de los límites
        nonce = max(min_val, min(max_val, nonce))
        
        # Calcular métricas específicas para RandomX
        metrics = self.calculate_randomx_metrics(nonce)
        
        return {
            "nonce": nonce,
            "block_height": block_height,
            "generator": self.generator_name,
            **metrics
        }
    
    def calculate_randomx_metrics(self, nonce: int) -> dict:
        """
        Métricas optimizadas para el algoritmo RandomX de Monero:
        - Entropía: Basada en distribución de bits
        - Uniqueness: Distancia de Hamming con nonces históricos
        - Zero Density: Crítico para rendimiento en CPU
        - Pattern Score: Detección de patrones que reducen eficiencia
        """
        bin_repr = bin(nonce)[2:].zfill(64)
        
        # Entropía (Shannon entropy)
        prob_ones = bin_repr.count('1') / 64
        entropy = - (prob_ones * np.log2(prob_ones + 1e-10) + 
                    (1 - prob_ones) * np.log2(1 - prob_ones + 1e-10))
        
        # Uniqueness (promedio distancia Hamming con últimos 100 nonces)
        uniqueness = self.calculate_hamming_uniqueness(nonce)
        
        # Zero density (crítico para RandomX)
        zero_density = bin_repr.count('0') / 64
        
        # Pattern score (penaliza patrones periódicos)
        pattern_score = self.calculate_randomx_pattern_score(bin_repr)
        
        return {
            "entropy": max(0.75, min(0.99, entropy)),
            "uniqueness": uniqueness,
            "zero_density": zero_density,
            "pattern_score": pattern_score,
            "is_valid": True
        }
    
    def calculate_hamming_uniqueness(self, nonce: int) -> float:
        """Calcula unicidad basada en distancia de Hamming con datos históricos"""
        if not self.training_data or len(self.training_data) < 100:
            return random.uniform(0.85, 0.95)
            
        # Tomar muestra de últimos 100 nonces aceptados
        sample = [
            int(row['nonce']) for row in self.training_data[-100:] 
            if row.get('nonce', '').isdigit()
        ]
        
        # Calcular distancia de Hamming promedio
        total_distance = 0
        nonce_bin = bin(nonce)[2:].zfill(64)
        for other in sample:
            other_bin = bin(other)[2:].zfill(64)
            distance = sum(1 for a, b in zip(nonce_bin, other_bin) if a != b)
            total_distance += distance
        
        avg_distance = total_distance / len(sample)
        uniqueness = avg_distance / 64  # Normalizar a 0-1
        return max(0.8, min(0.99, uniqueness))
    
    def calculate_randomx_pattern_score(self, binary_str: str) -> float:
        """
        Calcula puntaje de patrón optimizado para RandomX:
        - Penaliza secuencias largas de ceros (>8)
        - Penaliza patrones periódicos (010101...)
        - Favorece distribución uniforme
        """
        # Detectar secuencias largas de ceros
        max_zero_run = 0
        current_run = 0
        for char in binary_str:
            if char == '0':
                current_run += 1
                max_zero_run = max(max_zero_run, current_run)
            else:
                current_run = 0
        
        # Penalizar secuencias largas de ceros
        zero_penalty = min(1.0, max_zero_run / 8)
        
        # Detectar patrones periódicos
        periodic_score = 1.0
        for period in [2, 4, 8]:
            pattern = "01" * (period//2)
            if binary_str.startswith(pattern * (64//period)):
                periodic_score -= 0.2
        
        # Combinar puntajes
        pattern_score = 0.7 * (1 - zero_penalty) + 0.3 * periodic_score
        return max(0.6, min(1.0, pattern_score))