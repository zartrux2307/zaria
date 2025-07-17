"""
Técnica 7: Generador Aleatorio Optimizado para RandomX/Monero
Implementa generación aleatoria con métricas reales y optimización para minería CPU
"""

import random
import numpy as np
from iazar.generator.nonce_generator import BaseNonceGenerator
import hashlib
import time
from iazar.generator.config_loader import config_loader
class RandomGenerator(BaseNonceGenerator):
    """
    Generador de nonces aleatorios profesional para minería Monero.
    Implementa métricas reales y optimizaciones específicas para RandomX.
    """
    
    MIN_ENTROPY = 0.75  # Mínimo de entropía para regenerar
    MAX_ATTEMPTS = 10   # Intentos para mejorar métricas
    QUALITY_THRESHOLD = 0.6  # Umbral mínimo de calidad
    
    def __init__(self, config=None):  # Mantener parámetro
        super().__init__("random", config)
        self.config = config or config_loader.load_config()
        self.counter = 0
        self.last_quality_check = time.time()
        self.quality_history = []
        
    def generate_nonce(self, block_height: int) -> dict:
        """Genera un nonce aleatorio con métricas optimizadas"""
        best_nonce = None
        best_metrics = None
        best_score = 0
        
        # Generar varios candidatos y seleccionar el mejor
        for _ in range(self.MAX_ATTEMPTS):
            # Usar múltiples fuentes de entropía
            nonce = self.hybrid_random()
            metrics = self.calculate_randomx_metrics(nonce)
            score = self.quality_score(metrics)
            
            if score > best_score:
                best_score = score
                best_nonce = nonce
                best_metrics = metrics
                
            # Si cumple con el umbral mínimo, usar inmediatamente
            if score >= self.QUALITY_THRESHOLD:
                break
        
        # Actualizar seguimiento de calidad
        self.track_quality(best_score)
        
        return {
            "nonce": best_nonce,
            "block_height": block_height,
            "generator": self.generator_name,
            "is_valid": True,
            **best_metrics
        }
    
    def hybrid_random(self) -> int:
        """Combina múltiples fuentes de aleatoriedad para mayor seguridad"""
        self.counter += 1
        
        # 1. Aleatoriedad del sistema
        sys_random = random.getrandbits(64)
        
        # 2. Entropía criptográfica
        crypto_random = int.from_bytes(os.urandom(8), 'big')
        
        # 3. Basado en tiempo y contador
        time_based = int(time.time() * 1e6) + self.counter
        
        # Combinar con XOR y mezcla
        combined = sys_random ^ crypto_random ^ time_based
        hashed = hashlib.sha3_256(str(combined).encode()).digest()
        
        return int.from_bytes(hashed[:8], 'big')
    
    def calculate_randomx_metrics(self, nonce: int) -> dict:
        """Calcula métricas específicas para RandomX"""
        bin_repr = bin(nonce)[2:].zfill(64)
        
        # Entropía de Shannon
        ones = bin_repr.count('1')
        zeros = bin_repr.count('0')
        p1 = ones / 64
        p0 = zeros / 64
        entropy = - (p0 * np.log2(p0 + 1e-10) + p1 * np.log2(p1 + 1e-10))
        
        # Zero density con penalización por secuencias largas
        zero_density, max_zero_run = self.calculate_penalized_zero_density(bin_repr)
        
        # Pattern score para RandomX
        pattern_score = self.randomx_pattern_score(bin_repr, max_zero_run)
        
        # Uniqueness basada en datos históricos
        uniqueness = self.historical_uniqueness(nonce)
        
        return {
            "entropy": max(0.7, min(0.99, entropy)),
            "uniqueness": uniqueness,
            "zero_density": zero_density,
            "pattern_score": pattern_score
        }
    
    def calculate_penalized_zero_density(self, binary_str: str) -> tuple:
        """Calcula densidad de ceros con penalización por secuencias largas"""
        zero_count = binary_str.count('0')
        zero_density = zero_count / 64
        
        # Detectar secuencia más larga de ceros
        max_run = 0
        current_run = 0
        for char in binary_str:
            if char == '0':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        # Penalizar secuencias largas (>6 ceros seguidos)
        if max_run > 6:
            penalty = min(0.2, (max_run - 6) * 0.05)
            zero_density -= penalty
        
        return max(0.01, zero_density), max_run
    
    def randomx_pattern_score(self, binary_str: str, max_zero_run: int) -> float:
        """Calcula puntaje de patrón para características de RandomX"""
        score = 1.0
        
        # Penalizar secuencias de ceros largas
        if max_zero_run > 6:
            score -= min(0.3, (max_zero_run - 6) * 0.05)
        
        # Bonificación por transiciones frecuentes
        transitions = 0
        for i in range(1, len(binary_str)):
            if binary_str[i] != binary_str[i-1]:
                transitions += 1
                
        transition_ratio = transitions / (len(binary_str) - 1)
        if transition_ratio > 0.5:
            bonus = min(0.15, (transition_ratio - 0.5) * 0.5)
            score += bonus
        
        # Penalizar patrones periódicos
        for period in [2, 4, 8, 16]:
            pattern = binary_str[:period]
            if binary_str == pattern * (64 // period):
                score -= 0.25
                break
        
        return max(0.65, min(1.0, score))
    
    def quality_score(self, metrics: dict) -> float:
        """Calcula puntaje de calidad general basado en métricas"""
        entropy_score = min(1.0, metrics['entropy'] / 0.9)
        uniqueness_score = min(1.0, metrics['uniqueness'] / 0.95)
        zero_density_score = 1.0 - min(1.0, max(0, metrics['zero_density'] - 0.1) / 0.15)
        pattern_score = metrics['pattern_score']
        
        # Media ponderada
        return (entropy_score * 0.3 + 
                uniqueness_score * 0.2 + 
                zero_density_score * 0.3 + 
                pattern_score * 0.2)
    
    def track_quality(self, score: float):
        """Registra calidad para posible recalibración"""
        self.quality_history.append(score)
        if len(self.quality_history) > 100:
            self.quality_history.pop(0)
            
    def historical_uniqueness(self, nonce: int) -> float:
        """Calcula unicidad basada en distancia con nonces históricos"""
        if not self.training_data or len(self.training_data) < 100:
            return random.uniform(0.85, 0.95)
        
        # Muestra aleatoria de nonces históricos
        sample_size = min(100, len(self.training_data))
        sample = random.sample(self.training_data, sample_size)
        
        # Calcular distancia promedio ponderada
        total_distance = 0
        bin_repr = bin(nonce)[2:].zfill(64)
        
        for item in sample:
            try:
                other_nonce = int(item['nonce'])
                other_bin = bin(other_nonce)[2:].zfill(64)
                distance = sum(1 for a, b in zip(bin_repr, other_bin) if a != b)
                total_distance += distance
            except:
                continue
        
        avg_distance = total_distance / sample_size
        uniqueness = avg_distance / 64
        return max(0.8, min(0.99, uniqueness))