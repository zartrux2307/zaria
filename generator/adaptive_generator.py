"""
Técnica 5: Generador Adaptativo Optimizado para Monero/RandomX
Implementa aprendizaje por refuerzo con decaimiento exponencial y adaptación dinámica
"""

import random
import numpy as np
import time
import os
import json
from collections import deque
from threading import Lock

class AdaptiveGenerator:
    """
    Generador adaptativo profesional para minería Monero.
    Combina aprendizaje por refuerzo con optimizaciones específicas para RandomX.
    """
    
    DEFAULT_BINS = 50
    MAX_BINS = 200
    MIN_BINS = 20
    DECAY_RATE = 0.995
    ADAPTATION_INTERVAL = 500  # Reajustar cada 500 nonces
    PENALTY_FACTOR = 0.7
    REWARD_FACTOR = 1.3
    
    def __init__(self, config=None):
        self.generator_name = "adaptive"
        self.config = config or {}
        self.bins = self.DEFAULT_BINS
        self.bin_weights = None
        self.bin_success = None
        self.bin_attempts = None
        self.last_adaptation = 0
        self.lock = Lock()
        self.training_data = []
        self.initialize_state()
        
    def initialize_state(self):
        """Inicializa el estado de aprendizaje"""
        with self.lock:
            self.bin_weights = np.ones(self.bins)
            self.bin_success = np.zeros(self.bins)
            self.bin_attempts = np.zeros(self.bins)
            self.min_nonce = 0
            
            # Usar valor predeterminado si no hay configuración
            self.max_nonce = self.config.get('max_nonce', 2**64 - 1)
            
            # Cargar datos históricos para inicialización
            self.load_historical_data()
    
    def load_historical_data(self):
        """Carga datos históricos para inicializar pesos"""
        if not self.training_data:
            return
            
        for row in self.training_data:
            try:
                nonce = int(row['nonce'])
                is_valid = row.get('is_valid', 'False').lower() in ('true', '1', 'yes')
                bin_idx = self.nonce_to_bin(nonce)
                
                self.bin_attempts[bin_idx] += 1
                if is_valid:
                    self.bin_success[bin_idx] += 1
                    
            except (ValueError, KeyError):
                continue
        
        # Calcular pesos iniciales basados en datos históricos
        for i in range(self.bins):
            if self.bin_attempts[i] > 0:
                success_rate = self.bin_success[i] / self.bin_attempts[i]
                self.bin_weights[i] = max(0.1, success_rate * self.REWARD_FACTOR)
    
    def nonce_to_bin(self, nonce: int) -> int:
        """Mapea nonce a índice de bin"""
        bin_size = (self.max_nonce - self.min_nonce + 1) / self.bins
        idx = int((nonce - self.min_nonce) / bin_size)
        return max(0, min(self.bins - 1, idx))
    
    def bin_to_range(self, bin_idx: int) -> tuple:
        """Convierte índice de bin a rango de nonces"""
        bin_size = (self.max_nonce - self.min_nonce + 1) / self.bins
        start = int(self.min_nonce + bin_idx * bin_size)
        end = int(min(self.max_nonce, start + bin_size - 1))
        return (start, end)
    
    def update_bin_weights(self, bin_idx: int, success: bool):
        """Actualiza pesos con decaimiento exponencial y aprendizaje"""
        with self.lock:
            # Aplicar decaimiento a todos los bins
            self.bin_weights *= self.DECAY_RATE
            
            # Actualizar contadores
            self.bin_attempts[bin_idx] += 1
            if success:
                self.bin_success[bin_idx] += 1
                
            # Calcular nuevo peso
            if self.bin_attempts[bin_idx] > 0:
                success_rate = self.bin_success[bin_idx] / self.bin_attempts[bin_idx]
                adjustment = self.REWARD_FACTOR if success else self.PENALTY_FACTOR
                new_weight = success_rate * adjustment
                
                # Actualizar con momentum
                self.bin_weights[bin_idx] = 0.7 * self.bin_weights[bin_idx] + 0.3 * new_weight
    
    def adapt_bin_count(self):
        """Ajusta dinámicamente el número de bins basado en rendimiento"""
        if np.sum(self.bin_attempts) < 1000:
            return  # No hay suficientes datos
        
        avg_success = np.mean(self.bin_success / (self.bin_attempts + 1e-10))
        
        if avg_success < 0.3 and self.bins > self.MIN_BINS:
            # Reducir bins si el rendimiento es bajo
            new_bins = max(self.MIN_BINS, int(self.bins * 0.8))
            print(f"Decreasing bins from {self.bins} to {new_bins} (low success)")
            self.bins = new_bins
            self.initialize_state()
        elif avg_success > 0.7 and self.bins < self.MAX_BINS:
            # Aumentar bins si el rendimiento es alto
            new_bins = min(self.MAX_BINS, int(self.bins * 1.2))
            print(f"Increasing bins from {self.bins} to {new_bins} (high success)")
            self.bins = new_bins
            self.initialize_state()
    
    def select_bin(self) -> int:
        """Selecciona un bin basado en pesos con muestreo probabilístico"""
        with self.lock:
            # Normalizar pesos
            total_weight = np.sum(self.bin_weights)
            if total_weight <= 0:
                return random.randint(0, self.bins - 1)
                
            probabilities = self.bin_weights / total_weight
            
            # Muestreo aleatorio ponderado
            return np.random.choice(self.bins, p=probabilities)
    
    def generate_nonce(self, block_height: int) -> dict:
        """Genera un nonce usando estrategia adaptativa"""
        # Adaptar cada cierto intervalo
        if time.time() - self.last_adaptation > 60:
            self.adapt_bin_count()
            self.last_adaptation = time.time()
        
        # Seleccionar bin
        bin_idx = self.select_bin()
        start, end = self.bin_to_range(bin_idx)
        
        # Generar nonce con distribución gaussiana centrada
        center = (start + end) // 2
        std_dev = (end - start) / 6
        nonce = int(np.random.normal(center, std_dev))
        nonce = max(start, min(end, nonce))
        
        # Calcular métricas específicas para RandomX
        metrics = self.calculate_randomx_metrics(nonce)
        
        # Nota: La validación real la hará el proxy, esto es solo prevalidación
        is_valid = metrics['pattern_score'] > 0.7 and metrics['entropy'] > 0.8
        
        # Actualizar pesos (el resultado real se actualizará más tarde)
        self.update_bin_weights(bin_idx, is_valid)
        
        return {
            "nonce": nonce,
            "block_height": block_height,
            "generator": self.generator_name,
            "is_valid": is_valid,
            **metrics
        }
    
    def calculate_randomx_metrics(self, nonce: int) -> dict:
        """Métricas optimizadas para RandomX"""
        bin_repr = bin(nonce)[2:].zfill(64)
        
        # Entropía (Shannon)
        ones = bin_repr.count('1')
        zeros = bin_repr.count('0')
        p1 = ones / 64
        p0 = zeros / 64
        entropy = - (p0 * np.log2(p0 + 1e-10) + p1 * np.log2(p1 + 1e-10))
        
        # Zero density con penalización
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
    
    def report_success(self, nonce: int, success: bool):
        """Reporta éxito real después de validación por el proxy"""
        bin_idx = self.nonce_to_bin(nonce)
        self.update_bin_weights(bin_idx, success)