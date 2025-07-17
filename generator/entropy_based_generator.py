"""
Técnica 3: Generador basado en entropía avanzada para RandomX
Implementa medidas de entropía especializadas y optimizadas para minería Monero
"""

import numpy as np
import random
import time
from sklearn.neighbors import KernelDensity

class EntropyBasedGenerator:
    """
    Generador de nonces con foco en máxima entropía para RandomX.
    Combina múltiples medidas de entropía y adaptación dinámica.
    """
    
    MIN_ENTROPY_THRESHOLD = 5.8  # Mínimo absoluto para entropía de Shannon
    MAX_ATTEMPTS = 1000  # Límite para evitar bucles infinitos
    ADAPTIVE_WINDOW = 1000  # Tamaño de ventana para adaptación
    
    def __init__(self, config=None):
        self.generator_name = "entropy_based"
        self.config = config or {}
        self.adaptive_threshold = self.MIN_ENTROPY_THRESHOLD
        self.recent_entropies = []
        self.kde_model = None
        self.last_kde_update = 0
        self.training_data = []  # Inicializar lista vacía para datos de entrenamiento
        
    def generate_nonce(self, block_height: int) -> dict:
        """Genera un nonce con características de alta entropía"""
        # Actualizar modelo KDE periódicamente
        if time.time() - self.last_kde_update > 3600 or not self.kde_model:
            self.update_kde_model()
        
        best_nonce = None
        best_entropy = 0
        attempts = 0
        
        while attempts < self.MAX_ATTEMPTS:
            nonce = random.getrandbits(64)
            entropy_metrics = self.calculate_entropy_metrics(nonce)
            shannon_entropy = entropy_metrics['entropy']
            
            # Verificar si cumple con el umbral adaptativo
            if shannon_entropy >= self.adaptive_threshold:
                return {
                    "nonce": nonce,
                    "block_height": block_height,
                    "generator": self.generator_name,
                    "is_valid": True,
                    **entropy_metrics
                }
            
            # Trackear el mejor candidato encontrado
            if shannon_entropy > best_entropy:
                best_entropy = shannon_entropy
                best_nonce = nonce
                best_metrics = entropy_metrics
            
            attempts += 1
        
        # Si no se encontró candidato óptimo, devolver el mejor
        print(f"Warning: Using best candidate after {self.MAX_ATTEMPTS} attempts")
        return {
            "nonce": best_nonce,
            "block_height": block_height,
            "generator": self.generator_name,
            "is_valid": True,
            **best_metrics
        }
    
    def calculate_entropy_metrics(self, nonce: int) -> dict:
        """
        Calcula múltiples medidas de entropía optimizadas para RandomX:
        1. Entropía de Shannon (bits)
        2. Entropía de min-entropía
        3. Entropía diferencial
        4. Entropía de Rényi (orden 2)
        """
        bin_repr = bin(nonce)[2:].zfill(64)
        byte_repr = nonce.to_bytes(8, 'big')
        
        # Shannon entropy
        counts = np.bincount(np.frombuffer(byte_repr, dtype=np.uint8))
        probs = counts / counts.sum()
        shannon_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Min-entropy
        min_entropy = -np.log2(probs.max())
        
        # Differential entropy (aproximada)
        diff_entropy = self.calculate_differential_entropy(byte_repr)
        
        # Rényi entropy (order 2)
        renyi_entropy = -np.log2(np.sum(probs**2))
        
        # Calcular métricas adicionales para RandomX
        zero_density = bin_repr.count('0') / 64
        pattern_score = self.randomx_pattern_score(bin_repr)
        
        # Actualizar umbral adaptativo
        self.update_adaptive_threshold(shannon_entropy)
        
        return {
            "entropy": shannon_entropy,
            "min_entropy": min_entropy,
            "diff_entropy": diff_entropy,
            "renyi_entropy": renyi_entropy,
            "zero_density": zero_density,
            "pattern_score": pattern_score
        }
    
    def update_adaptive_threshold(self, entropy: float):
        """Ajusta dinámicamente el umbral de entropía"""
        self.recent_entropies.append(entropy)
        if len(self.recent_entropies) > self.ADAPTIVE_WINDOW:
            self.recent_entropies.pop(0)
            
        if len(self.recent_entropies) > 100:
            mean_entropy = np.mean(self.recent_entropies)
            std_entropy = np.std(self.recent_entropies)
            self.adaptive_threshold = max(
                self.MIN_ENTROPY_THRESHOLD,
                mean_entropy - 0.5 * std_entropy
            )
    
    def calculate_differential_entropy(self, byte_data: bytes) -> float:
        """Calcula entropía diferencial usando KDE"""
        if not self.kde_model:
            return 6.0  # Valor por defecto
            
        data = np.frombuffer(byte_data, dtype=np.uint8).reshape(-1, 1)
        log_prob = self.kde_model.score_samples(data)
        return -np.mean(log_prob) / np.log(2)
    
    def update_kde_model(self):
        """Actualiza el modelo de densidad kernel con datos recientes"""
        if not self.training_data or len(self.training_data) < 100:
            return
            
        # Obtener nonces recientes
        nonces = []
        for row in self.training_data[-1000:]:
            try:
                nonces.append(int(row['nonce']))
            except:
                continue
        
        if len(nonces) < 100:
            return
            
        # Convertir a representación de bytes
        byte_data = []
        for nonce in nonces:
            byte_data.extend(list(nonce.to_bytes(8, 'big')))
        
        # Entrenar modelo KDE
        data = np.array(byte_data).reshape(-1, 1)
        self.kde_model = KernelDensity(bandwidth=0.5, kernel='gaussian')
        self.kde_model.fit(data)
        self.last_kde_update = time.time()
    
    def randomx_pattern_score(self, binary_str: str) -> float:
        """Calcula puntaje de patrón para RandomX"""
        score = 1.0
        
        # Penalizar secuencias de ceros largas
        max_zero_run = self.calculate_max_zero_run(binary_str)
        if max_zero_run > 6:
            penalty = min(0.3, (max_zero_run - 6) * 0.05)
            score -= penalty
        
        # Bonificar alta densidad de transiciones
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
    
    def calculate_max_zero_run(self, binary_str: str) -> int:
        """Calcula la secuencia más larga de ceros consecutivos"""
        max_run = 0
        current_run = 0
        for bit in binary_str:
            if bit == '0':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run