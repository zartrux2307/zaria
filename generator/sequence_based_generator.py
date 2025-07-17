"""
Técnica 4: Generador de secuencias optimizadas para RandomX
Integra configuración global y parámetros específicos para minería Monero
"""

import os
import json
import numpy as np
import hashlib
import random
import time
from threading import Lock
from iazar.generator.nonce_generator import BaseNonceGenerator

class SequenceBasedGenerator(BaseNonceGenerator):
    """
    Generador profesional de secuencias de nonces para minería Monero/RandomX.
    Integra configuración global y parámetros específicos para máxima eficiencia.
    """
    
    SEQUENCE_VERSION = "rx_v3"
    CACHE_SIZE = 10000  # Nonces generados en caché
    RESEQUENCE_INTERVAL = 1000  # Recalcular secuencia cada X nonces
    
    def __init__(self, config=None):  # Mantener parámetro
        super().__init__("sequence_based", config)  # Pasar config
        self.sequence = []
        self.index = 0
        self.lock = Lock()
        self.global_config = self.load_global_config()
        self.sequence_params = self.load_sequence_params()
        self.generate_optimized_sequence()
        
    def load_global_config(self) -> dict:
        """Carga configuración global desde archivo"""
        config_path = os.path.join(
            "C:/zarturxia/src/iazar/config", 
            "global_config.json"
        )
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading global config: {e}")
            # Configuración de emergencia
            return {
                "paths": {
                    "data_dir": "C:/zarturxia/src/iazar/data",
                    "config_dir": "C:/zarturxia/src/iazar/config",
                    "model_dir": "C:/zarturxia/src/iazar/models"
                },
                "ia": {
                    "max_nonce": 2**64 - 1
                }
            }
        
    def load_sequence_params(self) -> dict:
        """Carga parámetros de secuencia específicos"""
        config_dir = self.global_config["paths"]["config_dir"]
        params_path = os.path.join(
            config_dir, 
            f"sequence_params_{self.SEQUENCE_VERSION}.json"
        )
        
        # Parámetros predeterminados optimizados
        default_params = {
            "prime_base": 15485867,
            "xor_mask": 0x1F2F3F4F5F6F7F8F,
            "entropy_min": 0.88,
            "zero_density_max": 0.12,
            "pattern_min": 0.85,
            "max_attempts": 100,
            "optimization_rules": {
                "max_zero_run": 6,
                "periodic_patterns": ["0101010101010101", "0011001100110011", "0000111100001111"],
                "entropy_threshold": 0.9
            }
        }
        
        try:
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"Using default sequence parameters (file not found: {params_path})")
        except Exception as e:
            print(f"Error loading sequence params: {e}")
            
        return default_params
    
    def generate_optimized_sequence(self):
        """Genera secuencia optimizada usando configuración"""
        print(f"Generating optimized sequence ({self.SEQUENCE_VERSION})...")
        sequence = []
        params = self.sequence_params
        prime = params['prime_base']
        xor_mask = params['xor_mask']
        max_nonce = self.global_config["ia"].get("max_nonce", 2**64 - 1)
        
        # Generar semilla basada en datos históricos
        seed = self.generate_sequence_seed()
        
        # Generar secuencia base
        base_nonce = seed
        for _ in range(self.CACHE_SIZE):
            # Algoritmo de generación pseudo-aleatorio mejorado
            base_nonce = (base_nonce * prime) % max_nonce
            candidate = base_nonce ^ xor_mask
            
            # Aplicar transformaciones para mejorar métricas
            candidate = self.apply_randomx_optimizations(candidate)
            
            # Filtrar y añadir si cumple criterios
            if self.is_nonce_optimized(candidate, params):
                sequence.append(candidate)
        
        # Usar mejores candidatos si no se alcanza tamaño completo
        if len(sequence) < self.CACHE_SIZE:
            sequence += self.generate_fallback_sequence(self.CACHE_SIZE - len(sequence))
        
        self.sequence = sequence
        print(f"Generated optimized sequence with {len(self.sequence)} nonces")
    
    def generate_fallback_sequence(self, count: int) -> list:
        """Genera secuencia de respaldo cuando no hay suficientes candidatos"""
        fallback = []
        max_nonce = self.global_config["ia"].get("max_nonce", 2**64 - 1)
        for _ in range(count):
            nonce = random.randint(0, max_nonce)
            fallback.append(nonce)
        return fallback
    
    def generate_sequence_seed(self) -> int:
        """Genera semilla basada en datos históricos y estado actual"""
        # Combinar datos de varias fuentes
        seed_data = b""
        
        # Último nonce aceptado
        if self.training_data:
            try:
                last_nonce = int(self.training_data[-1]['nonce'])
                seed_data += last_nonce.to_bytes(8, 'big')
            except:
                pass
        
        # Estado actual de la red
        seed_data += os.urandom(8)
        
        # Hash para crear semilla
        return int.from_bytes(hashlib.sha3_256(seed_data).digest()[:8], 'big')
    
    def apply_randomx_optimizations(self, nonce: int) -> int:
        """Aplica transformaciones para mejorar características de RandomX"""
        bin_str = bin(nonce)[2:].zfill(64)
        rules = self.sequence_params['optimization_rules']
        max_zero_run = rules['max_zero_run']
        
        # Reducir secuencias largas de ceros
        if '0' * (max_zero_run + 1) in bin_str:
            replacement = '10' * ((max_zero_run + 1) // 2)
            bin_str = bin_str.replace('0' * (max_zero_run + 1), replacement, 1)
            nonce = int(bin_str, 2)
        
        # Romper patrones periódicos
        for pattern in rules['periodic_patterns']:
            if len(pattern) > 0 and bin_str.startswith(pattern):
                mask = (1 << len(pattern)) - 1
                nonce ^= mask
                break
        
        return nonce
    
    def is_nonce_optimized(self, nonce: int, params: dict) -> bool:
        """Verifica si un nonce cumple con los criterios de optimización"""
        metrics = self.calculate_base_metrics(nonce)
        return (
            metrics['entropy'] >= params['entropy_min'] and
            metrics['zero_density'] <= params['zero_density_max'] and
            metrics['pattern_score'] >= params['pattern_min']
        )
    
    def generate_nonce(self, block_height: int) -> dict:
        """Devuelve el siguiente nonce en la secuencia optimizada"""
        with self.lock:
            # Regenerar secuencia periódicamente
            if self.index % self.RESEQUENCE_INTERVAL == 0:
                self.generate_optimized_sequence()
                self.index = 0
                
            # Obtener nonce de la secuencia
            if not self.sequence:
                # Fallback si la secuencia está vacía
                nonce = random.getrandbits(64)
                print("Warning: Sequence empty, using random fallback")
            else:
                nonce = self.sequence[self.index % len(self.sequence)]
                self.index += 1
            
            # Calcular métricas
            metrics = self.calculate_base_metrics(nonce)
            
            return {
                "nonce": nonce,
                "block_height": block_height,
                "generator": self.generator_name,
                "is_valid": True,
                **metrics
            }