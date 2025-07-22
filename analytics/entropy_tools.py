"""
entropy_tools.py - Herramientas profesionales de entropía para análisis IA.
© 2025 Zartrux AI Mining Project
"""

import math
import logging
from collections import Counter
from typing import Any, List, Optional, Union
import os
import sys

import pandas as pd
import json
import numpy as np
from iazar.utils.config_manager import ConfigManager 
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)
# Obtener configuración global
config_manager = ConfigManager()
app_config = config_manager.get_config('global_config')

# Columnas estándar globales basadas en configuración
COLUMNS = app_config.get('ia', {}).get('feature_columns', 
    ["nonce", "entropy", "uniqueness", "zero_density", "pattern_score", "is_valid"])

# Configurar logging basado en la configuración global
log_config = app_config.get('logging', {})
log_level = getattr(logging, log_config.get('level', 'INFO').upper(), logging.INFO)
log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger("EntropyTools")

def leer_nonces_csv(path):
    """Lee un CSV de nonces y garantiza estructura/cabecera estándar."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)

    try:
        df = pd.read_csv(path)
        # Manejar CSV vacío
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)

        # Asegurar columnas requeridas
        missing = [col for col in COLUMNS if col not in df.columns]
        for col in missing:
            df[col] = 0
        return df[COLUMNS].dropna()
    except Exception as e:
        logger.error(f"Error leyendo CSV: {e}")
        return pd.DataFrame(columns=COLUMNS)

def guardar_nonces_csv(df, path):
    """Guarda un DataFrame de nonces con la cabecera y orden estándar."""
    # Crear directorio si es necesario
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Asegurar columnas requeridas
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df[COLUMNS].to_csv(path, index=False)

def leer_nonces_json(path):
    """Lee un JSON de nonces como lista de dicts."""
    if not os.path.exists(path):
        return []

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        # Manejar JSON vacío/inválido
        if not isinstance(data, list):
            return []

        # Completar campos faltantes
        for item in data:
            for col in COLUMNS:
                if col not in item:
                    item[col] = 0
        return data
    except Exception as e:
        logger.error(f"Error leyendo JSON: {e}")
        return []

def guardar_nonces_json(lista, path):
    """Guarda una lista de dicts como JSON de nonces."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(lista, f, indent=2)

# Utilidades para blobs binarios
def hexstr_to_bytes(blob_hex):
    return bytes.fromhex(blob_hex) if isinstance(blob_hex, str) else blob_hex

def bytes_to_hexstr(blob_bytes):
    return blob_bytes.hex() if isinstance(blob_bytes, (bytes, bytearray)) else blob_bytes

def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy for byte data"""
    if not data:
        return 0.0
        
    counts = np.zeros(256, dtype=np.uint32)
    for byte in data:
        counts[byte] += 1
        
    probabilities = counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

class ShannonEntropyCalculator:
    """
    Calculadora de entropía de Shannon para cadenas, listas y bytes.
    Útil para estimar la aleatoriedad o dispersión de datos, como nonces.
    """

    @staticmethod
    def calculate(data: Union[str, bytes, List[Any]]) -> float:
        """
        Calcula la entropía de Shannon de una secuencia.

        Args:
            data (str|bytes|List[Any]): Datos a analizar.

        Returns:
            float: Entropía de Shannon (0=sin aleatoriedad, >4=alta aleatoriedad).
        """
        if not data or len(data) == 0:
            logger.warning("Se recibió una secuencia vacía para calcular entropía.")
            return 0.0

        # Convertir a bytes si es posible para optimización
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif isinstance(data, list) and all(isinstance(x, int) for x in data):
            try:
                data = bytes(data)
            except ValueError:
                pass  # Mantener como lista si no se puede convertir

        # Usar la función optimizada con numpy para bytes
        if isinstance(data, bytes):
            entropy = calculate_entropy(data)
            logger.debug(f"Entropía de Shannon (bytes) calculada: {entropy:.4f}")
            return entropy

        # Para otros tipos de datos, usar el método tradicional
        if isinstance(data, list):
            items = data
        else:
            items = list(data)

        counts = Counter(items)
        total = float(len(items))
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        logger.debug(f"Entropía de Shannon (genérica) calculada: {entropy:.4f}")
        return entropy

    @staticmethod
    def from_file(filepath: str, encoding: Optional[str] = None) -> float:
        """Calcula la entropía de un archivo (modo texto o binario)."""
        try:
            if encoding:
                with open(filepath, 'r', encoding=encoding) as f:
                    data = f.read()
            else:
                with open(filepath, 'rb') as f:
                    data = f.read()
            return ShannonEntropyCalculator.calculate(data)
        except Exception as ex:
            logger.error(f"Error leyendo archivo para entropía: {ex}")
            return 0.0


class EntropyTools:
    """
    Utilidades avanzadas para cálculo y comparación de entropía.
    Se integra fácilmente en análisis IA para filtrado de nonces, hash, streams, etc.
    """

    @staticmethod
    def shannon_entropy(data: Union[str, bytes, List[Any]]) -> float:
        """
        Interfaz directa para calcular entropía de Shannon.
        """
        return ShannonEntropyCalculator.calculate(data)

    @staticmethod
    def compare_entropy(a: Union[str, bytes, List[Any]],
                        b: Union[str, bytes, List[Any]]) -> float:
        """
        Compara la entropía de dos muestras, útil para detectar diferencias
        significativas en la dispersión de nonces entre dos lotes.

        Returns:
            float: Diferencia absoluta de entropía.
        """
        ea = EntropyTools.shannon_entropy(a)
        eb = EntropyTools.shannon_entropy(b)
        diff = abs(ea - eb)
        logger.info(f"Comparación de entropía: A={ea:.3f} B={eb:.3f} Δ={diff:.3f}")
        return diff

    @staticmethod
    def is_random_enough(data: Union[str, bytes, List[Any]], threshold: float = None) -> bool:
        """
        Determina si la entropía de los datos supera el umbral recomendado.
        Usa el umbral de configuración si no se especifica.
        """
        if threshold is None:
            threshold = app_config.get('ia', {}).get('min_entropy', 3.5)
        
        entropy = EntropyTools.shannon_entropy(data)
        logger.info(f"Entropía={entropy:.3f} (umbral={threshold})")
        return entropy >= threshold

    @staticmethod
    def analyze_nonce_quality(nonce: bytes) -> dict:
        """
        Analiza un nonce completo y devuelve métricas de calidad.
        """
        entropy = ShannonEntropyCalculator.calculate(nonce)
        
        # Calcular densidad de ceros
        zero_count = sum(1 for byte in nonce if byte == 0)
        zero_density = zero_count / len(nonce) if nonce else 0
        
        # Calcular unicidad (porcentaje de bytes únicos)
        unique_bytes = len(set(nonce))
        uniqueness = unique_bytes / len(nonce) if nonce else 0
        
        # Calcular puntaje de patrón (detección de secuencias repetitivas)
        pattern_score = 0
        if len(nonce) > 4:
            for i in range(len(nonce) - 3):
                chunk = nonce[i:i+4]
                if chunk.count(chunk[0]) == 4:  # Todos iguales
                    pattern_score += 0.5
                elif chunk[0] == chunk[2] and chunk[1] == chunk[3]:  # Patrón ABAB
                    pattern_score += 0.3
            pattern_score = min(1.0, pattern_score / (len(nonce) - 3))
        
        return {
            'entropy': entropy,
            'zero_density': zero_density,
            'uniqueness': uniqueness,
            'pattern_score': pattern_score,
            'quality_score': (entropy * 0.4) + (uniqueness * 0.3) + 
                            ((1 - zero_density) * 0.2) + ((1 - pattern_score) * 0.1)
        }

# Exports principales para importar en otros módulos:
__all__ = [
    "ShannonEntropyCalculator",
    "EntropyTools",
    "leer_nonces_csv",
    "guardar_nonces_csv",
    "leer_nonces_json",
    "guardar_nonces_json",
    "hexstr_to_bytes",
    "bytes_to_hexstr"
]
