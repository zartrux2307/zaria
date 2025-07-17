"""
Advanced Nonce Quality Filter for Monero/RandomX Mining

Features:
- Entropy-based filtering
- Pattern detection
- Statistical outlier rejection
- CSV integration with validation
- Configurable thresholds
- State tracking for incremental processing

Usage:
    from nonce_quality_filter import NonceQualityFilter

    # Filter nonces from CSV
    NonceQualityFilter.filter_from_csv()

    # Evaluate single nonce
    score = NonceQualityFilter.evaluate_nonce("a5d3e8f1c02b4d67")
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict

# Project configuration
PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"
STATE_FILE = DATA_DIR / "quality_filter_state.json"
RESULTS_DIR = DATA_DIR / "quality_reports"

sys.path.insert(0, str(PROJECT_DIR))
os.chdir(str(PROJECT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / "quality_filter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NonceQualityFilter")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

class EntropyTools:
    """Herramientas para análisis de entropía"""
    @staticmethod
    def analyze_nonce_quality(nonce_bytes: bytes) -> Dict[str, float]:
        """
        Analiza la calidad de un nonce basado en sus propiedades estadísticas
        
        Args:
            nonce_bytes: Nonce en formato de bytes
        
        Returns:
            Diccionario con métricas de calidad
        """
        # Convertir a binario
        binary_str = ''.join(f'{byte:08b}' for byte in nonce_bytes)
        
        # Calcular métricas básicas
        entropy = EntropyTools.calculate_binary_entropy(binary_str)
        uniqueness = EntropyTools.calculate_uniqueness(binary_str)
        zero_density = binary_str.count('0') / len(binary_str)
        pattern_score = EntropyTools.calculate_pattern_score(binary_str)
        
        return {
            'entropy': entropy,
            'uniqueness': uniqueness,
            'zero_density': zero_density,
            'pattern_score': pattern_score
        }
    
    @staticmethod
    def calculate_binary_entropy(binary_str: str) -> float:
        """Calcula entropía de Shannon para una cadena binaria"""
        if len(binary_str) == 0:
            return 0.0
        
        counts = {'0': 0, '1': 0}
        for bit in binary_str:
            counts[bit] += 1
        
        probs = [count / len(binary_str) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    @staticmethod
    def calculate_uniqueness(binary_str: str) -> float:
        """Calcula métrica de unicidad basada en subsecuencias"""
        # Dividir en subsecuencias de 4 bits
        chunks = [binary_str[i:i+4] for i in range(0, len(binary_str), 4)]
        return len(set(chunks)) / len(chunks)
    
    @staticmethod
    def calculate_pattern_score(binary_str: str) -> float:
        """Calcula puntuación anti-patrón"""
        # Penalizar secuencias repetitivas
        max_repeat = 0
        current = binary_str[0]
        count = 1
        
        for bit in binary_str[1:]:
            if bit == current:
                count += 1
            else:
                if count > max_repeat:
                    max_repeat = count
                current = bit
                count = 1
        
        if count > max_repeat:
            max_repeat = count
            
        repeat_penalty = min(max_repeat / 8, 1.0)  # Normalizar
        
        # Penalizar patrones periódicos
        periodic_penalty = 0
        for pattern in ['0101', '1010', '0011', '1100', '0000', '1111']:
            if pattern in binary_str:
                periodic_penalty += 0.15
        
        # Combinar penalizaciones
        total_penalty = min(repeat_penalty + periodic_penalty, 1.0)
        return 1.0 - total_penalty

class NonceQualityFilter:
    """Professional-grade nonce filter for Monero mining operations"""
    
    # Default configuration fallbacks
    DEFAULT_CONFIG = {
        'min_entropy': 3.8,
        'max_pattern_score': 0.25,
        'quality_threshold': 0.75,
        'max_zero_density': 0.35,
        'min_uniqueness': 0.7,
        'check_interval': 60  # segundos
    }
    
    @staticmethod
    def get_config() -> dict:
        """Retrieve quality thresholds from configuration"""
        config_path = CONFIG_DIR / "global_config.json"
        if not config_path.exists():
            return NonceQualityFilter.DEFAULT_CONFIG
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                ia_config = config.get('ia', {})
                return {
                    'min_entropy': ia_config.get('min_entropy', NonceQualityFilter.DEFAULT_CONFIG['min_entropy']),
                    'max_pattern_score': ia_config.get('max_pattern_score', NonceQualityFilter.DEFAULT_CONFIG['max_pattern_score']),
                    'quality_threshold': ia_config.get('quality_threshold', NonceQualityFilter.DEFAULT_CONFIG['quality_threshold']),
                    'max_zero_density': ia_config.get('max_zero_density', NonceQualityFilter.DEFAULT_CONFIG['max_zero_density']),
                    'min_uniqueness': ia_config.get('min_uniqueness', NonceQualityFilter.DEFAULT_CONFIG['min_uniqueness']),
                    'check_interval': ia_config.get('check_interval', NonceQualityFilter.DEFAULT_CONFIG['check_interval'])
                }
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return NonceQualityFilter.DEFAULT_CONFIG

    @staticmethod
    def evaluate_nonce(nonce: Union[str, bytes]) -> float:
        """
        Evaluate nonce quality using statistical characteristics
        
        Args:
            nonce: Hexadecimal string or bytes representing the nonce
        
        Returns:
            Quality score between 0.0 (poor) and 1.0 (excellent)
        """
        # Convert to bytes if needed
        if isinstance(nonce, str):
            try:
                nonce_bytes = bytes.fromhex(nonce)
            except ValueError:
                return 0.0
        else:
            nonce_bytes = nonce
        
        # Get quality metrics
        metrics = EntropyTools.analyze_nonce_quality(nonce_bytes)
        
        # Calculate composite quality score
        entropy_weight = 0.4
        pattern_weight = 0.3
        uniqueness_weight = 0.2
        zero_density_weight = 0.1
        
        entropy_score = min(1.0, metrics['entropy'] / 8.0)
        pattern_score = 1.0 - min(1.0, metrics['pattern_score'])
        uniqueness_score = metrics['uniqueness']
        zero_density_score = 1.0 - min(1.0, metrics['zero_density'] / 0.5)
        
        return (entropy_score * entropy_weight +
                pattern_score * pattern_weight +
                uniqueness_score * uniqueness_weight +
                zero_density_score * zero_density_weight)

    @staticmethod
    def filter_nonces(
        nonces: List[Union[str, bytes]], 
        threshold: Optional[float] = None
    ) -> List[Union[str, bytes]]:
        """
        Filter nonces meeting quality threshold
        
        Args:
            nonces: List of nonces to evaluate
            threshold: Quality threshold (0.0-1.0)
        
        Returns:
            List of nonces meeting quality standards
        """
        config = NonceQualityFilter.get_config()
        threshold = threshold or config['quality_threshold']
        return [n for n in nonces if NonceQualityFilter.evaluate_nonce(n) >= threshold]

    @staticmethod
    def batch_evaluate(
        nonces: List[Union[str, bytes]]
    ) -> List[Tuple[Union[str, bytes], float]]:
        """
        Efficient batch evaluation of nonces
        
        Args:
            nonces: List of nonces to evaluate
        
        Returns:
            List of tuples (nonce, quality_score)
        """
        return [(nonce, NonceQualityFilter.evaluate_nonce(nonce)) for nonce in nonces]

    @staticmethod
    def load_nonces_csv(path: str) -> pd.DataFrame:
        """
        Load nonces from CSV with validation
        
        Args:
            path: Path to CSV file
        
        Returns:
            DataFrame with nonce data
        """
        if not os.path.exists(path):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(path)
            required_columns = {'nonce', 'entropy', 'uniqueness', 
                               'zero_density', 'pattern_score', 
                               'is_valid', 'block_height'}
            
            # Validate columns
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                logger.warning(f"CSV missing required columns: {missing}")
                return pd.DataFrame()
            
            return df
        except Exception as e:
            logger.error(f"Error loading nonces CSV: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def save_nonces_csv(df: pd.DataFrame, path: str) -> None:
        """
        Save nonces to CSV with error handling
        
        Args:
            df: DataFrame to save
            path: Output file path
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
        except Exception as e:
            logger.error(f"Error saving nonces CSV: {str(e)}")

    @staticmethod
    def real_time_filter(nonce: Union[str, bytes]) -> bool:
        """
        Ultra-fast evaluation for real-time filtering
        
        Args:
            nonce: Nonce to evaluate
        
        Returns:
            True if nonce meets quality standards
        """
        config = NonceQualityFilter.get_config()
        score = NonceQualityFilter.evaluate_nonce(nonce)
        return score >= config['quality_threshold']

    @staticmethod
    def load_state() -> dict:
        """Cargar estado del filtro desde archivo"""
        default_state = {
            'last_processed_index': 0,
            'last_run': 0
        }
        
        if not STATE_FILE.exists():
            return default_state
        
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            return default_state

    @staticmethod
    def save_state(state: dict) -> None:
        """Guardar estado del filtro en archivo"""
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")

    @staticmethod
    def filter_incremental() -> int:
        """
        Filtrado incremental de nonces con seguimiento de estado
        
        Returns:
            Número de nonces aceptados en esta ejecución
        """
        config = NonceQualityFilter.get_config()
        state = NonceQualityFilter.load_state()
        
        input_path = DATA_DIR / "nonces_exitosos.csv"
        output_path = DATA_DIR / "nonces_aceptados.csv"
        
        # Cargar datos de entrada
        df = NonceQualityFilter.load_nonces_csv(input_path)
        if df.empty:
            return 0
        
        # Obtener solo nuevos datos
        last_index = state.get('last_processed_index', 0)
        if last_index >= len(df):
            # Reiniciar si el archivo se ha truncado
            last_index = 0
        
        new_df = df.iloc[last_index:]
        if new_df.empty:
            return 0
        
        # Evaluar nuevos nonces
        new_df['quality_score'] = new_df['nonce'].apply(NonceQualityFilter.evaluate_nonce)
        threshold = config['quality_threshold']
        filtered = new_df[new_df['quality_score'] >= threshold].copy()
        
        # Guardar resultados
        if not filtered.empty:
            if output_path.exists():
                # Modo append manteniendo el encabezado solo si el archivo existe
                filtered.to_csv(output_path, mode='a', header=False, index=False)
            else:
                filtered.to_csv(output_path, index=False)
        
        # Actualizar estado
        state['last_processed_index'] = len(df)
        state['last_run'] = time.time()
        NonceQualityFilter.save_state(state)
        
        # Generar reporte periódico
        if time.time() - state.get('last_report', 0) > 3600:  # Cada hora
            NonceQualityFilter.generate_quality_report(filtered)
            state['last_report'] = time.time()
            NonceQualityFilter.save_state(state)
        
        return len(filtered)

    @staticmethod
    def generate_quality_report(df: pd.DataFrame) -> None:
        """Generar reporte profesional de calidad de nonces"""
        if df.empty:
            return
        
        try:
            # Estadísticas de calidad
            report = {
                'timestamp': time.time(),
                'total_nonces': len(df),
                'avg_quality': df['quality_score'].mean(),
                'min_quality': df['quality_score'].min(),
                'max_quality': df['quality_score'].max(),
                'quality_distribution': {
                    'excellent': len(df[df['quality_score'] >= 0.9]),
                    'good': len(df[(df['quality_score'] >= 0.75) & (df['quality_score'] < 0.9)]),
                    'fair': len(df[(df['quality_score'] >= 0.6) & (df['quality_score'] < 0.75)]),
                    'poor': len(df[df['quality_score'] < 0.6])
                }
            }
            
            # Guardar reporte
            report_file = RESULTS_DIR / f"quality_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Quality report generated: {report_file}")
        except Exception as e:
            logger.error(f"Error generating quality report: {str(e)}")

def main():
    """Punto de entrada principal para el servicio de filtrado"""
    logger.info("🚀 Iniciando servicio de filtrado de calidad de nonces")
    
    # Inicializar estado si es necesario
    state = NonceQualityFilter.load_state()
    if 'last_processed_index' not in state:
        state['last_processed_index'] = 0
        NonceQualityFilter.save_state(state)
    
    # Bucle principal de servicio
    while True:
        try:
            config = NonceQualityFilter.get_config()
            interval = config.get('check_interval', 60)
            
            # Ejecutar filtrado incremental
            accepted_count = NonceQualityFilter.filter_incremental()
            
            if accepted_count > 0:
                logger.info(f"✅ Nonces aceptados: {accepted_count}")
            
            # Esperar hasta el próximo ciclo
            time.sleep(interval)
            
        except KeyboardInterrupt:
            logger.info("🛑 Servicio detenido por usuario")
            break
        except Exception as e:
            logger.error(f"Error crítico: {str(e)}")
            time.sleep(60)  # Esperar antes de reintentar

if __name__ == "__main__":
    main()