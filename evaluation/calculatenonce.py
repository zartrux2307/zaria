import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
from pathlib import Path
import json
import hashlib
from scipy import stats
from sklearn.ensemble import IsolationForest
import time
from iazar.generator.config_loader import config_loader
# Configuración profesional de rutas
PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = DATA_DIR / "analysis_results"

sys.path.insert(0, str(PROJECT_DIR))
os.chdir(str(PROJECT_DIR))

# Configuración profesional de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / "nonce_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NonceAnalysis")

class ConfigManager:
    """Gestión profesional de configuraciones"""
    _instance = None
    
    def __new__(cls, config_name: str = "global_config"):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.config_path = CONFIG_DIR / f"{config_name}.json"
            cls._instance.config = cls._instance._load_config()
            cls._instance.last_update = time.time()
        return cls._instance
    
    def _load_config(self) -> dict:
        """Carga configuración con verificación de integridad"""
        if not self.config_path.exists():
            logger.critical(f"Archivo de configuración no encontrado: {self.config_path}")
            raise FileNotFoundError(f"Config file missing: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            # Verificar hash de integridad
            config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            if config.get('integrity_hash') != config_hash:
                logger.warning("Configuración modificada! Verificar integridad")
                
            return config.get('nonce_analysis', {})
        except Exception as e:
            logger.exception(f"Error cargando configuración: {str(e)}")
            return {}
    
    def refresh_config(self):
        """Actualiza configuración si ha cambiado"""
        if time.time() - self.last_update > 300:  # Actualizar cada 5 minutos
            self.config = self._load_config()
            self.last_update = time.time()

class DataHandler:
    """Manejador profesional de datos para análisis en tiempo real"""
    REQUIRED_COLUMNS = ['nonce', 'entropy', 'uniqueness', 'zero_density', 
                        'pattern_score', 'is_valid', 'block_height', 'timestamp']
    
    def __init__(self):
        self.training_path = DATA_DIR / "nonce_training_data.csv"
        self.success_path = DATA_DIR / "nonces_exitosos.csv"
        self.accepted_path = DATA_DIR / "nonces_aceptados.csv"
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    def load_training_data(self) -> pd.DataFrame:
        """Carga datos históricos optimizados para análisis"""
        logger.info(f"Cargando datos de entrenamiento desde: {self.training_path}")
        
        if not self.training_path.exists():
            logger.error("Dataset de entrenamiento no encontrado!")
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        
        try:
            # Lectura optimizada con tipos de datos específicos
            dtype_map = {
                'nonce': np.uint32,
                'entropy': np.float32,
                'uniqueness': np.float32,
                'zero_density': np.float32,
                'pattern_score': np.float32,
                'is_valid': bool,
                'block_height': np.uint64
            }
            
            return pd.read_csv(
                self.training_path,
                usecols=self.REQUIRED_COLUMNS,
                dtype=dtype_map,
                parse_dates=['timestamp'],
                infer_datetime_format=True
            )
        except Exception as e:
            logger.exception(f"Error cargando datos: {str(e)}")
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
    
    def stream_successful_nonces(self, batch_size: int = 500) -> pd.DataFrame:
        """Stream de nonces exitosos para análisis continuo"""
        logger.info(f"Leyendo {batch_size} nonces exitosos para análisis")
        
        if not self.success_path.exists():
            logger.warning("Archivo de nonces exitosos no encontrado")
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        
        try:
            # Leer solo el último batch de nonces
            df = pd.read_csv(
                self.success_path,
                usecols=self.REQUIRED_COLUMNS,
                dtype={'nonce': np.uint32}
            ).tail(batch_size)
            
            # Limpiar archivo después de leer
            with open(self.success_path, 'w') as f:
                f.write(','.join(self.REQUIRED_COLUMNS) + '\n')
                
            logger.info(f"{len(df)} nonces exitosos cargados para análisis")
            return df
        except Exception as e:
            logger.exception(f"Error leyendo nonces exitosos: {str(e)}")
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

class RealTimeNonceAnalyzer:
    """Sistema profesional de análisis en tiempo real para operaciones mineras"""
    MAX_NONCE = 2**32  # 4,294,967,295
    
    def __init__(self, config: dict):
        self.config = config
        self.strategy_thresholds = config.get(
            'strategy_thresholds',
            {
                'low_range': (0, 100_000),
                'mid_range': (2.1e9, 2.2e9),
                'high_range': (4_294_000_000, MAX_NONCE)
            }
        )
        self.anomaly_model = self._init_anomaly_detector()
    
    def _init_anomaly_detector(self):
        """Inicializa modelo profesional de detección de anomalías"""
        return IsolationForest(
            n_estimators=150,
            contamination=0.005,
            random_state=42,
            n_jobs=-1
        )
    
    def analyze_distribution(self, df: pd.DataFrame) -> dict:
        """Análisis profesional de distribución de nonces"""
        if df.empty:
            return {}
        
        results = {
            'total_nonces': len(df),
            'valid_rate': df['is_valid'].mean() if 'is_valid' in df.columns else 0,
            'mean_entropy': df['entropy'].mean(),
            'mean_uniqueness': df['uniqueness'].mean()
        }
        
        # Análisis por estrategia
        for strategy, (low, high) in self.strategy_thresholds.items():
            mask = (df['nonce'] >= low) & (df['nonce'] <= high)
            count = mask.sum()
            results[f'{strategy}_count'] = count
            results[f'{strategy}_pct'] = count / len(df) * 100 if len(df) > 0 else 0
            
            if count > 0:
                subset = df[mask]
                results[f'{strategy}_entropy'] = subset['entropy'].mean()
                results[f'{strategy}_valid_rate'] = subset['is_valid'].mean() if 'is_valid' in df.columns else 0
        
        # Métricas avanzadas
        if len(df) > 10:
            results['entropy_std'] = df['entropy'].std()
            results['uniqueness_std'] = df['uniqueness'].std()
            results['nonce_skew'] = stats.skew(df['nonce'])
            results['nonce_kurtosis'] = stats.kurtosis(df['nonce'])
        
        return results
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detección profesional de nonces anómalos"""
        if len(df) < 100:
            return pd.DataFrame()
        
        # Preparar características para detección de anomalías
        features = df[['nonce', 'entropy', 'uniqueness']].copy()
        features['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Entrenar y predecir
        self.anomaly_model.fit(features)
        anomaly_scores = self.anomaly_model.decision_function(features)
        predictions = self.anomaly_model.predict(features)
        
        # Filtrar anomalías
        df['anomaly_score'] = anomaly_scores
        anomalies = df[predictions == -1]
        
        if not anomalies.empty:
            logger.warning(f"Detectadas {len(anomalies)} anomalías en nonces")
            # Guardar detecciones para investigación
            anomaly_path = RESULTS_DIR / "nonce_anomalies.csv"
            anomalies.to_csv(anomaly_path, mode='a', header=not anomaly_path.exists(), index=False)
        
        return anomalies
    
    def generate_recommendations(self, stats: dict) -> dict:
        """Genera recomendaciones estratégicas basadas en análisis"""
        recommendations = {
            'priority_strategy': 'mid_range',
            'entropy_threshold': 7.0,
            'adjustment_factor': 1.0
        }
        
        if not stats:
            return recommendations
        
        # Lógica profesional para optimización de estrategias
        if stats.get('high_range_pct', 0) > 40:
            recommendations['priority_strategy'] = 'mid_range'
            recommendations['adjustment_factor'] = 1.2
        elif stats.get('low_range_pct', 0) > 30:
            recommendations['priority_strategy'] = 'high_range'
            recommendations['adjustment_factor'] = 0.8
        
        # Ajuste basado en calidad de nonces
        if stats.get('mean_entropy', 0) < 6.5:
            recommendations['entropy_threshold'] = 6.8
            recommendations['adjustment_factor'] *= 1.15
        
        return recommendations

def analysis_cycle(config_manager: ConfigManager, data_handler: DataHandler):
    """Ciclo profesional de análisis continuo"""
    analyzer = RealTimeNonceAnalyzer(config_manager.config)
    
    while True:
        try:
            # Actualizar configuración
            config_manager.refresh_config()
            
            # Cargar datos históricos
            historical_data = data_handler.load_training_data()
            
            # Realizar análisis histórico
            if not historical_data.empty:
                historical_stats = analyzer.analyze_distribution(historical_data)
                analyzer.detect_anomalies(historical_data)
            
            # Analizar últimos nonces exitosos
            success_data = data_handler.stream_successful_nonces()
            if not success_data.empty:
                realtime_stats = analyzer.analyze_distribution(success_data)
                analyzer.detect_anomalies(success_data)
                
                # Generar recomendaciones
                recommendations = analyzer.generate_recommendations(realtime_stats)
                
                # Enviar recomendaciones al coordinador
                if 'priority_strategy' in recommendations:
                    logger.info(f"Recomendación: Estrategia prioritaria = {recommendations['priority_strategy']}")
            
            # Intervalo de análisis
            time.sleep(config_manager.config.get('analysis_interval', 60))
            
        except Exception as e:
            logger.error(f"Error en ciclo de análisis: {str(e)}")
            time.sleep(10)

def main():
    """Punto de entrada profesional para el sistema de análisis"""
    parser = argparse.ArgumentParser(description='Sistema Industrial de Análisis de Nonces')
    parser.add_argument('--config', type=str, default='global_config',
                        help='Nombre de configuración (sin extensión .json)')
    args = parser.parse_args()
    
    try:
        logger.info("=" * 60)
        logger.info("INICIANDO SISTEMA PROFESIONAL DE ANÁLISIS DE NONCES")
        logger.info("=" * 60)
        
        # Inicializar componentes profesionales
        config_manager = ConfigManager(args.config)
        data_handler = DataHandler()
        
        # Iniciar ciclo de análisis continuo
        analysis_cycle(config_manager, data_handler)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Análisis detenido por usuario")
        return 0
    except Exception as e:
        logger.critical(f"Fallo crítico: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)