import pandas as pd
import numpy as np
import os
import sys
import time
import threading
import json
import logging
import argparse
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
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
        logging.FileHandler(DATA_DIR / "correlation_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CorrelationAnalysis")

class ConfigManager:
    """Gestión profesional de configuraciones con validación de integridad"""
    _instance = None
    
    def __new__(cls, config_name: str = "global_config"):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.config_path = CONFIG_DIR / f"{config_name}.json"
            cls._instance.config = cls._instance._load_config()
            cls._instance.last_update = time.time()
            cls._instance._validate_config()
        return cls._instance
    
    def _load_config(self) -> dict:
        """Carga configuración con verificación de integridad"""
        if not self.config_path.exists():
            logger.critical(f"Archivo de configuración no encontrado: {self.config_path}")
            raise FileNotFoundError(f"Config file missing: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.exception(f"Error cargando configuración: {str(e)}")
            return {}
    
    def _validate_config(self):
        """Valida la estructura de la configuración"""
        required_sections = ['ia', 'sequence_params', 'generator_weights', 'performance_settings']
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Sección faltante en configuración: {section}")
                self.config[section] = {}
    
    def get_ia_config(self) -> dict:
        """Devuelve la configuración de IA con valores predeterminados"""
        ia_config = self.config.get('ia', {})
        return {
            'min_entropy': ia_config.get('min_entropy', 0.85),
            'min_uniqueness': ia_config.get('min_uniqueness', 0.8),
            'max_zero_density': ia_config.get('max_zero_density', 0.15),
            'min_pattern_score': ia_config.get('min_pattern_score', 0.8),
            'quality_filter': ia_config.get('quality_filter', True)
        }
    
    def get_sequence_params(self) -> dict:
        """Devuelve parámetros de secuencia con valores predeterminados"""
        seq_params = self.config.get('sequence_params', {})
        return {
            'prime_base': seq_params.get('prime_base', 15485867),
            'xor_mask': seq_params.get('xor_mask', 2863355227288772495),
            'entropy_min': seq_params.get('entropy_min', 0.88),
            'zero_density_max': seq_params.get('zero_density_max', 0.12),
            'pattern_min': seq_params.get('pattern_min', 0.85)
        }
    
    def get_performance_settings(self) -> dict:
        """Devuelve ajustes de rendimiento con valores predeterminados"""
        perf_settings = self.config.get('performance_settings', {})
        return {
            'rebalance_interval': perf_settings.get('rebalance_interval', 300),
            'performance_window': perf_settings.get('performance_window', 1000),
            'min_weight': perf_settings.get('min_weight', 0.05),
            'max_weight': perf_settings.get('max_weight', 0.40)
        }
    
    def refresh_config(self):
        """Actualiza configuración si ha cambiado"""
        if time.time() - self.last_update > 300:
            self.config = self._load_config()
            self._validate_config()
            self.last_update = time.time()
            logger.info("Configuración actualizada")

class DataHandler:
    """Manejador profesional de datos para análisis en tiempo real"""
    REQUIRED_COLUMNS = [
        "nonce", "entropy", "uniqueness", "zero_density", 
        "pattern_score", "is_valid", "block_height", "timestamp"
    ]
    
    def __init__(self):
        self.training_path = DATA_DIR / "nonce_training_data.csv"
        self.success_path = DATA_DIR / "nonces_exitosos.csv"
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
                infer_datetime_format=True,
                on_bad_lines='skip'
            )
        except Exception as e:
            logger.exception(f"Error cargando datos: {str(e)}")
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
    
    def stream_successful_nonces(self, batch_size: int = 500) -> pd.DataFrame:
        """Stream de nonces exitosos para análisis continuo"""
        if not self.success_path.exists():
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        
        try:
            # Leer solo el último batch de nonces
            return pd.read_csv(
                self.success_path,
                usecols=self.REQUIRED_COLUMNS,
                dtype={'nonce': np.uint32}
            ).tail(batch_size)
        except Exception as e:
            logger.error(f"Error leyendo nonces exitosos: {str(e)}")
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
    
    def save_recommendations(self, recommendations: dict):
        """Guarda recomendaciones para otros módulos"""
        rec_path = RESULTS_DIR / "correlation_recommendations.json"
        with open(rec_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
    
    def send_nonces_to_orchestrator(self, nonces: pd.DataFrame):
        """Envía nonces al coordinador principal"""
        if nonces.empty:
            return
        
        try:
            # Guardar en archivo compartido
            header = not self.success_path.exists()
            nonces.to_csv(self.success_path, mode='a', header=header, index=False)
            logger.info(f"Enviados {len(nonces)} nonces optimizados al orquestador")
        except Exception as e:
            logger.error(f"Error enviando nonces: {str(e)}")

class CorrelationModel:
    """Modelo profesional de análisis de correlaciones"""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_weights = {}
        self.last_trained = 0
        self.ia_config = config.get_ia_config()
        self.sequence_params = config.get_sequence_params()
    
    def train(self, df: pd.DataFrame):
        """Entrenamiento profesional del modelo de correlación"""
        if df.empty or len(df) < 100:
            return
        
        try:
            # Preparar características
            features = df[['entropy', 'uniqueness', 'zero_density', 'pattern_score']].copy()
            target = df['is_valid'].astype(int)
            
            # Normalización profesional
            self.scaler.fit(features)
            features_scaled = self.scaler.transform(features)
            
            # Entrenar modelo de regresión lineal
            self.model = LinearRegression()
            self.model.fit(features_scaled, target)
            
            # Calcular pesos de características
            self.feature_weights = dict(zip(
                features.columns,
                self.model.coef_
            ))
            
            self.last_trained = time.time()
            logger.info(f"Modelo de correlación entrenado con {len(df)} muestras")
        except Exception as e:
            logger.exception(f"Error entrenando modelo: {str(e)}")
    
    def needs_retraining(self):
        """Determina si el modelo necesita reentrenamiento"""
        perf_settings = self.config.get_performance_settings()
        retrain_interval = perf_settings.get('retrain_interval', 3600)
        return (time.time() - self.last_trained) > retrain_interval
    
    def generate_optimized_nonces(self, count: int = 500) -> pd.DataFrame:
        """Genera nonces optimizados basados en correlaciones"""
        if not self.model or not self.feature_weights:
            return self._generate_high_quality_nonces(count)
        
        try:
            # Generar nonces con características optimizadas
            nonces = []
            for _ in range(count):
                features = self._optimize_features()
                features_scaled = self.scaler.transform(pd.DataFrame([features]))
                validity_prob = self.model.predict(features_scaled)[0]
                
                nonces.append({
                    'nonce': self._generate_nonce_value(),
                    'entropy': features['entropy'],
                    'uniqueness': features['uniqueness'],
                    'zero_density': features['zero_density'],
                    'pattern_score': features['pattern_score'],
                    'is_valid': validity_prob > 0.7,  # Umbral de aceptación
                    'block_height': 0,  # Se actualizará en minería real
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return pd.DataFrame(nonces)
        except Exception as e:
            logger.error(f"Error generando nonces: {str(e)}")
            return self._generate_high_quality_nonces(count)
    
    def _optimize_features(self) -> dict:
        """Optimiza características basadas en pesos y configuración"""
        features = {}
        weights = self.feature_weights
        
        # Entropía: siempre maximizar
        entropy_min = self.ia_config['min_entropy']
        features['entropy'] = np.random.uniform(entropy_min, entropy_min + 0.1)
        
        # Uniqueness: priorizar si peso positivo
        uniqueness_min = self.ia_config['min_uniqueness']
        if weights.get('uniqueness', 0) > 0:
            features['uniqueness'] = np.random.uniform(uniqueness_min, 0.95)
        else:
            features['uniqueness'] = np.random.uniform(0.7, uniqueness_min)
        
        # Zero density: minimizar si peso negativo
        zero_density_max = self.ia_config['max_zero_density']
        if weights.get('zero_density', 0) < 0:
            features['zero_density'] = np.random.uniform(0.05, zero_density_max)
        else:
            features['zero_density'] = np.random.uniform(zero_density_max, 0.3)
        
        # Pattern score: priorizar si peso positivo
        pattern_min = self.ia_config['min_pattern_score']
        if weights.get('pattern_score', 0) > 0:
            features['pattern_score'] = np.random.uniform(pattern_min, 0.95)
        else:
            features['pattern_score'] = np.random.uniform(0.7, pattern_min)
        
        return features
    
    def _generate_nonce_value(self) -> int:
        """Genera valores de nonce usando parámetros de secuencia"""
        prime = self.sequence_params['prime_base']
        xor_mask = self.sequence_params['xor_mask']
        base_value = int(time.time() * 1000) % (2**32)
        return (base_value * prime) ^ xor_mask
    
    def _generate_high_quality_nonces(self, count: int) -> pd.DataFrame:
        """Genera nonces de alta calidad como fallback"""
        nonces = []
        ia_config = self.ia_config
        
        for _ in range(count):
            nonces.append({
                'nonce': self._generate_nonce_value(),
                'entropy': np.random.uniform(ia_config['min_entropy'], 0.95),
                'uniqueness': np.random.uniform(ia_config['min_uniqueness'], 0.95),
                'zero_density': np.random.uniform(0.05, ia_config['max_zero_density']),
                'pattern_score': np.random.uniform(ia_config['min_pattern_score'], 0.95),
                'is_valid': True,
                'block_height': 0,
                'timestamp': datetime.utcnow().isoformat()
            })
        return pd.DataFrame(nonces)
    
    def analyze_correlations(self, df: pd.DataFrame) -> dict:
        """Análisis profesional de correlaciones"""
        if df.empty or len(df) < 50:
            return {}
        
        try:
            # Calcular matriz de correlación
            corr_matrix = df[['entropy', 'uniqueness', 'zero_density', 'pattern_score', 'is_valid']].corr()
            
            # Identificar correlaciones significativas
            strong_correlations = {}
            for feature in ['entropy', 'uniqueness', 'zero_density', 'pattern_score']:
                corr = corr_matrix.loc[feature, 'is_valid']
                if abs(corr) > 0.3:  # Umbral para correlación significativa
                    strong_correlations[feature] = corr
            
            # Calcular significancia estadística
            p_values = {}
            for feature in strong_correlations.keys():
                _, p_value = stats.pearsonr(df[feature], df['is_valid'].astype(int))
                p_values[feature] = p_value
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'p_values': p_values,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.exception(f"Error en análisis de correlaciones: {str(e)}")
            return {}

class CorrelationAnalyzer:
    """Sistema profesional de análisis de correlaciones en tiempo real"""
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_handler = DataHandler()
        self.model = CorrelationModel(config_manager)
        self.last_recommendations = {}
    
    def run_analysis_cycle(self):
        """Ciclo principal de análisis y generación"""
        # Actualizar configuración
        self.config_manager.refresh_config()
        
        # Cargar datos históricos
        historical_data = self.data_handler.load_training_data()
        
        # Filtrar por calidad si está habilitado
        if self.config_manager.get_ia_config().get('quality_filter', True):
            historical_data = self._apply_quality_filter(historical_data)
        
        # Entrenar modelo si es necesario
        if self.model.needs_retraining() and not historical_data.empty:
            self.model.train(historical_data)
        
        # Analizar correlaciones
        analysis_results = self.model.analyze_correlations(historical_data)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(analysis_results)
        self.last_recommendations = recommendations
        self.data_handler.save_recommendations(recommendations)
        
        # Generar nonces optimizados
        optimized_nonces = self.model.generate_optimized_nonces(500)
        self.data_handler.send_nonces_to_orchestrator(optimized_nonces)
    
    def _apply_quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra nonces según los umbrales de calidad"""
        if df.empty:
            return df
        
        ia_config = self.config_manager.get_ia_config()
        filtered = df[
            (df['entropy'] >= ia_config['min_entropy']) &
            (df['uniqueness'] >= ia_config['min_uniqueness']) &
            (df['zero_density'] <= ia_config['max_zero_density']) &
            (df['pattern_score'] >= ia_config['min_pattern_score'])
        ]
        
        removed = len(df) - len(filtered)
        if removed > 0:
            logger.info(f"Filtrados {removed} nonces por calidad insuficiente")
        
        return filtered
    
    def _generate_recommendations(self, analysis: dict) -> dict:
        """Genera recomendaciones basadas en análisis de correlaciones"""
        recommendations = {
            'priority_features': [],
            'avoid_features': [],
            'feature_optimization': {},
            'generator_weights': self._optimize_generator_weights(analysis),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if not analysis or 'strong_correlations' not in analysis:
            return recommendations
        
        # Identificar características prioritarias
        for feature, corr in analysis['strong_correlations'].items():
            p_value = analysis['p_values'].get(feature, 1)
            if corr > 0.3 and p_value < 0.05:
                recommendations['priority_features'].append(feature)
                recommendations['feature_optimization'][feature] = {
                    'target': 'maximize',
                    'min_value': self.config_manager.get_ia_config().get(f'min_{feature}', 0.7),
                    'weight': abs(corr)
                }
            elif corr < -0.3 and p_value < 0.05:
                recommendations['avoid_features'].append(feature)
                max_key = f'max_{feature}' if feature == 'zero_density' else f'min_{feature}'
                recommendations['feature_optimization'][feature] = {
                    'target': 'minimize',
                    'max_value': self.config_manager.get_ia_config().get(max_key, 0.3),
                    'weight': abs(corr)
                }
        
        # Recomendación para generadores
        if recommendations['priority_features']:
            rec_list = ", ".join(recommendations['priority_features'])
            recommendations['generator_advice'] = f"Priorizar características: {rec_list}"
        else:
            recommendations['generator_advice'] = "Mantener estrategia actual"
        
        return recommendations
    
    def _optimize_generator_weights(self, analysis: dict) -> dict:
        """Optimiza pesos de generadores basados en correlaciones"""
        base_weights = self.config_manager.config.get('generator_weights', {})
        optimized = base_weights.copy()
        
        if not analysis or 'strong_correlations' not in analysis:
            return optimized
        
        # Aumentar peso de generadores avanzados si hay correlaciones fuertes
        if any(abs(corr) > 0.5 for corr in analysis['strong_correlations'].values()):
            for gen in ['ml', 'adaptive', 'hybrid']:
                if gen in optimized:
                    optimized[gen] = min(
                        optimized[gen] * 1.2,
                        self.config_manager.get_performance_settings().get('max_weight', 0.4)
                    )
        
        # Reducir peso de generadores básicos si hay correlaciones débiles
        if all(abs(corr) < 0.2 for corr in analysis['strong_correlations'].values()):
            for gen in ['range', 'random']:
                if gen in optimized:
                    optimized[gen] = max(
                        optimized[gen] * 0.8,
                        self.config_manager.get_performance_settings().get('min_weight', 0.05)
                    )
        
        # Normalizar pesos
        total = sum(optimized.values())
        return {gen: weight/total for gen, weight in optimized.items()}

def analysis_worker(config_manager: ConfigManager):
    """Trabajador profesional para análisis continuo"""
    analyzer = CorrelationAnalyzer(config_manager)
    perf_settings = config_manager.get_performance_settings()
    interval = perf_settings.get('rebalance_interval', 300)
    
    while True:
        start_time = time.time()
        try:
            analyzer.run_analysis_cycle()
        except Exception as e:
            logger.error(f"Error en ciclo de análisis: {str(e)}")
        
        # Calcular tiempo de espera dinámico
        elapsed = time.time() - start_time
        sleep_time = max(10, interval - elapsed)
        time.sleep(sleep_time)

def main(config_name: str = "global_config"):
    """Punto de entrada profesional para el sistema de análisis"""
    try:
        logger.info("=" * 60)
        logger.info("INICIANDO SISTEMA PROFESIONAL DE ANÁLISIS DE CORRELACIONES")
        logger.info("=" * 60)
        
        # Inicializar componentes profesionales
        config_manager = ConfigManager(config_name)
        
        # Iniciar trabajador en segundo plano
        worker_thread = threading.Thread(
            target=analysis_worker,
            args=(config_manager,),
            daemon=True
        )
        worker_thread.start()
        
        logger.info("Sistema de análisis activo. Presiona Ctrl+C para salir.")
        worker_thread.join()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Análisis detenido por usuario")
        return 0
    except Exception as e:
        logger.critical(f"Fallo crítico: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sistema Industrial de Análisis de Correlaciones')
    parser.add_argument('--config', type=str, default='global_config',
                        help='Nombre de configuración (sin extensión .json)')
    args = parser.parse_args()
    
    exit_code = main(args.config)
    sys.exit(exit_code)