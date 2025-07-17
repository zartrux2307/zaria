import numpy as np
import pandas as pd
import os
import sys
import time
import threading
import logging
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, kendalltau, norm
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # Para uso sin interfaz gráfica
import matplotlib.pyplot as plt
import seaborn as sns
from iazar.generator.config_loader import config_loader
# Configuración profesional de rutas
PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = DATA_DIR / "analysis_results"
MODELS_DIR = PROJECT_DIR / "models"

sys.path.insert(0, str(PROJECT_DIR))
os.chdir(str(PROJECT_DIR))

# Configuración profesional de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / "distribution_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DistributionAnalyzer")

# Columnas estándar globales
COLUMNS = ["nonce", "entropy", "uniqueness", "zero_density", "pattern_score", "is_valid", "block_height", "timestamp"]

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
    
    def get_performance_settings(self) -> dict:
        """Devuelve ajustes de rendimiento con valores predeterminados"""
        perf_settings = self.config.get('performance_settings', {})
        return {
            'analysis_interval': perf_settings.get('analysis_interval', 3600),
            'generation_interval': perf_settings.get('generation_interval', 1),
            'distribution_window': perf_settings.get('distribution_window', 100000)
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
    REQUIRED_COLUMNS = COLUMNS
    
    def __init__(self):
        self.training_path = DATA_DIR / "nonce_training_data.csv"
        self.success_path = DATA_DIR / "nonces_exitosos.csv"
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        MODELS_DIR.mkdir(exist_ok=True, parents=True)
    
    def load_training_data(self, max_rows: int = None) -> pd.DataFrame:
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
            
            # Leer solo las columnas necesarias
            usecols = [col for col in self.REQUIRED_COLUMNS if col != 'timestamp']
            usecols.append('timestamp')
            
            # Manejar archivos grandes con lectura eficiente
            if max_rows:
                return pd.read_csv(
                    self.training_path,
                    usecols=usecols,
                    dtype=dtype_map,
                    parse_dates=['timestamp'],
                    infer_datetime_format=True,
                    nrows=max_rows
                )
            else:
                return pd.read_csv(
                    self.training_path,
                    usecols=usecols,
                    dtype=dtype_map,
                    parse_dates=['timestamp'],
                    infer_datetime_format=True
                )
        except Exception as e:
            logger.exception(f"Error cargando datos: {str(e)}")
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
    
    def send_nonces_to_orchestrator(self, nonces: pd.DataFrame):
        """Envía nonces al coordinador principal"""
        if nonces.empty:
            return
        
        try:
            # Guardar en archivo compartido
            header = not self.success_path.exists()
            nonces.to_csv(self.success_path, mode='a', header=header, index=False)
            logger.info(f"Enviados {len(nonces)} nonces al orquestador")
        except Exception as e:
            logger.error(f"Error enviando nonces: {str(e)}")
    
    def save_distribution_model(self, model: dict, filename: str = "distribution_model.json"):
        """Guarda el modelo de distribución para uso futuro"""
        model_path = MODELS_DIR / filename
        with open(model_path, 'w') as f:
            json.dump(model, f, indent=2)
        logger.info(f"Modelo de distribución guardado en: {model_path}")

class DistributionModel:
    """Modelo profesional de distribución para análisis y generación"""
    MAX_NONCE = 2**32  # 4,294,967,295
    
    def __init__(self, config: dict):
        self.config = config
        self.distribution_params = {}
        self.correlation_matrix = pd.DataFrame()
        self.last_updated = 0
        self.feature_distributions = {}
    
    def fit(self, df: pd.DataFrame):
        """Ajusta modelos de distribución a los datos"""
        if df.empty:
            return
        
        try:
            # Ajustar distribución de nonces
            self._fit_nonce_distribution(df['nonce'])
            
            # Ajustar distribuciones de características
            for feature in ['entropy', 'uniqueness', 'zero_density', 'pattern_score']:
                if feature in df.columns:
                    self._fit_feature_distribution(df[feature], feature)
            
            # Calcular matriz de correlación
            self.correlation_matrix = df[['entropy', 'uniqueness', 'zero_density', 'pattern_score']].corr()
            
            self.last_updated = time.time()
            logger.info("Modelo de distribución ajustado exitosamente")
        except Exception as e:
            logger.exception(f"Error ajustando modelo de distribución: {str(e)}")
    
    def _fit_nonce_distribution(self, nonce_series: pd.Series):
        """Ajusta una distribución de probabilidad a los nonces"""
        # Usar histograma para estimar la densidad empírica
        hist, bin_edges = np.histogram(nonce_series, bins=1000, density=True)
        
        # Almacenar la distribución empírica
        self.distribution_params['nonce'] = {
            'type': 'empirical',
            'hist': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'min': int(nonce_series.min()),
            'max': int(nonce_series.max()),
            'mean': float(nonce_series.mean()),
            'std': float(nonce_series.std())
        }
    
    def _fit_feature_distribution(self, feature_series: pd.Series, feature_name: str):
        """Ajusta una distribución normal a las características"""
        try:
            # Intentar ajustar una distribución normal
            (mu, sigma) = norm.fit(feature_series)
            self.feature_distributions[feature_name] = {
                'type': 'normal',
                'mu': mu,
                'sigma': sigma
            }
        except:
            # Fallback a distribución empírica
            hist, bin_edges = np.histogram(feature_series, bins=100, density=True)
            self.feature_distributions[feature_name] = {
                'type': 'empirical',
                'hist': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
    
    def generate_nonces(self, count: int = 500) -> pd.DataFrame:
        """Genera nonces basados en la distribución aprendida"""
        if not self.distribution_params.get('nonce'):
            return self._generate_random_nonces(count)
        
        try:
            nonces = []
            nonce_dist = self.distribution_params['nonce']
            
            for _ in range(count):
                # Generar nonce basado en distribución empírica
                if nonce_dist['type'] == 'empirical':
                    nonce = self._sample_empirical_distribution(nonce_dist)
                else:
                    nonce = np.random.randint(nonce_dist['min'], nonce_dist['max'] + 1)
                
                # Generar características basadas en distribuciones
                features = {}
                for feature in ['entropy', 'uniqueness', 'zero_density', 'pattern_score']:
                    if feature in self.feature_distributions:
                        dist = self.feature_distributions[feature]
                        if dist['type'] == 'normal':
                            features[feature] = np.random.normal(dist['mu'], dist['sigma'])
                        else:
                            features[feature] = self._sample_empirical_distribution(dist)
                
                # Asegurar valores dentro de rangos válidos
                features['entropy'] = np.clip(features['entropy'], 0.01, 0.99)
                features['uniqueness'] = np.clip(features['uniqueness'], 0.01, 0.99)
                features['zero_density'] = np.clip(features['zero_density'], 0.01, 0.99)
                features['pattern_score'] = np.clip(features['pattern_score'], 0.01, 0.99)
                
                nonces.append({
                    'nonce': int(nonce),
                    'entropy': features['entropy'],
                    'uniqueness': features['uniqueness'],
                    'zero_density': features['zero_density'],
                    'pattern_score': features['pattern_score'],
                    'is_valid': False,  # Se validará durante la minería
                    'block_height': 0,  # Se actualizará en minería real
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return pd.DataFrame(nonces)
        except Exception as e:
            logger.error(f"Error generando nonces: {str(e)}")
            return self._generate_random_nonces(count)
    
    def _sample_empirical_distribution(self, dist: dict) -> float:
        """Muestrea de una distribución empírica basada en histograma"""
        # Seleccionar bin basado en probabilidades
        probs = dist['hist']
        bin_idx = np.random.choice(len(probs), p=probs/np.sum(probs))
        
        # Generar valor uniforme dentro del bin
        low = dist['bin_edges'][bin_idx]
        high = dist['bin_edges'][bin_idx + 1]
        return np.random.uniform(low, high)
    
    def _generate_random_nonces(self, count: int) -> pd.DataFrame:
        """Genera nonces aleatorios como fallback"""
        nonces = []
        ia_config = self.config.get_ia_config()
        
        for _ in range(count):
            nonces.append({
                'nonce': np.random.randint(0, self.MAX_NONCE),
                'entropy': np.random.uniform(ia_config['min_entropy'], 0.95),
                'uniqueness': np.random.uniform(ia_config['min_uniqueness'], 0.95),
                'zero_density': np.random.uniform(0.05, ia_config['max_zero_density']),
                'pattern_score': np.random.uniform(ia_config['min_pattern_score'], 0.95),
                'is_valid': False,
                'block_height': 0,
                'timestamp': datetime.utcnow().isoformat()
            })
        return pd.DataFrame(nonces)
    
    def plot_distribution(self, df: pd.DataFrame, feature: str, save_path: Path):
        """Visualización profesional de distribuciones"""
        if df.empty or feature not in df.columns:
            return
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Histograma de la característica
            sns.histplot(df[feature], kde=True, color='skyblue', stat='density')
            
            # Añadir línea de distribución ajustada si está disponible
            if feature in self.feature_distributions:
                dist = self.feature_distributions[feature]
                if dist['type'] == 'normal':
                    x = np.linspace(df[feature].min(), df[feature].max(), 100)
                    plt.plot(x, norm.pdf(x, dist['mu'], dist['sigma']), 'r-', lw=2)
            
            # Configuración profesional
            plt.title(f'Distribución de {feature}', fontsize=14)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('Densidad', fontsize=12)
            plt.grid(alpha=0.3)
            
            # Guardar figura
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            logger.info(f"Gráfico de distribución guardado en: {save_path}")
        except Exception as e:
            logger.error(f"Error generando gráfico: {str(e)}")

class DistributionAnalyzer:
    """Sistema profesional de análisis de distribución en tiempo real"""
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_handler = DataHandler()
        self.model = DistributionModel(config_manager)
        self.df = pd.DataFrame()
        self.last_analysis = 0
    
    def run_analysis(self):
        """Ejecuta análisis completo de distribución"""
        start_time = time.time()
        
        try:
            # Actualizar configuración
            self.config_manager.refresh_config()
            
            # Cargar datos históricos
            perf_settings = self.config_manager.get_performance_settings()
            max_rows = perf_settings.get('distribution_window', 100000)
            self.df = self.data_handler.load_training_data(max_rows)
            
            if self.df.empty:
                logger.warning("No hay datos para análisis de distribución")
                return
            
            # Filtrar por calidad si está habilitado
            if self.config_manager.get_ia_config().get('quality_filter', True):
                self._apply_quality_filter()
            
            # Ajustar modelo de distribución
            self.model.fit(self.df)
            
            # Generar visualizaciones
            self._generate_visualizations()
            
            # Guardar modelo
            self.data_handler.save_distribution_model({
                'distribution_params': self.model.distribution_params,
                'feature_distributions': self.model.feature_distributions,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info(f"Análisis completado en {time.time() - start_time:.2f} segundos")
            self.last_analysis = time.time()
        except Exception as e:
            logger.exception(f"Error en análisis de distribución: {str(e)}")
    
    def _apply_quality_filter(self):
        """Filtra nonces según los umbrales de calidad de configuración"""
        ia_config = self.config_manager.get_ia_config()
        initial_count = len(self.df)
        
        self.df = self.df[
            (self.df['entropy'] >= ia_config['min_entropy']) &
            (self.df['uniqueness'] >= ia_config['min_uniqueness']) &
            (self.df['zero_density'] <= ia_config['max_zero_density']) &
            (self.df['pattern_score'] >= ia_config['min_pattern_score'])
        ]
        
        removed = initial_count - len(self.df)
        if removed > 0:
            logger.info(f"Filtrados {removed} nonces por calidad insuficiente")
    
    def _generate_visualizations(self):
        """Genera visualizaciones profesionales de distribuciones"""
        # Crear directorio para reportes
        report_dir = RESULTS_DIR / "distribution_reports"
        report_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generar gráficos para cada característica
        for feature in ['nonce', 'entropy', 'uniqueness', 'zero_density', 'pattern_score']:
            if feature in self.df.columns:
                save_path = report_dir / f"{feature}_distribution_{timestamp}.png"
                self.model.plot_distribution(self.df, feature, save_path)
        
        # Generar matriz de correlación
        try:
            plt.figure(figsize=(10, 8))
            corr_matrix = self.df[['entropy', 'uniqueness', 'zero_density', 'pattern_score']].corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
            plt.title('Matriz de Correlación', fontsize=14)
            plt.tight_layout()
            plt.savefig(report_dir / f"correlation_matrix_{timestamp}.png", dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"Error generando matriz de correlación: {str(e)}")
    
    def generate_nonces(self, count: int = 500) -> pd.DataFrame:
        """Genera nonces basados en la distribución aprendida"""
        return self.model.generate_nonces(count)

def analysis_worker(config_manager: ConfigManager, analyzer: DistributionAnalyzer):
    """Trabajador profesional para análisis periódico"""
    logger.info("Iniciando trabajador de análisis de distribución")
    
    perf_settings = config_manager.get_performance_settings()
    interval = perf_settings.get('analysis_interval', 3600)
    
    while True:
        try:
            analyzer.run_analysis()
        except Exception as e:
            logger.error(f"Error en ciclo de análisis: {str(e)}")
        
        # Esperar hasta el próximo ciclo
        time.sleep(interval)

def generation_worker(analyzer: DistributionAnalyzer, data_handler: DataHandler):
    """Trabajador profesional para generación de nonces"""
    logger.info("Iniciando trabajador de generación de nonces")
    
    perf_settings = analyzer.config_manager.get_performance_settings()
    interval = perf_settings.get('generation_interval', 1)
    
    while True:
        start_time = time.time()
        try:
            # Generar nonces basados en la distribución
            nonces = analyzer.generate_nonces(500)
            
            # Enviar al orquestador
            data_handler.send_nonces_to_orchestrator(nonces)
        except Exception as e:
            logger.error(f"Error en ciclo de generación: {str(e)}")
        
        # Calcular tiempo de espera dinámico
        elapsed = time.time() - start_time
        sleep_time = max(0.1, interval - elapsed)
        time.sleep(sleep_time)

def main(config_name: str = "global_config"):
    """Punto de entrada profesional para el sistema de análisis"""
    try:
        logger.info("=" * 60)
        logger.info("INICIANDO SISTEMA PROFESIONAL DE ANÁLISIS DE DISTRIBUCIÓN")
        logger.info("=" * 60)
        
        # Inicializar componentes profesionales
        config_manager = ConfigManager(config_name)
        data_handler = DataHandler()
        analyzer = DistributionAnalyzer(config_manager)
        
        # Ejecutar análisis inicial
        analyzer.run_analysis()
        
        # Iniciar trabajadores en segundo plano
        analysis_thread = threading.Thread(
            target=analysis_worker,
            args=(config_manager, analyzer),
            daemon=True
        )
        generation_thread = threading.Thread(
            target=generation_worker,
            args=(analyzer, data_handler),
            daemon=True
        )
        
        analysis_thread.start()
        generation_thread.start()
        
        logger.info("Sistema de distribución activo. Presiona Ctrl+C para salir.")
        
        # Mantener el hilo principal activo
        while True:
            time.sleep(3600)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Análisis detenido por usuario")
        return 0
    except Exception as e:
        logger.critical(f"Fallo crítico: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sistema Industrial de Análisis de Distribución')
    parser.add_argument('--config', type=str, default='global_config',
                        help='Nombre de configuración (sin extensión .json)')
    args = parser.parse_args()
    
    exit_code = main(args.config)
    sys.exit(exit_code)