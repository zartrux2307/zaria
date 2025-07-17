import os
import sys
import time
import threading
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import entropy as scipy_entropy
from collections import Counter
import random
import argparse
import csv

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
        logging.FileHandler(DATA_DIR / "entropy_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EntropyAnalysis")

# Columnas estándar globales
COLUMNS = ["nonce", "entropy", "uniqueness", "zero_density", "pattern_score", "is_valid", "block_height", "timestamp"]

class ConfigManager:
    """Gestión profesional de configuraciones con validación de integridad"""
    _instance = None
    
    def __new__(cls, config_name: str = "global_config"):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.config_path = CONFIG_DIR / f"{config_name}.json"
            
            # Inicializar config con valores por defecto
            cls._instance.config = {}
            cls._instance._validate_config()
            
            try:
                # Intentar cargar configuración
                cls._instance.config = cls._instance._load_config()
            except Exception as e:
                logger.error(f"Error cargando configuración: {str(e)}")
                # Mantener valores por defecto si hay error
                
            cls._instance.last_update = time.time()
        return cls._instance
    
    def _load_config(self) -> dict:
        """Carga configuración con verificación de integridad"""
        if not self.config_path.exists():
            logger.error(f"Archivo de configuración no encontrado: {self.config_path}")
            return {}
        
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
            'quality_filter': ia_config.get('quality_filter', True),
            'target_entropy': ia_config.get('target_entropy', 0.92)
        }
    
    def get_performance_settings(self) -> dict:
        """Devuelve ajustes de rendimiento con valores predeterminados"""
        perf_settings = self.config.get('performance_settings', {})
        return {
            'analysis_interval': perf_settings.get('analysis_interval', 300),
            'generation_interval': perf_settings.get('generation_interval', 1),
            'entropy_window': perf_settings.get('entropy_window', 100000)
        }
    
    def refresh_config(self):
        """Actualiza configuración si ha cambiado"""
        if time.time() - self.last_update > 300:
            try:
                new_config = self._load_config()
                if new_config:
                    self.config = new_config
                    self._validate_config()
                    self.last_update = time.time()
                    logger.info("Configuración actualizada")
            except Exception as e:
                logger.error(f"Error actualizando configuración: {str(e)}")

class DataHandler:
    """Manejador profesional de datos para análisis en tiempo real"""
    REQUIRED_COLUMNS = COLUMNS
    
    def __init__(self):
        self.training_path = DATA_DIR / "nonce_training_data.csv"
        self.success_path = DATA_DIR / "nonces_exitosos.csv"
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
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
            logger.info(f"Enviados {len(nonces)} nonces optimizados al orquestador")
        except Exception as e:
            logger.error(f"Error enviando nonces: {str(e)}")
    
    def save_entropy_report(self, report: dict):
        """Guarda reporte de análisis de entropía"""
        report_path = RESULTS_DIR / "entropy_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Reporte de entropía guardado en: {report_path}")

class EntropyAnalyzer:
    """Sistema profesional de análisis y optimización de entropía"""
    MAX_NONCE = 2**32  # 4,294,967,295
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_handler = DataHandler()
        self.ia_config = config_manager.get_ia_config()
        self.perf_settings = config_manager.get_performance_settings()
        self.df = pd.DataFrame()
        self.last_analysis = 0
        self.entropy_threshold = self.ia_config['min_entropy']
    
    def run_analysis(self):
        """Ejecuta análisis completo de entropía"""
        start_time = time.time()
        
        try:
            # Actualizar configuración
            self.config_manager.refresh_config()
            self.ia_config = self.config_manager.get_ia_config()
            self.perf_settings = self.config_manager.get_performance_settings()
            
            # Cargar datos históricos
            max_rows = self.perf_settings.get('entropy_window', 100000)
            self.df = self.data_handler.load_training_data(max_rows)
            
            if self.df.empty:
                logger.warning("No hay datos para análisis de entropía")
                return
            
            # Filtrar por calidad si está habilitado
            if self.ia_config.get('quality_filter', True):
                self._apply_quality_filter()
            
            # Realizar análisis de entropía
            entropy_report = self._analyze_entropy()
            
            # Actualizar umbral de entropía
            self._update_entropy_threshold(entropy_report)
            
            # Guardar reporte
            self.data_handler.save_entropy_report(entropy_report)
            
            logger.info(f"Análisis de entropía completado en {time.time() - start_time:.2f} segundos")
            self.last_analysis = time.time()
        except Exception as e:
            logger.exception(f"Error en análisis de entropía: {str(e)}")
    
    def _apply_quality_filter(self):
        """Filtra nonces según los umbrales de calidad de configuración"""
        initial_count = len(self.df)
        
        self.df = self.df[
            (self.df['entropy'] >= self.ia_config['min_entropy']) &
            (self.df['uniqueness'] >= self.ia_config['min_uniqueness']) &
            (self.df['zero_density'] <= self.ia_config['max_zero_density']) &
            (self.df['pattern_score'] >= self.ia_config['min_pattern_score'])
        ]
        
        removed = initial_count - len(self.df)
        if removed > 0:
            logger.info(f"Filtrados {removed} nonces por calidad insuficiente")
    
    def _analyze_entropy(self) -> dict:
        """Realiza análisis profesional de entropía"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'data_points': len(self.df),
            'entropy_stats': {},
            'correlation': {}
        }
        
        if self.df.empty:
            return report
        
        try:
            # Estadísticas básicas de entropía
            entropy_series = self.df['entropy']
            report['entropy_stats'] = {
                'mean': float(entropy_series.mean()),
                'median': float(entropy_series.median()),
                'std': float(entropy_series.std()),
                'min': float(entropy_series.min()),
                'max': float(entropy_series.max()),
                'q1': float(entropy_series.quantile(0.25)),
                'q3': float(entropy_series.quantile(0.75))
            }
            
            # Correlación con otras métricas
            for metric in ['uniqueness', 'zero_density', 'pattern_score']:
                if metric in self.df.columns:
                    corr = self.df['entropy'].corr(self.df[metric])
                    report['correlation'][metric] = float(corr)
            
            # Distribución de entropía
            report['entropy_distribution'] = {
                'bins': [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
                'counts': pd.cut(
                    self.df['entropy'], 
                    bins=report['entropy_distribution']['bins'],
                    include_lowest=True
                ).value_counts().to_dict()
            }
            
            # Entropía vs validez
            if 'is_valid' in self.df.columns:
                valid_entropy = self.df[self.df['is_valid']]['entropy'].mean()
                invalid_entropy = self.df[~self.df['is_valid']]['entropy'].mean()
                report['validity_comparison'] = {
                    'valid_mean': float(valid_entropy),
                    'invalid_mean': float(invalid_entropy),
                    'difference': float(valid_entropy - invalid_entropy)
                }
            
            logger.info("Análisis de entropía completado exitosamente")
            return report
        except Exception as e:
            logger.error(f"Error en análisis de entropía: {str(e)}")
            return report
    
    def _update_entropy_threshold(self, report: dict):
        """Actualiza dinámicamente el umbral de entropía basado en el análisis"""
        if not report or 'entropy_stats' not in report:
            return
        
        try:
            # Estrategia adaptativa: aumentar umbral si la media es alta
            current_mean = report['entropy_stats']['mean']
            target_entropy = self.ia_config.get('target_entropy', 0.92)
            
            if current_mean > target_entropy:
                # Subir umbral si el 75% de los nonces están por encima del objetivo
                q3 = report['entropy_stats']['q3']
                if q3 > target_entropy:
                    self.entropy_threshold = min(
                        self.entropy_threshold * 1.05,
                        0.95  # Límite máximo
                    )
                    logger.info(f"Umbral de entropía aumentado a {self.entropy_threshold:.4f}")
            else:
                # Bajar umbral si la media es baja
                self.entropy_threshold = max(
                    self.entropy_threshold * 0.98,
                    self.ia_config['min_entropy']  # Límite mínimo
                )
                logger.info(f"Umbral de entropía reducido a {self.entropy_threshold:.4f}")
        except Exception as e:
            logger.error(f"Error actualizando umbral de entropía: {str(e)}")
    
    def generate_high_entropy_nonces(self, count: int = 500) -> pd.DataFrame:
        """Genera nonces con alta entropía optimizada"""
        nonces = []
        target_entropy = self.entropy_threshold
        
        for _ in range(count):
            # Generar nonce aleatorio
            nonce_val = random.randint(0, self.MAX_NONCE)
            
            # Convertir a binario y calcular métricas
            nonce_bin = bin(nonce_val)[2:].zfill(32)
            entropy = self._calculate_binary_entropy(nonce_bin)
            uniqueness = self._calculate_uniqueness(nonce_bin)
            zero_density = nonce_bin.count('0') / len(nonce_bin)
            pattern_score = self._calculate_pattern_score(nonce_bin)
            
            # Ajustar para alcanzar el umbral de entropía
            if entropy < target_entropy:
                nonce_val, entropy = self._optimize_entropy(nonce_val, target_entropy)
            
            nonces.append({
                'nonce': nonce_val,
                'entropy': entropy,
                'uniqueness': uniqueness,
                'zero_density': zero_density,
                'pattern_score': pattern_score,
                'is_valid': False,  # Se validará durante la minería
                'block_height': 0,  # Se actualizará en minería real
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return pd.DataFrame(nonces)
    
    def _calculate_binary_entropy(self, binary_str: str) -> float:
        """Calcula entropía de Shannon para una cadena binaria"""
        if len(binary_str) == 0:
            return 0.0
        
        counts = Counter(binary_str)
        probs = [count / len(binary_str) for count in counts.values()]
        return scipy_entropy(probs, base=2)
    
    def _calculate_uniqueness(self, binary_str: str) -> float:
        """Calcula métrica de unicidad basada en subsecuencias"""
        # Dividir en subsecuencias de 8 bits
        chunks = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]
        return len(set(chunks)) / len(chunks)
    
    def _calculate_pattern_score(self, binary_str: str) -> float:
        """Calcula puntuación anti-patrón"""
        # Penalizar secuencias repetitivas
        max_repeat = max(len(seq) for seq in binary_str.replace('01', '0 1').replace('10', '1 0').split())
        repeat_penalty = min(max_repeat / 8, 1.0)  # Normalizar
        
        # Penalizar patrones periódicos
        periodic_penalty = 0
        for pattern in ['01010101', '10101010', '00110011', '11001100']:
            if pattern in binary_str:
                periodic_penalty += 0.2
        
        # Combinar penalizaciones
        total_penalty = min(repeat_penalty + periodic_penalty, 1.0)
        return 1.0 - total_penalty
    
    def _optimize_entropy(self, nonce: int, target_entropy: float) -> tuple:
        """Optimiza un nonce para alcanzar la entropía objetivo"""
        original_bin = bin(nonce)[2:].zfill(32)
        best_entropy = self._calculate_binary_entropy(original_bin)
        best_nonce = nonce
        
        # Si ya cumple, retornar
        if best_entropy >= target_entropy:
            return nonce, best_entropy
        
        # Intentar optimizar con mutaciones aleatorias
        for _ in range(100):  # Límite de intentos
            # Crear mutación: voltear bits aleatorios
            mutated_bin = list(original_bin)
            num_flips = random.randint(1, 4)
            for _ in range(num_flips):
                idx = random.randint(0, 31)
                mutated_bin[idx] = '1' if mutated_bin[idx] == '0' else '0'
            
            mutated_bin = ''.join(mutated_bin)
            mutated_entropy = self._calculate_binary_entropy(mutated_bin)
            
            # Verificar si es mejor
            if mutated_entropy > best_entropy:
                best_entropy = mutated_entropy
                best_nonce = int(mutated_bin, 2)
                
                if best_entropy >= target_entropy:
                    break
        
        return best_nonce, best_entropy

def analysis_worker(config_manager: ConfigManager, analyzer: EntropyAnalyzer):
    """Trabajador profesional para análisis periódico"""
    logger.info("Iniciando trabajador de análisis de entropía")
    
    perf_settings = config_manager.get_performance_settings()
    interval = perf_settings.get('analysis_interval', 300)
    
    while True:
        try:
            analyzer.run_analysis()
        except Exception as e:
            logger.error(f"Error en ciclo de análisis: {str(e)}")
        
        # Esperar hasta el próximo ciclo
        time.sleep(interval)

def generation_worker(analyzer: EntropyAnalyzer, data_handler: DataHandler):
    """Trabajador profesional para generación de nonces"""
    logger.info("Iniciando trabajador de generación de nonces")
    
    perf_settings = analyzer.config_manager.get_performance_settings()
    interval = perf_settings.get('generation_interval', 1)
    
    while True:
        start_time = time.time()
        try:
            # Generar nonces con alta entropía
            nonces = analyzer.generate_high_entropy_nonces(500)
            
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
        logger.info("INICIANDO SISTEMA PROFESIONAL DE ANÁLISIS DE ENTROPÍA")
        logger.info("=" * 60)
        
        # Inicializar componentes profesionales
        config_manager = ConfigManager(config_name)
        data_handler = DataHandler()
        analyzer = EntropyAnalyzer(config_manager)
        
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
        
        logger.info("Sistema de entropía activo. Presiona Ctrl+C para salir.")
        
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
    parser = argparse.ArgumentParser(description='Sistema Industrial de Análisis de Entropía')
    parser.add_argument('--config', type=str, default='global_config',
                        help='Nombre de configuración (sin extensión .json)')
    args = parser.parse_args()
    
    exit_code = main(args.config)
    sys.exit(exit_code)