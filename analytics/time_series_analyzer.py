import logging
import mlflow
import json
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union
from iazar.utils.nonce_loader import NonceLoader
from iazar.utils.config_manager import ConfigManager
from iazar.utils.data_preprocessing import NonceDataPreprocessor
from iazar.utils.feature_utils import COLUMNS
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)
# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TimeSeriesAnalyzer")


class TimeSeriesAnalyzer:
    """Analizador avanzado de series temporales para datos de minería con generación de timestamps sintéticos."""

    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.loader = NonceLoader(self.config)
        self.preprocessor = NonceDataPreprocessor()
        self._initialize_parameters()

        # Configurar MLFlow si está activo
        self.mlflow_tracking = self.config.get_config_value('mlflow', 'tracking', False)
        if self.mlflow_tracking:
            mlflow.set_tracking_uri(self.config.get_config_value('mlflow', 'tracking_uri', 'http://localhost:5000'))
            mlflow.set_experiment(self.config.get_config_value('mlflow', 'experiment', 'iazar-timeseries'))

    def _initialize_parameters(self):
        """Carga parámetros de configuración con valores por defecto robustos"""
        self.window_sizes = self.config.get_config_value('ts_config', 'window_sizes', [10, 50, 100])
        self.metrics = self.config.get_config_value('ts_config', 'metrics', ['ma', 'ema', 'std'])

        # Parámetros para generación de timestamps sintéticos
        self.synthetic_start = self.config.get_config_value('ts_config', 'synthetic_start', '2023-01-01')
        self.synthetic_interval = self.config.get_config_value('ts_config', 'synthetic_interval', 60)  # segundos

        # Directorio de reportes
        reports_path = self.config.get_config_value('ts_config', 'reports_path', 'iazar/logs/reports/timeseries')
        self.report_path = Path(reports_path)
        self.report_path.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Carga datos y genera timestamps sintéticos si son necesarios"""
        try:
            df = self.loader.load_hash_data()

            # Verificar si ya tiene timestamp
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp').sort_index()
                    logger.info("Datos temporales cargados con timestamp existente")
                    return df
                except Exception as e:
                    logger.warning(f"Error procesando timestamp: {str(e)} - Generando timestamps sintéticos")

            # Generar timestamps sintéticos si no existen o son inválidos
            logger.info("Generando timestamps sintéticos para datos")
            start_time = pd.to_datetime(self.synthetic_start)
            intervals = pd.date_range(
                start=start_time,
                periods=len(df),
                freq=timedelta(seconds=self.synthetic_interval)
            )

            df['timestamp'] = intervals
            df = df.set_index('timestamp').sort_index()

            return self.preprocessor.preprocess(df)

        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            return pd.DataFrame()

    def validate_time_series(self, df: pd.DataFrame) -> bool:
        """Valida que el DataFrame tenga la estructura adecuada para análisis temporal"""
        if df.empty:
            logger.error("DataFrame vacío recibido para validación")
            return False

        required_columns = ['nonce', 'is_valid'] + [col for col in COLUMNS if col not in ['nonce', 'is_valid']]
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            logger.error(f"Columnas requeridas faltantes: {', '.join(missing)}")
            return False

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("El índice no es de tipo DateTimeIndex")
            return False

        if df.index.is_monotonic_increasing:
            logger.info("Índice temporal válido y ordenado")
        else:
            logger.warning("Índice temporal no está ordenado - Reordenando")
            df = df.sort_index()

        return True

    def calculate_moving_average(self, data: pd.Series, window: int) -> pd.Series:
        """Calcula media móvil con validación de ventana robusta"""
        if len(data) < window:
            new_window = max(1, len(data) // 2)
            logger.warning(f"Ventana ajustada de {window} a {new_window} por tamaño de datos insuficiente")
            window = new_window
        return data.rolling(window=window, min_periods=1).mean()

    def calculate_ema(self, data: pd.Series, span: int) -> pd.Series:
        """Calcula media móvil exponencial con ajuste automático de span"""
        if span > len(data):
            span = max(2, len(data) // 2)
            logger.warning(f"Span de EMA ajustado a {span}")
        return data.ewm(span=span, adjust=False).mean()

    def calculate_volatility(self, data: pd.Series, window: int) -> pd.Series:
        """Calcula volatilidad con manejo de datos insuficientes"""
        if len(data) < window:
            window = max(2, len(data) // 2)
            logger.warning(f"Ventana de volatilidad ajustada a {window}")
        return data.rolling(window=window).std()

    def analyze(self, feature: str = 'hash_score') -> Dict[str, Union[pd.Series, dict]]:
        """Ejecuta análisis completo de series temporales con validación"""
        try:
            df = self.load_and_prepare_data()
            if not self.validate_time_series(df):
                logger.error("Datos temporales inválidos - Abortando análisis")
                return {}

            if feature not in df.columns:
                logger.error(f"Característica '{feature}' no encontrada en los datos")
                return {}

            results = {}
            logger.info(f"Iniciando análisis de serie temporal para característica: {feature}")

            # Calcular métricas solicitadas
            for window in self.window_sizes:
                if 'ma' in self.metrics:
                    results[f'ma_{window}'] = self.calculate_moving_average(df[feature], window)
                if 'ema' in self.metrics:
                    results[f'ema_{window}'] = self.calculate_ema(df[feature], window)
                if 'std' in self.metrics:
                    results[f'std_{window}'] = self.calculate_volatility(df[feature], window)

            # Generar visualizaciones y métricas
            self._generate_visualizations(results, feature)
            analysis_report = {
                'series': {k: v.tolist() for k, v in results.items()},
                'cross_correlation': self._calculate_cross_correlation(results).to_dict(),
                'stationarity_test': self._check_stationarity(df[feature])
            }

            # Guardar reporte completo
            report_file = self.report_path / f"ts_analysis_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(analysis_report, f, indent=2)

            logger.info(f"Análisis completado y guardado en {report_file}")
            return analysis_report

        except Exception as e:
            logger.error(f"Error crítico en análisis: {str(e)}", exc_info=True)
            return {}

    def _generate_visualizations(self, data: dict, feature: str):
        """Genera visualizaciones profesionales con múltiples subplots"""
        try:
            fig, axes = plt.subplots(len(data), 1, figsize=(14, 8), sharex=True)
            fig.suptitle(f"Análisis Temporal de {feature}", fontsize=16)

            if len(data) == 1:
                axes = [axes]

            for i, (key, series) in enumerate(data.items()):
                ax = axes[i]
                ax.plot(series, label=key.replace('_', ' ').upper(), color='navy')
                ax.set_title(key.upper())
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

            plt.xlabel("Timestamp", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            plot_file = self.report_path / f"ts_analysis_{feature}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            if self.mlflow_tracking and mlflow.active_run():
                mlflow.log_artifact(str(plot_file))
                logger.info(f"Visualización guardada y registrada en MLflow: {plot_file}")

        except Exception as e:
            logger.error(f"Error generando visualizaciones: {str(e)}")

    def _log_metrics(self, data: dict):
        """Registra métricas clave en MLflow con prefijo"""
        try:
            if not self.mlflow_tracking or not mlflow.active_run():
                return

            metrics = {}
            for key, series in data.items():
                if len(series) > 0:
                    metrics[f"ts_{key}_final"] = series[-1]
                    metrics[f"ts_{key}_mean"] = np.mean(series)
                    metrics[f"ts_{key}_max"] = np.max(series)
                    metrics[f"ts_{key}_min"] = np.min(series)

            mlflow.log_metrics(metrics)
            logger.info(f"Métricas registradas en MLflow: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Error registrando métricas en MLflow: {str(e)}")

    def _calculate_cross_correlation(self, data: dict) -> pd.DataFrame:
        """Calcula matriz de correlación entre métricas con manejo de NaN"""
        df = pd.DataFrame(data)
        return df.corr()

    def _check_stationarity(self, series: pd.Series, test: str = 'adfuller') -> Dict:
        """Realiza test de estacionalidad con manejo robusto"""
        try:
            from statsmodels.tsa.stattools import adfuller

            # Preparar datos (eliminar NaNs)
            clean_series = series.dropna()
            if len(clean_series) < 10:
                logger.warning("Datos insuficientes para test de estacionalidad")
                return {'error': 'insufficient_data'}

            result = adfuller(clean_series)
            return {
                'test': test,
                'statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except ImportError:
            logger.error("statsmodels no instalado - No se puede realizar test de estacionalidad")
            return {'error': 'statsmodels_not_installed'}
        except Exception as e:
            logger.error(f"Error en test de estacionalidad: {str(e)}")
            return {'error': str(e)}

    def generate_features(self, target: str = 'hash_score') -> pd.DataFrame:
        """Genera características para modelos de forecasting con validación"""
        df = self.load_and_prepare_data()
        if not self.validate_time_series(df) or target not in df.columns:
            return pd.DataFrame()

        features = pd.DataFrame(index=df.index)
        logger.info(f"Generando características para forecasting de {target}")

        # Características temporales básicas
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month

        # Métricas de ventana
        for window in self.window_sizes:
            if 'ma' in self.metrics:
                features[f'ma_{window}'] = self.calculate_moving_average(df[target], window)
            if 'ema' in self.metrics:
                features[f'ema_{window}'] = self.calculate_ema(df[target], window)
            if 'std' in self.metrics:
                features[f'std_{window}'] = self.calculate_volatility(df[target], window)

        # Retardos temporales
        for lag in [1, 2, 3, 7]:
            features[f'lag_{lag}'] = df[target].shift(lag)

        features['target'] = df[target]
        features = features.dropna()

        # Guardar características generadas
        features_file = self.report_path / f"forecasting_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        features.to_csv(features_file)
        logger.info(f"Características para forecasting guardadas en {features_file}")

        return features

    @classmethod
    def example_usage(cls):
        """Ejemplo de integración con el proyecto"""
        logger.info("Ejecutando ejemplo de uso de TimeSeriesAnalyzer")

        try:
            config = ConfigManager()
            analyzer = cls(config)

            # Ejecutar análisis
            analysis_results = analyzer.analyze('nonce')
            logger.info("Análisis completado. Métricas calculadas: %s", list(analysis_results.get('series', {}).keys()))

            # Generar características para forecasting
            features = analyzer.generate_features('is_valid')
            logger.info("Características generadas para forecasting. Columnas: %s", features.columns.tolist())

            return features

        except Exception:
            logger.exception("Error en ejemplo de uso")
            return None


if __name__ == "__main__":
    # Configurar logging para ejecución directa
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("time_series_analyzer.log")
        ]
    )

    TimeSeriesAnalyzer.example_usage()
