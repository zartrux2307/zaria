import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from typing import Tuple, Dict, Any
import logging
import os
import sys
import json
import time

from iazar.utils.nonce_loader import NonceLoader
from iazar.data.create_initial_data import generate_core_datasets

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# Configuración avanzada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fourier_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FourierAnalyzer")

def safe_load_data(file_path: str) -> pd.DataFrame:
    """
    Carga segura de datos con generación de fallback si el archivo no existe o está vacío.
    
    Args:
        file_path (str): Ruta al archivo CSV a cargar
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados o generados
    """
    # Verificar si el archivo existe y no está vacío
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        logger.warning(f"Archivo no encontrado o vacío: {file_path}. Generando datasets iniciales...")
        try:
            generate_core_datasets()
            logger.info("Datasets iniciales generados exitosamente.")
        except Exception as e:
            logger.error(f"Error generando datasets iniciales: {str(e)}")
            return pd.DataFrame()
    
    # Intentar cargar el archivo
    try:
        # Detectar tamaño para estrategia de lectura
        file_size = os.path.getsize(file_path)
        logger.info(f"Tamaño del archivo: {file_size / (1024 * 1024):.2f} MB")

        if file_size > 50 * 1024 * 1024:  # > 50 MB
            logger.info("Archivo grande, usando lectura por chunks")
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=10000,
                                     on_bad_lines='skip', encoding='utf-8'):
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
    except Exception as e:
        logger.exception(f"Error crítico cargando datos: {str(e)}")
        return pd.DataFrame()


class FourierAnalyzer:
    def __init__(self, sampling_rate: float = 1.0, config: Dict[str, Any] = None):
        """
        Analizador de Fourier optimizado para datos de nonces.

        Args:
            sampling_rate (float): Tasa de muestreo de los datos.
            config (dict): Configuración para rutas de datos.
        """
        self.sampling_rate = sampling_rate
        self.loader = NonceLoader(config=config)
        self.data_path = os.path.join(self.loader.training_dir, "nonce_training_data.csv")
        self.results_dir = os.path.join(self.loader.data_dir, "fourier_results")
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_data(self) -> pd.DataFrame:
        """Carga datos de entrenamiento con manejo robusto de errores."""
        logger.info(f"Cargando datos desde: {self.data_path}")
        return safe_load_data(self.data_path)

    def _safe_column_access(self, df: pd.DataFrame, column_name: str) -> np.ndarray:
        """Acceso seguro a columnas con generación alternativa si es necesario."""
        if column_name in df.columns:
            return df[column_name].values
        else:
            logger.warning(f"Columna '{column_name}' no encontrada, generando datos alternativos")
            # Generar valores sintéticos basados en nonces si la columna no existe
            if 'nonce' in df.columns:
                return df['nonce'].apply(lambda x: float(x) if str(x).isdigit() else 0.0).values
            else:
                # Si no hay nonces, generar datos aleatorios
                return np.random.rand(len(df))

    def apply_rfft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica la Transformada Rápida de Fourier (FFT) optimizada para señales reales.

        Args:
            data (np.ndarray): Datos de entrada.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Frecuencias y amplitudes correspondientes.
        """
        n = len(data)

        # Usar FFT para datos reales (más eficiente)
        yf = rfft(data)

        # Calcular frecuencias positivas
        xf = rfftfreq(n, 1 / self.sampling_rate)

        # Normalizar amplitudes
        amplitudes = np.abs(yf) / n

        return xf, amplitudes

    def plot_spectrum(self, data: np.ndarray, title: str = "Espectro de Frecuencias",
                      save_plot: bool = True):
        """
        Grafica y guarda el espectro de frecuencias de los datos.

        Args:
            data (np.ndarray): Datos de entrada.
            title (str): Título del gráfico.
            save_plot (bool): Guardar la imagen en disco.
        """
        start_time = time.time()
        xf, yf = self.apply_rfft(data)
        calc_time = time.time() - start_time
        logger.info(f"Cálculo FFT completado en {calc_time:.4f}s para {len(data)} puntos")

        plt.figure(figsize=(12, 7))
        plt.plot(xf, yf)
        plt.title(title)
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Amplitud Normalizada")
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.tight_layout()

        if save_plot:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f"spectrum_{timestamp}.png")
            plt.savefig(plot_path, dpi=150)
            logger.info(f"Gráfico guardado en: {plot_path}")

        plt.show()
        return plot_path if save_plot else None

    def extract_dominant_features(self, data: np.ndarray, num_features: int = 5) -> Dict[str, float]:
        """
        Extrae las frecuencias y amplitudes dominantes de los datos.

        Args:
            data (np.ndarray): Datos de entrada.
            num_features (int): Número de características a extraer.

        Returns:
            Dict[str, float]: Diccionario con características dominantes.
        """
        xf, yf = self.apply_rfft(data)

        # Encontrar picos dominantes
        peak_indices = np.argsort(yf)[-num_features:][::-1]

        features = {
            "total_energy": np.sum(yf**2),
            "mean_amplitude": np.mean(yf),
            "max_amplitude": np.max(yf)
        }

        for i, idx in enumerate(peak_indices):
            features[f"freq_{i + 1}"] = xf[idx]
            features[f"amp_{i + 1}"] = yf[idx]

        return features

    def analyze_column(self, column_name: str = "nonce", num_features: int = 5):
        """
        Ejecuta análisis completo para una columna específica.

        Args:
            column_name (str): Nombre de la columna a analizar.
            num_features (int): Número de características dominantes a extraer.
        """
        logger.info(f"Iniciando análisis para columna: '{column_name}'")

        # Paso 1: Cargar datos
        df = self._load_data()
        if df.empty:
            logger.error("No se pudieron cargar datos. Abortando análisis.")
            return None

        # Paso 2: Preparar datos
        data = self._safe_column_access(df, column_name)

        # Paso 3: Generar gráfico
        plot_path = self.plot_spectrum(
            data,
            title=f"Espectro de Frecuencias - Columna '{column_name}'"
        )

        # Paso 4: Extraer características
        features = self.extract_dominant_features(data, num_features)

        # Paso 5: Guardar resultados
        results = {
            "column": column_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_points": len(data),
            "features": features,
            "plot_path": plot_path
        }

        # Guardar resultados en JSON
        results_path = os.path.join(self.results_dir, f"fourier_{column_name}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Resultados guardados en: {results_path}")
        return results


# Ejemplo de uso de la clase FourierAnalyzer
if __name__ == "__main__":
    # Configurar análisis
    analyzer = FourierAnalyzer(sampling_rate=0.1)

    # Columnas comunes para análisis
    columns_to_analyze = ["nonce", "entropy", "zero_density", "pattern_score"]

    # Ejecutar análisis para cada columna
    all_results = {}
    for column in columns_to_analyze:
        result = analyzer.analyze_column(column)
        if result:
            all_results[column] = result["features"]

    # Mostrar resumen
    print("\n" + "=" * 50)
    print("RESUMEN DE ANÁLISIS DE FOURIER")
    print("=" * 50)
    for col, features in all_results.items():
        print(f"\nCaracterísticas dominantes para '{col}':")
        for key, value in features.items():
            if key.startswith("freq") or key.startswith("amp"):
                print(f"- {key}: {value:.6f}")

    print("\nAnálisis completado. Resultados guardados en:", analyzer.results_dir)
