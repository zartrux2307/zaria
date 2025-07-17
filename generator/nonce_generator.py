
import csv
import os
import json
import threading
from abc import ABC, abstractmethod
from iazar.generator.config_loader import config_loader
class BaseNonceGenerator(ABC):
    """
    Clase base abstracta para generadores de nonces.
    Proporciona funcionalidad común para todos los generadores:
    - Carga de configuración centralizada
    - Acceso a datos de entrenamiento
    - Escritura segura de resultados
    - Formato estandarizado de salida
    """
    
    DEFAULT_DATA_PATHS = {
        "training_data": "C:/zarturxia/src/iazar/data/nonce_training_data.csv",
        "generated_nonces": "C:/zarturxia/src/iazar/data/nonces_exitosos.csv",
        "accepted_nonces": "C:/zarturxia/src/iazar/data/nonces_aceptados.csv"
    }
    
    FIELDNAMES = [
        "nonce", "entropy", "uniqueness", "zero_density", 
        "pattern_score", "is_valid", "block_height", "generator"
    ]
    
    def __init__(self, generator_name: str, config: dict = None):  # Cambiar parámetro
        self.generator_name = generator_name
        self.config = config or config_loader.load_config()  
        self.training_data = self.load_training_data()
        self.lock = threading.Lock()
        
    def load_training_data(self) -> list:
        """Carga datos históricos de entrenamiento con valores por defecto"""
        # Obtener rutas con valores por defecto
        data_paths = self.config.get('data_paths', {})
        training_path = data_paths.get(
            'training_data', 
            self.DEFAULT_DATA_PATHS['training_data']
        )
        
        if not os.path.exists(training_path):
            return []
            
        try:
            with open(training_path, 'r') as f:
                reader = csv.DictReader(f)
                return [row for row in reader if row.get('nonce')]
        except Exception as e:
            print(f"Error cargando datos de entrenamiento: {e}")
            return []
    
    @abstractmethod
    def generate_nonce(self, block_height: int) -> dict:
        """
        Método abstracto que debe implementar cada generador.
        Devuelve un diccionario con el formato:
        {
            "nonce": int,
            "entropy": float,
            "uniqueness": float,
            "zero_density": float,
            "pattern_score": float,
            "is_valid": bool,
            "block_height": int,
            "generator": str
        }
        """
        pass
    
    def save_nonce(self, nonce_data: dict):
        """
        Guarda un nonce en el archivo CSV de forma segura
        con bloqueo para acceso concurrente
        """
        # Obtener rutas con valores por defecto
        data_paths = self.config.get('data_paths', {})
        output_path = data_paths.get(
            'generated_nonces', 
            self.DEFAULT_DATA_PATHS['generated_nonces']
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with self.lock:
            write_header = not os.path.exists(output_path)
            with open(output_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                if write_header:
                    writer.writeheader()
                writer.writerow(nonce_data)
    
    def calculate_metrics(self, nonce: int) -> dict:
        """
        Calcula métricas básicas para un nonce (implementación base)
        Los generadores especializados deben sobreescribir este método
        """
        # Implementación simplificada - generadores reales deben implementar su propia lógica
        return {
            "entropy": random.uniform(0.7, 0.95),
            "uniqueness": random.uniform(0.8, 1.0),
            "zero_density": random.uniform(0.05, 0.2),
            "pattern_score": random.uniform(0.9, 1.0),
            "is_valid": True
        }
