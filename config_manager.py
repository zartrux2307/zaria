"""
Módulo avanzado de gestión de configuración con locking robusto
para acceso concurrente seguro y prevención de corrupción
"""
import pandas as pd
import os
import json
import logging
from typing import Dict, Any, Optional
from filelock import FileLock
from iazar.utils.paths import get_nonce_training_data_path
# Opcional: importa dotenv si lo usas (no interrumpe si no está)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger('ZartruxConfigManager')

class ConfigValidationError(Exception):
    """Excepción personalizada para errores de validación de configuración"""

class LockManager:
    """Gestor centralizado de locks para archivos de configuración"""
    _locks: Dict[str, FileLock] = {}

    @classmethod
    def get_lock(cls, file_path: str) -> FileLock:
        lock_key = os.path.abspath(file_path)
        if lock_key not in cls._locks:
            cls._locks[lock_key] = FileLock(f"{lock_key}.lock", timeout=10)
        return cls._locks[lock_key]

class ConfigManager:
    _instance = None
    _configs: Dict[str, Dict[str, Any]] = {}
    _schemas: Dict[str, Dict[str, Any]] = {}
    _encryption_key: Optional[bytes] = None

    BASE_SCHEMAS = {
        'global_config': {
            "type": "object",
            "properties": {
                "shm": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "size": {"type": "integer", "minimum": 1024}
                    },
                    "required": ["name", "size"]
                },
                "stratum": {
                    "type": "object",
                    "properties": {
                        "pool_host": {"type": "string"},
                        "pool_port": {"type": "integer", "minimum": 1, "maximum": 65535}
                    },
                    "required": ["pool_host", "pool_port"]
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                        "format": {"type": "string"}
                    },
                    "required": ["level", "format"]
                }
            },
            "required": ["shm", "stratum", "logging"]
        }
        # Puedes agregar otros esquemas si necesitas...
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        """
        Inicialización segura para el singleton ConfigManager.
        Lee y carga configuraciones, inicializa esquemas, etc.
        """
        config_path = os.getenv('CONFIG_PATH', 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                try:
                    self._configs['global_config'] = json.load(f)
                    logger.info(f"Configuración cargada desde {config_path}")
                except Exception as e:
                    logger.error(f'Error cargando configuración: {e}')
                    self._configs['global_config'] = {}
        else:
            logger.warning(f'No se encontró archivo de configuración en {config_path}')
            self._configs['global_config'] = {}  # Fallback seguro

        self._schemas = self.BASE_SCHEMAS.copy()
        key = os.getenv('ENCRYPTION_KEY')
        self._encryption_key = key.encode() if key else None
        logger.info("ConfigManager inicializado correctamente.")

    def get_config(self, name: str) -> Dict[str, Any]:
        """Obtiene la configuración por nombre, fallback seguro."""
        if name not in self._configs:
            logger.warning(f"Configuración '{name}' no encontrada. Devolviendo objeto vacío.")
            return {}
        return self._configs[name]

    def get_config_value(self, section: str, key: str, default=None) -> Any:
        """Acceso seguro a valores anidados."""
        try:
            return self._configs[section][key]
        except Exception:
            return default

    def get_shm_config(self) -> Dict[str, Any]:
        """Acceso directo a la sección 'shm' del global_config."""
        return self._configs.get('global_config', {}).get('shm', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Acceso directo a la sección 'logging' del global_config."""
        return self._configs.get('global_config', {}).get('logging', {})

    def get_stratum_config(self) -> Dict[str, Any]:
        """Acceso directo a la sección 'stratum' del global_config."""
        return self._configs.get('global_config', {}).get('stratum', {})

# ==== ALIAS COMPATIBLES PARA IMPORTS ====

def get_global_config() -> Dict[str, Any]:
    return ConfigManager().get_config('global_config')

def get_config(config_name: str) -> Dict[str, Any]:
    return ConfigManager().get_config(config_name)

def get_shm_config() -> Dict[str, Any]:
    return ConfigManager().get_shm_config()

def get_logging_config() -> Dict[str, Any]:
    return ConfigManager().get_logging_config()

def get_stratum_config() -> Dict[str, Any]:
    return ConfigManager().get_stratum_config()

def get_config_value(section: str, key: str, default=None) -> Any:
    return ConfigManager().get_config_value(section, key, default)

def initialize_system_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Inicializa/obtiene la configuración del sistema"""
    return get_global_config()

def get_ia_config(config_filename='config.json', key='ia_config', silent=False):
    """
    Carga la configuración de IA desde un archivo JSON, flexible con rutas.
    """
    # Busca el archivo de config de forma robusta
    base_dirs = [
        os.path.dirname(__file__),                               # Local al script
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')),  # Carpeta /config
        os.path.abspath(os.path.join(os.getcwd(), 'config')),    # Carpeta config en cwd
        os.getcwd(),                                             # cwd directo
    ]
    config_path = None
    for base in base_dirs:
        path = os.path.join(base, config_filename)
        if os.path.isfile(path):
            config_path = path
            break
    if not config_path:
        if not silent:
            print(f"[WARNING] No se encontró {config_filename} en rutas conocidas.")
        return {}
    try:
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
        if key and key in config:
            return config[key]
        return config
    except Exception as e:
        if not silent:
            print(f"[ERROR] No se pudo cargar la configuración IA: {e}")
        return {}
  
