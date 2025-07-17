import os
import json
import logging

logger = logging.getLogger("ConfigLoader")

def load_config(path: str = None) -> dict:
    """
    Carga el archivo JSON de configuración y devuelve un dict.
    Si el archivo no existe, lo crea con valores predeterminados.

    :param path: Ruta al archivo de configuración. Si no se especifica,
                 usa '<project_root>/config/config.json'.
    :return: Diccionario con la configuración.
    """
    # Determinar directorio raíz del proyecto (tres niveles arriba)
    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
    )
    # Ruta por defecto al config.json
    default_path = os.path.join(base_dir, "config", "config.json")
    config_path = path or default_path

    # Asegurar que el directorio exista
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Si no existe el archivo, generarlo con valores por defecto
    if not os.path.exists(config_path):
        default_cfg = get_default_config(base_dir)
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_cfg, f, indent=4)
            logger.info(f"Archivo de configuración creado: {config_path}")
        except Exception as e:
            logger.error(f"Error creando config por defecto: {e}")
        return default_cfg

    # Leer configuración existente
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        # Normalizar rutas relativas en data_paths
        data_paths = cfg.get('data_paths', {})
        for key, val in data_paths.items():
            if val and not os.path.isabs(val):
                data_paths[key] = os.path.join(base_dir, val)
        return cfg
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        return get_default_config(base_dir)


def get_default_config(base_dir: str) -> dict:
    """
    Devuelve un dict con la configuración por defecto.
    """
    data_dir = os.path.join(base_dir, "data")
    return {
        "prefix": "5555",
        "evaluation_interval": 300,
        "data_paths": {
            "training_data": os.path.join(data_dir, "nonce_training_data.csv"),
            "generated_nonces": os.path.join(data_dir, "nonces_exitosos.csv"),
            "accepted_nonces": os.path.join(data_dir, "nonces_aceptados.csv"),
            "reports": os.path.join(data_dir, "reports")
        },
        "hybrid_params": {"weights": [0.5, 0.5]},
        "adaptive_params": {"initial_parameters": {}},
        "sequence_params": {"step": 1},
        "generator_weights": {"adaptativo": 0.7, "hibrido": 0.3},
        "performance_settings": {"batch_size": 500, "max_nonce": 2**64 - 1}
    }
# Alias para compatibilidad con imports existentes
config_loader = load_config