import os
from iazar.utils.config_manager import ConfigManager
from iazar.generator.config_loader import config_loader
def setup_paths():
    config = ConfigManager().get_config('global_config')
    paths_config = config.get('paths', {})
    
    BASE_DIR = paths_config.get('data_dir', os.path.expanduser("~/iazar/data"))
    CONFIG_DIR = paths_config.get('config_dir', os.path.expanduser("~/iazar/config"))
    MODEL_DIR = paths_config.get('model_dir', os.path.expanduser("~/iazar/models"))
    os.makedirs("C:/zarturxia/src/iazar/data", exist_ok=True)
    os.makedirs("C:/zarturxia/src/iazar/models", exist_ok=True)
    os.makedirs("C:/zarturxia/src/iazar/config", exist_ok=True)
    # Crear directorios si no existen
    for path in [BASE_DIR, CONFIG_DIR, MODEL_DIR]:
        os.makedirs(path, exist_ok=True)
    
    return BASE_DIR, CONFIG_DIR, MODEL_DIR

BASE_DIR, CONFIG_DIR, MODEL_DIR = setup_paths()

def get_nonce_training_data_path():
    return os.path.join(BASE_DIR, "nonce_training_data.csv")

def get_model_path(filename):
    return os.path.join(MODEL_DIR, filename)