import os

# --- Carpeta base del proyecto, ajusta aquí si cambias la raíz ---
BASE_DIR = r"C:\zarturxia\src\iazar\data"
CONFIG_DIR = r"C:\zarturxia\src\iazar\config"

def get_nonce_training_data_path():
    return os.path.join(BASE_DIR, "nonce_training_data.csv")

def get_nonce_training_data_backup_path():
    return os.path.join(BASE_DIR, "nonce_training_data.backup.csv")

def get_nonces_exitosos_path():
    return os.path.join(BASE_DIR, "nonces_exitosos.csv")

def get_nonce_hashes_path():
    return os.path.join(BASE_DIR, "nonce_hashes.bin")

def get_nonce_preprocessed_path():
    return os.path.join(BASE_DIR, "nonce_preprocessed.csv")

def get_winner_blocks_path():
    return os.path.join(BASE_DIR, "winner_blocks.csv")

def get_inyectados_log_path():
    return os.path.join(BASE_DIR, "inyectados.log")

def get_ia_config_path():
    return os.path.join(CONFIG_DIR, "ia_config.json")

def get_global_config_path():
    return os.path.join(CONFIG_DIR, "global_config.json")

def get_config_path(filename):
    """Acceso flexible para cualquier archivo config"""
    return os.path.join(CONFIG_DIR, filename)
