import os
import sys
import logging
import pandas as pd
from pathlib import Path
from iazar.utils.config_manager import get_ia_config
import json
from iazar.utils.feature_utils import guardar_nonces_csv, COLUMNS

from iazar.utils.paths import get_nonce_training_data_path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)
# Columnas estándar globales
COLUMNS = ["nonce", "entropy", "uniqueness", "zero_density", "pattern_score", "is_valid"]


def leer_nonces_csv(path):
    """Lee un CSV de nonces y garantiza estructura/cabecera estándar."""
    if not os.path.exists(path):
        pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    missing = [col for col in COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = 0
    df = df[COLUMNS]
    df = df.dropna()  # Opcional, borra filas incompletas
    return df


def guardar_nonces_csv(df, path):
    """Guarda un DataFrame de nonces con la cabecera y orden estándar."""
    if not set(COLUMNS).issubset(df.columns):
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = 0
    df = df[COLUMNS]
    df.to_csv(path, index=False)


def leer_nonces_json(path):
    """Lee un JSON de nonces como lista de dicts."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump([], f)
        return []
    with open(path, 'r') as f:
        data = json.load(f)
    # Completa campos faltantes
    for item in data:
        for col in COLUMNS:
            if col not in item:
                item[col] = 0
    return data


def guardar_nonces_json(lista, path):
    """Guarda una lista de dicts como JSON de nonces."""
    with open(path, 'w') as f:
        json.dump(lista, f, indent=2)

# Utilidades para blobs binarios


def hexstr_to_bytes(blob_hex):
    return bytes.fromhex(blob_hex) if isinstance(blob_hex, str) else blob_hex


def bytes_to_hexstr(blob_bytes):
    return blob_bytes.hex() if isinstance(blob_bytes, (bytes, bytearray)) else blob_bytes

# Ejemplo de uso:
# df = leer_nonces_csv("ruta.csv")
# guardar_nonces_csv(df, "nueva_ruta.csv")
# nonces = leer_nonces_json("ruta.json")
# guardar_nonces_json(nonces, "nueva_ruta.json")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataCollection")


def main():
    config = get_ia_config()
    data_path = Path(config['data_paths']['successful_nonces'])
    data_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"📥 Generando datos de ejemplo en: {data_path}")

    # Datos de ejemplo - en una aplicación real aquí se recolectarían datos reales
    data = {
        'block_number': [1000000, 1000001, 1000002],
        'miner': ['miner1', 'miner2', 'miner3'],
        'nonce': [123456, 654321, 987654],
        'difficulty': [1000000, 1000001, 1000002],
        'timestamp': [1625097600, 1625097601, 1625097602]
    }

    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    logger.info(f"✅ Datos generados: {len(df)} registros guardados")


if __name__ == "__main__":
    main()
