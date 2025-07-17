import os
import logging
import pandas as pd
from typing import Dict, Any

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ruta absoluta al archivo ia_config.json
CONFIG_PATH = "C:/zarturxia/src/iazar/config/ia_config.json"

def load_config() -> Dict[str, Any]:
    import json
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"[ERROR] No se pudo cargar la configuración IA: {e}")
        return {}

def generate_core_datasets(cfg: Dict[str, Any]):
    try:
        data_paths = cfg["data_paths"]
        nonce_path = data_paths["nonce_training_data"]
        hashes_path = data_paths["nonce_hashes"]
        nonces_exitosos_path = data_paths["nonces_exitosos"]
        winner_blocks_path = data_paths["winner_blocks"]

        df = pd.read_csv(nonce_path)
        df_blocks = pd.read_csv(winner_blocks_path)

        df_merged = pd.merge(df, df_blocks, on="block_height", how="left")
        df_merged.to_csv(nonces_exitosos_path, index=False)
        logging.info(f"✅ Datos guardados correctamente en {nonces_exitosos_path}")

    except Exception as e:
        logging.error(f"❌ Error generando datasets: {e}")

def main():
    logging.info("==================================================")
    logging.info(" INICIANDO GENERACIÓN DE DATOS INICIALES")
    logging.info("==================================================")

    cfg = load_config()
    if not cfg:
        logging.error("❌ No se puede continuar sin configuración cargada.")
        return

    generate_core_datasets(cfg)

if __name__ == "__main__":
    main()
