import lmdb
import struct
import csv
import os
import sys
import time
import signal
import hashlib
from typing import Dict, Optional, List, Tuple
from tqdm.auto import tqdm
from datetime import datetime
from collections import deque
import concurrent.futures

# Importa la función para calcular características del nonce
from iazar.utils.feature_utils import calc_nonce_features

# Ruta destino (formato final requerido por el sistema IA)
CSV_OUTPUT_PATH = r"C:/zarturxia/src/iazar/data/nonce_training_data.csv"

CONFIG = {
    'lmdb_path': r"E:/monero-blockchain/lmdb",
    'csv_output': CSV_OUTPUT_PATH,
    'max_blocks': 1580000,
    'update_interval': 3000,
    'max_retries': 5,
    'nonce_offsets': {
        1: 43, 2: 47, 3: 51, 4: 55, 'default': 43
    },
    'hash_window': 1000,
    'batch_size': 1000
}

# Cabecera exacta requerida por create_initial_data.py
NONCE_COLUMNS = [
     "nonce","entropy","uniquenes","zero_density","pattern_score","is_valid"
]

class NonceExtractor:
    def __init__(self):
        self.processed_hashes = deque(maxlen=CONFIG['hash_window'])
        self.running = True
        self.processed_nonces = set()
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        self._load_existing_nonces()

    def _load_existing_nonces(self):
        if os.path.exists(CONFIG['csv_output']):
            try:
                with open(CONFIG['csv_output'], 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.processed_nonces.add(row.get("nonce"))
                print(f"[Info] Cargados {len(self.processed_nonces)} nonces existentes")
            except Exception as e:
                print(f"[Error] No se pudieron cargar nonces existentes: {str(e)}")

    def graceful_shutdown(self, signum, frame):
        print(f"\n[Info] Señal {signum} recibida. Terminando extracción...")
        self.running = False

    def write_csv(self, entries: list):
        try:
            os.makedirs(os.path.dirname(CONFIG['csv_output']), exist_ok=True)
            file_exists = os.path.exists(CONFIG['csv_output'])
            mode = 'a' if file_exists else 'w'

            new_entries = []
            for e in entries:
                nonce_str = str(e.get("nonce", ""))
                if nonce_str and nonce_str not in self.processed_nonces:
                    new_entries.append(e)
                    self.processed_nonces.add(nonce_str)

            if not new_entries:
                print("[Info] No hay nuevos registros para guardar.")
                return

            with open(CONFIG['csv_output'], mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=NONCE_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                for e in new_entries:
                    writer.writerow({k: e[k] for k in NONCE_COLUMNS})

            print(f"[Éxito] {len(new_entries)} registros escritos en {CONFIG['csv_output']}")

        except Exception as e:
            print(f"[Error CSV] {str(e)}")

    def _block_hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def parse_block(self, data: bytes) -> Tuple[Optional[int], Optional[int]]:
        try:
            major_version = struct.unpack('<B', data[0:1])[0]
            offset = CONFIG['nonce_offsets'].get(major_version, CONFIG['nonce_offsets']['default'])
            nonce_value = struct.unpack('<I', data[offset:offset + 4])[0]

            block_height = struct.unpack('<Q', data[1:9])[0]
            return nonce_value, block_height
        except Exception as e:
            print(f"[Error] parse_block: {e}")
            return None, None

    def process_block_batch(self, blocks: List[Tuple[bytes, str]]) -> List[Dict]:
        batch_entries = []
        for data, block_hash in blocks:
            if block_hash in self.processed_hashes:
                continue

            nonce_value, block_height = self.parse_block(data)
            if nonce_value is None or block_height is None:
                continue

            features = calc_nonce_features(nonce_value)
            features["block_height"] = block_height
            batch_entries.append(features)
            self.processed_hashes.append(block_hash)

        return batch_entries

    def process_blocks(self, cursor) -> list:
        new_entries = []
        cursor.last()
        block_count = 0
        retries = 0
        blocks_to_process = []

        with tqdm(total=CONFIG['max_blocks'], desc="Recolectando bloques") as pbar:
            while self.running and block_count < CONFIG['max_blocks'] and retries < CONFIG['max_retries']:
                try:
                    data = cursor.value()
                    block_hash = self._block_hash(data)

                    if block_hash not in self.processed_hashes:
                        blocks_to_process.append((data, block_hash))
                        block_count += 1
                        pbar.update(1)

                    if not cursor.prev():
                        break
                except Exception as e:
                    print(f"[Error LMDB] {str(e)}")
                    retries += 1
                    time.sleep(2 ** retries)

        batch_size = CONFIG['batch_size']
        total_batches = (len(blocks_to_process) + batch_size - 1) // batch_size

        with tqdm(total=len(blocks_to_process), desc="Procesando bloques") as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.process_block_batch, blocks_to_process[i:i + batch_size])
                    for i in range(0, len(blocks_to_process), batch_size)
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_entries = future.result()
                        new_entries.extend(batch_entries)
                        pbar.update(len(batch_entries))
                    except Exception as e:
                        print(f"[Error en lote] {str(e)}")

        return new_entries

    def run_extraction(self):
        try:
            env = lmdb.open(
                CONFIG['lmdb_path'],
                max_dbs=1,
                readonly=True,
                lock=False,
                metasync=False,
                readahead=False
            )
            with env.begin(db=env.open_db(b'blocks'), buffers=True) as txn:
                new_entries = self.process_blocks(txn.cursor())
                if new_entries:
                    self.write_csv(new_entries)
        except Exception as e:
            print(f"[Error extracción] {str(e)}")
        finally:
            if 'env' in locals():
                env.close()

    def main_loop(self):
        while self.running:
            print(f"\n[{datetime.now()}] Iniciando extracción de nonces...")
            self.run_extraction()
            print(f"Esperando {CONFIG['update_interval']}s antes del siguiente ciclo...\n")
            time.sleep(CONFIG['update_interval'])


if __name__ == "__main__":
    NonceExtractor().main_loop()
