import os
import csv
import struct
import logging

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
CSV_NONCES = os.path.join(DATA_DIR, 'nonces_exitosos.csv')
BIN_NONCES = os.path.join(DATA_DIR, 'nonce_hashes.bin')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

def main():
    if not os.path.isfile(CSV_NONCES):
        logging.error(f"No se encuentra el archivo CSV de nonces: {CSV_NONCES}")
        return

    count = 0
    with open(CSV_NONCES, 'r', newline='', encoding='utf-8') as csvfile, \
         open(BIN_NONCES, 'wb') as binfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                nonce = int(row['nonce'])
                # Guardar cada nonce como uint32 little-endian
                binfile.write(struct.pack('<I', nonce))
                count += 1
            except Exception as e:
                logging.warning(f"Fila inválida o sin nonce válido: {row} ({e})")

    if count:
        logging.info(f"✔️ Generado {BIN_NONCES} con {count} nonces.")
    else:
        logging.error("❌ Ningún nonce válido extraído. ¡Revisa el CSV de entrada!")

if __name__ == "__main__":
    main()
