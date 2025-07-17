# File: generate_missing_files.py
import pandas as pd
import os

DATA_DIR = r"C:\zarturxia\src\iazar\data"

# Crear winner_blocks.csv si no existe
if not os.path.exists(os.path.join(DATA_DIR, "winner_blocks.csv")):
    pd.DataFrame({
        'block_height': [1045678, 1045679],
        'nonce': ['a1b2c3d4', 'e5f6g7h8'],
        'timestamp': [1689200000, 1689200500]
    }).to_csv(os.path.join(DATA_DIR, "winner_blocks.csv"), index=False)

# Crear nonce_training_data.csv base
if not os.path.exists(os.path.join(DATA_DIR, "nonce_training_data.csv")):
    pd.DataFrame({
        'block_data': ['0000000000', '1111111111'],
        'nonce': ['00000000', 'FFFFFFFF']
    }).to_csv(os.path.join(DATA_DIR, "nonce_training_data.csv"), index=False)

print("✅ Critical files generated in data directory")
