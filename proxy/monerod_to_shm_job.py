import requests
import time
import sys
import os
import json
import logging
import uuid
import binascii

# Asegura el path al módulo shm_channels
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'proxy')))
from iazar.proxy.shm_channels import JobChannel

MONEROD_RPC_URL = "http://127.0.0.1:18081/json_rpc"
WALLET_ADDRESS = "44crWF5Y7gWDLCwhNSH7cbAbCPT6xScpCRFMMYhbCpFijJVUpPwze39GbvRRR1GsRZCvNMKZpU4sPT8bqRY3FY29Loyx1zc"
POLL_INTERVAL = 2.0  # segundos
PREFIX = "5555"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("monerod_to_shm_job")

def get_block_template():
    payload = {
        "jsonrpc": "2.0",
        "id": "0",
        "method": "get_block_template",
        "params": {
            "wallet_address": WALLET_ADDRESS,
            "reserve_size": 16
        }
    }
    resp = requests.post(MONEROD_RPC_URL, json=payload, timeout=10)
    resp.raise_for_status()
    j = resp.json()
    if "error" in j:
        raise RuntimeError(f"monerod RPC error: {j['error']}")
    result = j.get("result")
    if not result or not all(k in result for k in ["blocktemplate_blob", "difficulty", "height", "reserved_offset"]):
        raise RuntimeError(f"Respuesta inesperada de monerod: {j}")
    return {
        "blob": result["blocktemplate_blob"],
        "seed_hash": result.get("seed_hash", "00" * 32),
        "job_id": str(uuid.uuid5(uuid.NAMESPACE_OID, f"{result['height']}_{result['difficulty']}_{result['blocktemplate_blob'][:32]}")),
        "difficulty": int(result["difficulty"]),
        "height": int(result["height"]),
        "nonce_offset": int(result.get("reserved_offset", 39))
    }

def difficulty_to_target(difficulty: int):
    if difficulty <= 0:
        return b"\xff" * 32
    t = (1 << 256) // difficulty
    return t.to_bytes(32, "little")

def main():
    job_channel = JobChannel(PREFIX)
    last_job_id = None
    logger.info("Conectando a monerod local...")

    while True:
        try:
            tpl = get_block_template()
            if tpl["job_id"] != last_job_id:
                target_bytes = difficulty_to_target(tpl["difficulty"])
                job_dict = {
    "blob": tpl["blob"],               # Cambia "blob_hex" por "blob"
    "target": target_bytes,            # Cambia "target_qword" por "target" y usa los bytes completos
    "height": tpl["height"],
    "job_id": tpl["job_id"],
    "seed": tpl["seed_hash"],          # Cambia "seed_hash" por "seed"
    "nonce_offset": tpl["nonce_offset"]
}
                job_channel.set_job(job_dict)
                logger.info(f"Nuevo job injectado: altura={tpl['height']} job_id={tpl['job_id']} offset={tpl['nonce_offset']}")
                last_job_id = tpl["job_id"]
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.error(f"Error al obtener template: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
