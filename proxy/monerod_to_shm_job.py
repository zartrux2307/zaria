import requests
import time
import sys
import os
import json
import logging
import uuid
import traceback
import struct

# Configuración de rutas para compatibilidad con Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, '..', 'proxy'))

# Importación segura de módulos
try:
    from iazar.proxy.shm_channels import open_job_channel
except ImportError:
    from shm_channels import open_job_channel

# Configuración modificada para compatibilidad con Windows
MONEROD_RPC_URL = "http://127.0.0.1:18081/json_rpc"
WALLET_ADDRESS = "44crWF5Y7gWDLCwhNSH7cbAbCPT6xScpCRFMMYhbCpFijJVUpPwze39GbvRRR1GsRZCvNMKZpU4sPT8bqRY3FY29Loyx1zc"

# Configuración mejorada de logging
log_dir = os.path.join(os.environ.get("TEMP", "C:\\zarturxia\\tmp"), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "monerod_job_injector.log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(module)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("monerod_to_shm_job")

def get_block_template():
    """Obtiene plantilla de bloque de monerod con manejo robusto de errores"""
    payload = {
        "jsonrpc": "2.0",
        "id": "0",
        "method": "get_block_template",
        "params": {
            "wallet_address": WALLET_ADDRESS,
            "reserve_size": 16
        }
    }
    
    try:
        resp = requests.post(MONEROD_RPC_URL, json=payload, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        
        if "error" in j:
            error_msg = j["error"].get("message", "Error desconocido")
            error_code = j["error"].get("code", -1)
            raise RuntimeError(f"monerod RPC error [{error_code}]: {error_msg}")
        
        result = j.get("result", {})
        required_keys = ["blocktemplate_blob", "difficulty", "height", "reserved_offset"]
        
        if not all(key in result for key in required_keys):
            missing = [k for k in required_keys if k not in result]
            raise RuntimeError(f"Respuesta incompleta. Faltan: {missing}")
            
        return {
            "blob": result["blocktemplate_blob"],
            "seed_hash": result.get("seed_hash", "00" * 32),
            "job_id": str(uuid.uuid5(uuid.NAMESPACE_OID, 
                     f"{result['height']}_{result['difficulty']}_{result['blocktemplate_blob'][:32]}")),
            "difficulty": int(result["difficulty"]),
            "height": int(result["height"]),
            "nonce_offset": int(result.get("reserved_offset", 39))
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión con monerod: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error procesando respuesta: {str(e)}")
        logger.debug("Respuesta completa: %s", json.dumps(j, indent=2) if 'j' in locals() else "No response")
        raise

def difficulty_to_target(difficulty: int):
    """Convierte dificultad a target de 32 bytes (little-endian)"""
    if difficulty <= 0:
        return b"\xff" * 32
    
    try:
        # Cálculo seguro para grandes números
        target_value = (1 << 256) // difficulty
        
        # Manejo de overflow
        if target_value > (1 << 256) - 1:
            return b"\xff" * 32
            
        return target_value.to_bytes(32, "little")
    except Exception as e:
        logger.error(f"Error convirtiendo dificultad: {e}")
        return b"\xff" * 32

def main():
    prefix = "5555"  # Prefijo fijo para todo el sistema
    logger.info("Iniciando inyector de trabajos para SHM")
    logger.info("Configuración:")
    logger.info("  Prefijo SHM: %s", prefix)
    logger.info("  URL monerod: %s", MONEROD_RPC_URL)
    logger.info("  Wallet: %s", WALLET_ADDRESS)
    logger.info("  Log file: %s", log_file)

    job_channel = None
    last_job_id = None
    consecutive_errors = 0
    max_errors = 10

    while True:
        try:
            # Reconectar SHM si es necesario
            if job_channel is None:
                try:
                    job_channel = open_job_channel(prefix, size=8192)  # Tamaño aumentado
                    logger.info("Canal SHM inicializado correctamente")
                except Exception as e:
                    logger.critical("Error inicializando canal SHM: %s", e)
                    time.sleep(5)
                    continue
            
            # Obtener plantilla de bloque
            tpl = get_block_template()
            
            # Solo procesar si es un trabajo nuevo
            if tpl["job_id"] != last_job_id:
                target_bytes = difficulty_to_target(tpl["difficulty"])
                
                # Crear trabajo en formato compatible
                job_dict = {
                    "blob": tpl["blob"],
                    "seed": tpl["seed_hash"],
                    "job_id": tpl["job_id"],
                    "target": target_bytes,
                    "height": tpl["height"],
                    "nonce_offset": tpl["nonce_offset"],
                    "extra_nonce_size": 4  # Campo requerido por el sistema
                }
                
                # Enviar trabajo al canal SHM
                version = job_channel.set_job(job_dict)
                logger.info(
                    "Trabajo inyectado: altura=%d diff=%d job_id=%s version=%d",
                    tpl["height"], tpl["difficulty"], tpl["job_id"], version
                )
                last_job_id = tpl["job_id"]
                consecutive_errors = 0
            
            time.sleep(1.5)
        
        except KeyboardInterrupt:
            logger.info("Detenido por usuario")
            break
        except requests.exceptions.ConnectionError:
            consecutive_errors += 1
            logger.error("Monerod no responde. Reintento %d/%d", consecutive_errors, max_errors)
            time.sleep(min(5 * consecutive_errors, 30))
        except Exception as e:
            consecutive_errors += 1
            logger.error("Error crítico: %s", str(e))
            logger.debug("Traceback: %s", traceback.format_exc())
            
            if consecutive_errors >= max_errors:
                logger.critical("Demasiados errores consecutivos. Reiniciando canal SHM...")
                try:
                    if job_channel:
                        job_channel.close()
                        job_channel.unlink()
                except:
                    pass
                job_channel = None
                consecutive_errors = 0
            
            time.sleep(5)
    
    # Limpieza final
    if job_channel:
        try:
            job_channel.close()
            job_channel.unlink()
        except Exception as e:
            logger.warning("Error limpiando SHM: %s", e)

if __name__ == "__main__":
    main()