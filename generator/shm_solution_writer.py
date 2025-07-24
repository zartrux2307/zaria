import zmq
import logging
import struct
import threading
from typing import Optional

logger = logging.getLogger("generator.solutionwriter")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.propagate = True

DEFAULT_RECORD_SIZE = 160

def _pad_ascii(s: str, size: int) -> bytes:
    b = s.encode('utf-8')
    if len(b) > size:
        return b[:size]
    if len(b) < size:
        b += b'\0' * (size - len(b))
    return b

class ZmqSolutionWriter:
    """Escritor de soluciones usando ZeroMQ PUB-SUB"""
   
    def __init__(self, zmq_pub_address: str, record_size: int = DEFAULT_RECORD_SIZE):
        self.record_size = record_size
        self.zmq_pub_address = zmq_pub_address
        self._lock = threading.RLock()
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self._connect()

    def _connect(self):
        """Establece conexión ZMQ"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.zmq_pub_address)
        logger.info("ZMQ PUB socket conectado en %s", self.zmq_pub_address)

    def write_solution(self, nonce: int, hash_bytes: bytes, height: int,
                      job_id: str, seed_hash: str, accepted_flag: int = 0):
        if not self.socket:
            return

        # Normalizar hash (igual que en la implementación original)
        if len(hash_bytes) != 32:
            if isinstance(hash_bytes, (bytes, bytearray)):
                if len(hash_bytes) > 32:
                    hash_bytes = hash_bytes[:32]
                else:
                    hash_bytes = hash_bytes.ljust(32, b'\0')
            else:
                try:
                    hb = bytes.fromhex(str(hash_bytes))
                    hash_bytes = hb[:32].ljust(32, b'\0')
                except Exception:
                    hash_bytes = b'\0' * 32

        seed_bytes = bytes.fromhex(seed_hash)[:32] if seed_hash else b'\0' * 32
        seed_bytes = seed_bytes.ljust(32, b'\0')
        job_bytes = _pad_ascii(job_id or "", 36)

        # Construir mensaje binario (160 bytes)
        record = bytearray(self.record_size)
        record[0:4] = struct.pack('<I', nonce & 0xFFFFFFFF)
        record[4:36] = hash_bytes
        record[36:40] = struct.pack('<I', height & 0xFFFFFFFF)
        record[40:76] = job_bytes
        record[76:108] = seed_bytes
        record[108:112] = struct.pack('<I', accepted_flag & 0xFFFFFFFF)
        # Resto del buffer permanece como padding

        with self._lock:
            try:
                self.socket.send(bytes(record))
            except zmq.ZMQError as e:
                logger.error("Error enviando solución via ZMQ: %s", e)
                # Reconexión automática en caso de error
                self.close()
                self._connect()
                self.socket.send(bytes(record))

    def try_submit(self, job_id: str, nonce: int, hash_bytes: bytes, 
                  valid: bool = True, height: int = 0, seed_hash: str = "") -> bool:
        """Mantiene la misma interfaz que la implementación original"""
        try:
            accepted_flag = 1 if valid else 2
            self.write_solution(nonce, hash_bytes, height, job_id, seed_hash, accepted_flag)
            logger.debug("try_submit: job_id=%s nonce=%s valid=%s height=%s", 
                        job_id, nonce, valid, height)
            return True
        except Exception as e:
            logger.error("try_submit error: %s", e)
            return False

    def close(self):
        """Cierra conexiones ZMQ de forma segura"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.socket = None
        self.context = None
        logger.info("Conexión ZMQ cerrada")

    def __del__(self):
        self.close()