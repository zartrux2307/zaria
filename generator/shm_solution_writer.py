from __future__ import annotations
import multiprocessing.shared_memory as shm
import logging
import struct
import threading
import time
from typing import Optional, Dict, Any
from iazar.proxy.shm_channels import JobChannel
logger = logging.getLogger("generator.solutionwriter")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.propagate = True

DEFAULT_RECORD_SIZE = 160
DEFAULT_CAPACITY = 8192
HEADER_SIZE = 16
MAX_SHM_SIZE = 6 * 1024**3  # 6 GB máximo

def _pad_ascii(s: str, size: int) -> bytes:
    b = s.encode('utf-8')
    if len(b) > size:
        return b[:size]
    if len(b) < size:
        b += b'\0' * (size - len(b))
    return b

class SharedMemorySolutionWriter:
    """
    Segmento de soluciones (nonce, hash, etc). Soporta método thread-safe write_solution y try_submit.
    """
   
    def __init__(self, prefix="5555"):
        self.ch = JobChannel(prefix)
    
    def current_job(self):
        return self.ch.get_job()

    def __init__(self, prefix: str = "5555", create_if_missing: bool = True,
                 record_size: int = DEFAULT_RECORD_SIZE, capacity: int = DEFAULT_CAPACITY):
        self.prefix = prefix
        self.create_if_missing = create_if_missing
        self.record_size = record_size
        self.capacity = capacity
        self.version = 1
        self.solution_shm: Optional[shm.SharedMemory] = None
        self._lock = threading.RLock()
        self._standalone = False
        self._open_or_create()

    def _segment_name(self) -> str:
        return f"{self.prefix}_solution"

    def _total_size(self) -> int:
        return HEADER_SIZE + self.record_size * self.capacity

    def _open_or_create(self):
        name = self._segment_name()
        try:
            self.solution_shm = shm.SharedMemory(name=name)
            logger.info("Attached to existing solution SHM name=%s size=%d", name, self.solution_shm.size)
            self._validate_or_init_header(expect_existing=True)
        except FileNotFoundError:
            if not self.create_if_missing:
                raise
            
            # Calcula tamaño y limita a 6GB máximo
            total_size = self._total_size()
            if total_size > MAX_SHM_SIZE:
                max_capacity = (MAX_SHM_SIZE - HEADER_SIZE) // self.record_size
                if max_capacity < 1:
                    raise ValueError("El tamaño de registro es demasiado grande para caber en 6GB de memoria compartida")
                
                logger.warning(
                    "Capacidad solicitada %d (size=%d) supera límite de 6GB. "
                    "Reduciendo capacidad a %d",
                    self.capacity, total_size, max_capacity
                )
                self.capacity = max_capacity
                total_size = self._total_size()
            
            self.solution_shm = shm.SharedMemory(name=name, create=True, size=total_size)
            self._standalone = True
            logger.warning(
                "Creada nueva solución SHM (standalone) name=%s size=%d records=%d",
                name, total_size, self.capacity
            )
            self._validate_or_init_header(expect_existing=False)

    def _validate_or_init_header(self, expect_existing: bool):
        buf = self.solution_shm.buf  # type: ignore
        if not expect_existing:
            with self._lock:
                # write_index=0, record_size, capacity, version
                buf[0:4] = (0).to_bytes(4, 'little')
                buf[4:8] = self.record_size.to_bytes(4, 'little')
                buf[8:12] = self.capacity.to_bytes(4, 'little')
                buf[12:16] = self.version.to_bytes(4, 'little')
        else:
            existing_rec_size = int.from_bytes(buf[4:8], 'little')
            existing_capacity = int.from_bytes(buf[8:12], 'little')
            if existing_rec_size != self.record_size or existing_capacity != self.capacity:
                logger.warning("Solution SHM metadata mismatch (rec_size=%d/%d capacity=%d/%d)",
                               existing_rec_size, self.record_size, existing_capacity, self.capacity)

    def is_standalone(self) -> bool:
        return self._standalone

    def _next_offset(self) -> int:
        buf = self.solution_shm.buf  # type: ignore
        write_index = int.from_bytes(buf[0:4], 'little')
        slot = write_index % self.capacity
        return HEADER_SIZE + slot * self.record_size

    def _advance_index(self):
        buf = self.solution_shm.buf  # type: ignore
        write_index = int.from_bytes(buf[0:4], 'little')
        write_index += 1
        buf[0:4] = write_index.to_bytes(4, 'little')

    def write_solution(self, nonce: int, hash_bytes: bytes, height: int,
                       job_id: str, seed_hash: str, accepted_flag: int = 0):
        if not self.solution_shm:
            return
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

        with self._lock:
            offset = self._next_offset()
            buf = self.solution_shm.buf  # type: ignore
            
            # Estructura 160 bytes
            buf[offset:offset+4] = (nonce & 0xFFFFFFFF).to_bytes(4, 'little')
            buf[offset+4:offset+36] = hash_bytes
            buf[offset+36:offset+40] = (height & 0xFFFFFFFF).to_bytes(4, 'little')
            buf[offset+40:offset+76] = job_bytes
            buf[offset+76:offset+108] = seed_bytes
            buf[offset+108:offset+112] = (accepted_flag & 0xFFFFFFFF).to_bytes(4, 'little')
            buf[offset+112:offset+160] = b'\0' * 48
            
            self._advance_index()

    def try_submit(self, job_id: str, nonce: int, hash_bytes: bytes, valid: bool = True, height: int = 0, seed_hash: str = "") -> bool:
        """
        Interfaz compatible con nonce_orchestrator.
        Escribe la solución en memoria compartida.
        """
        try:
            accepted_flag = 1 if valid else 2  # 1=aceptada, 2=rechazada
            self.write_solution(nonce, hash_bytes, height, job_id, seed_hash, accepted_flag)
            logger.debug(f"try_submit: job_id={job_id} nonce={nonce} valid={valid} height={height}")
            return True
        except Exception as e:
            logger.error(f"try_submit error: {e}")
            return False

    def close(self):
        try:
            if self.solution_shm:
                self.solution_shm.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
