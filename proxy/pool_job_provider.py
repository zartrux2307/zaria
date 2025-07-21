from __future__ import annotations
import multiprocessing.shared_memory as shm
import struct
import threading
import time
import logging
import json
from typing import Optional, Dict

logger = logging.getLogger("proxy.jobprovider")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.propagate = True

# Layout binario (debe coincidir con el lado proxy escritor)
# Offsets:
#   0..83   -> blob (84 bytes, padded with \0)
#   84..91  -> target (8 bytes big endian)
#   92..123 -> seed_hash (32 bytes, hex ascii 32? *ojo*) aquí guardamos 32 bytes raw (o ascii)
#   124..159-> job_id (36 bytes máx) padded
#   160..163-> height (uint32 big endian)
#   164..179-> algo (16 bytes, e.g. "rx/0")
#   180      -> flag (1 byte) 1 = válido
JOB_STRUCT_SIZE = 180
FLAG_OFFSET = JOB_STRUCT_SIZE  # 180
TOTAL_SIZE = JOB_STRUCT_SIZE + 1

DEFAULT_POLL = 0.2


def _pad_bytes(src: bytes, size: int) -> bytes:
    if len(src) > size:
        return src[:size]
    if len(src) < size:
        return src + b"\0" * (size - len(src))
    return src


class SharedMemoryJobProvider:
    """
    Proveedor de jobs vía memoria compartida.
    Si el segmento no existe, lo crea e inicia en modo standalone (sin proxy).
    Permite inyección manual de jobs (update_job) para pruebas offline.
    """

    def __init__(self, prefix: str = "5555", poll_interval: float = DEFAULT_POLL, create_if_missing: bool = True):
        self.prefix = prefix
        self.poll_interval = poll_interval
        self.create_if_missing = create_if_missing
        self.job_shm: Optional[shm.SharedMemory] = None
        self._lock = threading.RLock()
        self._last_job: Optional[Dict] = None
        self._standalone = False
        self._open_or_create()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"jobprov-{prefix}")
        self._thread.start()
        logger.info("SharedMemoryJobProvider started prefix=%s standalone=%s", prefix, self._standalone)

    # -------- SHM Management --------
    def _open_or_create(self):
        name = f"{self.prefix}_job"
        try:
            self.job_shm = shm.SharedMemory(name=name)
            logger.info("Attached to existing SHM job segment name=%s size=%d", name, self.job_shm.size)
        except FileNotFoundError:
            if not self.create_if_missing:
                raise
            self.job_shm = shm.SharedMemory(name=name, create=True, size=TOTAL_SIZE)
            self._standalone = True
            self._init_segment()
            logger.warning("Created new SHM job segment (standalone mode) name=%s size=%d", name, self.job_shm.size)

    def _init_segment(self):
        if not self.job_shm:
            return
        buf = self.job_shm.buf
        for i in range(TOTAL_SIZE):
            buf[i] = 0
        # flag=0 => sin job válido

    # -------- Polling Loop --------
    def _loop(self):
        while True:
            try:
                if not self.job_shm:
                    time.sleep(1.0)
                    continue
                buf = self.job_shm.buf
                flag = buf[FLAG_OFFSET]
                if flag == 1:
                    raw = bytes(buf[:JOB_STRUCT_SIZE])
                    job = self._decode(raw)
                    if job:
                        with self._lock:
                            self._last_job = job
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.debug("JobProvider loop error: %s", e)
                time.sleep(1.0)

    # -------- Decode / Encode --------
    def _decode(self, data: bytes) -> Optional[Dict]:
        if len(data) != JOB_STRUCT_SIZE:
            return None
        try:
            blob = data[0:84].rstrip(b"\0").hex()
            target_be = data[84:92]
            target_int = struct.unpack(">Q", target_be)[0]
            seed_hash_raw = data[92:124].rstrip(b"\0")
            seed_hash = seed_hash_raw.hex()
            job_id = data[124:160].decode('utf-8', errors='ignore').rstrip("\0")
            height = struct.unpack(">I", data[160:164])[0]
            algo = data[164:180].decode('utf-8', errors='ignore').rstrip("\0")
            return {
                "blob_hex": blob,
                "seed_hash": seed_hash,
                "job_id": job_id,
                "height": height,
                "target_qword": target_int,
                "algo": algo
            }
        except Exception as e:
            logger.debug("Decode error: %s", e)
            return None

    def _encode(self, job: Dict) -> bytes:
        blob_hex = job.get("blob_hex", "")
        if len(blob_hex) % 2 != 0:
            blob_hex = blob_hex[:-1]
        blob = bytes.fromhex(blob_hex)[:84]
        blob = _pad_bytes(blob, 84)
        target_qword = int(job.get("target_qword", 0))
        target_be = struct.pack(">Q", target_qword)
        seed_hash_hex = job.get("seed_hash", "")
        if len(seed_hash_hex) % 2 != 0:
            seed_hash_hex = seed_hash_hex[:-1]
        seed_bytes = bytes.fromhex(seed_hash_hex)[:32]
        seed_bytes = _pad_bytes(seed_bytes, 32)
        job_id = _pad_bytes(job.get("job_id", "JOBDUMMY").encode('utf-8'), 36)
        height = struct.pack(">I", int(job.get("height", 0)))
        algo = _pad_bytes(job.get("algo", "rx/0").encode('utf-8'), 16)
        return blob + target_be + seed_bytes + job_id + height + algo  # 180 bytes

    # -------- Public API --------
    def current_job(self) -> Optional[Dict]:
        with self._lock:
            return dict(self._last_job) if self._last_job else None

    def update_job(self, job: Dict) -> None:
        """
        Inyecta / sobreescribe un job (sólo permitido en modo standalone).
        """
        if not self._standalone:
            logger.warning("update_job ignorado: no estamos en modo standalone (segmento externo).")
            return
        if not self.job_shm:
            return
        try:
            payload = self._encode(job)
            if len(payload) != JOB_STRUCT_SIZE:
                raise ValueError("Tamaño codificado incorrecto")
            buf = self.job_shm.buf
            buf[:JOB_STRUCT_SIZE] = payload
            buf[FLAG_OFFSET] = 1  # marcar válido
            with self._lock:
                self._last_job = job
            logger.info("Job inyectado en SHM (standalone) id=%s height=%s", job.get("job_id"), job.get("height"))
        except Exception as e:
            logger.error("Error inyectando job: %s", e)

    def is_standalone(self) -> bool:
        return self._standalone

    def close(self):
        try:
            if self.job_shm:
                self.job_shm.close()
        except Exception:
            pass
