from __future__ import annotations
"""
shm_channels.py
---------------
Canales de Memoria Compartida (Shared Memory) multi-pool para intercambio
de *jobs* y *solutions* entre Orchestrator (IA) y Proxy / Miners.

Estructuras:
- JobChannel: segmento fijo (default 4096 bytes)
- SolutionChannel: buffer circular (ring) compuesto por entradas de tamaño fijo.
"""

import os
import sys
import mmap
import struct
import logging
import threading
from typing import Optional, List, Dict, Any

try:
    from multiprocessing import shared_memory
    _HAS_SHARED_MEMORY = True
except ImportError:
    _HAS_SHARED_MEMORY = False

logger = logging.getLogger("shm.channels")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------- Excepciones ----------------
class SHMError(Exception): pass
class SHMFormatError(SHMError): pass
class SHMConsistencyError(SHMError): pass

# ---------------- Utilidades ----------------
def _ensure_shared_memory(name: str, size: int):
    if _HAS_SHARED_MEMORY:
        try:
            return shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            return shared_memory.SharedMemory(name=name, create=False, size=size)
        except Exception as e:
            logger.error("Fallo SharedMemory directa name=%s size=%d err=%s, fallback a archivo.", name, size, e)
    # Fallback file mapping (cross-platform)
    path = f"/tmp/{name}.shm" if os.name != "nt" else os.path.join(os.environ.get("TEMP", "."), f"{name}.shm")
    existed = os.path.exists(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    if not existed:
        os.ftruncate(fd, size)
    else:
        current = os.stat(path).st_size
        if current < size:
            os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size)
    class _Shim:
        def __init__(self, mm, fd, name, size):
            self.buf = mm
            self._fd = fd
            self.name = name
            self.size = size
        def close(self):
            try: self.buf.flush()
            except Exception: pass
            try: self.buf.close()
            except Exception: pass
            try: os.close(self._fd)
            except Exception: pass
        def unlink(self):
            try: os.remove(path)
            except Exception: pass
    return _Shim(mm, fd, name, size)

def _null_terminated_bytes(src: bytes, max_len: int) -> bytes:
    if len(src) >= max_len:
        return src[:max_len]
    return src + b"\x00" * (max_len - len(src))

# ---------------- Job Channel ----------------
class JobChannel:
    MAGIC = b"RJOB"
    VERSION = 1
    HEADER_SIZE = 40  # hasta extra_nonce_size
    DEFAULT_SIZE = 4096

    def __init__(self, prefix: str, size: int = DEFAULT_SIZE):
        self.prefix = prefix
        self.size = size
        self.name = f"{prefix}_job"
        self._shm = _ensure_shared_memory(self.name, size)
        self._lock = threading.RLock()
        self._mv = memoryview(self._shm.buf)
        self._init_if_empty()

    def _init_if_empty(self):
        with self._lock:
            if self._mv[:4].tobytes() != self.MAGIC:
                self._mv[:4] = self.MAGIC
                struct.pack_into("<H", self._mv, 4, self.VERSION)
                struct.pack_into("<H", self._mv, 6, 0)
                struct.pack_into("<I", self._mv, 8, 0)
                for off in (12,16,20,24,28,32,36):
                    struct.pack_into("<I", self._mv, off, 0)

    def close(self):
        self._shm.close()

    def unlink(self):
        try: self._shm.unlink()
        except Exception: pass

    def set_job(self, job: Dict[str, Any]) -> int:
        blob = job.get("blob", b"")
        if isinstance(blob, str):
            if all(c in "0123456789abcdefABCDEF" for c in blob.strip()):
                try: blob = bytes.fromhex(blob.strip())
                except Exception: blob = blob.encode("utf-8")
            else:
                blob = blob.encode("utf-8")
        seed = job.get("seed", b"")
        if isinstance(seed, str):
            if len(seed) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in seed):
                try: seed = bytes.fromhex(seed)
                except Exception: seed = seed.encode("utf-8")
            else:
                seed = seed.encode("utf-8")

        job_id = str(job.get("job_id", "none"))[:64]
        target = job.get("target", b"")
        if isinstance(target, int):
            target = target.to_bytes(32, "little", signed=False)
        elif isinstance(target, str):
            t = target.strip()
            if t.startswith("0x"): t = t[2:]
            if len(t) % 2 == 1:
                t = "0" + t
            try:
                raw = bytes.fromhex(t)
                if len(raw) < 32:
                    target = raw + b"\x00" * (32 - len(raw))
                else:
                    target = raw[:32]
            except Exception:
                target = t.encode("utf-8")
        elif isinstance(target, bytes):
            if len(target) < 32:
                target = target + b"\x00" * (32 - len(target))
            else:
                target = target[:32]
        else:
            target = b"\x00" * 32

        height = int(job.get("height", 0))
        nonce_offset = int(job.get("nonce_offset", 0))
        extra_nonce_size = int(job.get("extra_nonce_size", 0))

        with self._lock:
            cur_ver = struct.unpack_from("<I", self._mv, 8)[0]
            new_ver = (cur_ver + 1) & 0xFFFFFFFF

            blob_size = len(blob)
            seed_size = len(seed)
            job_id_b = job_id.encode("utf-8")
            job_id_size = len(job_id_b)
            target_size = len(target)

            needed = self.HEADER_SIZE + blob_size + seed_size + job_id_size + target_size
            if needed > self.size:
                raise SHMFormatError(f"Job payload too large needed={needed} size={self.size}")

            struct.pack_into("<I", self._mv, 12, blob_size)
            struct.pack_into("<I", self._mv, 16, seed_size)
            struct.pack_into("<I", self._mv, 20, job_id_size)
            struct.pack_into("<I", self._mv, 24, target_size)
            struct.pack_into("<I", self._mv, 28, height)
            struct.pack_into("<I", self._mv, 32, nonce_offset)
            struct.pack_into("<I", self._mv, 36, extra_nonce_size)

            offset = self.HEADER_SIZE
            self._mv[offset:offset+blob_size] = blob
            offset += blob_size
            self._mv[offset:offset+seed_size] = seed
            offset += seed_size
            self._mv[offset:offset+job_id_size] = job_id_b
            offset += job_id_size
            self._mv[offset:offset+target_size] = target

            struct.pack_into("<I", self._mv, 8, new_ver)
            return new_ver

    def get_job(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._mv[:4].tobytes() != self.MAGIC:
                return None
            version = struct.unpack_from("<I", self._mv, 8)[0]
            blob_size = struct.unpack_from("<I", self._mv, 12)[0]
            seed_size = struct.unpack_from("<I", self._mv, 16)[0]
            job_id_size = struct.unpack_from("<I", self._mv, 20)[0]
            target_size = struct.unpack_from("<I", self._mv, 24)[0]
            height = struct.unpack_from("<I", self._mv, 28)[0]
            nonce_offset = struct.unpack_from("<I", self._mv, 32)[0]
            extra_nonce_size = struct.unpack_from("<I", self._mv, 36)[0]

            offset = self.HEADER_SIZE
            end_blob = offset + blob_size
            end_seed = end_blob + seed_size
            end_jobid = end_seed + job_id_size
            end_target = end_jobid + target_size

            if end_target > self.size:
                raise SHMConsistencyError("Corrupt job segment (overflow)")

            blob = bytes(self._mv[offset:end_blob])
            seed = bytes(self._mv[end_blob:end_seed])
            job_id = self._mv[end_seed:end_jobid].tobytes().decode("utf-8", errors="replace")
            target = bytes(self._mv[end_jobid:end_target])

            return {
                "job_version": version,
                "blob": blob,
                "seed": seed,
                "job_id": job_id,
                "target": target,
                "height": height,
                "nonce_offset": nonce_offset,
                "extra_nonce_size": extra_nonce_size
            }

# ---------------- Solution Channel ----------------
class SolutionChannel:
    MAGIC = b"RSOL"
    VERSION = 1
    DEFAULT_CAPACITY = 16384       # número de entradas
    DEFAULT_ITEM_SIZE = 80         # bytes por entrada
    HEADER_SIZE = 24               # bytes

    def __init__(self, prefix: str, capacity: int = DEFAULT_CAPACITY, item_size: int = DEFAULT_ITEM_SIZE):
        if item_size < 64:
            raise ValueError("item_size demasiado pequeño")
        self.prefix = prefix
        self.capacity = int(capacity)
        self.item_size = int(item_size)
        total_size = self.HEADER_SIZE + self.capacity * self.item_size
        self.name = f"{prefix}_sol"
        self._shm = _ensure_shared_memory(self.name, total_size)
        self._mv = memoryview(self._shm.buf)
        self._lock = threading.RLock()
        self._init_if_empty()

    def _init_if_empty(self):
        with self._lock:
            if self._mv[:4].tobytes() != self.MAGIC:
                self._mv[:4] = self.MAGIC
                struct.pack_into("<H", self._mv, 4, self.VERSION)
                struct.pack_into("<H", self._mv, 6, 0)
                struct.pack_into("<I", self._mv, 8, self.capacity)
                struct.pack_into("<I", self._mv, 12, 0)  # write_index
                struct.pack_into("<I", self._mv, 16, 0)  # total_submitted
                struct.pack_into("<I", self._mv, 20, self.item_size)

    def close(self):
        self._shm.close()

    def unlink(self):
        try: self._shm.unlink()
        except Exception: pass

    def try_submit(self, job_version: int, nonce: int, hash_bytes: bytes, valid: bool, job_id: str, pool_id: str = "default") -> bool:
        if not isinstance(hash_bytes, (bytes, bytearray)) or len(hash_bytes) != 32:
            return False
        jb = job_id.encode("utf-8")[:16]
        pb = pool_id.encode("utf-8")[:16]
        jb = _null_terminated_bytes(jb, 16)
        pb = _null_terminated_bytes(pb, 16)
        flags = 0x01 if valid else 0x00

        with self._lock:
            capacity = struct.unpack_from("<I", self._mv, 8)[0]
            write_index = struct.unpack_from("<I", self._mv, 12)[0]
            total_submitted = struct.unpack_from("<I", self._mv, 16)[0]
            item_size = struct.unpack_from("<I", self._mv, 20)[0]

            if capacity != self.capacity or item_size != self.item_size:
                raise SHMConsistencyError("Solution header inconsistente (tam cambiado).")

            pos = self.HEADER_SIZE + (write_index % capacity) * item_size
            end = pos + item_size
            if end > len(self._mv):
                raise SHMConsistencyError("Overflow calculando posición ring buffer.")

            # Escribir campos
            struct.pack_into("<I", self._mv, pos, job_version & 0xFFFFFFFF)
            struct.pack_into("<I", self._mv, pos + 4, nonce & 0xFFFFFFFF)
            self._mv[pos + 8: pos + 40] = hash_bytes
            struct.pack_into("B", self._mv, pos + 40, flags)
            self._mv[pos + 41: pos + 48] = b"\x00" * 7
            self._mv[pos + 48: pos + 64] = jb
            self._mv[pos + 64: pos + 80] = pb
            if item_size > 80:
                self._mv[pos + 80: end] = b"\x00" * (item_size - 80)

            # Actualizar índices
            write_index = (write_index + 1) & 0xFFFFFFFF
            total_submitted = (total_submitted + 1) & 0xFFFFFFFF
            struct.pack_into("<I", self._mv, 12, write_index)
            struct.pack_into("<I", self._mv, 16, total_submitted)
            return True

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "capacity": struct.unpack_from("<I", self._mv, 8)[0],
                "write_index": struct.unpack_from("<I", self._mv, 12)[0],
                "total_submitted": struct.unpack_from("<I", self._mv, 16)[0],
                "item_size": struct.unpack_from("<I", self._mv, 20)[0]
            }

    def tail(self, max_items: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            st = self.stats()
            wi = st["write_index"]
            cap = st["capacity"]
            size = st["item_size"]
            out: List[Dict[str, Any]] = []
            n = min(max_items, cap, wi if wi < cap else cap)
            for i in range(n):
                index = (wi - 1 - i) % cap
                pos = self.HEADER_SIZE + index * size
                entry = self._parse_entry(pos, size)
                out.append(entry)
            return out

    def _parse_entry(self, pos: int, size: int) -> Dict[str, Any]:
        job_version = struct.unpack_from("<I", self._mv, pos)[0]
        nonce = struct.unpack_from("<I", self._mv, pos + 4)[0]
        hash_bytes = bytes(self._mv[pos + 8: pos + 40])
        flags = self._mv[pos + 40]
        job_id_raw = bytes(self._mv[pos + 48: pos + 64])
        pool_id_raw = bytes(self._mv[pos + 64: pos + 80])
        job_id = job_id_raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        pool_id = pool_id_raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        return {
            "job_version": job_version,
            "nonce": nonce,
            "hash": hash_bytes.hex(),
            "valid": bool(flags & 0x01),
            "job_id": job_id,
            "pool_id": pool_id
        }

# ---------------- Helpers Factory ----------------
def open_job_channel(prefix: str, size: int = JobChannel.DEFAULT_SIZE) -> JobChannel:
    return JobChannel(prefix=prefix, size=size)

def open_solution_channel(prefix: str,
                          capacity: int = SolutionChannel.DEFAULT_CAPACITY,
                          item_size: int = SolutionChannel.DEFAULT_ITEM_SIZE) -> SolutionChannel:
    return SolutionChannel(prefix=prefix, capacity=capacity, item_size=item_size)
