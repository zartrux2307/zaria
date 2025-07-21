from __future__ import annotations
"""
shm_channels.py
---------------
Canales de Memoria Compartida (Shared Memory) multi-pool para intercambio
de *jobs* y *solutions* entre Orchestrator (IA) y Proxy / Miners.

Mejoras implementadas:
- Compatibilidad multiplataforma mejorada (Windows/Linux)
- Capacidad aumentada para alta frecuencia (500+ nonces/seg)
- Auto-expansión controlada de buffers
- Validación reforzada de parámetros
- Registro de errores detallado
- Métodos utilitarios para gestión de soluciones
- Tratamiento seguro de strings
- Límites de tamaño para prevenir sobrecarga
"""

import os
import re
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
class SHMSizeError(SHMError): pass

# ---------------- Utilidades ----------------
def _ensure_shared_memory(name: str, size: int):
    # Limpieza de nombre para compatibilidad con Windows
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Límite de tamaño para prevenir sobrecarga
    MAX_SIZE = 1024 * 1024 * 128  # 128MB
    if size > MAX_SIZE:
        raise SHMSizeError(f"Tamaño de SHM excede límite máximo ({MAX_SIZE} bytes)")
    
    if _HAS_SHARED_MEMORY:
        try:
            return shared_memory.SharedMemory(name=safe_name, create=True, size=size)
        except FileExistsError:
            return shared_memory.SharedMemory(name=safe_name, create=False, size=size)
        except Exception as e:
            logger.error("Fallo SharedMemory directa name=%s size=%d err=%s, fallback a archivo.", 
                         safe_name, size, e)
    
    # Fallback file mapping (cross-platform)
    if os.name == "nt":
        # Configuración mejorada para Windows
        temp_dir = os.environ.get("TEMP", "C:\\zarturxia\\tmp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        path = os.path.join(temp_dir, f"{safe_name}.shm")
    else:
        path = f"/tmp/{safe_name}.shm"
    
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
            try: 
                self.buf.flush()
            except Exception: 
                pass
            try: 
                self.buf.close()
            except Exception: 
                pass
            try: 
                os.close(self._fd)
            except Exception: 
                pass
        
        def unlink(self):
            try: 
                os.remove(path)
            except Exception: 
                pass
    
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
    DEFAULT_SIZE = 8192  # Aumentado para trabajos más grandes

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
                logger.info("Inicializando nuevo segmento JobChannel")
                self._mv[:4] = self.MAGIC
                struct.pack_into("<H", self._mv, 4, self.VERSION)
                struct.pack_into("<H", self._mv, 6, 0)
                struct.pack_into("<I", self._mv, 8, 0)  # job_version
                for off in (12, 16, 20, 24, 28, 32, 36):
                    struct.pack_into("<I", self._mv, off, 0)

    def _resize(self, new_size: int):
        """Expande el segmento de memoria compartida"""
        logger.warning("Redimensionando JobChannel de %d a %d bytes", self.size, new_size)
        with self._lock:
            # Crear nuevo segmento
            new_shm = _ensure_shared_memory(self.name, new_size)
            new_mv = memoryview(new_shm.buf)
            
            # Copiar datos existentes
            new_mv[:self.size] = self._mv[:self.size]
            
            # Actualizar referencias
            self._shm.close()
            self._shm = new_shm
            self._mv = new_mv
            self.size = new_size

    def close(self):
        self._shm.close()

    def unlink(self):
        try: 
            self._shm.unlink()
        except Exception as e: 
            logger.error("Error al eliminar SHM: %s", e)

    def set_job(self, job: Dict[str, Any]) -> int:
        # Conversión segura de blob
        blob = job.get("blob", b"")
        if isinstance(blob, str):
            try:
                # Intentar decodificar como hex si es posible
                if len(blob) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in blob.strip()):
                    blob = bytes.fromhex(blob.strip())
                else:
                    blob = blob.encode("utf-8")
            except Exception as e:
                logger.error("Error convirtiendo blob: %s", e)
                blob = blob.encode("utf-8")

        # Conversión segura de seed
        seed = job.get("seed", b"")
        if isinstance(seed, str):
            try:
                if len(seed) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in seed):
                    seed = bytes.fromhex(seed)
                else:
                    seed = seed.encode("utf-8")
            except Exception as e:
                logger.error("Error convirtiendo seed: %s", e)
                seed = seed.encode("utf-8")

        # Tratamiento seguro de job_id
        job_id = str(job.get("job_id", "none"))[:64]
        
        # Conversión segura de target
        target = job.get("target", b"")
        if isinstance(target, int):
            target = target.to_bytes(32, "little", signed=False)
        elif isinstance(target, str):
            t = target.strip()
            if t.startswith("0x"): 
                t = t[2:]
            if len(t) % 2 == 1:
                t = "0" + t
            try:
                raw = bytes.fromhex(t)
                target = raw.ljust(32, b"\x00")[:32] if len(raw) < 32 else raw[:32]
            except Exception:
                target = t.encode("utf-8")
        elif isinstance(target, bytes):
            target = target.ljust(32, b"\x00")[:32] if len(target) < 32 else target[:32]
        else:
            target = b"\x00" * 32

        # Campos numéricos
        height = int(job.get("height", 0))
        nonce_offset = int(job.get("nonce_offset", 0))
        extra_nonce_size = int(job.get("extra_nonce_size", 0))

        with self._lock:
            cur_ver = struct.unpack_from("<I", self._mv, 8)[0]
            new_ver = (cur_ver + 1) & 0xFFFFFFFF

            # Calcular tamaño necesario
            blob_size = len(blob)
            seed_size = len(seed)
            job_id_b = job_id.encode("utf-8")
            job_id_size = len(job_id_b)
            target_size = len(target)
            needed = self.HEADER_SIZE + blob_size + seed_size + job_id_size + target_size

            # Auto-expansión si es necesario
            if needed > self.size:
                new_size = max(self.size * 2, needed + 1024)
                self._resize(new_size)

            # Escribir encabezado
            struct.pack_into("<I", self._mv, 12, blob_size)
            struct.pack_into("<I", self._mv, 16, seed_size)
            struct.pack_into("<I", self._mv, 20, job_id_size)
            struct.pack_into("<I", self._mv, 24, target_size)
            struct.pack_into("<I", self._mv, 28, height)
            struct.pack_into("<I", self._mv, 32, nonce_offset)
            struct.pack_into("<I", self._mv, 36, extra_nonce_size)

            # Escribir datos
            offset = self.HEADER_SIZE
            self._mv[offset:offset+blob_size] = blob
            offset += blob_size
            self._mv[offset:offset+seed_size] = seed
            offset += seed_size
            self._mv[offset:offset+job_id_size] = job_id_b
            offset += job_id_size
            self._mv[offset:offset+target_size] = target

            # Actualizar versión
            struct.pack_into("<I", self._mv, 8, new_ver)
            logger.debug("Nuevo trabajo establecido: job_id=%s, version=%d", job_id, new_ver)
            return new_ver

    def get_job(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._mv[:4].tobytes() != self.MAGIC:
                logger.warning("Segmento JobChannel no inicializado")
                return None
            
            try:
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
                    raise SHMConsistencyError(f"Overflow detectado: end_target={end_target}, size={self.size}")

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
            except Exception as e:
                logger.exception("Error al leer trabajo: %s", e)
                return None

# ---------------- Solution Channel ----------------
class SolutionChannel:
    MAGIC = b"RSOL"
    VERSION = 1
    DEFAULT_CAPACITY = 32768  # Capacidad aumentada para alta frecuencia
    DEFAULT_ITEM_SIZE = 96    # Espacio adicional para metadata
    HEADER_SIZE = 24           # bytes

    def __init__(self, prefix: str, capacity: int = DEFAULT_CAPACITY, item_size: int = DEFAULT_ITEM_SIZE):
        if item_size < 80:
            raise ValueError("item_size debe ser >= 80 bytes")
        
        self.prefix = prefix
        self.capacity = int(capacity)
        self.item_size = int(item_size)
        total_size = self.HEADER_SIZE + self.capacity * self.item_size
        
        # Prevenir sobrecarga de memoria
        MAX_SIZE = 1024 * 1024 * 64  # 64MB
        if total_size > MAX_SIZE:
            raise MemoryError(f"Tamaño de canal excede límite máximo ({MAX_SIZE} bytes)")
        
        self.name = f"{prefix}_sol"
        self._shm = _ensure_shared_memory(self.name, total_size)
        self._mv = memoryview(self._shm.buf)
        self._lock = threading.RLock()
        self._init_if_empty()

    def _init_if_empty(self):
        with self._lock:
            if self._mv[:4].tobytes() != self.MAGIC:
                logger.info("Inicializando nuevo SolutionChannel")
                self._mv[:4] = self.MAGIC
                struct.pack_into("<H", self._mv, 4, self.VERSION)
                struct.pack_into("<H", self._mv, 6, 0)  # reserved
                struct.pack_into("<I", self._mv, 8, self.capacity)
                struct.pack_into("<I", self._mv, 12, 0)  # write_index
                struct.pack_into("<I", self._mv, 16, 0)  # total_submitted
                struct.pack_into("<I", self._mv, 20, self.item_size)

    def close(self):
        self._shm.close()

    def unlink(self):
        try: 
            self._shm.unlink()
        except Exception as e: 
            logger.error("Error al eliminar SHM: %s", e)

    def try_submit(self, job_version: int, nonce: int, hash_bytes: bytes, 
                  valid: bool, job_id: str, pool_id: str = "default") -> bool:
        # Validación de entrada
        if not isinstance(hash_bytes, (bytes, bytearray)) or len(hash_bytes) != 32:
            logger.error("Longitud de hash_bytes inválida: %d, esperaba 32", len(hash_bytes))
            return False
        
        # Seguridad: truncar identificadores
        job_id = str(job_id)[:32]
        pool_id = str(pool_id)[:16]
        
        # Preparar datos
        jb = job_id.encode("utf-8")[:16]
        pb = pool_id.encode("utf-8")[:16]
        jb = _null_terminated_bytes(jb, 16)
        pb = _null_terminated_bytes(pb, 16)
        flags = 0x01 if valid else 0x00

        with self._lock:
            try:
                # Leer metadatos
                capacity = struct.unpack_from("<I", self._mv, 8)[0]
                write_index = struct.unpack_from("<I", self._mv, 12)[0]
                total_submitted = struct.unpack_from("<I", self._mv, 16)[0]
                item_size = struct.unpack_from("<I", self._mv, 20)[0]

                if capacity != self.capacity or item_size != self.item_size:
                    raise SHMConsistencyError("Solution header inconsistente (tam cambiado).")

                # Calcular posición
                pos = self.HEADER_SIZE + (write_index % capacity) * item_size
                end = pos + item_size
                if end > len(self._mv):
                    raise SHMConsistencyError("Overflow en ring buffer")

                # Escribir campos
                struct.pack_into("<I", self._mv, pos, job_version & 0xFFFFFFFF)
                struct.pack_into("<I", self._mv, pos + 4, nonce & 0xFFFFFFFF)
                self._mv[pos + 8: pos + 40] = hash_bytes
                struct.pack_into("B", self._mv, pos + 40, flags)
                self._mv[pos + 41: pos + 48] = b"\x00" * 7  # padding
                self._mv[pos + 48: pos + 64] = jb  # job_id
                self._mv[pos + 64: pos + 80] = pb  # pool_id
                
                # Rellenar espacio adicional si es necesario
                if item_size > 80:
                    self._mv[pos + 80: end] = b"\x00" * (item_size - 80)

                # Actualizar índices
                new_index = (write_index + 1) & 0xFFFFFFFF
                total_submitted = (total_submitted + 1) & 0xFFFFFFFF
                struct.pack_into("<I", self._mv, 12, new_index)
                struct.pack_into("<I", self._mv, 16, total_submitted)
                
                logger.debug("Solución enviada: job_id=%s, nonce=%d, válido=%s", 
                            job_id, nonce, valid)
                return True
            except Exception as e:
                logger.exception("Error al enviar solución: %s", e)
                return False

    def stats(self) -> Dict[str, int]:
        with self._lock:
            try:
                return {
                    "capacity": struct.unpack_from("<I", self._mv, 8)[0],
                    "write_index": struct.unpack_from("<I", self._mv, 12)[0],
                    "total_submitted": struct.unpack_from("<I", self._mv, 16)[0],
                    "item_size": struct.unpack_from("<I", self._mv, 20)[0]
                }
            except Exception as e:
                logger.error("Error al leer estadísticas: %s", e)
                return {
                    "capacity": 0,
                    "write_index": 0,
                    "total_submitted": 0,
                    "item_size": 0
                }

    def clear_solutions(self):
        """Reinicia el buffer de soluciones"""
        with self._lock:
            try:
                struct.pack_into("<I", self._mv, 12, 0)  # write_index
                struct.pack_into("<I", self._mv, 16, 0)  # total_submitted
                logger.info("Buffer de soluciones reiniciado")
            except Exception as e:
                logger.error("Error al reiniciar soluciones: %s", e)

    def get_all_solutions(self) -> List[Dict[str, Any]]:
        """Recupera todas las soluciones válidas del buffer"""
        with self._lock:
            try:
                stats = self.stats()
                solutions = []
                for i in range(stats['capacity']):
                    pos = self.HEADER_SIZE + i * stats['item_size']
                    entry = self._parse_entry(pos, stats['item_size'])
                    if entry['job_version'] > 0:  # Entrada válida
                        solutions.append(entry)
                return solutions
            except Exception as e:
                logger.error("Error al obtener soluciones: %s", e)
                return []

    def tail(self, max_items: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            try:
                st = self.stats()
                wi = st["write_index"]
                cap = st["capacity"]
                size = st["item_size"]
                out: List[Dict[str, Any]] = []
                
                # Calcular número de elementos a recuperar
                n = min(max_items, cap)
                if wi < cap:
                    n = min(n, wi)
                
                # Recuperar elementos más recientes
                for i in range(n):
                    index = (wi - 1 - i) % cap
                    pos = self.HEADER_SIZE + index * size
                    entry = self._parse_entry(pos, size)
                    out.append(entry)
                return out
            except Exception as e:
                logger.error("Error al obtener tail: %s", e)
                return []

    def _parse_entry(self, pos: int, size: int) -> Dict[str, Any]:
        try:
            job_version = struct.unpack_from("<I", self._mv, pos)[0]
            nonce = struct.unpack_from("<I", self._mv, pos + 4)[0]
            hash_bytes = bytes(self._mv[pos + 8: pos + 40])
            flags = self._mv[pos + 40]
            
            # Decodificar identificadores con manejo de nulos
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
        except Exception as e:
            logger.error("Error al analizar entrada: %s", e)
            return {
                "job_version": 0,
                "nonce": 0,
                "hash": "",
                "valid": False,
                "job_id": "error",
                "pool_id": "error"
            }

# ---------------- Helpers Factory ----------------
def open_job_channel(prefix: str, size: int = JobChannel.DEFAULT_SIZE) -> JobChannel:
    return JobChannel(prefix=prefix, size=size)

def open_solution_channel(prefix: str,
                          capacity: int = SolutionChannel.DEFAULT_CAPACITY,
                          item_size: int = SolutionChannel.DEFAULT_ITEM_SIZE) -> SolutionChannel:
    return SolutionChannel(prefix=prefix, capacity=capacity, item_size=item_size)