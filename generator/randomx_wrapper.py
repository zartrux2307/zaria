from __future__ import annotations
"""
randomx_wrapper.py
------------------
Wrapper robusto para la librería RandomX (Monero PoW) con:
 - Carga dinámica configurable (dll_path).
 - Pool reutilizable de VMs por seed (cache).
 - Flags RandomX configurables (JIT, LARGE_PAGES, HARD_AES, FULL_MEM, SECURE).
 - Thread-local binding de VM (evita contención).
 - Funciones utilitarias: hash simple, hash modificando nonce in-place, verificación de target.
 - Limpieza idempotente y manejo de errores sin sys.exit().
"""

import os
import ctypes
import threading
import logging
import atexit
from typing import Dict, List, Optional, Union

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("randomx.wrapper")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------
# Constantes Flags (subset común)
# ---------------------------------------------------------------------
RANDOMX_FLAG_DEFAULT      = 0
RANDOMX_FLAG_LARGE_PAGES  = 1 << 0
RANDOMX_FLAG_HARD_AES     = 1 << 1
RANDOMX_FLAG_FULL_MEM     = 1 << 2
RANDOMX_FLAG_JIT          = 1 << 3
RANDOMX_FLAG_SECURE       = 1 << 4   # Puede no estar soportado en tu build

_FLAG_NAME_MAP = {
    "LARGE_PAGES": RANDOMX_FLAG_LARGE_PAGES,
    "HARD_AES": RANDOMX_FLAG_HARD_AES,
    "FULL_MEM": RANDOMX_FLAG_FULL_MEM,
    "JIT": RANDOMX_FLAG_JIT,
    "SECURE": RANDOMX_FLAG_SECURE,
    "DEFAULT": RANDOMX_FLAG_DEFAULT
}

# ---------------------------------------------------------------------
# Excepciones específicas
# ---------------------------------------------------------------------
class RandomXError(Exception):
    pass

class RandomXFunctionMissing(RandomXError):
    pass

class RandomXSeedError(RandomXError):
    pass

# ---------------------------------------------------------------------
# Carga de librería
# ---------------------------------------------------------------------
def _load_library(dll_path: str) -> ctypes.WinDLL:
    if not os.path.exists(dll_path):
        raise RandomXError(f"No se encontró la DLL RandomX en: {dll_path}")
    try:
        os.environ['PATH'] = os.path.dirname(dll_path) + os.pathsep + os.environ.get('PATH', '')
        lib = ctypes.WinDLL(dll_path)
        logger.info("RandomX DLL cargada: %s", dll_path)
        return lib
    except Exception as e:
        raise RandomXError(f"Fallo al cargar DLL: {e}") from e

# ---------------------------------------------------------------------
# Funciones nativas requeridas
# ---------------------------------------------------------------------
_REQUIRED_FUNCS = {
    'randomx_alloc_cache': (ctypes.c_void_p, [ctypes.c_uint]),
    'randomx_init_cache': (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]),
    'randomx_create_vm': (ctypes.c_void_p, [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]),
    'randomx_calculate_hash': (None, [ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_ubyte),
                                      ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_ubyte)]),
    'randomx_destroy_vm': (None, [ctypes.c_void_p]),
    'randomx_release_cache': (None, [ctypes.c_void_p]),
}

# Opcional: si tu build soporta dataset
_OPTIONAL_FUNCS = {
    'randomx_alloc_dataset': (ctypes.c_void_p, [ctypes.c_uint]),
    'randomx_init_dataset': (None, [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_ulonglong, ctypes.c_ulonglong]),
    'randomx_release_dataset': (None, [ctypes.c_void_p]),
}

def _bind_functions(lib: ctypes.WinDLL):
    for name, (restype, argtypes) in {**_REQUIRED_FUNCS, **_OPTIONAL_FUNCS}.items():
        try:
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
        except AttributeError:
            if name in _REQUIRED_FUNCS:
                raise RandomXFunctionMissing(f"Función requerida no encontrada en DLL: {name}")
            else:
                logger.debug("Función opcional no presente: %s", name)

# ---------------------------------------------------------------------
# VM Manager
# ---------------------------------------------------------------------
class RandomXVMManager:
    """
    Gestiona caches y pools de VM por seed.
    - Cada seed (32 bytes hex) tiene un cache y lista de VMs disponibles.
    - Thread-local: cada hilo obtiene una VM y la retiene mientras la seed no cambie.
    - Limita el tamaño del pool por seed (configurable).
    """
    def __init__(self, lib: ctypes.WinDLL, flags: int, max_vms_per_seed: int = 8):
        self.lib = lib
        self.flags = flags
        self.max_vms_per_seed = max(1, max_vms_per_seed)
        self._caches: Dict[str, ctypes.c_void_p] = {}
        self._vm_pool: Dict[str, List[ctypes.c_void_p]] = {}
        self._lock = threading.Lock()
        self._local = threading.local()
        self._closed = False
        atexit.register(self.cleanup)

    def get_vm(self, seed_hex: str) -> ctypes.c_void_p:
        """
        Devuelve una VM lista para calcular hashes con la seed indicada.
        Mantiene binding por hilo mientras no cambie la seed.
        """
        if self._closed:
            raise RandomXError("RandomXVMManager cerrado.")

        if len(seed_hex) != 64:
            raise RandomXSeedError("Seed debe ser hex de 64 chars (32 bytes).")

        # Reusar VM si ya asociada en el hilo
        if getattr(self._local, "current_seed", None) == seed_hex and getattr(self._local, "vm_ptr", None):
            return self._local.vm_ptr

        # Si hay otra VM previa en el hilo -> devolverla al pool
        if getattr(self._local, "vm_ptr", None) and getattr(self._local, "current_seed", None):
            self._release_vm(self._local.current_seed, self._local.vm_ptr)
            self._local.vm_ptr = None
            self._local.current_seed = None

        vm_ptr = self._acquire_vm(seed_hex)
        self._local.vm_ptr = vm_ptr
        self._local.current_seed = seed_hex
        return vm_ptr

    # Internos --------------------------------------------------------
    def _acquire_vm(self, seed_hex: str) -> ctypes.c_void_p:
        with self._lock:
            if seed_hex not in self._caches:
                self._init_seed_resources(seed_hex)
            pool = self._vm_pool[seed_hex]
            if pool:
                vm = pool.pop()
                logger.debug("VM reutilizada seed=%s pool=%d", seed_hex[:8], len(pool))
                # METRIC: randomx_vm_reuse.inc()
                return vm
        # Crear fuera del lock (coste mayor)
        vm = self._create_vm(seed_hex)
        logger.debug("VM nueva seed=%s", seed_hex[:8])
        # METRIC: randomx_vm_created.inc()
        return vm

    def _init_seed_resources(self, seed_hex: str):
        cache = self.lib.randomx_alloc_cache(self.flags)
        if not cache:
            raise RandomXError("randomx_alloc_cache devolvió NULL")

        seed_bytes = bytes.fromhex(seed_hex)
        if len(seed_bytes) != 32:
            raise RandomXSeedError("Seed debe tener 32 bytes")

        seed_buf = (ctypes.c_ubyte * 32)(*seed_bytes)
        self.lib.randomx_init_cache(cache, seed_buf, 32)

        self._caches[seed_hex] = cache
        self._vm_pool[seed_hex] = []
        logger.info("Cache inicializada seed=%s", seed_hex[:8])

    def _create_vm(self, seed_hex: str) -> ctypes.c_void_p:
        cache = self._caches[seed_hex]
        vm = self.lib.randomx_create_vm(self.flags, cache, None)
        if not vm:
            raise RandomXError("randomx_create_vm devolvió NULL")
        return vm

    def _release_vm(self, seed_hex: str, vm_ptr: ctypes.c_void_p):
        with self._lock:
            pool = self._vm_pool.get(seed_hex)
            if pool is None:
                # Seed ya limpiada; destruir VM directamente
                try:
                    self.lib.randomx_destroy_vm(vm_ptr)
                except Exception:
                    pass
                return
            if len(pool) < self.max_vms_per_seed:
                pool.append(vm_ptr)
            else:
                try:
                    self.lib.randomx_destroy_vm(vm_ptr)
                except Exception:
                    pass

    def cleanup(self):
        if self._closed:
            return
        self._closed = True
        with self._lock:
            for seed, pool in self._vm_pool.items():
                for vm in pool:
                    try:
                        self.lib.randomx_destroy_vm(vm)
                    except Exception:
                        pass
                logger.debug("VMs destruidas seed=%s count=%d", seed[:8], len(pool))
            for seed, cache in self._caches.items():
                try:
                    self.lib.randomx_release_cache(cache)
                except Exception:
                    pass
                logger.debug("Cache liberada seed=%s", seed[:8])
            self._vm_pool.clear()
            self._caches.clear()
        logger.info("RandomXVMManager cleanup completo")

# ---------------------------------------------------------------------
# Estado global (inicializable on-demand)
# ---------------------------------------------------------------------
_global_lib: Optional[ctypes.WinDLL] = None
_global_manager: Optional[RandomXVMManager] = None
_global_flags: int = RANDOMX_FLAG_DEFAULT

def init_randomx(dll_path: Optional[str] = None,
                 flags: Union[int, List[str], None] = None,
                 max_vms_per_seed: int = 8) -> RandomXVMManager:
    """
    Inicializa (idempotente) el wrapper global.
    flags: int o lista de strings (ej: ["JIT","HARD_AES"])
    """
    global _global_lib, _global_manager, _global_flags
    if _global_manager is not None:
        return _global_manager

    dll_path = dll_path or os.environ.get("RANDOMX_DLL_PATH") or r"C:\zarturxia\src\libs\randomx.dll"

    if isinstance(flags, list):
        mask = 0
        for name in flags:
            upper = name.strip().upper()
            if upper not in _FLAG_NAME_MAP:
                raise RandomXError(f"Flag desconocido: {name}")
            mask |= _FLAG_NAME_MAP[upper]
        flags_int = mask
    elif isinstance(flags, int):
        flags_int = flags
    else:
        # Por defecto usar JIT y HARD_AES si están disponibles
        flags_int = RANDOMX_FLAG_JIT | RANDOMX_FLAG_HARD_AES

    _global_lib = _load_library(dll_path)
    _bind_functions(_global_lib)
    _global_flags = flags_int
    _global_manager = RandomXVMManager(_global_lib, flags_int, max_vms_per_seed=max_vms_per_seed)
    logger.info("RandomX inicializado flags=0x%x", flags_int)
    return _global_manager

# ---------------------------------------------------------------------
# Funciones de hash
# ---------------------------------------------------------------------
def compute_randomx_hash(blob: bytes, seed_hex: str) -> bytes:
    """
    Calcula hash RandomX de un blob completo (incluye nonce dentro).
    Retorna 32 bytes (little-endian).
    """
    if _global_manager is None:
        init_randomx()
    if len(seed_hex) != 64:
        raise RandomXSeedError("Seed inválida (64 hex chars esperados)")
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 76:
        raise ValueError("Blob inválido (esperado >=76 bytes para Monero)")
    vm = _global_manager.get_vm(seed_hex)
    in_buf = (ctypes.c_ubyte * len(blob))(*blob)
    out_buf = (ctypes.c_ubyte * 32)()
    _global_lib.randomx_calculate_hash(vm, in_buf, len(blob), out_buf)
    return bytes(out_buf)

def compute_hash_nonce(blob: Union[bytearray, bytes],
                       nonce_offset: int,
                       nonce32: int,
                       seed_hex: str = None) -> bytes:
    """
    Inserta nonce32 (uint32 little-endian) en `blob` en `nonce_offset`, calcula hash y lo retorna.
    blob debe ser bytearray si se modifica in-place.
    """
    if isinstance(blob, bytes):
        # Crear copia mutable
        blob = bytearray(blob)
    if nonce_offset < 0 or nonce_offset + 4 > len(blob):
        raise ValueError("nonce_offset fuera de rango")
    blob[nonce_offset:nonce_offset+4] = nonce32.to_bytes(4, 'little')
    if seed_hex is None:
        raise RandomXSeedError("seed_hex requerido")
    return compute_randomx_hash(bytes(blob), seed_hex)

def hash_meets_target(hash_bytes: bytes, target_int: int) -> bool:
    """
    Verifica si el hash (32 bytes, little-endian) es <= target (entero 256 bits).
    """
    if not isinstance(hash_bytes, (bytes, bytearray)) or len(hash_bytes) != 32:
        return False
    if target_int <= 0:
        return False
    hv = int.from_bytes(hash_bytes, 'little')
    return hv <= target_int

# ---------------------------------------------------------------------
# Limpieza manual (opcional)
# ---------------------------------------------------------------------
def close():
    if _global_manager:
        _global_manager.cleanup()

# ---------------------------------------------------------------------
# Auto-init perezoso si el módulo se importa y se necesita inmediatamente
# (No se hace aquí para permitir configuración previa.)
# ---------------------------------------------------------------------

__all__ = [
    "init_randomx",
    "compute_randomx_hash",
    "compute_hash_nonce",
    "hash_meets_target",
    "close",
    # Constantes
    "RANDOMX_FLAG_DEFAULT", "RANDOMX_FLAG_LARGE_PAGES", "RANDOMX_FLAG_HARD_AES",
    "RANDOMX_FLAG_FULL_MEM", "RANDOMX_FLAG_JIT", "RANDOMX_FLAG_SECURE"
]
