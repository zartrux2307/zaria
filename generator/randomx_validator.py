import os
import sys
import ctypes
import threading
import logging
import atexit
import secrets

# Logger setup
logger = logging.getLogger("RandomXValidator")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s"
    )

DLL_PATH = r"C:\zarturxia\src\libs\randomx.dll"

# Advanced RandomX Flags
RANDOMX_FLAG_DEFAULT = 0
RANDOMX_FLAG_LARGE_PAGES = 1
RANDOMX_FLAG_JIT = 2
RANDOMX_FLAG_FULL_MEM = 4
RANDOMX_FLAG_HARD_AES = 8
RANDOMX_FLAG_SECURE = 16

def get_randomx_flags(config=None):
    flags = RANDOMX_FLAG_DEFAULT
    cfg = config or {}
    if cfg.get("use_large_pages", True):
        flags |= RANDOMX_FLAG_LARGE_PAGES
    if cfg.get("use_jit", True):
        flags |= RANDOMX_FLAG_JIT
    if cfg.get("use_hard_aes", True):
        flags |= RANDOMX_FLAG_HARD_AES
    if cfg.get("use_full_mem", False):
        flags |= RANDOMX_FLAG_FULL_MEM
    if cfg.get("secure_mode", False):
        flags |= RANDOMX_FLAG_SECURE
    return flags

# DLL Loading
if not os.path.exists(DLL_PATH):
    logger.critical(f"RandomX DLL not found at {DLL_PATH}")
    sys.exit(1)

try:
    os.environ['PATH'] = os.path.dirname(DLL_PATH) + os.pathsep + os.environ.get('PATH', '')
    randomx_lib = ctypes.WinDLL(DLL_PATH)
    logger.info(f"RandomX library loaded: {DLL_PATH}")
except Exception as e:
    logger.critical(f"Failed to load RandomX DLL: {e}")
    sys.exit(1)

# Function Definitions
_FUNCTIONS = {
    'randomx_alloc_cache': (ctypes.c_void_p, [ctypes.c_uint]),
    'randomx_init_cache': (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]),
    'randomx_create_vm': (ctypes.c_void_p, [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]),
    'randomx_calculate_hash': (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]),
    'randomx_destroy_vm': (None, [ctypes.c_void_p]),
    'randomx_release_cache': (None, [ctypes.c_void_p]),
}

for name, (restype, argtypes) in _FUNCTIONS.items():
    try:
        func = getattr(randomx_lib, name)
        func.restype = restype
        func.argtypes = argtypes
    except AttributeError:
        logger.critical(f"RandomX DLL missing function: {name}")
        sys.exit(1)

# VM Manager (Thread-Safe Pool)
class OptimizedVMManager:
    MAX_POOL_PER_SEED = 8

    def __init__(self, flags=RANDOMX_FLAG_JIT | RANDOMX_FLAG_HARD_AES | RANDOMX_FLAG_LARGE_PAGES):
        self.caches = {}
        self.vm_pool = {}
        self.lock = threading.Lock()
        self.local = threading.local()
        self.flags = flags
        atexit.register(self.cleanup)

    def get_vm(self, seed: str):
        if hasattr(self.local, 'current_seed') and self.local.current_seed == seed:
            return self.local.vm_ptr
        if hasattr(self.local, 'vm_ptr'):
            self._release_vm(self.local.vm_ptr, self.local.current_seed)
            del self.local.vm_ptr
            del self.local.current_seed
        with self.lock:
            if seed not in self.vm_pool:
                self._init_seed_resources(seed)
            pool = self.vm_pool[seed]
            if pool:
                vm_ptr = pool.pop()
                logger.debug(f"Reusing VM for seed {seed[:8]}")
            else:
                vm_ptr = self._create_vm(seed)
                logger.debug(f"Created new VM for seed {seed[:8]}")
        self.local.vm_ptr = vm_ptr
        self.local.current_seed = seed
        return vm_ptr

    def _init_seed_resources(self, seed: str):
        cache = randomx_lib.randomx_alloc_cache(self.flags)
        if not cache:
            raise MemoryError("randomx_alloc_cache failed")
        seed_bytes = bytes.fromhex(seed)
        if len(seed_bytes) != 32:
            raise ValueError("Seed must be 32 bytes (hex-64)")
        seed_buf = (ctypes.c_ubyte * 32)(*seed_bytes)
        randomx_lib.randomx_init_cache(cache, seed_buf, 32)
        self.caches[seed] = cache
        self.vm_pool[seed] = []
        logger.info(f"RandomX seed resources initialized: {seed[:8]}...")

    def _create_vm(self, seed: str):
        cache = self.caches[seed]
        vm_ptr = randomx_lib.randomx_create_vm(self.flags, cache, None)
        if not vm_ptr:
            raise RuntimeError("randomx_create_vm failed")
        return vm_ptr

    def _release_vm(self, vm_ptr, seed: str):
        with self.lock:
            pool = self.vm_pool.get(seed, [])
            if len(pool) < self.MAX_POOL_PER_SEED:
                pool.append(vm_ptr)
            else:
                randomx_lib.randomx_destroy_vm(vm_ptr)
                logger.info(f"Destroyed VM (pool full) for seed {seed[:8]}")

    def cleanup(self):
        with self.lock:
            for seed, vms in self.vm_pool.items():
                for vm_ptr in vms:
                    randomx_lib.randomx_destroy_vm(vm_ptr)
                logger.debug(f"Destroyed all VMs for seed {seed[:8]}")
            for seed, cache in self.caches.items():
                randomx_lib.randomx_release_cache(cache)
                logger.debug(f"Released cache for seed {seed[:8]}")
            self.vm_pool.clear()
            self.caches.clear()
            logger.info("RandomX resources cleaned up.")

# Singleton for all VM management
vm_manager = OptimizedVMManager()

# Main Validator Interface
class RandomXValidator:
    def __init__(self, config=None):
        self.flags = get_randomx_flags(config)
    
    def validate(self, nonce: int, block_data: dict) -> bool:
        try:
            if 'blob' not in block_data or len(block_data['blob']) < 76:
                block_data['blob'] = secrets.token_bytes(76)
            if 'seed' not in block_data:
                block_data['seed'] = secrets.token_hex(32)
            if 'target' not in block_data:
                block_data['target'] = 2**256 - 1
            
            blob = self._make_blob(nonce, block_data)
            seed = block_data.get('seed')
            target = block_data.get('target')
            hash_bytes = self.compute_hash(blob, seed)
            return hash_meets_target(hash_bytes, target)
        except Exception as ex:
            logger.error(f"[RandomXValidator] Validation error: {ex}")
            return False
    
    def compute_hash(self, blob: bytes, seed: str) -> bytes:
        if not blob or len(blob) < 76:
            raise ValueError("Invalid blob")
        if len(seed) != 64:
            raise ValueError("Seed must be 32 bytes (64 hex chars)")
        vm_ptr = vm_manager.get_vm(seed)
        input_buf = (ctypes.c_ubyte * len(blob))(*blob)
        output_buf = (ctypes.c_ubyte * 32)()
        randomx_lib.randomx_calculate_hash(vm_ptr, input_buf, len(blob), output_buf)
        return bytes(output_buf)
    
    @staticmethod
    def _make_blob(nonce: int, block_data: dict) -> bytes:
        blob = block_data.get('blob')
        if not blob or len(blob) < 76:
            raise ValueError("Missing or invalid blob in block_data")
        return blob[:39] + nonce.to_bytes(4, 'little') + blob[43:]

def hash_meets_target(hash_bytes: bytes, target_int: int) -> bool:
    try:
        # CORRECCIÓN: Usar big-endian para evitar errores de conversión
        hash_int = int.from_bytes(hash_bytes, 'big')
        return hash_int < target_int
    except Exception as e:
        logger.error(f"Target comparison error: {e}")
        return False

logger.info("RandomXValidator ready for enterprise mining.")