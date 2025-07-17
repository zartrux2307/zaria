import os
import sys
import ctypes
import threading
import logging
import atexit

logger = logging.getLogger("RandomXWrapper")

# Configuración inicial (sin cambios)
DLL_PATH = r"C:\zarturxia\src\libs\randomx.dll"

if not os.path.exists(DLL_PATH):
    logger.critical(f"ERROR: No se encontró la DLL en {DLL_PATH}")
    sys.exit(1)
    
try:
    os.environ['PATH'] = os.path.dirname(DLL_PATH) + os.pathsep + os.environ['PATH']
    randomx_lib = ctypes.WinDLL(DLL_PATH)
    logger.info(f"Biblioteca cargada: {DLL_PATH}")
except Exception as e:
    logger.critical(f"ERROR: Fallo al cargar DLL: {str(e)}")
    sys.exit(1)

FUNCTION_DEFS = {
    'randomx_alloc_cache': (ctypes.c_void_p, [ctypes.c_uint]),
    'randomx_init_cache': (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]),
    'randomx_create_vm': (ctypes.c_void_p, [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]),
    'randomx_calculate_hash': (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]),
    'randomx_destroy_vm': (None, [ctypes.c_void_p]),
    'randomx_release_cache': (None, [ctypes.c_void_p])
}

for func_name, (restype, argtypes) in FUNCTION_DEFS.items():
    try:
        func = getattr(randomx_lib, func_name)
        func.restype = restype
        func.argtypes = argtypes
    except AttributeError:
        logger.critical(f"ERROR: Función no encontrada: {func_name}")
        sys.exit(1)

# ===== GESTIÓN OPTIMIZADA DE VMs =====
class OptimizedVMManager:
    def __init__(self):
        self.caches = {}
        self.vm_pool = {}
        self.lock = threading.Lock()
        self.local = threading.local()
        atexit.register(self.cleanup)

    def get_vm(self, seed):
        """Obtiene una VM para la seed actual del hilo"""
        # 1. Verificar si el hilo ya tiene una VM configurada
        if hasattr(self.local, 'current_seed') and self.local.current_seed == seed:
            return self.local.vm_ptr
        
        # 2. Limpiar VM anterior si existe
        if hasattr(self.local, 'vm_ptr'):
            self._release_vm(self.local.vm_ptr, self.local.current_seed)
            delattr(self.local, 'vm_ptr')
            delattr(self.local, 'current_seed')
        
        # 3. Obtener o crear nueva VM
        with self.lock:
            if seed not in self.vm_pool:
                self._init_seed_resources(seed)
            
            # Tomar VM del pool
            if self.vm_pool[seed]:
                vm_ptr = self.vm_pool[seed].pop()
                logger.debug(f"VM reutilizada para seed {seed[:8]}... (hilo: {threading.current_thread().name})")
            else:
                vm_ptr = self._create_vm(seed)
                logger.debug(f"Nueva VM creada para seed {seed[:8]}... (hilo: {threading.current_thread().name})")
        
        # 4. Configurar en almacenamiento local del hilo
        self.local.vm_ptr = vm_ptr
        self.local.current_seed = seed
        return vm_ptr

    def _init_seed_resources(self, seed):
        """Inicializa recursos para una nueva seed"""
        # Crear cache
        cache = randomx_lib.randomx_alloc_cache(0)
        if not cache:
            raise MemoryError("Error creando cache")
        
        seed_bytes = bytes.fromhex(seed)
        if len(seed_bytes) != 32:
            raise ValueError("Seed debe ser 32 bytes")
        
        seed_buf = (ctypes.c_ubyte * 32)(*seed_bytes)
        randomx_lib.randomx_init_cache(cache, seed_buf, 32)
        
        # Inicializar estructuras
        self.caches[seed] = cache
        self.vm_pool[seed] = []
        logger.info(f"Recursos inicializados para seed {seed[:8]}...")

    def _create_vm(self, seed):
        """Crea una nueva VM para una seed específica"""
        cache = self.caches[seed]
        vm_ptr = randomx_lib.randomx_create_vm(0, cache, None)
        if not vm_ptr:
            raise RuntimeError("Error creando VM")
        return vm_ptr

    def _release_vm(self, vm_ptr, seed):
        """Devuelve una VM al pool para reutilización"""
        with self.lock:
            if seed in self.vm_pool:
                self.vm_pool[seed].append(vm_ptr)

    def cleanup(self):
        """Limpieza final de todos los recursos"""
        with self.lock:
            # Destruir todas las VMs
            for seed, vm_list in self.vm_pool.items():
                for vm_ptr in vm_list:
                    randomx_lib.randomx_destroy_vm(vm_ptr)
                logger.debug(f"Destruidas {len(vm_list)} VMs para seed {seed[:8]}...")
            
            # Liberar caches
            for seed, cache in self.caches.items():
                randomx_lib.randomx_release_cache(cache)
                logger.debug(f"Cache liberada para seed {seed[:8]}...")
            
            self.vm_pool.clear()
            self.caches.clear()

# Instancia global optimizada
vm_manager = OptimizedVMManager()

# Funciones principales (sin cambios)
def compute_randomx_hash(blob, seed):
    if not blob or len(blob) < 76:
        raise ValueError("Blob inválido")
    if len(seed) != 64:
        raise ValueError("Seed inválida")
    
    vm_ptr = vm_manager.get_vm(seed)
    input_buf = (ctypes.c_ubyte * len(blob))(*blob)
    output_buf = (ctypes.c_ubyte * 32)()
    randomx_lib.randomx_calculate_hash(vm_ptr, input_buf, len(blob), output_buf)
    return bytes(output_buf)

def hash_meets_target(hash_bytes, target_int):
    try:
        return int.from_bytes(hash_bytes, 'little') < (target_int << 192)
    except:
        return False