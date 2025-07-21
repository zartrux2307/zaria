from __future__ import annotations
import os
import ctypes
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

LOGGER_NAME = "generator.randomx_validator"
logger = logging.getLogger(LOGGER_NAME)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Flags RandomX
RANDOMX_FLAG_DEFAULT       = 0x0
RANDOMX_FLAG_LARGE_PAGES   = 1 << 0
RANDOMX_FLAG_HARD_AES      = 1 << 2
RANDOMX_FLAG_FULL_MEM      = 1 << 3
RANDOMX_FLAG_JIT           = 1 << 4
RANDOMX_FLAG_SECURE        = 1 << 5

DEFAULT_FLAGS = ["HARD_AES"]

class RandomXValidator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.dll_path = Path(self.config.get("dll_path") or
                             os.environ.get("RANDOMX_DLL") or
                             "C:/zarturxia/src/libs/randomx.dll")
        self.flags_cfg = self.config.get("flags", DEFAULT_FLAGS)
        self._flags_value = self._parse_flags(self.flags_cfg)
        self.available = False
        self._lib = None
        self._cache = None
        self._dataset = None
        self._seed_key = None
        self._lock = threading.RLock()
        self._fallback = False
        self.mode = self.config.get("mode", "target")
        self._load_library()
        logger.info("RandomXValidator init | dll=%s available=%s flags=0x%x mode=%s",
                    self.dll_path, self.available, self._flags_value, self.mode)

    def _parse_flags(self, flags_list):
        v = 0
        for f in flags_list:
            f = f.upper()
            if f == "LARGE_PAGES":
                v |= RANDOMX_FLAG_LARGE_PAGES
            elif f == "HARD_AES":
                v |= RANDOMX_FLAG_HARD_AES
            elif f == "FULL_MEM":
                v |= RANDOMX_FLAG_FULL_MEM
            elif f == "JIT":
                v |= RANDOMX_FLAG_JIT
            elif f == "SECURE":
                v |= RANDOMX_FLAG_SECURE
        return v

    def _load_library(self):
        if not self.dll_path.exists():
            logger.error("RandomX DLL no encontrada: %s", self.dll_path)
            return
        try:
            self._lib = ctypes.CDLL(str(self.dll_path))
        except Exception as e:
            logger.error("Error cargando DLL RandomX: %s", e)
            return
        # Firmas simplificadas
        try:
            self._lib.randomx_alloc_cache.restype = ctypes.c_void_p
            self._lib.randomx_alloc_cache.argtypes = [ctypes.c_int]
            self._lib.randomx_init_cache.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_void_p,
                                                     ctypes.c_size_t]
            self._lib.randomx_release_cache.argtypes = [ctypes.c_void_p]
            self._lib.randomx_alloc_dataset.restype = ctypes.c_void_p
            self._lib.randomx_alloc_dataset.argtypes = [ctypes.c_int]
            self._lib.randomx_dataset_item_count.restype = ctypes.c_ulonglong
            self._lib.randomx_init_dataset.argtypes = [ctypes.c_void_p,
                                                       ctypes.c_void_p,
                                                       ctypes.c_ulonglong,
                                                       ctypes.c_ulonglong]
            self._lib.randomx_release_dataset.argtypes = [ctypes.c_void_p]
            self._lib.randomx_create_vm.restype = ctypes.c_void_p
            self._lib.randomx_create_vm.argtypes = [ctypes.c_int,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p]
            self._lib.randomx_destroy_vm.argtypes = [ctypes.c_void_p]
            self._lib.randomx_calculate_hash_first.argtypes = [ctypes.c_void_p]
            self._lib.randomx_calculate_hash_last.argtypes = [ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_size_t,
                                                              ctypes.c_void_p]
            self.available = True
        except Exception as e:
            logger.error("Asignación de firmas RandomX falló: %s", e)
            self.available = False

    # ---------------- Seed / Dataset ----------------
    def _ensure_seed(self, seed: bytes):
        if not self.available or self._fallback:
            return
        with self._lock:
            if self._seed_key == seed:
                return
            # liberar previos
            if self._cache:
                self._lib.randomx_release_cache(self._cache)
                self._cache = None
            if self._dataset:
                self._lib.randomx_release_dataset(self._dataset)
                self._dataset = None

            cache = self._lib.randomx_alloc_cache(self._flags_value)
            if not cache:
                if (self._flags_value & RANDOMX_FLAG_LARGE_PAGES):
                    logger.warning("Fallo alloc_cache con LARGE_PAGES; reintentando sin LARGE_PAGES.")
                    alt = self._flags_value & ~RANDOMX_FLAG_LARGE_PAGES
                    cache = self._lib.randomx_alloc_cache(alt)
                    if cache:
                        logger.info("Reintento alloc_cache exitoso sin LARGE_PAGES (flags=0x%x)", alt)
                        self._flags_value = alt
                if not cache:
                    logger.error("randomx_alloc_cache devolvió NULL (abort seed init)")
                    self._fallback = True
                    return

            self._cache = cache
            # init cache
            seed_buf = ctypes.create_string_buffer(seed)
            self._lib.randomx_init_cache(self._cache, seed_buf, len(seed))
            # dataset (opcional si FULL_MEM)
            if self._flags_value & RANDOMX_FLAG_FULL_MEM:
                ds = self._lib.randomx_alloc_dataset(self._flags_value)
                if ds:
                    count = self._lib.randomx_dataset_item_count()
                    self._lib.randomx_init_dataset(ds, self._cache, 0, count)
                    self._dataset = ds
            self._seed_key = seed

    # ---------------- Validation ----------------
    def validate(self, nonce: int, block_data: Dict[str, Any], return_hash: bool = False):
        # Placeholder hash generation (sin implementación VM completa para brevedad)
        # Puedes extender con VM real si tienes binding completo.
        if not self.available or self._fallback:
            return (True, b"\x00" * 32) if return_hash else True
        seed_hex = block_data.get("seed_hash") or ("00" * 32)
        try:
            seed = bytes.fromhex(seed_hex)
        except Exception:
            seed = b"\x00" * 32
        self._ensure_seed(seed)
        # Hash dummy determinista (XOR simple con nonce) – reemplazar con VM real
        h = bytearray(32)
        n = nonce.to_bytes(4, "little")
        for i in range(32):
            h[i] = n[i % 4] ^ seed[i]
        valid = True  # Aquí pondrías la evaluación según target
        if return_hash:
            return valid, bytes(h)
        return valid

    def close(self):
        with self._lock:
            if self._cache:
                try:
                    self._lib.randomx_release_cache(self._cache)
                except Exception:
                    pass
                self._cache = None
            if self._dataset:
                try:
                    self._lib.randomx_release_dataset(self._dataset)
                except Exception:
                    pass
                self._dataset = None
