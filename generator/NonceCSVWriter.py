from __future__ import annotations
"""
NonceCSVWriter (Refactor Producción)
-----------------------------------
Writer centralizado y eficiente para nonces con esquema estándar.

Características:
- Singleton por ruta (evita múltiples instancias compitiendo en el mismo proceso).
- Buffer configurable + flush automático seguro.
- Rotación por fecha y por tamaño opcional.
- Sanitización de tipos y máscara 32-bit para 'nonce'.
- Opción de flush atómico (archivo temporal + append / rename).
- Escapado básico contra fórmulas CSV (Excel injection).
- Estadísticas internas consultables.
- Preparado para instrumentación (# METRIC: ...).

Campos estándar: ["nonce","entropy","uniqueness","zero_density","pattern_score","is_valid","block_height","hash" (opcional)]

Uso rápido:
    writer = NonceCSVWriter("C:/zarturxia/src/iazar/data/nonces_exitosos.csv", batch_size=500)
    writer.write_many(records)
    writer.flush()
"""

import csv
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Iterable, ClassVar, Union

STANDARD_FIELDNAMES = [
    "nonce", "entropy", "uniqueness", "zero_density",
    "pattern_score", "is_valid", "block_height", "hash"
]

_ESCAPE_PREFIXES = ('=', '+', '-', '@')

def _escape_csv(value):
    if isinstance(value, str) and value.startswith(_ESCAPE_PREFIXES):
        return "'" + value
    return value

class NonceCSVWriter:
    _instances: ClassVar[dict[str, "NonceCSVWriter"]] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, file_path: Union[str, Path], **kwargs):
        file_path = str(file_path)
        with cls._instances_lock:
            inst = cls._instances.get(file_path)
            if inst is None:
                inst = super().__new__(cls)
                cls._instances[file_path] = inst
            return inst

    def __init__(
        self,
        file_path: Union[str, Path],
        fieldnames: Optional[List[str]] = None,
        batch_size: int = 1000,
        max_buffer: int = 50_000,
        atomic_flush: bool = False,
        rotate_daily: bool = False,
        max_bytes: Optional[int] = None,
        date_pattern: str = "%Y%m%d",
    ):
        if hasattr(self, "_initialized"):
            return
        self.path = Path(file_path)
        self.fieldnames = fieldnames or STANDARD_FIELDNAMES
        self.batch_size = int(batch_size)
        self.max_buffer = int(max_buffer)
        self.atomic_flush = bool(atomic_flush)
        self.rotate_daily = bool(rotate_daily)
        self.max_bytes = max_bytes
        self.date_pattern = date_pattern

        self._buffer: List[Dict] = []
        self._lock = threading.RLock()
        self._last_rotation_tag = self._current_date_tag() if rotate_daily else None
        self._writer_header_written = self.path.exists() and self.path.stat().st_size > 0
        self._closed = False

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Métricas internas
        self._rows_written = 0
        self._flush_count = 0

        import atexit
        atexit.register(self._atexit_flush)

        # Logger local opcional
        self._log = None

        self._initialized = True

    # -------------------- Public API --------------------
    def write(self, row: Dict):
        if self._closed:
            raise RuntimeError("Writer closed.")
        sanitized = self._sanitize_row(row)
        with self._lock:
            self._buffer.append(sanitized)
            if len(self._buffer) >= self.batch_size or len(self._buffer) >= self.max_buffer:
                self._flush_locked()

    def write_many(self, rows: Iterable[Dict]):
        if self._closed:
            raise RuntimeError("Writer closed.")
        with self._lock:
            for row in rows:
                self._buffer.append(self._sanitize_row(row))
                if len(self._buffer) >= self.batch_size:
                    self._flush_locked()
            if len(self._buffer) >= self.max_buffer:
                self._flush_locked()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def rotate_if_needed(self):
        """Chequea condiciones de rotación y ejecuta si procede (llamar bajo lock)."""
        if self._closed:
            return
        if self.rotate_daily:
            tag = self._current_date_tag()
            if tag != self._last_rotation_tag:
                self._perform_rotation(tag, reason="daily")
        if self.max_bytes and self.path.exists():
            if self.path.stat().st_size >= self.max_bytes:
                self._perform_rotation(self._current_date_tag(), reason="size")

    def close(self):
        with self._lock:
            if not self._closed:
                self._flush_locked()
                self._closed = True

    @classmethod
    def close_all(cls):
        with cls._instances_lock:
            for inst in list(cls._instances.values()):
                try:
                    inst.close()
                except Exception:
                    pass

    # -------------------- Internal --------------------
    def _sanitize_row(self, row: Dict) -> Dict:
        out = {}
        for f in self.fieldnames:
            if f not in row:
                # Default según tipo
                if f == "is_valid":
                    out[f] = False
                elif f == "block_height":
                    out[f] = 0
                elif f == "nonce":
                    out[f] = 0
                else:
                    out[f] = 0.0
                continue
            v = row[f]
            if f == "nonce":
                try:
                    v = int(v) & 0xFFFFFFFF
                except Exception:
                    v = 0
            elif f in ("entropy", "uniqueness", "zero_density", "pattern_score"):
                try:
                    v = float(v)
                except Exception:
                    v = 0.0
            elif f == "is_valid":
                v = bool(v)
            elif f == "block_height":
                try:
                    v = int(v)
                except Exception:
                    v = 0
            elif f == "hash":
                if v is None:
                    v = ""
                else:
                    v = str(v)
            out[f] = _escape_csv(v)
        return out

    def _flush_locked(self):
        if not self._buffer:
            return
        rows = self._buffer
        self._buffer = []
        self._write_rows(rows)

    def _write_rows(self, rows: List[Dict]):
        self.rotate_if_needed()
        file_exists = self.path.exists()
        need_header = not self._writer_header_written

        tmp_path = self.path.with_suffix(".tmp") if self.atomic_flush else self.path

        try:
            with open(tmp_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if need_header:
                    writer.writeheader()
                    self._writer_header_written = True
                writer.writerows(rows)
        except Exception:
            # Reinsertar para no perder datos
            self._buffer[:0] = rows
            raise
        finally:
            if self.atomic_flush and tmp_path.exists():
                if file_exists:
                    try:
                        with open(tmp_path, "r", encoding="utf-8") as src, \
                             open(self.path, "a", encoding="utf-8") as dst:
                            # Saltar encabezado del tmp si ya existía archivo principal
                            lines = src.readlines()
                            start_idx = 1 if file_exists and lines and lines[0].startswith("nonce") else 0
                            for line in lines[start_idx:]:
                                dst.write(line)
                    finally:
                        try:
                            tmp_path.unlink()
                        except Exception:
                            pass
                else:
                    # rename directo
                    try:
                        tmp_path.replace(self.path)
                    except Exception:
                        pass

        self._rows_written += len(rows)
        self._flush_count += 1
        # METRIC: nonce_writer_rows_written.inc(len(rows))
        # METRIC: nonce_writer_flush_total.inc()

    def _perform_rotation(self, new_tag: str, reason: str):
        if not self.path.exists():
            self._last_rotation_tag = new_tag
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        rotated = self.path.with_name(f"{self.path.stem}_{new_tag}_{reason}_{ts}{self.path.suffix}")
        try:
            self.path.replace(rotated)
            self._writer_header_written = False
            self._last_rotation_tag = new_tag
            # METRIC: nonce_writer_rotations_total.inc()
        except Exception:
            # Rotación fallida -> continuar sin interrumpir
            pass

    def _current_date_tag(self) -> str:
        return time.strftime(self.date_pattern)

    def _atexit_flush(self):
        try:
            self.close()
        except Exception:
            pass

    # -------------------- Info / Stats --------------------
    def stats(self) -> dict:
        with self._lock:
            return {
                "rows_written": self._rows_written,
                "flush_count": self._flush_count,
                "buffer_len": len(self._buffer),
                "path": str(self.path),
                "rotated_tag": self._last_rotation_tag,
                "closed": self._closed
            }
