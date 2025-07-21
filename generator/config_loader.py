from __future__ import annotations
"""
config_loader.py
----------------
Carga y cachea `global_config.json` con soporte de:
 - Reload automático por mtime.
 - Acceso segmentado por secciones.
 - Override por variables de entorno (prefijo IAZAR_CFG__SECTION__KEY).
 - Método utility para obtener sub‑config de generadores.

Uso:
    from iazar.generator.config_loader import config_loader
    cfg = config_loader.get()
    rnd_cfg = config_loader.section("random_generator")
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

class _ConfigLoader:
    _lock = threading.RLock()
    _cache: Dict[str, Any] = {}
    _mtime: float = 0.0

    def __init__(self):
        self.base_dir = Path(os.environ.get("IAZAR_BASE", "C:/zarturxia/src/iazar")).resolve()
        self.config_path = self.base_dir / "config" / "global_config.json"

    def _load_raw(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _apply_env_overrides(self, config: Dict[str, Any]):
        """
        Variables de entorno:
          IAZAR_CFG__<SECTION>__<KEY>=value
          IAZAR_CFG__ROOT__clave=value  (para nivel root)

        Conversión básica de tipos: int, float, bool, json si empieza con { o [.
        """
        prefix = "IAZAR_CFG__"
        for k, v in os.environ.items():
            if not k.startswith(prefix):
                continue
            parts = k[len(prefix):].split("__")
            if len(parts) != 2:
                continue
            section, key = parts
            raw = v.strip()
            # Coerción
            val: Any
            if raw.lower() in ("true", "false"):
                val = raw.lower() == "true"
            else:
                try:
                    if raw.startswith("{") or raw.startswith("["):
                        val = json.loads(raw)
                    elif "." in raw:
                        val = float(raw)
                    else:
                        val = int(raw)
                except Exception:
                    val = raw
            if section.upper() == "ROOT":
                config[key] = val
            else:
                sec = config.setdefault(section, {})
                if isinstance(sec, dict):
                    sec[key] = val

    def get(self, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            mtime = self.config_path.stat().st_mtime if self.config_path.exists() else 0.0
            if force or not self._cache or mtime != self._mtime:
                data = self._load_raw()
                self._apply_env_overrides(data)
                self._cache = data
                self._mtime = mtime
            return self._cache

    def section(self, name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = self.get()
        return cfg.get(name, default or {})

config_loader = _ConfigLoader()
