import json
import os
import threading
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Configuración de logging para este módulo
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class ConfigLoader:
    """
    Carga y cachea `global_config.json` con:
      - Rutas absolutas robustas
      - Recarga automática cuando cambia mtime
      - Overrides por variables de entorno prefijadas
      - Acceso thread-safe mediante RLock
    """

    def __init__(self):
        # Bloque reentrante para operaciones atómicas
        self._lock = threading.RLock()
        # Determina directorio base: preferencia a IAZAR_BASE, si no, usa relativa al paquete
        env_base = os.environ.get("IAZAR_BASE")
        if env_base:
            self.base_dir = Path(env_base).expanduser().resolve()
        else:
            # __file__ está en iazar/generator/config_loader.py -> sube tres niveles
            self.base_dir = (Path(__file__).parent.parent.parent).resolve()
        self.config_path = self.base_dir / "config" / "global_config.json"
        self._cache: Dict[str, Any] = {}
        self._mtime: float = 0.0

    def _load_raw(self) -> Dict[str, Any]:
        if not self.config_path.is_file():
            logger.warning("Archivo de configuración no encontrado: %s", self.config_path)
            return {}
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Error al parsear JSON en %s: %s", self.config_path, e)
            return {}
        except Exception as e:
            logger.exception("Fallo inesperado al leer %s", self.config_path)
            return {}

    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        prefix = "IAZAR_CFG__"
        for key_env, raw_val in os.environ.items():
            if not key_env.startswith(prefix):
                continue
            parts = key_env[len(prefix):].split("__", maxsplit=1)
            if len(parts) != 2:
                continue
            section, key = parts
            raw = raw_val.strip()
            # Coerción básica de tipos
            try:
                if raw.lower() in ("true", "false"):
                    val: Any = raw.lower() == "true"
                elif raw.startswith("{") or raw.startswith("["):
                    val = json.loads(raw)
                elif "." in raw:
                    val = float(raw)
                else:
                    val = int(raw)
            except Exception:
                val = raw
            # Asignación
            if section.upper() == "ROOT":
                config[key] = val
            else:
                sec = config.setdefault(section, {})
                if isinstance(sec, dict):
                    sec[key] = val

    def get(self, force: bool = False) -> Dict[str, Any]:
        """
        Devuelve la configuración completa. Recarga si:
         - force=True
         - cache vacío
         - mtime ha cambiado
        """
        with self._lock:
            try:
                mtime = self.config_path.stat().st_mtime
            except OSError as e:
                logger.error("No se pudo obtener mtime de %s: %s", self.config_path, e)
                mtime = 0.0
            # Compara y recarga si es necesario
            if force or not self._cache or mtime != self._mtime:
                data = self._load_raw()
                self._apply_env_overrides(data)
                # Actualiza cache de forma atómica
                self._cache = data
                self._mtime = mtime
            # Devolvemos copia para evitar modificaciones externas
            return dict(self._cache)

    def section(self, name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Obtiene una sección específica de la configuración, o default si no existe.
        """
        cfg = self.get()
        return cfg.get(name, default or {})

# Instancia global
config_loader = ConfigLoader()
