from __future__ import annotations
"""
ia_proxy_main.py (RandomX integrado)
------------------------------------
Proxy reactor multi-pool para coordinar mineros y la IA vía SHM, con
validación real de shares usando RandomX (opcional).

Mejoras implementadas:
- Compatibilidad mejorada con el nuevo sistema SHM
- Validación optimizada de shares con RandomX
- Manejo robusto de errores y estadísticas
- Configuración TLS mejorada
- Integración con la estructura de directorios
"""

import os
import ssl
import sys
import json
import time
import math
import socket
import argparse
import selectors
import logging
import random
import traceback
import subprocess
import threading
from typing import Dict, Any, List, Optional, Tuple

# Usamos los nuevos canales SHM mejorados
from iazar.proxy.shm_channels import open_job_channel, open_solution_channel, JobChannel, SolutionChannel

from iazar.proxy.pool_connection import PoolConnection
from iazar.proxy.miner_session import MinerSession

# ---- IMPORTA RandomXValidator ----
try:
    from iazar.proxy.randomx_validator import RandomXValidator
    _HAS_RANDOMX = True
except Exception:
    _HAS_RANDOMX = False

logger = logging.getLogger("proxy.main")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ============================================================
# Utilidades
# ============================================================

def generate_self_signed_cert(cert_path: str, key_path: str):
    # Directorio seguro para certificados en Windows
    cert_dir = os.path.join(os.environ.get("TEMP", "C:\\zarturxia\\tmp"), "certs")
    os.makedirs(cert_dir, exist_ok=True)
    
    cert_file = os.path.join(cert_dir, "proxy_cert.pem")
    key_file = os.path.join(cert_dir, "proxy_key.pem")
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        return cert_file, key_file
    
    logger.warning("Generando certificado auto-firmado en: %s", cert_dir)
    try:
        subprocess.check_call([
            "openssl", "req", "-x509", "-nodes",
            "-newkey", "rsa:2048",
            "-keyout", key_file,
            "-out", cert_file,
            "-days", "365",
            "-subj", "/C=NA/ST=NA/L=NA/O=IAZAR/OU=DEV/CN=localhost"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return cert_file, key_file
    except Exception as e:
        logger.error("No se pudo generar certificado: %s", e)
        return None, None

def hex_nonce(nonce: int) -> str:
    return f"{nonce:08x}"

def _decode_hex_maybe(x):
    if isinstance(x, bytes):
        return x
    if not isinstance(x, str):
        return b""
    h = x.strip()
    if h.startswith("0x"):
        h = h[2:]
    if len(h) % 2 == 1:
        h = "0" + h
    try:
        return bytes.fromhex(h)
    except Exception:
        return h.encode("utf-8", errors="replace")

# ============================================================
# ShareValidator (RandomX) - Mejorado
# ============================================================

class ShareValidator:
    """
    Encapsula la validación de shares con manejo mejorado de errores
    y estadísticas detalladas
    """
    def __init__(self, rx_validator: RandomXValidator, sampling: float = 1.0):
        self.rx = rx_validator
        self.sampling = max(0.0, min(1.0, sampling))
        self.random = random.Random(1337)

        # Contadores mejorados
        self.total_submits = 0
        self.validated = 0
        self.skipped = 0
        self.accepted = 0
        self.rejected = 0
        self.errors = {
            "no_blob": 0,
            "bad_nonce_offset": 0,
            "rx_hash_fail": 0,
            "hash_mismatch": 0,
            "low_diff": 0,
            "exception": 0
        }

    def validate_share(self, job: Dict[str, Any], nonce: int, result_hex: str | None) -> Dict[str, Any]:
        """
        Retorna dict con resultados de validación mejorado
        """
        self.total_submits += 1

        # Sampling decision con registro
        do_validate = True
        if self.sampling < 0.999:
            if self.random.random() > self.sampling:
                self.skipped += 1
                return {"validated": False, "accepted": True, "error": None, "hash_hex": None}

        self.validated += 1

        try:
            # Manejo robusto de blob
            blob = job.get("blob")
            if blob is None:
                self.errors["no_blob"] += 1
                return {"validated": True, "accepted": False, "error": "no_blob", "hash_hex": None}
            
            blob_bytes = _decode_hex_maybe(blob)
            seed_bytes = _decode_hex_maybe(job.get("seed", b""))
            target = job.get("target", b"")
            target_bytes = _decode_hex_maybe(target)
            nonce_offset = int(job.get("nonce_offset", 0))

            # Validación de nonce_offset
            if nonce_offset < 0 or nonce_offset + 4 > len(blob_bytes):
                self.errors["bad_nonce_offset"] += 1
                return {"validated": True, "accepted": False, "error": "bad_nonce_offset", "hash_hex": None}

            # Inserta nonce (little endian)
            mutable = bytearray(blob_bytes)
            mutable[nonce_offset:nonce_offset+4] = nonce.to_bytes(4, "little", signed=False)

            # Hash RandomX con manejo de errores
            try:
                h = self.rx.hash(bytes(mutable), seed_bytes)
            except Exception as e:
                logger.error("Error en RandomX.hash: %s", e)
                self.errors["rx_hash_fail"] += 1
                return {"validated": True, "accepted": False, "error": "rx_hash_fail", "hash_hex": None}
            
            if not isinstance(h, (bytes, bytearray)) or len(h) != 32:
                self.errors["rx_hash_fail"] += 1
                return {"validated": True, "accepted": False, "error": "rx_hash_fail", "hash_hex": None}

            # Comparar target con manejo de formato
            if len(target_bytes) < 32:
                target_bytes = target_bytes + b"\x00" * (32 - len(target_bytes))
            elif len(target_bytes) > 32:
                target_bytes = target_bytes[:32]

            try:
                hash_val = int.from_bytes(h, "little")
                target_val = int.from_bytes(target_bytes, "little")
            except Exception as e:
                logger.error("Error convirtiendo valores: %s", e)
                self.errors["exception"] += 1
                return {"validated": True, "accepted": False, "error": f"conversion_error:{e}", "hash_hex": h.hex()}

            if target_val == 0:
                target_val = 1  # Evitar división por cero

            meets = hash_val < target_val

            # Comprobación de hash declarado por minero
            if result_hex and len(result_hex) >= 64:
                miner_h_first = result_hex[:64].lower()
                our_first = h.hex()[:64]
                if miner_h_first != our_first:
                    self.errors["hash_mismatch"] += 1
                    self.rejected += 1
                    return {"validated": True, "accepted": False, "error": "hash_mismatch", "hash_hex": h.hex()}

            if meets:
                self.accepted += 1
                return {"validated": True, "accepted": True, "error": None, "hash_hex": h.hex()}
            else:
                self.errors["low_diff"] += 1
                self.rejected += 1
                return {"validated": True, "accepted": False, "error": "low_diff", "hash_hex": h.hex()}

        except Exception as e:
            self.errors["exception"] += 1
            self.rejected += 1
            logger.exception("Excepción en validate_share: %s", e)
            return {"validated": True, "accepted": False, "error": f"exception:{e}", "hash_hex": None}

    def stats(self) -> Dict[str, Any]:
        return {
            "total_submits": self.total_submits,
            "validated": self.validated,
            "sampling": self.sampling,
            "skipped": self.skipped,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "errors": self.errors
        }

# ============================================================
# MultiPoolManager (Adaptado para nuevo SHM)
# ============================================================

class MultiPoolManager:
    def __init__(self, pool_defs: List[Dict[str, Any]], solutions_capacity: int):
        self.pools: Dict[str, Dict[str, Any]] = {}
        self.solutions_capacity = solutions_capacity
        for p in pool_defs:
            pool_id = p["pool_id"]
            prefix = p.get("prefix") or pool_id
            # Usamos los nuevos canales SHM con capacidad aumentada
            job_ch = open_job_channel(prefix, size=8192)  # Tamaño aumentado
            sol_ch = open_solution_channel(prefix, capacity=solutions_capacity, item_size=96)  # Item size aumentado
            conn = PoolConnection(
                pool_id=pool_id,
                job_channel=job_ch,
                url=p.get("url", ""),
                port=p.get("port", 0),
                user=p.get("user", ""),
                password=p.get("password", "x"),
                tls=not p.get("notls", False),
                tls_fingerprint=p.get("tls_fingerprint"),
                enabled=p.get("enabled", True)
            )
            self.pools[pool_id] = {
                "job_ch": job_ch,
                "sol_ch": sol_ch,
                "conn": conn,
                "last_job_version": 0
            }

    def start_all(self):
        for obj in self.pools.values():
            obj["conn"].start()

    def stop_all(self):
        for obj in self.pools.values():
            obj["conn"].stop()

    def get_current_job(self) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[str]]:
        for pid, obj in self.pools.items():
            job = obj["job_ch"].get_job()
            if job and job.get("job_version", 0) > 0:
                return job, job["job_version"], pid
        return None, None, None

    def poll_new_jobs(self) -> List[Tuple[str, Dict[str, Any], int]]:
        updates = []
        for pid, obj in self.pools.items():
            job = obj["job_ch"].get_job()
            if not job:
                continue
            ver = job.get("job_version", 0)
            if ver > obj["last_job_version"]:
                obj["last_job_version"] = ver
                updates.append((pid, job, ver))
        return updates

    def submit_solution_from_ai(self, forward_to_pool: bool = True) -> int:
        # Placeholder para integración con IA
        return 0

    def stats(self) -> Dict[str, Any]:
        out = {}
        for pid, obj in self.pools.items():
            out[pid] = obj["conn"].stats()
        return out

# ============================================================
# ProxyServer (Mejorado)
# ============================================================

class ProxyServer:
    def __init__(self,
                 listen_host: str,
                 listen_port_plain: Optional[int],
                 listen_port_tls: Optional[int],
                 tls_cert: Optional[str],
                 tls_key: Optional[str],
                 multipool: MultiPoolManager,
                 stats_interval: int = 30,
                 submit_forward: bool = True,
                 max_sessions: int = 10000,
                 max_out_buffer: int = 512 * 1024,
                 share_validator: Optional[ShareValidator] = None):
        self.listen_host = listen_host
        self.listen_port_plain = listen_port_plain
        self.listen_port_tls = listen_port_tls
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.multipool = multipool
        self.stats_interval = stats_interval
        self.submit_forward = submit_forward
        self.max_sessions = max_sessions
        self.max_out_buffer = max_out_buffer
        self.share_validator = share_validator

        self.selector = selectors.DefaultSelector()
        self.sessions: Dict[int, MinerSession] = {}
        self._stop = False
        self._last_stats_time = time.time()
        self._next_housekeeping = time.time()

        self._lsock_plain = None
        self._lsock_tls = None
        self._tls_context = None
        if self.listen_port_tls is not None:
            self._prepare_tls_context()

        # Métricas mejoradas
        self.total_submits = 0
        self.accepted_submits = 0
        self.rejected_submits = 0
        self.sampled_out = 0
        self.session_count = 0

    def _prepare_tls_context(self):
        # Generar certificados si no se proporcionan
        if not (self.tls_cert and self.tls_key):
            self.tls_cert, self.tls_key = generate_self_signed_cert(None, None)
            if not self.tls_cert or not self.tls_key:
                logger.error("No se pudo crear contexto TLS")
                self.listen_port_tls = None
                return
        
        try:
            ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            ctx.load_cert_chain(certfile=self.tls_cert, keyfile=self.tls_key)
            self._tls_context = ctx
        except Exception as e:
            logger.error("Error creando contexto TLS: %s", e)
            self.listen_port_tls = None

    # -------- Public --------
    def start(self):
        self._open_listeners()
        self.multipool.start_all()
        logger.info("Proxy Server started (host=%s plain=%s tls=%s randomx=%s sampling=%s)",
                    self.listen_host, self.listen_port_plain, self.listen_port_tls,
                    bool(self.share_validator), getattr(self.share_validator, "sampling", None))
        self._reactor_loop()

    def stop(self):
        self._stop = True
        for s in list(self.sessions.values()):
            s.close("server_stop")
        for ls in (self._lsock_plain, self._lsock_tls):
            if ls:
                try: ls.close()
                except Exception: pass
        self.multipool.stop_all()

    # -------- Internals --------
    def _open_listener_socket(self, port: int, use_tls: bool):
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((self.listen_host, port))
        lsock.listen(512)
        lsock.setblocking(False)
        self.selector.register(lsock, selectors.EVENT_READ, data=("accept_tls" if use_tls else "accept_plain"))
        return lsock

    def _open_listeners(self):
        if self.listen_port_plain is not None:
            self._lsock_plain = self._open_listener_socket(self.listen_port_plain, use_tls=False)
            logger.info("Escuchando en puerto plain: %d", self.listen_port_plain)
        
        if self.listen_port_tls is not None and self._tls_context:
            self._lsock_tls = self._open_listener_socket(self.listen_port_tls, use_tls=True)
            logger.info("Escuchando en puerto TLS: %d", self.listen_port_tls)

    def _reactor_loop(self):
        try:
            while not self._stop:
                events = self.selector.select(timeout=0.5)
                for key, mask in events:
                    data = key.data
                    if data == "accept_plain":
                        self._accept_new(False, key.fileobj)
                    elif data == "accept_tls":
                        self._accept_new(True, key.fileobj)
                    else:
                        sess: MinerSession = data
                        if mask & selectors.EVENT_READ:
                            sess.on_readable()
                            if sess.closed:
                                self._drop_session(sess)
                                continue
                        if mask & selectors.EVENT_WRITE:
                            sess.on_writable()
                            if sess.closed:
                                self._drop_session(sess)
                                continue
                        self._update_selector_interest(sess)
                now = time.time()
                if now >= self._next_housekeeping:
                    self._housekeeping(now)
                    self._next_housekeeping = now + 1.0
                if self.stats_interval and now - self._last_stats_time >= self.stats_interval:
                    self._print_stats()
                    self._last_stats_time = now
        except KeyboardInterrupt:
            logger.info("Deteniendo (KeyboardInterrupt)")
        except Exception as e:
            logger.error("Error en loop principal: %s", e, exc_info=True)
        finally:
            self.stop()

    def _accept_new(self, is_tls: bool, sockobj):
        try:
            conn, addr = sockobj.accept()
            if is_tls:
                try:
                    conn = self._tls_context.wrap_socket(conn, server_side=True)
                except Exception as e:
                    logger.debug("Fallo handshake TLS desde %s: %s", addr, e)
                    conn.close()
                    return
            if len(self.sessions) >= self.max_sessions:
                conn.close()
                logger.warning("Conexión rechazada: máximo de sesiones alcanzado")
                return
            
            conn.setblocking(False)
            job, ver, pid = self.multipool.get_current_job()
            initial = (job, ver) if job and ver else None
            
            sess = MinerSession(
                sock=conn,
                addr=addr,
                server_callbacks={
                    "on_submit": self._on_submit,
                    "current_job_provider": self.multipool.get_current_job,
                    "on_disconnect": self._on_disconnect_session,
                    "metrics_inc": self._metrics_inc_stub
                },
                initial_job=initial
            )
            
            self.sessions[sess.id] = sess
            self.selector.register(conn, selectors.EVENT_READ, data=sess)
            self.session_count += 1
            logger.info("Miner conectado id=%d addr=%s tls=%s sesiones=%d", 
                        sess.id, addr, is_tls, len(self.sessions))
        except Exception as e:
            logger.debug("Error aceptando conexión: %s", e, exc_info=True)

    def _drop_session(self, sess: MinerSession):
        try:
            self.selector.unregister(sess.sock)
        except Exception:
            pass
        if sess.id in self.sessions:
            del self.sessions[sess.id]
            logger.info("Sesión cerrada id=%d sesiones=%d", sess.id, len(self.sessions))

    def _update_selector_interest(self, sess: MinerSession):
        events = selectors.EVENT_READ
        if sess.want_write():
            events |= selectors.EVENT_WRITE
        try:
            self.selector.modify(sess.sock, events, data=sess)
        except Exception as e:
            logger.debug("Error actualizando selector: %s", e)

    def _housekeeping(self, now: float):
        # Actualizar trabajos para mineros conectados
        updates = self.multipool.poll_new_jobs()
        if updates:
            for pid, job, ver in updates:
                for s in list(self.sessions.values()):
                    s.push_job(job, ver)

        # Limpieza de sesiones
        for s in list(self.sessions.values()):
            if s.closed:
                self._drop_session(s)
                continue
            s.tick(now)
            if len(s.outbuf) > self.max_out_buffer:
                s.close("output_overflow")
                self._drop_session(s)

        # Reenvío de soluciones de IA
        if self.submit_forward:
            self.multipool.submit_solution_from_ai(forward_to_pool=False)

    def _print_stats(self):
        pools_stats = self.multipool.stats()
        extra = {}
        if self.share_validator:
            extra["share_validator"] = self.share_validator.stats()
        
        logger.info("[Stats] sessions=%d submits=%d acc=%d rej=%d pools=%s extra=%s",
                    len(self.sessions),
                    self.total_submits,
                    self.accepted_submits,
                    self.rejected_submits,
                    json.dumps(pools_stats, separators=(',', ':')),
                    json.dumps(extra, separators=(',', ':')))

    # -------- Submit Callback --------
    def _on_submit(self, job_id: str, nonce_int: int, result_hex: str, session: MinerSession) -> Dict[str, Any]:
        self.total_submits += 1
        job, ver, pid = self.multipool.get_current_job()
        if not job:
            return {"result": False, "error": "No job"}
        if job_id != job.get("job_id"):
            return {"result": False, "error": "Job mismatch"}
        if nonce_int < 0 or nonce_int > 0xFFFFFFFF:
            return {"result": False, "error": "Bad nonce"}

        accepted = True
        error = None
        hash_hex = None

        # Validación con RandomX si está activado
        if self.share_validator:
            res = self.share_validator.validate_share(job, nonce_int, result_hex)
            hash_hex = res.get("hash_hex")
            if not res.get("accepted"):
                accepted = False
                error = res.get("error")

        if accepted:
            pool_obj = self.multipool.pools.get(pid)
            if pool_obj:
                sol_ch: SolutionChannel = pool_obj["sol_ch"]
                
                # Usar el hash calculado si está disponible
                if hash_hex:
                    hbytes = bytes.fromhex(hash_hex[:64])
                elif result_hex and len(result_hex) >= 64:
                    hbytes = bytes.fromhex(result_hex[:64])
                else:
                    hbytes = b"\x00" * 32
                
                # Enviar solución al canal de soluciones
                success = sol_ch.try_submit(
                    job_version=ver,
                    nonce=nonce_int,
                    hash_bytes=hbytes,
                    valid=True,
                    job_id=job_id,
                    pool_id=pid
                )
                
                if not success:
                    logger.warning("Error enviando solución al canal SHM")
                
                # Reenviar al pool remoto si está habilitado
                conn: PoolConnection = pool_obj["conn"]
                if conn and conn.enabled and conn.stats().get("connected"):
                    submit_msg = {
                        "id": 1000 + session.id,
                        "method": "mining.submit",
                        "params": [
                            session.worker_name,
                            job_id,
                            hex_nonce(nonce_int),
                            hash_hex if hash_hex else result_hex
                        ]
                    }
                    try:
                        conn._send_json(submit_msg)  # pylint: disable=protected-access
                    except Exception as e:
                        logger.debug("Error reenviando submit: %s", e)
            
            self.accepted_submits += 1
            return {"result": True}
        else:
            self.rejected_submits += 1
            return {"result": False, "error": error or "Rejected"}

    def _on_disconnect_session(self, session: MinerSession):
        self._drop_session(session)

    def _metrics_inc_stub(self, name: str, value: int = 1):
        # Placeholder para métricas extendidas
        pass

# ============================================================
# CLI / Main (Mejorado)
# ============================================================

def parse_pool_arg(arg: str) -> Dict[str, Any]:
    parts = arg.split(",")
    if len(parts) < 5:
        raise ValueError("Formato pool inválido (pool_id,url,port,user,password[,notls][,fp=sha256])")
    
    pool_id, url, port, user, password, *rest = parts
    notls = False
    tls_fp = None
    
    for r in rest:
        r = r.strip().lower()
        if r == "notls":
            notls = True
        elif r.startswith("fp="):
            tls_fp = r.split("=", 1)[1]
    
    return {
        "pool_id": pool_id,
        "url": url,
        "port": int(port),
        "user": user,
        "password": password,
        "notls": notls,
        "tls_fingerprint": tls_fp,
        "enabled": True
    }

def main():
    ap = argparse.ArgumentParser(description="IAZAR Multi-Pool Mining Proxy (RandomX integrado)")
    ap.add_argument("--listen-host", default="0.0.0.0")
    ap.add_argument("--listen-port", type=int, default=3333, help="Puerto plain (si --plain)")
    ap.add_argument("--listen-port-tls", type=int, default=0, help="Puerto TLS (0=off)")
    ap.add_argument("--plain", action="store_true", help="Habilitar puerto plain")
    ap.add_argument("--tls-cert", default=None, help="Ruta certificado TLS")
    ap.add_argument("--tls-key", default=None, help="Ruta clave privada TLS")
    ap.add_argument("--pool", action="append", default=[], help="pool_id,url,port,user,password[,notls][,fp=sha256]")
    ap.add_argument("--stats-interval", type=int, default=30, help="Intervalo de estadísticas en segundos")
    ap.add_argument("--solutions-capacity", type=int, default=32768, help="Capacidad del canal de soluciones")
    ap.add_argument("--no-forward", action="store_true", help="No reenviar shares al pool")
    ap.add_argument("--max-sessions", type=int, default=5000, help="Máximo de sesiones concurrentes")
    
    # RandomX
    ap.add_argument("--randomx", action="store_true", help="Activar validación real RandomX")
    ap.add_argument("--rx-sampling", type=float, default=1.0, help="Proporción de shares que se validan (0-1)")
    ap.add_argument("--rx-warmup", type=int, default=1000, help="Hashes dummy para precalentar cache/dataset")
    ap.add_argument("--rx-config-json", default=None, help="Path JSON config randomx (opcional)")
    
    args = ap.parse_args()

    pool_defs = []
    for p in args.pool:
        try:
            pool_defs.append(parse_pool_arg(p))
        except Exception as e:
            logger.error("Error parseando --pool %s: %s", p, e)
            sys.exit(1)

    if not pool_defs:
        logger.warning("No pools configurados (modo solo IA).")

    listen_port_plain = args.listen_port if args.plain else None
    listen_port_tls = args.listen_port_tls if args.listen_port_tls > 0 else None
    
    if listen_port_plain is None and listen_port_tls is None:
        logger.error("Debes habilitar al menos un puerto (plain o tls).")
        sys.exit(1)

    # RandomX ShareValidator (opcional)
    share_validator = None
    if args.randomx:
        if not _HAS_RANDOMX:
            logger.error("randomx_validator no disponible en el entorno.")
            sys.exit(1)
        
        rx_cfg = {}
        if args.rx_config_json and os.path.exists(args.rx_config_json):
            try:
                with open(args.rx_config_json, "r", encoding="utf-8") as f:
                    rx_cfg = json.load(f)
            except Exception as e:
                logger.warning("No se pudo cargar rx-config-json: %s", e)
        
        try:
            rx_validator = RandomXValidator(rx_cfg)
        except Exception as e:
            logger.error("Error inicializando RandomXValidator: %s", e)
            sys.exit(1)
        
        # Warmup mejorado
        if args.rx_warmup > 0:
            logger.info("RandomX warmup: %d hashes dummy...", args.rx_warmup)
            dummy_seed = b"\x00" * 32
            dummy_blob = bytearray(b"\x11" * 128)
            
            for i in range(args.rx_warmup):
                nn = i & 0xFFFFFFFF
                dummy_blob[39:43] = nn.to_bytes(4, "little")
                try:
                    rx_validator.hash(bytes(dummy_blob), dummy_seed)
                except Exception as e:
                    logger.error("Error en warmup RandomX: %s", e)
                    break
        
        share_validator = ShareValidator(rx_validator, sampling=args.rx_sampling)
        logger.info("Validación RandomX activada (sampling=%.2f)", args.rx_sampling)

    multipool = MultiPoolManager(pool_defs, solutions_capacity=args.solutions_capacity)
    
    server = ProxyServer(
        listen_host=args.listen_host,
        listen_port_plain=listen_port_plain,
        listen_port_tls=listen_port_tls,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        multipool=multipool,
        stats_interval=args.stats_interval,
        submit_forward=not args.no_forward,
        max_sessions=args.max_sessions,
        share_validator=share_validator
    )

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Servidor detenido por usuario")
    except Exception as e:
        logger.error("Error en servidor: %s", e)
    finally:
        server.stop()
        logger.info("Servidor completamente detenido")


def launch_job_injector():
    """
    Lanza monerod_to_shm_job.py como subproceso en segundo plano
    y reinicia si muere accidentalmente.
    """
    # Directorio seguro para logs
    log_dir = os.path.join(os.environ.get("TEMP", "C:\\zarturxia\\tmp"), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "job_injector.log")
    
    # Ruta del script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(base_dir, "monerod_to_shm_job.py")
    
    while True:
        try:
            logger.info("Iniciando job injector: %s", script)
            with open(log_file, "a") as log:
                p = subprocess.Popen(
                    [sys.executable, script],
                    stdout=log,
                    stderr=subprocess.STDOUT
                )
                exit_code = p.wait()
                if exit_code == 0:
                    logger.warning("Job injector terminó normalmente. Reiniciando en 5s...")
                else:
                    logger.error("Job injector terminó con código %d. Reiniciando en 10s...", exit_code)
                    time.sleep(10)
                time.sleep(5)
        except Exception as e:
            logger.error("Fallo arrancando job injector: %s. Reintento en 15s...", e)
            time.sleep(15)

if __name__ == "__main__":
    # Arranca el job injector en un hilo aparte SOLO SI no hay pools externas
    if len(sys.argv) == 1 or all("--pool" not in a for a in sys.argv):
        th = threading.Thread(target=launch_job_injector, daemon=True)
        th.start()
        logger.info("Hilo job injector iniciado")
    
    main()