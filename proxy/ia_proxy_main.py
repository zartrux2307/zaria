from __future__ import annotations
"""
ia_proxy_main.py (RandomX integrado)
------------------------------------
Proxy reactor multi-pool para coordinar mineros y la IA vía SHM, con
validación real de shares usando RandomX (opcional).

Novedades respecto a versión anterior:
- Integración RandomXValidator real (si --randomx).
- Clase ShareValidator para encapsular lógica hash + target check.
- Sampling de validación (--share-sampling) para reducir carga CPU.
- Métricas internas simples (contadores).
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
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return
    logger.warning("Generando certificado auto-firmado temporal (testing).")
    try:
        import subprocess
        subprocess.check_call([
            "openssl", "req", "-x509", "-nodes",
            "-newkey", "rsa:2048",
            "-keyout", key_path,
            "-out", cert_path,
            "-days", "365",
            "-subj", "/C=NA/ST=NA/L=NA/O=IAZAR/OU=DEV/CN=localhost"
        ])
    except Exception as e:
        logger.error("No se pudo generar certificado: %s", e)

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
# ShareValidator (RandomX)
# ============================================================

class ShareValidator:
    """
    Encapsula la validación de shares:
      - Inserta nonce en blob
      - Calcula hash RandomX
      - Compara hash con target
      - Verifica (opcional) hash declarado por el minero (result_hex)
    """
    def __init__(self, rx_validator: RandomXValidator, sampling: float = 1.0):
        self.rx = rx_validator
        self.sampling = max(0.0, min(1.0, sampling))
        self.random = random.Random(1337)

        # Contadores
        self.total_submits = 0
        self.validated = 0
        self.skipped = 0
        self.accepted = 0
        self.rejected = 0

    def validate_share(self, job: Dict[str, Any], nonce: int, result_hex: str | None) -> Dict[str, Any]:
        """
        Retorna dict:
          {
            "validated": bool,
            "accepted": bool,
            "error": str|None,
            "hash_hex": str or None
          }
        """
        self.total_submits += 1

        # Sampling decision
        do_validate = True
        if self.sampling < 0.999:
            if self.random.random() > self.sampling:
                do_validate = False

        if not do_validate:
            self.skipped += 1
            # Aceptación optimista (ya que no validamos) pero se puede marcar 'validated': False
            return {"validated": False, "accepted": True, "error": None, "hash_hex": None}

        self.validated += 1

        try:
            blob = job.get("blob")
            if blob is None:
                return {"validated": True, "accepted": False, "error": "no_blob", "hash_hex": None}
            blob_bytes = _decode_hex_maybe(blob)
            seed_bytes = _decode_hex_maybe(job.get("seed"))
            target = job.get("target")
            target_bytes = _decode_hex_maybe(target)
            nonce_offset = int(job.get("nonce_offset", 0))

            if nonce_offset < 0 or nonce_offset + 4 > len(blob_bytes):
                return {"validated": True, "accepted": False, "error": "bad_nonce_offset", "hash_hex": None}

            # Inserta nonce (little endian)
            mutable = bytearray(blob_bytes)
            mutable[nonce_offset:nonce_offset+4] = nonce.to_bytes(4, "little", signed=False)

            # Hash RandomX
            # Asumimos RandomXValidator posee método .hash(data, seed) -> bytes32
            h = self.rx.hash(bytes(mutable), seed_bytes)
            if not isinstance(h, (bytes, bytearray)) or len(h) != 32:
                return {"validated": True, "accepted": False, "error": "rx_hash_fail", "hash_hex": None}

            # Comparar target
            # Interpretamos target_bytes little-endian (Monero)
            if len(target_bytes) < 32:
                target_bytes = target_bytes + b"\x00" * (32 - len(target_bytes))
            elif len(target_bytes) > 32:
                target_bytes = target_bytes[:32]

            hash_val = int.from_bytes(h, "little")
            target_val = int.from_bytes(target_bytes, "little")
            if target_val == 0:
                # Evitar división por cero / target inválido
                target_val = 1

            meets = hash_val < target_val

            # (Opcional) Comprobar coincidencia con result_hex que envía minero
            if result_hex and len(result_hex) >= 64:
                miner_h_first = result_hex[:64].lower()
                our_first = h.hex()[:64]
                if miner_h_first != our_first:
                    # Si no coincide, la share se invalida
                    self.rejected += 1
                    return {"validated": True, "accepted": False, "error": "hash_mismatch", "hash_hex": h.hex()}

            if meets:
                self.accepted += 1
                return {"validated": True, "accepted": True, "error": None, "hash_hex": h.hex()}
            else:
                self.rejected += 1
                return {"validated": True, "accepted": False, "error": "low_diff", "hash_hex": h.hex()}

        except Exception as e:
            self.rejected += 1
            return {"validated": True, "accepted": False, "error": f"exception:{e}", "hash_hex": None}

    def stats(self) -> Dict[str, Any]:
        return {
            "total_submits": self.total_submits,
            "validated": self.validated,
            "sampling": self.sampling,
            "skipped": self.skipped,
            "accepted": self.accepted,
            "rejected": self.rejected
        }

# ============================================================
# MultiPoolManager (sin cambios de lógica principal)
# ============================================================

class MultiPoolManager:
    def __init__(self, pool_defs: List[Dict[str, Any]], solutions_capacity: int):
        self.pools: Dict[str, Dict[str, Any]] = {}
        self.solutions_capacity = solutions_capacity
        for p in pool_defs:
            pool_id = p["pool_id"]
            prefix = p.get("prefix") or pool_id
            job_ch = open_job_channel(prefix)
            sol_ch = open_solution_channel(prefix, capacity=solutions_capacity)
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
        # Placeholder: no reenvío implementado (igual que antes).
        return 0

    def stats(self) -> Dict[str, Any]:
        out = {}
        for pid, obj in self.pools.items():
            out[pid] = obj["conn"].stats()
        return out

# ============================================================
# ProxyServer
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

        # Métricas básicas
        self.total_submits = 0
        self.accepted_submits = 0
        self.rejected_submits = 0
        self.sampled_out = 0

    def _prepare_tls_context(self):
        if not (self.tls_cert and self.tls_key):
            cert_path = "proxy_cert.pem"
            key_path = "proxy_key.pem"
            generate_self_signed_cert(cert_path, key_path)
            self.tls_cert = cert_path
            self.tls_key = key_path
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.load_cert_chain(certfile=self.tls_cert, keyfile=self.tls_key)
        self._tls_context = ctx

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
        if self.listen_port_tls is not None:
            self._lsock_tls = self._open_listener_socket(self.listen_port_tls, use_tls=True)

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
            logger.info("Stopping (KeyboardInterrupt)")
        except Exception as e:
            logger.error("Main loop error: %s", e, exc_info=True)
        finally:
            self.stop()

    def _accept_new(self, is_tls: bool, sockobj):
        try:
            conn, addr = sockobj.accept()
            if is_tls:
                try:
                    conn = self._tls_context.wrap_socket(conn, server_side=True)
                except Exception as e:
                    logger.debug("TLS handshake failed from %s: %s", addr, e)
                    conn.close()
                    return
            if len(self.sessions) >= self.max_sessions:
                conn.close()
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
            logger.info("Miner connected id=%d addr=%s tls=%s", sess.id, addr, is_tls)
        except Exception as e:
            logger.debug("Accept failed: %s", e, exc_info=True)

    def _drop_session(self, sess: MinerSession):
        try:
            self.selector.unregister(sess.sock)
        except Exception:
            pass
        self.sessions.pop(sess.id, None)

    def _update_selector_interest(self, sess: MinerSession):
        events = selectors.EVENT_READ
        if sess.want_write():
            events |= selectors.EVENT_WRITE
        try:
            self.selector.modify(sess.sock, events, data=sess)
        except Exception:
            pass

    def _housekeeping(self, now: float):
        updates = self.multipool.poll_new_jobs()
        if updates:
            for pid, job, ver in updates:
                for s in list(self.sessions.values()):
                    s.push_job(job, ver)

        for s in list(self.sessions.values()):
            if s.closed:
                self._drop_session(s)
                continue
            s.tick(now)
            if len(s.outbuf) > self.max_out_buffer:
                s.close("output_overflow")
                self._drop_session(s)

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
                hbytes = bytes.fromhex(hash_hex[:64]) if hash_hex else (bytes.fromhex(result_hex[:64]) if result_hex and len(result_hex) >= 64 else b"\x00" * 32)
                sol_ch.try_submit(
                    job_version=ver,
                    nonce=nonce_int,
                    hash_bytes=hbytes,
                    valid=True,
                    job_id=job_id,
                    pool_id=pid
                )
                # Forward al pool remoto
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
                        logger.debug("Forward submit fail: %s", e)
            self.accepted_submits += 1
            return {"result": True}
        else:
            self.rejected_submits += 1
            return {"result": False, "error": error or "Rejected"}

    def _on_disconnect_session(self, session: MinerSession):
        pass

    def _metrics_inc_stub(self, name: str, value: int = 1):
        pass

# ============================================================
# CLI / Main
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
    ap = argparse.ArgumentParser(description="IAZAR Multi-Pool Mining Proxy (RandomX integrated)")
    ap.add_argument("--listen-host", default="0.0.0.0")
    ap.add_argument("--listen-port", type=int, default=3333, help="Puerto plain (si --plain)")
    ap.add_argument("--listen-port-tls", type=int, default=0, help="Puerto TLS (0=off)")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--tls-cert", default=None)
    ap.add_argument("--tls-key", default=None)
    ap.add_argument("--pool", action="append", default=[], help="pool_id,url,port,user,password[,notls][,fp=sha256]")
    ap.add_argument("--stats-interval", type=int, default=30)
    ap.add_argument("--solutions-capacity", type=int, default=8192)
    ap.add_argument("--no-forward", action="store_true", help="No reenviar shares al pool")
    ap.add_argument("--max-sessions", type=int, default=5000)
    # RandomX
    ap.add_argument("--randomx", action="store_true", help="Activar validación real RandomX")
    ap.add_argument("--rx-sampling", type=float, default=1.0, help="Proporción de shares que se validan (0-1)")
    ap.add_argument("--rx-warmup", type=int, default=0, help="Hashes dummy para precalentar cache/dataset")
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
        rx_validator = RandomXValidator(rx_cfg)
        # Warmup
        if args.rx_warmup > 0:
            logger.info("RandomX warmup: %d hashes dummy...", args.rx_warmup)
            dummy_seed = b"\x00" * 32
            dummy_blob = bytearray(b"\x11" * 128)
            for i in range(args.rx_warmup):
                nn = i & 0xFFFFFFFF
                dummy_blob[39:43] = nn.to_bytes(4, "little")
                try:
                    rx_validator.hash(bytes(dummy_blob), dummy_seed)
                except Exception:
                    break
        share_validator = ShareValidator(rx_validator, sampling=args.rx_sampling)

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
    finally:
        server.stop()


def launch_job_injector():
    """
    Lanza monerod_to_shm_job.py como subproceso en segundo plano
    y reinicia si muere accidentalmente.
    """
    script = os.path.join(os.path.dirname(__file__), "monerod_to_shm_job.py")
    while True:
        try:
            logger.info("Arrancando job injector: %s", script)
            p = subprocess.Popen([sys.executable, script])
            p.wait()
            logger.warning("Job injector terminó (código %s). Reiniciando en 5s...", p.returncode)
            time.sleep(5)
        except Exception as e:
            logger.error("Fallo arrancando job injector: %s", e)
            time.sleep(10)

if __name__ == "__main__":
    # Arranca el job injector en un hilo aparte SOLO SI no hay pools externas
    if len(sys.argv) == 1 or all("--pool" not in a for a in sys.argv):
        th = threading.Thread(target=launch_job_injector, daemon=True)
        th.start()
    main()