from __future__ import annotations
"""
pool_connection.py
------------------
Gestión robusta de conexión Stratum-like (Monero) a un pool, obtención de jobs
y publicación en JobChannel (SHM) para consumo por Orchestrator / Generadores.

Características:
- Conexión TCP/TLS opcional.
- Reconexión exponencial con jitter.
- Ping keepalive basado en inactividad.
- Parsing incremental de JSON líneas.
- Validación y normalización de jobs.
- Modo 'dummy' (sin pool) si disabled.
- Multi-pool listo (cada instancia tiene pool_id y prefix).

Uso rápido:
    jc = JobChannel(prefix="5555")
    pc = PoolConnection(pool_id="p1",
                        job_channel=jc,
                        url="pool.hashvault.pro",
                        port=443,
                        user=wallet_address,
                        password="x",
                        tls=True)
    pc.start()
    ...
    pc.stop()

Protocolo esperado (subset):
  mining.subscribe
  mining.authorize / login
  mining.set_difficulty (opcional)
  mining.notify (job)
  mining.job (algunos pools)
"""
import os
import ssl
import json
import time
import math
import socket
import random
import threading
import logging
from typing import Optional, Dict, Any, List, Tuple

from .shm_channels import JobChannel

__all__ = ["PoolConnection", "PoolJob"]

logger = logging.getLogger("proxy.pool")

if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class PoolJob:
    """Estructura interna job normalizada."""
    __slots__ = ("job_id", "seed_hash", "blob_hex", "target_qword",
                 "height", "nonce_offset", "raw", "pool_difficulty")

    def __init__(self,
                 job_id: str,
                 seed_hash: str,
                 blob_hex: str,
                 target_qword: int,
                 height: int,
                 nonce_offset: int,
                 pool_difficulty: int,
                 raw: Dict[str, Any]):
        self.job_id = job_id
        self.seed_hash = seed_hash
        self.blob_hex = blob_hex
        self.target_qword = target_qword
        self.height = height
        self.nonce_offset = nonce_offset
        self.pool_difficulty = pool_difficulty
        self.raw = raw


class PoolConnection:
    """Conexión a un único pool remoto y publicación de jobs en Shared Memory."""

    DEFAULT_TIMEOUT = 15
    MAX_LINE = 65536
    PING_INTERVAL = 90
    INACTIVITY_PING = 60
    RECONNECT_MAX_DELAY = 120
    SUBSCRIBE_ID = 1

    def __init__(self,
                 pool_id: str,
                 job_channel: JobChannel,
                 url: str = "",
                 port: int = 0,
                 user: str = "",
                 password: str = "x",
                 tls: bool = True,
                 tls_fingerprint: Optional[str] = None,
                 enabled: bool = True,
                 nicehash: bool = False,
                 rig_id: Optional[str] = None,
                 keepalive: bool = True,
                 use_subscribe: bool = True,
                 retry_base_delay: float = 2.0,
                 max_reconnect_delay: float = RECONNECT_MAX_DELAY):
        self.pool_id = pool_id
        self.job_channel = job_channel
        self.url = url
        self.port = int(port)
        self.user = user
        self.password = password
        self.tls = tls
        self.tls_fingerprint = tls_fingerprint
        self.enabled = enabled and bool(url)
        self.nicehash = nicehash
        self.rig_id = rig_id
        self.keepalive = keepalive
        self.use_subscribe = use_subscribe
        self.retry_base_delay = retry_base_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._buffer = bytearray()
        self._next_id = 10
        self._authorized = False
        self._subscribed = False

        self._current_job: Optional[PoolJob] = None
        self._job_version_local = 0
        self._last_recv = time.time()
        self._last_ping = 0.0
        self._difficulty = 0
        self._reconnect_attempt = 0

        # Métricas internas
        self._jobs_received = 0
        self._jobs_published = 0
        self._last_error: Optional[str] = None

    # --------------- Public API ---------------
    def start(self):
        if not self.enabled:
            logger.info("Pool %s disabled (modo solo IA).", self.pool_id)
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"pool-{self.pool_id}", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2)

    def stats(self) -> Dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "enabled": self.enabled,
            "connected": bool(self._sock),
            "authorized": self._authorized,
            "jobs_received": self._jobs_received,
            "jobs_published": self._jobs_published,
            "difficulty": self._difficulty,
            "current_job_id": self._current_job.job_id if self._current_job else None,
            "last_error": self._last_error,
            "reconnect_attempt": self._reconnect_attempt
        }

    def current_job(self) -> Optional[Dict[str, Any]]:
        job = self._current_job
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "seed_hash": job.seed_hash,
            "blob_hex": job.blob_hex,
            "target_qword": job.target_qword,
            "height": job.height,
            "nonce_offset": job.nonce_offset
        }

    # --------------- Internal Loop ---------------
    def _run(self):
        while not self._stop.is_set():
            try:
                self._connect_and_loop()
            except Exception as e:
                self._last_error = str(e)
                logger.warning("Pool %s loop error: %s", self.pool_id, e, exc_info=True)
            self._cleanup_socket()
            if self._stop.is_set():
                break
            delay = min(self.max_reconnect_delay,
                        self.retry_base_delay * (2 ** min(self._reconnect_attempt, 6)))
            delay = delay * (0.7 + 0.6 * random.random())  # jitter
            self._reconnect_attempt += 1
            logger.info("Pool %s reconnect in %.1fs (attempt %d)", self.pool_id, delay, self._reconnect_attempt)
            time.sleep(delay)

    def _connect_and_loop(self):
        self._reconnect_attempt = 0
        self._open_socket()
        self._last_recv = time.time()
        self._handshake()

        while not self._stop.is_set():
            self._recv_lines()
            now = time.time()
            if now - self._last_recv > self.INACTIVITY_PING and now - self._last_ping > self.PING_INTERVAL:
                self._send_ping()
            time.sleep(0.01)

    # --------------- Networking ---------------
    def _open_socket(self):
        sock = socket.create_connection((self.url, self.port), timeout=self.DEFAULT_TIMEOUT)
        sock.settimeout(5)
        if self.tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            ssock = ctx.wrap_socket(sock, server_hostname=self.url)
            if self.tls_fingerprint:
                fp = ssock.getpeercert(binary_form=True)
                import hashlib
                h = hashlib.sha256(fp).hexdigest()
                if h.lower() != self.tls_fingerprint.lower():
                    raise RuntimeError(f"TLS fingerprint mismatch got={h} exp={self.tls_fingerprint}")
            self._sock = ssock
        else:
            self._sock = sock
        logger.info("Pool %s connected (tls=%s)", self.pool_id, self.tls)

    def _cleanup_socket(self):
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._authorized = False
        self._subscribed = False

    def _send_json(self, obj: Dict[str, Any]):
        data = (json.dumps(obj, separators=(',', ':')) + "\n").encode("utf-8")
        try:
            if self._sock:
                self._sock.sendall(data)
        except Exception as e:
            self._last_error = str(e)
            raise

    def _recv_lines(self):
        if not self._sock:
            return
        try:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("EOF")
            self._buffer.extend(chunk)
            self._last_recv = time.time()
        except socket.timeout:
            return
        except Exception as e:
            raise ConnectionError(f"recv error: {e}")

        # Procesar líneas completas
        while True:
            nl = self._buffer.find(b"\n")
            if nl == -1:
                if len(self._buffer) > self.MAX_LINE:
                    raise ConnectionError("line buffer overflow")
                break
            line = self._buffer[:nl].decode("utf-8", errors="replace").strip()
            del self._buffer[:nl+1]
            if line:
                self._handle_line(line)

    # --------------- Protocolo ---------------
    def _handshake(self):
        if self.use_subscribe:
            self._send_json({
                "id": self.SUBSCRIBE_ID,
                "method": "mining.subscribe",
                "params": {
                    "agent": f"iazar-proxy/{self.pool_id}",
                    "id": self.rig_id or "iazar"
                }
            })
        # Algunos pools requieren login directo
        self._send_json({
            "id": self._next_id_inc(),
            "method": "mining.authorize",
            "params": [self.user, self.password]
        })

    def _send_ping(self):
        self._last_ping = time.time()
        try:
            self._send_json({"id": self._next_id_inc(), "method": "mining.ping", "params": []})
        except Exception as e:
            logger.debug("Ping failed: %s", e)

    def _handle_line(self, line: str):
        try:
            msg = json.loads(line)
        except Exception:
            logger.debug("Malformed JSON line (ignored) pool=%s: %s", self.pool_id, line)
            return

        # Respuesta a authorize / subscribe
        if "id" in msg and msg.get("id") == self.SUBSCRIBE_ID:
            self._subscribed = True
        if msg.get("id") and isinstance(msg.get("result"), (bool, dict)):
            if msg.get("result") is True and not self._authorized:
                self._authorized = True

        method = msg.get("method")
        if method in ("mining.notify", "job", "mining.job"):
            params = msg.get("params") or msg.get("result") or {}
            self._process_job_message(params)
        elif method == "mining.set_difficulty":
            params = msg.get("params") or []
            if params:
                try:
                    self._difficulty = int(params[0])
                except Exception:
                    pass
        elif method == "mining.ping":
            # PONG improv
            pass
        # Otros métodos se ignoran

    def _process_job_message(self, params: Any):
        try:
            if isinstance(params, list):
                # Estructura tipo: [job_id, blob, target, seed_hash, height, ...]
                job_id = params[0]
                blob_hex = params[1]
                target_hex = params[2]
                seed_hash = params[3] if len(params) > 3 else ""
                height = int(params[4]) if len(params) > 4 else 0
            elif isinstance(params, dict):
                job_id = params.get("job_id") or params.get("id") or "unknown"
                blob_hex = params.get("blob") or params.get("blob_hex") or ""
                target_hex = params.get("target") or ""
                seed_hash = params.get("seed_hash") or params.get("seed") or ""
                height = int(params.get("height", 0))
            else:
                logger.debug("Job params formato desconocido: %s", params)
                return

            if not (blob_hex and seed_hash and target_hex):
                logger.debug("Job incompleto ignorado.")
                return

            # target en qword: pools envían un hex little endian de 16 chars? (adaptamos)
            try:
                # Interpretar los primeros 16 hex (64 bits) little-endian.
                target_qword = int(target_hex[0:16], 16)
            except Exception:
                target_qword = 0

            nonce_offset = self._guess_nonce_offset(blob_hex)

            pj = PoolJob(
                job_id=job_id,
                seed_hash=seed_hash,
                blob_hex=blob_hex,
                target_qword=target_qword,
                height=height,
                nonce_offset=nonce_offset,
                pool_difficulty=self._difficulty,
                raw={"params": params}
            )
            self._jobs_received += 1
            self._publish_job(pj)
        except Exception as e:
            logger.debug("Error procesando job: %s", e, exc_info=True)

    def _guess_nonce_offset(self, blob_hex: str) -> int:
        # Heurística: usar offset típico Monero (39) si longitud >= 80 bytes (160 hex).
        if len(blob_hex) >= 160:
            return 39
        return 0

    def _publish_job(self, pj: PoolJob):
        self._current_job = pj
        # Construir dict para JobChannel
        try:
            target_32b = pj.target_qword.to_bytes(32, "little", signed=False)
        except Exception:
            target_32b = b"\xff" * 32
        job_dict = {
            "blob": pj.blob_hex,
            "seed": pj.seed_hash,
            "job_id": pj.job_id,
            "target": target_32b,
            "height": pj.height,
            "nonce_offset": pj.nonce_offset
        }
        ver = self.job_channel.set_job(job_dict)
        self._job_version_local = ver
        self._jobs_published += 1
        logger.info("Pool %s job published id=%s height=%s ver=%d diff=%s",
                    self.pool_id, pj.job_id, pj.height, ver, self._difficulty)

    # --------------- Helpers ---------------
    def _next_id_inc(self) -> int:
        self._next_id += 1
        return self._next_id

# --------------- CLI Debug ---------------
if __name__ == "__main__":
    import argparse
    from .shm_channels import open_job_channel

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-id", default="p1")
    ap.add_argument("--prefix", default="5555")
    ap.add_argument("--url", default="pool.hashvault.pro")
    ap.add_argument("--port", type=int, default=443)
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", default="x")
    ap.add_argument("--no-tls", action="store_true")
    args = ap.parse_args()

    jc = open_job_channel(args.prefix)
    pc = PoolConnection(
        pool_id=args.pool_id,
        job_channel=jc,
        url=args.url,
        port=args.port,
        user=args.user,
        password=args.password,
        tls=not args.no_tls
    )
    pc.start()
    try:
        while True:
            time.sleep(5)
            print(pc.stats())
    except KeyboardInterrupt:
        pass
    finally:
        pc.stop()
