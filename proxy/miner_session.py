from __future__ import annotations
"""
miner_session.py
----------------
Sesión de un minero gestionada por un reactor (selectors). No crea threads propios.

API:
  session = MinerSession(sock, addr, server_callbacks=callbacks_dict)
  session.fileno() -> para registrar en selector
  session.on_readable() / session.on_writable() -> invocadas por loop
  session.push_job(job_dict, job_version) -> cola JSON mining.notify
  session.close()

Callbacks esperados en server_callbacks:
  {
     "on_submit": callable(job_id, nonce_int, result_hex, session) -> dict(result=True/False,error=None),
     "current_job_provider": callable() -> (job_dict, version) | (None, None),
     "on_disconnect": callable(session),
     "metrics_inc": callable(metric_name, value=1),
  }

NOTAS:
- Validación completa del share se delega al servidor (que puede usar RandomX).
- Rate limiting simple configurable.
"""

import json
import time
import selectors
import logging
from typing import Optional, Dict, Any, Tuple, Callable
import socket
import math

logger = logging.getLogger("proxy.session")

MAX_LINE_BYTES = 64 * 1024
MAX_BUFFER_BYTES = 512 * 1024
WRITE_CHUNK = 16384
SUBMIT_RATE_PER_SEC = 20
STRIKE_LIMIT = 5

# ---------------- Token Bucket simple ----------------
class _TokenBucket:
    __slots__ = ("rate", "capacity", "tokens", "ts")

    def __init__(self, rate: float, capacity: Optional[float] = None):
        self.rate = rate
        self.capacity = capacity or rate
        self.tokens = self.capacity
        self.ts = time.perf_counter()

    def consume(self, n: float = 1.0) -> bool:
        now = time.perf_counter()
        delta = now - self.ts
        self.ts = now
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

# ---------------- Miner Session ----------------
class MinerSession:
    __slots__ = (
        "sock", "addr", "id", "server_cb",
        "inbuf", "outbuf",
        "authorized", "subscribed", "worker_name",
        "last_recv", "last_send", "last_job_version",
        "submit_bucket", "strikes",
        "valid_submits", "invalid_submits",
        "selector_key", "closed",
        "extra_nonce", "pending_close_reason",
        "protocol_features"
    )

    _ID_SEQ = 0

    def __init__(self,
                 sock: socket.socket,
                 addr,
                 server_callbacks: Dict[str, Callable],
                 initial_job: Optional[Tuple[Dict[str, Any], int]] = None):
        MinerSession._ID_SEQ += 1
        self.id = MinerSession._ID_SEQ
        self.sock = sock
        self.addr = addr
        self.server_cb = server_callbacks
        self.inbuf = bytearray()
        self.outbuf = bytearray()
        self.authorized = False
        self.subscribed = False
        self.worker_name = f"miner-{self.id}"
        self.last_recv = time.time()
        self.last_send = time.time()
        self.last_job_version = -1
        self.submit_bucket = _TokenBucket(SUBMIT_RATE_PER_SEC, SUBMIT_RATE_PER_SEC)
        self.strikes = 0
        self.valid_submits = 0
        self.invalid_submits = 0
        self.selector_key = None
        self.closed = False
        self.extra_nonce = 0
        self.pending_close_reason = None
        self.protocol_features: Dict[str, Any] = {}

        if initial_job and initial_job[0]:
            job_dict, ver = initial_job
            self._queue_job_notify(job_dict, ver)

        self.sock.setblocking(False)

    # -------- Public --------
    def fileno(self):
        return self.sock.fileno()

    def on_readable(self):
        if self.closed:
            return
        try:
            chunk = self.sock.recv(4096)
        except BlockingIOError:
            return
        except Exception as e:
            self.close(f"read error {e}")
            return
        if not chunk:
            self.close("peer closed")
            return
        self.inbuf.extend(chunk)
        self.last_recv = time.time()
        if len(self.inbuf) > MAX_BUFFER_BYTES:
            self.close("input buffer overflow")
            return
        self._process_incoming_lines()

    def on_writable(self):
        if self.closed or not self.outbuf:
            return
        try:
            sent = self.sock.send(self.outbuf[:WRITE_CHUNK])
            if sent > 0:
                del self.outbuf[:sent]
                self.last_send = time.time()
        except BlockingIOError:
            return
        except Exception as e:
            self.close(f"write error {e}")
            return

    def want_write(self) -> bool:
        return bool(self.outbuf)

    def push_job(self, job_dict: Dict[str, Any], job_version: int):
        if job_version <= self.last_job_version:
            return
        self._queue_job_notify(job_dict, job_version)

    def close(self, reason: str = "close"):
        if self.closed:
            return
        self.closed = True
        self.pending_close_reason = reason
        try:
            self.sock.close()
        except Exception:
            pass
        cb = self.server_cb.get("on_disconnect")
        if cb:
            try:
                cb(self)
            except Exception:
                pass
        logger.info("Session %s closed (%s)", self.addr, reason)

    def stats(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "addr": f"{self.addr[0]}:{self.addr[1]}" if self.addr else None,
            "authorized": self.authorized,
            "subscribed": self.subscribed,
            "valid_submits": self.valid_submits,
            "invalid_submits": self.invalid_submits,
            "strikes": self.strikes,
            "last_job_version": self.last_job_version,
            "want_write": self.want_write()
        }

    # -------- Internal Processing --------
    def _process_incoming_lines(self):
        while True:
            nl = self.inbuf.find(b"\n")
            if nl == -1:
                break
            line = self.inbuf[:nl].decode("utf-8", errors="replace").strip()
            del self.inbuf[:nl+1]
            if line:
                if len(line) > MAX_LINE_BYTES:
                    self.close("line too long")
                    return
                self._handle_json_line(line)

    def _handle_json_line(self, line: str):
        try:
            msg = json.loads(line)
        except Exception:
            self._send_error(None, "Parse error")
            self._strike("bad_json")
            return
        if not isinstance(msg, dict):
            self._send_error(None, "Invalid message")
            self._strike("bad_msg")
            return

        mid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params")

        if method == "mining.subscribe":
            self.subscribed = True
            self.protocol_features = params if isinstance(params, dict) else {}
            self._send_result(mid, {"status": "OK"})
        elif method in ("mining.authorize", "login"):
            # params: [user, pass, rig_id?] o dict
            user = None
            if isinstance(params, list) and params:
                user = params[0]
            elif isinstance(params, dict):
                user = params.get("login")
            self.worker_name = user or self.worker_name
            self.authorized = True
            self._send_result(mid, {"status": "OK"})
            # Enviar job actual inmediatamente
            self._maybe_push_current_job()
        elif method == "mining.submit":
            self._handle_submit(mid, params)
        elif method in ("mining.keepalived", "mining.ping"):
            self._send_result(mid, True)
        else:
            self._send_error(mid, "Unknown method")

    def _maybe_push_current_job(self):
        cb = self.server_cb.get("current_job_provider")
        if cb:
            try:
                job, ver = cb()
                if job and ver is not None:
                    self.push_job(job, ver)
            except Exception:
                logger.debug("current_job_provider failed", exc_info=True)

    def _handle_submit(self, mid, params):
        if not self.authorized:
            self._send_error(mid, "Not authorized")
            self._strike("unauth_submit")
            return
        if not self.submit_bucket.consume():
            self._send_error(mid, "Rate limited")
            self._strike("rate")
            return

        # XMRig typical: params = [worker, job_id, nonce_hex, result_hex]
        job_id = None
        nonce_hex = None
        result_hex = None
        if isinstance(params, list) and len(params) >= 4:
            job_id = params[1]
            nonce_hex = params[2]
            result_hex = params[3]
        elif isinstance(params, dict):
            job_id = params.get("job_id")
            nonce_hex = params.get("nonce")
            result_hex = params.get("result")

        try:
            nonce_int = int(nonce_hex, 16) if nonce_hex else 0
        except Exception:
            self._send_error(mid, "Bad nonce")
            self._strike("bad_nonce")
            return

        cb = self.server_cb.get("on_submit")
        if not cb:
            self._send_error(mid, "Submit disabled")
            self._strike("nosubmit")
            return

        try:
            resp = cb(job_id, nonce_int, result_hex, self)
        except Exception as e:
            logger.debug("Submit callback exception: %s", e, exc_info=True)
            self._send_error(mid, "Internal error")
            self._strike("cb_exc")
            return

        ok = bool(resp.get("result")) if isinstance(resp, dict) else False
        if ok:
            self.valid_submits += 1
            self._send_result(mid, True)
        else:
            self.invalid_submits += 1
            self._send_error(mid, resp.get("error") if isinstance(resp, dict) else "Rejected")
            self._strike("invalid_share")

    def _strike(self, reason: str):
        self.strikes += 1
        if self.strikes >= STRIKE_LIMIT:
            self.close(f"too_many_strikes:{reason}")

    # -------- Envío JSON --------
    def _send_result(self, mid, result):
        msg = {"id": mid, "jsonrpc": "2.0", "result": result}
        self._queue_json(msg)

    def _send_error(self, mid, err: str):
        msg = {"id": mid, "jsonrpc": "2.0", "error": {"code": -1, "message": err}}
        self._queue_json(msg)

    def _queue_json(self, obj: Dict[str, Any]):
        try:
            data = (json.dumps(obj, separators=(',', ':')) + "\n").encode("utf-8")
            self.outbuf.extend(data)
        except Exception:
            pass

    def _queue_job_notify(self, job: Dict[str, Any], version: int):
        """
        job: dict con campos (blob, seed, job_id, target, height, nonce_offset)
        Se envía como 'mining.notify' con subset de campos estándar.
        """
        self.last_job_version = version
        notify = {
            "jsonrpc": "2.0",
            "method": "mining.notify",
            "params": [
                job.get("job_id", "job"),
                job.get("blob_hex") or (job.get("blob").hex() if isinstance(job.get("blob"), bytes) else job.get("blob")),
                job.get("target_hex") or (job.get("target").hex() if isinstance(job.get("target"), bytes) else ""),
                job.get("seed_hash") or (job.get("seed").hex() if isinstance(job.get("seed"), bytes) else ""),
                job.get("height", 0)
            ]
        }
        self._queue_json(notify)

    # -------- Housekeeping --------
    def tick(self, now: float):
        """
        Llamada periódica por el servidor (ej. cada seg) para housekeeping:
         - Reenviar job si hace falta
         - Cerrar por inactividad excesiva (opcional)
        """
        # Podrías añadir cierre por inactividad o ping aquí
        pass
