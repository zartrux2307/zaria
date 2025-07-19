import sys
import os
import ssl
import json
import time
import socket
import logging
import threading
import traceback
import struct
import multiprocessing.shared_memory as shm
import uuid
import heapq
from collections import defaultdict
from typing import Dict, Optional, Tuple

# === Configuración inicial ===
PREFIX = "5555"
BUFFER_SIZE = 2097152

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

share_stats = defaultdict(int)
start_time = time.time()

# --- Logger Configuration ---
logger = logging.getLogger("IA-Zar-Proxy")
logger.setLevel(logging.INFO)
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
handlers = [
    logging.StreamHandler(),
    logging.FileHandler(os.path.join(log_dir, 'proxy.log'), encoding='utf-8')
]
for handler in handlers:
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

JOB_STRUCT_SIZE = 180
SOLUTION_STRUCT_SIZE = 73
SHM_JOB_SIZE = JOB_STRUCT_SIZE + 1
SHM_SOLUTION_SIZE = SOLUTION_STRUCT_SIZE + 1

# === Auto-certificados (si no existen, se crean) ===
def ensure_certificates(cert_file, key_file):
    from pathlib import Path
    Path(cert_file).parent.mkdir(parents=True, exist_ok=True)
    Path(key_file).parent.mkdir(parents=True, exist_ok=True)
    if not Path(cert_file).exists() or not Path(key_file).exists():
        logger.info("No TLS cert/key found, creating self-signed...")
        if os.name == "nt":
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from datetime import datetime, timedelta

            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, u"ZZ"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"IA-Zar"),
                x509.NameAttribute(NameOID.COMMON_NAME, u"IA-ZarProxy"),
            ])
            cert = x509.CertificateBuilder().subject_name(subject
                ).issuer_name(subject
                ).public_key(key.public_key()
                ).serial_number(x509.random_serial_number()
                ).not_valid_before(datetime.utcnow()
                ).not_valid_after(datetime.utcnow() + timedelta(days=365)
                ).add_extension(x509.SubjectAlternativeName([x509.DNSName(u"localhost")]), False
                ).sign(key, hashes.SHA256())

            with open(cert_file, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            with open(key_file, "wb") as f:
                f.write(key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))
        else:
            cmd = (
                f'openssl req -new -newkey rsa:4096 -days 365 -nodes -x509 '
                f'-subj "/CN=IA-ZarProxy" '
                f'-keyout "{key_file}" -out "{cert_file}"'
            )
            os.system(cmd)

CERT_FILE = os.path.join(PROJECT_DIR, "iazar", "proxy", "cert.pem")
KEY_FILE = os.path.join(PROJECT_DIR, "iazar", "proxy", "cert_key.pem")
ensure_certificates(CERT_FILE, KEY_FILE)

# --- Shared Memory Management ---
class BinSharedMemoryManager:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.job_shm = None
        self.solution_shm = None
        self._initialize_shm()

    def _initialize_shm(self):
        job_shm_name = f"{self.prefix}_job"
        solution_shm_name = f"{self.prefix}_solution"
        try:
            self.job_shm = shm.SharedMemory(name=job_shm_name, create=True, size=SHM_JOB_SIZE)
            self.job_shm.buf[SHM_JOB_SIZE-1] = 0
        except FileExistsError:
            self.job_shm = shm.SharedMemory(name=job_shm_name)
        try:
            self.solution_shm = shm.SharedMemory(name=solution_shm_name, create=True, size=SHM_SOLUTION_SIZE)
            self.solution_shm.buf[SHM_SOLUTION_SIZE-1] = 0
        except FileExistsError:
            self.solution_shm = shm.SharedMemory(name=solution_shm_name)

    @staticmethod
    def serialize_job(job: Dict) -> bytes:
        try:
            blob_bytes = bytes.fromhex(job['blob'])[:84].ljust(84, b'\0')
            target_bytes = struct.pack('>Q', int(job['target'], 16))
            seed_bytes = bytes.fromhex(job['seed_hash'])[:32].ljust(32, b'\0')
            job_id_bytes = job['job_id'].encode('utf-8')[:36].ljust(36, b'\0')
            height_bytes = struct.pack('>I', job.get('height', 0))
            algo_bytes = job.get('algo', 'rx/0').encode('utf-8')[:16].ljust(16, b'\0')
            return blob_bytes + target_bytes + seed_bytes + job_id_bytes + height_bytes + algo_bytes
        except Exception as e:
            logger.error(f"Job serialization error: {str(e)}")
            return b''

    @staticmethod
    def deserialize_solution(data: bytes) -> Dict:
        try:
            return {
                'job_id': data[0:36].decode('utf-8').rstrip('\0'),
                'nonce': struct.unpack('>I', data[36:40])[0],
                'hash': data[40:72].hex(),
                'is_valid': bool(data[72])
            }
        except Exception as e:
            logger.error(f"Solution deserialization error: {str(e)}")
            return {}

    def set_job(self, job: Dict):
        if not self.job_shm:
            return
        while self.job_shm.buf[SHM_JOB_SIZE-1] == 1:
            time.sleep(0.001)
        job_data = self.serialize_job(job)
        if len(job_data) != JOB_STRUCT_SIZE:
            logger.error(f"Invalid job size: {len(job_data)} != {JOB_STRUCT_SIZE}")
            return
        self.job_shm.buf[:JOB_STRUCT_SIZE] = job_data
        self.job_shm.buf[SHM_JOB_SIZE-1] = 1
        share_stats['ai_solutions'] += 1

    def get_solution(self, timeout: float = 3.0) -> Optional[Dict]:
        if not self.solution_shm:
            return None
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            if self.solution_shm.buf[SHM_SOLUTION_SIZE-1] == 1:
                solution_data = bytes(self.solution_shm.buf[:SOLUTION_STRUCT_SIZE])
                self.solution_shm.buf[SHM_SOLUTION_SIZE-1] = 0
                solution = self.deserialize_solution(solution_data)
                if solution: share_stats['valid_ai_solutions'] += int(solution['is_valid'])
                return solution
            time.sleep(0.001)
        return None

    def close(self):
        if self.job_shm: self.job_shm.close()
        if self.solution_shm: self.solution_shm.close()

# --- Priority Job Distributor ---
class PriorityJobDistributor:
    def __init__(self, maxlen=200):
        self.queue: list[Tuple[int, float, Dict]] = []
        self.lock = threading.Lock()
        self.maxlen = maxlen

    def add_job(self, job: Dict, priority=0):
        with self.lock:
            if len(self.queue) >= self.maxlen:
                heapq.heappop(self.queue)
            heapq.heappush(self.queue, (-priority, time.time(), job))

    def get_job(self) -> Optional[Dict]:
        with self.lock:
            if self.queue:
                return heapq.heappop(self.queue)[2]
            return None

    def jobs_available(self) -> int:
        with self.lock:
            return len(self.queue)

# --- Miner Connection Object ---
class MinerConnection:
    def __init__(self, sock, addr, connection_id):
        self.sock = sock
        self.addr = addr
        self.id = connection_id
        self.worker_name = None
        self.subscribed = False
        self.authorized = False
        self.buffer = b""
        self.last_job_id = None
        self.sock.settimeout(5.0)
        self.active = True

    def send(self, message):
        try:
            if not isinstance(message, bytes):
                message = message.encode()
            self.sock.sendall(message + b"\n")
            return True
        except Exception as e:
            logger.error(f"Send error to {self.addr}: {e}")
            self.active = False
            return False

# --- IA-Zar Proxy Main ---
class IAZarProxy:
    def __init__(self, wallet, pool_host="127.0.0.1", pool_port=3333, 
                 listen_port=5555, miner_password="x", shm_prefix="5555",
                 tls_fingerprint=None):
        self.wallet = wallet
        self.pool_host = pool_host
        self.pool_port = pool_port
        self.listen_port = listen_port
        self.miner_password = miner_password
        self.conn = None
        self.last_job = None
        self.miner_connections = {}
        self.miner_connection_counter = 0
        self.message_id_counter = 0
        self.lock = threading.Lock()
        self.session_id = None
        self.pool_buffer = b""
        self.is_connected = False
        self.expected_fingerprint = tls_fingerprint
        self.connection_attempts = 0

        self.shm_manager = BinSharedMemoryManager(prefix=shm_prefix)
        self.job_buffer = PriorityJobDistributor(maxlen=1000)
        self.start_miners_listener()
        logger.info(f"Proxy started on port {listen_port}")

        threading.Thread(target=self.ai_orchestrator_listener, daemon=True).start()

    def next_msg_id(self):
        self.message_id_counter += 1
        return self.message_id_counter

    def connect_to_pool(self):
        self.connection_attempts += 1
        max_retries = 10
        for attempt in range(max_retries):
            try:
                self.pool_buffer = b""
                logger.info(f"Connecting to pool {self.pool_host}:{self.pool_port} (attempt {attempt+1}/{max_retries})")
                raw_sock = socket.create_connection(
                    (self.pool_host, self.pool_port), timeout=15
                )
                use_tls = self.pool_port in [443, 5555, 3333, 4444, 7777, 8888, 9999]
                if use_tls:
                    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    sock = context.wrap_socket(raw_sock, server_hostname=self.pool_host)
                    if self.expected_fingerprint:
                        cert = sock.getpeercert(binary_form=True)
                        import hashlib
                        fp = hashlib.sha256(cert).hexdigest()
                        fp_clean = self.expected_fingerprint.replace(':', '').lower()
                        if fp.lower() != fp_clean:
                            logger.error(f"TLS fingerprint mismatch! Expected: {fp_clean}, Got: {fp}")
                            sock.close()
                            raise ssl.SSLError("TLS fingerprint does not match!")
                else:
                    sock = raw_sock

                self.conn = sock
                self.conn.settimeout(15)
                login_msg = {
                    "id": self.next_msg_id(),
                    "jsonrpc": "2.0",
                    "method": "login",
                    "params": {
                        "login": self.wallet,
                        "pass": self.miner_password,
                        "agent": "IA-ZarProxy/1.0"
                    }
                }
                self._send_json(login_msg)
                response = self._recv_line(30)
                if not response:
                    raise ConnectionError("No login response")
                response_data = json.loads(response)
                if "result" in response_data:
                    self.session_id = response_data["result"].get("id")
                    logger.info(f"Connected to pool, session ID: {self.session_id}")
                    self.is_connected = True
                    self.connection_attempts = 0
                    return True
                else:
                    error = response_data.get('error', {}).get('message', 'Unknown error')
                    raise ConnectionError(f"Login failed: {error}")
            except Exception as e:
                logger.error(f"Connection attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        logger.critical("Failed to connect to pool")
        self.is_connected = False
        return False

    def _send_json(self, data):
        try:
            payload = (json.dumps(data) + "\n").encode('utf-8')
            self.conn.sendall(payload)
            share_stats['shares_submitted_to_pool'] += 1
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.reconnect_to_pool()
            return False

    def _recv_line(self, timeout=15) -> Optional[str]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if b"\n" in self.pool_buffer:
                line, self.pool_buffer = self.pool_buffer.split(b"\n", 1)
                try:
                    return line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    return line.decode('latin-1', errors='replace').strip()
            try:
                chunk = self.conn.recv(4096)
                if not chunk:
                    logger.warning("Connection closed by pool")
                    self.reconnect_to_pool()
                    return None
                self.pool_buffer += chunk
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                self.reconnect_to_pool()
                return None
        logger.warning(f"Receive timeout after {timeout} seconds, buffer: {self.pool_buffer[:100]!r}")
        return None

    def ai_orchestrator_listener(self):
        while True:
            try:
                if not self.miner_connections:
                    if self.last_job:
                        self.shm_manager.set_job(self.last_job)
                    solution = self.shm_manager.get_solution(timeout=1.0)
                    if solution and solution.get('is_valid'):
                        self.submit_ai_solution(solution)
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"[AI Orchestrator Listener] {e}")
                time.sleep(2)

    def start_miners_listener(self):
        def listener():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # TLS listener: allow up to 300k concurrent miners with TLS
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
            ssl_sock = context.wrap_socket(sock, server_side=True)
            
            ssl_sock.bind(('0.0.0.0', self.listen_port))
            ssl_sock.listen(300000)
            logger.info(f"Miner listener started on port {self.listen_port} (SSL only)")
            
            while True:
                client_sock = None
                addr = None
                try:
                    client_sock, addr = ssl_sock.accept()
                    client_sock.settimeout(5)
                    with self.lock:
                        self.miner_connection_counter += 1
                        miner_conn = MinerConnection(client_sock, addr, self.miner_connection_counter)
                        self.miner_connections[miner_conn.id] = miner_conn
                    threading.Thread(target=self.handle_miner, args=(miner_conn,), daemon=True).start()
                except ssl.SSLError as e:
                    # Filtra conexiones no SSL
                    if 'wrong version number' in str(e).lower():
                        if addr:
                            logger.debug(f"Rejected non-SSL connection from {addr[0]}:{addr[1]}")
                        else:
                            logger.debug("Rejected non-SSL connection from unknown address")
                    else:
                        logger.warning(f"SSL error from {addr if addr else 'unknown'}: {e}")
                    if client_sock:
                        try:
                            client_sock.close()
                        except:
                            pass
                except Exception as e:
                    logger.error(f"Error accepting connection: {e}")
                    if client_sock:
                        try:
                            client_sock.close()
                        except:
                            pass
        threading.Thread(target=listener, daemon=True).start()

    def handle_miner(self, miner_conn):
        logger.info(f"New miner connected: {miner_conn.addr}")
        try:
            while miner_conn.active:
                try:
                    data = miner_conn.sock.recv(4096)
                    if not data:
                        break
                    miner_conn.buffer += data
                    while b"\n" in miner_conn.buffer:
                        msg, miner_conn.buffer = miner_conn.buffer.split(b"\n", 1)
                        self.process_miner_message(miner_conn, msg)
                except socket.timeout:
                    continue
                except ssl.SSLError as e:
                    if 'timed out' not in str(e):
                        logger.error(f"SSL error with miner {miner_conn.addr}: {e}")
                    break
                except Exception as e:
                    logger.error(f"Handle miner error: {e}")
                    break
        finally:
            try:
                miner_conn.sock.close()
            except:
                pass
            with self.lock:
                if miner_conn.id in self.miner_connections:
                    del self.miner_connections[miner_conn.id]
                    logger.info(f"Miner disconnected: {miner_conn.addr}")

    def send_job_to_miner(self, miner_conn, job):
        notify_msg = {
            "id": None,
            "method": "mining.notify",
            "params": [
                job['job_id'],
                job['blob'],
                job['seed_hash'],
                "00000000",
                "00000000",
                "00000000",
                [],
                job['target'],
                "00000000",
                True
            ]
        }
        msg_str = json.dumps(notify_msg)
        if miner_conn.send(msg_str):
            share_stats['jobs_to_miners'] += 1

    def process_miner_message(self, miner_conn, msg_bytes):
        try:
            msg = json.loads(msg_bytes.decode("utf-8"))
            logger.debug(f"Received miner message: {msg}")

            if msg.get("method") in ("login", "mining.subscribe"):
                miner_conn.subscribed = True
                resp = {
                    "id": msg.get("id"),
                    "jsonrpc": "2.0",
                    "result": {
                        "status": "OK"
                    }
                }
                miner_conn.send(json.dumps(resp))

                # ✅ Enviar trabajo si ya lo tenemos
                if self.last_job:
                    self.send_job_to_miner(miner_conn, self.last_job)

            elif msg.get("method") == "mining.authorize":
                miner_conn.authorized = True
                resp = {"id": msg.get("id"), "jsonrpc": "2.0", "result": True}
                miner_conn.send(json.dumps(resp))

            elif msg.get("method") == "mining.submit":
                job_id = msg.get("params", {}).get("job_id", "")
                if self.last_job and job_id == self.last_job.get("job_id"):
                    resp = {"id": msg.get("id"), "jsonrpc": "2.0", "result": True}
                else:
                    resp = {
                        "id": msg.get("id"),
                        "jsonrpc": "2.0",
                        "error": {"code": -1, "message": "Invalid job_id"}
                    }
                miner_conn.send(json.dumps(resp))

            else:
                resp = {
                    "id": msg.get("id"),
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Unknown method"}
                }
                miner_conn.send(json.dumps(resp))

        except Exception as e:
            logger.error(f"Error processing miner message: {e}")

    def broadcast_job(self, job):
        if not job:
            return
        with self.lock:
            for conn_id, miner_conn in list(self.miner_connections.items()):
                if miner_conn.subscribed:
                    self.send_job_to_miner(miner_conn, job)

    def get_next_job(self, timeout=60):
        start = time.time()
        while time.time() - start < timeout:
            line = self._recv_line(timeout=timeout)
            if not line:
                logger.debug("No data received from pool, reconnecting...")
                self.is_connected = False
                return None
            try:
                msg = json.loads(line)
                job_data = None
                if "result" in msg and "job" in msg["result"]:
                    job_data = msg["result"]["job"]
                    job_data["target"] = msg["result"]["target"]
                    job_data["seed_hash"] = msg["result"].get("seed_hash", job_data.get("seed_hash", ""))
                    job_data["height"] = msg["result"].get("height", 0)
                    job_data["algo"] = msg["result"].get("algo", "rx/0")
                elif msg.get("method") == "mining.notify":
                    params = msg.get("params", [])
                    if len(params) < 8:
                        logger.warning(f"Invalid mining.notify params: {params}")
                        continue
                    job_data = {
                        "job_id": params[0],
                        "blob": params[1],
                        "seed_hash": params[2],
                        "target": params[7] if len(params) > 7 else "0",
                        "height": 0,
                        "algo": "rx/0",
                    }
                if job_data:
                    logger.info(f"Received new job from pool: {job_data['job_id']}")
                    return job_data
                else:
                    logger.debug(f"Non-job message from pool: {msg}")
            except Exception as e:
                logger.error(f"Error parsing pool job: {e} -- Raw: {line}")
        logger.warning("Timeout waiting for pool job")
        return None

    def run(self):
        logger.info("Starting proxy main loop")
        last_heartbeat = time.time()
        last_job_time = time.time()
        JOB_TIMEOUT = 300

        while True:
            try:
                if not self.is_connected:
                    if not self.connect_to_pool():
                        time.sleep(min(5 * self.connection_attempts, 60))
                        continue
                    else:
                        last_job_time = time.time()
                current_time = time.time()

                # Heartbeat cada 30 segundos
                if current_time - last_heartbeat > 30:
                    try:
                        self._send_json({"id": self.next_msg_id(), "method": "keepalive"})
                        last_heartbeat = current_time
                        logger.debug("Heartbeat sent to pool")
                    except Exception as e:
                        logger.error(f"Heartbeat failed: {e}")
                        self.is_connected = False
                        continue

                # Obtener nuevo trabajo
                job = self.get_next_job()
                if job:
                    last_job_time = current_time
                    self.last_job = job
                    self.broadcast_job(job)
                    self.job_buffer.add_job(job)

                # Compartir trabajo con IA
                if self.last_job:
                    self.shm_manager.set_job(self.last_job)

                # Procesar soluciones de IA
                solution = self.shm_manager.get_solution()
                if solution and solution.get('is_valid'):
                    logger.info(f"Submitting AI solution for job: {solution['job_id']}")
                    self.submit_ai_solution(solution)

                # Verificar timeout de trabajo
                if current_time - last_job_time > JOB_TIMEOUT:
                    logger.warning(f"No job received in {JOB_TIMEOUT} seconds, reconnecting...")
                    self.is_connected = False
                    last_job_time = current_time
                    continue

                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                traceback.print_exc()
                time.sleep(5)

    def submit_ai_solution(self, solution):
        submit_msg = {
            "id": self.next_msg_id(),
            "jsonrpc": "2.0",
            "method": "submit",
            "params": {
                "id": solution["job_id"],
                "job_id": solution["job_id"],
                "nonce": format(solution["nonce"], "08x"),
                "result": solution["hash"],
                "worker": "x"
            }
        }
        return self._send_json(submit_msg)

    def health_check(self):
        return {
            "status": "OK" if self.is_connected else "DISCONNECTED",
            "connections": len(self.miner_connections),
            "shm_jobs": self.shm_manager.job_shm.buf[-1] if self.shm_manager.job_shm else 0,
            "shm_solutions": self.shm_manager.solution_shm.buf[-1] if self.shm_manager.solution_shm else 0,
            "queue_load": self.job_buffer.jobs_available(),
            "uptime": time.time() - start_time
        }

    def __del__(self):
        self.shm_manager.close()

def print_usage():
    print("Usage:")
    print("  python -m src.iazar.proxy.ia_proxy_main <wallet_address> [pool_host] [pool_port] [shm_prefix] [tls_fingerprint]")
    print("Ejemplo:")
    print("  python -m src.iazar.proxy.ia_proxy_main 44crWF5... pool.hashvault.pro 443 5555 420c7850e09b7c0bdcf748a7da9eb3647daf8515718f36d9ccfdd6b9ff834b14")
    print("  (puedes omitir tls_fingerprint para probar sin TLS strict)")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_usage()
        sys.exit(1)

    wallet = sys.argv[1]
    pool_host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    pool_port = int(sys.argv[3]) if len(sys.argv) > 3 else 3333
    shm_prefix = sys.argv[4] if len(sys.argv) > 4 else "5555"
    tls_fingerprint = sys.argv[5] if len(sys.argv) > 5 else None

    logger.info(f"Launching IA-Zar Proxy with config:")
    logger.info(f"  Wallet:           {wallet}")
    logger.info(f"  Pool host:        {pool_host}")
    logger.info(f"  Pool port:        {pool_port}")
    logger.info(f"  SHM prefix:       {shm_prefix}")
    logger.info(f"  TLS fingerprint:  {tls_fingerprint if tls_fingerprint else '[none]'}")
    logger.info(f"  Listen port:      5555 (local miners)")

    try:
        proxy = IAZarProxy(
            wallet=wallet,
            pool_host=pool_host,
            pool_port=pool_port,
            listen_port=5555,
            miner_password="x",
            shm_prefix=shm_prefix,
            tls_fingerprint=tls_fingerprint,
        )
        proxy.run()
    except Exception as e:
        logger.critical(f"Fatal error on startup: {e}", exc_info=True)
        sys.exit(2)