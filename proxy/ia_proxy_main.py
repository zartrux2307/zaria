import sys
import os
import ssl
import json
import time
import socket
import logging
import threading
import traceback
import random
import hashlib
import struct
import multiprocessing.shared_memory as shm
import uuid
from typing import Dict, Optional
from collections import defaultdict

# === Configuración de memoria compartida ===
try:
    PREFIX = "5555"
    BUFFER_SIZE = 2097152
    print(f"[DEBUG] Usando configuración para p2pool: PREFIX={PREFIX}, BUFFER_SIZE={BUFFER_SIZE}")
except Exception as e:
    print(f"Error loading config: {str(e)}")
    PREFIX = "5555"
    BUFFER_SIZE = 2097152

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

share_stats = defaultdict(int)

def diagnostic_printer():
    while True:
        print(
            f"\n==== [IA-Zar DIAG] ===="
            f"\nJobs Pool→Proxy:         {share_stats['jobs_from_pool']}"
            f"\nJobs Proxy→Miners:       {share_stats['jobs_to_miners']}"
            f"\nShares Miner→Proxy:      {share_stats['shares_from_miners']}"
            f"\nSoluciones IA SHM:       {share_stats['ai_solutions']}"
            f"\nIA Solutions Válidas:    {share_stats['valid_ai_solutions']}"
            f"\nShares Proxy→Pool:       {share_stats['shares_submitted_to_pool']}"
            f"\nShares Pool Accepted:    {share_stats['shares_accepted_by_pool']}"
            f"\nShares Pool Rejected:    {share_stats['shares_rejected_by_pool']}"
            f"\n========================\n"
        )
        time.sleep(30)

threading.Thread(target=diagnostic_printer, daemon=True).start()

# --- Logger Configuration ---
logger = logging.getLogger("IA-Zar-Proxy")
logger.setLevel(logging.INFO)

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

handlers = [
    logging.StreamHandler(),
    logging.FileHandler(
        os.path.join(log_dir, 'proxy.log'),
        encoding='utf-8'
    )
]

for handler in handlers:
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

PROXY_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROXY_DIR)
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

JOB_STRUCT_SIZE = 180
SOLUTION_STRUCT_SIZE = 73
SHM_JOB_SIZE = JOB_STRUCT_SIZE + 1
SHM_SOLUTION_SIZE = SOLUTION_STRUCT_SIZE + 1

class BinSharedMemoryManager:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.job_shm = None
        self.solution_shm = None
        self._initialize_shm()

    def _initialize_shm(self):
        job_shm_name = f"{self.prefix}_job"
        solution_shm_name = f"{self.prefix}_solution"
        logger.info(f"Initializing shared memory: job='{job_shm_name}', solution='{solution_shm_name}'")
        try:
            self.job_shm = shm.SharedMemory(name=job_shm_name, create=True, size=SHM_JOB_SIZE)
            self.job_shm.buf[SHM_JOB_SIZE-1] = 0
            logger.info(f"Created job shared memory: {job_shm_name} ({SHM_JOB_SIZE} bytes)")
        except FileExistsError:
            self.job_shm = shm.SharedMemory(name=job_shm_name)
            logger.info(f"Connected to existing job shared memory: {job_shm_name}")
        try:
            self.solution_shm = shm.SharedMemory(name=solution_shm_name, create=True, size=SHM_SOLUTION_SIZE)
            self.solution_shm.buf[SHM_SOLUTION_SIZE-1] = 0
            logger.info(f"Created solution shared memory: {solution_shm_name} ({SHM_SOLUTION_SIZE} bytes)")
        except FileExistsError:
            self.solution_shm = shm.SharedMemory(name=solution_shm_name)
            logger.info(f"Connected to existing solution shared memory: {solution_shm_name}")

    @staticmethod
    def serialize_job(job: Dict) -> bytes:
        try:
            blob_bytes = bytes.fromhex(job['blob'])[:84]
            blob_bytes = blob_bytes.ljust(84, b'\0')
            target_bytes = struct.pack('>Q', int(job['target'], 16))
            seed_bytes = bytes.fromhex(job['seed_hash'])[:32]
            seed_bytes = seed_bytes.ljust(32, b'\0')
            job_id_bytes = job['job_id'].encode('utf-8')[:36]
            job_id_bytes = job_id_bytes.ljust(36, b'\0')
            height_bytes = struct.pack('>I', job.get('height', 0))
            algo_bytes = job.get('algo', 'rx/0').encode('utf-8')[:16]
            algo_bytes = algo_bytes.ljust(16, b'\0')
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
        logger.debug(f"Job sent to AI: {job['job_id']}")
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
                if solution:
                    logger.info(f"AI solution received: job_id={solution['job_id']}")
                    share_stats['valid_ai_solutions'] += 1 if solution['is_valid'] else 0
                return solution
            time.sleep(0.001)
        return None

    def close(self):
        if self.job_shm:
            self.job_shm.close()
        if self.solution_shm:
            self.solution_shm.close()

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

    def send(self, message):
        try:
            if not isinstance(message, bytes):
                message = message.encode()
            self.sock.sendall(message + b"\n")
            return True
        except Exception as e:
            logger.error(f"Send error to {self.addr}: {e}")
            return False

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
        self.expected_fingerprint = tls_fingerprint  # SHA256 fingerprint

        self.shm_manager = BinSharedMemoryManager(prefix=shm_prefix)
        self.connect_to_pool()
        self.start_miners_listener()
        logger.info(f"Proxy started on port {listen_port}")

    def next_msg_id(self):
        self.message_id_counter += 1
        return self.message_id_counter

    def _send_json(self, data):
        try:
            payload = (json.dumps(data) + "\n").encode('utf-8')
            self.conn.sendall(payload)
            logger.debug(f"Sent: {data}")
            share_stats['shares_submitted_to_pool'] += 1
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.reconnect_to_pool()
            return False

    def _recv_line(self, timeout=10) -> Optional[str]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if b"\n" in self.pool_buffer:
                line, self.pool_buffer = self.pool_buffer.split(b"\n", 1)
                try:
                    return line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    logger.warning("Non-UTF8 message received, trying latin-1")
                    try:
                        return line.decode('latin-1').strip()
                    except Exception:
                        logger.error("Error decoding message, using raw representation")
                        return str(line)
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
        return None

    def get_next_job(self):
        try:
            data = self._recv_line(timeout=1)
            if not data:
                return None
            if data.startswith("{"):
                try:
                    message = json.loads(data)
                    method = message.get("method")
                    if method == "job":
                        job = self.parse_job(message)
                        if job:
                            logger.info(f"Job received: {job['job_id']}")
                            share_stats['jobs_from_pool'] += 1
                            return job
                    elif method == "submit_result":
                        result = message.get("result")
                        if result and result.get("status") == "OK":
                            logger.info(f"Share accepted: {message.get('id')}")
                            share_stats['shares_accepted_by_pool'] += 1
                        else:
                            logger.warning(f"Share rejected: {message.get('error')}")
                            share_stats['shares_rejected_by_pool'] += 1
                    else:
                        logger.debug(f"Received non-job JSON message: {message}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {data[:100]}...")
            else:
                if "job" in data.lower() or "blob" in data.lower():
                    logger.warning(f"Suspicious message received: {data[:200]}")
                else:
                    logger.info(f"Non-JSON message: {data[:200]}...")
            return None
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Socket error: {e}")
            self.reconnect_to_pool()
            return None

    def connect_to_pool(self):
        """Establish connection to mining pool with TLS support and fingerprint check."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.pool_buffer = b""
                raw_sock = socket.create_connection(
                    (self.pool_host, self.pool_port),
                    timeout=10
                )
                use_tls = self.pool_port in [443, 5555, 3333, 4444, 7777, 8888, 9999]
                if use_tls:
                    context = ssl.create_default_context()
                    if self.expected_fingerprint:
                        def verify_fingerprint(sock, hostname):
                            cert = sock.getpeercert(binary_form=True)
                            import hashlib
                            fp = hashlib.sha256(cert).hexdigest()
                            fp_clean = self.expected_fingerprint.replace(':', '').lower()
                            if fp.lower() != fp_clean:
                                raise ssl.SSLError("TLS fingerprint does not match!")
                        context.check_hostname = False
                        context.verify_mode = ssl.CERT_NONE
                        sock = context.wrap_socket(raw_sock, server_hostname=self.pool_host)
                        verify_fingerprint(sock, self.pool_host)
                    else:
                        sock = context.wrap_socket(raw_sock, server_hostname=self.pool_host)
                else:
                    sock = raw_sock
                self.conn = sock
                self.conn.settimeout(10)
                login_msg = {
                    "id": self.next_msg_id(),
                    "jsonrpc": "2.0",
                    "method": "login",
                    "params": {
                        "login": self.wallet,
                        "pass": getattr(self, 'miner_password', 'x'),
                        "agent": "IA-ZarProxy/1.0"
                    }
                }
                self._send_json(login_msg)
                response = self._recv_line(15)
                if not response:
                    raise ConnectionError("No login response")
                response_data = json.loads(response)
                if "result" in response_data:
                    self.session_id = response_data["result"].get("id")
                    logger.info(f"Connected to pool, session ID: {self.session_id}")
                    self.is_connected = True
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

    def start_miners_listener(self):
        def listener():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', self.listen_port))
            sock.listen(100)
            logger.info(f"Miner listener started on port {self.listen_port}")
            while True:
                client_sock, addr = sock.accept()
                with self.lock:
                    self.miner_connection_counter += 1
                    miner_conn = MinerConnection(client_sock, addr, self.miner_connection_counter)
                    self.miner_connections[miner_conn.id] = miner_conn
                if addr[0] == "127.0.0.1":
                    logger.info(f"NonceOrchestrator connected: {addr}")
                threading.Thread(target=self.handle_miner, args=(miner_conn,), daemon=True).start()
        threading.Thread(target=listener, daemon=True).start()

    def handle_miner(self, miner_conn):
        logger.info(f"New miner connected: {miner_conn.addr}")
        try:
            while True:
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
                except Exception as e:
                    logger.error(f"Handle miner error: {e}")
                    break
        finally:
            miner_conn.sock.close()
            with self.lock:
                if miner_conn.id in self.miner_connections:
                    del self.miner_connections[miner_conn.id]
                    logger.info(f"Miner disconnected: {miner_conn.addr}")

    def process_miner_message(self, miner_conn, msg_bytes):
        try:
            msg = json.loads(msg_bytes.decode('utf-8', errors='replace'))
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {miner_conn.addr}: {msg_bytes[:100]}")
            return

        method = msg.get("method")
        params = msg.get("params")
        msg_id = msg.get("id")
        logger.debug(f"[{miner_conn.addr}] Method: {method}, ID: {msg_id}")

        if method == "mining.subscribe":
            response = {
                "id": msg_id,
                "result": [
                    [["mining.notify", random.randint(10000000, 99999999)], "08000002", 4],
                    "08000002"
                ],
                "error": None
            }
            miner_conn.send(json.dumps(response))
            miner_conn.subscribed = True
            logger.info(f"[{miner_conn.addr}] Subscribed")

        elif method == "mining.authorize":
            if not params or len(params) < 1:
                response = {"id": msg_id, "error": ["-1", "Invalid parameters", ""]}
            else:
                login = params[0]
                password = params[1] if len(params) > 1 else "x"
                if password != self.miner_password:
                    response = {"id": msg_id, "error": ["-1", "Invalid password", ""]}
                else:
                    miner_conn.worker_name = login
                    miner_conn.authorized = True
                    response = {"id": msg_id, "result": True, "error": None}
                    logger.info(f"[{miner_conn.addr}] Authorized as {login}")
            miner_conn.send(json.dumps(response))

        elif method == "login":
            if isinstance(params, dict):
                login = params.get("login")
                password = params.get("pass", "x")
            else:
                login = None
            if login:
                miner_conn.worker_name = login
                miner_conn.authorized = True
                miner_conn.subscribed = True
                # XMRig espera que la respuesta incluya un job válido
                job_dict = None
                if self.last_job:
                    job_dict = {
                        "job_id": self.last_job["job_id"],
                        "blob": self.last_job["blob"],
                        "target": self.last_job["target"],
                        "seed_hash": self.last_job["seed_hash"],
                        "height": self.last_job["height"],
                        "algo": self.last_job["algo"]
                    }
                response = {
                    "id": msg_id,
                    "jsonrpc": "2.0",
                    "result": {
                        "id": f"proxy_{miner_conn.id}",
                        "job": job_dict,
                        "status": "OK"
                    },
                    "error": None
                }
                miner_conn.send(json.dumps(response))
                logger.info(f"[{miner_conn.addr}] XMRig login autorizado como {login}")
                if self.last_job:
                    self.send_job_to_miner(miner_conn, self.last_job)
            else:
                error_msg = {"id": msg_id, "error": ["-1", "Missing login parameter", ""]}
                miner_conn.send(json.dumps(error_msg))

        elif method in ("mining.submit", "submit"):
            if not miner_conn.authorized:
                miner_conn.send(json.dumps({"id": msg_id, "error": ["-1", "Unauthorized", ""]}))
                return

            # Normalize both list- and dict-style params
            worker_name = None
            job_id = None
            nonce = None
            result_hash = None

            if isinstance(params, list) and len(params) >= 4:
                worker_name, job_id, nonce, result_hash = params[:4]
            elif isinstance(params, dict):
                worker_name = params.get("worker") or params.get("login")
                job_id      = params.get("job_id") or params.get("id")
                nonce       = params.get("nonce")
                result_hash = params.get("result")

            if not all([worker_name, job_id, nonce, result_hash]):
                miner_conn.send(json.dumps({"id": msg_id, "error": ["-1", "Invalid parameters", ""]}))
                return

            submit_msg = {
                "id": self.next_msg_id(),
                "jsonrpc": "2.0",
                "method": "submit",
                "params": {
                    "id": job_id,
                    "job_id": job_id,
                    "nonce": nonce,
                    "result": result_hash,
                    "worker": worker_name
                }
            }

            if self._send_json(submit_msg):
                miner_conn.send(json.dumps({"id": msg_id, "result": True, "error": None}))
                logger.info(f"Share submitted: {worker_name} job={job_id}")
                share_stats['shares_from_miners'] += 1
            else:
                miner_conn.send(json.dumps({"id": msg_id, "error": ["-1", "Proxy error", ""]}))

        elif method == "mining.configure":
            response = {
                "id": msg_id,
                "result": {
                    "version-rolling": True,
                    "version-rolling.mask": "1fffe000",
                    "version-rolling.min-bit-count": 16
                },
                "error": None
            }
            miner_conn.send(json.dumps(response))
            logger.debug(f"[{miner_conn.addr}] Configured")

        else:
            logger.warning(f"Unsupported method from {miner_conn.addr}: {method}")
            miner_conn.send(json.dumps({
                "id": msg_id,
                "error": ["-1", "Unsupported method", ""]
            }))

    def broadcast_job(self, job):
        if not job:
            return
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
        with self.lock:
            for conn_id, miner_conn in list(self.miner_connections.items()):
                if miner_conn.subscribed:
                    if not miner_conn.send(msg_str):
                        del self.miner_connections[conn_id]
                    else:
                        share_stats['jobs_to_miners'] += 1
                        logger.debug(f"Job sent to {miner_conn.addr}")

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
            logger.debug(f"Job sent to {miner_conn.addr}")

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

    def parse_job(self, message):
        try:
            params = message.get("params", {})
            return {
                "job_id": params.get("job_id") or str(uuid.uuid4()),
                "blob": params["blob"],
                "seed_hash": params["seed_hash"],
                "target": params["target"],
                "height": params.get("height", 0),
                "algo": "rx/0"
            }
        except Exception as e:
            logger.error(f"Job parsing error: {e}")
            return None

    def reconnect_to_pool(self):
        logger.warning("Reconnecting to pool...")
        self.is_connected = False
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            try:
                if self.conn:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                self.pool_buffer = b""
                if self.connect_to_pool():
                    logger.info("Reconnected successfully")
                    return True
            except Exception as e:
                logger.error(f"Reconnection error (attempt {attempts+1}): {e}")
                traceback.print_exc()
            wait_time = min(2 ** attempts, 30)
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            attempts += 1
        logger.critical("Failed to reconnect after multiple attempts")
        return False

    def run(self):
        logger.info("Starting proxy main loop")
        last_heartbeat = time.time()
        last_job_time = time.time()
        JOB_TIMEOUT = 120
        while True:
            try:
                if not self.is_connected:
                    if not self.reconnect_to_pool():
                        time.sleep(5)
                        continue
                    else:
                        last_job_time = time.time()
                current_time = time.time()
                if current_time - last_heartbeat > 30:
                    try:
                        self._send_json({"id": self.next_msg_id(), "method": "keepalive"})
                        last_heartbeat = current_time
                        logger.debug("Heartbeat sent")
                    except Exception as e:
                        logger.error(f"Heartbeat failed: {e}")
                        self.is_connected = False
                        continue
                job = self.get_next_job()
                if job:
                    last_job_time = current_time
                    self.last_job = job
                    logger.info(f"New job: {job['job_id']}")
                    self.broadcast_job(job)
                    self.shm_manager.set_job(job)
                solution = self.shm_manager.get_solution()
                if solution and solution.get('is_valid'):
                    logger.info(f"Valid AI solution: job={solution['job_id']}")
                    self.submit_ai_solution(solution)
                if current_time - last_job_time > JOB_TIMEOUT:
                    logger.warning(f"No job received in {JOB_TIMEOUT} seconds, reconnecting...")
                    self.is_connected = False
                    last_job_time = current_time
                    continue
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                traceback.print_exc()
                time.sleep(1)

    def __del__(self):
        self.shm_manager.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ia_proxy_main.py <wallet_address> [pool_host] [pool_port] [shm_prefix] [tls_fingerprint]")
        sys.exit(1)
    wallet = sys.argv[1]
    pool_host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    pool_port = int(sys.argv[3]) if len(sys.argv) > 3 else 3333
    shm_prefix = sys.argv[4] if len(sys.argv) > 4 else "5555"
    tls_fingerprint = sys.argv[5] if len(sys.argv) > 5 else None
    logger.info(f"""
    ======================================
    Starting IA-Zar Proxy for Monero/Pool
    Wallet: {wallet}
    Pool: {pool_host}:{pool_port}
    Miner Port: 5555
    SHM Prefix: {shm_prefix}
    TLS Fingerprint: {tls_fingerprint or 'None'}
    ======================================
    """)
    proxy = IAZarProxy(
        wallet,
        pool_host=pool_host,
        pool_port=pool_port,
        listen_port=5555,
        shm_prefix=shm_prefix,
        tls_fingerprint=tls_fingerprint
    )
    try:
        proxy.run()
    except KeyboardInterrupt:
        logger.info("Proxy stopped by user")
        sys.exit(0)
