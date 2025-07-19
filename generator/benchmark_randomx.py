import os
import sys
import time
import random
import binascii
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
except ImportError:
    np = None

# Ajustar path para imports locales
sys.path.append(os.path.dirname(__file__))

from randomx_wrapper import RandomXManager, get_randomx_flags

# ========================
# Configuración General
# ========================
SEEDS = [f"{i:0>64}" for i in range(1, 9)]   # 8 semillas realistas tipo Monero
THREADS = int(os.environ.get("BENCH_THREADS", 4))
BATCH_SIZE = int(os.environ.get("BENCH_BATCH", 2000))
POOL_SIZE = int(os.environ.get("BENCH_POOL", 8))
RUNTIME = int(os.environ.get("BENCH_TIME", 10))   # Tiempo de prueba en segundos
STRESS_MODE = bool(int(os.environ.get("STRESS_MODE", 0)))

def random_blob() -> bytes:
    """
    Genera un blob aleatorio tipo mining (76 bytes, header simulado).
    """
    arr = bytearray(os.urandom(76))
    arr[0] = random.randint(1, 6)   # versión
    arr[35:43] = os.urandom(8)      # campo nonce mining
    return bytes(arr)

def hash_validation(hash_bytes: bytes) -> bool:
    """
    Verifica que el hash tenga formato correcto (32 bytes).
    """
    return isinstance(hash_bytes, bytes) and len(hash_bytes) == 32

def worker_bench(thread_id: int, hashes_per_thread: int, result_list: list, latency_list: list, manager: RandomXManager, seeds: list):
    """
    Worker de benchmarking. Ejecuta hashes y mide latencia.
    """
    count = 0
    latencies = []
    for i in range(hashes_per_thread):
        seed = random.choice(seeds)
        blob = random_blob()
        t0 = time.perf_counter_ns()
        hash_bytes = manager.compute_randomx_hash(blob, seed)
        dt = (time.perf_counter_ns() - t0) / 1e6  # ms
        latencies.append(dt)
        assert hash_validation(hash_bytes), "Hash inválido"
        count += 1
        if i % 250 == 0 and i > 0:
            print(f"[Thread-{thread_id}] {i}/{hashes_per_thread} hashes...")
    result_list[thread_id] = count
    latency_list[thread_id] = latencies

def show_latency_report(all_latencies):
    """
    Reporta estadísticas detalladas de latencia (ms).
    """
    flat = [x for lst in all_latencies if lst for x in lst]
    if not flat:
        print("[BENCH] No latency data collected")
        return
    if np:
        arr = np.array(flat)
        print(f"  Latencia (ms) ->  mean: {arr.mean():.2f} | median: {np.median(arr):.2f} | p95: {np.percentile(arr,95):.2f} | p99: {np.percentile(arr,99):.2f} | max: {arr.max():.2f}")
    else:
        flat.sort()
        print(f"  Latencia (ms) ->  mean: {statistics.mean(flat):.2f} | median: {statistics.median(flat):.2f} | p95: {flat[int(0.95*len(flat))]:.2f} | max: {max(flat):.2f}")

def main():
    print(f"[BENCH] RandomX Benchmark | Threads={THREADS} | Pool={POOL_SIZE} | Batch={BATCH_SIZE} | Seeds={len(SEEDS)} | Stress={STRESS_MODE}")
    manager = RandomXManager(max_pool_per_seed=POOL_SIZE, flags=get_randomx_flags())

    # =====================
    # Calentamiento Pool/Cache
    # =====================
    print("[BENCH] Calentando pool y caches RandomX ...")
    for i in range(min(4, len(SEEDS))):
        for _ in range(4):
            manager.compute_randomx_hash(random_blob(), SEEDS[i])
    time.sleep(1)
    print("[BENCH] Calentamiento OK.\n")

    hashes_per_thread = BATCH_SIZE
    total_hashes = THREADS * hashes_per_thread

    results = [0] * THREADS
    latencies = [None] * THREADS
    threads = []

    # =====================
    # Benchmark
    # =====================
    print(f"[BENCH] Ejecutando prueba real ({total_hashes} hashes, ~{RUNTIME}s)...")
    t_start = time.time()

    for i in range(THREADS):
        t = threading.Thread(
            target=worker_bench,
            args=(i, hashes_per_thread, results, latencies, manager, SEEDS)
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    t_end = time.time()
    elapsed = t_end - t_start
    total = sum(results)
    print(f"\n[RESULTADO] {total} hashes en {elapsed:.2f}s | {total/elapsed:.2f} H/s global")

    for idx, (n, lst) in enumerate(zip(results, latencies)):
        hrate = n / (sum(lst)/1000 if lst else 1)
        print(f"   [Thread {idx}] {n} hashes | Hashrate: {hrate:.2f} H/s | avg latency: {statistics.mean(lst):.2f} ms")

    show_latency_report(latencies)

    # ================
    # Modo Estrés (Opcional)
    # ================
    if STRESS_MODE:
        print("[BENCH] Stress mode activado: Reinicializando pool repetidamente...")
        for rep in range(3):
            manager.cleanup()
            time.sleep(0.5)
            for i in range(min(4, len(SEEDS))):
                manager.compute_randomx_hash(random_blob(), SEEDS[i])
            print(f"  [Stress] Iteración {rep+1}/3 OK.")

    del manager
    print("[BENCH] Benchmark finalizado.\n")

if __name__ == "__main__":
    main()
