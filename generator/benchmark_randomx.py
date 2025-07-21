from __future__ import annotations
"""
benchmark_randomx.py
--------------------
Benchmark de rendimiento hashing RandomX (H/s) usando RandomXValidator.

Opciones CLI:
  --seconds 10          Duración aproximada.
  --batch 1000          Nonces por lote.
  --pool 4              Tamaño pool VM.
  --threads 4           Hilos Python generando trabajo (cada uno pide VMs).
  --seed SEEDHEX        Seed/clave (hex / texto).
  --mode target|accept_all
  --dll PATH            Ruta explícita DLL.
  --flags HARD_AES,LARGE_PAGES,JIT,FULL_MEM

Ejemplo:
  python -m src.iazar.generator.benchmark_randomx --seconds 20 --batch 2000 --pool 8 --threads 8 --mode accept_all
"""

import argparse
import os
import time
import threading
import statistics
from typing import List

from iazar.proxy.randomx_validator import RandomXValidator

def parse_flags(s: str):
    return [f.strip() for f in s.split(",") if f.strip()]

def worker(validator: RandomXValidator, seed_block, batch_size: int, stop_event: threading.Event,
           results: List[int], idx: int):
    local_count = 0
    nonce = 0
    while not stop_event.is_set():
        for _ in range(batch_size):
            ok, h = validator.validate(nonce, seed_block, return_hash=True)
            nonce += 1
            local_count += 1
        # Muy ligero sleep cooperativo
        if local_count >= batch_size:
            pass
    results[idx] = local_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=str, default="benchmark-seed")
    ap.add_argument("--mode", type=str, default="accept_all")
    ap.add_argument("--dll", type=str, default="")
    ap.add_argument("--flags", type=str, default="HARD_AES,JIT")
    ap.add_argument("--mt-dataset", type=int, default=4)
    args = ap.parse_args()

    cfg = {
        "dll_path": args.dll if args.dll else None,
        "flags": parse_flags(args.flags),
        "vm_pool_size": args.pool,
        "verify_mode": args.mode,
        "auto_dataset": True,
        "dataset_mt_threads": args.mt_dataset,
        "hash_little_endian": True
    }
    validator = RandomXValidator(cfg)

    seed_block = {"seed": args.seed, "target": (1 << 256) - 1}

    # Warmup
    validator.validate(0, seed_block, return_hash=True)

    stop_event = threading.Event()
    results = [0] * args.threads
    threads = []
    t_start = time.perf_counter()

    for i in range(args.threads):
        th = threading.Thread(target=worker,
                              args=(validator, seed_block, args.batch, stop_event, results, i),
                              daemon=True)
        th.start()
        threads.append(th)

    time.sleep(args.seconds)
    stop_event.set()
    for th in threads:
        th.join()

    elapsed = time.perf_counter() - t_start
    total = sum(results)
    hps = total / elapsed if elapsed > 0 else 0.0

    print(f"\n=== RandomX Benchmark ===")
    print(f"Duration        : {elapsed:.2f} s")
    print(f"VM Pool Size    : {args.pool}")
    print(f"Threads (Python): {args.threads}")
    print(f"Batch size loop : {args.batch}")
    print(f"Total hashes    : {total}")
    print(f"Hashrate (H/s)  : {hps:,.2f}")
    if args.threads > 1:
        print(f"Per-thread avg  : {statistics.mean(results):,.0f}  (std={statistics.pstdev(results):,.0f})")

if __name__ == "__main__":
    main()
