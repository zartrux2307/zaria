import subprocess
from iazar.proxy.ia_proxy_main import MiningProxy
from iazar.generator.nonce_orchestrator import NonceOrchestrator

def main():
    # Iniciar componentes
    processes = [
        subprocess.Popen(["python", "-m", "src.iazar.proxy.monerod_to_shm_job"]),
        subprocess.Popen(["python", "-m", "src.iazar.generator.nonce_orchestrator"]),
    ]
    
    # Iniciar proxy
    proxy = MiningProxy()
    proxy.start()
    
    # Mantener sistema activo
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()