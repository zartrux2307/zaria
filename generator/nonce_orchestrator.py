"""Orquestador Optimizado para Intel i5-2330 (4 hilos/8GB RAM)"""

import os
import sys
import json
import time
import logging
import struct
import threading
import queue
import multiprocessing.shared_memory as shm
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# === CONFIGURACIÓN INICIAL DE LOGGING ===
# Configuración básica temporal para capturar errores iniciales
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NonceOrchestrator")

# Insertar ruta actual para importar randomx_wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import randomx_wrapper 
except Exception as e:
    logger.error(f"Error importando randomx_wrapper: {str(e)}")
    sys.exit(1)

# Configurar rutas del proyecto
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# === CONFIGURACIÓN COMPLETA DE LOGGING ===
# Directorio para logs
log_dir = os.path.join(PROJECT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

# Handler para archivo
fh = logging.FileHandler(os.path.join(log_dir, "orquestador.log"))
fh.setLevel(logging.INFO)

# Handler para consola
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Reemplazar configuración básica con configuración completa
logger.handlers = []
logger.addHandler(fh)
logger.addHandler(ch)
logger.propagate = False

# Ahora importar módulos del proyecto
from datetime import datetime
try:
    from iazar.generator.config_loader import load_config
    # Importar generadores
    from iazar.generator.adaptive_generator import AdaptiveGenerator
    from iazar.generator.entropy_based_generator import EntropyBasedGenerator
    from iazar.generator.hybrid_generator import HybridGenerator
    from iazar.generator.ml_based_generator import MLBasedGenerator
    from iazar.generator.random_generator import RandomGenerator
    from iazar.generator.range_based_generator import RangeBasedGenerator
    from iazar.generator.sequence_based_generator import SequenceBasedGenerator

    # Importar módulos de evaluación
    from iazar.evaluation import calculatenonce, correlation_analysis
    from iazar.evaluation import distribution_analyzer, entropy_analysis
    from iazar.evaluation import nonce_quality_filter, survival_analyzer
except ImportError as e:
    logger.error(f"Error importando módulos: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)

# === CONFIGURACIÓN MEJORADA ===
# Cargar configuración con manejo de errores
try:
    config = load_config()
    logger.info("Configuración cargada exitosamente")
except Exception as e:
    logger.error(f"Error cargando configuración: {e}")
    config = {
        'data_paths': {
            'training_data': os.path.join('data', 'nonce_training_data.csv'),
            'generated_nonces': os.path.join('data', 'nonces_exitosos.csv'),
            'accepted_nonces': os.path.join('data', 'nonces_aceptados.csv')
        },
        'performance_settings': {'batch_size': 500},
        'quality_threshold': 0.75
    }

# Estructura de rutas de datos con validación
DATA_DIR = os.path.abspath('data')
DEFAULT_DATA_PATHS = {
    "training_data": os.path.join(DATA_DIR, "nonce_training_data.csv"),
    "generated_nonces": os.path.join(DATA_DIR, "nonces_exitosos.csv"),
    "accepted_nonces": os.path.join(DATA_DIR, "nonces_aceptados.csv")
}

# Crear directorios necesarios
os.makedirs(DATA_DIR, exist_ok=True)
for key in DEFAULT_DATA_PATHS:
    path = config.get('data_paths', {}).get(key, DEFAULT_DATA_PATHS[key])
    os.makedirs(os.path.dirname(path), exist_ok=True)

# === CONFIGURACIÓN DE MEMORIA COMPARTIDA ACTUALIZADA ===
PREFIX = str(config.get("prefix", "5555"))

# Nuevos tamaños de estructura basados en los formatos corregidos
JOB_STRUCT_SIZE = 184  # 32 (job_id) + 84 (blob) + 32 (seed_hash) + 32 (target) + 4 (height)
SOLUTION_STRUCT_SIZE = 68  # 32 (job_id) + 4 (nonce) + 32 (hash)
SHM_JOB_SIZE = JOB_STRUCT_SIZE + 1
SHM_SOLUTION_SIZE = SOLUTION_STRUCT_SIZE + 1

# Parámetros de rendimiento optimizados para 4 núcleos/8GB RAM
NONCES_POR_LOTE = config.get("performance_settings", {}).get("batch_size", 500)
INTERVALO_LOTE = 1.0  # 1 segundo por lote de 500 nonces
EVALUATION_INTERVAL = int(config.get("evaluation_interval", 300))  # 5 minutos

# === GENERADOR DE RESERVA MEJORADO ===
class RobustFallbackGenerator:
    """Generador de respaldo mejorado con características adaptativas básicas"""
    def __init__(self):
        self.name = "RobustFallback"
        self.counter = 0
        self.last_success = 0
        
    def generate_nonce(self, height):
        self.counter += 1
        # Adaptación simple cada 1000 nonces
        if self.counter - self.last_success > 1000:
            return {"nonce": random.randint(0, 0x7FFFFFFF)}  # Probar rango positivo
        return {"nonce": random.randint(0, 0xFFFFFFFF)}
    
    def record_success(self):
        self.last_success = self.counter

# === CLASES CORREGIDAS ===
class GestorMemoriaCompartida:
    """Gestión profesional de memoria compartida con reconexión"""
    
    def __init__(self, prefix):
        self.prefix = prefix
        self.job_shm = None
        self.solution_shm = None
        self.conectar()
    
    def conectar(self):
        """Conexión robusta con manejo de errores"""
        try:
            self.job_shm = shm.SharedMemory(name=f"{self.prefix}_job")
            self.solution_shm = shm.SharedMemory(name=f"{self.prefix}_solution")
            logger.info("Conexión a memoria compartida establecida")
        except FileNotFoundError:
            try:
                self.job_shm = shm.SharedMemory(
                    name=f"{self.prefix}_job", create=True, size=SHM_JOB_SIZE)
                self.solution_shm = shm.SharedMemory(
                    name=f"{self.prefix}_solution", create=True, size=SHM_SOLUTION_SIZE)
                logger.info("Segmentos de memoria compartida creados")
            except Exception as e:
                logger.error(f"Error creando memoria: {e}")
        except Exception as e:
            logger.error(f"Error conectando: {e}")
            self.reconectar()
    
    def reconectar(self):
        """Reconexión automática"""
        logger.warning("Reconectando a memoria compartida...")
        time.sleep(1)
        self.conectar()
    
    def cerrar(self):
        """Liberación segura de recursos"""
        try:
            if self.job_shm:
                self.job_shm.close()
            if self.solution_shm:
                self.solution_shm.close()
        except Exception as e:
            logger.error(f"Error liberando memoria: {e}")

class GestorGeneradores:
    """Gestión profesional del ciclo de vida de generadores con fallback"""
    
    def __init__(self, config):
        self.config = config
        self.generadores = []
        self.generadores_activos = {}
        self.rendimiento_generadores = {}
        self.inicializar_generadores()
        
    def inicializar_generadores(self):
        """Inicialización robusta con manejo de errores"""
        generadores_clases = {
            'AdaptiveGenerator': AdaptiveGenerator,
            'EntropyBasedGenerator': EntropyBasedGenerator,
            'HybridGenerator': HybridGenerator,
            'MLBasedGenerator': MLBasedGenerator,
            'RandomGenerator': RandomGenerator,
            'RangeBasedGenerator': RangeBasedGenerator,
            'SequenceBasedGenerator': SequenceBasedGenerator
        }
        
        for nombre, clase in generadores_clases.items():
            try:
                # Configuración segura con valores por defecto
                gen_config = self.config.get('generators', {}).get(nombre, {})
                if not isinstance(gen_config, dict):
                    gen_config = {}
                
                # Verificar parámetros requeridos por el generador
                sig = inspect.signature(clase.__init__)
                params = sig.parameters
                
                # Crear diccionario de argumentos basado en parámetros esperados
                init_kwargs = {}
                for param_name in params:
                    if param_name == 'self':
                        continue
                    if param_name in gen_config:
                        init_kwargs[param_name] = gen_config[param_name]
                    elif param_name == 'load_config':
                        init_kwargs[param_name] = load_config  # Pasar la función
                
                generador = clase(**init_kwargs)
                self.generadores.append(generador)
                self.generadores_activos[nombre] = generador
                self.rendimiento_generadores[nombre] = {'contador': 0, 'exitosos': 0}
                logger.info(f"Generador {nombre} inicializado")
            except Exception as e:
                logger.error(f"Error en {nombre}: {e}")
                # Usar generador de respaldo mejorado para este slot
                logger.warning(f"Usando generador de respaldo mejorado para {nombre}")
                fallback = RobustFallbackGenerator()
                self.generadores.append(fallback)
                self.generadores_activos[nombre] = fallback
                self.rendimiento_generadores[nombre] = {'contador': 0, 'exitosos': 0}
        
        # Crear generador de respaldo adicional si todos fallan
        if not self.generadores_activos:
            try:
                logger.warning("Creando generador de respaldo adicional")
                fallback = RobustFallbackGenerator()
                self.generadores.append(fallback)
                self.generadores_activos['RobustFallback'] = fallback
                self.rendimiento_generadores['RobustFallback'] = {'contador': 0, 'exitosos': 0}
            except Exception as e:
                logger.critical(f"Error crítico: {e}")
    
    def seleccionar_generador(self):
        """Selección con reintento de inicialización"""
        if not self.generadores_activos:
            logger.warning("Reintentando inicialización de generadores")
            self.inicializar_generadores()
            
        if self.generadores_activos:
            return random.choice(list(self.generadores_activos.values()))
        return None

    def actualizar_rendimiento(self, nombre_generador, exito):
        """Actualizar estadísticas de rendimiento de un generador"""
        if nombre_generador in self.rendimiento_generadores:
            self.rendimiento_generadores[nombre_generador]['contador'] += 1
            if exito:
                self.rendimiento_generadores[nombre_generador]['exitosos'] += 1
                # Registrar éxito en generadores de respaldo
                if isinstance(self.generadores_activos[nombre_generador], RobustFallbackGenerator):
                    self.generadores_activos[nombre_generador].record_success()

# === SISTEMA DE EVALUACIÓN CORREGIDO ===
class SistemaEvaluacion:
    """Sistema profesional de evaluación de calidad de nonces"""
    
    def __init__(self, config):
        self.config = config
        self.evento_parada = threading.Event()
        self.cola_evaluacion = queue.Queue(maxsize=1000)
        self.metricas = {
            'nonces_evaluados': 0,
            'nonces_aceptados': 0,
            'nonces_rechazados': 0
        }
        
    def iniciar_evaluacion(self):
        """Iniciar subsistema de evaluación"""
        hilo_evaluacion = threading.Thread(
            target=self.bucle_evaluacion,
            daemon=True
        )
        hilo_evaluacion.start()
        logger.info("Sistema de evaluación iniciado")
        
    def bucle_evaluacion(self):
        """Bucle principal de evaluación"""
        while not self.evento_parada.is_set():
            try:
                # Obtener datos para evaluación
                datos = self.cola_evaluacion.get(timeout=5)
                if datos:
                    self.evaluar_calidad(datos)
                    self.metricas['nonces_evaluados'] += 1
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error en evaluación: {str(e)}")
                
    def evaluar_calidad(self, datos):
        """Evaluar profesionalmente la calidad de un nonce"""
        try:
            # Ejecutar análisis de calidad
            resultados = {
                'entropia': entropy_analysis.calcular_entropia(datos['nonce']),
                'distribucion': distribution_analyzer.analizar_distribucion(datos['nonce']),
                'correlacion': correlation_analysis.analizar_correlacion(datos['nonce']),
                'calidad': calculatenonce.calcular_calidad(datos['nonce'])
            }
            
            # Aplicar filtro de calidad
            aceptado = nonce_quality_filter.filtrar_por_calidad(
                datos['nonce'],
                resultados['calidad'],
                self.config.get('quality_threshold', 0.75)
            )
            
            if aceptado:
                self.metricas['nonces_aceptados'] += 1
                # Guardar nonce aceptado
                self.guardar_nonce_aceptado(datos, resultados)
            else:
                self.metricas['nonces_rechazados'] += 1
                logger.debug(f"Nonce rechazado: {datos['nonce']}")
                
        except Exception as e:
            logger.error(f"Error evaluando calidad: {str(e)}")
            
    def guardar_nonce_aceptado(self, datos, resultados):
        """Guardar profesionalmente nonce aceptado"""
        try:
            # Guardar en archivo CSV
            ruta = self.config.get('data_paths', {}).get('accepted_nonces', DEFAULT_DATA_PATHS['accepted_nonces'])
            with open(ruta, 'a') as f:
                f.write(f"{time.time()},{datos['nonce']},{resultados['calidad']}\n")
                
            logger.info(f"Nonce aceptado: {datos['nonce']} Calidad={resultados['calidad']:.2f}")
        except Exception as e:
            logger.error(f"Error guardando nonce: {str(e)}")

# === NÚCLEO DEL ORQUESTADOR CORREGIDO ===
class NonceOrchestrator:
    
    def __init__(self):
        self.config = config
        self.gestor_memoria = GestorMemoriaCompartida(PREFIX)
        self.gestor_generadores = GestorGeneradores(config)
        self.sistema_evaluacion = SistemaEvaluacion(config)
        self.lock_job = threading.Lock()
        self.evento_parada = threading.Event()
        self.offset_nonce = 39
        self.job_actual = None
        self.cola_soluciones = queue.Queue(maxsize=100)
        self.estadisticas_rendimiento = {
            'nonces_generados': 0,
            'nonces_validados': 0,
            'soluciones_encontradas': 0,
            'ultima_solucion': 0
        }
        self.ultimo_lote = time.time()
        
        # Iniciar subsistemas
        self.iniciar_sistema_evaluacion()
        self.iniciar_enviador_soluciones()
        logger.info("🎯 Orquestador optimizado inicializado")

    def iniciar_sistema_evaluacion(self):
        """Iniciar subsistema de evaluación"""
        self.sistema_evaluacion.iniciar_evaluacion()
        
    def iniciar_enviador_soluciones(self):
        """Iniciar hilo dedicado para enviar soluciones"""
        hilo_enviador = threading.Thread(
            target=self.bucle_enviador_soluciones,
            daemon=True
        )
        hilo_enviador.start()
        logger.info("Hilo enviador de soluciones iniciado")

    def bucle_enviador_soluciones(self):
        """Hilo dedicado para enviar soluciones"""
        while not self.evento_parada.is_set():
            try:
                solucion = self.cola_soluciones.get(timeout=1)
                self.enviar_solucion(solucion)
                self.estadisticas_rendimiento['soluciones_encontradas'] += 1
                self.estadisticas_rendimiento['ultima_solucion'] = time.time()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error enviando solución: {str(e)}")

    def obtener_ultimo_job(self):
        """Obtener último job de memoria compartida"""
        try:
            if self.gestor_memoria.job_shm.buf[SHM_JOB_SIZE - 1] == 1:
                data = bytes(self.gestor_memoria.job_shm.buf[:JOB_STRUCT_SIZE])
                self.gestor_memoria.job_shm.buf[SHM_JOB_SIZE - 1] = 0
                return deserializar_job(data)
        except Exception as e:
            logger.error(f"Error obteniendo job: {str(e)}")
            # Intentar reconectar a memoria compartida
            self.gestor_memoria.conectar()
        return None

    def enviar_solucion(self, solucion):
        """Enviar solución mediante memoria compartida"""
        if not solucion:
            return
            
        sol_bytes = serializar_solucion(solucion)
        if not sol_bytes or len(sol_bytes) != SOLUTION_STRUCT_SIZE:
            logger.error(f"Formato de solución inválido. Tamaño: {len(sol_bytes) if sol_bytes else 0}")
            return
            
        try:
            # Esperar hasta que el buffer esté disponible (máx 50ms)
            start_wait = time.time()
            while self.gestor_memoria.solution_shm.buf[SOLUTION_STRUCT_SIZE] == 1:
                if time.time() - start_wait > 0.05:
                    logger.warning("Buffer de soluciones lleno, descartando solución")
                    return
                time.sleep(0.001)
                
            # Escribir solución
            self.gestor_memoria.solution_shm.buf[:SOLUTION_STRUCT_SIZE] = sol_bytes
            self.gestor_memoria.solution_shm.buf[SOLUTION_STRUCT_SIZE] = 1
            
            # Log con representación hexadecimal para depuración
            logger.info(f"✅ Solución enviada: nonce={solucion['nonce']} job_id={solucion['job_id'][:8].hex()}...")
        except Exception as e:
            logger.error(f"Error enviando solución: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    # === VALIDACIÓN DE NONCES OPTIMIZADA ===
    def validar_nonce(self, nonce, job):
        """Validación optimizada de nonce con RandomX usando memoryview"""
        try:
            # Pre-calcular valores estáticos
            blob_bytes = bytes.fromhex(job['blob'])
            if len(blob_bytes) < self.offset_nonce + 4:
                logger.error(f"Blob demasiado corto: {len(blob_bytes)} bytes")
                return None
                
            seed_hash = job['seed_hash']
            target = int(job['target'], 16)
            target_256 = target << 192
            
            # Usar memoryview para modificación sin copia
            blob_mutable = bytearray(blob_bytes)
            with memoryview(blob_mutable) as mv:
                # Insertar nonce directamente en el blob
                mv[self.offset_nonce:self.offset_nonce + 4] = nonce.to_bytes(4, "little")
                
                # Calcular hash con RandomX
                hash_bloque = randomx_wrapper.compute_randomx_hash(
                    blob=bytes(mv.obj),
                    seed=seed_hash
                )
            
            # Verificar si cumple el target
            hash_int = int.from_bytes(hash_bloque, 'little')
            cumple_target = hash_int < target_256
            
            self.estadisticas_rendimiento['nonces_validados'] += 1
            
            if cumple_target:
                logger.debug(f"Nonce válido encontrado: {nonce}")
            
            return {
                "job_id": job['job_id'],
                "nonce": nonce,
                "hash": hash_bloque.hex(),
                "is_valid": cumple_target
            }
        except Exception as e:
            logger.error(f"Error validando: {str(e)}")
            return None

    # === EJECUCIÓN DE LOTES PARALELIZADA ===
    def ejecutar_lote(self, job):
        """Ejecutar un lote de nonces en paralelo"""
        tiempo_actual = time.time()
        tiempo_transcurrido = tiempo_actual - self.ultimo_lote
        
        # Esperar hasta completar el intervalo de 1 segundo
        if tiempo_transcurrido < INTERVALO_LOTE:
            time.sleep(INTERVALO_LOTE - tiempo_transcurrido)
        
        inicio_lote = time.time()
        generador = self.gestor_generadores.seleccionar_generador()
        if not generador:
            logger.error("No hay generadores disponibles. Reintentando en 1 segundo...")
            time.sleep(1)
            return
            
        nombre_gen = next((k for k, v in self.gestor_generadores.generadores_activos.items() if v == generador), 'desconocido')
        
        try:
            # Generar nonces en paralelo
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(generador.generate_nonce, job['height']) for _ in range(NONCES_POR_LOTE)]
                nonces = [f.result()["nonce"] for f in futures]
            
            self.estadisticas_rendimiento['nonces_generados'] += len(nonces)
            
            # Validar nonces en paralelo
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self.validar_nonce, nonce, job): nonce for nonce in nonces}
                for future in as_completed(futures):
                    solucion = future.result()
                    if solucion and solucion["is_valid"]:
                        self.cola_soluciones.put(solucion)
                        self.gestor_generadores.actualizar_rendimiento(nombre_gen, True)
                        
        except Exception as e:
            logger.error(f"Error en generador: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        self.ultimo_lote = time.time()
        tiempo_procesamiento = self.ultimo_lote - inicio_lote
        logger.info(f"Lote procesado: {NONCES_POR_LOTE} nonces en {tiempo_procesamiento:.4f}s")

    def observador_jobs(self):
        """Hilo dedicado para observar nuevos jobs"""
        logger.info("👀 Observador de jobs iniciado")
        last_job_id = b''
        while not self.evento_parada.is_set():
            job = self.obtener_ultimo_job()
            if job and job['job_id'] != last_job_id:
                with self.lock_job:
                    self.job_actual = job
                    last_job_id = job['job_id']
                    # Log con representación hexadecimal
                    logger.info(f"🆕 Nuevo job: {last_job_id[:8].hex()}... altura={job['height']}")
                    # Limpiar cola de soluciones del job anterior
                    while not self.cola_soluciones.empty():
                        try:
                            self.cola_soluciones.get_nowait()
                        except queue.Empty:
                            break
            time.sleep(0.05)  # Sondeo más rápido

    def bucle_mineria(self):
        """Bucle principal de minería optimizado"""
        logger.info("⚒️ Bucle de minería iniciado")
        while not self.evento_parada.is_set():
            if self.job_actual:
                self.ejecutar_lote(self.job_actual)
            else:
                time.sleep(0.1)

    def monitor_rendimiento(self):
        """Hilo de monitoreo de rendimiento"""
        logger.info("📈 Monitor de rendimiento iniciado")
        while not self.evento_parada.is_set():
            time.sleep(60)
            logger.info(
                f"Rendimiento: Generados={self.estadisticas_rendimiento['nonces_generados']} "
                f"Validados={self.estadisticas_rendimiento['nonces_validados']} "
                f"Soluciones={self.estadisticas_rendimiento['soluciones_encontradas']}"
            )
            # Registrar rendimiento de generadores
            for nombre, stats in self.gestor_generadores.rendimiento_generadores.items():
                if stats['contador'] > 0:
                    tasa_exito = (stats['exitosos'] / stats['contador']) * 100
                    logger.info(
                        f"Generador {nombre}: "
                        f"Intentos={stats['contador']} "
                        f"Exitosos={stats['exitosos']} "
                        f"Tasa={tasa_exito:.2f}%"
                    )

    def iniciar(self):
        """Iniciar todos los subsistemas del orquestador"""
        # Iniciar observador de jobs
        threading.Thread(target=self.observador_jobs, daemon=True).start()
        
        # Iniciar monitor de rendimiento
        threading.Thread(target=self.monitor_rendimiento, daemon=True).start()
        
        # Iniciar bucle principal de minería
        self.bucle_mineria()
        
        logger.info("🏁 Orquestador iniciado exitosamente")

    def detener(self):
        """Apagado controlado"""
        self.evento_parada.set()
        self.gestor_memoria.cerrar()
        logger.info("🛑 Orquestador detenido")

# Funciones de serialización/deserialización CORREGIDAS
def deserializar_job(data):
    """Deserializa datos binarios a un diccionario de trabajo"""
    try:
        # Nueva estructura: [job_id(32)|blob(84)|seed_hash(32)|target(32)|height(4)]
        job_id_bytes = data[0:32]
        blob = data[32:116].hex()  # 84 bytes -> hex string
        seed_hash = data[116:148].hex()  # 32 bytes -> hex string
        target = data[148:180].hex()  # 32 bytes -> hex string
        height = struct.unpack('<I', data[180:184])[0]  # uint32 little-endian
        return {
            'job_id': job_id_bytes,
            'blob': blob,
            'seed_hash': seed_hash,
            'target': target,
            'height': height
        }
    except Exception as e:
        logger.error(f"Error deserializando job: {str(e)}")
        return None

def serializar_solucion(solucion):
    """Serializa una solución a formato binario usando bytes directamente"""
    try:
        job_id_bytes = solucion['job_id']  # Ya es bytes
        nonce = solucion['nonce']
        hash_bytes = bytes.fromhex(solucion['hash'])
        
        # Nueva estructura: [job_id(32)|nonce(4)|hash(32)]
        return struct.pack('<32sI32s', job_id_bytes, nonce, hash_bytes)
    except Exception as e:
        logger.error(f"Error serializando solución: {str(e)}")
        return None

if __name__ == "__main__":
    orquestador = NonceOrchestrator()
    try:
        orquestador.iniciar()
    except KeyboardInterrupt:
        logger.info("⛔ Señal de apagado recibida")
    except Exception as e:
        logger.error(f"Error crítico: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        orquestador.detener()
        sys.exit(0)