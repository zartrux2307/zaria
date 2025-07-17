
import os
import sys
import pandas as pd
import logging
import shutil
from datetime import datetime
from iazar.utils.feature_utils import COLUMNS
from iazar.utils.nonce_loader import NonceLoader

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)
# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nonce_cleaner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NonceCleaner")

# Columnas estándar globales
COLUMNS = ["nonce", "entropy", "uniqueness", "zero_density", "pattern_score", "is_valid"]

# Campos requeridos actualizados
CSV_FIELDS = [
    'timestamp', 'nonce', 'nonce_hex', 'major_ver', 'minor_ver',
    'block_timestamp', 'block_size', 'block_hash',
    'accepted', 'predicted_by_ia',
    'entropy', 'uniqueness', 'zero_density', 'pattern_score', 'is_valid'
]


def initialize_paths():
    """Inicializa rutas usando NonceLoader"""
    loader = NonceLoader()
    data_dir = loader.data_dir
    training_dir = os.path.join(data_dir, 'training')
    backups_dir = os.path.join(data_dir, 'backups')

    # Crear directorios si no existen
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(backups_dir, exist_ok=True)

    return {
        'csv_path': os.path.join(training_dir, "nonce_training_data.csv"),
        'backup_path': os.path.join(backups_dir, f"nonce_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
        'backups_dir': backups_dir
    }


def leer_nonces_csv(path):
    """Lee un CSV de nonces y garantiza estructura/cabecera estándar."""
    logger.info(f"Leyendo CSV de nonces: {path}")

    if not os.path.exists(path):
        logger.warning(f"Archivo no encontrado, creando nuevo: {path}")
        pd.DataFrame(columns=CSV_FIELDS).to_csv(path, index=False)
        return pd.DataFrame(columns=CSV_FIELDS)

    try:
        # Detectar tamaño para determinar estrategia de lectura
        file_size = os.path.getsize(path)
        logger.info(f"Tamaño del archivo: {file_size / (1024 * 1024):.2f} MB")

        if file_size > 100 * 1024 * 1024:  # > 100 MB
            logger.info("Archivo grande, usando lectura por chunks")
            chunks = []
            for chunk in pd.read_csv(path, chunksize=10000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path)
    except Exception:
        logger.error(f"Error leyendo CSV: {str(e)}")
        return pd.DataFrame(columns=CSV_FIELDS)

    # Verificar estructura de columnas
    logger.info("Verificando estructura de columnas...")
    missing = [col for col in CSV_FIELDS if col not in df.columns]
    if missing:
        logger.warning(f"Faltan columnas: {', '.join(missing)}")
        for col in missing:
            df[col] = None  # Usar None en lugar de 0 para datos faltantes

    # Mantener solo campos válidos
    valid_cols = [col for col in CSV_FIELDS if col in df.columns]
    df = df[valid_cols]

    # Eliminar filas completamente vacías
    initial_count = len(df)
    df = df.dropna(how='all')
    if initial_count != len(df):
        logger.info(f"Eliminadas {initial_count - len(df)} filas completamente vacías")

    return df


def guardar_nonces_csv(df, path):
    """Guarda un DataFrame de nonces con la cabecera y orden estándar."""
    logger.info(f"Guardando CSV de nonces: {path}")

    # Verificar y completar columnas faltantes
    missing = [col for col in CSV_FIELDS if col not in df.columns]
    if missing:
        logger.warning(f"Añadiendo columnas faltantes: {', '.join(missing)}")
        for col in missing:
            df[col] = None

    # Ordenar columnas
    df = df[CSV_FIELDS]

    # Guardar con manejo de errores
    try:
        # Guardar temporalmente primero
        temp_path = path + ".tmp"
        df.to_csv(temp_path, index=False)

        # Reemplazar archivo original
        if os.path.exists(path):
            os.replace(temp_path, path)
        else:
            os.rename(temp_path, path)

        logger.info(f"Guardado exitoso. Filas: {len(df)}")
    except Exception:
        logger.error(f"Error guardando CSV: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def create_backup(source_path, backup_dir):
    """Crea un backup seguro con rotación automática"""
    logger.info("Creando backup...")

    # Generar nombre único para el backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"nonce_training_data_{timestamp}.csv")

    try:
        # Copiar en lugar de mover para mantener original
        shutil.copy2(source_path, backup_path)
        logger.info(f"Backup creado: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Error creando backup: {str(e)}")
        return None


def clean_backups(backup_dir, max_backups=5):
    """Limpia backups antiguos manteniendo solo los más recientes"""
    logger.info(f"Limpiando backups en {backup_dir} (máximo {max_backups})")

    try:
        # Listar todos los backups
        backups = []
        for f in os.listdir(backup_dir):
            if f.startswith("nonce_training_data_") and f.endswith(".csv"):
                file_path = os.path.join(backup_dir, f)
                backups.append((file_path, os.path.getmtime(file_path)))

        # Ordenar por fecha (más antiguos primero)
        backups.sort(key=lambda x: x[1])

        # Eliminar los más antiguos si excedemos el máximo
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            try:
                os.remove(oldest[0])
                logger.info(f"Eliminado backup antiguo: {oldest[0]}")
            except Exception as e:
                logger.error(f"Error eliminando backup {oldest[0]}: {str(e)}")

    except Exception as e:
        logger.error(f"Error limpiando backups: {str(e)}")


def clean_duplicates():
    """Limpia duplicados y verifica la integridad de los datos"""
    # Inicializar rutas
    paths = initialize_paths()
    csv_path = paths['csv_path']
    backups_dir = paths['backups_dir']

    logger.info("=" * 50)
    logger.info(f"INICIANDO LIMPIEZA DE DATOS: {csv_path}")
    logger.info("=" * 50)

    # Paso 1: Leer datos existentes
    df = leer_nonces_csv(csv_path)
    if df.empty:
        logger.warning("No hay datos para procesar")
        return

    initial_count = len(df)
    logger.info(f"Registros iniciales: {initial_count}")

    # Paso 2: Crear backup antes de cualquier modificación
    backup_path = create_backup(csv_path, backups_dir)

    # Paso 3: Eliminar duplicados
    logger.info("Buscando duplicados...")

    # Verificar que 'nonce' existe
    if 'nonce' not in df.columns:
        logger.error("Columna 'nonce' no encontrada en los datos. Abortando.")
        return

    # Identificar duplicados manteniendo el primero
    duplicates_mask = df.duplicated(subset=['nonce'], keep='first')
    duplicate_count = duplicates_mask.sum()

    if duplicate_count > 0:
        logger.info(f"Encontrados {duplicate_count} duplicados")

        # Crear DataFrame con duplicados para registro
        duplicates_df = df[duplicates_mask]
        dup_report_path = os.path.join(backups_dir, f"duplicates_{datetime.now().strftime('%Y%m%d')}.csv")
        duplicates_df.to_csv(dup_report_path, index=False)
        logger.info(f"Reporte de duplicados guardado en: {dup_report_path}")

        # Eliminar duplicados
        df = df[~duplicates_mask]
        logger.info(f"Duplicados eliminados. Registros restantes: {len(df)}")
    else:
        logger.info("No se encontraron duplicados")

    # Paso 4: Verificar valores faltantes críticos
    logger.info("Verificando valores faltantes...")
    critical_cols = ['nonce', 'block_hash', 'accepted']
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"{null_count} valores faltantes en columna '{col}'")

    # Paso 5: Guardar datos limpios
    guardar_nonces_csv(df, csv_path)

    # Paso 6: Limpiar backups antiguos
    clean_backups(backups_dir, max_backups=5)

    # Resumen final
    logger.info("=" * 50)
    logger.info(f"LIMPIEZA COMPLETADA")
    logger.info(f"Registros iniciales: {initial_count}")
    logger.info(f"Registros finales:   {len(df)}")
    logger.info(f"Duplicados eliminados: {duplicate_count}")
    logger.info(f"Backup creado: {backup_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    try:
        clean_duplicates()
    except Exception:
        logger.exception("Error fatal durante la limpieza:")
