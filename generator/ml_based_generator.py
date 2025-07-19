import os
import threading
import time
import logging
import random
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb

from iazar.generator.nonce_generator import BaseNonceGenerator
from iazar.generator.config_loader import config_loader
from iazar.generator.randomx_validator import RandomXValidator

# ================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s"
)
logger = logging.getLogger("MLBasedGenerator")

# ================== CONFIGURACIÓN ==================
MODEL_VERSION = "v2.2"
RETRAIN_INTERVAL = 1800  # segundos
MIN_TRAINING_SAMPLES = 10000
CANDIDATE_BATCH_SIZE = 2000
VALIDATION_WORKERS = 4

# Directorio base para datos
DATA_DIR = './data'

class MLBasedGenerator(BaseNonceGenerator):
    """
    Generador profesional basado en Machine Learning para minería Monero.
    - Ensemble con XGBoost y LightGBM
    - Selección y validación por lotes
    - Entrenamiento incremental en background
    - 100% CPU, robusto para entornos productivos
    """
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("ml_based", config)
        self.model = None
        self.scaler = None
        self.validator = RandomXValidator(self.config)
        self.lock = threading.RLock()
        self.last_trained = 0
        self.feature_importances = {}
        self.acceptance_rate = 0.0
        self.training_data_df = None
        self.model_path = os.path.join(
            self.config.get('model_storage', './models'),
            f"nonce_model_{MODEL_VERSION}.joblib"
        )
        self.data_refreshed = False
        self._load_or_train_model()
        logger.info("[MLBasedGenerator] Ready for enterprise mining.")
    
    def _get_data_path(self, key: str) -> str:
        """Obtener ruta de datos profesional"""
        return os.path.join(DATA_DIR, f"ml_{key}.csv")

    def _load_or_train_model(self):
        try:
            if os.path.exists(self.model_path):
                with self.lock:
                    model_data = joblib.load(self.model_path)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.last_trained = model_data['timestamp']
                    self.feature_importances = model_data.get('feature_importances', {})
                    logger.info(f"[MLBasedGenerator] Model v{MODEL_VERSION} loaded (accuracy: {model_data.get('accuracy', 0):.2f})")
                return
        except Exception as e:
            logger.error(f"[MLBasedGenerator] Model load failed: {e}")
        threading.Thread(target=self._train_model, daemon=True).start()
        self._initialize_fallback_model()

    def _refresh_training_data(self) -> pd.DataFrame:
        """Carga profesional de datos de entrenamiento"""
        with self.lock:
            if self.data_refreshed:
                return self.training_data_df
            
            path = self._get_data_path('training_data')
            if not os.path.exists(path):
                logger.warning(f"[MLBasedGenerator] Training data not found at {path}")
                return pd.DataFrame()
            
            try:
                df = pd.read_csv(path)
                required_cols = ['nonce', 'entropy', 'uniqueness', 
                                'zero_density', 'pattern_score', 'is_valid']
                
                # Validar y completar columnas faltantes
                for col in required_cols:
                    if col not in df.columns:
                        if col == 'is_valid':
                            df[col] = False  # Valor por defecto para columna de validación
                        else:
                            df[col] = np.nan  # NaN para características numéricas
                
                # Filtrar filas con valores faltantes
                df = df.dropna(subset=required_cols)
                
                # Feature engineering
                df['bit_variance'] = df['nonce'].apply(
                    lambda x: np.var([int(b) for b in bin(int(x))[2:].zfill(64)])
                )
                df['byte_entropy'] = df['nonce'].apply(
                    lambda x: self._byte_level_entropy(int(x)))
                
                self.training_data_df = df
                self.data_refreshed = True
                logger.info(f"[MLBasedGenerator] Training samples loaded: {len(df)}")
                return df
            except Exception as e:
                logger.error(f"[MLBasedGenerator] Error loading training data: {e}")
                return pd.DataFrame()

    def _byte_level_entropy(self, nonce: int) -> float:
        bytes_repr = nonce.to_bytes(8, 'little')
        counts = np.bincount(np.frombuffer(bytes_repr, dtype=np.uint8), minlength=256)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _train_model(self):
        start_time = time.time()
        df = self._refresh_training_data()
        if len(df) < MIN_TRAINING_SAMPLES:
            logger.warning("[MLBasedGenerator] Insufficient data for training.")
            return
        features = df[[
            'entropy', 'uniqueness', 'zero_density', 
            'pattern_score', 'bit_variance', 'byte_entropy'
        ]]
        target = df['is_valid'].astype(int)
        # Remove outliers
        outlier_detector = IsolationForest(contamination=0.05, random_state=42)
        mask = outlier_detector.fit_predict(features) == 1
        features = features[mask]
        target = target[mask]
        # Scaling
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(features)
        x_train, x_test, y_train, y_test = train_test_split(
            X_scaled, target, test_size=0.2, random_state=42, stratify=target
        )
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.85,
            tree_method='hist',
            predictor='auto',
            eval_metric='logloss',
            objective='binary:logistic',
            verbosity=1,
            use_label_encoder=False
        )
        xgb_model.fit(x_train, y_train)
        acc_xgb = accuracy_score(y_test, xgb_model.predict(x_test))
        f1_xgb = f1_score(y_test, xgb_model.predict(x_test))
        logger.info(f"[MLBasedGenerator] XGBoost Accuracy={acc_xgb:.4f}, F1={f1_xgb:.4f}")

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=48,
            device='cpu',
            metric='binary_logloss',
            boosting_type='gbdt'
        )
        lgb_model.fit(x_train, y_train)
        acc_lgb = accuracy_score(y_test, lgb_model.predict(x_test))
        f1_lgb = f1_score(y_test, lgb_model.predict(x_test))
        logger.info(f"[MLBasedGenerator] LightGBM Accuracy={acc_lgb:.4f}, F1={f1_lgb:.4f}")

        # Ensemble
        class EnsembleModel:
            def predict_proba(self, X):
                p1 = xgb_model.predict_proba(X)[:, 1]
                p2 = lgb_model.predict_proba(X)[:, 1]
                avg = (p1 + p2) / 2
                return np.vstack([1 - avg, avg]).T
            def predict(self, X):
                return ((self.predict_proba(X)[:, 1] > 0.5).astype(int))
        self.model = EnsembleModel()
        self.feature_importances = {
            'xgb': xgb_model.feature_importances_,
            'lgb': lgb_model.feature_importances_
        }
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'timestamp': time.time(),
            'accuracy': (acc_xgb + acc_lgb) / 2,
            'f1_score': (f1_xgb + f1_lgb) / 2,
            'feature_importances': self.feature_importances,
            'version': MODEL_VERSION,
            'training_size': len(df)
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model_data, self.model_path)
        self.last_trained = time.time()
        logger.info(f"[MLBasedGenerator] Model trained and saved in {time.time()-start_time:.2f}s")

    def _initialize_fallback_model(self):
        self.scaler = RobustScaler()
        dummy_features = np.array([[0.85, 0.90, 0.12, 0.95, 0.25, 1.8]])
        self.scaler.fit(dummy_features)
        class ProductionFallback:
            def predict_proba(self, X):
                # Simple heuristics
                probs = []
                for x in X:
                    prob = 0.65
                    if x[0] > 0.8: prob += 0.15
                    if x[1] > 0.85: prob += 0.10
                    if x[2] < 0.15: prob += 0.05
                    if x[3] > 0.9: prob += 0.05
                    probs.append([1 - min(1, prob), min(1, prob)])
                return np.array(probs)
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] > 0.7).astype(int)
        self.model = ProductionFallback()
        self.last_trained = time.time()
        logger.warning("[MLBasedGenerator] Using CPU fallback model.")

    def _generate_candidates_batch(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        nonces = np.random.randint(0, 2**63, size=n, dtype=np.uint64)
        metrics = np.zeros((n, 6), dtype=np.float32)
        seen = set()
        for i, nonce in enumerate(nonces):
            if nonce in seen:
                nonce = random.randint(0, 2**64 - 1)
            seen.add(nonce)
            bin_repr = bin(nonce)[2:].zfill(64)
            p0 = bin_repr.count('0') / 64
            p1 = 1 - p0
            entropy = - (p0 * np.log2(p0 + 1e-10) + p1 * np.log2(p1 + 1e-10))
            zero_density, max_zero_run = self._calc_zero_metrics(bin_repr)
            pattern_score = self._calc_pattern_score(bin_repr, max_zero_run)
            uniqueness = self._calc_uniqueness(nonce)
            bit_variance = np.var([int(b) for b in bin_repr])
            byte_entropy = self._byte_level_entropy(nonce)
            metrics[i] = [
                max(0.7, min(0.99, entropy)),
                uniqueness,
                zero_density,
                pattern_score,
                bit_variance,
                byte_entropy
            ]
        return nonces, metrics

    def _calc_zero_metrics(self, bin_str: str) -> Tuple[float, int]:
        zero_count = bin_str.count('0')
        zero_density = zero_count / 64
        max_run = max(len(run) for run in bin_str.split('1')) if '1' in bin_str else 64
        return max(0.01, zero_density), max_run

    def _calc_pattern_score(self, bin_str: str, max_zero_run: int) -> float:
        arr = np.array([int(b) for b in bin_str])
        diff = np.diff(arr, prepend=arr[0]-1, append=arr[-1]-1)
        run_starts = np.where(diff != 0)[0]
        run_lengths = np.diff(run_starts)
        max_run = np.max(run_lengths) if run_lengths.size > 0 else 0
        periodic_score = 0
        for period in [2, 4, 8, 16]:
            if bin_str == (bin_str[:period] * (64//period)):
                periodic_score += 0.3
        transitions = np.sum(arr[:-1] != arr[1:])
        transition_score = min(0.2, transitions / 63)
        return max(0.5, 1.0 - min(0.3, max_run/32) - periodic_score + transition_score)

    def _calc_uniqueness(self, nonce: int) -> float:
        if self.training_data_df is None or len(self.training_data_df) < 100:
            return np.random.uniform(0.85, 0.95)
        sample_size = min(500, len(self.training_data_df))
        sample = self.training_data_df.sample(sample_size)['nonce'].values
        bin_repr = bin(nonce)[2:].zfill(64)
        total_distance = 0
        for other in sample:
            other_bin = bin(int(other))[2:].zfill(64)
            total_distance += sum(a != b for a, b in zip(bin_repr, other_bin))
        return max(0.8, min(0.99, total_distance / (64 * sample_size)))

    def _calculate_dynamic_threshold(self) -> float:
        target_rate = 0.4
        adjustment = 0.05 * (target_rate - self.acceptance_rate)
        return max(0.5, min(0.9, 0.7 + adjustment))

    def run_generation(self, block_height: int, block_data: dict, batch_size: int = 500) -> List[dict]:
        start_time = time.time()
        if time.time() - self.last_trained > RETRAIN_INTERVAL:
            threading.Thread(target=self._train_model, daemon=True).start()
        nonces, features = self._generate_candidates_batch(CANDIDATE_BATCH_SIZE)
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        proba = self.model.predict_proba(features_scaled)[:, 1]
        threshold = self._calculate_dynamic_threshold()
        top_indices = np.argsort(proba)[-batch_size:]
        top_nonces = nonces[top_indices]
        valid_nonces = []
        with ThreadPoolExecutor(max_workers=VALIDATION_WORKERS) as executor:
            futures = [
                executor.submit(self.validator.validate, nonce=nonce, block_data=block_data)
                for nonce in top_nonces
            ]
            for i, future in enumerate(futures):
                if future.result():
                    valid_nonces.append({
                        "nonce": int(top_nonces[i]),
                        "block_height": block_height,
                        "generator": self.generator_name,
                        "is_valid": True,
                        "entropy": float(features[top_indices[i], 0]),
                        "uniqueness": float(features[top_indices[i], 1]),
                        "zero_density": float(features[top_indices[i], 2]),
                        "pattern_score": float(features[top_indices[i], 3])
                    })
        self.acceptance_rate = len(valid_nonces) / batch_size
        logger.info(f"[MLBasedGenerator] Acceptance rate: {self.acceptance_rate:.2%}")
        if valid_nonces:
            self._save_nonces_batch(valid_nonces)
        logger.info(f"[MLBasedGenerator] Generated {len(valid_nonces)} valid nonces in {time.time()-start_time:.2f}s")
        return valid_nonces

    def _save_nonces_batch(self, nonces: List[dict]):
        if not nonces:
            return
        output_path = self._get_data_path('generated_nonces')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = self.FIELDNAMES
        rows = [{k: v for k, v in item.items() if k in fieldnames} for item in nonces]
        with self.lock:
            file_exists = os.path.exists(output_path)
            import csv
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)