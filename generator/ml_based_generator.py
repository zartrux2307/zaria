

import numpy as np
import pandas as pd
import joblib
import time
import os
import random  # Added missing import
from datetime import datetime
from iazar.generator.nonce_generator import BaseNonceGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

class MLBasedGenerator(BaseNonceGenerator):
    """
    Generador de nonces con modelo ML optimizado para CPU/RandomX.
    Implementa aprendizaje continuo y características avanzadas.
    """
    
    MODEL_VERSION = "v1.2"
    RETRAIN_INTERVAL = 3600  # Segundos entre reentrenamientos
    MIN_TRAINING_SAMPLES = 5000
    
    def __init__(self, config=None):
        # Handle callable config (evaluate if it's a function)
        evaluated_config = config() if callable(config) else config
        
        super().__init__("ml_based", evaluated_config)
        self.model = None
        self.scaler = None
        self.last_trained = 0
        self.model_path = os.path.join(
            self.config['model_storage'],
            f"randomx_ml_model_{self.MODEL_VERSION}.joblib"
        )
        self.load_or_train_model()
        self.training_data = self.get_training_data() 
        
    def load_or_train_model(self):
        """Carga el modelo o entrena uno nuevo si es necesario"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.last_trained = model_data['timestamp']
                print(f"Loaded ML model from {self.model_path}")
                return
            except Exception as e:
                print(f"Model loading failed: {str(e)}")
        
        print("Training new ML model...")
        self.train_model()
    
    def train_model(self):
        """Entrena un nuevo modelo con datos históricos"""
        training_data = self.get_training_data()
        if len(training_data) < self.MIN_TRAINING_SAMPLES:
            print(f"Warning: Insufficient training data ({len(training_data)} samples)")
            self.initialize_fallback_model()
            return
        
        # Preprocesamiento de datos
        features = training_data[['entropy', 'uniqueness', 'zero_density', 'pattern_score']]
        target = training_data['is_valid'].astype(int)
        
        # Normalización
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)
        
        # Entrenamiento con XGBoost (más preciso)
        x_train, x_test, y_train, y_test = train_test_split(
            X_scaled, target, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo con características avanzadas
        start_time = time.time()
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',
            device='cpu'
        )
        model.fit(x_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluación
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"XGBoost training complete | Accuracy: {accuracy:.4f} | Time: {train_time:.2f}s")
        
        # Convertir a LightGBM para inferencia más rápida
        start_time = time.time()
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=32
        )
        lgb_model.fit(x_train, y_train)
        conversion_time = time.time() - start_time
        
        # Evaluar LightGBM
        y_pred_lgb = lgb_model.predict(x_test)
        accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
        print(f"LightGBM conversion | Accuracy: {accuracy_lgb:.4f} | Time: {conversion_time:.2f}s")
        
        # Guardar modelo
        self.model = lgb_model
        self.last_trained = time.time()
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'timestamp': self.last_trained,
            'accuracy': accuracy_lgb,
            'version': self.MODEL_VERSION
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model_data, self.model_path)
    
    def get_training_data(self) -> pd.DataFrame:
        """Carga y prepara datos de entrenamiento"""
        training_path = self.config['data_paths']['training_data']
        accepted_path = self.config['data_paths']['accepted_nonces']
        
        # Combinar datos de entrenamiento y nonces aceptados
        dfs = []
        if os.path.exists(training_path):
            train_df = pd.read_csv(training_path)
            dfs.append(train_df)
        
        if os.path.exists(accepted_path):
            accepted_df = pd.read_csv(accepted_path)
            # Filtrar solo nonces válidos
            accepted_df = accepted_df[accepted_df['is_valid'] == True]
            dfs.append(accepted_df)
        
        if not dfs:
            return pd.DataFrame()
        
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Limpieza de datos
        full_df = full_df.dropna(subset=[
            'entropy', 'uniqueness', 'zero_density', 'pattern_score', 'is_valid'
        ])
        
        # Asegurar tipos correctos
        full_df = full_df.astype({
            'entropy': 'float32',
            'uniqueness': 'float32',
            'zero_density': 'float32',
            'pattern_score': 'float32',
            'is_valid': 'bool'
        })
        
        return full_df
    
    def initialize_fallback_model(self):
        """Modelo de emergencia cuando no hay datos suficientes"""
        print("Initializing fallback model...")
        self.scaler = StandardScaler()
        dummy_features = np.array([[0.85, 0.90, 0.12, 0.95]])
        self.scaler.fit(dummy_features)
        
        # Modelo simple basado en reglas
        class FallbackModel:
            def predict_proba(self, X):
                # Reglas heurísticas básicas
                probs = []
                for x in X:
                    entropy, uniqueness, zero_density, pattern_score = x
                    prob = 0.7
                    if entropy > 0.8: prob += 0.1
                    if uniqueness > 0.85: prob += 0.1
                    if zero_density < 0.15: prob += 0.05
                    if pattern_score > 0.9: prob += 0.05
                    probs.append([1 - min(1, prob), min(1, prob)])
                return np.array(probs)
        
        self.model = FallbackModel()
        self.last_trained = time.time()
    
    def generate_nonce(self, block_height: int) -> dict:
        """Genera un nonce con alta probabilidad de éxito según modelo ML"""
        # Verificar si necesita reentrenamiento
        if time.time() - self.last_trained > self.RETRAIN_INTERVAL:
            print("Retraining model...")
            self.train_model()
        
        # Generar candidatos hasta encontrar uno con alta probabilidad
        for _ in range(100):  # Límite de intentos
            # Generar nonce aleatorio
            nonce = random.randint(0, 2**64 - 1)
            
            # Calcular métricas optimizadas para RandomX
            metrics = self.calculate_randomx_metrics(nonce)
            
            # Preparar características para el modelo
            features = np.array([[
                metrics['entropy'],
                metrics['uniqueness'],
                metrics['zero_density'],
                metrics['pattern_score']
            ]])
            
            # Normalizar características
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Predecir probabilidad
            prob_valid = self.model.predict_proba(features)[0][1]
            
            # Umbral dinámico basado en dificultad
            threshold = self.calculate_dynamic_threshold(block_height)
            
            if prob_valid >= threshold:
                return {
                    "nonce": nonce,
                    "block_height": block_height,
                    "generator": self.generator_name,
                    "is_valid": True,  # El proxy tiene la última palabra
                    **metrics
                }
        
        # Si no se encuentra candidato, devolver el mejor intento
        return {
            "nonce": nonce,
            "block_height": block_height,
            "generator": self.generator_name,
            "is_valid": True,
            **metrics
        }
    
    def calculate_dynamic_threshold(self, block_height: int) -> float:
        """Calcula umbral dinámico basado en condiciones de red"""
        # Umbral base
        threshold = 0.75
        
        # Aumentar umbral si hay muchos rechazos recientes
        if hasattr(self, 'recent_acceptance_rate'):
            if self.recent_acceptance_rate < 0.3:
                threshold = min(0.85, threshold + 0.1)
        
        # Reducir umbral durante dificultad alta
        if block_height > 2800000:  # Post hardfork
            threshold = max(0.65, threshold - 0.05)
        
        return threshold
    
    def calculate_randomx_metrics(self, nonce: int) -> dict:
        """
        Métricas optimizadas para RandomX:
        - Entropía de Shannon mejorada
        - Uniqueness basado en distancia de Hamming ponderada
        - Zero density con penalización por secuencias largas
        - Pattern score para patrones específicos de RandomX
        """
        bin_repr = bin(nonce)[2:].zfill(64)
        
        # Entropía de Shannon mejorada
        p0 = bin_repr.count('0') / 64
        p1 = 1 - p0
        entropy = - (p0 * np.log2(p0 + 1e-10) + p1 * np.log2(p1 + 1e-10))
        
        # Zero density con penalización por runs largos
        zero_density, max_zero_run = self.calculate_penalized_zero_density(bin_repr)
        
        # Pattern score para RandomX
        pattern_score = self.randomx_pattern_score(bin_repr, max_zero_run)
        
        # Uniqueness basada en datos históricos
        uniqueness = self.historical_uniqueness(nonce)
        
        return {
            "entropy": max(0.7, min(0.99, entropy)),
            "uniqueness": uniqueness,
            "zero_density": zero_density,
            "pattern_score": pattern_score
        }
    
    def calculate_penalized_zero_density(self, binary_str: str) -> tuple:
        """Calcula densidad de ceros con penalización por secuencias largas"""
        zero_count = binary_str.count('0')
        zero_density = zero_count / 64
        
        # Detectar secuencia más larga de ceros
        max_run = 0
        current_run = 0
        for char in binary_str:
            if char == '0':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        # Penalizar secuencias largas (>8 ceros seguidos)
        if max_run > 8:
            penalty = min(0.2, (max_run - 8) * 0.05)
            zero_density -= penalty
        
        return max(0.01, zero_density), max_run
    
    def randomx_pattern_score(self, binary_str: str, max_zero_run: int) -> float:
        """Calcula puntaje de patrón para características de RandomX"""
        score = 1.0
        
        # Penalizar secuencias de ceros largas
        if max_zero_run > 8:
            score -= min(0.3, (max_zero_run - 8) * 0.05)
        
        # Penalizar patrones periódicos
        periodic_penalty = 0
        for period in [2, 4, 8, 16]:
            pattern = binary_str[:period]
            repeats = 64 // period
            if binary_str == pattern * repeats:
                periodic_penalty += 0.2
        
        score -= periodic_penalty
        
        # Bonificación por distribución uniforme
        transitions = sum(1 for i in range(1, 64) if binary_str[i] != binary_str[i-1])
        transition_ratio = transitions / 63
        if transition_ratio > 0.5:
            score += min(0.1, (transition_ratio - 0.5) * 0.5)
        
        return max(0.6, min(1.0, score))
    
    def historical_uniqueness(self, nonce: int) -> float:
        """Calcula unicidad basada en distancia con nonces históricos"""
        if not self.training_data or len(self.training_data) < 100:
            return random.uniform(0.85, 0.95)
        
        # Muestra aleatoria de nonces históricos
        sample_size = min(100, len(self.training_data))
        sample = random.sample(self.training_data, sample_size)
        
        # Calcular distancia promedio ponderada
        total_distance = 0
        bin_repr = bin(nonce)[2:].zfill(64)
        
        for item in sample:
            try:
                other_nonce = int(item['nonce'])
                other_bin = bin(other_nonce)[2:].zfill(64)
                distance = sum(1 for a, b in zip(bin_repr, other_bin) if a != b)
                total_distance += distance
            except:
                continue
        
        avg_distance = total_distance / sample_size
        uniqueness = avg_distance / 64
        return max(0.8, min(0.99, uniqueness))