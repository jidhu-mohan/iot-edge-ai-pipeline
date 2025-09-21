import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import tensorflow as tf
from tensorflow import keras
import os
from typing import Dict, Tuple, List, Optional
import time
import json

class EdgeMLModel:
    def __init__(self, model_type='sklearn'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_metadata = {}

    def create_neural_network(self, input_dim: int, output_dim: int = 1):
        """Create a lightweight neural network for edge deployment"""
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_dim=input_dim),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(output_dim, activation='sigmoid' if output_dim == 1 else 'softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy' if output_dim == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def create_random_forest(self, n_estimators: int = 50, max_depth: int = 10):
        """Create a lightweight Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        return self.model

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not created. Call create_neural_network() or create_random_forest() first.")

        # Store feature names if provided as DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        if self.model_type == 'tensorflow':
            # Split data for neural network
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=validation_split, random_state=42
            )

            # Train neural network
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )

            # Store training metadata
            self.model_metadata = {
                'train_accuracy': float(max(history.history['accuracy'])),
                'val_accuracy': float(max(history.history['val_accuracy'])),
                'epochs_trained': len(history.history['accuracy'])
            }

        else:
            # Train sklearn model
            self.model.fit(X_scaled, y)

            # Calculate training accuracy
            train_pred = self.model.predict(X_scaled)
            train_accuracy = accuracy_score(y, train_pred)

            self.model_metadata = {
                'train_accuracy': float(train_accuracy),
                'feature_importance': self.model.feature_importances_.tolist() if hasattr(self.model, 'feature_importances_') else None
            }

    def predict(self, X: np.ndarray) -> Dict:
        """Make prediction with timing and confidence"""
        if self.model is None:
            raise ValueError("Model not trained")

        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            X = X.values

        start_time = time.time()

        # Scale features
        X_scaled = self.scaler.transform(X)

        if self.model_type == 'tensorflow':
            # Neural network prediction
            predictions = self.model.predict(X_scaled, verbose=0)
            probabilities = predictions.flatten()
            binary_predictions = (probabilities > 0.5).astype(int)

        else:
            # Sklearn prediction
            binary_predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        return {
            'prediction': int(binary_predictions[0]) if len(binary_predictions) == 1 else binary_predictions.tolist(),
            'probability': float(probabilities[0]) if len(probabilities) == 1 else probabilities.tolist(),
            'inference_time_ms': inference_time,
            'model_type': self.model_type
        }

    def save_model(self, filepath: str):
        """Save model to disk"""
        model_dir = os.path.dirname(filepath)
        os.makedirs(model_dir, exist_ok=True)

        if self.model_type == 'tensorflow':
            # Save TensorFlow model
            self.model.save(f"{filepath}_tf_model")

            # Save scaler and metadata separately
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")

        else:
            # Save sklearn model and scaler
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metadata': self.model_metadata
            }, f"{filepath}.pkl")

        # Save metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'metadata': self.model_metadata
            }, f, indent=2)

    def load_model(self, filepath: str):
        """Load model from disk"""
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)

        self.model_type = metadata['model_type']
        self.feature_names = metadata['feature_names']
        self.model_metadata = metadata['metadata']

        if self.model_type == 'tensorflow':
            # Load TensorFlow model
            self.model = keras.models.load_model(f"{filepath}_tf_model")
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        else:
            # Load sklearn model
            saved_data = joblib.load(f"{filepath}.pkl")
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']

    def optimize_for_edge(self):
        """Optimize model for edge deployment"""
        if self.model_type == 'tensorflow':
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

            # Apply optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Quantization for smaller model size
            converter.target_spec.supported_types = [tf.float16]

            # Convert model
            tflite_model = converter.convert()

            return tflite_model
        else:
            # For sklearn models, already lightweight
            return self.model

class AnomalyDetectionModel(EdgeMLModel):
    def __init__(self):
        super().__init__(model_type='sklearn')
        self.threshold = 0.5

    def create_isolation_forest(self, contamination: float = 0.1):
        """Create Isolation Forest for anomaly detection"""
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        return self.model

    def train_anomaly_detector(self, X: np.ndarray):
        """Train anomaly detection model"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled)

        # Calculate anomaly scores for training data
        scores = self.model.decision_function(X_scaled)
        self.threshold = np.percentile(scores, 10)  # Bottom 10% as anomalies

    def detect_anomaly(self, X: np.ndarray) -> Dict:
        """Detect anomalies in data"""
        if self.model is None:
            raise ValueError("Model not trained")

        start_time = time.time()

        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get anomaly score
        anomaly_score = self.model.decision_function(X_scaled)[0]
        is_anomaly = anomaly_score < self.threshold

        inference_time = (time.time() - start_time) * 1000

        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'threshold': float(self.threshold),
            'inference_time_ms': inference_time
        }

class ModelValidator:
    def __init__(self):
        self.validation_results = {}

    def validate_model_performance(self, model: EdgeMLModel, test_data: Tuple) -> Dict:
        """Validate model performance"""
        X_test, y_test = test_data

        # Get predictions
        predictions = []
        probabilities = []
        inference_times = []

        for i in range(len(X_test)):
            result = model.predict(X_test[i:i+1])
            predictions.append(result['prediction'])
            probabilities.append(result['probability'])
            inference_times.append(result['inference_time_ms'])

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        avg_inference_time = np.mean(inference_times)
        max_inference_time = np.max(inference_times)

        return {
            'accuracy': float(accuracy),
            'avg_inference_time_ms': float(avg_inference_time),
            'max_inference_time_ms': float(max_inference_time),
            'predictions_count': len(predictions),
            'model_size_mb': self._calculate_model_size(model)
        }

    def _calculate_model_size(self, model: EdgeMLModel) -> float:
        """Estimate model size in MB"""
        if model.model_type == 'tensorflow':
            # Rough estimation for TensorFlow model
            total_params = model.model.count_params()
            size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        else:
            # For sklearn models, use joblib size estimation
            import pickle
            size_bytes = len(pickle.dumps(model.model))
            size_mb = size_bytes / (1024 * 1024)

        return size_mb

    def validate_edge_constraints(self, model: EdgeMLModel,
                                max_inference_time_ms: float = 100,
                                max_model_size_mb: float = 10) -> Dict:
        """Validate model meets edge deployment constraints"""
        # Test with dummy data
        dummy_input = np.random.randn(10, 10)  # 10 samples, 10 features

        results = []
        for i in range(len(dummy_input)):
            result = model.predict(dummy_input[i:i+1])
            results.append(result)

        avg_inference_time = np.mean([r['inference_time_ms'] for r in results])
        model_size = self._calculate_model_size(model)

        constraints_met = {
            'inference_time_ok': avg_inference_time <= max_inference_time_ms,
            'model_size_ok': model_size <= max_model_size_mb
        }

        return {
            'constraints_met': all(constraints_met.values()),
            'avg_inference_time_ms': float(avg_inference_time),
            'model_size_mb': float(model_size),
            'max_inference_time_ms': max_inference_time_ms,
            'max_model_size_mb': max_model_size_mb,
            'details': constraints_met
        }

if __name__ == "__main__":
    # Example usage
    print("Creating and testing edge ML model...")

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Create and train model
    model = EdgeMLModel(model_type='sklearn')
    model.create_random_forest(n_estimators=20, max_depth=5)
    model.train(X, y)

    # Test prediction
    test_sample = X[0:1]
    result = model.predict(test_sample)
    print(f"Prediction: {result}")

    # Validate model
    validator = ModelValidator()
    X_test, y_test = X[800:], y[800:]
    performance = validator.validate_model_performance(model, (X_test, y_test))
    print(f"Performance: {performance}")

    constraints = validator.validate_edge_constraints(model)
    print(f"Edge constraints: {constraints}")