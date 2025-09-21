import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
import numpy as np
from dataclasses import dataclass
import threading
import queue
import logging

from ..preprocessing import DataPipeline
from ..ml_models import EdgeMLModel, AnomalyDetectionModel
from ..sensors import MQTTDataIngestion

@dataclass
class InferenceResult:
    timestamp: str
    sensor_id: str
    prediction: Dict
    features: Dict
    inference_time_ms: float
    model_version: str

class RealTimeInferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.models = {}
        self.data_pipeline = DataPipeline()
        self.inference_queue = queue.Queue(maxsize=1000)
        self.results_queue = queue.Queue(maxsize=1000)
        self.running = False
        self.callbacks = []

        # Performance monitoring
        self.stats = {
            'total_inferences': 0,
            'avg_inference_time': 0.0,
            'max_inference_time': 0.0,
            'anomalies_detected': 0,
            'last_inference_time': None
        }

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_name: str, model_path: str, model_type: str = 'classification'):
        """Load a trained model for inference"""
        try:
            if model_type == 'anomaly_detection':
                model = AnomalyDetectionModel()
            else:
                model = EdgeMLModel()

            model.load_model(model_path)
            self.models[model_name] = {
                'model': model,
                'type': model_type,
                'loaded_at': datetime.now().isoformat()
            }

            self.logger.info(f"Loaded model '{model_name}' from {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model '{model_name}': {e}")
            return False

    def add_callback(self, callback: Callable[[InferenceResult], None]):
        """Add callback for inference results"""
        self.callbacks.append(callback)

    def process_sensor_data(self, sensor_data: Dict) -> Optional[InferenceResult]:
        """Process sensor data and run inference"""
        try:
            start_time = time.time()

            # Process data through pipeline
            features = self.data_pipeline.process_data_point(sensor_data)
            if features is None:
                return None

            # Run inference with all loaded models
            predictions = {}
            total_inference_time = 0

            for model_name, model_info in self.models.items():
                model = model_info['model']
                model_type = model_info['type']

                # Convert features to numpy array
                feature_vector = self._features_to_vector(features)

                if model_type == 'anomaly_detection':
                    result = model.detect_anomaly(feature_vector)
                else:
                    result = model.predict(feature_vector)

                predictions[model_name] = result
                total_inference_time += result.get('inference_time_ms', 0)

            # Create inference result
            inference_result = InferenceResult(
                timestamp=datetime.now().isoformat(),
                sensor_id=sensor_data.get('sensor_id', 'unknown'),
                prediction=predictions,
                features=features,
                inference_time_ms=total_inference_time,
                model_version=f"models_count_{len(self.models)}"
            )

            # Update statistics
            self._update_stats(inference_result)

            # Call callbacks
            for callback in self.callbacks:
                try:
                    callback(inference_result)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")

            return inference_result

        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return None

    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dictionary to numpy vector"""
        # Filter out non-numeric features
        numeric_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                numeric_features[key] = value

        # Sort keys for consistent ordering
        sorted_keys = sorted(numeric_features.keys())
        vector = np.array([numeric_features[key] for key in sorted_keys])

        return vector.reshape(1, -1)

    def _update_stats(self, result: InferenceResult):
        """Update inference statistics"""
        self.stats['total_inferences'] += 1
        self.stats['last_inference_time'] = result.timestamp

        # Update average inference time
        current_avg = self.stats['avg_inference_time']
        total_count = self.stats['total_inferences']
        new_avg = ((current_avg * (total_count - 1)) + result.inference_time_ms) / total_count
        self.stats['avg_inference_time'] = new_avg

        # Update max inference time
        if result.inference_time_ms > self.stats['max_inference_time']:
            self.stats['max_inference_time'] = result.inference_time_ms

        # Count anomalies
        for model_name, prediction in result.prediction.items():
            if isinstance(prediction, dict) and prediction.get('is_anomaly', False):
                self.stats['anomalies_detected'] += 1

    def start_mqtt_processing(self, mqtt_broker: str = 'localhost', mqtt_port: int = 1883):
        """Start processing MQTT data in real-time"""
        self.mqtt_ingestion = MQTTDataIngestion(mqtt_broker, mqtt_port)

        def mqtt_callback(data):
            result = self.process_sensor_data(data)
            if result:
                self.results_queue.put(result)

        self.mqtt_ingestion.add_callback(mqtt_callback)

        try:
            self.mqtt_ingestion.connect()
            self.running = True
            self.logger.info("Started MQTT processing")
        except Exception as e:
            self.logger.error(f"Failed to start MQTT processing: {e}")

    def stop_mqtt_processing(self):
        """Stop MQTT processing"""
        self.running = False
        if hasattr(self, 'mqtt_ingestion'):
            self.mqtt_ingestion.disconnect()
        self.logger.info("Stopped MQTT processing")

    def get_stats(self) -> Dict:
        """Get inference engine statistics"""
        return self.stats.copy()

    def get_recent_results(self, count: int = 10) -> List[InferenceResult]:
        """Get recent inference results"""
        results = []
        temp_results = []

        # Drain queue
        while not self.results_queue.empty() and len(temp_results) < count:
            try:
                result = self.results_queue.get_nowait()
                temp_results.append(result)
            except queue.Empty:
                break

        # Return most recent results
        return temp_results[:count]

class EdgeInferenceAPI:
    def __init__(self, inference_engine: RealTimeInferenceEngine):
        self.engine = inference_engine
        self.alert_thresholds = {}

    def set_alert_threshold(self, model_name: str, threshold: float):
        """Set alert threshold for a model"""
        self.alert_thresholds[model_name] = threshold

    def check_alerts(self, result: InferenceResult) -> List[Dict]:
        """Check if any alerts should be triggered"""
        alerts = []

        for model_name, prediction in result.prediction.items():
            # Anomaly detection alerts
            if isinstance(prediction, dict) and prediction.get('is_anomaly', False):
                alerts.append({
                    'type': 'anomaly',
                    'model': model_name,
                    'sensor_id': result.sensor_id,
                    'timestamp': result.timestamp,
                    'details': prediction
                })

            # Probability threshold alerts
            if model_name in self.alert_thresholds:
                prob = prediction.get('probability', 0)
                if prob > self.alert_thresholds[model_name]:
                    alerts.append({
                        'type': 'threshold',
                        'model': model_name,
                        'sensor_id': result.sensor_id,
                        'timestamp': result.timestamp,
                        'probability': prob,
                        'threshold': self.alert_thresholds[model_name]
                    })

        return alerts

    def process_with_alerts(self, sensor_data: Dict) -> Dict:
        """Process data and check for alerts"""
        result = self.engine.process_sensor_data(sensor_data)
        if result is None:
            return {'status': 'error', 'message': 'Failed to process data'}

        alerts = self.check_alerts(result)

        return {
            'status': 'success',
            'result': {
                'timestamp': result.timestamp,
                'sensor_id': result.sensor_id,
                'predictions': result.prediction,
                'inference_time_ms': result.inference_time_ms
            },
            'alerts': alerts,
            'feature_count': len(result.features)
        }

class InferenceMonitor:
    def __init__(self, engine: RealTimeInferenceEngine):
        self.engine = engine
        self.monitoring_active = False
        self.monitor_thread = None

    def start_monitoring(self, interval_seconds: int = 30):
        """Start monitoring inference performance"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitoring_loop(self, interval_seconds: int):
        """Monitoring loop"""
        while self.monitoring_active:
            stats = self.engine.get_stats()
            self._log_stats(stats)
            time.sleep(interval_seconds)

    def _log_stats(self, stats: Dict):
        """Log performance statistics"""
        logging.info(f"Inference Stats: {json.dumps(stats, indent=2)}")

        # Check for performance issues
        if stats['avg_inference_time'] > 100:  # 100ms threshold
            logging.warning(f"High average inference time: {stats['avg_inference_time']:.2f}ms")

        if stats['max_inference_time'] > 500:  # 500ms threshold
            logging.warning(f"High max inference time: {stats['max_inference_time']:.2f}ms")

if __name__ == "__main__":
    # Example usage
    print("Setting up inference engine...")

    # Create inference engine
    engine = RealTimeInferenceEngine()

    # Example callback for results
    def result_callback(result: InferenceResult):
        print(f"Inference result for {result.sensor_id}: {result.prediction}")

    engine.add_callback(result_callback)

    # Create API wrapper
    api = EdgeInferenceAPI(engine)
    api.set_alert_threshold('anomaly_model', 0.8)

    # Example data processing
    sample_data = {
        'sensor_id': 'temp_001',
        'sensor_type': 'temperature',
        'timestamp': datetime.now().isoformat(),
        'value': 25.5,
        'unit': '°C',
        'location': 'factory_floor'
    }

    result = api.process_with_alerts(sample_data)
    print(f"API Result: {result}")

    # Start monitoring
    monitor = InferenceMonitor(engine)
    monitor.start_monitoring(interval_seconds=10)

    print("Inference engine ready!")