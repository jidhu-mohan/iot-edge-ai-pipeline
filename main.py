#!/usr/bin/env python3
"""
IoT Edge AI Pipeline - Main Application
Real-time sensor data processing with edge AI inference
"""

import argparse
import time
import yaml
import logging
from datetime import datetime
import signal
import sys

from src.sensors import IoTSensorSimulator, MQTTSensorPublisher, MQTTDataIngestion
from src.preprocessing import DataPipeline
from src.ml_models import EdgeMLModel, AnomalyDetectionModel, ModelValidator
from src.inference import RealTimeInferenceEngine, EdgeInferenceAPI, InferenceMonitor

class IoTEdgeAIPipeline:
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config = self.load_config(config_path)
        self.components = {}
        self.running = False

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()

    def get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'mqtt': {'broker_host': 'localhost', 'broker_port': 1883},
            'data_processing': {'window_size': 100, 'feature_window': 10},
            'inference': {'max_inference_time_ms': 100},
            'monitoring': {'log_level': 'INFO', 'stats_interval_seconds': 30},
            'sensors': {'simulation': {'enabled': True, 'sensors': []}}
        }

    def setup_sensor_simulation(self):
        """Setup IoT sensor simulation"""
        if not self.config['sensors']['simulation']['enabled']:
            return

        self.logger.info("Setting up sensor simulation...")

        # Create sensor simulator
        simulator = IoTSensorSimulator()

        # Add sensors from config
        for sensor_config in self.config['sensors']['simulation']['sensors']:
            simulator.add_sensor(
                sensor_id=sensor_config['id'],
                sensor_type=sensor_config['type'],
                location=sensor_config['location'],
                min_val=sensor_config['min_val'],
                max_val=sensor_config['max_val'],
                unit=sensor_config['unit'],
                sampling_rate=sensor_config['sampling_rate'],
                noise_factor=sensor_config['noise_factor']
            )

        self.components['simulator'] = simulator

        # Setup MQTT publisher
        mqtt_config = self.config['mqtt']
        publisher = MQTTSensorPublisher(
            broker_host=mqtt_config['broker_host'],
            broker_port=mqtt_config['broker_port']
        )

        self.components['mqtt_publisher'] = publisher

    def setup_inference_engine(self):
        """Setup the inference engine"""
        self.logger.info("Setting up inference engine...")

        # Create inference engine
        engine = RealTimeInferenceEngine()

        # Load models if specified in config
        if 'models' in self.config and 'default_models' in self.config['models']:
            for model_config in self.config['models']['default_models']:
                # For demo purposes, create and save a simple model if it doesn't exist
                self.create_demo_model(model_config)

        # Setup API wrapper
        api = EdgeInferenceAPI(engine)

        # Set alert thresholds from config
        if 'alert_thresholds' in self.config['inference']:
            for threshold_name, threshold_value in self.config['inference']['alert_thresholds'].items():
                if threshold_name != 'anomaly_score':  # Skip non-model thresholds
                    continue
                api.set_alert_threshold('anomaly_detector', threshold_value)

        self.components['inference_engine'] = engine
        self.components['inference_api'] = api

        # Setup monitoring
        if self.config['monitoring']['performance_monitoring']:
            monitor = InferenceMonitor(engine)
            monitor.start_monitoring(
                interval_seconds=self.config['monitoring']['stats_interval_seconds']
            )
            self.components['monitor'] = monitor

    def create_demo_model(self, model_config: dict):
        """Create a demo model for testing purposes"""
        import os
        import numpy as np

        model_path = model_config['path']
        model_dir = os.path.dirname(model_path)

        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Check if model already exists
        if os.path.exists(f"{model_path}_metadata.json"):
            self.logger.info(f"Model {model_config['name']} already exists")
            return

        self.logger.info(f"Creating demo model: {model_config['name']}")

        # Generate dummy training data
        np.random.seed(42)
        X = np.random.randn(1000, 10)  # 1000 samples, 10 features
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple binary classification

        # Create and train model
        if model_config['type'] == 'anomaly_detection':
            model = AnomalyDetectionModel()
            model.create_isolation_forest(contamination=0.1)
            model.train_anomaly_detector(X)
        else:
            model = EdgeMLModel(model_type='sklearn')
            model.create_random_forest(n_estimators=20, max_depth=5)
            model.train(X, y)

        # Save model
        model.save_model(model_path)
        self.logger.info(f"Saved demo model to {model_path}")

    def setup_data_pipeline(self):
        """Setup MQTT data ingestion and processing"""
        self.logger.info("Setting up data pipeline...")

        mqtt_config = self.config['mqtt']
        ingestion = MQTTDataIngestion(
            broker_host=mqtt_config['broker_host'],
            broker_port=mqtt_config['broker_port']
        )

        # Connect inference engine to MQTT data
        if 'inference_engine' in self.components:
            def data_callback(sensor_data):
                result = self.components['inference_api'].process_with_alerts(sensor_data)
                if result['status'] == 'success':
                    self.logger.debug(f"Processed data from {result['result']['sensor_id']}")

                    # Log alerts
                    if result['alerts']:
                        for alert in result['alerts']:
                            self.logger.warning(f"ALERT: {alert}")

            ingestion.add_callback(data_callback)

        self.components['mqtt_ingestion'] = ingestion

    def start_simulation(self):
        """Start sensor simulation and MQTT publishing"""
        if 'simulator' not in self.components or 'mqtt_publisher' not in self.components:
            self.logger.warning("Simulation components not setup")
            return

        simulator = self.components['simulator']
        publisher = self.components['mqtt_publisher']

        # Connect MQTT publisher
        publisher.connect()

        # Setup callback to publish sensor data
        def publish_callback(reading):
            success = publisher.publish_sensor_data(reading)
            if success:
                self.logger.debug(f"Published: {reading.sensor_id} = {reading.value} {reading.unit}")

        # Start simulation
        simulator.start_simulation(publish_callback)
        self.logger.info("Sensor simulation started")

    def start_inference(self):
        """Start inference pipeline"""
        if 'mqtt_ingestion' not in self.components:
            self.logger.warning("MQTT ingestion not setup")
            return

        # Connect MQTT ingestion
        ingestion = self.components['mqtt_ingestion']
        ingestion.connect()
        self.logger.info("Inference pipeline started")

    def run(self):
        """Run the complete pipeline"""
        self.logger.info("Starting IoT Edge AI Pipeline...")

        try:
            # Setup all components
            self.setup_sensor_simulation()
            self.setup_inference_engine()
            self.setup_data_pipeline()

            # Start services
            self.start_simulation()
            self.start_inference()

            self.running = True
            self.logger.info("Pipeline running. Press Ctrl+C to stop.")

            # Keep running
            while self.running:
                time.sleep(1)

                # Print periodic stats
                if hasattr(self, '_last_stats_time'):
                    if time.time() - self._last_stats_time > 60:  # Every minute
                        self.print_stats()
                        self._last_stats_time = time.time()
                else:
                    self._last_stats_time = time.time()

        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
            self.stop()

    def print_stats(self):
        """Print pipeline statistics"""
        if 'inference_engine' in self.components:
            stats = self.components['inference_engine'].get_stats()
            self.logger.info(f"Pipeline Stats: {stats}")

    def stop(self):
        """Stop the pipeline"""
        self.running = False

        # Stop simulation
        if 'simulator' in self.components:
            self.components['simulator'].stop_simulation()

        # Disconnect MQTT
        if 'mqtt_publisher' in self.components:
            self.components['mqtt_publisher'].disconnect()

        if 'mqtt_ingestion' in self.components:
            self.components['mqtt_ingestion'].disconnect()

        # Stop monitoring
        if 'monitor' in self.components:
            self.components['monitor'].stop_monitoring()

        self.logger.info("Pipeline stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='IoT Edge AI Pipeline')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', choices=['full', 'inference', 'simulation'],
                       default='full', help='Run mode')

    args = parser.parse_args()

    # Create pipeline
    pipeline = IoTEdgeAIPipeline(config_path=args.config)

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutdown signal received...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run pipeline
    if args.mode == 'full':
        pipeline.run()
    elif args.mode == 'simulation':
        pipeline.setup_sensor_simulation()
        pipeline.start_simulation()
        while True:
            time.sleep(1)
    elif args.mode == 'inference':
        pipeline.setup_inference_engine()
        pipeline.setup_data_pipeline()
        pipeline.start_inference()
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()