#!/usr/bin/env python3
"""
Quick example script to demonstrate the IoT Edge AI Pipeline
"""

import time
import logging
import numpy as np
from datetime import datetime

from src.sensors import setup_default_sensors, SensorDataLogger
from src.preprocessing import DataPipeline
from src.ml_models import EdgeMLModel, AnomalyDetectionModel
from src.inference import RealTimeInferenceEngine, EdgeInferenceAPI

def run_simple_example():
    """Run a simple example of the pipeline without MQTT"""
    print("=== IoT Edge AI Pipeline - Simple Example ===\n")

    # Setup logging - suppress inference errors for cleaner demo output
    logging.basicConfig(level=logging.INFO)

    # Suppress inference engine errors for demo
    logging.getLogger('src.inference.inference_engine').setLevel(logging.CRITICAL)

    # 1. Create sensor simulator
    print("1. Setting up sensor simulation...")
    simulator = setup_default_sensors()
    logger = SensorDataLogger('data/raw/example_sensor_data.json')

    # 2. Setup data processing pipeline
    print("2. Setting up data processing pipeline...")
    data_pipeline = DataPipeline()

    # 3. Create and train a simple model
    print("3. Creating and training demo models...")

    # Create some training data
    import numpy as np
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    # Classification model
    clf_model = EdgeMLModel(model_type='sklearn')
    clf_model.create_random_forest(n_estimators=20, max_depth=5)
    clf_model.train(X_train, y_train)

    # Anomaly detection model
    anomaly_model = AnomalyDetectionModel()
    anomaly_model.create_isolation_forest(contamination=0.1)
    anomaly_model.train_anomaly_detector(X_train)

    # 4. Setup inference engine
    print("4. Setting up inference engine...")
    inference_engine = RealTimeInferenceEngine()

    # Add models to engine (simulate loading)
    inference_engine.models = {
        'classifier': {'model': clf_model, 'type': 'classification'},
        'anomaly_detector': {'model': anomaly_model, 'type': 'anomaly_detection'}
    }

    api = EdgeInferenceAPI(inference_engine)

    # 5. Process some sensor data
    print("5. Processing sensor data and running inference...\n")

    data_points = []
    results = []

    def data_callback(reading):
        # Log raw data
        logger.log_reading(reading)

        # Convert to dict for processing
        sensor_data = {
            'sensor_id': reading.sensor_id,
            'sensor_type': reading.sensor_type,
            'timestamp': reading.timestamp,
            'value': reading.value,
            'unit': reading.unit,
            'location': reading.location,
            'metadata': reading.metadata
        }

        data_points.append(sensor_data)

        # Process through inference API
        result = api.process_with_alerts(sensor_data)
        if result['status'] == 'success':
            results.append(result)

            # Print interesting results
            inference_result = result['result']
            print(f"Sensor: {inference_result['sensor_id']} | "
                  f"Value: {sensor_data['value']} {sensor_data['unit']} | "
                  f"Inference: {inference_result['inference_time_ms']:.1f}ms")

            # Print any alerts
            if result['alerts']:
                for alert in result['alerts']:
                    print(f"  🚨 ALERT: {alert['type']} - {alert.get('details', '')}")

    # Start simulation
    simulator.start_simulation(data_callback)

    try:
        print("Running simulation for 10 seconds...\n")
        time.sleep(10)

    finally:
        simulator.stop_simulation()

    # 6. Show results summary
    print(f"\n=== Results Summary ===")
    print(f"Data points collected: {len(data_points)}")
    print(f"Successful inferences: {len(results)}")

    if results:
        inference_times = [r['result']['inference_time_ms'] for r in results]
        print(f"Average inference time: {np.mean(inference_times):.2f}ms")
        print(f"Max inference time: {np.max(inference_times):.2f}ms")

        # Count alerts
        total_alerts = sum(len(r['alerts']) for r in results)
        print(f"Total alerts generated: {total_alerts}")

    # Show engine stats
    stats = inference_engine.get_stats()
    print(f"\nEngine stats: {stats}")

    print("\n=== Example Complete ===")
    print("Check 'data/raw/example_sensor_data.json' for logged sensor data")

def run_feature_demo():
    """Demonstrate feature extraction capabilities"""
    print("\n=== Feature Extraction Demo ===")

    # Create data pipeline
    pipeline = DataPipeline()

    # Generate sample data from different sensors
    sample_data = [
        {
            'sensor_id': 'temp_001',
            'sensor_type': 'temperature',
            'timestamp': datetime.now().isoformat(),
            'value': 25.5,
            'unit': '°C',
            'location': 'factory_floor'
        },
        {
            'sensor_id': 'hum_001',
            'sensor_type': 'humidity',
            'timestamp': datetime.now().isoformat(),
            'value': 65.2,
            'unit': '%',
            'location': 'factory_floor'
        },
        {
            'sensor_id': 'vib_001',
            'sensor_type': 'vibration',
            'timestamp': datetime.now().isoformat(),
            'value': 2.1,
            'unit': 'm/s²',
            'location': 'motor_1'
        }
    ]

    print("Processing sample sensor data...")
    for i, data in enumerate(sample_data * 5):  # Process multiple times to build history
        # Add some variation
        data = data.copy()
        data['value'] += np.random.normal(0, 0.1)
        data['timestamp'] = datetime.now().isoformat()

        features = pipeline.process_data_point(data)

        if features and i == len(sample_data) * 5 - 1:  # Show features from last processing
            print(f"\nExtracted {len(features)} features:")
            for key, value in list(features.items())[:10]:  # Show first 10
                print(f"  {key}: {value}")
            print("  ...")

if __name__ == "__main__":
    try:
        run_simple_example()
        run_feature_demo()

    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()