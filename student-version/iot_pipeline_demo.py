#!/usr/bin/env python3
"""
IoT Edge AI Pipeline - Educational 
=======================================

This is a simplified, educational version of an IoT Edge AI Pipeline.
It demonstrates the complete flow from sensor data collection to AI inference
in a single, easy-to-understand script.

LEARNING OBJECTIVES:
1. Understand IoT sensor data simulation
2. Learn data preprocessing and feature extraction
3. See machine learning model training and inference
4. Observe real-time data processing patterns

PIPELINE FLOW:
[Sensors] → [Data Collection] → [Feature Engineering] → [ML Model] → [Prediction]
"""

import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class IoTSensor:
    """
    Simulates a single IoT sensor (like temperature, humidity, vibration)

    EDUCATIONAL NOTE:
    Real IoT sensors send data at regular intervals. This class simulates
    that behavior by generating realistic sensor readings with:
    - Normal operating ranges
    - Random noise (real sensors aren't perfectly accurate)
    - Occasional anomalies (equipment issues, environmental changes)
    """

    def __init__(self, sensor_id: str, sensor_type: str, min_val: float, max_val: float, unit: str):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        self.current_value = (min_val + max_val) / 2  # Start at middle value

    def read_sensor(self) -> Dict[str, Any]:
        """
        Simulate reading data from a physical sensor

        EDUCATIONAL NOTE:
        This method simulates what happens when you query a real sensor:
        1. Get current timestamp
        2. Read sensor value (with some randomness)
        3. Package data in a standard format
        """

        # Occasionally inject anomalies (05% chance for better demonstration)
        if random.random() < 0.05:
            # Create significant anomalies that go outside normal bounds
            anomaly_factor = random.choice([-1.0, 1.0])
            anomaly_value = self.current_value + anomaly_factor * (self.max_val - self.min_val) * 0.3
            self.current_value = anomaly_value
        else:
            # Add realistic variation to sensor readings
            noise = random.uniform(-0.1, 0.1) * (self.max_val - self.min_val)
            self.current_value += noise

            # Keep normal readings within realistic bounds
            self.current_value = max(self.min_val, min(self.max_val, self.current_value))

        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'timestamp': datetime.now().isoformat(),
            'value': round(self.current_value, 2),
            'unit': self.unit
        }

class DataProcessor:
    """
    Processes raw sensor data and extracts features for machine learning

    EDUCATIONAL NOTE:
    Raw sensor readings aren't directly useful for ML. We need to:
    1. Collect multiple readings over time
    2. Calculate statistical features (mean, std deviation, trends)
    3. Create features that help ML models make better predictions
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.sensor_history = {}  # Store recent readings for each sensor

    def add_reading(self, reading: Dict[str, Any]) -> Dict[str, float]:
        """
        Add a new sensor reading and extract features

        EDUCATIONAL NOTE:
        This is where "feature engineering" happens - converting raw data
        into meaningful features that machine learning models can use.
        """

        sensor_id = reading['sensor_id']

        # Initialize history for new sensors
        if sensor_id not in self.sensor_history:
            self.sensor_history[sensor_id] = []

        # Add new reading to history
        self.sensor_history[sensor_id].append(reading)

        # Keep only recent readings (sliding window)
        if len(self.sensor_history[sensor_id]) > self.window_size:
            self.sensor_history[sensor_id] = self.sensor_history[sensor_id][-self.window_size:]

        # Extract features if we have enough data
        if len(self.sensor_history[sensor_id]) >= 3:
            return self._extract_features(sensor_id)

        return {}

    def _extract_features(self, sensor_id: str) -> Dict[str, float]:
        """
        Extract statistical features from sensor history

        EDUCATIONAL NOTE:
        These features help ML models understand patterns:
        - mean: average value (is it running hot/cold?)
        - std: variability (is it stable or fluctuating?)
        - trend: is it increasing/decreasing over time?
        - min/max: extreme values in recent history
        """

        readings = self.sensor_history[sensor_id]
        values = [r['value'] for r in readings]

        features = {
            f'{sensor_id}_mean': np.mean(values),
            f'{sensor_id}_std': np.std(values),
            f'{sensor_id}_min': np.min(values),
            f'{sensor_id}_max': np.max(values),
            f'{sensor_id}_range': np.max(values) - np.min(values),
        }

        # Calculate trend (is it going up or down?)
        if len(values) >= 3:
            # Simple trend: difference between recent and older values
            recent_avg = np.mean(values[-3:])
            older_avg = np.mean(values[:3])
            features[f'{sensor_id}_trend'] = recent_avg - older_avg

        return features

class AnomalyDetector:
    """
    Machine Learning model to detect unusual sensor patterns

    EDUCATIONAL NOTE:
    Anomaly detection helps identify when equipment is behaving unusually.
    This could indicate:
    - Equipment failure
    - Environmental changes
    - Security issues
    - Maintenance needed
    """

    def __init__(self):
        self.model = IsolationForest(contamination=0.2, random_state=42, n_estimators=50)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = None  # Store feature order for consistency

    def train(self, training_data: List[Dict[str, float]]):
        """
        Train the anomaly detection model

        EDUCATIONAL NOTE:
        Training teaches the model what "normal" looks like.
        When it sees new data, it can identify what's unusual.
        """

        if not training_data:
            return

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(training_data)

        if df.empty:
            return

        # Store feature column order for consistency
        self.feature_columns = sorted(df.columns.tolist())

        # Reorder columns and fill missing values
        df_ordered = df[self.feature_columns].fillna(0)

        # Normalize features (important for ML models)
        X = self.scaler.fit_transform(df_ordered)

        # Train the model
        self.model.fit(X)
        self.is_trained = True
        print(f"✓ Anomaly detector trained on {len(training_data)} samples with {len(self.feature_columns)} features")

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict if current sensor readings are anomalous

        Returns:
        - is_anomaly: True if unusual pattern detected
        - anomaly_score: How unusual (lower = more unusual)
        - confidence: How confident the model is
        """

        if not self.is_trained or not features or not self.feature_columns:
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'confidence': 0.0}

        try:
            # Create feature vector in the same order as training
            feature_values = [features.get(col, 0.0) for col in self.feature_columns]
            X = np.array(feature_values).reshape(1, -1)

            # Normalize features using the same scaler from training
            X_scaled = self.scaler.transform(X)

            # Make prediction
            prediction = self.model.predict(X_scaled)[0]  # 1 = normal, -1 = anomaly
            anomaly_score = self.model.decision_function(X_scaled)[0]

            is_anomaly = prediction == -1
            confidence = abs(anomaly_score)  # Higher absolute value = more confident

            return {
                'is_anomaly': is_anomaly,
                'anomaly_score': float(anomaly_score),
                'confidence': float(confidence)
            }

        except Exception:
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'confidence': 0.0}

class IoTPipeline:
    """
    Main pipeline that orchestrates the entire IoT to AI flow

    EDUCATIONAL NOTE:
    This class brings everything together:
    1. Manages multiple sensors
    2. Processes incoming data
    3. Runs machine learning inference
    4. Provides insights and alerts
    """

    def __init__(self):
        self.sensors = {}
        self.data_processor = DataProcessor()
        self.anomaly_detector = AnomalyDetector()
        self.all_features = []  # For training the ML model

    def add_sensor(self, sensor_id: str, sensor_type: str, min_val: float, max_val: float, unit: str):
        """Add a new sensor to the pipeline"""
        self.sensors[sensor_id] = IoTSensor(sensor_id, sensor_type, min_val, max_val, unit)
        print(f"✓ Added {sensor_type} sensor: {sensor_id}")

    def collect_training_data(self, duration_seconds: int = 30):
        """
        Collect data to train the machine learning model

        EDUCATIONAL NOTE:
        Before we can detect anomalies, we need to show the model
        what "normal" operation looks like. This function collects
        normal sensor data for training.
        """

        print(f"\nCollecting training data for {duration_seconds} seconds...")
        print("This teaches the AI what normal sensor behavior looks like.")

        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < duration_seconds:
            # Read from all sensors
            for sensor in self.sensors.values():
                reading = sensor.read_sensor()
                features = self.data_processor.add_reading(reading)

                if features:  # Only add if we extracted features
                    self.all_features.append(features)
                    sample_count += 1

            time.sleep(0.5)  # Wait before next reading

        print(f"✓ Collected {sample_count} feature samples for training")

        # Train the anomaly detector
        if self.all_features:
            self.anomaly_detector.train(self.all_features)

    def run_real_time_monitoring(self, duration_seconds: int = 60):
        """
        Run real-time monitoring with AI inference

        EDUCATIONAL NOTE:
        This is the "production" phase where the system:
        1. Continuously reads sensor data
        2. Processes it in real-time
        3. Runs AI inference to detect issues
        4. Alerts when anomalies are found
        """

        print(f"\nStarting real-time monitoring for {duration_seconds} seconds...")
        print("The AI will now analyze sensor data and detect anomalies in real-time.\n")

        start_time = time.time()
        reading_count = 0
        anomaly_count = 0

        while time.time() - start_time < duration_seconds:
            # Read from all sensors
            for sensor in self.sensors.values():
                reading = sensor.read_sensor()
                features = self.data_processor.add_reading(reading)

                if features:
                    # Run AI inference
                    result = self.anomaly_detector.predict(features)

                    reading_count += 1

                    # Display results
                    status = "🚨 ANOMALY" if result['is_anomaly'] else "✅ Normal"
                    confidence = result['confidence']

                    print(f"[{reading['timestamp'][:19]}] "
                          f"{reading['sensor_id']}: {reading['value']}{reading['unit']} "
                          f"| {status} (confidence: {confidence:.2f})")

                    if result['is_anomaly']:
                        anomaly_count += 1
                        print(f"   ⚠️  Alert: Unusual pattern detected in {reading['sensor_id']}")

            time.sleep(1)  # Wait 1 second between readings

        # Summary
        print(f"\nMonitoring Summary:")
        print(f"   Total readings processed: {reading_count}")
        print(f"   Anomalies detected: {anomaly_count}")
        print(f"   Normal operation: {reading_count - anomaly_count} readings")

        if anomaly_count > 0:
            anomaly_rate = (anomaly_count / reading_count) * 100
            print(f"   Anomaly rate: {anomaly_rate:.1f}%")

def main():
    """
    Main function - demonstrates the complete IoT Edge AI pipeline

    EDUCATIONAL FLOW:
    1. Setup sensors (simulate IoT devices)
    2. Collect training data (teach AI what's normal)
    3. Run real-time monitoring (detect anomalies)
    4. Show results and insights
    """

    print("IoT Edge AI Pipeline - Educational Demo")
    print("=" * 50)
    print("\nThis demo shows how IoT sensor data flows through an AI pipeline:")
    print("1. IoT sensors generate realistic data")
    print("2. Data is processed and features are extracted")
    print("3. Machine learning detects unusual patterns")
    print("4. Real-time alerts are generated")

    # Create the pipeline
    pipeline = IoTPipeline()

    # Add different types of sensors (like you'd find in a factory)
    print("\nSetting up sensors...")
    pipeline.add_sensor('temp_001', 'temperature', 18.0, 28.0, '°C')
    pipeline.add_sensor('humid_001', 'humidity', 40.0, 70.0, '%')
    pipeline.add_sensor('vibration_001', 'vibration', 0.5, 3.0, 'm/s²')
    pipeline.add_sensor('pressure_001', 'pressure', 980.0, 1020.0, 'hPa')

    # Phase 1: Collect training data
    pipeline.collect_training_data(duration_seconds=15)

    # Phase 2: Real-time monitoring
    pipeline.run_real_time_monitoring(duration_seconds=30)

    print("\nLearning Points:")
    print("• Sensors generate continuous data streams")
    print("• Feature engineering converts raw data to ML-ready format")
    print("• Anomaly detection identifies unusual patterns automatically")
    print("• Real-time processing enables immediate alerts")
    print("• This approach scales to thousands of sensors in production")

    print("\nDemo completed! This shows the foundation of industrial IoT AI systems.")

if __name__ == "__main__":
    """
    EDUCATIONAL NOTE:
    This is the entry point. When you run this script, it will:
    1. Set up a simulated IoT environment
    2. Train an AI model
    3. Show real-time anomaly detection
    4. Demonstrate key concepts for students
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    except Exception as error:
        print(f"\nError: {error}")
        print("Check that you have the required packages installed:")
        print("pip install numpy pandas scikit-learn")