import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import json

class RealTimeDataProcessor:
    def __init__(self, window_size: int = 100, feature_window: int = 10):
        self.window_size = window_size
        self.feature_window = feature_window
        self.data_buffer = deque(maxlen=window_size)
        self.sensor_buffers = {}

    def add_data_point(self, data: Dict):
        """Add new data point to the processing pipeline"""
        self.data_buffer.append(data)

        # Maintain separate buffers for each sensor
        sensor_id = data.get('sensor_id')
        if sensor_id:
            if sensor_id not in self.sensor_buffers:
                self.sensor_buffers[sensor_id] = deque(maxlen=self.window_size)
            self.sensor_buffers[sensor_id].append(data)

    def get_sensor_features(self, sensor_id: str) -> Dict:
        """Extract features for a specific sensor"""
        if sensor_id not in self.sensor_buffers or len(self.sensor_buffers[sensor_id]) < self.feature_window:
            return {}

        buffer = list(self.sensor_buffers[sensor_id])[-self.feature_window:]
        values = [point['value'] for point in buffer]

        features = {
            f'{sensor_id}_mean': np.mean(values),
            f'{sensor_id}_std': np.std(values),
            f'{sensor_id}_min': np.min(values),
            f'{sensor_id}_max': np.max(values),
            f'{sensor_id}_median': np.median(values),
            f'{sensor_id}_trend': self._calculate_trend(values),
            f'{sensor_id}_anomaly_score': self._calculate_anomaly_score(values),
            f'{sensor_id}_current': values[-1]
        }

        return features

    def get_all_features(self) -> Dict:
        """Get features for all sensors"""
        all_features = {}

        for sensor_id in self.sensor_buffers:
            sensor_features = self.get_sensor_features(sensor_id)
            all_features.update(sensor_features)

        # Add cross-sensor features
        cross_features = self._calculate_cross_sensor_features()
        all_features.update(cross_features)

        return all_features

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of recent values"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope

    def _calculate_anomaly_score(self, values: List[float]) -> float:
        """Calculate anomaly score based on statistical deviation"""
        if len(values) < 3:
            return 0.0

        mean_val = np.mean(values[:-1])
        std_val = np.std(values[:-1])

        if std_val == 0:
            return 0.0

        z_score = abs((values[-1] - mean_val) / std_val)
        return min(z_score / 3.0, 1.0)  # Normalize to 0-1

    def _calculate_cross_sensor_features(self) -> Dict:
        """Calculate features across multiple sensors"""
        features = {}

        # Get current values for all sensors
        current_values = {}
        for sensor_id, buffer in self.sensor_buffers.items():
            if buffer:
                current_values[sensor_id] = buffer[-1]['value']

        if len(current_values) > 1:
            values = list(current_values.values())
            features['cross_sensor_std'] = np.std(values)
            features['cross_sensor_mean'] = np.mean(values)

            # Temperature-humidity correlation if both present
            temp_sensors = [k for k in current_values.keys() if 'temp' in k]
            hum_sensors = [k for k in current_values.keys() if 'hum' in k]

            if temp_sensors and hum_sensors:
                temp_val = current_values[temp_sensors[0]]
                hum_val = current_values[hum_sensors[0]]
                features['temp_hum_ratio'] = temp_val / (hum_val + 1e-6)

        return features

class FeatureExtractor:
    def __init__(self):
        self.feature_history = []

    def extract_time_features(self, timestamp: str) -> Dict:
        """Extract time-based features"""
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        return {
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'is_weekend': 1 if dt.weekday() >= 5 else 0,
            'is_business_hours': 1 if 9 <= dt.hour <= 17 else 0,
            'minute': dt.minute,
            'second': dt.second
        }

    def extract_statistical_features(self, values: List[float], prefix: str = '') -> Dict:
        """Extract statistical features from a series of values"""
        if not values:
            return {}

        features = {
            f'{prefix}mean': np.mean(values),
            f'{prefix}std': np.std(values),
            f'{prefix}var': np.var(values),
            f'{prefix}min': np.min(values),
            f'{prefix}max': np.max(values),
            f'{prefix}median': np.median(values),
            f'{prefix}q25': np.percentile(values, 25),
            f'{prefix}q75': np.percentile(values, 75),
            f'{prefix}iqr': np.percentile(values, 75) - np.percentile(values, 25),
            f'{prefix}skewness': self._calculate_skewness(values),
            f'{prefix}kurtosis': self._calculate_kurtosis(values)
        }

        return features

    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of values"""
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return 0.0

        return np.mean(((values - mean_val) / std_val) ** 3)

    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of values"""
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return 0.0

        return np.mean(((values - mean_val) / std_val) ** 4) - 3

class DataValidator:
    def __init__(self):
        self.validation_rules = {}

    def add_validation_rule(self, sensor_type: str, min_val: float, max_val: float):
        """Add validation rule for sensor type"""
        self.validation_rules[sensor_type] = {'min': min_val, 'max': max_val}

    def validate_data_point(self, data: Dict) -> Tuple[bool, str]:
        """Validate a single data point"""
        sensor_type = data.get('sensor_type')
        value = data.get('value')

        if sensor_type is None or value is None:
            return False, "Missing sensor_type or value"

        if sensor_type in self.validation_rules:
            rules = self.validation_rules[sensor_type]
            if value < rules['min'] or value > rules['max']:
                return False, f"Value {value} outside valid range [{rules['min']}, {rules['max']}]"

        # Check for timestamp
        if 'timestamp' not in data:
            return False, "Missing timestamp"

        try:
            datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            return False, "Invalid timestamp format"

        return True, "Valid"

    def clean_data_batch(self, data_batch: List[Dict]) -> List[Dict]:
        """Clean a batch of data points"""
        cleaned_data = []

        for data_point in data_batch:
            is_valid, message = self.validate_data_point(data_point)
            if is_valid:
                cleaned_data.append(data_point)
            else:
                print(f"Invalid data point: {message}")

        return cleaned_data

class DataPipeline:
    def __init__(self, window_size: int = 100):
        self.processor = RealTimeDataProcessor(window_size)
        self.feature_extractor = FeatureExtractor()
        self.validator = DataValidator()

        # Setup default validation rules
        self._setup_default_validation()

    def _setup_default_validation(self):
        """Setup default validation rules for common sensor types"""
        self.validator.add_validation_rule('temperature', -50.0, 100.0)
        self.validator.add_validation_rule('humidity', 0.0, 100.0)
        self.validator.add_validation_rule('pressure', 50.0, 200.0)
        self.validator.add_validation_rule('vibration', 0.0, 50.0)

    def process_data_point(self, data: Dict) -> Optional[Dict]:
        """Process a single data point through the pipeline"""
        # Validate data
        is_valid, message = self.validator.validate_data_point(data)
        if not is_valid:
            print(f"Validation failed: {message}")
            return None

        # Add to processor
        self.processor.add_data_point(data)

        # Extract features
        features = self.processor.get_all_features()

        # Add time features
        time_features = self.feature_extractor.extract_time_features(data['timestamp'])
        features.update(time_features)

        # Add metadata
        features['data_timestamp'] = data['timestamp']
        features['processed_timestamp'] = datetime.now().isoformat()

        return features

    def get_feature_vector(self, sensor_list: List[str] = None) -> np.ndarray:
        """Get feature vector for ML model input"""
        features = self.processor.get_all_features()

        if sensor_list:
            # Filter features for specific sensors
            filtered_features = {}
            for key, value in features.items():
                if any(sensor in key for sensor in sensor_list):
                    filtered_features[key] = value
            features = filtered_features

        # Convert to numpy array
        return np.array(list(features.values()))

if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline()

    # Simulate some data processing
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
        }
    ]

    for data_point in sample_data:
        features = pipeline.process_data_point(data_point)
        if features:
            print(f"Extracted {len(features)} features")
            print(f"Sample features: {list(features.keys())[:5]}")