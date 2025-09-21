import unittest
import numpy as np
from datetime import datetime

from src.preprocessing import RealTimeDataProcessor, FeatureExtractor, DataValidator, DataPipeline

class TestRealTimeDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = RealTimeDataProcessor(window_size=10, feature_window=5)

    def test_add_data_point(self):
        """Test adding data points to processor"""
        data = {
            'sensor_id': 'test_sensor',
            'value': 25.5,
            'timestamp': datetime.now().isoformat()
        }

        self.processor.add_data_point(data)

        # Check data was added to main buffer
        self.assertEqual(len(self.processor.data_buffer), 1)

        # Check sensor-specific buffer was created
        self.assertIn('test_sensor', self.processor.sensor_buffers)
        self.assertEqual(len(self.processor.sensor_buffers['test_sensor']), 1)

    def test_feature_extraction(self):
        """Test feature extraction for sensor data"""
        # Add multiple data points
        for i in range(6):
            data = {
                'sensor_id': 'test_sensor',
                'value': 20.0 + i,  # Increasing values
                'timestamp': datetime.now().isoformat()
            }
            self.processor.add_data_point(data)

        features = self.processor.get_sensor_features('test_sensor')

        # Should have extracted features
        self.assertGreater(len(features), 0)

        # Check specific features exist
        self.assertIn('test_sensor_mean', features)
        self.assertIn('test_sensor_std', features)
        self.assertIn('test_sensor_trend', features)

        # Trend should be positive (increasing values)
        self.assertGreater(features['test_sensor_trend'], 0)

    def test_cross_sensor_features(self):
        """Test cross-sensor feature calculation"""
        # Add data for multiple sensors
        sensors = ['temp_001', 'hum_001']
        for sensor_id in sensors:
            for i in range(6):
                data = {
                    'sensor_id': sensor_id,
                    'value': 20.0 + i,
                    'timestamp': datetime.now().isoformat()
                }
                self.processor.add_data_point(data)

        features = self.processor.get_all_features()

        # Should have cross-sensor features
        self.assertIn('cross_sensor_std', features)
        self.assertIn('cross_sensor_mean', features)

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_time_features(self):
        """Test time-based feature extraction"""
        timestamp = "2023-12-01T14:30:45"
        features = self.extractor.extract_time_features(timestamp)

        self.assertIn('hour', features)
        self.assertIn('day_of_week', features)
        self.assertIn('is_weekend', features)
        self.assertIn('is_business_hours', features)

        self.assertEqual(features['hour'], 14)
        self.assertEqual(features['is_business_hours'], 1)  # 14:30 is business hours

    def test_statistical_features(self):
        """Test statistical feature extraction"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        features = self.extractor.extract_statistical_features(values, prefix='test_')

        self.assertIn('test_mean', features)
        self.assertIn('test_std', features)
        self.assertIn('test_median', features)

        self.assertAlmostEqual(features['test_mean'], 5.5)
        self.assertAlmostEqual(features['test_median'], 5.5)

class TestDataValidator(unittest.TestCase):
    def setUp(self):
        self.validator = DataValidator()
        self.validator.add_validation_rule('temperature', -10.0, 50.0)

    def test_valid_data(self):
        """Test validation of valid data"""
        data = {
            'sensor_type': 'temperature',
            'value': 25.5,
            'timestamp': datetime.now().isoformat()
        }

        is_valid, message = self.validator.validate_data_point(data)

        self.assertTrue(is_valid)
        self.assertEqual(message, "Valid")

    def test_invalid_value_range(self):
        """Test validation of out-of-range values"""
        data = {
            'sensor_type': 'temperature',
            'value': 100.0,  # Too high
            'timestamp': datetime.now().isoformat()
        }

        is_valid, message = self.validator.validate_data_point(data)

        self.assertFalse(is_valid)
        self.assertIn("outside valid range", message)

    def test_missing_fields(self):
        """Test validation with missing required fields"""
        data = {
            'sensor_type': 'temperature'
            # Missing value and timestamp
        }

        is_valid, message = self.validator.validate_data_point(data)

        self.assertFalse(is_valid)
        self.assertIn("Missing", message)

    def test_clean_data_batch(self):
        """Test cleaning a batch of data"""
        data_batch = [
            {
                'sensor_type': 'temperature',
                'value': 25.5,
                'timestamp': datetime.now().isoformat()
            },
            {
                'sensor_type': 'temperature',
                'value': 100.0,  # Invalid - too high
                'timestamp': datetime.now().isoformat()
            },
            {
                'sensor_type': 'temperature',
                'value': 20.0,
                'timestamp': datetime.now().isoformat()
            }
        ]

        cleaned_data = self.validator.clean_data_batch(data_batch)

        # Should have 2 valid data points (1st and 3rd)
        self.assertEqual(len(cleaned_data), 2)
        self.assertEqual(cleaned_data[0]['value'], 25.5)
        self.assertEqual(cleaned_data[1]['value'], 20.0)

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = DataPipeline(window_size=10)

    def test_pipeline_processing(self):
        """Test complete pipeline processing"""
        data = {
            'sensor_id': 'test_sensor',
            'sensor_type': 'temperature',
            'value': 25.5,
            'timestamp': datetime.now().isoformat(),
            'unit': '°C',
            'location': 'test_location'
        }

        features = self.pipeline.process_data_point(data)

        # Should have processed successfully
        self.assertIsNotNone(features)

        # Should have timestamp features
        self.assertIn('hour', features)
        self.assertIn('data_timestamp', features)
        self.assertIn('processed_timestamp', features)

    def test_invalid_data_handling(self):
        """Test pipeline handling of invalid data"""
        invalid_data = {
            'sensor_id': 'test_sensor',
            'sensor_type': 'temperature',
            'value': 200.0,  # Way too high for temperature
            'timestamp': datetime.now().isoformat()
        }

        features = self.pipeline.process_data_point(invalid_data)

        # Should return None for invalid data
        self.assertIsNone(features)

    def test_feature_vector_generation(self):
        """Test feature vector generation for ML models"""
        # Add some data points
        for i in range(10):
            data = {
                'sensor_id': 'test_sensor',
                'sensor_type': 'temperature',
                'value': 20.0 + i,
                'timestamp': datetime.now().isoformat()
            }
            self.pipeline.process_data_point(data)

        # Get feature vector
        feature_vector = self.pipeline.get_feature_vector()

        # Should be a numpy array
        self.assertIsInstance(feature_vector, np.ndarray)
        self.assertGreater(len(feature_vector), 0)

if __name__ == '__main__':
    unittest.main()