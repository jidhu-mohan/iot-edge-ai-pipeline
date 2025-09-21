import unittest
import time
import tempfile
import os
from datetime import datetime

from src.sensors import IoTSensorSimulator, SensorDataLogger, setup_default_sensors

class TestIoTSensorSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = IoTSensorSimulator()

    def test_add_sensor(self):
        """Test adding a sensor to the simulator"""
        self.simulator.add_sensor(
            sensor_id='test_temp',
            sensor_type='temperature',
            location='test_location',
            min_val=0.0,
            max_val=100.0,
            unit='°C'
        )

        self.assertIn('test_temp', self.simulator.sensors)
        self.assertEqual(self.simulator.sensors['test_temp']['type'], 'temperature')

    def test_sensor_simulation(self):
        """Test that sensor simulation generates realistic values"""
        self.simulator.add_sensor(
            sensor_id='test_temp',
            sensor_type='temperature',
            location='test_location',
            min_val=20.0,
            max_val=30.0,
            unit='°C'
        )

        # Generate multiple values and check they're in range
        values = []
        for _ in range(10):
            value = self.simulator._generate_realistic_value('test_temp')
            values.append(value)

        # All values should be within reasonable bounds (allow for noise variation)
        for value in values:
            self.assertGreaterEqual(value, 10.0)  # Allow more variation for realistic noise
            self.assertLessEqual(value, 40.0)

    def test_callback_system(self):
        """Test that callbacks are called correctly"""
        callback_data = []

        def test_callback(reading):
            callback_data.append(reading)

        self.simulator.add_sensor(
            sensor_id='test_sensor',
            sensor_type='temperature',
            location='test',
            min_val=20.0,
            max_val=25.0,
            unit='°C',
            sampling_rate=10.0  # High frequency for faster testing
        )

        self.simulator.start_simulation(test_callback)
        time.sleep(0.2)  # Wait a bit for some readings
        self.simulator.stop_simulation()

        # Should have received some readings
        self.assertGreater(len(callback_data), 0)

        # Check reading structure
        reading = callback_data[0]
        self.assertEqual(reading.sensor_id, 'test_sensor')
        self.assertEqual(reading.sensor_type, 'temperature')
        self.assertEqual(reading.unit, '°C')

class TestSensorDataLogger(unittest.TestCase):
    def test_data_logging(self):
        """Test data logging functionality"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            logger = SensorDataLogger(temp_filename)

            # Create test reading
            from src.sensors.sensor_simulator import SensorReading
            reading = SensorReading(
                sensor_id='test_sensor',
                sensor_type='temperature',
                timestamp=datetime.now().isoformat(),
                value=25.5,
                unit='°C',
                location='test_location',
                metadata={'quality': 'good'}
            )

            # Log reading
            logger.log_reading(reading)
            logger._flush_buffer()  # Force flush

            # Check file exists and has content
            self.assertTrue(os.path.exists(temp_filename))

            with open(temp_filename, 'r') as f:
                content = f.read()
                self.assertIn('test_sensor', content)
                self.assertIn('25.5', content)

        finally:
            # Cleanup
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

class TestDefaultSensors(unittest.TestCase):
    def test_setup_default_sensors(self):
        """Test default sensor setup"""
        simulator = setup_default_sensors()

        # Should have multiple sensors
        self.assertGreater(len(simulator.sensors), 0)

        # Should have different sensor types
        sensor_types = set()
        for sensor_id, sensor_data in simulator.sensors.items():
            sensor_types.add(sensor_data['type'])

        expected_types = {'temperature', 'humidity', 'vibration', 'pressure'}
        self.assertTrue(expected_types.issubset(sensor_types))

if __name__ == '__main__':
    unittest.main()