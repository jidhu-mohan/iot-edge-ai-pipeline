import json
import time
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import threading
import uuid

@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: str
    timestamp: str
    value: float
    unit: str
    location: str
    metadata: Dict

class IoTSensorSimulator:
    def __init__(self):
        self.sensors = {}
        self.running = False
        self.threads = []

    def add_sensor(self, sensor_id: str, sensor_type: str, location: str,
                   min_val: float, max_val: float, unit: str,
                   sampling_rate: float = 1.0, noise_factor: float = 0.1):
        """Add a sensor to the simulation"""
        self.sensors[sensor_id] = {
            'type': sensor_type,
            'location': location,
            'min_val': min_val,
            'max_val': max_val,
            'unit': unit,
            'sampling_rate': sampling_rate,
            'noise_factor': noise_factor,
            'last_value': random.uniform(min_val, max_val)
        }

    def _generate_realistic_value(self, sensor_id: str) -> float:
        """Generate realistic sensor values with trends and noise"""
        sensor = self.sensors[sensor_id]

        # Base patterns for different sensor types
        if sensor['type'] == 'temperature':
            # Daily temperature pattern
            hour = datetime.now().hour
            base_temp = 20 + 10 * np.sin(2 * np.pi * (hour - 6) / 24)
            noise = np.random.normal(0, sensor['noise_factor'])
            value = base_temp + noise

        elif sensor['type'] == 'humidity':
            # Inverse correlation with temperature
            hour = datetime.now().hour
            base_humidity = 60 - 20 * np.sin(2 * np.pi * (hour - 6) / 24)
            noise = np.random.normal(0, sensor['noise_factor'] * 5)
            value = max(0, min(100, base_humidity + noise))

        elif sensor['type'] == 'vibration':
            # Random spikes for anomaly detection
            if random.random() < 0.05:  # 5% chance of anomaly
                value = random.uniform(sensor['max_val'] * 0.8, sensor['max_val'])
            else:
                value = random.uniform(sensor['min_val'], sensor['max_val'] * 0.3)

        elif sensor['type'] == 'pressure':
            # Slow drift with occasional changes
            drift = np.random.normal(0, 0.1)
            sensor['last_value'] += drift
            value = max(sensor['min_val'], min(sensor['max_val'], sensor['last_value']))

        else:
            # Generic sensor with random walk
            change = np.random.normal(0, sensor['noise_factor'])
            sensor['last_value'] += change
            value = max(sensor['min_val'], min(sensor['max_val'], sensor['last_value']))

        return round(value, 2)

    def _sensor_thread(self, sensor_id: str, callback=None):
        """Thread function for individual sensor"""
        sensor = self.sensors[sensor_id]

        while self.running:
            value = self._generate_realistic_value(sensor_id)

            reading = SensorReading(
                sensor_id=sensor_id,
                sensor_type=sensor['type'],
                timestamp=datetime.now().isoformat(),
                value=value,
                unit=sensor['unit'],
                location=sensor['location'],
                metadata={
                    'sampling_rate': sensor['sampling_rate'],
                    'quality': 'good' if random.random() > 0.1 else 'poor'
                }
            )

            if callback:
                callback(reading)

            time.sleep(1.0 / sensor['sampling_rate'])

    def start_simulation(self, callback=None):
        """Start all sensor simulations"""
        self.running = True

        for sensor_id in self.sensors:
            thread = threading.Thread(
                target=self._sensor_thread,
                args=(sensor_id, callback),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)

    def stop_simulation(self):
        """Stop all sensor simulations"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
        self.threads.clear()

class SensorDataLogger:
    def __init__(self, output_file: str = 'data/raw/sensor_data.json'):
        self.output_file = output_file
        self.data_buffer = []
        self.buffer_size = 100

    def log_reading(self, reading: SensorReading):
        """Log sensor reading to buffer and file"""
        self.data_buffer.append(asdict(reading))

        if len(self.data_buffer) >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffer to file"""
        try:
            with open(self.output_file, 'a') as f:
                for reading in self.data_buffer:
                    f.write(json.dumps(reading) + '\n')
            self.data_buffer.clear()
        except Exception as e:
            print(f"Error writing to file: {e}")

def setup_default_sensors() -> IoTSensorSimulator:
    """Setup a default set of sensors for the simulation"""
    simulator = IoTSensorSimulator()

    # Temperature sensors
    simulator.add_sensor('temp_001', 'temperature', 'factory_floor',
                        15.0, 35.0, '°C', 0.5, 1.0)
    simulator.add_sensor('temp_002', 'temperature', 'warehouse',
                        10.0, 30.0, '°C', 0.2, 0.8)

    # Humidity sensors
    simulator.add_sensor('hum_001', 'humidity', 'factory_floor',
                        30.0, 90.0, '%', 0.5, 2.0)
    simulator.add_sensor('hum_002', 'humidity', 'warehouse',
                        40.0, 80.0, '%', 0.2, 1.5)

    # Vibration sensors
    simulator.add_sensor('vib_001', 'vibration', 'motor_1',
                        0.0, 10.0, 'm/s²', 2.0, 0.5)
    simulator.add_sensor('vib_002', 'vibration', 'motor_2',
                        0.0, 8.0, 'm/s²', 2.0, 0.3)

    # Pressure sensors
    simulator.add_sensor('press_001', 'pressure', 'pipe_main',
                        100.0, 150.0, 'kPa', 1.0, 0.2)

    return simulator

if __name__ == "__main__":
    # Example usage
    logger = SensorDataLogger()
    simulator = setup_default_sensors()

    def data_callback(reading):
        print(f"{reading.sensor_id}: {reading.value} {reading.unit}")
        logger.log_reading(reading)

    try:
        simulator.start_simulation(data_callback)
        print("Sensor simulation started. Press Ctrl+C to stop.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping simulation...")
        simulator.stop_simulation()
        print("Simulation stopped.")