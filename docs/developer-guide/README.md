# Developer Guide

This guide is for developers who want to extend, customize, or contribute to the IoT Edge AI Pipeline project.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Architecture Overview](#architecture-overview)
3. [Adding New Sensors](#adding-new-sensors)
4. [Creating Custom Models](#creating-custom-models)
5. [Extending the Pipeline](#extending-the-pipeline)
6. [Testing](#testing)
7. [Performance Optimization](#performance-optimization)
8. [Contributing Guidelines](#contributing-guidelines)

---

## Development Setup

### Development Environment

1. **Clone the repository:**
```bash
git clone <repository-url>
cd iot-edge-ai-pipeline
```

2. **Create development environment:**
```bash
python -m venv dev-env
source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate
```

3. **Install development dependencies:**
```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

4. **Setup pre-commit hooks:**
```bash
pip install pre-commit
pre-commit install
```

### Project Structure for Developers

```
iot-edge-ai-pipeline/
├── src/                    # Main source code
│   ├── sensors/           # Sensor simulation and MQTT
│   ├── preprocessing/     # Data processing pipeline
│   ├── ml_models/        # Machine learning models
│   ├── inference/        # Real-time inference engine
│   └── __init__.py
├── tests/                 # Test suite
├── docs/                  # Documentation
├── config/               # Configuration files
├── data/                 # Data storage
├── models/               # Trained models
├── main.py              # Main application
├── run_example.py       # Demo script
└── requirements.txt     # Dependencies
```

### Code Style

We follow PEP 8 with these specifics:

- **Line length**: 88 characters (Black default)
- **Imports**: Use absolute imports
- **Docstrings**: Google style
- **Type hints**: Required for public APIs

Example:
```python
def process_sensor_data(
    sensor_data: Dict[str, Any],
    validation_rules: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, float]]:
    """Process sensor data through the pipeline.

    Args:
        sensor_data: Dictionary containing sensor information
        validation_rules: Optional validation rules to apply

    Returns:
        Dictionary of extracted features or None if processing failed

    Raises:
        ValueError: If sensor_data format is invalid
    """
    pass
```

---

## Architecture Overview

### Component Interaction

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   IoT Sensors   │───▶│ Data Processing  │───▶│  ML Inference   │
│                 │    │                  │    │                 │
│ • Simulation    │    │ • Validation     │    │ • Classification│
│ • MQTT Pub/Sub  │    │ • Feature Eng.   │    │ • Anomaly Det.  │
│ • Real Hardware │    │ • Buffering      │    │ • Edge Optimized│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Storage  │    │   Monitoring     │    │     Alerts      │
│                 │    │                  │    │                 │
│ • Raw Data      │    │ • Performance    │    │ • Thresholds    │
│ • Features      │    │ • Health Checks  │    │ • Notifications │
│ • Model Results │    │ • Metrics        │    │ • Actions       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Design Principles

1. **Modularity**: Each component is independent and replaceable
2. **Scalability**: Support for multiple sensors and models
3. **Performance**: Optimized for real-time edge deployment
4. **Extensibility**: Easy to add new sensors, models, and features
5. **Reliability**: Comprehensive error handling and monitoring

### Data Flow

1. **Sensor Data Generation**: IoT sensors or simulators generate data
2. **Data Ingestion**: MQTT or direct API ingestion
3. **Validation**: Data quality checks and filtering
4. **Feature Extraction**: Statistical and time-based features
5. **Model Inference**: ML predictions and anomaly detection
6. **Alert Processing**: Threshold checks and notifications
7. **Storage**: Raw data and results persistence

---

## Adding New Sensors

### Creating a New Sensor Type

1. **Define the sensor class:**

```python
# In src/sensors/custom_sensor.py
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class CustomSensorReading:
    sensor_id: str
    sensor_type: str
    timestamp: str
    value: float
    unit: str
    location: str
    metadata: Dict[str, Any]
    # Add custom fields
    custom_field: str

class CustomSensor:
    def __init__(self, sensor_id: str, location: str, **kwargs):
        self.sensor_id = sensor_id
        self.location = location
        self.config = kwargs

    def generate_reading(self) -> CustomSensorReading:
        """Generate a realistic sensor reading."""
        # Implement custom data generation logic
        value = self._generate_custom_value()

        return CustomSensorReading(
            sensor_id=self.sensor_id,
            sensor_type='custom_type',
            timestamp=datetime.now().isoformat(),
            value=value,
            unit='custom_unit',
            location=self.location,
            metadata={'quality': 'good'},
            custom_field='custom_data'
        )

    def _generate_custom_value(self) -> float:
        """Implement custom value generation logic."""
        # Add realistic patterns, noise, trends, etc.
        return np.random.normal(50.0, 5.0)
```

2. **Integrate with the simulator:**

```python
# In src/sensors/sensor_simulator.py
from .custom_sensor import CustomSensor

class IoTSensorSimulator:
    def add_custom_sensor(self, sensor_id: str, sensor_class, **kwargs):
        """Add a custom sensor type to the simulation."""
        sensor = sensor_class(sensor_id, **kwargs)
        self.custom_sensors[sensor_id] = sensor
```

3. **Add validation rules:**

```python
# In src/preprocessing/data_processor.py
class DataValidator:
    def __init__(self):
        super().__init__()
        # Add validation for custom sensor
        self.add_validation_rule('custom_type', 0.0, 100.0)
```

### Hardware Integration Example

For real hardware sensors:

```python
# Hardware interface example
import serial
import json

class ArduinoSensorInterface:
    def __init__(self, port: str, baud_rate: int = 9600):
        self.serial_conn = serial.Serial(port, baud_rate)

    def read_sensor_data(self) -> Dict[str, Any]:
        """Read data from Arduino sensor."""
        try:
            line = self.serial_conn.readline().decode('utf-8').strip()
            data = json.loads(line)
            return {
                'sensor_id': data['id'],
                'sensor_type': data['type'],
                'timestamp': datetime.now().isoformat(),
                'value': float(data['value']),
                'unit': data['unit'],
                'location': data.get('location', 'unknown'),
                'metadata': data.get('metadata', {})
            }
        except Exception as e:
            print(f"Error reading sensor data: {e}")
            return None
```

---

## Creating Custom Models

### Custom Classification Model

```python
# In src/ml_models/custom_models.py
from .edge_models import EdgeMLModel
import numpy as np
from sklearn.svm import SVC

class CustomSVMModel(EdgeMLModel):
    def __init__(self):
        super().__init__(model_type='sklearn')

    def create_svm_model(self, kernel='rbf', C=1.0):
        """Create SVM model optimized for edge deployment."""
        self.model = SVC(
            kernel=kernel,
            C=C,
            probability=True,  # Enable probability estimates
            gamma='scale'
        )
        return self.model

    def optimize_for_edge(self):
        """Custom optimization for SVM."""
        # Reduce support vectors if possible
        if hasattr(self.model, 'support_vectors_'):
            n_support = len(self.model.support_vectors_)
            print(f"Model has {n_support} support vectors")

        return self.model
```

### Custom Neural Network Architecture

```python
import tensorflow as tf
from tensorflow import keras

class CustomNeuralNetwork(EdgeMLModel):
    def __init__(self):
        super().__init__(model_type='tensorflow')

    def create_lstm_model(self, input_shape, output_dim=1):
        """Create LSTM model for time series prediction."""
        model = keras.Sequential([
            keras.layers.LSTM(32, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(16, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(output_dim, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train_time_series(self, X_sequence, y, epochs=50, batch_size=32):
        """Train on time series data."""
        # Reshape data for LSTM input
        X_reshaped = X_sequence.reshape((X_sequence.shape[0], X_sequence.shape[1], 1))

        history = self.model.fit(
            X_reshaped, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )

        return history
```

### Custom Feature Engineering

```python
# In src/preprocessing/custom_features.py
from typing import List, Dict
import numpy as np
from scipy import signal

class AdvancedFeatureExtractor:
    def extract_frequency_features(self, values: List[float], sampling_rate: float) -> Dict[str, float]:
        """Extract frequency domain features."""
        if len(values) < 4:
            return {}

        # FFT analysis
        fft = np.fft.fft(values)
        freqs = np.fft.fftfreq(len(values), 1/sampling_rate)
        magnitude = np.abs(fft)

        # Find dominant frequency
        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]

        return {
            'dominant_frequency': dominant_freq,
            'spectral_centroid': np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2]),
            'spectral_rolloff': self._calculate_spectral_rolloff(freqs, magnitude),
            'spectral_bandwidth': np.sqrt(np.sum(((freqs[:len(freqs)//2] - dominant_freq) ** 2) * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2]))
        }

    def extract_wavelet_features(self, values: List[float]) -> Dict[str, float]:
        """Extract wavelet-based features."""
        try:
            from pywt import wavedec

            # Wavelet decomposition
            coeffs = wavedec(values, 'db4', level=3)

            features = {}
            for i, coeff in enumerate(coeffs):
                features[f'wavelet_level_{i}_energy'] = np.sum(coeff ** 2)
                features[f'wavelet_level_{i}_entropy'] = self._calculate_entropy(coeff)

            return features

        except ImportError:
            print("PyWavelets not installed, skipping wavelet features")
            return {}

    def _calculate_spectral_rolloff(self, freqs: np.ndarray, magnitude: np.ndarray, rolloff_percent: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        cumulative_energy = np.cumsum(magnitude[:len(magnitude)//2])
        total_energy = cumulative_energy[-1]
        rolloff_energy = rolloff_percent * total_energy

        rolloff_idx = np.where(cumulative_energy >= rolloff_energy)[0]
        return freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0

    def _calculate_entropy(self, signal: np.ndarray) -> float:
        """Calculate signal entropy."""
        signal_normalized = signal / np.sum(np.abs(signal))
        entropy = -np.sum(signal_normalized * np.log(np.abs(signal_normalized) + 1e-10))
        return entropy
```

---

## Extending the Pipeline

### Adding a New Processing Stage

```python
# In src/preprocessing/custom_processor.py
from typing import Dict, Any, Optional

class CustomProcessingStage:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Custom processing logic."""
        try:
            # Implement your custom processing
            processed_data = self._custom_processing_logic(data)
            return processed_data

        except Exception as e:
            print(f"Error in custom processing: {e}")
            return None

    def _custom_processing_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement your specific processing logic here."""
        # Example: Add computed fields
        data['computed_field'] = data['value'] * 2
        return data

# Integrate into the main pipeline
class ExtendedDataPipeline(DataPipeline):
    def __init__(self, window_size: int = 100):
        super().__init__(window_size)
        self.custom_stage = CustomProcessingStage({})

    def process_data_point(self, data: Dict) -> Optional[Dict]:
        """Extended processing with custom stage."""
        # Run base processing
        features = super().process_data_point(data)

        if features:
            # Apply custom processing
            enhanced_features = self.custom_stage.process(features)
            return enhanced_features

        return None
```

### Custom Alert Handlers

```python
# In src/inference/custom_alerts.py
import smtplib
import requests
from typing import Dict, Any

class EmailAlertHandler:
    def __init__(self, smtp_server: str, username: str, password: str):
        self.smtp_server = smtp_server
        self.username = username
        self.password = password

    def send_alert(self, alert: Dict[str, Any]):
        """Send alert via email."""
        try:
            subject = f"IoT Alert: {alert['type']}"
            body = f"Alert Details:\n{json.dumps(alert, indent=2)}"

            msg = f"Subject: {subject}\n\n{body}"

            server = smtplib.SMTP(self.smtp_server, 587)
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.username, "admin@company.com", msg)
            server.quit()

        except Exception as e:
            print(f"Failed to send email alert: {e}")

class WebhookAlertHandler:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_alert(self, alert: Dict[str, Any]):
        """Send alert to webhook."""
        try:
            response = requests.post(
                self.webhook_url,
                json=alert,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            response.raise_for_status()

        except Exception as e:
            print(f"Failed to send webhook alert: {e}")

# Integration example
class EnhancedInferenceAPI(EdgeInferenceAPI):
    def __init__(self, inference_engine):
        super().__init__(inference_engine)
        self.alert_handlers = []

    def add_alert_handler(self, handler):
        """Add custom alert handler."""
        self.alert_handlers.append(handler)

    def process_with_alerts(self, sensor_data: Dict) -> Dict:
        """Process with enhanced alert handling."""
        result = super().process_with_alerts(sensor_data)

        # Send alerts to custom handlers
        if result.get('alerts'):
            for alert in result['alerts']:
                for handler in self.alert_handlers:
                    handler.send_alert(alert)

        return result
```

---

## Testing

### Unit Testing

```python
# In tests/test_custom_features.py
import unittest
import numpy as np
from src.preprocessing.custom_features import AdvancedFeatureExtractor

class TestAdvancedFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = AdvancedFeatureExtractor()

    def test_frequency_features(self):
        """Test frequency domain feature extraction."""
        # Generate test signal
        sampling_rate = 100
        t = np.linspace(0, 1, sampling_rate)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave

        features = self.extractor.extract_frequency_features(signal.tolist(), sampling_rate)

        self.assertIn('dominant_frequency', features)
        self.assertAlmostEqual(features['dominant_frequency'], 10.0, delta=1.0)

    def test_wavelet_features(self):
        """Test wavelet feature extraction."""
        signal = np.random.randn(64)
        features = self.extractor.extract_wavelet_features(signal.tolist())

        # Should extract multiple levels
        level_keys = [k for k in features.keys() if 'wavelet_level' in k]
        self.assertGreater(len(level_keys), 0)
```

### Integration Testing

```python
# In tests/test_integration.py
import unittest
import tempfile
import os
from src.sensors import IoTSensorSimulator
from src.preprocessing import DataPipeline
from src.ml_models import EdgeMLModel

class TestIntegration(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        """Test complete pipeline integration."""
        # Setup components
        simulator = IoTSensorSimulator()
        pipeline = DataPipeline()

        simulator.add_sensor(
            sensor_id='test_sensor',
            sensor_type='temperature',
            location='test',
            min_val=20.0,
            max_val=25.0,
            unit='°C'
        )

        # Collect data
        collected_features = []

        def collect_data(reading):
            sensor_data = {
                'sensor_id': reading.sensor_id,
                'sensor_type': reading.sensor_type,
                'timestamp': reading.timestamp,
                'value': reading.value,
                'unit': reading.unit,
                'location': reading.location
            }

            features = pipeline.process_data_point(sensor_data)
            if features:
                collected_features.append(features)

        # Run simulation briefly
        simulator.start_simulation(collect_data)
        time.sleep(2)
        simulator.stop_simulation()

        # Verify data collection
        self.assertGreater(len(collected_features), 0)

        # Verify feature structure
        features = collected_features[0]
        self.assertIn('hour', features)
        self.assertIn('processed_timestamp', features)
```

### Performance Testing

```python
# In tests/test_performance.py
import time
import unittest
import numpy as np
from src.ml_models import EdgeMLModel

class TestPerformance(unittest.TestCase):
    def test_inference_speed(self):
        """Test that inference meets speed requirements."""
        # Create lightweight model
        model = EdgeMLModel(model_type='sklearn')
        model.create_random_forest(n_estimators=10, max_depth=5)

        # Train with dummy data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model.train(X, y)

        # Test inference speed
        test_input = np.random.randn(1, 10)

        times = []
        for _ in range(100):
            start_time = time.time()
            result = model.predict(test_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        max_time = np.max(times)

        # Assert performance requirements
        self.assertLess(avg_time, 50.0, f"Average inference time {avg_time:.2f}ms exceeds 50ms")
        self.assertLess(max_time, 100.0, f"Max inference time {max_time:.2f}ms exceeds 100ms")
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_custom_features.py -v

# Run performance tests
python -m pytest tests/test_performance.py -v
```

---

## Performance Optimization

### Profiling

Use Python profilers to identify bottlenecks:

```python
import cProfile
import pstats
from src.preprocessing import DataPipeline

def profile_pipeline():
    """Profile the data processing pipeline."""
    pipeline = DataPipeline()

    # Generate test data
    test_data = {
        'sensor_id': 'test',
        'sensor_type': 'temperature',
        'timestamp': '2023-12-01T12:00:00',
        'value': 25.0,
        'unit': '°C',
        'location': 'test'
    }

    # Profile processing
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(1000):
        pipeline.process_data_point(test_data)

    profiler.disable()

    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

if __name__ == "__main__":
    profile_pipeline()
```

### Memory Optimization

```python
import tracemalloc
import numpy as np

def monitor_memory_usage():
    """Monitor memory usage during processing."""
    tracemalloc.start()

    # Your processing code here
    data = np.random.randn(10000, 100)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

    tracemalloc.stop()
```

### Optimization Strategies

1. **Vectorization**: Use NumPy operations instead of loops
2. **Caching**: Cache computed features and model predictions
3. **Batch Processing**: Process multiple data points together
4. **Memory Management**: Use generators for large datasets
5. **Model Optimization**: Quantization, pruning, distillation

Example optimization:

```python
# Before: Slow loop-based processing
def slow_feature_extraction(values):
    features = {}
    for i, val in enumerate(values):
        features[f'lag_{i}'] = val
    return features

# After: Vectorized processing
def fast_feature_extraction(values):
    values_array = np.array(values)
    return {f'lag_{i}': val for i, val in enumerate(values_array)}

# Even better: Use built-in functions
def optimized_feature_extraction(values):
    values_array = np.array(values)
    return {
        'mean': np.mean(values_array),
        'std': np.std(values_array),
        'min': np.min(values_array),
        'max': np.max(values_array)
    }
```

---

## Contributing Guidelines

### Code Contribution Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes** following the code style guidelines
4. **Add tests** for new functionality
5. **Run the test suite**: `python -m pytest tests/`
6. **Update documentation** if needed
7. **Submit a pull request**

### Pull Request Guidelines

- **Clear description** of changes and motivation
- **Tests included** for new features
- **Documentation updated** for API changes
- **Code style** follows project standards
- **Performance** impact considered

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and cover new functionality
- [ ] Documentation is updated
- [ ] Performance impact is acceptable
- [ ] Error handling is comprehensive
- [ ] Security considerations addressed

### Issue Reporting

When reporting bugs:

1. **Environment details**: Python version, OS, dependencies
2. **Reproduction steps**: Clear steps to reproduce the issue
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Logs/errors**: Include relevant error messages

### Feature Requests

For new features:

1. **Use case description**: Why is this needed?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: Other approaches evaluated
4. **Impact assessment**: Performance, compatibility considerations

---

## Advanced Topics

### Custom Data Sources

```python
# Example: Database integration
import sqlite3
from typing import Iterator, Dict, Any

class DatabaseSensorSource:
    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path)

    def stream_sensor_data(self, batch_size: int = 100) -> Iterator[Dict[str, Any]]:
        """Stream sensor data from database."""
        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT sensor_id, sensor_type, timestamp, value, unit, location
            FROM sensor_readings
            ORDER BY timestamp
        """)

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                yield {
                    'sensor_id': row[0],
                    'sensor_type': row[1],
                    'timestamp': row[2],
                    'value': row[3],
                    'unit': row[4],
                    'location': row[5]
                }
```

### Distributed Processing

```python
# Example: Multi-process inference
from multiprocessing import Pool, Queue
from concurrent.futures import ThreadPoolExecutor

class DistributedInferenceEngine:
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def process_batch_async(self, data_batch: List[Dict]) -> List[Future]:
        """Process data batch asynchronously."""
        futures = []

        for data_point in data_batch:
            future = self.executor.submit(self.process_single, data_point)
            futures.append(future)

        return futures

    def process_single(self, data: Dict) -> Dict:
        """Process single data point."""
        # Your processing logic here
        return result
```

This developer guide provides comprehensive information for extending and customizing the IoT Edge AI Pipeline. For specific implementation details, refer to the source code and API documentation.