# User Guide

Welcome to the IoT Edge AI Pipeline User Guide. This guide will walk you through everything you need to know to use the system effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Configuration](#configuration)
4. [Running Simulations](#running-simulations)
5. [Working with Real Sensors](#working-with-real-sensors)
6. [Model Training](#model-training)
7. [Monitoring and Alerts](#monitoring-and-alerts)
8. [Examples and Tutorials](#examples-and-tutorials)

---

## Getting Started

### Quick Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd iot-edge-ai-pipeline
```

2. **Create virtual environment:**
```bash
python -m venv iot-edge-env
source iot-edge-env/bin/activate  # On Windows: iot-edge-env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python run_example.py
```

### First Run

Your first successful run should show output like:
```
=== IoT Edge AI Pipeline - Simple Example ===

1. Setting up sensor simulation...
2. Setting up data processing pipeline...
3. Creating and training demo models...
4. Setting up inference engine...
5. Processing sensor data and running inference...

Sensor: temp_001 | Value: 25.5 °C | Inference: 2.1ms
Sensor: hum_001 | Value: 65.2 % | Inference: 1.8ms
...
```

---

## Basic Usage

### Running the Quick Demo

The easiest way to understand the system is to run the example script:

```bash
python run_example.py
```

This demonstrates:
- Sensor data simulation
- Real-time data processing
- Machine learning inference
- Anomaly detection
- Performance monitoring

### Running the Full Pipeline

For production-like usage with MQTT:

```bash
python main.py
```

This starts:
- IoT sensor simulation
- MQTT data publishing
- Real-time data ingestion
- ML inference engine
- Performance monitoring

### Command Line Options

```bash
# Run different modes
python main.py --mode simulation    # Only sensor simulation
python main.py --mode inference     # Only inference pipeline
python main.py --mode full         # Complete pipeline (default)

# Use custom configuration
python main.py --config custom_config.yaml

# Get help
python main.py --help
```

---

## Configuration

### Basic Configuration

The system is configured via `config/config.yaml`. Here are the key sections:

#### MQTT Settings
```yaml
mqtt:
  broker_host: localhost    # MQTT broker address
  broker_port: 1883        # MQTT broker port
```

#### Data Processing
```yaml
data_processing:
  window_size: 100         # Number of data points to keep in memory
  feature_window: 10       # Window for feature extraction
  buffer_size: 1000        # Processing buffer size
```

#### Sensor Simulation
```yaml
sensors:
  simulation:
    enabled: true
    sensors:
      - id: "temp_001"
        type: "temperature"
        location: "factory_floor"
        min_val: 15.0
        max_val: 35.0
        unit: "°C"
        sampling_rate: 0.5    # Readings per second
        noise_factor: 1.0     # Amount of random variation
```

### Advanced Configuration

#### Model Settings
```yaml
models:
  default_models:
    - name: "anomaly_detector"
      type: "anomaly_detection"
      path: "models/anomaly_model"
    - name: "temperature_classifier"
      type: "classification"
      path: "models/temp_classifier"
```

#### Performance Constraints
```yaml
inference:
  max_inference_time_ms: 100    # Maximum allowed inference time
  max_model_size_mb: 10         # Maximum model size for edge deployment
```

#### Alert Thresholds
```yaml
inference:
  alert_thresholds:
    anomaly_score: 0.8          # Anomaly detection threshold
    temperature_alert: 35.0     # Temperature alert threshold
    humidity_alert: 85.0        # Humidity alert threshold
```

---

## Running Simulations

### Default Sensor Setup

The system comes with pre-configured sensors:

- **Temperature sensors**: Factory floor and warehouse
- **Humidity sensors**: Environmental monitoring
- **Vibration sensors**: Motor monitoring for predictive maintenance
- **Pressure sensors**: Pipeline monitoring

### Adding Custom Sensors

You can add sensors programmatically:

```python
from src.sensors import IoTSensorSimulator

simulator = IoTSensorSimulator()

# Add custom sensor
simulator.add_sensor(
    sensor_id='ph_001',
    sensor_type='ph',
    location='water_treatment',
    min_val=6.0,
    max_val=8.0,
    unit='pH',
    sampling_rate=1.0,
    noise_factor=0.1
)
```

Or by modifying the configuration file:

```yaml
sensors:
  simulation:
    sensors:
      - id: "ph_001"
        type: "ph"
        location: "water_treatment"
        min_val: 6.0
        max_val: 8.0
        unit: "pH"
        sampling_rate: 1.0
        noise_factor: 0.1
```

### Sensor Data Patterns

Different sensor types generate realistic patterns:

- **Temperature**: Daily cycles with gradual changes
- **Humidity**: Inverse correlation with temperature
- **Vibration**: Random spikes for anomaly simulation (5% chance)
- **Pressure**: Slow drift with occasional changes

---

## Working with Real Sensors

### MQTT Integration

To connect real sensors via MQTT:

1. **Setup MQTT broker** (e.g., Mosquitto):
```bash
# Install Mosquitto (Ubuntu/Debian)
sudo apt-get install mosquitto mosquitto-clients

# Start broker
sudo systemctl start mosquitto
```

2. **Configure sensor devices** to publish to topics:
```
sensors/{sensor_id}/data
```

3. **Data format** should match:
```json
{
  "sensor_id": "temp_001",
  "sensor_type": "temperature",
  "timestamp": "2023-12-01T14:30:45.123Z",
  "value": 25.5,
  "unit": "°C",
  "location": "factory_floor",
  "metadata": {
    "quality": "good",
    "battery_level": 85
  }
}
```

### HTTP/REST Integration

For sensors that don't support MQTT, you can create a bridge:

```python
from src.sensors import MQTTSensorPublisher
import requests

publisher = MQTTSensorPublisher()
publisher.connect()

# Poll HTTP endpoint and publish to MQTT
def poll_http_sensor():
    response = requests.get('http://sensor-device/api/data')
    sensor_data = response.json()

    # Convert to SensorReading format
    reading = SensorReading(
        sensor_id=sensor_data['id'],
        sensor_type=sensor_data['type'],
        timestamp=sensor_data['timestamp'],
        value=sensor_data['value'],
        unit=sensor_data['unit'],
        location=sensor_data['location'],
        metadata=sensor_data.get('metadata', {})
    )

    publisher.publish_sensor_data(reading)
```

---

## Model Training

### Using Pre-trained Models

The system comes with demo models that are automatically created on first run. For production, you'll want to train models on your specific data.

### Training Custom Models

#### Classification Model Example

```python
from src.ml_models import EdgeMLModel
import numpy as np
import pandas as pd

# Load your training data
data = pd.read_csv('your_training_data.csv')
X = data[['feature1', 'feature2', 'feature3']].values
y = data['target'].values

# Create and train model
model = EdgeMLModel(model_type='sklearn')
model.create_random_forest(n_estimators=50, max_depth=10)
model.train(X, y)

# Save trained model
model.save_model('models/custom_classifier')
```

#### Anomaly Detection Model Example

```python
from src.ml_models import AnomalyDetectionModel

# Load normal operating data (no anomalies)
normal_data = pd.read_csv('normal_operations.csv')
X_normal = normal_data[['feature1', 'feature2', 'feature3']].values

# Create and train anomaly detector
anomaly_model = AnomalyDetectionModel()
anomaly_model.create_isolation_forest(contamination=0.1)
anomaly_model.train_anomaly_detector(X_normal)

# Save model
anomaly_model.save_model('models/custom_anomaly_detector')
```

#### Neural Network Model

```python
# For more complex patterns
model = EdgeMLModel(model_type='tensorflow')
model.create_neural_network(input_dim=10, output_dim=1)
model.train(X, y, validation_split=0.2)

# Optimize for edge deployment
optimized_model = model.optimize_for_edge()
```

### Model Validation

Always validate your models meet edge constraints:

```python
from src.ml_models import ModelValidator

validator = ModelValidator()

# Check performance
performance = validator.validate_model_performance(model, (X_test, y_test))
print(f"Model accuracy: {performance['accuracy']:.3f}")
print(f"Average inference time: {performance['avg_inference_time_ms']:.1f}ms")

# Check edge constraints
constraints = validator.validate_edge_constraints(
    model,
    max_inference_time_ms=100,
    max_model_size_mb=10
)
print(f"Edge deployment ready: {constraints['constraints_met']}")
```

---

## Monitoring and Alerts

### Real-time Monitoring

The system provides comprehensive monitoring:

```python
from src.inference import InferenceMonitor

# Start monitoring
monitor = InferenceMonitor(inference_engine)
monitor.start_monitoring(interval_seconds=30)

# Monitor will log performance stats every 30 seconds
```

### Performance Metrics

Key metrics monitored:

- **Inference Times**: Average and maximum per prediction
- **Throughput**: Data points processed per second
- **Model Performance**: Accuracy and confidence scores
- **System Resources**: Memory and CPU usage
- **Alert Frequency**: Number of alerts triggered

### Alert Configuration

Configure alerts in your config file:

```yaml
inference:
  alert_thresholds:
    anomaly_score: 0.8        # Trigger alert if anomaly score > 0.8
    temperature_alert: 35.0   # Alert if temperature > 35°C
    humidity_alert: 85.0      # Alert if humidity > 85%
```

### Custom Alert Handlers

Add custom alert processing:

```python
def alert_handler(alert):
    if alert['type'] == 'anomaly':
        print(f"🚨 ANOMALY DETECTED: {alert['sensor_id']}")
        # Send email, SMS, or webhook notification
    elif alert['type'] == 'threshold':
        print(f"⚠️ THRESHOLD EXCEEDED: {alert['sensor_id']} = {alert['probability']}")

# Add to inference API
api.add_alert_callback(alert_handler)
```

---

## Examples and Tutorials

### Example 1: Basic Sensor Monitoring

Monitor temperature and humidity with alerts:

```python
from src.sensors import setup_default_sensors, SensorDataLogger
from src.preprocessing import DataPipeline
from src.inference import RealTimeInferenceEngine, EdgeInferenceAPI

# Setup components
simulator = setup_default_sensors()
pipeline = DataPipeline()
engine = RealTimeInferenceEngine()
api = EdgeInferenceAPI(engine)

# Set alert thresholds
api.set_alert_threshold('temp_alert', 30.0)

# Process data
def process_data(reading):
    sensor_data = {
        'sensor_id': reading.sensor_id,
        'sensor_type': reading.sensor_type,
        'timestamp': reading.timestamp,
        'value': reading.value,
        'unit': reading.unit,
        'location': reading.location
    }

    result = api.process_with_alerts(sensor_data)

    if result['alerts']:
        for alert in result['alerts']:
            print(f"Alert: {alert}")

# Start simulation
simulator.start_simulation(process_data)
```

### Example 2: Predictive Maintenance

Monitor vibration sensors for equipment health:

```python
import numpy as np
from src.ml_models import AnomalyDetectionModel

# Train on normal vibration data
normal_vibration = np.random.normal(2.0, 0.5, (1000, 1))  # Normal operation ~2.0 m/s²

model = AnomalyDetectionModel()
model.create_isolation_forest(contamination=0.05)  # Expect 5% anomalies
model.train_anomaly_detector(normal_vibration)

# Use in real-time monitoring
def check_equipment_health(reading):
    if reading.sensor_type == 'vibration':
        result = model.detect_anomaly([[reading.value]])

        if result['is_anomaly']:
            print(f"⚠️ Equipment anomaly detected on {reading.sensor_id}")
            print(f"Vibration: {reading.value} m/s² (score: {result['anomaly_score']:.3f})")
            # Schedule maintenance, send alert, etc.
```

### Example 3: Environmental Control

Automated HVAC control based on sensor data:

```python
class HVACController:
    def __init__(self):
        self.target_temp = 22.0
        self.target_humidity = 50.0

    def control_hvac(self, temp_reading, humidity_reading):
        # Simple control logic
        if temp_reading.value > self.target_temp + 2:
            print("🔧 Activating cooling")
            # API call to HVAC system

        elif temp_reading.value < self.target_temp - 2:
            print("🔧 Activating heating")

        if humidity_reading.value > self.target_humidity + 10:
            print("🔧 Activating dehumidifier")

# Integrate with sensor processing
hvac = HVACController()

def process_environmental_data(reading):
    if reading.sensor_type == 'temperature':
        # Get corresponding humidity reading
        humidity_reading = get_latest_humidity(reading.location)
        if humidity_reading:
            hvac.control_hvac(reading, humidity_reading)
```

### Example 4: Data Logging and Analysis

Log sensor data for later analysis:

```python
from src.sensors import SensorDataLogger
import pandas as pd
import matplotlib.pyplot as plt

# Setup logging
logger = SensorDataLogger('data/raw/sensor_log.json')

# Log data for a period
simulator.start_simulation(logger.log_reading)
# Let it run for desired time...
simulator.stop_simulation()

# Analyze logged data
data = []
with open('data/raw/sensor_log.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Plot temperature trends
temp_data = df[df['sensor_type'] == 'temperature']
plt.figure(figsize=(12, 6))
for sensor_id in temp_data['sensor_id'].unique():
    sensor_df = temp_data[temp_data['sensor_id'] == sensor_id]
    plt.plot(pd.to_datetime(sensor_df['timestamp']),
             sensor_df['value'],
             label=sensor_id)

plt.title('Temperature Trends')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

---

## Tips and Best Practices

### Performance Optimization

1. **Batch Processing**: Process multiple readings together when possible
2. **Feature Caching**: Cache computed features for repeated use
3. **Model Selection**: Use simpler models for better edge performance
4. **Data Filtering**: Filter out unnecessary data early in the pipeline

### Production Deployment

1. **Configuration Management**: Use environment-specific config files
2. **Logging**: Configure appropriate log levels for production
3. **Monitoring**: Set up comprehensive monitoring and alerting
4. **Backup**: Regularly backup trained models and configurations

### Troubleshooting

1. **Check Logs**: Always check application logs first
2. **Validate Data**: Ensure sensor data format is correct
3. **Test Models**: Validate models meet performance requirements
4. **Network Issues**: Check MQTT broker connectivity

### Security Considerations

1. **MQTT Security**: Use TLS encryption and authentication
2. **Data Validation**: Always validate incoming sensor data
3. **Access Control**: Implement proper access controls
4. **Monitoring**: Monitor for unusual data patterns

---

## Next Steps

1. **Read the [Developer Guide](../developer-guide/README.md)** for customization
2. **Check the [Deployment Guide](../deployment/README.md)** for production setup
3. **Review the [API Documentation](../api/README.md)** for detailed technical information
4. **See the [Troubleshooting Guide](../troubleshooting/README.md)** if you encounter issues

## Support

For additional help:
- Check the troubleshooting guide
- Review the API documentation
- Look at the example scripts
- Open an issue on GitHub