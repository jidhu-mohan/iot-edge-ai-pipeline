# API Documentation

This document provides comprehensive API documentation for the IoT Edge AI Pipeline components.

## Table of Contents

1. [Sensor APIs](#sensor-apis)
2. [Preprocessing APIs](#preprocessing-apis)
3. [ML Model APIs](#ml-model-apis)
4. [Inference Engine APIs](#inference-engine-apis)
5. [Configuration APIs](#configuration-apis)

---

## Sensor APIs

### IoTSensorSimulator

Main class for simulating IoT sensors with realistic data patterns.

#### Constructor
```python
IoTSensorSimulator()
```

#### Methods

##### `add_sensor(sensor_id, sensor_type, location, min_val, max_val, unit, sampling_rate=1.0, noise_factor=0.1)`
Add a new sensor to the simulation.

**Parameters:**
- `sensor_id` (str): Unique identifier for the sensor
- `sensor_type` (str): Type of sensor ('temperature', 'humidity', 'vibration', 'pressure')
- `location` (str): Physical location of the sensor
- `min_val` (float): Minimum possible value
- `max_val` (float): Maximum possible value
- `unit` (str): Unit of measurement
- `sampling_rate` (float): Samples per second (default: 1.0)
- `noise_factor` (float): Amount of noise to add (default: 0.1)

**Example:**
```python
simulator = IoTSensorSimulator()
simulator.add_sensor(
    sensor_id='temp_001',
    sensor_type='temperature',
    location='factory_floor',
    min_val=15.0,
    max_val=35.0,
    unit='°C',
    sampling_rate=0.5
)
```

##### `start_simulation(callback=None)`
Start the sensor simulation with optional callback for data processing.

**Parameters:**
- `callback` (callable): Function to call with each sensor reading

**Example:**
```python
def data_handler(reading):
    print(f"Sensor {reading.sensor_id}: {reading.value} {reading.unit}")

simulator.start_simulation(data_handler)
```

##### `stop_simulation()`
Stop all sensor simulations.

### SensorReading

Data class representing a single sensor reading.

**Attributes:**
- `sensor_id` (str): Sensor identifier
- `sensor_type` (str): Type of sensor
- `timestamp` (str): ISO format timestamp
- `value` (float): Measured value
- `unit` (str): Unit of measurement
- `location` (str): Sensor location
- `metadata` (dict): Additional metadata

### MQTTSensorPublisher

Publishes sensor data to MQTT broker.

#### Constructor
```python
MQTTSensorPublisher(broker_host='localhost', broker_port=1883)
```

#### Methods

##### `connect()`
Connect to MQTT broker.

##### `disconnect()`
Disconnect from MQTT broker.

##### `publish_sensor_data(reading)`
Publish a sensor reading to MQTT.

**Parameters:**
- `reading` (SensorReading): Sensor reading to publish

**Returns:**
- `bool`: True if successful

### MQTTDataIngestion

Ingests sensor data from MQTT broker.

#### Constructor
```python
MQTTDataIngestion(broker_host='localhost', broker_port=1883)
```

#### Methods

##### `connect()`
Connect to MQTT broker and start listening.

##### `disconnect()`
Disconnect from MQTT broker.

##### `add_callback(callback)`
Add callback function for incoming data.

**Parameters:**
- `callback` (callable): Function to call with incoming data

---

## Preprocessing APIs

### RealTimeDataProcessor

Processes streaming sensor data and extracts features.

#### Constructor
```python
RealTimeDataProcessor(window_size=100, feature_window=10)
```

**Parameters:**
- `window_size` (int): Size of the data buffer
- `feature_window` (int): Window size for feature extraction

#### Methods

##### `add_data_point(data)`
Add new data point to processing buffer.

**Parameters:**
- `data` (dict): Data point with sensor information

##### `get_sensor_features(sensor_id)`
Extract features for a specific sensor.

**Parameters:**
- `sensor_id` (str): ID of the sensor

**Returns:**
- `dict`: Dictionary of extracted features

**Features extracted:**
- `{sensor_id}_mean`: Mean value
- `{sensor_id}_std`: Standard deviation
- `{sensor_id}_min`: Minimum value
- `{sensor_id}_max`: Maximum value
- `{sensor_id}_median`: Median value
- `{sensor_id}_trend`: Trend (slope)
- `{sensor_id}_anomaly_score`: Anomaly score
- `{sensor_id}_current`: Current value

##### `get_all_features()`
Get features for all sensors including cross-sensor features.

**Returns:**
- `dict`: All extracted features

### FeatureExtractor

Utility class for extracting various types of features.

#### Methods

##### `extract_time_features(timestamp)`
Extract time-based features from timestamp.

**Parameters:**
- `timestamp` (str): ISO format timestamp

**Returns:**
- `dict`: Time features including hour, day_of_week, is_weekend, is_business_hours

##### `extract_statistical_features(values, prefix='')`
Extract statistical features from value series.

**Parameters:**
- `values` (list): List of numeric values
- `prefix` (str): Prefix for feature names

**Returns:**
- `dict`: Statistical features (mean, std, var, min, max, median, q25, q75, iqr, skewness, kurtosis)

### DataValidator

Validates sensor data against configured rules.

#### Methods

##### `add_validation_rule(sensor_type, min_val, max_val)`
Add validation rule for sensor type.

**Parameters:**
- `sensor_type` (str): Type of sensor
- `min_val` (float): Minimum valid value
- `max_val` (float): Maximum valid value

##### `validate_data_point(data)`
Validate a single data point.

**Parameters:**
- `data` (dict): Data point to validate

**Returns:**
- `tuple`: (is_valid: bool, message: str)

##### `clean_data_batch(data_batch)`
Clean a batch of data points.

**Parameters:**
- `data_batch` (list): List of data points

**Returns:**
- `list`: List of valid data points

### DataPipeline

Complete data processing pipeline combining all preprocessing components.

#### Constructor
```python
DataPipeline(window_size=100)
```

#### Methods

##### `process_data_point(data)`
Process a single data point through the complete pipeline.

**Parameters:**
- `data` (dict): Sensor data point

**Returns:**
- `dict`: Extracted features or None if invalid

##### `get_feature_vector(sensor_list=None)`
Get feature vector for ML model input.

**Parameters:**
- `sensor_list` (list): Optional list of sensors to filter

**Returns:**
- `numpy.ndarray`: Feature vector

---

## ML Model APIs

### EdgeMLModel

Base class for edge-optimized machine learning models.

#### Constructor
```python
EdgeMLModel(model_type='sklearn')
```

**Parameters:**
- `model_type` (str): Type of model ('sklearn' or 'tensorflow')

#### Methods

##### `create_neural_network(input_dim, output_dim=1)`
Create lightweight neural network.

**Parameters:**
- `input_dim` (int): Number of input features
- `output_dim` (int): Number of output classes

**Returns:**
- `tensorflow.keras.Model`: Created model

##### `create_random_forest(n_estimators=50, max_depth=10)`
Create Random Forest model.

**Parameters:**
- `n_estimators` (int): Number of trees
- `max_depth` (int): Maximum tree depth

**Returns:**
- `sklearn.ensemble.RandomForestClassifier`: Created model

##### `train(X, y, validation_split=0.2)`
Train the model.

**Parameters:**
- `X` (numpy.ndarray): Training features
- `y` (numpy.ndarray): Training labels
- `validation_split` (float): Validation data proportion

##### `predict(X)`
Make predictions.

**Parameters:**
- `X` (numpy.ndarray): Input features

**Returns:**
- `dict`: Prediction results with timing information

**Response format:**
```python
{
    'prediction': int or list,
    'probability': float or list,
    'inference_time_ms': float,
    'model_type': str
}
```

##### `save_model(filepath)`
Save model to disk.

**Parameters:**
- `filepath` (str): Path to save model

##### `load_model(filepath)`
Load model from disk.

**Parameters:**
- `filepath` (str): Path to load model from

##### `optimize_for_edge()`
Optimize model for edge deployment.

**Returns:**
- Model optimized for edge (TensorFlow Lite or original)

### AnomalyDetectionModel

Specialized model for anomaly detection, inherits from EdgeMLModel.

#### Methods

##### `create_isolation_forest(contamination=0.1)`
Create Isolation Forest for anomaly detection.

**Parameters:**
- `contamination` (float): Expected proportion of anomalies

##### `train_anomaly_detector(X)`
Train anomaly detection model.

**Parameters:**
- `X` (numpy.ndarray): Training data (normal samples)

##### `detect_anomaly(X)`
Detect anomalies in data.

**Parameters:**
- `X` (numpy.ndarray): Input data

**Returns:**
- `dict`: Anomaly detection results

**Response format:**
```python
{
    'is_anomaly': bool,
    'anomaly_score': float,
    'threshold': float,
    'inference_time_ms': float
}
```

### ModelValidator

Validates model performance for edge deployment.

#### Methods

##### `validate_model_performance(model, test_data)`
Validate model performance metrics.

**Parameters:**
- `model` (EdgeMLModel): Model to validate
- `test_data` (tuple): (X_test, y_test)

**Returns:**
- `dict`: Performance metrics

##### `validate_edge_constraints(model, max_inference_time_ms=100, max_model_size_mb=10)`
Validate model meets edge constraints.

**Parameters:**
- `model` (EdgeMLModel): Model to validate
- `max_inference_time_ms` (float): Maximum allowed inference time
- `max_model_size_mb` (float): Maximum allowed model size

**Returns:**
- `dict`: Constraint validation results

---

## Inference Engine APIs

### RealTimeInferenceEngine

Main engine for real-time inference processing.

#### Constructor
```python
RealTimeInferenceEngine(model_path=None)
```

#### Methods

##### `load_model(model_name, model_path, model_type='classification')`
Load a trained model for inference.

**Parameters:**
- `model_name` (str): Name to assign to model
- `model_path` (str): Path to model file
- `model_type` (str): Type of model ('classification' or 'anomaly_detection')

**Returns:**
- `bool`: True if successful

##### `add_callback(callback)`
Add callback for inference results.

**Parameters:**
- `callback` (callable): Function to call with InferenceResult

##### `process_sensor_data(sensor_data)`
Process sensor data and run inference.

**Parameters:**
- `sensor_data` (dict): Sensor data to process

**Returns:**
- `InferenceResult`: Inference result or None

##### `start_mqtt_processing(mqtt_broker='localhost', mqtt_port=1883)`
Start processing MQTT data in real-time.

**Parameters:**
- `mqtt_broker` (str): MQTT broker hostname
- `mqtt_port` (int): MQTT broker port

##### `stop_mqtt_processing()`
Stop MQTT processing.

##### `get_stats()`
Get inference engine statistics.

**Returns:**
- `dict`: Performance statistics

##### `get_recent_results(count=10)`
Get recent inference results.

**Parameters:**
- `count` (int): Number of recent results

**Returns:**
- `list`: List of recent InferenceResult objects

### InferenceResult

Data class for inference results.

**Attributes:**
- `timestamp` (str): Result timestamp
- `sensor_id` (str): Source sensor ID
- `prediction` (dict): Prediction results from all models
- `features` (dict): Extracted features
- `inference_time_ms` (float): Total inference time
- `model_version` (str): Model version identifier

### EdgeInferenceAPI

High-level API wrapper for inference engine.

#### Constructor
```python
EdgeInferenceAPI(inference_engine)
```

#### Methods

##### `set_alert_threshold(model_name, threshold)`
Set alert threshold for a model.

**Parameters:**
- `model_name` (str): Name of the model
- `threshold` (float): Alert threshold

##### `check_alerts(result)`
Check if any alerts should be triggered.

**Parameters:**
- `result` (InferenceResult): Inference result to check

**Returns:**
- `list`: List of alert dictionaries

##### `process_with_alerts(sensor_data)`
Process data and check for alerts.

**Parameters:**
- `sensor_data` (dict): Sensor data to process

**Returns:**
- `dict`: API response with results and alerts

**Response format:**
```python
{
    'status': str,
    'result': {
        'timestamp': str,
        'sensor_id': str,
        'predictions': dict,
        'inference_time_ms': float
    },
    'alerts': list,
    'feature_count': int
}
```

### InferenceMonitor

Monitors inference engine performance.

#### Constructor
```python
InferenceMonitor(engine)
```

#### Methods

##### `start_monitoring(interval_seconds=30)`
Start monitoring with specified interval.

**Parameters:**
- `interval_seconds` (int): Monitoring interval

##### `stop_monitoring()`
Stop monitoring.

---

## Configuration APIs

### Configuration File Format

The system uses YAML configuration files. Here's the complete schema:

```yaml
mqtt:
  broker_host: string          # MQTT broker hostname
  broker_port: integer         # MQTT broker port
  topics:
    sensor_data: string        # Topic pattern for sensor data
    alerts: string             # Topic pattern for alerts

data_processing:
  window_size: integer         # Data buffer size
  feature_window: integer      # Feature extraction window
  buffer_size: integer         # Processing buffer size

models:
  default_models:             # List of models to load
    - name: string            # Model name
      type: string            # Model type
      path: string            # Model file path

inference:
  max_inference_time_ms: integer    # Maximum inference time
  max_model_size_mb: integer        # Maximum model size
  alert_thresholds:                 # Alert thresholds
    anomaly_score: float
    temperature_alert: float
    humidity_alert: float

monitoring:
  log_level: string                 # Logging level
  stats_interval_seconds: integer   # Stats reporting interval
  performance_monitoring: boolean   # Enable performance monitoring

sensors:
  simulation:
    enabled: boolean                # Enable sensor simulation
    sensors:                        # List of simulated sensors
      - id: string                  # Sensor ID
        type: string                # Sensor type
        location: string            # Sensor location
        min_val: float              # Minimum value
        max_val: float              # Maximum value
        unit: string                # Unit of measurement
        sampling_rate: float        # Sampling rate (Hz)
        noise_factor: float         # Noise factor
```

## Error Handling

All API methods include comprehensive error handling:

### Common Error Types
- `ValueError`: Invalid parameter values
- `ConnectionError`: MQTT connection issues
- `FileNotFoundError`: Missing model or config files
- `RuntimeError`: Runtime processing errors

### Error Response Format
```python
{
    'status': 'error',
    'message': str,
    'error_type': str,
    'timestamp': str
}
```

## Performance Considerations

### Latency Requirements
- **Data Processing**: <50ms per data point
- **Feature Extraction**: <20ms per sensor
- **Model Inference**: <100ms per prediction
- **End-to-end Pipeline**: <200ms total

### Memory Usage
- **Data Buffers**: Configurable, typically <100MB
- **Model Memory**: <500MB total for all models
- **Feature Storage**: <50MB for feature vectors

### Throughput Capabilities
- **Sensor Data**: 1000+ readings per second
- **Inference Requests**: 100+ predictions per second
- **MQTT Messages**: 5000+ messages per second

---

## Examples

See the [User Guide](../user-guide/README.md) for complete examples and tutorials.