# IoT Edge AI Pipeline - Student Version

**A simplified, educational demonstration of IoT sensors connected to AI for real-time anomaly detection**

## What You'll Learn

This educational demo teaches you the fundamentals of Industrial IoT and AI systems:

1. **IoT Sensor Simulation** - How sensors generate data in real-world systems
2. **Data Processing** - Converting raw sensor readings into useful information
3. **Feature Engineering** - Preparing data for machine learning models
4. **Anomaly Detection** - Using AI to automatically identify unusual patterns
5. **Real-time Processing** - Handling continuous data streams

## Real-World Applications

This simplified pipeline demonstrates concepts used in:

- **Manufacturing**: Monitoring equipment health and preventing failures
- **Smart Buildings**: Optimizing HVAC and detecting issues
- **Agriculture**: Monitoring soil conditions and crop health
- **Healthcare**: Monitoring patient vitals and medical equipment
- **Transportation**: Monitoring vehicle performance and maintenance needs

## Quick Start

### Prerequisites

```bash
# Install required Python packages
pip install -r requirements.txt

# Or install manually:
pip install numpy pandas scikit-learn streamlit plotly
```

### Run the Demo

**Option 1: Command Line Demo (Original)**
```bash
cd student-version
python iot_pipeline_demo.py
```

**Option 2: Interactive Web Dashboard (Recommended)**
```bash
cd student-version
streamlit run streamlit_dashboard.py
```

The web dashboard will open in your browser at `http://localhost:8501` and provide:
- Real-time interactive charts
- Live anomaly detection visualization
- Educational explanations and controls
- Data tables and metrics

## What Happens When You Run It

### Phase 1: Training (15 seconds)
```
Collecting training data for 15 seconds...
This teaches the AI what normal sensor behavior looks like.
✓ Collected 45 feature samples for training
✓ Anomaly detector trained on 45 samples
```

The system collects "normal" sensor data to teach the AI model what typical operation looks like.

### Phase 2: Real-time Monitoring (30 seconds)
```
Starting real-time monitoring for 30 seconds...
The AI will now analyze sensor data and detect anomalies in real-time.

[2024-01-15 14:30:21] temp_001: 22.5°C | ✅ Normal (confidence: 0.45)
[2024-01-15 14:30:22] humid_001: 58.2% | ✅ Normal (confidence: 0.32)
[2024-01-15 14:30:23] vibration_001: 1.8m/s² | 🚨 ANOMALY (confidence: 0.67)
   ⚠️  Alert: Unusual pattern detected in vibration_001
```

The system processes live sensor data and the AI identifies unusual patterns in real-time.

### Interactive Web Dashboard
```
🏭 IoT Edge AI Pipeline - Live Dashboard

📊 Real-Time Metrics
Total Readings: 156    Anomalies Detected: 23    Anomaly Rate: 14.7%    Active Sensors: 4

🌡️ Live Sensor Data
[Interactive charts showing real-time sensor data with anomaly highlighting]

🚨 Anomaly Detection Summary
[Bar chart of anomalies by sensor type + Recent alerts panel]
```

The web dashboard provides:
- **Interactive charts**: Zoom, pan, and hover for details
- **Real-time updates**: Data refreshes every second
- **Visual anomaly detection**: Red dots for anomalies, blue for normal
- **Educational controls**: Start/stop pipeline, clear data
- **Live metrics**: Running totals and statistics

## Understanding the Code Structure

### 1. IoTSensor Class
```python
class IoTSensor:
    def read_sensor(self) -> Dict[str, Any]:
        # Simulates reading from a physical sensor
        # Returns: timestamp, sensor_id, value, unit
```

**What it does**: Simulates real IoT sensors like temperature probes, humidity sensors, accelerometers, etc.

**Key concept**: In real systems, you'd interface with actual hardware using protocols like Modbus, I2C, or HTTP APIs.

### 2. DataProcessor Class
```python
class DataProcessor:
    def _extract_features(self, sensor_id: str) -> Dict[str, float]:
        # Converts raw readings into ML-ready features
        # Calculates: mean, std, min, max, range, trend
```

**What it does**: Transforms raw sensor values into statistical features that machine learning models can understand.

**Key concept**: Raw sensor readings (like "23.5°C") aren't directly useful for ML. We need to calculate patterns over time.

### 3. AnomalyDetector Class
```python
class AnomalyDetector:
    def train(self, training_data):
        # Learns what "normal" operation looks like

    def predict(self, features):
        # Identifies if current readings are unusual
```

**What it does**: Uses machine learning (Isolation Forest algorithm) to automatically detect when sensor patterns are unusual.

**Key concept**: Instead of setting fixed thresholds, ML can learn complex patterns and adapt to changing conditions.

### 4. IoTPipeline Class
```python
class IoTPipeline:
    def run_real_time_monitoring(self):
        # Orchestrates the complete data flow
        # Sensors → Processing → ML → Alerts
```

**What it does**: Coordinates the entire system - manages sensors, processes data, runs AI inference, and generates alerts.

## Key Concepts Explained

### Feature Engineering
```python
features = {
    'temp_001_mean': 22.3,      # Average temperature over time window
    'temp_001_std': 0.8,        # How much variation (stability)
    'temp_001_trend': 0.2,      # Is it increasing or decreasing?
    'temp_001_range': 2.1       # Difference between min and max
}
```

These features tell the AI much more than individual readings:
- **Mean**: Is the equipment running hot or cold?
- **Standard deviation**: Is it stable or fluctuating wildly?
- **Trend**: Is there a gradual change happening?
- **Range**: Are there sudden spikes or drops?

### Anomaly Detection Process

1. **Training Phase**: Show the AI 50+ examples of normal operation
2. **Learning**: AI builds a model of what "normal" looks like
3. **Inference**: When new data arrives, AI compares it to learned patterns
4. **Decision**: If the pattern is sufficiently different, flag as anomaly

### Why This Matters

Traditional monitoring uses fixed thresholds:
```python
if temperature > 30:  # Fixed threshold
    alert("Temperature too high!")
```

AI-based monitoring is adaptive:
```python
if ai_model.predict(features)['is_anomaly']:  # Learned patterns
    alert("Unusual temperature pattern detected!")
```

The AI approach can detect:
- Gradual changes that indicate wear
- Unusual combinations of sensor readings
- Patterns that precede failures
- Seasonal or time-based variations

## Extending the Demo

### Add New Sensor Types

```python
# Add a pressure sensor
pipeline.add_sensor('press_002', 'pressure', 1000.0, 1050.0, 'Pa')

# Add a flow rate sensor
pipeline.add_sensor('flow_001', 'flow_rate', 10.0, 50.0, 'L/min')
```

### Modify Anomaly Sensitivity

```python
# More sensitive (detects smaller anomalies)
self.model = IsolationForest(contamination=0.05)

# Less sensitive (only major anomalies)
self.model = IsolationForest(contamination=0.2)
```

### Add Custom Features

```python
def _extract_features(self, sensor_id: str) -> Dict[str, float]:
    # Add custom calculations
    features['custom_ratio'] = values[-1] / values[0]  # Latest vs first
    features['acceleration'] = np.diff(values[-3:]).mean()  # Rate of change
```

## Common Questions

**Q: How is this different from simple threshold alerts?**
A: Traditional systems alert when values exceed fixed limits. This AI system learns normal patterns and detects unusual combinations, trends, and relationships between sensors.

**Q: What makes this "edge AI"?**
A: The processing happens locally (on edge devices) rather than sending all data to the cloud. This reduces latency, bandwidth costs, and improves reliability.

**Q: How would this scale to real industrial systems?**
A: Real systems might have thousands of sensors. The same principles apply, but you'd need distributed processing, data storage, and more sophisticated ML models.

**Q: What happens if the AI makes mistakes?**
A: All AI systems have false positives and negatives. In production, you'd combine AI insights with human expertise and gradually improve the model with feedback.

## Next Steps for Students

1. **Explore the web dashboard** - Use the interactive Streamlit interface to visualize data
2. **Modify the demo** - Change sensor parameters, add new sensors, adjust timing
3. **Experiment with features** - Try different statistical calculations
4. **Test different ML models** - Replace Isolation Forest with other algorithms
5. **Compare interfaces** - Run both command-line and web versions to see differences
6. **Study the main codebase** - See how this scales to production systems

### Dashboard Features to Explore

- **Real-time charts**: Watch sensor data update live with anomaly highlighting
- **Interactive controls**: Start/stop the pipeline, clear data, adjust parameters
- **Data analysis**: Use the raw data table to understand feature engineering
- **Pattern recognition**: Observe how different sensors behave and correlate

## Connections to Production Systems

This educational demo simplifies many production concepts:

- **Data Storage**: Real systems use databases (InfluxDB, PostgreSQL)
- **Communication**: MQTT, HTTP APIs, industrial protocols
- **Scalability**: Container orchestration, load balancing
- **Security**: Authentication, encryption, secure communication
- **Monitoring**: Dashboards, logging, performance metrics
- **Model Management**: Versioning, A/B testing, continuous learning

## Further Learning

- **IoT Protocols**: MQTT, CoAP, LoRaWAN
- **Time Series Analysis**: Forecasting, seasonal decomposition
- **Machine Learning**: Supervised learning, deep learning for time series
- **Edge Computing**: TensorFlow Lite, ONNX, model optimization
- **Industrial Systems**: OPC-UA, Modbus, industrial networking

---

**This demo provides a foundation for understanding how modern industrial IoT systems work. The same principles scale from simple monitoring to complex predictive maintenance and optimization systems used in manufacturing, energy, and smart cities.**