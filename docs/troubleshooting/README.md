# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the IoT Edge AI Pipeline.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Installation Problems](#installation-problems)
3. [MQTT Connection Issues](#mqtt-connection-issues)
4. [Model Loading Problems](#model-loading-problems)
5. [Performance Issues](#performance-issues)
6. [Data Processing Errors](#data-processing-errors)
7. [Memory and Resource Issues](#memory-and-resource-issues)
8. [Debugging Tools](#debugging-tools)
9. [Getting Help](#getting-help)

---

## Common Issues

### Quick Diagnostic Checklist

Before diving into specific troubleshooting, run this quick checklist:

```bash
# 1. Check if Python environment is correct
python --version  # Should be 3.8+

# 2. Verify all dependencies are installed
pip list | grep -E "(numpy|pandas|scikit-learn|tensorflow|paho-mqtt)"

# 3. Test basic functionality
python -c "import src.sensors; print('Sensors module OK')"
python -c "import src.preprocessing; print('Preprocessing module OK')"
python -c "import src.ml_models; print('ML models module OK')"

# 4. Check configuration file
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# 5. Test MQTT connectivity (if using)
ping localhost  # Or your MQTT broker host

# 6. Check available disk space
df -h

# 7. Check memory usage
free -h
```

### Log Analysis

Always check logs first:

```bash
# Check application logs
tail -f logs/iot-pipeline.log

# Check system logs
sudo journalctl -u iot-pipeline -f

# Check for Python errors
grep -i "error\|exception\|traceback" logs/iot-pipeline.log
```

---

## Installation Problems

### Python Version Issues

**Problem:** `ImportError` or compatibility issues

**Solution:**
```bash
# Check Python version
python --version

# If using wrong version, create new environment
python3.8 -m venv iot-env
source iot-env/bin/activate

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependency Installation Failures

**Problem:** `pip install` fails for certain packages

**Common solutions:**

**For NumPy/SciPy issues:**
```bash
# Install system dependencies first
sudo apt-get install python3-dev build-essential

# Or use conda instead of pip
conda install numpy scipy scikit-learn
```

**For TensorFlow issues:**
```bash
# For CPU-only version
pip install tensorflow-cpu

# For older systems
pip install tensorflow==2.8.0

# For ARM devices (Raspberry Pi)
pip install tensorflow-aarch64
```

**For compilation errors:**
```bash
# Install build tools
sudo apt-get install gcc g++ make

# Increase swap space on limited memory systems
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Missing System Dependencies

**Problem:** `ModuleNotFoundError` for system libraries

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev libffi-dev libssl-dev

# For HDF5 (used by some ML libraries)
sudo apt-get install libhdf5-dev

# For audio processing (if needed)
sudo apt-get install libasound2-dev

# macOS
brew install hdf5 c-blosc
export HDF5_DIR=/opt/homebrew/
```

---

## MQTT Connection Issues

### Cannot Connect to MQTT Broker

**Problem:** `ConnectionRefusedError` when connecting to MQTT

**Diagnosis:**
```bash
# Check if MQTT broker is running
sudo systemctl status mosquitto

# Test connection manually
mosquitto_pub -h localhost -t test -m "hello"
mosquitto_sub -h localhost -t test

# Check firewall
sudo ufw status
```

**Solutions:**

**Start MQTT broker:**
```bash
# Install if not present
sudo apt-get install mosquitto mosquitto-clients

# Start service
sudo systemctl start mosquitto
sudo systemctl enable mosquitto
```

**Fix connection configuration:**
```python
# In your code, add error handling
try:
    publisher = MQTTSensorPublisher(broker_host='localhost', broker_port=1883)
    publisher.connect()
except Exception as e:
    print(f"MQTT connection failed: {e}")
    # Fallback to file logging or direct processing
```

### MQTT Authentication Issues

**Problem:** Authentication failures with MQTT broker

**Solution:**
```bash
# Create MQTT password file
sudo mosquitto_passwd -c /etc/mosquitto/passwd username

# Configure Mosquitto
sudo tee /etc/mosquitto/conf.d/auth.conf > /dev/null <<EOF
allow_anonymous false
password_file /etc/mosquitto/passwd
EOF

# Restart broker
sudo systemctl restart mosquitto
```

### SSL/TLS Connection Problems

**Problem:** SSL certificate errors

**Solution:**
```python
# Disable SSL verification for testing (NOT for production)
import ssl
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

# Or provide proper certificates
context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
context.load_verify_locations('/path/to/ca.crt')
context.load_cert_chain('/path/to/client.crt', '/path/to/client.key')
```

---

## Model Loading Problems

### Model File Not Found

**Problem:** `FileNotFoundError` when loading models

**Diagnosis:**
```bash
# Check if model files exist
ls -la models/
find models/ -name "*.pkl" -o -name "*_metadata.json"

# Check file permissions
ls -la models/your_model*
```

**Solutions:**

**Create demo models:**
```python
# Run this to create missing demo models
from src.ml_models import EdgeMLModel
import numpy as np

# Generate dummy data and train model
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

model = EdgeMLModel(model_type='sklearn')
model.create_random_forest(n_estimators=10)
model.train(X, y)
model.save_model('models/demo_model')
```

**Fix file permissions:**
```bash
sudo chown -R $(whoami):$(whoami) models/
chmod -R 644 models/*.pkl
chmod -R 644 models/*.json
```

### Model Compatibility Issues

**Problem:** Model fails to load due to version mismatch

**Solutions:**

**Check versions:**
```python
import sklearn
import tensorflow as tf
print(f"scikit-learn: {sklearn.__version__}")
print(f"TensorFlow: {tf.__version__}")
```

**Retrain with current versions:**
```python
# Load old training data and retrain
import pandas as pd
from src.ml_models import EdgeMLModel

# Load your training data
data = pd.read_csv('training_data.csv')
X = data.drop('target', axis=1).values
y = data['target'].values

# Retrain model
model = EdgeMLModel()
model.create_random_forest()
model.train(X, y)
model.save_model('models/updated_model')
```

### Memory Issues During Model Loading

**Problem:** `MemoryError` when loading large models

**Solutions:**

**Optimize model size:**
```python
# Use smaller models for edge deployment
model = EdgeMLModel()
model.create_random_forest(n_estimators=20, max_depth=10)  # Smaller model

# Use model compression
optimized_model = model.optimize_for_edge()
```

**Use lazy loading:**
```python
class LazyModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = EdgeMLModel()
            self._model.load_model(self.model_path)
        return self._model
```

---

## Performance Issues

### Slow Inference Times

**Problem:** Inference takes longer than expected

**Diagnosis:**
```python
import time
import numpy as np

# Profile inference time
def profile_inference(model, num_tests=100):
    times = []
    test_data = np.random.randn(1, 10)

    for _ in range(num_tests):
        start = time.time()
        result = model.predict(test_data)
        end = time.time()
        times.append((end - start) * 1000)

    print(f"Average: {np.mean(times):.2f}ms")
    print(f"Max: {np.max(times):.2f}ms")
    print(f"95th percentile: {np.percentile(times, 95):.2f}ms")
```

**Solutions:**

**Optimize model:**
```python
# Use smaller, faster models
model = EdgeMLModel()
model.create_random_forest(n_estimators=10, max_depth=5)

# Use CPU-optimized libraries
# Install: pip install intel-scikit-learn
import intel_extension_for_sklearn
intel_extension_for_sklearn.patch_sklearn()
```

**Optimize data processing:**
```python
# Cache feature computation
from functools import lru_cache

class OptimizedDataProcessor:
    @lru_cache(maxsize=1000)
    def compute_features(self, sensor_id, values_hash):
        # Cached feature computation
        return self._compute_features(values)
```

**Use batch processing:**
```python
# Process multiple samples at once
def batch_inference(model, data_batch, batch_size=32):
    results = []
    for i in range(0, len(data_batch), batch_size):
        batch = data_batch[i:i+batch_size]
        batch_results = model.predict(np.array(batch))
        results.extend(batch_results)
    return results
```

### High CPU Usage

**Problem:** Application consuming too much CPU

**Solutions:**

**Reduce sampling frequency:**
```yaml
# In config.yaml
sensors:
  simulation:
    sensors:
      - sampling_rate: 0.1  # Reduce from 1.0 to 0.1 Hz
```

**Optimize processing:**
```python
# Use more efficient data structures
from collections import deque
import numpy as np

class EfficientBuffer:
    def __init__(self, maxsize):
        self.buffer = deque(maxlen=maxsize)

    def add_data(self, data):
        self.buffer.append(data)

    def get_array(self):
        return np.array(list(self.buffer))
```

**Use multiprocessing for CPU-bound tasks:**
```python
from multiprocessing import Pool

def process_sensor_data_parallel(data_list):
    with Pool(processes=2) as pool:
        results = pool.map(process_single_sensor, data_list)
    return results
```

---

## Data Processing Errors

### Data Validation Failures

**Problem:** Sensor data fails validation

**Diagnosis:**
```python
# Enable debug logging for validation
import logging
logging.getLogger('src.preprocessing').setLevel(logging.DEBUG)

# Test validation manually
from src.preprocessing import DataValidator

validator = DataValidator()
test_data = {
    'sensor_id': 'test',
    'sensor_type': 'temperature',
    'value': 25.0,
    'timestamp': '2023-12-01T12:00:00'
}

is_valid, message = validator.validate_data_point(test_data)
print(f"Valid: {is_valid}, Message: {message}")
```

**Solutions:**

**Relax validation rules:**
```python
# Adjust validation ranges
validator = DataValidator()
validator.add_validation_rule('temperature', -50.0, 100.0)  # Wider range
```

**Add data cleaning:**
```python
def clean_sensor_data(data):
    # Handle missing values
    if 'value' not in data or data['value'] is None:
        return None

    # Convert string values to float
    try:
        data['value'] = float(data['value'])
    except (ValueError, TypeError):
        return None

    # Add timestamp if missing
    if 'timestamp' not in data:
        data['timestamp'] = datetime.now().isoformat()

    return data
```

### Feature Extraction Errors

**Problem:** Feature extraction fails or produces NaN values

**Solutions:**

**Handle edge cases:**
```python
import numpy as np

def safe_feature_extraction(values):
    features = {}

    if len(values) == 0:
        return features

    # Handle NaN values
    clean_values = [v for v in values if not np.isnan(v)]

    if len(clean_values) == 0:
        return features

    # Safe statistical calculations
    features['mean'] = np.mean(clean_values)
    features['std'] = np.std(clean_values) if len(clean_values) > 1 else 0.0

    return features
```

**Add error recovery:**
```python
def robust_data_pipeline(data):
    try:
        return process_data_point(data)
    except Exception as e:
        logger.warning(f"Data processing failed: {e}")
        # Return basic features as fallback
        return {
            'timestamp': data.get('timestamp'),
            'sensor_id': data.get('sensor_id'),
            'value': data.get('value', 0.0)
        }
```

---

## Memory and Resource Issues

### Memory Leaks

**Problem:** Memory usage grows over time

**Diagnosis:**
```python
import psutil
import gc
import tracemalloc

def monitor_memory():
    tracemalloc.start()

    # Your application code here

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")

    # Get process memory
    process = psutil.Process()
    print(f"Process RSS: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

**Solutions:**

**Limit buffer sizes:**
```python
class MemoryEfficientProcessor:
    def __init__(self, max_buffer_size=1000):
        self.data_buffer = deque(maxlen=max_buffer_size)
        self.cleanup_interval = 100
        self.operation_count = 0

    def add_data(self, data):
        self.data_buffer.append(data)
        self.operation_count += 1

        # Periodic cleanup
        if self.operation_count % self.cleanup_interval == 0:
            gc.collect()
```

**Use generators for large datasets:**
```python
def process_data_stream(data_source):
    for data_batch in data_source:
        # Process batch
        yield process_batch(data_batch)

        # Clear batch from memory
        del data_batch
        gc.collect()
```

### Disk Space Issues

**Problem:** Running out of disk space

**Solutions:**

**Implement log rotation:**
```python
import logging.handlers

# Configure rotating file handler
handler = logging.handlers.RotatingFileHandler(
    'logs/iot-pipeline.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

**Clean old data:**
```bash
#!/bin/bash
# cleanup.sh

# Remove data older than 30 days
find data/raw -name "*.json" -mtime +30 -delete

# Remove old log files
find logs/ -name "*.log.*" -mtime +7 -delete

# Remove temporary model files
find models/ -name "*.tmp" -delete
```

**Monitor disk usage:**
```python
import shutil

def check_disk_space(path='/'):
    total, used, free = shutil.disk_usage(path)

    free_percent = (free / total) * 100

    if free_percent < 10:
        logger.warning(f"Low disk space: {free_percent:.1f}% free")
        # Trigger cleanup
        cleanup_old_files()

    return free_percent
```

---

## Debugging Tools

### Debugging Configuration

**Enable debug logging:**
```python
# In your main application
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Enable debug for specific modules
logging.getLogger('src.sensors').setLevel(logging.DEBUG)
logging.getLogger('src.preprocessing').setLevel(logging.DEBUG)
```

### Performance Profiling

**Profile application performance:**
```python
import cProfile
import pstats

def profile_application():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run your application code
    main_application_function()

    profiler.disable()

    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    # Save profile data
    stats.dump_stats('profile_results.prof')
```

**Visualize profile results:**
```bash
# Install snakeviz for visual profiling
pip install snakeviz

# Visualize profile
snakeviz profile_results.prof
```

### Network Debugging

**Debug MQTT communications:**
```python
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")

def on_message(client, userdata, msg):
    print(f"Received: {msg.topic} - {msg.payload}")

def on_disconnect(client, userdata, rc):
    print(f"Disconnected with result code {rc}")

# Create debug client
debug_client = mqtt.Client()
debug_client.on_connect = on_connect
debug_client.on_message = on_message
debug_client.on_disconnect = on_disconnect

debug_client.connect("localhost", 1883, 60)
debug_client.subscribe("sensors/+/data")
debug_client.loop_forever()
```

### Data Validation Tools

**Create validation scripts:**
```python
# validate_data.py
import json
import sys
from src.preprocessing import DataValidator

def validate_file(filename):
    validator = DataValidator()
    errors = []

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                is_valid, message = validator.validate_data_point(data)

                if not is_valid:
                    errors.append(f"Line {line_num}: {message}")

            except json.JSONDecodeError:
                errors.append(f"Line {line_num}: Invalid JSON")

    return errors

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_data.py <data_file>")
        sys.exit(1)

    errors = validate_file(sys.argv[1])

    if errors:
        print("Validation errors found:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
    else:
        print("Data validation passed")
```

---

## Getting Help

### Collecting Diagnostic Information

**Before asking for help, collect this information:**

```bash
#!/bin/bash
# collect_diagnostics.sh

echo "=== System Information ===" > diagnostics.txt
uname -a >> diagnostics.txt
echo "" >> diagnostics.txt

echo "=== Python Environment ===" >> diagnostics.txt
python --version >> diagnostics.txt
pip list >> diagnostics.txt
echo "" >> diagnostics.txt

echo "=== Application Logs ===" >> diagnostics.txt
tail -100 logs/iot-pipeline.log >> diagnostics.txt
echo "" >> diagnostics.txt

echo "=== Configuration ===" >> diagnostics.txt
cat config/config.yaml >> diagnostics.txt
echo "" >> diagnostics.txt

echo "=== Process Information ===" >> diagnostics.txt
ps aux | grep python >> diagnostics.txt
echo "" >> diagnostics.txt

echo "=== System Resources ===" >> diagnostics.txt
free -h >> diagnostics.txt
df -h >> diagnostics.txt
echo "" >> diagnostics.txt

echo "Diagnostics collected in diagnostics.txt"
```

### Error Reporting Template

**When reporting issues, include:**

1. **Environment details:**
   - Operating system and version
   - Python version
   - Package versions (from `pip list`)

2. **Problem description:**
   - What you were trying to do
   - What you expected to happen
   - What actually happened

3. **Error messages:**
   - Complete error traceback
   - Relevant log entries

4. **Configuration:**
   - Your config.yaml (remove sensitive data)
   - Command used to run the application

5. **Reproduction steps:**
   - Minimal steps to reproduce the issue
   - Sample data if relevant

### Common Support Resources

1. **Check existing documentation:**
   - [User Guide](../user-guide/README.md)
   - [API Documentation](../api/README.md)
   - [Deployment Guide](../deployment/README.md)

2. **Search existing issues:**
   - GitHub Issues
   - Stack Overflow with tags: `iot`, `edge-ai`, `mqtt`

3. **Community forums:**
   - IoT developer communities
   - Machine learning forums
   - MQTT/messaging communities

4. **Professional support:**
   - Consider professional consulting for production deployments
   - Enterprise support options

### Self-Help Checklist

Before seeking help, try these steps:

- [ ] Restart the application
- [ ] Check log files for errors
- [ ] Verify configuration file syntax
- [ ] Test with minimal configuration
- [ ] Update dependencies to latest versions
- [ ] Run the diagnostic script
- [ ] Search documentation for similar issues
- [ ] Try the example scripts to isolate the problem

---

## Quick Reference

### Common Commands

```bash
# Check service status
sudo systemctl status iot-pipeline

# View live logs
tail -f logs/iot-pipeline.log

# Test MQTT connectivity
mosquitto_pub -h localhost -t test -m "hello"

# Check memory usage
free -h

# Check disk space
df -h

# Restart service
sudo systemctl restart iot-pipeline

# Run diagnostics
python -m pytest tests/ -v
```

### Emergency Recovery

**If the system is completely unresponsive:**

1. **Stop the service:**
   ```bash
   sudo systemctl stop iot-pipeline
   ```

2. **Check system resources:**
   ```bash
   top
   free -h
   df -h
   ```

3. **Clear temporary files:**
   ```bash
   rm -rf /tmp/iot-pipeline-*
   find data/ -name "*.tmp" -delete
   ```

4. **Reset to minimal configuration:**
   ```bash
   cp config/minimal.yaml config/config.yaml
   ```

5. **Start with basic functionality:**
   ```bash
   python run_example.py
   ```

This troubleshooting guide should help you resolve most common issues. For persistent problems, don't hesitate to reach out to the community or professional support.