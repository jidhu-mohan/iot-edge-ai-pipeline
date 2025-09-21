# Deployment Guide

This guide covers deploying the IoT Edge AI Pipeline in various environments, from development to production edge devices.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Development Deployment](#development-deployment)
3. [Production Deployment](#production-deployment)
4. [Edge Device Deployment](#edge-device-deployment)
5. [Cloud Integration](#cloud-integration)
6. [Docker Deployment](#docker-deployment)
7. [Kubernetes Deployment](#kubernetes-deployment)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)

---

## Deployment Overview

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Production Environment                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Edge Device │    │ Edge Device │    │ Edge Device │     │
│  │             │    │             │    │             │     │
│  │ • Sensors   │    │ • Sensors   │    │ • Sensors   │     │
│  │ • ML Models │    │ • ML Models │    │ • ML Models │     │
│  │ • Local AI  │    │ • Local AI  │    │ • Local AI  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│        │                   │                   │            │
│        └───────────────────┼───────────────────┘            │
│                            │                                │
│  ┌─────────────────────────▼─────────────────────────┐      │
│  │               MQTT Broker                        │      │
│  │         (Mosquitto / AWS IoT / Azure)            │      │
│  └─────────────────────────┬─────────────────────────┘      │
│                            │                                │
│  ┌─────────────────────────▼─────────────────────────┐      │
│  │            Inference Engine                      │      │
│  │                                                  │      │
│  │ • Data Processing    • Model Management          │      │
│  │ • Feature Engineering • Alert Processing         │      │
│  │ • Real-time ML       • Performance Monitoring    │      │
│  └─────────────────────────┬─────────────────────────┘      │
│                            │                                │
│  ┌─────────────────────────▼─────────────────────────┐      │
│  │              Data Storage & Analytics            │      │
│  │                                                  │      │
│  │ • Time Series DB     • Model Storage             │      │
│  │ • Alert Logs         • Performance Metrics       │      │
│  │ • Historical Data    • Dashboards                │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Options

1. **Single Machine**: Development and small-scale deployments
2. **Edge Devices**: Raspberry Pi, NVIDIA Jetson, industrial PCs
3. **Container Orchestration**: Docker Swarm, Kubernetes
4. **Cloud Integration**: AWS IoT, Azure IoT Hub, Google Cloud IoT
5. **Hybrid**: Edge processing with cloud analytics

---

## Development Deployment

### Local Development Setup

**Requirements:**
- Python 3.8+
- 4GB+ RAM
- 10GB+ storage
- Network connectivity

**Installation:**
```bash
# Clone repository
git clone <repository-url>
cd iot-edge-ai-pipeline

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python main.py --config config/config.yaml
```

### Development with MQTT Broker

**Install Mosquitto:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients

# macOS
brew install mosquitto

# Windows
# Download from https://mosquitto.org/download/

# Start broker
sudo systemctl start mosquitto  # Linux
brew services start mosquitto   # macOS
```

**Test MQTT Connection:**
```bash
# Subscribe to sensor data
mosquitto_sub -h localhost -t "sensors/+/data"

# Publish test data
mosquitto_pub -h localhost -t "sensors/test/data" -m '{"sensor_id":"test","value":25.5}'
```

### Development Configuration

Create `config/dev_config.yaml`:
```yaml
mqtt:
  broker_host: localhost
  broker_port: 1883

monitoring:
  log_level: DEBUG
  stats_interval_seconds: 10
  performance_monitoring: true

sensors:
  simulation:
    enabled: true
    sensors:
      - id: "dev_temp_001"
        type: "temperature"
        sampling_rate: 2.0  # Higher rate for testing

inference:
  max_inference_time_ms: 200  # Relaxed for development
```

---

## Production Deployment

### Production Requirements

**Hardware:**
- **CPU**: 4+ cores, 2.5GHz+
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 50GB+ SSD
- **Network**: Stable internet connection
- **Power**: UPS for critical deployments

**Software:**
- **OS**: Ubuntu 20.04+ LTS, CentOS 8+, Windows Server 2019+
- **Python**: 3.8+
- **MQTT Broker**: Mosquitto, RabbitMQ, or cloud service
- **Database**: PostgreSQL, InfluxDB (optional)
- **Monitoring**: Prometheus, Grafana (optional)

### Production Installation

**1. System Setup:**
```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y python3 python3-pip python3-venv git

# Create application user
sudo useradd -m -s /bin/bash iotpipeline
sudo su - iotpipeline
```

**2. Application Installation:**
```bash
# Clone and setup application
git clone <repository-url> iot-edge-ai-pipeline
cd iot-edge-ai-pipeline

# Create production environment
python3 -m venv prod-env
source prod-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create production directories
mkdir -p logs data/raw data/processed models/production
```

**3. Production Configuration:**
```yaml
# config/production.yaml
mqtt:
  broker_host: production-mqtt-broker.company.com
  broker_port: 8883  # TLS port
  username: "${MQTT_USERNAME}"
  password: "${MQTT_PASSWORD}"
  tls_enabled: true

monitoring:
  log_level: INFO
  log_file: logs/iot-pipeline.log
  stats_interval_seconds: 60
  performance_monitoring: true

inference:
  max_inference_time_ms: 100
  max_model_size_mb: 10
  alert_thresholds:
    anomaly_score: 0.8

data_storage:
  raw_data_path: data/raw
  processed_data_path: data/processed
  retention_days: 30
```

### Service Configuration

**Create systemd service:**
```bash
sudo tee /etc/systemd/system/iot-pipeline.service > /dev/null <<EOF
[Unit]
Description=IoT Edge AI Pipeline
After=network.target

[Service]
Type=simple
User=iotpipeline
Group=iotpipeline
WorkingDirectory=/home/iotpipeline/iot-edge-ai-pipeline
Environment=PATH=/home/iotpipeline/iot-edge-ai-pipeline/prod-env/bin
ExecStart=/home/iotpipeline/iot-edge-ai-pipeline/prod-env/bin/python main.py --config config/production.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable iot-pipeline
sudo systemctl start iot-pipeline

# Check status
sudo systemctl status iot-pipeline
```

### Security Configuration

**1. MQTT Security:**
```bash
# Generate certificates for MQTT TLS
sudo openssl req -new -x509 -days 365 -extensions v3_ca -keyout ca.key -out ca.crt
sudo openssl genrsa -out server.key 2048
sudo openssl req -new -key server.key -out server.csr
sudo openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365

# Configure Mosquitto with TLS
sudo tee /etc/mosquitto/conf.d/tls.conf > /dev/null <<EOF
port 8883
cafile /etc/mosquitto/certs/ca.crt
certfile /etc/mosquitto/certs/server.crt
keyfile /etc/mosquitto/certs/server.key
tls_version tlsv1.2
EOF
```

**2. Firewall Configuration:**
```bash
# Configure firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8883/tcp  # MQTT TLS
sudo ufw allow 1883/tcp  # MQTT (if needed)
```

**3. Environment Variables:**
```bash
# Create environment file
sudo tee /etc/environment >> /dev/null <<EOF
MQTT_USERNAME=production_user
MQTT_PASSWORD=secure_password
MODEL_ENCRYPTION_KEY=your_encryption_key
EOF
```

---

## Edge Device Deployment

### Raspberry Pi Deployment

**Requirements:**
- Raspberry Pi 4 (4GB+ RAM recommended)
- 32GB+ microSD card (Class 10)
- Stable power supply
- Network connectivity

**Installation:**
```bash
# Update Raspberry Pi OS
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3 python3-pip python3-venv git

# Install system dependencies for ML libraries
sudo apt-get install -y libatlas-base-dev libhdf5-dev

# Clone and setup
git clone <repository-url> iot-edge-ai-pipeline
cd iot-edge-ai-pipeline

# Create lightweight environment
python3 -m venv edge-env
source edge-env/bin/activate

# Install optimized dependencies
pip install -r requirements-edge.txt
```

**Create `requirements-edge.txt`:**
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow-lite-runtime>=2.8.0  # Lightweight TensorFlow
paho-mqtt>=1.6.0
pyyaml>=6.0
psutil>=5.8.0
```

**Edge-specific configuration:**
```yaml
# config/edge.yaml
data_processing:
  window_size: 50      # Reduced for memory constraints
  feature_window: 5    # Smaller feature window

inference:
  max_inference_time_ms: 200  # Relaxed for edge hardware
  max_model_size_mb: 5        # Smaller models for edge

monitoring:
  log_level: WARNING          # Reduce log verbosity
  stats_interval_seconds: 120 # Less frequent stats

models:
  optimization:
    quantization: true        # Enable model quantization
    pruning: true            # Enable model pruning
```

### NVIDIA Jetson Deployment

**For GPU-accelerated inference:**
```bash
# Install JetPack SDK (includes TensorFlow GPU support)
# Follow NVIDIA's installation guide

# Install additional dependencies
sudo apt-get install -y python3-dev

# Setup CUDA environment
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install GPU-optimized packages
pip install tensorflow-gpu==2.8.0
pip install cupy-cuda111  # For GPU-accelerated NumPy operations
```

### Industrial PC Deployment

**For harsh environment deployment:**
```bash
# Configure for fanless operation
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Setup watchdog for automatic restart
sudo tee /etc/systemd/system/iot-watchdog.service > /dev/null <<EOF
[Unit]
Description=IoT Pipeline Watchdog
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/watchdog -v -c /etc/watchdog.conf
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

---

## Cloud Integration

### AWS IoT Core Integration

**Setup AWS IoT:**
```python
# aws_iot_integration.py
import boto3
import json
from src.sensors import MQTTSensorPublisher

class AWSIoTPublisher(MQTTSensorPublisher):
    def __init__(self, endpoint, cert_path, key_path, ca_path):
        import ssl
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client()
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_verify_locations(ca_path)
        context.load_cert_chain(cert_path, key_path)

        self.client.tls_set_context(context)
        self.client.connect(endpoint, 8883, 60)

    def publish_to_aws(self, sensor_data, thing_name):
        topic = f"iot/{thing_name}/sensor-data"
        self.client.publish(topic, json.dumps(sensor_data))
```

**AWS deployment configuration:**
```yaml
# config/aws.yaml
cloud:
  provider: aws
  iot_endpoint: your-endpoint.iot.us-east-1.amazonaws.com
  thing_name: edge-device-001
  certificates:
    cert_path: /etc/ssl/aws-iot/cert.pem
    key_path: /etc/ssl/aws-iot/private.key
    ca_path: /etc/ssl/aws-iot/ca.pem

data_pipeline:
  cloud_sync: true
  sync_interval: 300  # 5 minutes
```

### Azure IoT Hub Integration

```python
# azure_iot_integration.py
from azure.iot.device import IoTHubDeviceClient

class AzureIoTClient:
    def __init__(self, connection_string):
        self.client = IoTHubDeviceClient.create_from_connection_string(connection_string)

    async def send_sensor_data(self, sensor_data):
        message = Message(json.dumps(sensor_data))
        message.content_encoding = "utf-8"
        message.content_type = "application/json"

        await self.client.send_message(message)
```

### Google Cloud IoT Integration

```python
# gcp_iot_integration.py
import jwt
import datetime
from google.cloud import iot_v1

class GCPIoTClient:
    def __init__(self, project_id, registry_id, device_id, private_key_file):
        self.project_id = project_id
        self.registry_id = registry_id
        self.device_id = device_id
        self.private_key_file = private_key_file

    def create_jwt_token(self):
        token = {
            'iat': datetime.datetime.utcnow(),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
            'aud': self.project_id
        }

        with open(self.private_key_file, 'r') as f:
            private_key = f.read()

        return jwt.encode(token, private_key, algorithm='RS256')
```

---

## Docker Deployment

### Basic Docker Setup

**Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 iotuser && chown -R iotuser:iotuser /app
USER iotuser

# Expose ports
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["python", "main.py", "--config", "config/docker.yaml"]
```

**Create docker-compose.yml:**
```yaml
version: '3.8'

services:
  iot-pipeline:
    build: .
    container_name: iot-edge-ai-pipeline
    restart: unless-stopped
    environment:
      - MQTT_BROKER_HOST=mosquitto
      - MQTT_BROKER_PORT=1883
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - mosquitto
    networks:
      - iot-network

  mosquitto:
    image: eclipse-mosquitto:2.0
    container_name: mosquitto
    restart: unless-stopped
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - ./mosquitto/config:/mosquitto/config
      - ./mosquitto/data:/mosquitto/data
      - ./mosquitto/log:/mosquitto/log
    networks:
      - iot-network

  influxdb:
    image: influxdb:2.0
    container_name: influxdb
    restart: unless-stopped
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=password
      - DOCKER_INFLUXDB_INIT_ORG=iot-org
      - DOCKER_INFLUXDB_INIT_BUCKET=sensor-data
    ports:
      - "8086:8086"
    volumes:
      - influxdb-data:/var/lib/influxdb2
    networks:
      - iot-network

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - iot-network

networks:
  iot-network:
    driver: bridge

volumes:
  influxdb-data:
  grafana-data:
```

**Deploy with Docker Compose:**
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f iot-pipeline

# Scale inference service
docker-compose up -d --scale iot-pipeline=3

# Stop services
docker-compose down
```

---

## Kubernetes Deployment

### Kubernetes Manifests

**Namespace:**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: iot-pipeline
```

**ConfigMap:**
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: iot-pipeline-config
  namespace: iot-pipeline
data:
  config.yaml: |
    mqtt:
      broker_host: mosquitto-service
      broker_port: 1883
    inference:
      max_inference_time_ms: 100
    monitoring:
      log_level: INFO
```

**Deployment:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iot-pipeline
  namespace: iot-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iot-pipeline
  template:
    metadata:
      labels:
        app: iot-pipeline
    spec:
      containers:
      - name: iot-pipeline
        image: iot-edge-ai-pipeline:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        env:
        - name: MQTT_BROKER_HOST
          value: "mosquitto-service"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: iot-pipeline-config
```

**Service:**
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: iot-pipeline-service
  namespace: iot-pipeline
spec:
  selector:
    app: iot-pipeline
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

**Deploy to Kubernetes:**
```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n iot-pipeline

# View logs
kubectl logs -f deployment/iot-pipeline -n iot-pipeline

# Scale deployment
kubectl scale deployment iot-pipeline --replicas=5 -n iot-pipeline
```

---

## Monitoring and Maintenance

### Monitoring Setup

**1. Application Metrics:**
```python
# monitoring.py
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
INFERENCE_COUNTER = Counter('inference_total', 'Total inferences')
INFERENCE_DURATION = Histogram('inference_duration_seconds', 'Inference duration')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')

class MetricsCollector:
    def __init__(self):
        start_http_server(8000)  # Prometheus metrics endpoint

    def record_inference(self, duration):
        INFERENCE_COUNTER.inc()
        INFERENCE_DURATION.observe(duration)

    def update_system_metrics(self):
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)
```

**2. Log Aggregation:**
```python
# logging_config.py
import logging
import logging.handlers

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/iot-pipeline.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )

    # Console handler
    console_handler = logging.StreamHandler()

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
```

### Maintenance Tasks

**1. Automated Backups:**
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/iot-pipeline"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# Backup configuration
cp -r config/ $BACKUP_DIR/config_$DATE/

# Backup recent data (last 7 days)
find data/ -name "*.json" -mtime -7 -exec cp {} $BACKUP_DIR/data_$DATE/ \;

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

**2. Health Checks:**
```python
# health_check.py
import requests
import sys
import time

def check_service_health():
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            print("Service is healthy")
            return True
        else:
            print(f"Service unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def check_mqtt_connection():
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client()
        client.connect("localhost", 1883, 60)
        client.disconnect()
        print("MQTT broker is reachable")
        return True
    except Exception as e:
        print(f"MQTT check failed: {e}")
        return False

if __name__ == "__main__":
    checks = [
        check_service_health,
        check_mqtt_connection
    ]

    for check in checks:
        if not check():
            sys.exit(1)

    print("All health checks passed")
```

**3. Performance Monitoring:**
```bash
#!/bin/bash
# monitor.sh

LOG_FILE="/var/log/iot-pipeline-monitor.log"

while true; do
    echo "$(date): Starting health checks" >> $LOG_FILE

    # Check service status
    if ! systemctl is-active --quiet iot-pipeline; then
        echo "$(date): Service down, restarting..." >> $LOG_FILE
        systemctl restart iot-pipeline
    fi

    # Check disk space
    DISK_USAGE=$(df /home/iotpipeline | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ $DISK_USAGE -gt 85 ]; then
        echo "$(date): Disk usage high: ${DISK_USAGE}%" >> $LOG_FILE
        # Cleanup old logs
        find /home/iotpipeline/iot-edge-ai-pipeline/logs -name "*.log.*" -mtime +7 -delete
    fi

    # Check memory usage
    MEM_USAGE=$(free | awk 'NR==2{printf "%.2f", $3*100/$2}')
    if (( $(echo "$MEM_USAGE > 90" | bc -l) )); then
        echo "$(date): High memory usage: ${MEM_USAGE}%" >> $LOG_FILE
    fi

    sleep 300  # Check every 5 minutes
done
```

### Cron Jobs

```bash
# Add to crontab: crontab -e

# Daily backup
0 2 * * * /home/iotpipeline/scripts/backup.sh

# Hourly health check
0 * * * * /usr/bin/python3 /home/iotpipeline/scripts/health_check.py

# Weekly log rotation
0 0 * * 0 /usr/sbin/logrotate /etc/logrotate.d/iot-pipeline

# Monthly model retraining (if applicable)
0 0 1 * * /home/iotpipeline/scripts/retrain_models.sh
```

This deployment guide provides comprehensive coverage for deploying the IoT Edge AI Pipeline across various environments. Choose the deployment strategy that best fits your requirements and infrastructure.