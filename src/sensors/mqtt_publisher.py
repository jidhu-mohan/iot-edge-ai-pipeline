import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
from .sensor_simulator import IoTSensorSimulator, setup_default_sensors, SensorReading

class MQTTSensorPublisher:
    def __init__(self, broker_host='localhost', broker_port=1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client()
        self.connected = False

        # Setup callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        else:
            print(f"Failed to connect to MQTT broker: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print("Disconnected from MQTT broker")

    def _on_publish(self, client, userdata, mid):
        pass  # Can add logging here if needed

    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()

            # Wait for connection
            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1

            if not self.connected:
                raise Exception("Failed to connect within timeout")

        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")
            raise

    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()

    def publish_sensor_data(self, reading: SensorReading):
        """Publish sensor reading to MQTT topic"""
        if not self.connected:
            print("Not connected to MQTT broker")
            return False

        topic = f"sensors/{reading.sensor_id}/data"
        payload = {
            'sensor_id': reading.sensor_id,
            'sensor_type': reading.sensor_type,
            'timestamp': reading.timestamp,
            'value': reading.value,
            'unit': reading.unit,
            'location': reading.location,
            'metadata': reading.metadata
        }

        try:
            result = self.client.publish(topic, json.dumps(payload))
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"Error publishing to MQTT: {e}")
            return False

class MQTTDataIngestion:
    def __init__(self, broker_host='localhost', broker_port=1883):
        self.client = mqtt.Client()
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.data_queue = []
        self.connected = False
        self.callbacks = []

        # Setup callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print("Connected to MQTT broker for data ingestion")
            client.subscribe("sensors/+/data")  # Subscribe to all sensor data
        else:
            print(f"Failed to connect to MQTT broker: {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            # Decode message
            payload = json.loads(msg.payload.decode())
            payload['mqtt_timestamp'] = datetime.now().isoformat()
            payload['topic'] = msg.topic

            # Add to queue
            self.data_queue.append(payload)

            # Call registered callbacks
            for callback in self.callbacks:
                callback(payload)

        except Exception as e:
            print(f"Error processing MQTT message: {e}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print("Disconnected from MQTT broker")

    def connect(self):
        """Connect to MQTT broker and start listening"""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()

            # Wait for connection
            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1

            if not self.connected:
                raise Exception("Failed to connect within timeout")

        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")
            raise

    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()

    def add_callback(self, callback):
        """Add callback function for incoming data"""
        self.callbacks.append(callback)

    def get_latest_data(self, count=10):
        """Get latest received data"""
        return self.data_queue[-count:] if self.data_queue else []

    def clear_queue(self):
        """Clear the data queue"""
        self.data_queue.clear()

if __name__ == "__main__":
    # Example: Publisher
    publisher = MQTTSensorPublisher()
    simulator = setup_default_sensors()

    try:
        publisher.connect()

        def publish_callback(reading):
            success = publisher.publish_sensor_data(reading)
            if success:
                print(f"Published: {reading.sensor_id} = {reading.value} {reading.unit}")

        simulator.start_simulation(publish_callback)
        print("Publishing sensor data to MQTT. Press Ctrl+C to stop.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        simulator.stop_simulation()
        publisher.disconnect()
        print("Stopped.")