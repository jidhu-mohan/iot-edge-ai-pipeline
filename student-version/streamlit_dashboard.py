#!/usr/bin/env python3
"""
IoT Edge AI Pipeline - Streamlit Dashboard
==========================================

Interactive web dashboard for visualizing IoT sensor data and AI anomaly detection
in real-time. This provides students with a visual interface to understand the
pipeline better.

FEATURES:
1. Real-time sensor data visualization
2. Anomaly detection alerts and charts
3. Interactive controls for pipeline parameters
4. Educational explanations of what's happening
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Import our pipeline components
from iot_pipeline_demo import IoTSensor, DataProcessor, AnomalyDetector

# Configure Streamlit page
st.set_page_config(
    page_title="IoT Edge AI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = defaultdict(lambda: deque(maxlen=100))
if 'anomaly_data' not in st.session_state:
    st.session_state.anomaly_data = deque(maxlen=100)
if 'pipeline_components' not in st.session_state:
    st.session_state.pipeline_components = None
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

class StreamlitPipeline:
    """Simplified pipeline that works synchronously with Streamlit"""

    def __init__(self):
        self.sensors = {}
        self.data_processor = DataProcessor()
        self.anomaly_detector = AnomalyDetector()
        self.is_trained = False

    def add_sensor(self, sensor_id: str, sensor_type: str, min_val: float, max_val: float, unit: str):
        """Add a new sensor to the pipeline"""
        self.sensors[sensor_id] = IoTSensor(sensor_id, sensor_type, min_val, max_val, unit)

    def train_model(self, num_samples=50):
        """Train the anomaly detection model"""
        training_data = []

        for _ in range(num_samples):
            for sensor in self.sensors.values():
                reading = sensor.read_sensor()
                features = self.data_processor.add_reading(reading)
                if features:
                    training_data.append(features)

        if training_data:
            self.anomaly_detector.train(training_data)
            self.is_trained = True

    def generate_reading(self):
        """Generate one reading from all sensors"""
        results = []

        for sensor in self.sensors.values():
            reading = sensor.read_sensor()
            features = self.data_processor.add_reading(reading)

            if features and self.is_trained:
                # Run anomaly detection
                result = self.anomaly_detector.predict(features)

                # Package data for dashboard
                dashboard_data = {
                    'timestamp': datetime.now(),
                    'sensor_id': reading['sensor_id'],
                    'sensor_type': reading['sensor_type'],
                    'value': reading['value'],
                    'unit': reading['unit'],
                    'is_anomaly': result['is_anomaly'],
                    'anomaly_score': result['anomaly_score'],
                    'confidence': result['confidence']
                }
                results.append(dashboard_data)

        return results

def create_sensor_chart(sensor_data, sensor_id, sensor_type, unit):
    """Create a real-time chart for a single sensor"""
    if not sensor_data:
        return go.Figure().add_annotation(text="No data yet...", xref="paper", yref="paper", x=0.5, y=0.5)

    df = pd.DataFrame(sensor_data)

    # Create figure with secondary y-axis for anomaly scores
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Color points based on anomaly detection
    colors = ['red' if anomaly else 'blue' for anomaly in df['is_anomaly']]

    # Add sensor values
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines+markers',
            name=f'{sensor_type.title()} ({unit})',
            line=dict(color='blue'),
            marker=dict(color=colors, size=8),
            hovertemplate=f'<b>{sensor_type.title()}</b><br>' +
                         'Time: %{x}<br>' +
                         f'Value: %{{y}} {unit}<br>' +
                         '<extra></extra>'
        ),
        secondary_y=False,
    )

    # Add anomaly scores
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['anomaly_score'],
            mode='lines',
            name='Anomaly Score',
            line=dict(color='orange', dash='dot'),
            opacity=0.7,
            hovertemplate='<b>Anomaly Score</b><br>' +
                         'Time: %{x}<br>' +
                         'Score: %{y:.3f}<br>' +
                         '<extra></extra>'
        ),
        secondary_y=True,
    )

    # Update layout
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text=f"{sensor_type.title()} ({unit})", secondary_y=False)
    fig.update_yaxes(title_text="Anomaly Score", secondary_y=True)

    fig.update_layout(
        title=f"{sensor_id} - {sensor_type.title()} Sensor",
        hovermode='x unified',
        height=300,
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def create_anomaly_summary_chart(anomaly_data):
    """Create a summary chart of all anomalies"""
    if not anomaly_data:
        return go.Figure().add_annotation(text="No anomaly data yet...", xref="paper", yref="paper", x=0.5, y=0.5)

    df = pd.DataFrame(anomaly_data)

    # Count anomalies by sensor type
    anomaly_counts = df[df['is_anomaly']].groupby('sensor_type').size().reset_index(name='count')

    if anomaly_counts.empty:
        return go.Figure().add_annotation(text="No anomalies detected yet", xref="paper", yref="paper", x=0.5, y=0.5)

    fig = px.bar(
        anomaly_counts,
        x='sensor_type',
        y='count',
        title="Anomalies Detected by Sensor Type",
        color='count',
        color_continuous_scale='Reds'
    )

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def process_new_data(data_list):
    """Process new data from the pipeline and update session state"""
    for data in data_list:
        # Store sensor data
        sensor_key = data['sensor_id']
        st.session_state.sensor_data[sensor_key].append({
            'timestamp': data['timestamp'],
            'value': data['value'],
            'is_anomaly': data['is_anomaly'],
            'anomaly_score': data['anomaly_score'],
            'confidence': data['confidence']
        })

        # Store anomaly data
        st.session_state.anomaly_data.append({
            'timestamp': data['timestamp'],
            'sensor_id': data['sensor_id'],
            'sensor_type': data['sensor_type'],
            'is_anomaly': data['is_anomaly'],
            'anomaly_score': data['anomaly_score'],
            'confidence': data['confidence']
        })

def initialize_pipeline():
    """Initialize the pipeline components"""
    if st.session_state.pipeline_components is None:
        pipeline = StreamlitPipeline()

        # Add sensors
        pipeline.add_sensor('temp_001', 'temperature', 18.0, 28.0, '°C')
        pipeline.add_sensor('humid_001', 'humidity', 40.0, 70.0, '%')
        pipeline.add_sensor('vibration_001', 'vibration', 0.5, 3.0, 'm/s²')
        pipeline.add_sensor('pressure_001', 'pressure', 980.0, 1020.0, 'hPa')

        # Train the model
        with st.spinner("Training AI model..."):
            pipeline.train_model(num_samples=50)

        st.session_state.pipeline_components = pipeline

def generate_new_data():
    """Generate new sensor readings if pipeline is running"""
    if st.session_state.pipeline_running and st.session_state.pipeline_components:
        current_time = time.time()

        # Generate new data every 2 seconds
        if current_time - st.session_state.last_update > 2:
            new_data = st.session_state.pipeline_components.generate_reading()
            if new_data:
                process_new_data(new_data)
            st.session_state.last_update = current_time

def main():
    """Main dashboard application"""

    # Title and description
    st.title("🏭 IoT Edge AI Pipeline - Live Dashboard")
    st.markdown("""
    **Educational Demo**: Watch IoT sensors generate data in real-time while AI detects anomalies.
    This dashboard shows the same concepts as the command-line demo but with interactive visualizations.
    """)

    # Initialize pipeline components
    initialize_pipeline()

    # Sidebar controls
    st.sidebar.title("🎛️ Controls")

    # Start/Stop buttons
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("▶️ Start Pipeline", type="primary"):
            st.session_state.pipeline_running = True
            st.sidebar.success("Pipeline started!")

    with col2:
        if st.button("⏹️ Stop Pipeline"):
            st.session_state.pipeline_running = False
            st.sidebar.info("Pipeline stopped!")

    # Pipeline status
    status = "🟢 Running" if st.session_state.pipeline_running else "🔴 Stopped"
    st.sidebar.markdown(f"**Status**: {status}")

    # Clear data button
    if st.sidebar.button("🗑️ Clear Data"):
        st.session_state.sensor_data.clear()
        st.session_state.anomaly_data.clear()
        st.sidebar.success("Data cleared!")

    # Educational information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 What You're Seeing")
    st.sidebar.markdown("""
    - **Blue dots**: Normal sensor readings
    - **Red dots**: Anomalies detected by AI
    - **Orange line**: Anomaly confidence score
    - **Charts update**: Every 2 seconds with new data
    """)

    # Generate new data if pipeline is running
    generate_new_data()

    # Main dashboard layout
    if st.session_state.pipeline_running or st.session_state.sensor_data:

        # Summary metrics
        st.markdown("### 📊 Real-Time Metrics")

        total_readings = sum(len(data) for data in st.session_state.sensor_data.values())
        total_anomalies = sum(1 for data in st.session_state.anomaly_data if data.get('is_anomaly', False))
        anomaly_rate = (total_anomalies / max(total_readings, 1)) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Readings", total_readings)
        col2.metric("Anomalies Detected", total_anomalies)
        col3.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        col4.metric("Active Sensors", len(st.session_state.sensor_data))

        # Sensor charts
        st.markdown("### 🌡️ Live Sensor Data")

        # Create two columns for sensor charts
        col1, col2 = st.columns(2)

        sensor_configs = [
            ('temp_001', 'temperature', '°C'),
            ('humid_001', 'humidity', '%'),
            ('vibration_001', 'vibration', 'm/s²'),
            ('pressure_001', 'pressure', 'hPa')
        ]

        for i, (sensor_id, sensor_type, unit) in enumerate(sensor_configs):
            sensor_data = list(st.session_state.sensor_data.get(sensor_id, []))

            if i % 2 == 0:
                with col1:
                    fig = create_sensor_chart(sensor_data, sensor_id, sensor_type, unit)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    fig = create_sensor_chart(sensor_data, sensor_id, sensor_type, unit)
                    st.plotly_chart(fig, use_container_width=True)

        # Anomaly summary
        st.markdown("### 🚨 Anomaly Detection Summary")

        col1, col2 = st.columns([2, 1])

        with col1:
            anomaly_fig = create_anomaly_summary_chart(list(st.session_state.anomaly_data))
            st.plotly_chart(anomaly_fig, use_container_width=True)

        with col2:
            st.markdown("#### Recent Alerts")
            recent_anomalies = [data for data in list(st.session_state.anomaly_data)[-10:] if data.get('is_anomaly', False)]

            if recent_anomalies:
                for anomaly in reversed(recent_anomalies[-5:]):  # Show last 5
                    timestamp = anomaly['timestamp'].strftime('%H:%M:%S')
                    st.error(f"🚨 {timestamp} - {anomaly['sensor_id']}: Anomaly detected!")
            else:
                st.info("No recent anomalies detected")

        # Data table
        if st.checkbox("📋 Show Raw Data"):
            st.markdown("### 📋 Recent Sensor Readings")

            # Combine all recent data
            all_data = []
            for sensor_id, data_list in st.session_state.sensor_data.items():
                for data_point in list(data_list)[-20:]:  # Last 20 points per sensor
                    all_data.append({
                        'Sensor ID': sensor_id,
                        'Timestamp': data_point['timestamp'].strftime('%H:%M:%S'),
                        'Value': f"{data_point['value']:.2f}",
                        'Anomaly': "🚨 Yes" if data_point['is_anomaly'] else "✅ No",
                        'Confidence': f"{data_point['confidence']:.3f}"
                    })

            if all_data:
                df = pd.DataFrame(all_data)
                st.dataframe(df.sort_values('Timestamp', ascending=False), use_container_width=True)

    else:
        # Welcome screen
        st.markdown("### 🚀 Welcome to the IoT Edge AI Dashboard!")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            This interactive dashboard demonstrates how IoT sensors connect to AI systems for real-time monitoring.

            **To get started:**
            1. Click "▶️ Start Pipeline" in the sidebar
            2. Watch as sensors generate data
            3. See AI detect anomalies in real-time
            4. Explore the interactive charts and metrics

            **Educational Value:**
            - Visualize the complete IoT-to-AI flow
            - Understand how anomaly detection works
            - See real-time data processing in action
            - Learn about industrial IoT monitoring
            """)

            st.info("💡 **Tip**: This dashboard complements the command-line demo in `iot_pipeline_demo.py`")

    # Auto-refresh for real-time updates
    if st.session_state.pipeline_running:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()