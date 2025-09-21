#!/usr/bin/env python3
"""
Test script to verify dashboard components work without Streamlit
"""

from streamlit_dashboard import StreamlitPipeline, create_sensor_chart
import time

def test_pipeline():
    """Test the StreamlitPipeline components"""
    print("Testing StreamlitPipeline components...")

    # Create pipeline
    pipeline = StreamlitPipeline()

    # Add sensors
    pipeline.add_sensor('temp_001', 'temperature', 18.0, 28.0, '°C')
    pipeline.add_sensor('humid_001', 'humidity', 40.0, 70.0, '%')

    print("✓ Sensors added successfully")

    # Train model
    print("Training model...")
    pipeline.train_model(num_samples=20)
    print("✓ Model trained successfully")

    # Generate some readings
    print("Generating test readings...")
    for i in range(5):
        readings = pipeline.generate_reading()
        if readings:
            for reading in readings:
                status = "🚨 ANOMALY" if reading['is_anomaly'] else "✅ Normal"
                print(f"  {reading['sensor_id']}: {reading['value']:.2f} | {status}")
        time.sleep(0.5)

    print("\n✅ All pipeline components working correctly!")

if __name__ == "__main__":
    test_pipeline()