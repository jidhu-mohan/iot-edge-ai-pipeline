#!/usr/bin/env python3
"""
Simple script to test if our Streamlit dashboard can be imported and run
"""

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")

        import plotly.graph_objects as go
        print("✓ Plotly imported successfully")

        import pandas as pd
        import numpy as np
        print("✓ Data processing libraries imported successfully")

        from iot_pipeline_demo import IoTSensor, DataProcessor, AnomalyDetector, IoTPipeline
        print("✓ IoT pipeline components imported successfully")

        print("\n🎉 All dependencies are working!")
        print("\nTo start the dashboard run:")
        print("streamlit run streamlit_dashboard.py")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    print("🔍 Testing dashboard dependencies...\n")
    test_imports()