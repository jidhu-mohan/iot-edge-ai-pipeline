# IoT Edge AI Dashboard Instructions

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test components (optional):**
   ```bash
   python test_dashboard_components.py
   ```

3. **Start the dashboard:**
   ```bash
   streamlit run streamlit_dashboard.py
   ```

4. **Open your browser** to `http://localhost:8501`

## How to Use the Dashboard

### Controls
- **▶️ Start Pipeline**: Begin IoT sensor simulation and AI processing
- **⏹️ Stop Pipeline**: Pause the data generation
- **🗑️ Clear Data**: Reset all charts and start fresh

### What You'll See

1. **Real-Time Metrics**: Total readings, anomaly count, and rates
2. **Live Sensor Charts**:
   - Blue dots = Normal readings
   - Red dots = Anomalies detected by AI
   - Orange dashed line = Anomaly confidence score
3. **Anomaly Summary**: Bar chart showing which sensors have the most anomalies
4. **Recent Alerts**: Live feed of anomalies as they're detected
5. **Raw Data Table**: Detailed view of sensor readings (optional)

### Auto-Refresh
- Charts update every 2 seconds when pipeline is running
- Page automatically refreshes to show new data
- No manual refresh needed!

## Educational Features

### Learning Objectives
- **Visualize IoT data flow**: See how sensors generate continuous data streams
- **Understand anomaly detection**: Watch AI identify unusual patterns in real-time
- **Explore feature engineering**: See how raw sensor values become ML features
- **Interactive analysis**: Zoom, hover, and explore the data

### Key Concepts Demonstrated
- **Time-series visualization**: Sensor data plotted over time
- **Machine learning inference**: Real-time AI predictions
- **Data preprocessing**: Feature extraction from raw sensor readings
- **Alert systems**: Automated notifications when anomalies occur

## Troubleshooting

### Common Issues

**Dashboard won't start:**
```bash
# Check if Streamlit is installed
pip install streamlit plotly

# Verify components work
python test_dashboard_components.py
```

**No data showing:**
- Click "▶️ Start Pipeline" in the sidebar
- Wait a few seconds for AI model training
- Charts will populate as data is generated

**Charts not updating:**
- Ensure pipeline status shows "🟢 Running"
- The page auto-refreshes every 2 seconds
- Try stopping and restarting the pipeline

**Browser issues:**
- Try refreshing the page (F5)
- Clear browser cache
- Try a different browser (Chrome, Firefox, Safari)

## Technical Notes

- **Data Generation**: New sensor readings every 2 seconds
- **AI Training**: Model trains on 50 samples before starting predictions
- **Memory Management**: Charts store last 100 data points per sensor
- **Performance**: Optimized for educational use, not production scale

## Comparison with Command-Line Demo

| Feature | Command-Line | Web Dashboard |
|---------|-------------|---------------|
| **Visualization** | Text output | Interactive charts |
| **Real-time** | Scrolling text | Live updating graphs |
| **Control** | Run once | Start/stop/clear |
| **Analysis** | Manual | Interactive zoom/hover |
| **Learning** | Sequential | Visual patterns |

Both versions teach the same concepts - choose based on your learning preference!

---

**💡 Tip**: Run both the command-line demo (`python iot_pipeline_demo.py`) and web dashboard to see different perspectives of the same IoT AI system.