# Intelligent AWS Anomaly Detection System (SIH 2026)

Real-time anomaly detection system for Automatic Weather Stations using three core parameters: **Temperature (°C)**, **Atmospheric Pressure (hPa)**, and **Relative Humidity (%)**.

The system distinguishes between genuine severe weather events (convective storms, downbursts, cold fronts) and sensor malfunctions (spikes, flatlines, calibration drift, packet dropouts) by coupling WMO-No. 8 quality control with atmospheric thermodynamics and multivariate machine learning.

---

## Key Capabilities

* **Physics-Guided Hybrid AI**: Integrates WMO-No. 8 physical limits, thermodynamic consistency checks (Magnus-Tetens Dew Point, Vapor Pressure Deficit), and multivariate Isolation Forest.
* **Severe Weather vs. Sensor Fault Disentanglement**: Prevents false alarms during rapid meteorological shifts by evaluating multi-parameter coupling (e.g., simultaneous rain cooling and humidity saturation).
* **Sensor-Aware Persistence Calibration**: Automatically infers instrument resolution and flatline run distributions from baseline historical data, avoiding false stuck-sensor alerts on coarsely quantized barometers.
* **Dual-Rate Adaptive Streaming**: Runs at 15-minute intervals during calm conditions to minimize telemetry bandwidth and switches to 1-minute sampling upon detecting rapid atmospheric shifts.
* **Edge Deployment on Microcontrollers**: Standalone, dependency-free C99 library (`edge/esp32_anomaly_detector.h`) designed for low-power ESP32 and ARM Cortex microcontrollers with sub-millisecond execution and a low memory footprint.
* **Explainable Diagnostics (XAI)**: Provides TreeSHAP feature attributions and diagnostic explanations for maintenance crews.
* **Predictive Maintenance**: Tracks signal-to-noise ratio, drift rate, and fault recurrence to compute a continuous Sensor Health Index (SHI: 0–100%).
* **Thermodynamic Data Imputation**: Fills missing or corrupted points using cubic spline and trend extrapolation bounded by physical constraints ($0 \le RH \le 100\%$, $T_d \le T$).

---

## Repository Structure

```
├── app.py                      # Streamlit web dashboard entrypoint
├── index.py                    # Dashboard implementation
├── main.py                     # CLI for streaming simulation and CSV evaluation
├── benchmark.py                # Automated benchmark and validation suite
├── requirements.txt            # Python dependencies
├── DOCUMENTATION.md            # Technical documentation and use-case analysis
├── src/
│   ├── physics.py              # Thermodynamic relations and storm signature logic
│   ├── quality_control.py      # WMO-No. 8 range, step-change, and persistence checks
│   ├── feature_engineering.py  # Lag, rolling statistics, and cyclical temporal features
│   ├── detector.py             # Multi-tier anomaly detection engine and XAI
│   ├── imputer.py              # Physics-constrained value reconstruction
│   ├── health_monitor.py       # Predictive maintenance and sensor health scoring
│   └── data_simulator.py       # Weather data generator with anomaly injection
├── edge/
│   ├── esp32_anomaly_detector.h # Standalone C99/C++ header for embedded targets
│   └── esp32_anomaly_detector.ino # Arduino/ESP32 example firmware sketch
└── tests/
    ├── test_physics.py         # Thermodynamic calculations unit tests
    └── test_detector.py        # QC, detector, and imputation unit tests
```

---

## Setup and Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### 1. Web Dashboard

Launch the interactive dashboard to visualize live data streams, test real-time anomaly injection, and replay station datasets:

```bash
streamlit run app.py
```

### 2. Command-Line Simulation Demo

Run a quick streaming demo in the terminal:

```bash
python main.py --demo
```

### 3. Evaluate CSV Telemetry Data

Evaluate an existing AWS dataset with train/test window configuration:

```bash
python main.py --evaluate --file incompass_kanpur_1min.csv
```

For non-interactive scripted execution:

```bash
python main.py --evaluate --file incompass_kanpur_1min.csv --non-interactive
```

### 4. Run Benchmark Suite

Execute the benchmark suite on injected fault scenarios to compute accuracy, precision, recall, and false alarm metrics:

```bash
python benchmark.py
```

### 5. Run Unit Tests

```bash
python -m unittest discover tests
```
