# AI/ML Intelligent AWS Anomaly Detection System (SIH 2026)

> **Smart India Hackathon 2026 — Minimum Viable Product (MVP)**
> Real-time intelligent anomaly detection for Automatic Weather Stations using **Temperature (°C)**, **Pressure (hPa)**, and **Relative Humidity (%)**.

---

## Key Highlights

* **Physics-Guided Hybrid AI**: Combines WMO-No. 8 standards, atmospheric thermodynamics (Magnus-Tetens Dew Point, Vapor Pressure Deficit), and Multivariate Machine Learning (Isolation Forest).
* **Genuine Storms vs Sensor Faults**: Cleanly disentangles severe convective downbursts and cold fronts from sensor hardware failures, achieving **0% false alarm rate on genuine storm events**.
* **Edge AI for ESP32**: Includes a zero-dependency C/C++ embedded library (`edge/esp32_anomaly_detector.h`) with $< 0.1\text{ ms}$ latency and $< 2\text{ KB}$ RAM footprint.
* **Explainable AI (XAI)**: Feature attribution scores (TreeSHAP) and natural language diagnostic rationale for field engineers.
* **Predictive Maintenance**: Real-time **Sensor Health Index (SHI: 0–100%)** tracking drift, noise floor, and failure recurrence.
* **Physics-Constrained Imputation**: Automatically reconstructs corrupted or missing readings within thermodynamic laws ($0 \le RH \le 100\%$, $T_d \le T$).

---

## Project Structure

```
├── app.py                      # Interactive Streamlit Web Dashboard
├── main.py                     # CLI for streaming demo and CSV evaluation
├── benchmark.py                # Comprehensive automated benchmark & metrics runner
├── requirements.txt            # Project dependencies
├── DOCUMENTATION.md            # In-depth technical architecture and use-case report
├── src/
│   ├── physics.py              # Magnus-Tetens thermodynamics & storm signature physics
│   ├── quality_control.py      # WMO-No. 8 plausibility, rate-of-change & persistence tests
│   ├── feature_engineering.py  # 29 sliding window, thermodynamic & cyclical features
│   ├── detector.py             # 5-Tier Hybrid AI Anomaly Engine & XAI
│   ├── imputer.py              # Physics-constrained real-time data reconstructor
│   ├── health_monitor.py       # Predictive maintenance & Sensor Health Index
│   └── data_simulator.py       # Realistic diurnal AWS generator & Anomaly Studio
├── edge/
│   ├── esp32_anomaly_detector.h # Standalone C/C++ library for ESP32
│   └── esp32_anomaly_detector.ino # Arduino/ESP32 sketch
└── tests/
    ├── test_physics.py         # Thermodynamic unit tests
    └── test_detector.py        # QC, detection, imputation & health unit tests
```

---

## Quickstart

### 1. Launch the Web Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` to test the live stream and on-the-fly **Anomaly Injection Studio**.

### 2. Run the Benchmark

```bash
python benchmark.py
```

### 3. Run Unit Tests

```bash
python -m unittest discover tests
```

### 4. Interactive Terminal Demo

```bash
python main.py --demo
```

### 5. Evaluate Any CSV File

```bash
python main.py --evaluate --file your_station_data.csv
```
