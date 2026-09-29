# 🌦️ AI/ML Intelligent AWS Anomaly Detection System (SIH 2026)

> **Smart India Hackathon 2026 — Minimum Viable Product (MVP)**
> Real-time intelligent anomaly detection for Automatic Weather Stations using **Temperature (°C)**, **Pressure (hPa)**, and **Relative Humidity (%)**.

---

## 🚀 Key Highlights

* **Physics-Guided Hybrid AI**: Combines WMO-No. 8 standards, atmospheric thermodynamics (Magnus-Tetens Dew Point, Vapor Pressure Deficit), and Multivariate Machine Learning (Isolation Forest).
* **Sensor-Aware Persistence Calibration**: Automatically learns physical sensor reporting resolution and 99th-percentile flatline run lengths from the training window, eliminating false stuck-sensor alarms on coarsely-quantized barometers.
* **Adaptive Dual-Rate Replay (15-min Routine / 1-min Storm)**: Operates at standard 15-minute synoptic intervals to save ~93% telemetry and compute bandwidth, and autonomously switches to 1-minute high-frequency mode during convective storms and squalls.
* **Strict Anti-Leakage Train/Test Engine**: Interactive or CLI-driven date window selection guaranteeing zero data leakage and strict chronological integrity (`Test Start > Train End`).
* **Genuine Storms vs Sensor Faults**: Cleanly disentangles severe convective downbursts and cold fronts from sensor hardware failures, achieving **0% false alarm rate on genuine storm events**.
* **Edge AI for ESP32**: Includes a zero-dependency C/C++ embedded library (`edge/esp32_anomaly_detector.h`) with $< 0.1\text{ ms}$ latency and $< 2\text{ KB}$ RAM footprint.
* **Explainable AI (XAI)**: Feature attribution scores (TreeSHAP) and natural language diagnostic rationale for field engineers.
* **Predictive Maintenance**: Real-time **Sensor Health Index (SHI: 0–100%)** tracking drift, noise floor, and failure recurrence.
* **Physics-Constrained Imputation**: Automatically reconstructs corrupted or missing readings within thermodynamic laws ($0 \le RH \le 100\%$, $T_d \le T$).

---

## 🛠️ Project Structure

```
├── app.py                      # Interactive Streamlit Web Dashboard (SkyGuard)
├── main.py                     # CLI for streaming demo and adaptive CSV evaluation
├── benchmark.py                # Comprehensive automated benchmark & metrics runner
├── eval_kanpur_train_test_interactive.py # Dedicated Kanpur station evaluation replay
├── diagnose_sensor_resolution.py # Empirical sensor resolution & flatline distribution diagnostic
├── requirements.txt            # Project dependencies
├── DOCUMENTATION.md            # In-depth technical architecture and use-case report
├── src/
│   ├── physics.py              # Magnus-Tetens thermodynamics & storm signature physics
│   ├── quality_control.py      # WMO-No. 8 plausibility, rate-of-change & adaptive persistence
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

## ⚡ Quickstart

### 1. Launch the Web Dashboard

```bash
streamlit run app.py
```

* **Live Simulation**: On-the-fly **Fault Injection Studio** (spikes, drifts, stuck sensors, convective storms).
* **Real CSV Replay**: Select **Kanpur Station 1-Min Telemetry** or upload any station CSV. Configure **Train & Test Date Windows** with one click to calibrate sensor profiles and stream test observations in **Adaptive (15m/1m)** mode.

### 2. Interactive Terminal Demo (Default)

```bash
python main.py --demo
```

### 3. Evaluate Any CSV File with Interactive Train/Test Windows

```bash
python main.py --evaluate --file incompass_kanpur_1min.csv
```

* Prompts interactively for **Training START/END** and **Testing START/END** dates.
* Automatically validates anti-leakage and chronological order.
* Calibrates sensor-specific resolution and stuck-run limits from training data.
* Runs adaptive 15-min normal / 1-min storm replay, and outputs summary + CSV logs.

### 4. Non-Interactive / Scripted Evaluation

```bash
python main.py --evaluate --file incompass_kanpur_1min.csv --non-interactive
```

### 5. Run the Automated Benchmark

```bash
python benchmark.py
```

### 6. Run Unit Tests

```bash
python -m unittest discover tests
```
