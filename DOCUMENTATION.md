# Intelligent Real-Time AWS Anomaly Detection & Diagnostic System
### Technical Specification and Use-Case Documentation (SIH 2026)

---

## 1. Problem Context and Technical Approach

Automatic Weather Stations (AWS) deployed across India by agencies like the India Meteorological Department (IMD), state agricultural boards, and airport operators face harsh, remote operating conditions. These stations continuously report three foundational surface variables:

1. **Air Temperature (°C)**
2. **Atmospheric Pressure (hPa)**
3. **Relative Humidity (%)**

Existing automated anomaly detection setups tend to fail in two predictable ways:
* **False alarms during real storms:** Standard statistical outlier checks and generic unsupervised ML algorithms (such as plain Isolation Forest or Autoencoders) flag genuine, abrupt weather events—such as thunderstorm downdrafts, squall lines, and frontal boundaries—as sensor failures.
* **Undetected sensor faults:** Low-amplitude hardware failures like frozen ADC converters (flatlines), calibration drift, contact noise, and physical inconsistencies often slip past simple min/max bounds.

### Proposed Architecture

To address these challenges, we built a hybrid detection pipeline combining **WMO-No. 8 operational standards**, **thermodynamic consistency checks** (Magnus-Tetens Dew Point, Vapor Pressure Deficit), **multivariate machine learning**, **feature attribution (TreeSHAP)**, and **C-based edge preprocessing for ESP32 hardware**.

---

## 2. System Architecture

```
+-------------------------------------------------------------------------+
|                  AWS Raw Telemetry Stream (1-15 min)                    |
|               Temperature (°C) | Pressure (hPa) | Humidity (%)          |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|  Stage 1: WMO-No. 8 Standard Quality Control                            |
|  - Plausibility limits ([-40, 60]°C, [800, 1085] hPa, [0, 100]%)        |
|  - Rate-of-change thresholds (max dT/dt, dP/dt, dRH/dt)                 |
|  - Persistence and flatline detection with adaptive resolution learning |
|  - Missing values and communication dropout tracking                    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|  Stage 2: Meteorological Event Disentanglement                          |
|  - Multi-parameter convective cooling and saturation checks             |
|  - Preserves legitimate storm dynamics as GENUINE_WEATHER_EVENT          |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|  Stage 3: Multivariate Machine Learning Engine                           |
|  - 29 thermodynamic, lag, rolling window, and temporal cyclical features|
|  - Calibrated Isolation Forest anomaly scoring                          |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|  Stage 4: Root-Cause Classification                                     |
|  - NORMAL | GENUINE_WEATHER_EVENT | SPIKE | STUCK_SENSOR                |
|  - SENSOR_DRIFT | OUT_OF_BOUNDS | PHYSICAL_INCONSISTENCY | MISSING     |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|  Stage 5: Explainability, Imputation & Health Monitoring                |
|  - TreeSHAP feature attribution and contextual diagnostic notes         |
|  - Physics-constrained cubic spline and trend imputation               |
|  - Sensor Health Index (SHI: 0-100%) tracking degradation               |
+-------------------------------------------------------------------------+
```

---

## 3. Atmospheric Thermodynamics

Rather than relying strictly on unconstrained numerical features, the pipeline evaluates core thermodynamic relationships derived from $T$, $P$, and $RH$:

### 3.1 Saturation Vapor Pressure $e_s(T)$
Calculated using the Sonntag (1990) formulation of the Magnus-Tetens equation:
$$e_s(T) = 6.112 \times \exp\left(\frac{17.67 \cdot T}{T + 243.5}\right) \text{ hPa}$$

For sub-zero temperatures over ice ($T < 0^\circ\text{C}$):
$$e_{s,\text{ice}}(T) = 6.112 \times \exp\left(\frac{22.46 \cdot T}{T + 272.62}\right) \text{ hPa}$$

### 3.2 Actual Vapor Pressure $e$
$$e = e_s(T) \times \left(\frac{RH}{100}\right) \text{ hPa}$$

### 3.3 Dew Point Temperature ($T_d$)
The temperature to which air must be cooled at constant pressure to reach saturation:
$$\gamma(T, RH) = \ln\left(\frac{RH}{100}\right) + \frac{17.67 \cdot T}{243.5 + T}$$
$$T_d = \frac{243.5 \cdot \gamma(T, RH)}{17.67 - \gamma(T, RH)}$$

### 3.4 Dew Point Depression ($DD$)
$$DD = T - T_d$$
* In ambient unsaturated air, $DD \ge 0^\circ\text{C}$ holds by definition.
* If $DD < -0.2^\circ\text{C}$, the system flags a `PHYSICAL_INCONSISTENCY` (impossible ambient supersaturation), indicating circuit contamination or hygrometer drift.

### 3.5 Vapor Pressure Deficit ($VPD$)
$$VPD = e_s(T) - e$$
High $VPD$ corresponds to dry air with high evaporative demand, while values near zero indicate fog or near-surface condensation.

### 3.6 Potential Temperature ($\theta$)
Calculated at the standard 1000 hPa reference level:
$$\theta = (T + 273.15) \times \left(\frac{1000}{P}\right)^{0.286} - 273.15$$

---

## 4. Differentiating Storms from Sensor Faults

During severe weather (e.g., convective storms, gust fronts, or squall lines), surface weather instruments observe sharp, correlated shifts:
* Ambient temperature drops $4^\circ\text{C}$ to $8^\circ\text{C}$ in under 20 minutes due to rain-cooled downdrafts.
* Relative humidity jumps rapidly toward saturation ($85\%\text{--}100\%$).
* Barometric pressure displays a sharp perturbation or gust front nose ($|\Delta P| \ge 1.0\text{ hPa}$).

Standard rate-of-change checks flag these abrupt jumps as spikes. The Stage 2 filter examines the joint multi-parameter vector:
$$\mathbf{\Delta}_{30} = [\Delta T_{30m}, \Delta P_{30m}, \Delta RH_{30m}]$$

When negative temperature deltas correlate with positive humidity deltas and pressure fluctuations, the reading is marked as `GENUINE_WEATHER_EVENT`, preventing false sensor alarms while preserving critical meteorological data for forecasting models.

---

## 5. Edge Deployment for ESP32 Microcontrollers

In remote installations (such as mountain passes, desert outposts, and offshore buoys), cellular and satellite bandwidth is limited and power budgets are tight.

### Embedded C Library (`edge/esp32_anomaly_detector.h`)
* **Zero Dependencies:** Pure C99/C++ code. No external libraries or RTOS requirements.
* **Low Memory Footprint:** Uses less than 1.5 KB RAM with a 16-sample circular buffer, fitting comfortably in constrained microcontrollers.
* **Low Latency:** Executes in under 0.1 ms per sample on a standard ESP32 at 240 MHz.
* **Bandwidth Savings:** Can report only anomaly flags or state transitions, reducing satellite transmission costs during calm periods.

---

## 6. Sensor Health Index and Predictive Maintenance

The health monitoring module tracks hardware degradation trends:
* **Temperature (RTD / Thermistor):** Tracks high-frequency signal variance, contact noise floor, and erratic single-sample spikes.
* **Pressure (Piezoresistive / Barometer):** Tracks diurnal tidal wave attenuation and port blockage.
* **Humidity (Capacitive Polymer):** Monitors sensor drift, saturation latching (hanging near 100%), and hysteresis errors.

### Health Index Tiers:
| Health Score | Status | Maintenance Recommendation |
|---|---|---|
| **90 - 100%** | Nominal | Sensor operating within standard bounds. Continue routine monitoring. |
| **75 - 89%** | Good | Minor signal noise detected. Plan inspection during next regular site visit. |
| **50 - 74%** | Degraded | Clean or replace protective filter cap; verify cable connections and shields. |
| **< 50%** | Critical | Immediate technician dispatch required. Sensor shows flatline or persistent drift. |

---

## 7. Real-World Applications

### 1. National Meteorological Networks (e.g., IMD)
* 1,500+ automatic weather stations reporting synoptic observations.
* Automates initial quality control before observations enter Numerical Weather Prediction (NWP) data assimilation cycles.

### 2. Agricultural Weather Networks and Crop Insurance
* Panchayat-level weather stations used for index-based crop insurance (e.g., PMFBY).
* Protects against erroneous payout claims caused by stuck or drifted sensors while keeping valid drought or heatwave records intact.

### 3. Aviation Ground Observations (AWOS / METAR)
* Runway weather monitoring stations providing wind shear, temperature, and altimeter settings.
* Immediately separates gust front downbursts from sensor failures to support airport safety operations.

### 4. Flood and Landslide Early Warning
* River basin catchment networks monitoring cloudburst precursors.
* Edge screening on low-power hardware triggers local sirens even when remote uplinks are temporarily unavailable.

---

## 8. Benchmark Evaluation

Evaluated across a 5-day continuous dataset (7,200 observations at 1-minute resolution) with injected ground-truth faults:

| Evaluation Metric | Measured Value | Target Baseline |
|---|---|---|
| **Overall Accuracy** | **98.8%** | > 90.0% |
| **Precision** | **92.0%** | > 85.0% |
| **Recall (Detection Rate)** | **97.0%** | > 90.0% |
| **F1-Score** | **0.945** | > 0.850 |
| **Specificity** | **99.0%** | > 95.0% |
| **Average Latency (Python)** | **~8.3 ms** / sample | < 50.0 ms |
| **Edge Latency (ESP32)** | **< 0.1 ms** / sample | < 1.0 ms |

---

## 9. Running the Pipeline

### Interactive Dashboard
```bash
streamlit run app.py
```

### Terminal Streaming Demo
```bash
python main.py --demo
```

### CSV Telemetry Evaluation
```bash
python main.py --evaluate --file incompass_kanpur_1min.csv
```

### Automated Benchmark Suite
```bash
python benchmark.py
```

### Unit Tests
```bash
python -m unittest discover tests
```
