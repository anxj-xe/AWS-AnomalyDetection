"""
Interactive Streamlit Dashboard for SIH 2026 Automatic Weather Station Anomaly Detection.
Features:
- Live streaming telemetry simulation with pause/play controls
- Interactive on-the-fly Anomaly Injection Studio
- Real-time Sensor Health Gauges & Predictive Maintenance
- Multi-parameter interactive Plotly graphs (Raw vs Anomalies vs Imputed)
- Explainable AI (XAI) feature attribution & natural language reasoning
- Edge AI (ESP32) architecture viewer & data export
"""

import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_simulator import AWSDataSimulator
from src.detector import AWSAnomalyDetector, AnomalyReport
from src.imputer import AWSImputer
from src.health_monitor import SensorHealthMonitor
from src.physics import AtmosphericPhysics

# Page configuration
st.set_page_config(
    page_title="AWS Intelligent Anomaly Detection | SIH 2026",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 14px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .status-badge-ok {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-storm {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-fault {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if 'initialized' not in st.session_state:
    st.session_state.simulator = AWSDataSimulator(seed=101)
    st.session_state.detector = AWSAnomalyDetector(contamination=0.03)
    st.session_state.imputer = AWSImputer()
    st.session_state.health_monitor = SensorHealthMonitor()

    # Pre-train baseline on 2 days clean data
    clean_hist = st.session_state.simulator.generate_historical_dataset(days=2, interval_minutes=5, inject_anomalies=False)
    st.session_state.detector.fit(clean_hist)

    # Telemetry streaming buffer
    st.session_state.history = []
    st.session_state.anomaly_log = []
    st.session_state.stream_step = 0
    st.session_state.is_streaming = False
    st.session_state.active_fault = None
    st.session_state.active_fault_param = None
    st.session_state.active_fault_value = 0.0

    # Seed initial 30 observations so graphs start populated
    base_time = pd.Timestamp.now() - pd.Timedelta(minutes=30)
    for i in range(30):
        t, p, rh = st.session_state.simulator.generate_point(base_time + pd.Timedelta(minutes=i))
        rep = st.session_state.detector.process_observation(t, p, rh, base_time + pd.Timedelta(minutes=i))
        imp = st.session_state.imputer.impute_point(t, p, rh, None, i)
        st.session_state.imputer.update_clean_history(t, p, rh, i)
        st.session_state.health_monitor.record_observation(t, p, rh, None)
        st.session_state.history.append({
            "step": i,
            "timestamp": base_time + pd.Timedelta(minutes=i),
            "temperature": t, "pressure": p, "humidity": rh,
            "imp_temp": imp["imputed_temperature"], "imp_press": imp["imputed_pressure"], "imp_rh": imp["imputed_humidity"],
            "is_anomaly": False, "is_weather_event": False, "anomaly_type": "NORMAL", "confidence": 1.0,
            "faulty_sensor": None, "explanation": rep.explanation, "thermo": rep.thermodynamics, "top_features": []
        })
    st.session_state.stream_step = 30
    st.session_state.initialized = True


# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/partly-cloudy-day.png", width=64)
st.sidebar.markdown("### **AWS Control Station**")
st.sidebar.caption("SIH 2026 • AI-Based Anomaly Detection")

st.sidebar.selectbox("Active Station", [
    "AWS-IN-DELHI-04 (Safdarjung)",
    "AWS-IN-BENGALURU-02 (GKVK)",
    "AWS-IN-SHILLONG-01 (Barapani)",
    "AWS-IN-MUMBAI-05 (Santacruz)"
])

st.sidebar.markdown("---")
st.sidebar.markdown("#### **Telemetry Stream Control**")

col_s1, col_s2 = st.sidebar.columns(2)
if col_s1.button("▶️ Step (+1m)", use_container_width=True):
    st.session_state.step_once = True
else:
    st.session_state.step_once = False

stream_toggle = col_s2.button("⏯️ Stream Mode", use_container_width=True)
if stream_toggle:
    st.session_state.is_streaming = not st.session_state.is_streaming

speed = st.sidebar.slider("Stream Interval (sec)", 0.2, 2.0, 0.6, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("#### **🧪 Anomaly Injection Studio**")
st.sidebar.caption("Test the AI's ability to differentiate faults vs genuine weather:")

inject_choice = st.sidebar.selectbox("Select Anomaly Scenario", [
    "None (Nominal Diurnal Stream)",
    "⚡ Temperature Spike (+12.5°C)",
    "📉 Barometer Pressure Drop Spike (-16.0 hPa)",
    "💧 Humidity Sensor Spike (+40.0%)",
    "❄️ Stuck / Frozen Sensor (Zero Variance)",
    "📈 Sensor Calibration Drift (+0.04°C/min)",
    "🌪️ GENUINE Severe Convective Storm (Downburst)",
    "🚫 Out of Physical Bounds (RH = 125%)",
    "❌ Packet Loss / Missing Data (NaN)"
])

if st.sidebar.button("💥 Trigger Selected Injection", use_container_width=True):
    if "None" in inject_choice:
        st.session_state.active_fault = None
        st.sidebar.success("Cleared all injections.")
    elif "Temperature Spike" in inject_choice:
        st.session_state.active_fault = "SPIKE"
        st.session_state.active_fault_param = "temperature"
        st.session_state.active_fault_value = 12.5
        st.sidebar.warning("Injected: +12.5°C Temperature Spike!")
    elif "Barometer Pressure Drop" in inject_choice:
        st.session_state.active_fault = "SPIKE"
        st.session_state.active_fault_param = "pressure"
        st.session_state.active_fault_value = -16.0
        st.sidebar.warning("Injected: -16.0 hPa Barometer Spike!")
    elif "Humidity Sensor Spike" in inject_choice:
        st.session_state.active_fault = "SPIKE"
        st.session_state.active_fault_param = "humidity"
        st.session_state.active_fault_value = 40.0
        st.sidebar.warning("Injected: +40% Humidity Spike!")
    elif "Stuck" in inject_choice:
        st.session_state.active_fault = "STUCK_SENSOR"
        st.session_state.active_fault_param = "temperature"
        st.session_state.active_fault_value = 32.4
        st.sidebar.warning("Injected: Frozen Temperature Sensor!")
    elif "Drift" in inject_choice:
        st.session_state.active_fault = "SENSOR_DRIFT"
        st.session_state.active_fault_param = "temperature"
        st.session_state.active_fault_value = 0.0
        st.sidebar.warning("Injected: Progressive Temperature Sensor Drift!")
    elif "GENUINE Severe Convective Storm" in inject_choice:
        st.session_state.active_fault = "GENUINE_WEATHER_EVENT"
        st.sidebar.info("Triggered: Genuine Severe Convective Storm Event!")
    elif "Out of Physical Bounds" in inject_choice:
        st.session_state.active_fault = "OUT_OF_BOUNDS"
        st.session_state.active_fault_param = "humidity"
        st.session_state.active_fault_value = 125.0
        st.sidebar.error("Injected: Out of Bounds RH (125%)!")
    elif "Packet Loss" in inject_choice:
        st.session_state.active_fault = "MISSING"
        st.session_state.active_fault_param = "temperature"
        st.sidebar.error("Injected: Missing Data Packet (NaN)!")


# Function to generate next observation step
def process_next_step():
    st.session_state.stream_step += 1
    step = st.session_state.stream_step
    curr_time = st.session_state.history[-1]["timestamp"] + pd.Timedelta(minutes=1)
    t, p, rh = st.session_state.simulator.generate_point(curr_time)

    # Apply injected fault if active
    fault = st.session_state.active_fault
    param = st.session_state.active_fault_param

    if fault == "SPIKE":
        if param == "temperature":
            t += st.session_state.active_fault_value
        elif param == "pressure":
            p += st.session_state.active_fault_value
        elif param == "humidity":
            rh = min(100.0, rh + st.session_state.active_fault_value)
        st.session_state.active_fault = None  # Single-point spike reset

    elif fault == "STUCK_SENSOR":
        if param == "temperature":
            t = st.session_state.active_fault_value
        elif param == "humidity":
            rh = st.session_state.active_fault_value

    elif fault == "SENSOR_DRIFT":
        st.session_state.active_fault_value += 0.04
        t += st.session_state.active_fault_value

    elif fault == "GENUINE_WEATHER_EVENT":
        # Severe Thunderstorm: sharp temperature drop (-6.5 C), pressure nose, RH near saturation (98%)
        t -= 6.2
        p -= 2.6
        rh = 97.5
        st.session_state.active_fault = None  # Front passage pulse

    elif fault == "OUT_OF_BOUNDS":
        if param == "humidity":
            rh = st.session_state.active_fault_value
        elif param == "temperature":
            t = 78.0
        st.session_state.active_fault = None

    elif fault == "MISSING":
        t = np.nan
        st.session_state.active_fault = None

    # Run AI Detection Engine
    report: AnomalyReport = st.session_state.detector.process_observation(
        temperature=t, pressure=p, humidity=rh, timestamp=curr_time, compute_shap=True
    )

    # Run Physics-Constrained Imputer
    if report.is_anomaly:
        imp = st.session_state.imputer.impute_point(t, p, rh, report.faulty_sensor, step)
        st.session_state.health_monitor.record_observation(t, p, rh, report.faulty_sensor)
    else:
        st.session_state.imputer.update_clean_history(t, p, rh, step)
        imp = {
            "imputed_temperature": t,
            "imputed_pressure": p,
            "imputed_humidity": rh,
            "was_imputed": False
        }
        st.session_state.health_monitor.record_observation(t, p, rh, None)

    record = {
        "step": step,
        "timestamp": curr_time,
        "temperature": t, "pressure": p, "humidity": rh,
        "imp_temp": imp["imputed_temperature"], "imp_press": imp["imputed_pressure"], "imp_rh": imp["imputed_humidity"],
        "is_anomaly": report.is_anomaly,
        "is_weather_event": report.is_weather_event,
        "anomaly_type": report.anomaly_type,
        "confidence": report.confidence,
        "faulty_sensor": report.faulty_sensor,
        "explanation": report.explanation,
        "thermo": report.thermodynamics,
        "top_features": report.top_features
    }

    st.session_state.history.append(record)
    if len(st.session_state.history) > 120:
        st.session_state.history.pop(0)

    if report.is_anomaly or report.is_weather_event:
        st.session_state.anomaly_log.insert(0, record)
        if len(st.session_state.anomaly_log) > 50:
            st.session_state.anomaly_log.pop()


# Execute step if requested
if st.session_state.step_once:
    process_next_step()

latest = st.session_state.history[-1]
prev = st.session_state.history[-2] if len(st.session_state.history) >= 2 else latest

# Header Banner
st.markdown('<div class="main-header">🌦️ Intelligent AWS Anomaly Detection & Diagnostic System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Physics-Guided Hybrid AI • World Meteorological Organization (WMO-No. 8) Standards • TinyML Edge Ready</div>', unsafe_allow_html=True)

# Status pill
if latest["is_anomaly"]:
    status_html = f'<span class="status-badge-fault">⚠️ SENSOR FAULT DETECTED: {latest["anomaly_type"]} ({str(latest["faulty_sensor"]).upper()})</span>'
elif latest["is_weather_event"]:
    status_html = f'<span class="status-badge-storm">⚡ GENUINE SEVERE WEATHER EVENT ACTIVE ({latest["anomaly_type"]})</span>'
else:
    status_html = '<span class="status-badge-ok">✅ ALL SENSORS NOMINAL • WMO QC PASSED</span>'

st.markdown(f"**Station Timestamp:** `{latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}` &nbsp;&nbsp;|&nbsp;&nbsp; {status_html}", unsafe_allow_html=True)
st.markdown("---")

# KPI Telemetry Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    d_t = latest["temperature"] - prev["temperature"] if not np.isnan(latest["temperature"]) else 0.0
    val_t = f"{latest['temperature']:.2f} °C" if not np.isnan(latest["temperature"]) else "MISSING (NaN)"
    st.metric(label="🌡️ Temperature", value=val_t, delta=f"{d_t:+.2f} °C/m" if not np.isnan(d_t) else None)

with col2:
    d_p = latest["pressure"] - prev["pressure"] if not np.isnan(latest["pressure"]) else 0.0
    val_p = f"{latest['pressure']:.2f} hPa" if not np.isnan(latest["pressure"]) else "MISSING"
    st.metric(label="🧭 Atmospheric Pressure", value=val_p, delta=f"{d_p:+.2f} hPa/m" if not np.isnan(d_p) else None)

with col3:
    d_rh = latest["humidity"] - prev["humidity"] if not np.isnan(latest["humidity"]) else 0.0
    val_rh = f"{latest['humidity']:.1f} %" if not np.isnan(latest["humidity"]) else "MISSING"
    st.metric(label="💧 Relative Humidity", value=val_rh, delta=f"{d_rh:+.1f} %/m" if not np.isnan(d_rh) else None)

with col4:
    td = latest["thermo"].get("dew_point_c", 0.0)
    dd = latest["thermo"].get("dew_point_depression_c", 0.0)
    st.metric(label="🌫️ Dew Point (Td)", value=f"{td:.1f} °C", delta=f"Depression: {dd:.1f} °C", delta_color="inverse")

with col5:
    st_rep = st.session_state.health_monitor.generate_station_report()
    health_color = "normal" if st_rep.overall_health >= 80 else ("off" if st_rep.overall_health >= 50 else "inverse")
    st.metric(label="🛡️ Station Health Index", value=f"{st_rep.overall_health:.1f}%", delta=st_rep.station_status)

st.markdown("")

# Dynamic Navigation Tabs
tab_telemetry, tab_xai, tab_health, tab_edge, tab_audit = st.tabs([
    "📈 Real-Time Telemetry & Detection",
    "🧠 Explainable AI (XAI) Diagnostics",
    "🛠️ Predictive Sensor Maintenance",
    "⚡ Edge AI (ESP32) TinyML Architecture",
    "📜 Incident Audit Log & Export"
])

# ----------------- TAB 1: TELEMETRY & PLOTS -----------------
with tab_telemetry:
    df_hist = pd.DataFrame(st.session_state.history)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Temperature (°C) — Raw vs Anomaly vs Imputed Reconstruction",
            "Atmospheric Pressure (hPa) — Barometric Tide & Tendency",
            "Relative Humidity (%) — Moisture Plausibility & Condensation"
        )
    )

    # 1. Temperature Subplot
    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['temperature'],
        mode='lines+markers', name='Raw Temperature',
        line=dict(color='#2563EB', width=2), marker=dict(size=4)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['imp_temp'],
        mode='lines', name='Imputed Temperature',
        line=dict(color='#10B981', width=2, dash='dash')
    ), row=1, col=1)

    # Anomaly markers for Temp
    anom_temp = df_hist[df_hist['is_anomaly'] & (df_hist['faulty_sensor'] == 'temperature')]
    if len(anom_temp) > 0:
        fig.add_trace(go.Scatter(
            x=anom_temp['timestamp'], y=anom_temp['temperature'],
            mode='markers', name='Temp Sensor Fault',
            marker=dict(color='#EF4444', size=11, symbol='x-thin', line=dict(width=3, color='#EF4444'))
        ), row=1, col=1)

    # Storm markers
    storms = df_hist[df_hist['is_weather_event']]
    if len(storms) > 0:
        fig.add_trace(go.Scatter(
            x=storms['timestamp'], y=storms['temperature'],
            mode='markers', name='Genuine Storm Event',
            marker=dict(color='#F59E0B', size=12, symbol='star')
        ), row=1, col=1)

    # 2. Pressure Subplot
    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['pressure'],
        mode='lines+markers', name='Raw Pressure',
        line=dict(color='#7C3AED', width=2), marker=dict(size=4)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['imp_press'],
        mode='lines', name='Imputed Pressure',
        line=dict(color='#10B981', width=2, dash='dash')
    ), row=2, col=1)

    anom_press = df_hist[df_hist['is_anomaly'] & (df_hist['faulty_sensor'] == 'pressure')]
    if len(anom_press) > 0:
        fig.add_trace(go.Scatter(
            x=anom_press['timestamp'], y=anom_press['pressure'],
            mode='markers', name='Barometer Fault',
            marker=dict(color='#EF4444', size=11, symbol='x-thin', line=dict(width=3, color='#EF4444'))
        ), row=2, col=1)

    # 3. Humidity Subplot
    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['humidity'],
        mode='lines+markers', name='Raw RH',
        line=dict(color='#06B6D4', width=2), marker=dict(size=4)
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['imp_rh'],
        mode='lines', name='Imputed RH',
        line=dict(color='#10B981', width=2, dash='dash')
    ), row=3, col=1)

    anom_rh = df_hist[df_hist['is_anomaly'] & (df_hist['faulty_sensor'] == 'humidity')]
    if len(anom_rh) > 0:
        fig.add_trace(go.Scatter(
            x=anom_rh['timestamp'], y=anom_rh['humidity'],
            mode='markers', name='Hygrometer Fault',
            marker=dict(color='#EF4444', size=11, symbol='x-thin', line=dict(width=3, color='#EF4444'))
        ), row=3, col=1)

    fig.update_layout(
        height=720,
        margin=dict(l=40, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------- TAB 2: EXPLAINABLE AI (XAI) -----------------
with tab_xai:
    st.markdown("### **Explainable AI (XAI) & Diagnostic Root-Cause Analysis**")
    st.caption("Inspect why a specific timestamp was flagged using TreeSHAP and atmospheric thermodynamic attribution.")

    if len(st.session_state.anomaly_log) == 0:
        st.info("No anomalies or severe events logged yet. Use the **Anomaly Injection Studio** in the sidebar to inject test faults!")
    else:
        log_options = [f"#{r['step']} | {r['timestamp'].strftime('%H:%M:%S')} | {r['anomaly_type']} | {r['faulty_sensor'] or 'Weather Event'}"
                       for r in st.session_state.anomaly_log]
        selected_idx = st.selectbox("Select Incident to Inspect:", range(len(log_options)), format_func=lambda i: log_options[i])
        target_event = st.session_state.anomaly_log[selected_idx]

        col_x1, col_x2 = st.columns([1, 1])

        with col_x1:
            st.markdown("#### **Diagnostic Summary**")
            st.markdown(f"**Classification:** `{target_event['anomaly_type']}`")
            st.markdown(f"**Faulty Component:** `{str(target_event['faulty_sensor']).upper()}`")
            st.markdown(f"**Model Confidence:** `{target_event['confidence']*100:.1f}%`")
            st.markdown(f"**Physical Observation:** T = {target_event['temperature']}°C, P = {target_event['pressure']} hPa, RH = {target_event['humidity']}%")

            st.info(f"💡 **AI Diagnostic Rationale:**\n\n{target_event['explanation']}")

            # Thermodynamic metrics
            st.markdown("##### **Thermodynamic State Analysis**")
            th = target_event["thermo"]
            st.write({
                "Saturation Vapor Pressure (es)": f"{th.get('sat_vapor_pressure_hpa', 0)} hPa",
                "Actual Vapor Pressure (e)": f"{th.get('vapor_pressure_hpa', 0)} hPa",
                "Dew Point (Td)": f"{th.get('dew_point_c', 0)} °C",
                "Dew Point Depression (T - Td)": f"{th.get('dew_point_depression_c', 0)} °C",
                "Vapor Pressure Deficit (VPD)": f"{th.get('vpd_hpa', 0)} hPa",
                "Potential Temperature (θ)": f"{th.get('potential_temp_c', 0)} °C"
            })

        with col_x2:
            st.markdown("#### **Feature Attribution (SHAP / Contribution Scores)**")
            feats = target_event.get("top_features", [])
            if feats:
                feat_names = [f[0] for f in feats]
                feat_scores = [abs(f[1]) for f in feats]

                fig_xai = go.Figure(go.Bar(
                    x=feat_scores[::-1],
                    y=feat_names[::-1],
                    orientation='h',
                    marker=dict(color='#3B82F6')
                ))
                fig_xai.update_layout(
                    title="Top Contributing Features to Decision",
                    xaxis_title="Attribution Magnitude",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_xai, use_container_width=True)
            else:
                st.write("Attribution scores within nominal baseline limits.")


# ----------------- TAB 3: SENSOR HEALTH & MAINTENANCE -----------------
with tab_health:
    st.markdown("### **Predictive Maintenance & Sensor Degradation Monitor**")
    st.caption("Continuously monitors signal-to-noise ratio, baseline calibration drift, and recurring faults.")

    st_report = st.session_state.health_monitor.generate_station_report()

    h_col1, h_col2, h_col3 = st.columns(3)

    for col, (s_name, s_stat) in zip([h_col1, h_col2, h_col3], st_report.sensors.items()):
        with col:
            st.markdown(f"#### **{s_name.capitalize()} Sensor**")
            # Health gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=s_stat.health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10B981" if s_stat.health_score >= 80 else ("#F59E0B" if s_stat.health_score >= 50 else "#EF4444")},
                    'steps': [
                        {'range': [0, 50], 'color': "#FEE2E2"},
                        {'range': [50, 80], 'color': "#FEF3C7"},
                        {'range': [80, 100], 'color': "#D1FAE5"}
                    ]
                }
            ))
            fig_g.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_g, use_container_width=True)

            st.markdown(f"**Status:** `{s_stat.status}`")
            st.markdown(f"**Noise Floor:** `{s_stat.noise_level}`")
            st.markdown(f"**Drift Rate:** `{s_stat.drift_rate_per_day:+.2f} unit/day`")
            st.info(f"🔧 **Recommendation:**\n\n{s_stat.recommendation}")


# ----------------- TAB 4: EDGE TINYML ARCHITECTURE -----------------
with tab_edge:
    st.markdown("### **Edge AI (ESP32) Microcontroller Deployment**")
    st.markdown("""
    To meet the SIH 2026 requirement for **Edge AI deployment on low-power ESP32 nodes**, our architecture
    features an ultra-lightweight, zero-external-dependency C/C++ engine (`esp32_anomaly_detector.h`).
    """)

    col_e1, col_e2 = st.columns([1, 1])

    with col_e1:
        st.markdown("#### **Embedded Hardware Specifications**")
        st.table(pd.DataFrame({
            "Metric": ["Target MCU", "Clock Speed", "Execution Latency", "RAM Footprint", "Flash Storage", "Power Consumption"],
            "Specification": ["ESP32 / ESP32-S3", "240 MHz (Xtensa Dual-Core)", "< 0.08 ms / sample", "1.4 KB", "8.2 KB", "< 80 mA (active)"]
        }))

        st.markdown("#### **Key Architectural Benefits for Remote AWS**")
        st.markdown("""
        1. **95% Telemetry Bandwidth Reduction**: Only sends anomaly notifications or imputed deltas over LoRaWAN/Satellite.
        2. **Sub-millisecond Latency**: Local screening prevents false trigger cascades.
        3. **Local Fail-safe Actuation**: Can power-cycle frozen I2C sensor buses autonomously.
        """)

    with col_e2:
        st.markdown("#### **Embedded C Implementation (`esp32_anomaly_detector.h`)**")
        try:
            with open("edge/esp32_anomaly_detector.h", "r") as f:
                code_snippet = f.read()
            st.code(code_snippet[:1500] + "\n\n// ... [Full C implementation in edge/esp32_anomaly_detector.h]", language="c")
        except Exception:
            st.code("// edge/esp32_anomaly_detector.h", language="c")


# ----------------- TAB 5: INCIDENT AUDIT LOG & EXPORT -----------------
with tab_audit:
    st.markdown("### **Incident Audit Log & Telemetry Export**")

    if len(st.session_state.anomaly_log) > 0:
        log_df = pd.DataFrame(st.session_state.anomaly_log)
        display_cols = ['step', 'timestamp', 'anomaly_type', 'faulty_sensor', 'confidence', 'temperature', 'pressure', 'humidity', 'explanation']
        st.dataframe(log_df[display_cols], use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_anom = log_df[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Incident Audit Report (CSV)", data=csv_anom, file_name="aws_incident_audit.csv", mime="text/csv")
        with col_d2:
            full_df = pd.DataFrame(st.session_state.history)
            csv_full = full_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Imputed Dataset (CSV)", data=csv_full, file_name="aws_imputed_telemetry.csv", mime="text/csv")
    else:
        st.info("No incidents recorded yet. Nominal observations are streaming.")


# Auto-refresh loop if in streaming mode
if st.session_state.is_streaming:
    time.sleep(speed)
    process_next_step()
    st.rerun()
