"""
SkyGuard — Real-time AWS Anomaly Detection Console (SIH 2026, PS73)
Streamlit front end for the 5-tier hybrid AI detection engine.

This file only changes presentation. Detection, imputation, and health-scoring
logic is called through src.detector / src.imputer / src.health_monitor exactly
as before — nothing in src/ needs to change for this UI.
"""

import time
import os
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

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="SkyGuard — AWS Anomaly Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# THEME — dark control-room palette
# =============================================================================
COLOR = {
    "bg": "#0B1120",
    "panel": "#161E2E",
    "panel_alt": "#1C2538",
    "border": "#26324A",
    "text": "#E7ECF3",
    "text_dim": "#8592A6",
    "normal": "#2DD4BF",
    "storm": "#F5A524",
    "fault": "#F0616B",
    "raw_line": "#5B8DEF",
    "imputed_line": "#2DD4BF",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {COLOR["bg"]} !important;
    color: {COLOR["text"]};
    font-family: 'IBM Plex Sans', sans-serif;
}}
[data-testid="stHeader"] {{ background-color: transparent !important; }}
[data-testid="stSidebar"] {{
    background-color: {COLOR["panel"]} !important;
    border-right: 1px solid {COLOR["border"]};
}}
[data-testid="stSidebar"] * {{ color: {COLOR["text"]}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1300px; }}

h1, h2, h3, h4, h5, p, span, label, div {{ color: {COLOR["text"]}; }}

/* Buttons */
.stButton > button {{
    background-color: {COLOR["panel_alt"]};
    color: {COLOR["text"]};
    border: 1px solid {COLOR["border"]};
    border-radius: 6px;
    font-weight: 500;
}}
.stButton > button:hover {{
    border-color: {COLOR["raw_line"]};
    color: {COLOR["raw_line"]};
}}
.stButton > button[kind="primary"] {{
    background-color: {COLOR["raw_line"]};
    border-color: {COLOR["raw_line"]};
    color: #0B1120;
    font-weight: 600;
}}

/* Inputs */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
.stSlider, [data-testid="stFileUploader"] {{
    background-color: {COLOR["panel_alt"]} !important;
    border-color: {COLOR["border"]} !important;
    color: {COLOR["text"]} !important;
}}

/* Expanders */
[data-testid="stExpander"] {{
    background-color: {COLOR["panel"]};
    border: 1px solid {COLOR["border"]};
    border-radius: 10px;
}}
[data-testid="stExpander"] summary {{
    font-weight: 600;
    color: {COLOR["text"]};
}}

/* Dataframe */
[data-testid="stDataFrame"] {{ border: 1px solid {COLOR["border"]}; border-radius: 8px; }}

hr {{ border-color: {COLOR["border"]}; }}

/* ---- SkyGuard custom components ---- */
.sg-brand {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: {COLOR["text"]};
    margin-bottom: 0;
}}
.sg-brand-by {{ color: {COLOR["text_dim"]}; font-size: 0.78rem; margin-top: -4px; }}

.sg-topbar {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 1.1rem;
}}
.sg-topbar-title {{ font-size: 1.5rem; font-weight: 700; margin: 0; }}
.sg-topbar-sub {{ color: {COLOR["text_dim"]}; font-size: 0.92rem; }}
.sg-topbar-meta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: {COLOR["text_dim"]};
    text-align: right;
}}
.sg-live-dot {{
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 6px; position: relative; top: -1px;
}}

/* Hero card */
.sg-hero {{
    border-radius: 12px;
    padding: 26px 30px;
    border: 1px solid {COLOR["border"]};
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 24px;
}}
.sg-hero--normal {{ background: linear-gradient(135deg, rgba(45,212,191,0.10), {COLOR["panel"]} 60%); border-color: rgba(45,212,191,0.35); }}
.sg-hero--storm  {{ background: linear-gradient(135deg, rgba(245,165,36,0.12), {COLOR["panel"]} 60%); border-color: rgba(245,165,36,0.4); }}
.sg-hero--fault  {{ background: linear-gradient(135deg, rgba(240,97,107,0.12), {COLOR["panel"]} 60%); border-color: rgba(240,97,107,0.4); }}

.sg-hero-status {{ font-size: 2.5rem; font-weight: 700; line-height: 1.05; margin: 0; }}
.sg-hero-status--normal {{ color: {COLOR["normal"]}; }}
.sg-hero-status--storm {{ color: {COLOR["storm"]}; }}
.sg-hero-status--fault {{ color: {COLOR["fault"]}; }}
.sg-hero-sub {{ color: {COLOR["text_dim"]}; font-size: 0.9rem; margin-top: 4px; }}
.sg-hero-time {{ font-family: 'IBM Plex Mono', monospace; color: {COLOR["text_dim"]}; font-size: 0.85rem; margin-top: 10px; }}

.sg-reading-grid {{ display: flex; gap: 34px; }}
.sg-reading {{ text-align: right; }}
.sg-reading-val {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; }}
.sg-reading-label {{ color: {COLOR["text_dim"]}; font-size: 0.78rem; margin-top: 2px; }}

.sg-hero-note {{
    flex-basis: 100%;
    border-top: 1px solid {COLOR["border"]};
    padding-top: 12px;
    margin-top: 4px;
    color: {COLOR["text_dim"]};
    font-size: 0.88rem;
}}

/* Previous readings strip */
.sg-strip {{ display: flex; gap: 10px; overflow-x: auto; padding: 14px 2px 4px 2px; }}
.sg-strip-item {{
    flex: 0 0 auto;
    background-color: {COLOR["panel"]};
    border: 1px solid {COLOR["border"]};
    border-radius: 8px;
    padding: 7px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: {COLOR["text_dim"]};
    text-align: center;
    min-width: 74px;
}}
.sg-strip-time {{ display: block; color: {COLOR["text"]}; font-weight: 500; }}
.sg-strip-type {{ display: block; margin-top: 2px; }}

/* Sensor health badges */
.sg-health-row {{ display: flex; gap: 12px; margin: 6px 0 2px 0; }}
.sg-health-chip {{
    flex: 1;
    background-color: {COLOR["panel"]};
    border: 1px solid {COLOR["border"]};
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.sg-health-name {{ font-size: 0.85rem; color: {COLOR["text_dim"]}; }}
.sg-health-val {{ font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1rem; }}
.sg-health-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; }}

.sg-panel-title {{ font-size: 1rem; font-weight: 600; margin-bottom: 2px; }}
.sg-panel-sub {{ color: {COLOR["text_dim"]}; font-size: 0.82rem; margin-bottom: 10px; }}
</style>
""", unsafe_allow_html=True)


def status_class(is_anomaly: bool, is_weather_event: bool) -> str:
    if is_anomaly:
        return "fault"
    if is_weather_event:
        return "storm"
    return "normal"


def status_label(record: dict) -> str:
    if record["is_anomaly"]:
        friendly = {
            "SPIKE": "Sensor spike",
            "STUCK_SENSOR": "Sensor stuck",
            "SENSOR_DRIFT": "Sensor drift",
            "OUT_OF_BOUNDS": "Out of bounds",
            "PHYSICAL_INCONSISTENCY": "Physical inconsistency",
            "MISSING": "Missing data",
        }
        return friendly.get(record["anomaly_type"], "Anomaly detected")
    if record["is_weather_event"]:
        return "Genuine weather event"
    return "Normal"


HEALTH_COLOR = lambda score: COLOR["normal"] if score >= 75 else (COLOR["storm"] if score >= 50 else COLOR["fault"])


# =============================================================================
# SESSION STATE INITIALIZATION (unchanged from original logic)
# =============================================================================
if 'initialized' not in st.session_state:
    st.session_state.simulator = AWSDataSimulator(seed=101)
    st.session_state.detector = AWSAnomalyDetector(contamination=0.03)
    st.session_state.imputer = AWSImputer()
    st.session_state.health_monitor = SensorHealthMonitor()

    clean_hist = st.session_state.simulator.generate_historical_dataset(days=2, interval_minutes=5, inject_anomalies=False)
    st.session_state.detector.fit(clean_hist)

    st.session_state.history = []
    st.session_state.anomaly_log = []
    st.session_state.stream_step = 0
    st.session_state.is_streaming = False
    st.session_state.active_fault = None
    st.session_state.active_fault_param = None
    st.session_state.active_fault_value = 0.0

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


# =============================================================================
# SIDEBAR — station + stream controls + injection studio
# =============================================================================
st.sidebar.markdown('<p class="sg-brand">SKYGUARD</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sg-brand-by">by Vyoma &middot; SIH 2026 PS73</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")

station = st.sidebar.selectbox("Active station", [
    "AWS-IN-DELHI-04 (Safdarjung)",
    "AWS-IN-BENGALURU-02 (GKVK)",
    "AWS-IN-SHILLONG-01 (Barapani)",
    "AWS-IN-MUMBAI-05 (Santacruz)"
])

st.sidebar.markdown("#### Stream control")
col_s1, col_s2 = st.sidebar.columns(2)
if col_s1.button("Step +1m", use_container_width=True):
    st.session_state.step_once = True
else:
    st.session_state.step_once = False

stream_toggle = col_s2.button("Start / stop", use_container_width=True)
if stream_toggle:
    st.session_state.is_streaming = not st.session_state.is_streaming

speed = st.sidebar.slider("Stream interval (sec)", 0.2, 2.0, 0.6, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Anomaly injection")
st.sidebar.caption("Force a scenario to test fault-vs-weather disentangling.")

inject_choice = st.sidebar.selectbox("Scenario", [
    "None (nominal diurnal stream)",
    "Temperature spike (+12.5°C)",
    "Barometer pressure drop spike (-16.0 hPa)",
    "Humidity sensor spike (+40.0%)",
    "Stuck / frozen sensor (zero variance)",
    "Sensor calibration drift (+0.04°C/min)",
    "Genuine severe convective storm (downburst)",
    "Out of physical bounds (RH = 125%)",
    "Packet loss / missing data (NaN)"
])

stuck_sensor_choice = "temperature"
drift_sensor_choice = "temperature"
if "Stuck" in inject_choice:
    stuck_sensor_choice = st.sidebar.selectbox("Sensor to freeze", ["temperature", "pressure", "humidity"], format_func=lambda x: f"Freeze {x.capitalize()}")
elif "drift" in inject_choice:
    drift_sensor_choice = st.sidebar.selectbox("Sensor to drift", ["temperature", "humidity", "pressure"], format_func=lambda x: f"Drift {x.capitalize()}")

if st.sidebar.button("Trigger injection", use_container_width=True):
    if "None" in inject_choice:
        st.session_state.active_fault = None
        st.sidebar.success("Cleared all injections.")
    elif "Temperature spike" in inject_choice:
        st.session_state.active_fault = "SPIKE"
        st.session_state.active_fault_param = "temperature"
        st.session_state.active_fault_value = 12.5
        st.sidebar.warning("Injected: +12.5°C temperature spike")
    elif "Barometer pressure drop" in inject_choice:
        st.session_state.active_fault = "SPIKE"
        st.session_state.active_fault_param = "pressure"
        st.session_state.active_fault_value = -16.0
        st.sidebar.warning("Injected: -16.0 hPa barometer spike")
    elif "Humidity sensor spike" in inject_choice:
        st.session_state.active_fault = "SPIKE"
        st.session_state.active_fault_param = "humidity"
        st.session_state.active_fault_value = 40.0
        st.sidebar.warning("Injected: +40% humidity spike")
    elif "Stuck" in inject_choice:
        st.session_state.active_fault = "STUCK_SENSOR"
        st.session_state.active_fault_param = stuck_sensor_choice
        latest_hist = st.session_state.history[-1] if st.session_state.history else None
        if latest_hist and stuck_sensor_choice in latest_hist:
            st.session_state.active_fault_value = float(latest_hist[stuck_sensor_choice])
        else:
            st.session_state.active_fault_value = 25.4 if stuck_sensor_choice == "temperature" else (1013.2 if stuck_sensor_choice == "pressure" else 62.0)
        st.sidebar.warning(f"Injected: frozen {stuck_sensor_choice} sensor (stuck at {st.session_state.active_fault_value:.2f})")
    elif "drift" in inject_choice:
        st.session_state.active_fault = "SENSOR_DRIFT"
        st.session_state.active_fault_param = drift_sensor_choice
        st.session_state.active_fault_value = 0.0
        st.sidebar.warning(f"Injected: progressive {drift_sensor_choice} sensor drift")
    elif "Genuine severe convective storm" in inject_choice:
        st.session_state.active_fault = "GENUINE_WEATHER_EVENT"
        st.sidebar.info("Triggered: genuine severe convective storm")
    elif "Out of physical bounds" in inject_choice:
        st.session_state.active_fault = "OUT_OF_BOUNDS"
        st.session_state.active_fault_param = "humidity"
        st.session_state.active_fault_value = 125.0
        st.sidebar.error("Injected: out-of-bounds RH (125%)")
    elif "Packet loss" in inject_choice:
        st.session_state.active_fault = "MISSING"
        st.session_state.active_fault_param = "temperature"
        st.sidebar.error("Injected: missing data packet (NaN)")


# =============================================================================
# STREAM STEP LOGIC (unchanged from original logic)
# =============================================================================
def process_next_step():
    st.session_state.stream_step += 1
    step = st.session_state.stream_step
    curr_time = st.session_state.history[-1]["timestamp"] + pd.Timedelta(minutes=1)
    t, p, rh = st.session_state.simulator.generate_point(curr_time)

    fault = st.session_state.active_fault
    param = st.session_state.active_fault_param

    if fault == "SPIKE":
        if param == "temperature":
            t += st.session_state.active_fault_value
        elif param == "pressure":
            p += st.session_state.active_fault_value
        elif param == "humidity":
            rh = min(100.0, rh + st.session_state.active_fault_value)
        st.session_state.active_fault = None

    elif fault == "STUCK_SENSOR":
        if st.session_state.active_fault_value is None:
            st.session_state.active_fault_value = t if param == "temperature" else (p if param == "pressure" else rh)
        if param == "temperature":
            t = st.session_state.active_fault_value
        elif param == "pressure":
            p = st.session_state.active_fault_value
        elif param == "humidity":
            rh = st.session_state.active_fault_value

    elif fault == "SENSOR_DRIFT":
        if param == "temperature":
            st.session_state.active_fault_value += 0.04
            t += st.session_state.active_fault_value
        elif param == "humidity":
            st.session_state.active_fault_value += 0.10
            rh = min(100.0, max(0.0, rh + st.session_state.active_fault_value))
        elif param == "pressure":
            st.session_state.active_fault_value += 0.03
            p += st.session_state.active_fault_value

    elif fault == "GENUINE_WEATHER_EVENT":
        t -= 6.2
        p -= 2.6
        rh = 97.5
        st.session_state.active_fault = None

    elif fault == "OUT_OF_BOUNDS":
        if param == "humidity":
            rh = st.session_state.active_fault_value
        elif param == "temperature":
            t = 78.0
        st.session_state.active_fault = None

    elif fault == "MISSING":
        t = np.nan
        st.session_state.active_fault = None

    report: AnomalyReport = st.session_state.detector.process_observation(
        temperature=t, pressure=p, humidity=rh, timestamp=curr_time, compute_shap=True
    )

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


if st.session_state.step_once:
    process_next_step()

latest = st.session_state.history[-1]
prev_readings = st.session_state.history[-16:-1][::-1]  # most recent first, excluding latest


# =============================================================================
# TOP BAR
# =============================================================================
live_color = COLOR["normal"] if st.session_state.is_streaming else COLOR["text_dim"]
live_text = f"Streaming &middot; {speed:.1f}s interval" if st.session_state.is_streaming else "Paused"

top_l, top_r = st.columns([3, 2])
with top_l:
    st.markdown('<p class="sg-topbar-title">SkyGuard</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sg-topbar-sub">{station.split(" (")[0]} &middot; live anomaly console</p>', unsafe_allow_html=True)
with top_r:
    st.markdown(
        f'<div class="sg-topbar-meta">'
        f'<span class="sg-live-dot" style="background-color:{live_color}"></span>{live_text}<br>'
        f'step #{latest["step"]} &middot; confidence {latest["confidence"]*100:.0f}%'
        f'</div>', unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# HERO CARD — current reading & status
# =============================================================================
s_class = status_class(latest["is_anomaly"], latest["is_weather_event"])
s_label = status_label(latest)

hero_note = ""
if latest["is_anomaly"]:
    hero_note = f'<div class="sg-hero-note">{latest["explanation"]} &middot; faulty component: {str(latest["faulty_sensor"]).upper()}</div>'
elif latest["is_weather_event"]:
    hero_note = f'<div class="sg-hero-note">{latest["explanation"]}</div>'

st.markdown(f"""
<div class="sg-hero sg-hero--{s_class}">
    <div>
        <p class="sg-hero-status sg-hero-status--{s_class}">{s_label}</p>
        <p class="sg-hero-sub">{station.split(" (")[0]}</p>
        <p class="sg-hero-time">{latest["timestamp"].strftime("%Y-%m-%d %H:%M")}</p>
    </div>
    <div class="sg-reading-grid">
        <div class="sg-reading">
            <div class="sg-reading-val">{latest["temperature"]:.1f}&deg;C</div>
            <div class="sg-reading-label">Temperature</div>
        </div>
        <div class="sg-reading">
            <div class="sg-reading-val">{latest["pressure"]:.1f}</div>
            <div class="sg-reading-label">Pressure (hPa)</div>
        </div>
        <div class="sg-reading">
            <div class="sg-reading-val">{latest["humidity"]:.1f}%</div>
            <div class="sg-reading-label">Humidity</div>
        </div>
    </div>
    {hero_note}
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PREVIOUS READINGS STRIP — neutral, fading with age
# =============================================================================
strip_html = '<div class="sg-strip">'
n = len(prev_readings)
for idx, r in enumerate(prev_readings):
    opacity = max(0.28, 1.0 - (idx / max(n, 1)) * 0.75)
    label = status_label(r)
    short = {"Normal": "Normal", "Genuine weather event": "Storm"}.get(label, "Fault")
    strip_html += (
        f'<div class="sg-strip-item" style="opacity:{opacity:.2f}">'
        f'<span class="sg-strip-time">{r["timestamp"].strftime("%H:%M")}</span>'
        f'<span class="sg-strip-type">{short}</span>'
        f'</div>'
    )
strip_html += '</div>'
st.markdown(strip_html, unsafe_allow_html=True)


# =============================================================================
# SENSOR HEALTH — persistent badge row
# =============================================================================
st_report = st.session_state.health_monitor.generate_station_report()
health_html = '<div class="sg-health-row">'
for s_name, s_stat in st_report.sensors.items():
    c = HEALTH_COLOR(s_stat.health_score)
    health_html += (
        f'<div class="sg-health-chip">'
        f'<span class="sg-health-name"><span class="sg-health-dot" style="background-color:{c}"></span>{s_name.capitalize()}</span>'
        f'<span class="sg-health-val" style="color:{c}">{s_stat.health_score:.0f}%</span>'
        f'</div>'
    )
health_html += '</div>'
st.markdown(health_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# TWO-PANEL LAYOUT — telemetry graphs · root cause / XAI
# =============================================================================
panel_l, panel_r = st.columns([3, 2])

with panel_l:
    st.markdown('<p class="sg-panel-title">Telemetry &amp; detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sg-panel-sub">Raw signal, imputed reconstruction, and flagged faults per parameter.</p>', unsafe_allow_html=True)

    with st.expander("Temperature", expanded=False):
        df_hist = pd.DataFrame(st.session_state.history)
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['temperature'], mode='lines', name='Raw', line=dict(color=COLOR["raw_line"], width=2)))
        fig_t.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['imp_temp'], mode='lines', name='Imputed', line=dict(color=COLOR["imputed_line"], width=2, dash='dash')))
        anom_temp = df_hist[df_hist['is_anomaly'] & (df_hist['faulty_sensor'] == 'temperature')]
        if len(anom_temp) > 0:
            fig_t.add_trace(go.Scatter(x=anom_temp['timestamp'], y=anom_temp['temperature'], mode='markers', name='Fault', marker=dict(color=COLOR["fault"], size=10, symbol='x-thin', line=dict(width=3, color=COLOR["fault"]))))
        storms = df_hist[df_hist['is_weather_event']]
        if len(storms) > 0:
            fig_t.add_trace(go.Scatter(x=storms['timestamp'], y=storms['temperature'], mode='markers', name='Storm', marker=dict(color=COLOR["storm"], size=11, symbol='star')))
        fig_t.update_layout(height=280, margin=dict(l=30, r=10, t=10, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLOR["text_dim"]), legend=dict(orientation="h", y=1.1), xaxis=dict(gridcolor=COLOR["border"]), yaxis=dict(gridcolor=COLOR["border"]))
        st.plotly_chart(fig_t, use_container_width=True)

    with st.expander("Pressure", expanded=False):
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['pressure'], mode='lines', name='Raw', line=dict(color=COLOR["raw_line"], width=2)))
        fig_p.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['imp_press'], mode='lines', name='Imputed', line=dict(color=COLOR["imputed_line"], width=2, dash='dash')))
        anom_press = df_hist[df_hist['is_anomaly'] & (df_hist['faulty_sensor'] == 'pressure')]
        if len(anom_press) > 0:
            fig_p.add_trace(go.Scatter(x=anom_press['timestamp'], y=anom_press['pressure'], mode='markers', name='Fault', marker=dict(color=COLOR["fault"], size=10, symbol='x-thin', line=dict(width=3, color=COLOR["fault"]))))
        fig_p.update_layout(height=280, margin=dict(l=30, r=10, t=10, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLOR["text_dim"]), legend=dict(orientation="h", y=1.1), xaxis=dict(gridcolor=COLOR["border"]), yaxis=dict(gridcolor=COLOR["border"]))
        st.plotly_chart(fig_p, use_container_width=True)

    with st.expander("Humidity", expanded=False):
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['humidity'], mode='lines', name='Raw', line=dict(color=COLOR["raw_line"], width=2)))
        fig_h.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['imp_rh'], mode='lines', name='Imputed', line=dict(color=COLOR["imputed_line"], width=2, dash='dash')))
        anom_rh = df_hist[df_hist['is_anomaly'] & (df_hist['faulty_sensor'] == 'humidity')]
        if len(anom_rh) > 0:
            fig_h.add_trace(go.Scatter(x=anom_rh['timestamp'], y=anom_rh['humidity'], mode='markers', name='Fault', marker=dict(color=COLOR["fault"], size=10, symbol='x-thin', line=dict(width=3, color=COLOR["fault"]))))
        fig_h.update_layout(height=280, margin=dict(l=30, r=10, t=10, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLOR["text_dim"]), legend=dict(orientation="h", y=1.1), xaxis=dict(gridcolor=COLOR["border"]), yaxis=dict(gridcolor=COLOR["border"]))
        st.plotly_chart(fig_h, use_container_width=True)

with panel_r:
    st.markdown('<p class="sg-panel-title">Root cause &amp; explainability</p>', unsafe_allow_html=True)
    st.markdown('<p class="sg-panel-sub">Diagnosis for the most recent reading, or step back through prior incidents.</p>', unsafe_allow_html=True)

    if len(st.session_state.anomaly_log) == 0:
        st.info("No incidents logged yet. The stream is nominal.")
    else:
        log_options = [f"#{r['step']} {r['timestamp'].strftime('%H:%M:%S')} — {status_label(r)}" for r in st.session_state.anomaly_log]
        selected_idx = st.selectbox("Incident", range(len(log_options)), format_func=lambda i: log_options[i])
        target_event = st.session_state.anomaly_log[selected_idx]

        t_class = status_class(target_event["is_anomaly"], target_event["is_weather_event"])
        t_color = COLOR[t_class]

        st.markdown(f"""
        <div style="border-left:3px solid {t_color}; padding-left:12px; margin-bottom:12px;">
            <div style="font-weight:600; color:{t_color};">{status_label(target_event)}</div>
            <div style="color:{COLOR['text_dim']}; font-size:0.85rem;">
                Faulty component: {str(target_event['faulty_sensor']).upper() if target_event['faulty_sensor'] else '—'}
                &middot; Confidence: {target_event['confidence']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.9rem; color:{COLOR['text_dim']}'>{target_event['explanation']}</p>", unsafe_allow_html=True)

        feats = target_event.get("top_features", [])
        if feats:
            fig_xai = go.Figure(go.Bar(
                x=[abs(f[1]) for f in feats][::-1],
                y=[f[0] for f in feats][::-1],
                orientation='h', marker=dict(color=COLOR["raw_line"])
            ))
            fig_xai.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLOR["text_dim"]),
                xaxis=dict(gridcolor=COLOR["border"]), yaxis=dict(gridcolor=COLOR["border"])
            )
            st.plotly_chart(fig_xai, use_container_width=True)


# =============================================================================
# SECONDARY DETAIL — sensor health, edge deployment, audit log
# =============================================================================
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Sensor health detail"):
    h_col1, h_col2, h_col3 = st.columns(3)
    for col, (s_name, s_stat) in zip([h_col1, h_col2, h_col3], st_report.sensors.items()):
        with col:
            c = HEALTH_COLOR(s_stat.health_score)
            st.markdown(f"**{s_name.capitalize()} sensor**")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=s_stat.health_score,
                number={'font': {'color': COLOR["text"]}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': COLOR["text_dim"]}, 'bar': {'color': c}, 'bgcolor': COLOR["panel_alt"], 'bordercolor': COLOR["border"]}
            ))
            fig_g.update_layout(height=180, margin=dict(l=20, r=20, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)', font=dict(color=COLOR["text_dim"]))
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(f"Status: {s_stat.status}")
            st.markdown(f"<span style='font-size:0.85rem; color:{COLOR['text_dim']}'>{s_stat.recommendation}</span>", unsafe_allow_html=True)

with st.expander("Edge AI deployment (ESP32)"):
    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        st.table(pd.DataFrame({
            "Metric": ["Target MCU", "Clock speed", "Execution latency", "RAM footprint", "Bandwidth saving"],
            "Specification": ["ESP32 / ESP32-S3", "240 MHz Xtensa", "< 0.08 ms / sample", "1.4 KB", "Up to 95% reduction"]
        }))
    with col_e2:
        try:
            with open("edge/esp32_anomaly_detector.h", "r") as f:
                st.code(f.read()[:1500] + "\n\n// ... full source in edge/esp32_anomaly_detector.h", language="c")
        except Exception:
            st.code("// edge/esp32_anomaly_detector.h", language="c")

with st.expander("Incident audit log & export"):
    if len(st.session_state.anomaly_log) > 0:
        log_df = pd.DataFrame(st.session_state.anomaly_log)
        disp_cols = ['step', 'timestamp', 'anomaly_type', 'faulty_sensor', 'confidence', 'temperature', 'pressure', 'humidity', 'explanation']
        st.dataframe(log_df[disp_cols], use_container_width=True)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("Download incident audit report (CSV)", data=log_df[disp_cols].to_csv(index=False).encode('utf-8'), file_name="aws_incident_audit.csv", mime="text/csv")
        with col_d2:
            full_df = pd.DataFrame(st.session_state.history)
            st.download_button("Download full imputed dataset (CSV)", data=full_df.to_csv(index=False).encode('utf-8'), file_name="aws_imputed_telemetry.csv", mime="text/csv")
    else:
        st.info("No incidents recorded yet.")


# =============================================================================
# AUTO-REFRESH LOOP
# =============================================================================
if st.session_state.is_streaming:
    time.sleep(speed)
    process_next_step()
    st.rerun()