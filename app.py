"""
Interactive Streamlit Dashboard for SIH 2026 Automatic Weather Station Anomaly Detection.
Real-Time Telemetry Streaming Simulation with On-the-Fly Anomaly Injection Studio.
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

# Page configuration (No emojis)
st.set_page_config(
    page_title="SkyGuard by Vyoma — AWS Anomaly Detection & Diagnostics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling (High-contrast, professional, zero blue-on-black)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0B1120 !important;
        color: #E2E8F0 !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }

    /* Top Brand Headers */
    .brand-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #FFFFFF !important;
        margin-bottom: 0.15rem;
    }
    .brand-subtitle {
        font-size: 0.88rem;
        color: #94A3B8 !important;
        margin-bottom: 1.2rem;
        line-height: 1.4;
    }

    /* Sidebar Section Headers */
    .sidebar-section-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8 !important;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }

    /* Enterprise Buttons */
    .stButton > button {
        background-color: #1F2937 !important;
        color: #F8FAFC !important;
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #374151 !important;
        border-color: #64748B !important;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        border-color: #3B82F6 !important;
        color: #FFFFFF !important;
    }

    /* Selectbox and Inputs */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #1F2937 !important;
        border-color: #374151 !important;
        color: #F8FAFC !important;
    }

    /* =========================================================================
       HERO BOX FOR RECENT READING (Dynamic Tinting ONLY for Hero Box)
       ========================================================================= */
    .hero-container {
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 22px;
        transition: all 0.25s ease;
    }

    /* RED TINT FOR ANOMALY */
    .hero-tint-anomaly {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(239, 68, 68, 0.03) 100%);
        border: 1.5px solid rgba(239, 68, 68, 0.55);
        box-shadow: 0 4px 24px -2px rgba(239, 68, 68, 0.18);
    }
    /* ORANGE/AMBER TINT FOR GENUINE WEATHER EVENT */
    .hero-tint-weather {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.13) 0%, rgba(245, 158, 11, 0.03) 100%);
        border: 1.5px solid rgba(245, 158, 11, 0.55);
        box-shadow: 0 4px 24px -2px rgba(245, 158, 11, 0.18);
    }
    /* BLUE TINT FOR NORMAL */
    .hero-tint-normal {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.10) 0%, rgba(59, 130, 246, 0.03) 100%);
        border: 1.5px solid rgba(59, 130, 246, 0.45);
        box-shadow: 0 4px 24px -2px rgba(59, 130, 246, 0.18);
    }

    .hero-top-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 14px;
        margin-bottom: 18px;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero-status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .hero-pill-anomaly {
        background-color: rgba(239, 68, 68, 0.22);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.50);
    }
    .hero-pill-weather {
        background-color: rgba(245, 158, 11, 0.22);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.50);
    }
    .hero-pill-normal {
        background-color: rgba(59, 130, 246, 0.20);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.45);
    }

    .hero-status-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        line-height: 1.2;
    }
    .hero-status-desc {
        font-size: 0.84rem;
        color: #94A3B8;
        margin-top: 3px;
    }

    .hero-meta-panel {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #94A3B8;
        text-align: right;
        line-height: 1.5;
    }
    .hero-meta-accent {
        color: #E2E8F0;
        font-weight: 600;
    }

    .hero-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-bottom: 16px;
    }

    .hero-metric-tile {
        background: rgba(17, 24, 39, 0.60);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 14px;
    }

    .hero-metric-caption {
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 4px;
    }

    .hero-metric-number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.45rem;
        font-weight: 600;
        color: #FFFFFF;
        line-height: 1.1;
    }

    .hero-metric-tendency {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        margin-top: 4px;
        color: #94A3B8;
    }

    .hero-footer-row {
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.84rem;
        color: #CBD5E1;
        line-height: 1.45;
    }
    .hero-footer-label {
        font-weight: 600;
        color: #F8FAFC;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #94A3B8 !important;
        border-radius: 6px 6px 0 0;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #3B82F6 !important;
    }

    /* Dataframe & Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #1E293B;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_and_normalize_aws_csv(file_or_path):
    """
    Intelligently maps and normalizes heterogeneous real AWS CSV datasets.
    Handles variable headers (Temp, Temperature, TEMP_C, SLP, MSLP, Pressure, RH, Humidity, Date_Time, etc.)
    Uses fast format parsing and caching for instant response.
    """
    if isinstance(file_or_path, str):
        df_raw = pd.read_csv(file_or_path)
    else:
        df_raw = pd.read_csv(file_or_path)

    col_map = {}
    for col in df_raw.columns:
        cl = str(col).lower().strip()
        # Check timestamp first to avoid collision with temperature
        if any(k in cl for k in ['timestamp', 'datetime', 'date_time', 'time', 'date']) and 'timestamp' not in col_map:
            col_map['timestamp'] = col
        elif any(k in cl for k in ['temperature', 'temp', 't_dry', 't_air']) and 'temperature' not in col_map:
            col_map['temperature'] = col
        elif any(k in cl for k in ['pressure', 'press', 'baro', 'mslp', 'slp']) and 'pressure' not in col_map:
            col_map['pressure'] = col
        elif any(k in cl for k in ['humidity', 'humid', 'rh']) and 'humidity' not in col_map:
            col_map['humidity'] = col

    # Fallback to check single-letter symbols
    for col in df_raw.columns:
        cl = str(col).lower().strip()
        if cl in ['t', 'ta'] and 'temperature' not in col_map:
            col_map['temperature'] = col
        elif cl in ['p', 'pa'] and 'pressure' not in col_map:
            col_map['pressure'] = col
        elif cl in ['u', 'h'] and 'humidity' not in col_map:
            col_map['humidity'] = col

    cols = list(df_raw.columns)
    t_col = col_map.get('temperature', cols[1] if len(cols) > 1 else cols[0])
    p_col = col_map.get('pressure', cols[2] if len(cols) > 2 else cols[0])
    rh_col = col_map.get('humidity', cols[3] if len(cols) > 3 else cols[0])

    clean_df = pd.DataFrame()
    if 'timestamp' in col_map:
        raw_ts = df_raw[col_map['timestamp']]
        try:
            clean_df['timestamp'] = pd.to_datetime(raw_ts, format='%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                clean_df['timestamp'] = pd.to_datetime(raw_ts, format='ISO8601')
            except Exception:
                clean_df['timestamp'] = pd.to_datetime(raw_ts, errors='coerce')
    else:
        clean_df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_raw), freq='15min')

    clean_df['temperature'] = pd.to_numeric(df_raw[t_col], errors='coerce')
    clean_df['pressure'] = pd.to_numeric(df_raw[p_col], errors='coerce')
    clean_df['humidity'] = pd.to_numeric(df_raw[rh_col], errors='coerce')

    clean_df = clean_df.dropna(subset=['temperature', 'pressure', 'humidity']).reset_index(drop=True)
    return clean_df


# Initialize Session State
if 'initialized' not in st.session_state:
    st.session_state.simulator = AWSDataSimulator(seed=101)
    st.session_state.detector = AWSAnomalyDetector(contamination=0.03)
    st.session_state.imputer = AWSImputer()
    st.session_state.health_monitor = SensorHealthMonitor()

    # Pre-train baseline on 2 days clean data (15-min synoptic interval)
    clean_hist = st.session_state.simulator.generate_historical_dataset(days=2, interval_minutes=15, inject_anomalies=False)
    st.session_state.detector.fit(clean_hist)

    # Telemetry streaming buffer
    st.session_state.history = []
    st.session_state.anomaly_log = []
    st.session_state.stream_step = 0
    st.session_state.is_streaming = False
    st.session_state.active_fault = None
    st.session_state.active_fault_param = None
    st.session_state.active_fault_value = 0.0

    # CSV streaming replay state
    st.session_state.csv_stream_idx = 0
    st.session_state.loaded_csv_df = None
    st.session_state.active_csv_name = ""
    st.session_state.replay_finished = False

    # Seed initial 30 observations at 15-minute intervals
    base_time = pd.Timestamp.now() - pd.Timedelta(minutes=30 * 15)
    for i in range(30):
        t, p, rh = st.session_state.simulator.generate_point(base_time + pd.Timedelta(minutes=i * 15))
        rep = st.session_state.detector.process_observation(t, p, rh, base_time + pd.Timedelta(minutes=i * 15))
        imp = st.session_state.imputer.impute_point(t, p, rh, None, i)
        st.session_state.imputer.update_clean_history(t, p, rh, i)
        st.session_state.health_monitor.record_observation(t, p, rh, None)
        st.session_state.history.append({
            "step": i,
            "timestamp": base_time + pd.Timedelta(minutes=i * 15),
            "temperature": t, "pressure": p, "humidity": rh,
            "imp_temp": imp["imputed_temperature"], "imp_press": imp["imputed_pressure"], "imp_rh": imp["imputed_humidity"],
            "is_anomaly": False, "is_weather_event": False, "anomaly_type": "NORMAL", "confidence": 1.0,
            "faulty_sensor": None, "explanation": rep.explanation, "thermo": rep.thermodynamics, "top_features": []
        })
    st.session_state.stream_step = 30
    st.session_state.initialized = True


def process_next_step(interval_mins=15):
    """Advance the streaming simulation or CSV replay by one observation."""
    st.session_state.stream_step += 1
    step = st.session_state.stream_step

    is_csv_active = (
        st.session_state.get("data_source_type") == "Real AWS Dataset Streamer (CSV Replay)"
        and st.session_state.loaded_csv_df is not None
        and len(st.session_state.loaded_csv_df) > 0
    )

    if is_csv_active:
        df_src = st.session_state.get("loaded_csv_test_slice")
        if df_src is None or len(df_src) == 0:
            df_src = st.session_state.loaded_csv_df

        idx = st.session_state.csv_stream_idx
        if idx >= len(df_src):
            st.session_state.is_streaming = False
            st.session_state.replay_finished = True
            return

        row = df_src.iloc[idx]
        t = float(row["temperature"])
        p = float(row["pressure"])
        rh = float(row["humidity"])
        if pd.notna(row["timestamp"]):
            curr_time = row["timestamp"]
        else:
            base_t = st.session_state.history[-1]["timestamp"] if st.session_state.history else pd.Timestamp.now()
            curr_time = base_t + pd.Timedelta(minutes=interval_mins)

        # Dynamic interval resolution: 1m in storm mode, 15m in nominal
        freq_mode = st.session_state.get("freq_mode", "Adaptive (Auto-Detect: 1m Storm / 15m Routine)")
        latest_hist_rec = st.session_state.history[-1] if st.session_state.history else None
        is_storm_condition = (
            (latest_hist_rec and latest_hist_rec.get("is_weather_event", False))
            or (st.session_state.active_fault == "GENUINE_WEATHER_EVENT")
        )

        if "Adaptive" in freq_mode and not is_storm_condition and pd.notna(row["timestamp"]):
            curr_ts = row["timestamp"]
            next_target = curr_ts + pd.Timedelta(minutes=15)
            next_idx = idx + 1
            while next_idx < len(df_src):
                if df_src.iloc[next_idx]["timestamp"] >= next_target:
                    break
                next_idx += 1
            st.session_state.csv_stream_idx = next_idx
        else:
            st.session_state.csv_stream_idx += 1

        if st.session_state.csv_stream_idx >= len(df_src):
            st.session_state.is_streaming = False
            st.session_state.replay_finished = True
    else:
        base_t = st.session_state.history[-1]["timestamp"] if st.session_state.history else pd.Timestamp.now()
        curr_time = base_t + pd.Timedelta(minutes=interval_mins)
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
        drift_rate = 0.04 if interval_mins <= 2 else 0.15
        if param == "temperature":
            st.session_state.active_fault_value += drift_rate
            t += st.session_state.active_fault_value
        elif param == "humidity":
            st.session_state.active_fault_value += (drift_rate * 2.5)
            rh = min(100.0, max(0.0, rh + st.session_state.active_fault_value))
        elif param == "pressure":
            st.session_state.active_fault_value += (drift_rate * 0.75)
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


def run_batch_test_window(train_subset, test_subset):
    """
    Run fast adaptive evaluation on test_subset, calibrated by train_subset.
    Matches the terminal workflow from main.py --evaluate.
    """
    profile = st.session_state.detector.calibrate_sensors(train_subset)
    st.session_state.calibrated_profile = profile

    train_15 = train_subset[train_subset['timestamp'].dt.minute.isin([0, 15, 30, 45])]
    if len(train_15) < 30:
        train_15 = train_subset
    if len(train_15) > 1000:
        train_15 = train_15.iloc[-1000:]

    st.session_state.detector.fit(train_15, calibrate_persistence=False)
    st.session_state.detector.reset_stream()

    test_df = test_subset.sort_values("timestamp").reset_index(drop=True)
    n_test = len(test_df)
    timestamps = test_df["timestamp"].tolist()

    history = []
    anomaly_log = []
    mode = "15MIN"
    next_15 = timestamps[0]
    switches = 0

    for pos in range(n_test):
        ts = timestamps[pos]
        if mode == "15MIN" and ts < next_15:
            continue

        row = test_df.iloc[pos]
        t = float(row["temperature"])
        p = float(row["pressure"])
        rh = float(row["humidity"])

        rep = st.session_state.detector.process_observation(t, p, rh, ts)
        rec = {
            "step": pos,
            "timestamp": ts,
            "temperature": t, "pressure": p, "humidity": rh,
            "imp_temp": t, "imp_press": p, "imp_rh": rh,
            "is_anomaly": rep.is_anomaly,
            "is_weather_event": rep.is_weather_event,
            "anomaly_type": rep.anomaly_type,
            "confidence": rep.confidence,
            "faulty_sensor": rep.faulty_sensor,
            "explanation": rep.explanation,
            "thermo": rep.thermodynamics,
            "top_features": rep.top_features
        }
        history.append(rec)
        if rep.is_anomaly or rep.is_weather_event:
            anomaly_log.insert(0, rec)

        if rep.is_weather_event:
            if mode == "15MIN":
                mode = "1MIN"
                switches += 1
        elif mode == "1MIN":
            mode = "15MIN"
            switches += 1
            next_15 = ts + pd.Timedelta(minutes=15)
        else:
            next_15 = ts + pd.Timedelta(minutes=15)

    st.session_state.history = history[-120:] if len(history) > 120 else history
    st.session_state.anomaly_log = anomaly_log[:100]
    st.session_state.loaded_csv_test_slice = test_df
    st.session_state.csv_stream_idx = n_test
    st.session_state.replay_finished = True
    st.session_state.is_streaming = False
    st.session_state.batch_eval_summary = {
        "total_test_rows": n_test,
        "processed_samples": len(history),
        "anomalies": sum(1 for r in history if r["is_anomaly"]),
        "weather_events": sum(1 for r in history if r["is_weather_event"]),
        "telemetry_savings_pct": (1.0 - len(history) / max(1, n_test)) * 100.0,
        "mode_switches": switches
    }


# Sidebar Navigation & Station Selection (No Emojis)
st.sidebar.markdown("### SkyGuard by Vyoma")
st.sidebar.caption("AWS Intelligent Anomaly Detection & Diagnostics")

selected_station = st.sidebar.selectbox("Active Station", [
    "Durg",
    "Raipur",
    "Bilaspur",
    "Raigarh"
])

st.sidebar.markdown('<div class="sidebar-section-title">Telemetry Data Source</div>', unsafe_allow_html=True)
data_source_type = st.sidebar.radio(
    "Ingestion Mode",
    ["Simulated Diurnal Stream", "Real AWS Dataset Streamer (CSV Replay)"],
    label_visibility="collapsed"
)

if data_source_type == "Real AWS Dataset Streamer (CSV Replay)":
    csv_presets = {
        "Kanpur Station 1-Min Telemetry (incompass_kanpur_1min.csv)": "incompass_kanpur_1min.csv",
        "Benchmark Dataset (7,200 Observations)": "data/aws_benchmark_dataset.csv",
        "Upload Real Station CSV...": "CUSTOM"
    }
    preset_choice = st.sidebar.selectbox("Select AWS Dataset", list(csv_presets.keys()))
    csv_target = csv_presets[preset_choice]

    if csv_target == "CUSTOM":
        custom_file = st.sidebar.file_uploader("Upload Station CSV (Any standard AWS format)", type=["csv"])
        if custom_file is not None:
            if st.session_state.get("active_csv_name") != custom_file.name:
                with st.spinner("Normalizing uploaded AWS CSV..."):
                    df_loaded = load_and_normalize_aws_csv(custom_file)
                    st.session_state.loaded_csv_df = df_loaded
                    st.session_state.active_csv_name = custom_file.name
                    st.session_state.loaded_csv_test_slice = None
                    st.session_state.csv_stream_idx = 0
                    st.sidebar.success(f"Loaded {len(df_loaded)} valid records!")
    else:
        if st.session_state.get("active_csv_name") != csv_target:
            if os.path.exists(csv_target):
                with st.spinner(f"Loading {preset_choice}..."):
                    df_loaded = load_and_normalize_aws_csv(csv_target)
                    st.session_state.loaded_csv_df = df_loaded
                    st.session_state.active_csv_name = csv_target
                    st.session_state.loaded_csv_test_slice = None
                    st.session_state.csv_stream_idx = 0
                    st.sidebar.success(f"Loaded {len(df_loaded)} rows from {os.path.basename(csv_target)}")

    if st.session_state.loaded_csv_df is not None and len(st.session_state.loaded_csv_df) > 0:
        df_src = st.session_state.loaded_csv_df
        has_dates = 'timestamp' in df_src.columns and pd.notna(df_src['timestamp'].iloc[0])

        if has_dates:
            with st.sidebar.expander("Train / Test Date Windows", expanded=True):
                min_ts = df_src['timestamp'].min()
                max_ts = df_src['timestamp'].max()
                st.caption(f"Dataset Range: {min_ts.strftime('%Y-%m-%d')} to {max_ts.strftime('%Y-%m-%d')}")

                total_span = max_ts - min_ts
                default_train_end = min_ts + total_span * 0.25
                default_test_start = default_train_end + pd.Timedelta(minutes=15)

                col_tw1, col_tw2 = st.columns(2)
                t_start_d = col_tw1.date_input("Train Start", min_ts.date(), min_value=min_ts.date(), max_value=max_ts.date(), key="train_sd")
                t_end_d = col_tw2.date_input("Train End", default_train_end.date(), min_value=min_ts.date(), max_value=max_ts.date(), key="train_ed")

                col_ts1, col_ts2 = st.columns(2)
                test_start_d = col_ts1.date_input("Test Start", default_test_start.date(), min_value=min_ts.date(), max_value=max_ts.date(), key="test_sd")
                test_end_d = col_ts2.date_input("Test End", max_ts.date(), min_value=min_ts.date(), max_value=max_ts.date(), key="test_ed")

                col_b1, col_b2 = st.columns(2)
                btn_stream = col_b1.button("Calibrate & Replay", width='stretch')
                btn_batch = col_b2.button("Run Batch Eval", width='stretch')

                if btn_stream or btn_batch:
                    t_s_dt = pd.Timestamp(t_start_d)
                    t_e_dt = pd.Timestamp(t_end_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    test_s_dt = pd.Timestamp(test_start_d)
                    test_e_dt = pd.Timestamp(test_end_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                    if t_s_dt >= t_e_dt:
                        st.error("Training start must be before training end.")
                    elif test_s_dt <= t_e_dt:
                        st.error("Testing must start strictly AFTER training ends (Zero Data Leakage).")
                    else:
                        train_subset = df_src[(df_src['timestamp'] >= t_s_dt) & (df_src['timestamp'] <= t_e_dt)].copy()
                        test_subset = df_src[(df_src['timestamp'] >= test_s_dt) & (df_src['timestamp'] <= test_e_dt)].copy()

                        if len(train_subset) < 20:
                            st.error(f"Selected training window has only {len(train_subset)} rows (need >= 20).")
                        elif len(test_subset) == 0:
                            st.error("Selected testing window contains no observations.")
                        else:
                            if btn_batch:
                                with st.spinner(f"Evaluating {len(test_subset):,} observations in adaptive 15m/1m mode..."):
                                    run_batch_test_window(train_subset, test_subset)
                                st.sidebar.success(f"Batch eval complete! {len(test_subset):,} test rows processed.")
                            else:
                                with st.spinner("Calibrating sensor profiles and fitting baseline..."):
                                    profile = st.session_state.detector.calibrate_sensors(train_subset)
                                    st.session_state.calibrated_profile = profile

                                    train_15 = train_subset[train_subset['timestamp'].dt.minute.isin([0, 15, 30, 45])]
                                    if len(train_15) < 30:
                                        train_15 = train_subset
                                    if len(train_15) > 1000:
                                        train_15 = train_15.iloc[-1000:]

                                    st.session_state.detector.fit(train_15, calibrate_persistence=False)
                                    st.session_state.detector.reset_stream()
                                    st.session_state.loaded_csv_test_slice = test_subset.reset_index(drop=True)
                                    st.session_state.csv_stream_idx = 0
                                    st.session_state.history = []
                                    st.session_state.anomaly_log = []
                                    st.session_state.stream_step = 0
                                    st.session_state.replay_finished = False
                                    st.session_state.batch_eval_summary = None
                                    st.session_state.step_once = True
                                    st.sidebar.success(f"Calibrated on {len(train_subset):,} rows! Ready to replay {len(test_subset):,} test rows.")

            if st.session_state.get("calibrated_profile"):
                with st.sidebar.expander("Learned Sensor Calibration", expanded=False):
                    prof = st.session_state.calibrated_profile
                    for s_k, p_val in prof.items():
                        st.markdown(f"**{s_k.capitalize()}:**")
                        st.caption(f"Resolution: `{p_val['minimum_real_change']:.4f}` | Max Normal Flatline: `{p_val['normal_run_q99']:.1f}` | Stuck Threshold: `{p_val['learned_run_limit']}` readings")

        target_df = st.session_state.get("loaded_csv_test_slice")
        if target_df is None or len(target_df) == 0:
            target_df = st.session_state.loaded_csv_df

        total_rows = len(target_df)
        curr_idx = st.session_state.csv_stream_idx
        progress_val = min(1.0, curr_idx / max(1, total_rows))
        st.sidebar.progress(progress_val)
        label_prefix = "Test Window Replay" if st.session_state.get("loaded_csv_test_slice") is not None else "CSV Replay"
        st.sidebar.caption(f"{label_prefix}: Row {curr_idx} / {total_rows} ({progress_val*100:.1f}%)")
        if st.sidebar.button("Reset Replay", width='stretch'):
            st.session_state.csv_stream_idx = 0
            st.sidebar.info("Replay reset to row 0.")

st.sidebar.markdown('<div class="sidebar-section-title">Interval & Frequency Mode</div>', unsafe_allow_html=True)
freq_mode = st.sidebar.selectbox("Sampling Frequency", [
    "Adaptive (Auto-Detect: 1m Storm / 15m Routine)",
    "15-Minute Synoptic (Standard IMD Routine)",
    "1-Minute Rapid (Convective Storm Mode)"
])

# Dynamic interval resolution
latest_hist_rec = st.session_state.history[-1] if st.session_state.history else None
is_storm_condition = (
    (latest_hist_rec and latest_hist_rec.get("is_weather_event", False))
    or (st.session_state.active_fault == "GENUINE_WEATHER_EVENT")
)

if "Adaptive" in freq_mode:
    active_interval_mins = 1 if is_storm_condition else 15
elif "1-Minute" in freq_mode:
    active_interval_mins = 1
else:
    active_interval_mins = 15

st.sidebar.markdown('<div class="sidebar-section-title">Telemetry Stream Controls</div>', unsafe_allow_html=True)
col_s1, col_s2 = st.sidebar.columns(2)
if col_s1.button(f"Step (+{active_interval_mins}m)", width='stretch'):
    st.session_state.step_once = True
else:
    st.session_state.step_once = False

stream_toggle = col_s2.button("Toggle Stream", width='stretch')
if stream_toggle:
    st.session_state.is_streaming = not st.session_state.is_streaming

speed = st.sidebar.slider("Stream Speed (seconds/step)", 0.2, 2.0, 0.6, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section-title">Fault Injection Studio</div>', unsafe_allow_html=True)
st.sidebar.caption("Evaluate multi-tier AI differentiation between hardware faults and meteorological events:")

inject_choice = st.sidebar.selectbox("Select Anomaly Scenario", [
    "None (Nominal Diurnal Stream)",
    "Temperature Spike (+12.5°C)",
    "Barometer Pressure Drop Spike (-16.0 hPa)",
    "Humidity Sensor Spike (+40.0%)",
    "Stuck / Frozen Sensor (Zero Variance)",
    "Sensor Calibration Drift (+0.04°C/min)",
    "Genuine Severe Convective Storm (Downburst)",
    "Out of Physical Bounds (RH = 125%)",
    "Packet Loss / Missing Data (NaN)"
])

stuck_sensor_choice = "temperature"
drift_sensor_choice = "temperature"
if "Stuck" in inject_choice:
    stuck_sensor_choice = st.sidebar.selectbox("Sensor to Freeze", ["temperature", "pressure", "humidity"], format_func=lambda x: f"Freeze {x.capitalize()}")
elif "Drift" in inject_choice:
    drift_sensor_choice = st.sidebar.selectbox("Sensor to Drift", ["temperature", "humidity", "pressure"], format_func=lambda x: f"Drift {x.capitalize()}")

if st.sidebar.button("Trigger Selected Scenario", width='stretch'):
    if "None" in inject_choice:
        st.session_state.active_fault = None
        st.sidebar.success("Cleared all injections. Nominal stream resumed.")
    elif "Temperature Spike" in inject_choice:
        st.session_state.active_fault = "SPIKE"
        st.session_state.active_fault_param = "temperature"
        st.session_state.active_fault_value = 12.5
        st.sidebar.warning("Injected: +12.5°C Temperature Spike.")
        if not st.session_state.is_streaming:
            st.session_state.step_once = True
    elif "Barometer Pressure Drop" in inject_choice:
        st.session_state.active_fault = "SPIKE"
        st.session_state.active_fault_param = "pressure"
        st.session_state.active_fault_value = -16.0
        st.sidebar.warning("Injected: -16.0 hPa Barometer Spike.")
        if not st.session_state.is_streaming:
            st.session_state.step_once = True
    elif "Humidity Sensor Spike" in inject_choice:
        st.session_state.active_fault = "SPIKE"
        st.session_state.active_fault_param = "humidity"
        st.session_state.active_fault_value = 40.0
        st.sidebar.warning("Injected: +40% Humidity Spike.")
        if not st.session_state.is_streaming:
            st.session_state.step_once = True
    elif "Stuck" in inject_choice:
        st.session_state.active_fault = "STUCK_SENSOR"
        st.session_state.active_fault_param = stuck_sensor_choice
        latest_hist = st.session_state.history[-1] if st.session_state.history else None
        if latest_hist and stuck_sensor_choice in latest_hist:
            st.session_state.active_fault_value = float(latest_hist[stuck_sensor_choice])
        else:
            st.session_state.active_fault_value = 25.4 if stuck_sensor_choice == "temperature" else (1013.2 if stuck_sensor_choice == "pressure" else 62.0)
        st.sidebar.warning(f"Injected: Frozen {stuck_sensor_choice.capitalize()} Sensor (stuck at {st.session_state.active_fault_value:.2f}).")
        if not st.session_state.is_streaming:
            st.session_state.step_once = True
    elif "Drift" in inject_choice:
        st.session_state.active_fault = "SENSOR_DRIFT"
        st.session_state.active_fault_param = drift_sensor_choice
        st.session_state.active_fault_value = 0.0
        st.sidebar.warning(f"Injected: Progressive {drift_sensor_choice.capitalize()} Sensor Drift.")
        if not st.session_state.is_streaming:
            st.session_state.step_once = True
    elif "Genuine Severe Convective Storm" in inject_choice or "Convective Storm" in inject_choice or "GENUINE" in inject_choice:
        st.session_state.active_fault = "GENUINE_WEATHER_EVENT"
        st.sidebar.info("Triggered: Genuine Severe Convective Storm Downburst.")
        if not st.session_state.is_streaming:
            st.session_state.step_once = True
    elif "Out of Physical Bounds" in inject_choice:
        st.session_state.active_fault = "OUT_OF_BOUNDS"
        st.session_state.active_fault_param = "humidity"
        st.session_state.active_fault_value = 125.0
        st.sidebar.error("Injected: Out of Physical Bounds RH (125%).")
        if not st.session_state.is_streaming:
            st.session_state.step_once = True
    elif "Packet Loss" in inject_choice:
        st.session_state.active_fault = "MISSING"
        st.session_state.active_fault_param = "temperature"
        st.sidebar.error("Injected: Missing Data Packet (NaN).")
        if not st.session_state.is_streaming:
            st.session_state.step_once = True


# Step execution
if st.session_state.step_once:
    process_next_step(interval_mins=active_interval_mins)
    st.session_state.step_once = False

if not st.session_state.history:
    process_next_step(interval_mins=active_interval_mins)

latest = st.session_state.history[-1]
prev = st.session_state.history[-2] if len(st.session_state.history) >= 2 else latest

# Main Header
st.markdown('<div class="brand-title">SkyGuard by Vyoma</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Automatic Weather Station (AWS) Intelligent Anomaly Detection & Diagnostics System • Physics-Guided AI • WMO-No. 8 Standards</div>', unsafe_allow_html=True)

if st.session_state.get("batch_eval_summary"):
    s = st.session_state.batch_eval_summary
    fault_color = "#F87171" if s['anomalies'] > 0 else "#34D399"
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #3B82F6; border-radius: 8px; padding: 14px 18px; margin-bottom: 18px;">
        <div style="color: #60A5FA; font-weight: 700; text-transform: uppercase; font-size: 0.76rem; letter-spacing: 0.08em; margin-bottom: 6px;">Adaptive Test Window Evaluation Summary</div>
        <div style="display: flex; gap: 24px; flex-wrap: wrap; font-family: 'IBM Plex Mono', monospace;">
            <div><span style="color: #94A3B8; font-size: 0.78rem;">Test Rows:</span> <strong style="color: #FFFFFF;">{s['total_test_rows']:,}</strong></div>
            <div><span style="color: #94A3B8; font-size: 0.78rem;">Evaluated Points:</span> <strong style="color: #FFFFFF;">{s['processed_samples']:,}</strong></div>
            <div><span style="color: #94A3B8; font-size: 0.78rem;">Bandwidth Saved:</span> <strong style="color: #34D399;">{s['telemetry_savings_pct']:.1f}%</strong></div>
            <div><span style="color: #94A3B8; font-size: 0.78rem;">Sensor Faults:</span> <strong style="color: {fault_color};">{s['anomalies']:,}</strong></div>
            <div><span style="color: #94A3B8; font-size: 0.78rem;">Severe Storms:</span> <strong style="color: #FBBF24;">{s['weather_events']:,}</strong></div>
            <div><span style="color: #94A3B8; font-size: 0.78rem;">Rate Switches:</span> <strong style="color: #E2E8F0;">{s['mode_switches']:,}</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.get("replay_finished", False):
    st.info("Test Window Replay Finished: All test observations evaluated.")

# Dynamic Hero Box for Recent Reading (Tint changes only for Hero Box)
if latest["is_anomaly"]:
    tint_class = "hero-tint-anomaly"
    pill_class = "hero-pill-anomaly"
    status_label = "SENSOR FAULT DETECTED"
    status_headline = f"{latest['anomaly_type']} — {str(latest['faulty_sensor']).upper()} SENSOR"
    status_detail = f"Confidence: {latest['confidence']*100:.1f}% • Imputation Engine Activated"
elif latest["is_weather_event"]:
    tint_class = "hero-tint-weather"
    pill_class = "hero-pill-weather"
    status_label = "GENUINE METEOROLOGICAL EVENT"
    status_headline = "SEVERE CONVECTIVE STORM / DOWNBURST ACTIVE"
    status_detail = "Multi-variable atmospheric coupling confirmed. Sensor fault alarms suppressed."
else:
    tint_class = "hero-tint-normal"
    pill_class = "hero-pill-normal"
    status_label = "SYSTEM NOMINAL"
    status_headline = "ALL SENSORS NOMINAL • WMO STANDARDS COMPLIANT"
    status_detail = "Physical plausibility, rate-of-change, and diurnal thermodynamic checks verified."

d_t = latest["temperature"] - prev["temperature"] if not np.isnan(latest["temperature"]) and not np.isnan(prev["temperature"]) else 0.0
d_p = latest["pressure"] - prev["pressure"] if not np.isnan(latest["pressure"]) and not np.isnan(prev["pressure"]) else 0.0
d_rh = latest["humidity"] - prev["humidity"] if not np.isnan(latest["humidity"]) and not np.isnan(prev["humidity"]) else 0.0

val_t = f"{latest['temperature']:.2f} °C" if not np.isnan(latest["temperature"]) else "MISSING"
val_p = f"{latest['pressure']:.2f} hPa" if not np.isnan(latest["pressure"]) else "MISSING"
val_rh = f"{latest['humidity']:.1f} %" if not np.isnan(latest["humidity"]) else "MISSING"

st_rep = st.session_state.health_monitor.generate_station_report()

frame_desc = f"{active_interval_mins}m interval • {'Storm Rapid' if latest['is_weather_event'] else 'Synoptic Routine'}"

hero_html = f"""
<div class="hero-container {tint_class}">
    <div class="hero-top-row">
        <div>
            <span class="hero-status-pill {pill_class}">{status_label}</span>
            <div class="hero-status-title">{status_headline}</div>
            <div class="hero-status-desc">{status_detail}</div>
        </div>
        <div class="hero-meta-panel">
            <div>Station: <span class="hero-meta-accent">{selected_station}</span></div>
            <div>Timestamp: <span class="hero-meta-accent">{latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}</span></div>
            <div>Telemetry Frame: <span class="hero-meta-accent">Step #{latest['step']} ({frame_desc})</span></div>
        </div>
    </div>
    <div class="hero-metrics-grid">
        <div class="hero-metric-tile">
            <div class="hero-metric-caption">Temperature</div>
            <div class="hero-metric-number">{val_t}</div>
            <div class="hero-metric-tendency">Delta: {d_t:+.2f} °C / {active_interval_mins}m</div>
        </div>
        <div class="hero-metric-tile">
            <div class="hero-metric-caption">Atmospheric Pressure</div>
            <div class="hero-metric-number">{val_p}</div>
            <div class="hero-metric-tendency">Delta: {d_p:+.2f} hPa / {active_interval_mins}m</div>
        </div>
        <div class="hero-metric-tile">
            <div class="hero-metric-caption">Relative Humidity</div>
            <div class="hero-metric-number">{val_rh}</div>
            <div class="hero-metric-tendency">Delta: {d_rh:+.1f} % / {active_interval_mins}m</div>
        </div>
        <div class="hero-metric-tile">
            <div class="hero-metric-caption">Station Health Index</div>
            <div class="hero-metric-number">{st_rep.overall_health:.1f}%</div>
            <div class="hero-metric-tendency">Status: {st_rep.station_status}</div>
        </div>
    </div>
    <div class="hero-footer-row">
        <span class="hero-footer-label">AI Diagnostic Rationale:</span> &nbsp;{latest['explanation']}
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# Detailed Operations Tabs
tab_telemetry, tab_xai, tab_health, tab_audit = st.tabs([
    "Real-Time Telemetry & Detection",
    "Explainable AI (TreeSHAP) & Diagnostics",
    "Predictive Sensor Maintenance",
    "Incident Audit Log & Export"
])

with tab_telemetry:
    df_hist = pd.DataFrame(st.session_state.history)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Temperature (°C) — Raw vs Flagged Fault vs Imputed Reconstruction",
            "Atmospheric Pressure (hPa) — Barometric Pressure Profile",
            "Relative Humidity (%) — Moisture Plausibility & Condensation Boundary"
        )
    )

    # Temperature Traces
    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['temperature'],
        mode='lines+markers', name='Raw Temperature',
        line=dict(color='#38BDF8', width=2), marker=dict(size=4)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['imp_temp'],
        mode='lines', name='Imputed Temperature',
        line=dict(color='#34D399', width=2, dash='dash')
    ), row=1, col=1)

    anom_temp = df_hist[df_hist['is_anomaly'] & (df_hist['faulty_sensor'] == 'temperature')]
    if len(anom_temp) > 0:
        fig.add_trace(go.Scatter(
            x=anom_temp['timestamp'], y=anom_temp['temperature'],
            mode='markers', name='Temperature Fault',
            marker=dict(color='#F87171', size=10, symbol='x-thin', line=dict(width=3, color='#F87171'))
        ), row=1, col=1)

    storms = df_hist[df_hist['is_weather_event']]
    if len(storms) > 0:
        fig.add_trace(go.Scatter(
            x=storms['timestamp'], y=storms['temperature'],
            mode='markers', name='Genuine Storm Event',
            marker=dict(color='#FBBF24', size=11, symbol='star')
        ), row=1, col=1)

    # Pressure Traces
    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['pressure'],
        mode='lines+markers', name='Raw Pressure',
        line=dict(color='#A78BFA', width=2), marker=dict(size=4)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['imp_press'],
        mode='lines', name='Imputed Pressure',
        line=dict(color='#34D399', width=2, dash='dash')
    ), row=2, col=1)

    anom_press = df_hist[df_hist['is_anomaly'] & (df_hist['faulty_sensor'] == 'pressure')]
    if len(anom_press) > 0:
        fig.add_trace(go.Scatter(
            x=anom_press['timestamp'], y=anom_press['pressure'],
            mode='markers', name='Pressure Fault',
            marker=dict(color='#F87171', size=10, symbol='x-thin', line=dict(width=3, color='#F87171'))
        ), row=2, col=1)

    # Humidity Traces
    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['humidity'],
        mode='lines+markers', name='Raw Humidity',
        line=dict(color='#22D3EE', width=2), marker=dict(size=4)
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df_hist['timestamp'], y=df_hist['imp_rh'],
        mode='lines', name='Imputed Humidity',
        line=dict(color='#34D399', width=2, dash='dash')
    ), row=3, col=1)

    anom_rh = df_hist[df_hist['is_anomaly'] & (df_hist['faulty_sensor'] == 'humidity')]
    if len(anom_rh) > 0:
        fig.add_trace(go.Scatter(
            x=anom_rh['timestamp'], y=anom_rh['humidity'],
            mode='markers', name='Humidity Fault',
            marker=dict(color='#F87171', size=10, symbol='x-thin', line=dict(width=3, color='#F87171'))
        ), row=3, col=1)

    fig.update_layout(
        height=680,
        margin=dict(l=40, r=20, t=40, b=20),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.65)",
        font=dict(color="#E2E8F0", family="IBM Plex Sans, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#E2E8F0")
        )
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.12)", zerolinecolor="rgba(148, 163, 184, 0.2)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.12)", zerolinecolor="rgba(148, 163, 184, 0.2)")

    st.plotly_chart(fig, width='stretch', key="main_telemetry_chart")

with tab_xai:
    st.markdown("### Explainable AI (TreeSHAP) & Root-Cause Diagnostics")
    if len(st.session_state.anomaly_log) == 0:
        st.info("No anomalies logged yet. Use the Fault Injection Studio or Real CSV Streamer in the sidebar to simulate operational incidents.")
    else:
        log_options = [
            f"Step #{r['step']} | {r['timestamp'].strftime('%H:%M:%S UTC')} | {r['anomaly_type']} | {r['faulty_sensor'] or 'Atmospheric Event'}"
            for r in st.session_state.anomaly_log
        ]
        selected_idx = st.selectbox("Select Logged Incident to Inspect", range(len(log_options)), format_func=lambda i: log_options[i])
        target_event = st.session_state.anomaly_log[selected_idx]

        col_x1, col_x2 = st.columns([1, 1])
        with col_x1:
            st.markdown("#### Diagnostic Summary")
            st.markdown(f"**Classification:** `{target_event['anomaly_type']}`")
            st.markdown(f"**Target Component:** `{str(target_event['faulty_sensor']).upper()}`")
            st.markdown(f"**Model Confidence:** `{target_event['confidence']*100:.1f}%`")
            st.info(f"**AI Diagnostic Rationale:**\n\n{target_event['explanation']}")
        with col_x2:
            st.markdown("#### Feature Attribution (TreeSHAP Scores)")
            feats = target_event.get("top_features", [])
            if feats:
                fig_xai = go.Figure(go.Bar(
                    x=[abs(f[1]) for f in feats][::-1],
                    y=[f[0] for f in feats][::-1],
                    orientation='h',
                    marker=dict(color='#3B82F6')
                ))
                fig_xai.update_layout(
                    title="TreeSHAP Feature Contributions to Decision",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15, 23, 42, 0.65)",
                    font=dict(color="#E2E8F0")
                )
                fig_xai.update_xaxes(gridcolor="rgba(148, 163, 184, 0.12)")
                st.plotly_chart(fig_xai, width='stretch', key=f"xai_chart_{selected_idx}")
            else:
                st.info("Feature attribution is calculated for multi-variate statistical events.")

with tab_health:
    st.markdown("### Predictive Sensor Health & Degradation Monitor")
    st_report = st.session_state.health_monitor.generate_station_report()
    h_col1, h_col2, h_col3 = st.columns(3)
    for col, (s_name, s_stat) in zip([h_col1, h_col2, h_col3], st_report.sensors.items()):
        with col:
            st.markdown(f"#### {s_name.capitalize()} Sensor")
            gauge_color = "#10B981" if s_stat.health_score >= 80 else ("#F59E0B" if s_stat.health_score >= 50 else "#EF4444")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=s_stat.health_score,
                number={'suffix': "%", 'font': {'color': "#FFFFFF", 'family': "IBM Plex Mono"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#94A3B8"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "rgba(30, 41, 59, 0.5)",
                    'borderwidth': 1,
                    'bordercolor': "#334155"
                }
            ))
            fig_g.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0")
            )
            st.plotly_chart(fig_g, width='stretch', key=f"gauge_{s_name}")
            st.markdown(f"**Operational Status:** `{s_stat.status}`")
            st.info(f"**Maintenance Recommendation:**\n\n{s_stat.recommendation}")

with tab_audit:
    st.markdown("### Incident Audit Log & Telemetry Export")
    if len(st.session_state.anomaly_log) > 0:
        log_df = pd.DataFrame(st.session_state.anomaly_log)
        disp_cols = ['step', 'timestamp', 'anomaly_type', 'faulty_sensor', 'confidence', 'temperature', 'pressure', 'humidity', 'explanation']
        st.dataframe(log_df[disp_cols], width='stretch')
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "Download Incident Audit Report (CSV)",
                data=log_df[disp_cols].to_csv(index=False).encode('utf-8'),
                file_name="aws_incident_audit.csv",
                mime="text/csv",
                width='stretch'
            )
        with col_d2:
            full_df = pd.DataFrame(st.session_state.history)
            st.download_button(
                "Download Full Imputed Dataset (CSV)",
                data=full_df.to_csv(index=False).encode('utf-8'),
                file_name="aws_imputed_telemetry.csv",
                mime="text/csv",
                width='stretch'
            )
    else:
        st.info("No incidents recorded yet. Nominal observations are streaming.")

# =============================================================================
# Edge AI (ESP32) Microcontroller Deployment (Bottom of website)
# =============================================================================
st.markdown("---")
st.markdown("### Edge AI (ESP32) Microcontroller Deployment")
st.caption("TinyML Firmware Specification & Ultra-Low Latency Sensor Inference Engine")
col_e1, col_e2 = st.columns([1, 1])
with col_e1:
    st.table(pd.DataFrame({
        "Specification Metric": ["Target Microcontroller", "CPU Core / Clock Speed", "Inference Latency", "SRAM Memory Footprint", "Bandwidth Reduction"],
        "Edge Performance": ["ESP32 / ESP32-S3", "240 MHz Xtensa LX7", "< 0.08 ms per sample", "1.4 KB ring buffer", "Up to 95% telemetry savings"]
    }))
with col_e2:
    try:
        with open("edge/esp32_anomaly_detector.h", "r") as f:
            st.code(f.read()[:1500] + "\n\n// ... [Complete implementation in edge/esp32_anomaly_detector.h]", language="c")
    except Exception:
        st.code("// edge/esp32_anomaly_detector.h", language="c")

# Auto refresh loop
if st.session_state.is_streaming:
    time.sleep(speed)
    process_next_step(interval_mins=active_interval_mins)
    st.rerun()