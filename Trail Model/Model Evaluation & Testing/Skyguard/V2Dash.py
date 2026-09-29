import streamlit as st
import time
import pandas as pd
from pathlib import Path

from model_runner import StreamProcessor

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛡️",
    layout="wide"
)


st.markdown(
    """
    <style>

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        .main-title {
            font-size: 2.6rem;
            font-weight: 750;
            margin-bottom: 0.1rem;
        }

        .subtitle {
            color: #888888;
            font-size: 1rem;
            margin-bottom: 1.8rem;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        .batch-title {
            font-size: 1.15rem;
            font-weight: 700;
        }

        .small-text {
            color: #888888;
            font-size: 0.88rem;
        }

        .big-number {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .live-strip {
            padding: 0.85rem 1rem;
            border: 1px solid #333842;
            border-radius: 0.65rem;
            background: #11141a;
            margin-bottom: 1rem;
        }

        .live-status {
            font-weight: 700;
            letter-spacing: 0.03em;
        }

    </style>
    """,
    unsafe_allow_html=True
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV_PATH = PROJECT_ROOT / "Model Evaluation & Testing" / "AWS_Weather_5000_Clean.csv"
STREAM_CSV_PATH = PROJECT_ROOT / "ModelV2" / "ART3" / "AWS_Weather_5000_With_Anomalies.csv"


@st.cache_data
def load_data():

    clean_df = pd.read_csv(CLEAN_CSV_PATH)
    stream_df = pd.read_csv(STREAM_CSV_PATH)

    for name, frame in (("clean training", clean_df), ("stream", stream_df)):
        if "timestamp" not in frame.columns:
            raise ValueError(f"{name.title()} CSV must contain a timestamp column.")

        frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)
    stream_df = stream_df.sort_values("timestamp").reset_index(drop=True)

    return clean_df, stream_df


try:
    clean_df, df = load_data()
except Exception as e:
    st.error(f"Could not load datasets: {e}")
    st.stop()


if "processor" not in st.session_state:

    st.session_state.processor = None


if "visible_blocks" not in st.session_state:

    st.session_state.visible_blocks = 1


if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

if "stream_speed" not in st.session_state:
    st.session_state.stream_speed = 0.6


with st.sidebar:

    st.header("Live Telemetry")

    st.caption(
        "Simulated AWS telemetry from the anomaly dataset. "
        "ART3 processes one reading at a time."
    )

    live_col1, live_col2 = st.columns(2)

    with live_col1:
        start_stream = st.button(
            "▶ Start Stream",
            use_container_width=True,
            disabled=st.session_state.is_streaming
        )

    with live_col2:
        pause_stream = st.button(
            "⏸ Pause",
            use_container_width=True,
            disabled=not st.session_state.is_streaming
        )

    stream_speed = st.slider(
        "Telemetry Interval (seconds / reading)",
        min_value=0.2,
        max_value=2.0,
        value=float(st.session_state.stream_speed),
        step=0.1,
        help="Real-time delay between simulated telemetry readings. Each reading represents the next record in the CSV stream."
    )
    st.session_state.stream_speed = stream_speed

    if st.session_state.is_streaming:
        st.success("● LIVE — receiving telemetry")
    elif st.session_state.processor is not None and st.session_state.processor.finished():
        st.info("✓ STREAM COMPLETE")
    else:
        st.caption("Ⅱ PAUSED — stream position is preserved")

    st.divider()

    st.subheader("Stream Status")

    if st.session_state.processor is not None:
        p = st.session_state.processor
        processed_sidebar, total_sidebar, _ = p.progress()

        st.caption("Records Processed")
        st.markdown(
            f'<div class="big-number">{processed_sidebar:,} / {total_sidebar:,}</div>',
            unsafe_allow_html=True
        )

        current_sidebar = p.last_batch.get("latest_reading", {}) if p.last_batch else {}
        current_ts_sidebar = current_sidebar.get("timestamp", "—")
        current_batch_sidebar = getattr(p, "live_batch_number", 0)

        st.caption("Current Record")
        st.write(f"#{processed_sidebar:,}")

        st.caption("Simulated Time")
        st.write(str(current_ts_sidebar))

        st.caption("Current Batch")
        st.write(f"#{current_batch_sidebar}")
    else:
        st.caption("Records Processed")
        st.write("0 / 0")
        st.caption("Current Record")
        st.write("—")
        st.caption("Simulated Time")
        st.write("—")
        st.caption("Current Batch")
        st.write("—")

    st.divider()

    with st.expander("⚙ Advanced Configuration", expanded=False):

        st.caption(
            "These settings control model training and stream range. "
            "They are not required for normal live operation."
        )

        baseline_size = st.number_input(
            "Baseline Size",
            min_value=1,
            max_value=len(clean_df),
            value=len(clean_df),
            step=100,
            help="Number of clean readings used to train the ART3 model."
        )

        st.caption("Detection Range")
        range_col1, range_col2 = st.columns(2)

        with range_col1:
            start_index = st.number_input(
                "Start",
                min_value=0,
                max_value=max(0, len(df) - 1),
                value=0,
                step=10,
                label_visibility="collapsed",
                help="First record in the simulated anomaly stream."
            )

        with range_col2:
            end_index = st.number_input(
                "End",
                min_value=1,
                max_value=len(df),
                value=len(df),
                step=10,
                label_visibility="collapsed",
                help="Last exclusive index in the simulated anomaly stream."
            )

        batch_size = st.number_input(
            "Feed Group Size",
            min_value=1,
            max_value=len(df),
            value=50,
            step=10,
            help="Number of telemetry readings grouped into one Live Processing Feed batch. ART3 still processes each reading individually."
        )

        apply_settings = st.button(
            "Apply Settings",
            use_container_width=True,
            type="primary"
        )

    if start_stream:
        if st.session_state.processor is None:
            try:
                st.session_state.processor = StreamProcessor(
                    df=df,
                    baseline_size=baseline_size,
                    start_index=start_index,
                    end_index=end_index,
                    batch_size=batch_size,
                    baseline_df=clean_df
                )
                st.session_state.visible_blocks = 1
            except Exception as e:
                st.error(f"Could not start the live stream: {e}")
                st.stop()

        st.session_state.is_streaming = True
        st.rerun()

    if pause_stream:
        st.session_state.is_streaming = False
        st.rerun()

    reset_detection = st.button(
        "↻ Reset Detection",
        use_container_width=True
    )

    st.divider()

    st.caption(f"Dataset: {len(df):,} simulated telemetry records")
    st.caption("Training: Clean AWS telemetry")
    st.caption("Detection Model: ART3")


if apply_settings:

    try:

        st.session_state.is_streaming = False

        st.session_state.processor = StreamProcessor(
            df=df,

            baseline_size=baseline_size,

            start_index=start_index,

            end_index=end_index,

            batch_size=batch_size,

            baseline_df=clean_df
        )

        st.session_state.visible_blocks = 1

        st.success(
            "Processing settings applied."
        )

    except Exception as e:

        st.error(str(e))


if reset_detection:

    st.session_state.is_streaming = False

    if st.session_state.processor is not None:

        st.session_state.processor.reset()

    st.session_state.visible_blocks = 1

    st.info(
        "Detection has been reset."
    )


processor = st.session_state.processor


st.markdown(
    '<div class="main-title">SKYGUARD AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Intelligent Weather Anomaly Monitoring System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    'Live Processing Feed'
    '</div>',
    unsafe_allow_html=True
)

if processor is None:

    st.info(
        "Press ▶ Start Stream to begin simulated live telemetry. "
        "The clean CSV trains ART3; the anomaly CSV is processed one reading at a time."
    )

else:

    processed, total, remaining = (
        processor.progress()
    )

    progress_value = (

        processed / total

        if total > 0

        else 0
    )

    st.progress(
        progress_value,

        text=(
            f"{processed} of {total} "
            f"detection readings processed"
        )
    )

    latest = processor.last_batch.get("latest_reading", {}) if processor.last_batch else {}
    simulated_time = latest.get("timestamp", "—")
    current_batch = getattr(processor, "live_batch_number", 0)

    if st.session_state.is_streaming:
        status_text = "● LIVE"
        status_detail = f"Receiving telemetry every {st.session_state.stream_speed:.1f}s"
    elif processor.finished():
        status_text = "✓ COMPLETE"
        status_detail = "All selected telemetry records processed"
    else:
        status_text = "Ⅱ PAUSED"
        status_detail = "Stream stopped — processing position preserved"

    st.markdown(
        "<div class=\"live-strip\">"
        f"<div class=\"live-status\">{status_text}</div>"
        f"<div class=\"small-text\">{status_detail}</div>"
        "</div>",
        unsafe_allow_html=True
    )

    anomaly_count = sum(
        bool(result.get("is_anomaly", False))
        for result in processor.results
    )
    normal_rate = ((processed - anomaly_count) / processed * 100) if processed else 0

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Telemetry", f"{processed:,}", f"of {total:,}")

    with k2:
        st.metric("Anomalies", f"{anomaly_count:,}")

    with k3:
        st.metric("Normal Rate", f"{normal_rate:.1f}%")

    with k4:
        st.metric("ART3", "ACTIVE")


    st.markdown(
        '<div class="section-title">Current Telemetry</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        current_col1, current_col2, current_col3, current_col4 = st.columns(4)

        with current_col1:
            st.caption("Temperature")
            value = latest.get("temperature_c")
            if pd.notna(value):
                st.markdown(f'<div class="big-number">{value:.2f} °C</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="big-number">—</div>', unsafe_allow_html=True)

        with current_col2:
            st.caption("Humidity")
            value = latest.get("humidity_pct")
            if pd.notna(value):
                st.markdown(f'<div class="big-number">{value:.2f} %</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="big-number">—</div>', unsafe_allow_html=True)

        with current_col3:
            st.caption("Pressure")
            value = latest.get("pressure_hpa")
            if pd.notna(value):
                st.markdown(f'<div class="big-number">{value:.2f} hPa</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="big-number">—</div>', unsafe_allow_html=True)

        with current_col4:
            st.caption("Current Status")
            if processor.last_batch and processor.last_batch.get("status") == "ANOMALY":
                st.error("ANOMALY")
            elif processed > 0:
                st.success("NORMAL")
            else:
                st.info("WAITING")

    st.caption(
        f"Simulated time: {simulated_time}   •   Record #{processed:,}   •   Feed batch #{current_batch}"
    )

    batches = list(reversed(processor.batch_history))

    if not batches:

        st.info(
            "No readings have been processed yet. "
            "Press Start Stream to begin simulated live telemetry, or use Process Next Batch for manual testing."
        )

    else:

        feed_container = st.container(
            height=520,
            border=True
        )

        with feed_container:

            visible_batches = batches[
                :st.session_state.visible_blocks
            ]

            for batch in visible_batches:

                with st.container(border=True):

                    latest = batch["latest_reading"]
                    status = batch["status"]

                    heading_col, status_col = (
                        st.columns([4, 1])
                    )

                    with heading_col:

                        st.markdown(
                            f"### Processing Batch "
                            f"{batch['batch_number']}"
                        )

                        st.caption(
                            f"{batch['start_timestamp']}"
                            f" → "
                            f"{batch['end_timestamp']}"
                            f"   •   "
                            f"{batch['readings_processed']} readings"
                        )

                    with status_col:

                        if status == "ANOMALY":

                            st.error(
                                "ANOMALY DETECTED"
                            )

                        else:

                            st.success(
                                "NORMAL"
                            )

                    st.divider()

                    temp_col, humidity_col, pressure_col = (
                        st.columns(3)
                    )

                    with temp_col:

                        st.caption("Temperature")

                        temp = latest.get(
                            "temperature_c"
                        )

                        if pd.notna(temp):

                            st.markdown(
                                f"### {temp:.2f} °C"
                            )

                        else:

                            st.markdown("### —")

                    with humidity_col:

                        st.caption("Humidity")

                        humidity = latest.get(
                            "humidity_pct"
                        )

                        if pd.notna(humidity):

                            st.markdown(
                                f"### {humidity:.2f} %"
                            )

                        else:

                            st.markdown("### —")

                    with pressure_col:

                        st.caption("Pressure")

                        pressure = latest.get(
                            "pressure_hpa"
                        )

                        if pd.notna(pressure):

                            st.markdown(
                                f"### {pressure:.2f} hPa"
                            )

                        else:

                            st.markdown("### —")

                    st.divider()

                    info1, info2, info3 = (
                        st.columns(3)
                    )

                    with info1:

                        st.caption(
                            "Anomalies in Batch"
                        )

                        st.write(
                            f"**{batch['anomaly_count']}**"
                        )

                    with info2:

                        st.caption(
                            "Detection Result"
                        )

                        if batch["anomaly_count"] > 0:

                            st.write(
                                "**Anomaly detected**"
                            )

                        else:

                            st.write(
                                "**No anomaly**"
                            )

                    with info3:

                        st.caption(
                            "Adaptive Model"
                        )

                        if batch["model_refitted"]:

                            st.write(
                                "**Model retrained**"
                            )

                        else:

                            st.write(
                                "**No retraining**"
                            )

                    if batch["anomalies"]:

                        with st.expander(
                            "Detection Details"
                        ):

                            for number, anomaly in enumerate(
                                batch["anomalies"],
                                start=1
                            ):

                                st.markdown(
                                    f"#### Anomaly {number}"
                                )

                                detail1, detail2 = (
                                    st.columns(2)
                                )

                                with detail1:

                                    st.write(
                                        f"**Timestamp:** "
                                        f"{anomaly.get('timestamp', '—')}"
                                    )

                                    # ART3 provides the complete trigger list.
                                    triggers = anomaly.get("triggers", [])

                                    if triggers:
                                        detector_text = " + ".join(triggers)
                                    else:
                                        detector_text = anomaly.get(
                                            "anomaly_type", "Unknown"
                                        )

                                    st.write(
                                        f"**Triggered By:** "
                                        f"{detector_text}"
                                    )

                                    st.write(
                                        f"**Root Cause:** "
                                        f"{anomaly.get('root_cause', 'Not classified')}"
                                    )

                                    st.write(
                                        f"**Affected Sensor:** "
                                        f"{anomaly.get('affected_sensor', '—')}"
                                    )

                                with detail2:

                                    score = anomaly.get(
                                        "iforest_score"
                                    )

                                    zscore = anomaly.get(
                                        "zscore_max"
                                    )

                                    if isinstance(
                                        score,
                                        (int, float)
                                    ):

                                        st.write(
                                            f"**Isolation Forest "
                                            f"Score:** "
                                            f"{score:.4f}"
                                        )

                                    else:

                                        st.write(
                                            "**Isolation Forest "
                                            "Score:** —"
                                        )

                                    if isinstance(
                                        zscore,
                                        (int, float)
                                    ):

                                        st.write(
                                            f"**Maximum Z-Score:** "
                                            f"{zscore:.4f}"
                                        )

                                    else:

                                        st.write(
                                            "**Maximum Z-Score:** —"
                                        )

                                    st.write(
                                        f"**Confidence:** "
                                        f"{anomaly.get('confidence', '—')}"
                                    )

                                    st.write(
                                        f"**Severity:** "
                                        f"{anomaly.get('severity', '—')}"
                                    )

                                if (
                                    number
                                    < len(
                                        batch["anomalies"]
                                    )
                                ):

                                    st.divider()

            if len(batches) > st.session_state.visible_blocks:

                st.markdown(
                    "<div style='text-align:center; "
                    "color:#888; padding:0.2rem 0;'>"
                    "More processed batches are available"
                    "</div>",
                    unsafe_allow_html=True
                )

        has_older_batches = (
            len(batches) > st.session_state.visible_blocks
        )

        if st.session_state.visible_blocks > 1:

            nav_col1, nav_col2 = st.columns(2)

            with nav_col1:

                if has_older_batches:

                    if st.button(
                        "⌄  Show More Batches",
                        use_container_width=True
                    ):

                        st.session_state.visible_blocks += 3
                        st.rerun()

            with nav_col2:

                if st.button(
                    "⌃  Hide Batches",
                    use_container_width=True
                ):

                    st.session_state.visible_blocks = 1
                    st.rerun()

        elif has_older_batches:

            if st.button(
                "⌄  Show More Batches",
                use_container_width=True
            ):

                st.session_state.visible_blocks += 3
                st.rerun()

        else:

            st.caption(
                "All processed batches are visible."
            )

st.markdown(
    '<div class="section-title">'
    'System Intelligence'
    '</div>',
    unsafe_allow_html=True
)

left_panel, right_panel = st.columns(2)


with left_panel:

    with st.container(border=True):

        st.subheader(
            "Last Anomaly"
        )

        if (
            processor is None
            or processor.last_anomaly is None
        ):

            st.info(
                "No anomaly has been detected yet."
            )

        else:

            anomaly = (
                processor.last_anomaly
            )

            st.error(
                "ANOMALY DETECTED"
            )

            st.write(
                f"**Type:** "
                f"{anomaly.get('anomaly_type', 'Unknown')}"
            )

            st.write(
                f"**Timestamp:** "
                f"{anomaly.get('timestamp', '—')}"
            )

            triggers = anomaly.get("triggers", [])

            if triggers:
                detector_text = " + ".join(triggers)
            else:
                detector_text = anomaly.get(
                    "anomaly_type", "Unknown"
                )

            st.write(
                f"**Triggered By:** "
                f"{detector_text}"
            )

            score = anomaly.get(
                "iforest_score"
            )

            zscore = anomaly.get(
                "zscore_max"
            )

            if isinstance(
                score,
                (int, float)
            ):

                st.write(
                    f"**Isolation Forest Score:** "
                    f"{score:.4f}"
                )

            else:

                st.write(
                    "**Isolation Forest Score:** —"
                )

            if isinstance(
                zscore,
                (int, float)
            ):

                st.write(
                    f"**Maximum Z-Score:** "
                    f"{zscore:.4f}"
                )

            else:

                st.write(
                    "**Maximum Z-Score:** —"
                )

            st.markdown("### Root-Cause Diagnosis")

            st.write(
                f"**Root Cause:** "
                f"{anomaly.get('root_cause', 'Not classified')}"
            )

            st.write(
                f"**Affected Sensor:** "
                f"{anomaly.get('affected_sensor', '—')}"
            )

            rc_col1, rc_col2 = st.columns(2)

            with rc_col1:
                st.write(
                    f"**Confidence:** {anomaly.get('confidence', '—')}"
                )

            with rc_col2:
                st.write(
                    f"**Severity:** {anomaly.get('severity', '—')}"
                )

            action = anomaly.get('recommended_action', '—')
            evidence = anomaly.get('diagnostic_evidence', '—')

            st.info(f"**Recommended Action:** {action}")

            with st.expander("Diagnostic Evidence"):
                st.write(evidence)

            try:

                anomaly_timestamp = (
                    pd.to_datetime(
                        anomaly.get(
                            "timestamp"
                        )
                    )
                )

                matching_rows = df[
                    df["timestamp"]
                    == anomaly_timestamp
                ]

                if not matching_rows.empty:

                    anomaly_row = (
                        matching_rows.iloc[-1]
                    )

                    st.divider()

                    temp_col, humidity_col, pressure_col = (
                        st.columns(3)
                    )

                    with temp_col:

                        st.caption(
                            "Temperature"
                        )

                        st.write(
                            f"**"
                            f"{anomaly_row['temperature_c']:.2f}"
                            f" °C**"
                        )

                    with humidity_col:

                        st.caption(
                            "Humidity"
                        )

                        st.write(
                            f"**"
                            f"{anomaly_row['humidity_pct']:.2f}"
                            f" %**"
                        )

                    with pressure_col:

                        st.caption(
                            "Pressure"
                        )

                        st.write(
                            f"**"
                            f"{anomaly_row['pressure_hpa']:.2f}"
                            f" hPa**"
                        )

            except Exception:

                pass

with right_panel:

    with st.container(border=True):

        st.subheader(
            "System / Model Intelligence"
        )

        if processor is None:

            st.info(
                "The model is not initialized yet."
            )

        else:

            processed, total, remaining = (
                processor.progress()
            )

            anomaly_count = sum(
                result.get(
                    "is_anomaly",
                    False
                )

                for result in processor.results
            )

            anomaly_rate = (

                (anomaly_count / processed)
                * 100

                if processed > 0

                else 0
            )

            st.write(
                "**Model:** "
                "ART3 / AdaptiveRTModel"
            )

            st.write(
                "**Isolation Forest:** ACTIVE"
            )

            st.write(
                "**Rolling Z-Score:** ACTIVE"
            )

            st.write(
                "**ART3 Rule Detectors:** ACTIVE"
            )

            st.write(
                "**Root-Cause Engine:** ACTIVE"
            )

            st.write(
                "**Adaptive Learning:** ACTIVE"
            )

            st.divider()

            stat1, stat2 = (
                st.columns(2)
            )

            with stat1:

                st.caption(
                    "Readings Processed"
                )

                st.markdown(
                    f'<div class="big-number">'
                    f'{processed}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with stat2:

                st.caption(
                    "Anomalies Detected"
                )

                st.markdown(
                    f'<div class="big-number">'
                    f'{anomaly_count}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            stat3, stat4 = (
                st.columns(2)
            )

            with stat3:

                st.caption(
                    "Anomaly Rate"
                )

                st.markdown(
                    f'<div class="big-number">'
                    f'{anomaly_rate:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with stat4:

                st.caption(
                    "Current Batch"
                )

                st.markdown(
                    f'<div class="big-number">'
                    f'{getattr(processor, "live_batch_number", processor.batch_number)}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.divider()

            st.write(
                f"**Baseline Size:** "
                f"{processor.baseline_size}"
            )

            st.write(
                f"**Readings Since Refit:** "
                f"{processor.adaptive.count_since_refit}"
            )

            st.write(
                f"**Remaining:** "
                f"{remaining}"
            )

            if processor.last_batch is not None:

                if processor.last_batch[
                    "model_refitted"
                ]:

                    st.success(
                        "Adaptive model was retrained "
                        "during the latest batch."
                    )

                else:

                    st.caption(
                        "No model retraining occurred "
                        "during the latest batch."
                    )


if processor is not None and processed > 0:

    st.markdown(
        '<div class="section-title">Telemetry Trends</div>',
        unsafe_allow_html=True
    )

    chart_start = int(start_index)
    chart_end = chart_start + int(processed)
    chart_df = df.iloc[chart_start:chart_end].copy().tail(100)

    if not chart_df.empty:
        chart_df = chart_df.set_index("timestamp")

        chart_col1, chart_col2, chart_col3 = st.columns(3)

        with chart_col1:
            st.caption("Temperature — last 100 readings")
            st.line_chart(
                chart_df[["temperature_c"]],
                use_container_width=True
            )

        with chart_col2:
            st.caption("Humidity — last 100 readings")
            st.line_chart(
                chart_df[["humidity_pct"]],
                use_container_width=True
            )

        with chart_col3:
            st.caption("Pressure — last 100 readings")
            st.line_chart(
                chart_df[["pressure_hpa"]],
                use_container_width=True
            )


# One simulated telemetry record is processed per cycle.
# Streamlit reruns the script after each record, preserving
# all detector state in st.session_state.

if st.session_state.is_streaming:

    if processor is None:
        st.session_state.is_streaming = False
        st.warning("Start Stream could not initialize the ART3 processor.")
        st.stop()

    if processor.finished():
        st.session_state.is_streaming = False
        st.success("Live telemetry stream completed.")
        st.rerun()

    time.sleep(st.session_state.stream_speed)
    processor.process_next_row()
    st.rerun()
