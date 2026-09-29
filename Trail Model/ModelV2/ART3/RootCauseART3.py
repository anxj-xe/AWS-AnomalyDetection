import os
import sys
import sqlite3
import datetime
import inspect
from collections import deque

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART3_DIR = os.path.join(PROJECT_DIR, "ModelV2", "ART3")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ARTmodel as art3

FEATURES = list(art3.FEATURES)
AdaptiveModel = art3.AdaptiveModel
RollingZScore = art3.RollingZScore

MODEL_NAME = "ART3"
PROJECT_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..")
)

CSV_PATH = os.path.join(PROJECT_DIR, "Model Evaluation & Testing", "AWS_Weather_5000_Clean.csv")
DB_PATH = os.path.join(PROJECT_DIR, "Model Evaluation & Testing", "experiment_log.db")
XLSX_PATH = os.path.join(PROJECT_DIR, "Model Evaluation & Testing", "experiment_log.xlsx")
OUTPUT_CSV = os.path.join(PROJECT_DIR, "Model Evaluation & Testing", "ART3_RootCauseResults.csv")

BASELINE_ROWS = 2000
TEST_ROWS = 3000

TEMP_SPIKE_THRESHOLD = 1.1677
HUMIDITY_SPIKE_THRESHOLD = 3.4070
PRESSURE_SPIKE_THRESHOLD = 1.5923

PHYSICAL_LIMITS = {
    "temperature_c": (-80.0, 60.0),
    "humidity_pct": (0.0, 100.0),
    "pressure_hpa": (800.0, 1100.0),
}

PERSISTENCE_WINDOW = 7
FROZEN_MIN_COUNT = 5
FROZEN_TOLERANCE = 0.001
DRIFT_Z_THRESHOLD = 2.0


def inject_spikes(df, n, rng):
    if n <= 0:
        return df
    n = min(n, len(df))
    indices = rng.choice(len(df), size=n, replace=False)
    for idx in indices:
        df.loc[idx, "temperature_c"] += 10.0 * rng.choice([-1, 1])
        df.loc[idx, "is_anomaly"] = 1
        df.loc[idx, "anomaly_type"] = "SPIKE"
    return df


def inject_frozen(df, n, rng):
    if n <= 0:
        return df
    for _ in range(n):
        start = int(rng.integers(0, max(1, len(df) - 15)))
        length = int(rng.integers(10, 16))
        end = min(start + length, len(df))
        for sensor in FEATURES:
            value = float(df.loc[start, sensor])
            df.loc[start:end - 1, sensor] = value
        df.loc[start:end - 1, "is_anomaly"] = 1
        df.loc[start:end - 1, "anomaly_type"] = "FROZEN"
    return df


def inject_drift(df, n, rng):
    if n <= 0:
        return df
    for _ in range(n):
        start = int(rng.integers(0, max(1, len(df) - 30)))
        end = min(start + 30, len(df))
        length = end - start
        if length < 2:
            continue
        offset = np.linspace(0.0, 8.0, length)
        df.loc[start:end - 1, "temperature_c"] += offset
        df.loc[start:end - 1, "is_anomaly"] = 1
        df.loc[start:end - 1, "anomaly_type"] = "DRIFT"
    return df


def inject_dropout(df, n, rng):
    if n <= 0:
        return df
    n = min(n, len(df))
    indices = rng.choice(len(df), size=n, replace=False)
    for idx in indices:
        choice = int(rng.integers(0, 3))
        if choice == 0:
            df.loc[idx, "temperature_c"] = 78.0
        elif choice == 1:
            df.loc[idx, "humidity_pct"] = 125.0
        else:
            df.loc[idx, "pressure_hpa"] = 1200.0
        df.loc[idx, "is_anomaly"] = 1
        df.loc[idx, "anomaly_type"] = "DROPOUT"
    return df


def get_anomaly_config():
    print("\nSelect anomaly type(s) to inject:")
    print("0 = None")
    print("1 = Spike")
    print("2 = Frozen value")
    print("3 = Gradual drift")
    print("4 = Dropout / noise")
    print("5 = All four")
    print("Example: 1,3 -> Spike + Gradual drift")
    print("         2,4 -> Frozen + Dropout")
    print("         0   -> Clean test data")

    raw = input("\nEnter option number(s): ").strip()

    if raw == "0" or raw == "":
        return {}

    if raw == "5":
        selected = {1, 2, 3, 4}
    else:
        try:
            selected = {int(x.strip()) for x in raw.split(",")}
        except ValueError:
            raise ValueError("Use numbers like 1,2 or 1,3,4.")

        if not selected.issubset({1, 2, 3, 4}):
            raise ValueError("Valid anomaly options are 0, 1, 2, 3, 4 or 5.")

    config = {}

    names = {
        1: "spike",
        2: "frozen",
        3: "drift",
        4: "dropout",
    }

    for option in sorted(selected):
        while True:
            try:
                value = int(input(f"How many '{names[option]}' anomalies/windows? "))
                if value >= 0:
                    config[option] = value
                    break
            except ValueError:
                pass
            print("Enter a non-negative whole number.")

    return config


def inject_selected_anomalies(test_df, config, rng):
    stream = test_df.copy().reset_index(drop=True)
    stream["is_anomaly"] = 0
    stream["anomaly_type"] = "normal"

    if 1 in config:
        inject_spikes(stream, config[1], rng)
    if 2 in config:
        inject_frozen(stream, config[2], rng)
    if 3 in config:
        inject_drift(stream, config[3], rng)
    if 4 in config:
        inject_dropout(stream, config[4], rng)

    return stream


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def calculate_sensor_deltas(row, previous_row):
    if previous_row is None:
        return {sensor: 0.0 for sensor in FEATURES}

    return {
        sensor: safe_float(row[sensor]) - safe_float(previous_row[sensor])
        for sensor in FEATURES
    }


def detect_spikes(deltas):
    return {
        "temperature_c": abs(deltas["temperature_c"]) > TEMP_SPIKE_THRESHOLD,
        "humidity_pct": abs(deltas["humidity_pct"]) > HUMIDITY_SPIKE_THRESHOLD,
        "pressure_hpa": abs(deltas["pressure_hpa"]) > PRESSURE_SPIKE_THRESHOLD,
    }


def update_frozen_counts(deltas, frozen_counts):
    for sensor in FEATURES:
        if abs(deltas[sensor]) <= FROZEN_TOLERANCE:
            frozen_counts[sensor] += 1
        else:
            frozen_counts[sensor] = 0

    return {
        sensor: frozen_counts[sensor] >= FROZEN_MIN_COUNT
        for sensor in FEATURES
    }


def calculate_dew_point(temperature_c, humidity_pct):
    temperature_c = safe_float(temperature_c)
    humidity_pct = safe_float(humidity_pct)

    if not np.isfinite(temperature_c) or not np.isfinite(humidity_pct):
        return np.nan
    if humidity_pct <= 0.0 or humidity_pct > 100.0:
        return np.nan

    a = 17.27
    b = 237.7
    gamma = (a * temperature_c) / (b + temperature_c) + np.log(humidity_pct / 100.0)
    denominator = a - gamma

    if abs(denominator) < 1e-12:
        return np.nan

    return (b * gamma) / denominator


def check_dew_point_consistency(row):
    temperature = safe_float(row.get("temperature_c"))
    humidity = safe_float(row.get("humidity_pct"))
    dew_point = calculate_dew_point(temperature, humidity)

    if not np.isfinite(dew_point):
        return False

    return bool(dew_point > temperature + 0.2 or humidity > 100.0)


def check_communication(previous_timestamp, current_timestamp):
    if previous_timestamp is None:
        return False

    try:
        difference = pd.Timestamp(current_timestamp) - pd.Timestamp(previous_timestamp)
    except Exception:
        return True

    return bool(
        difference > pd.Timedelta(minutes=1)
        or difference <= pd.Timedelta(0)
    )


def detect_persistent_drift(sensor_history, z_scores):
    flags = {sensor: False for sensor in FEATURES}

    for sensor in FEATURES:
        history = sensor_history[sensor]
        if len(history) < PERSISTENCE_WINDOW:
            continue

        z = safe_float(z_scores.get(sensor, 0.0))
        if not np.isfinite(z) or abs(z) < DRIFT_Z_THRESHOLD:
            continue

        values = np.asarray(list(history), dtype=float)
        reference = float(np.median(values[:-1]))
        current = float(values[-1])

        if current > reference:
            same_side = int(np.sum(values > reference))
        elif current < reference:
            same_side = int(np.sum(values < reference))
        else:
            same_side = 0

        flags[sensor] = same_side >= 5

    return flags


def check_physical_limits(row):
    flags = {}

    for sensor, (low, high) in PHYSICAL_LIMITS.items():
        value = safe_float(row.get(sensor))
        flags[sensor] = (
            not np.isfinite(value)
            or value < low
            or value > high
        )

    return flags


def run_art3_detection(row_dict, model, roller):
    """Bridge the current ART3 model API into the RCA pipeline."""
    x = pd.DataFrame(
        [[row_dict[f] for f in FEATURES]],
        columns=FEATURES
    )

    iforest_pred = model.predict(x)[0]
    iforest_score = model.decision_function(x)[0]

    detect_params = inspect.signature(art3.detect_anomaly).parameters

    if len(detect_params) == 4:
        # ART3 version where Isolation Forest prediction/score are
        # calculated by the caller and passed into detect_anomaly().
        return art3.detect_anomaly(
            row_dict,
            bool(iforest_pred == -1),
            float(iforest_score),
            roller
        )

    # Compatibility with the older ART3 API where detect_anomaly()
    # receives the fitted model directly.
    return art3.detect_anomaly(
        row_dict,
        model,
        roller
    )


def identify_sensor(z_scores, spike_flags, drift_flags):
    drifted = [s.replace("_c", "").replace("_pct", "").replace("_hpa", "")
               for s in FEATURES if drift_flags.get(s, False)]
    if drifted:
        return ", ".join(drifted)

    spiked = [s.replace("_c", "").replace("_pct", "").replace("_hpa", "")
              for s in FEATURES if spike_flags.get(s, False)]
    if spiked:
        return ", ".join(spiked)

    valid = {s: abs(v) for s, v in z_scores.items() if np.isfinite(v)}
    if valid:
        dominant = max(valid, key=valid.get)
        return dominant.replace("_c", "").replace("_pct", "").replace("_hpa", "")

    return "station-level"


def classify_root_cause(result, spike_flags, frozen_flags, drift_flags,
                        physical_flags, dew_point_issue, communication_error,
                        deltas, z_scores):
    if not result["is_anomaly"]:
        return "NORMAL"

    if any(physical_flags.values()):
        return "OUT_OF_BOUNDS"

    if dew_point_issue:
        return "PHYSICAL_INCONSISTENCY"

    if communication_error:
        return "DATA_DROPOUT"

    if any(frozen_flags.values()):
        return "STUCK_SENSOR"

    if any(drift_flags.values()):
        return "SENSOR_DRIFT"

    if any(spike_flags.values()):
        return "SPIKE"

    if result["iforest_flag"] or result["z_flag"]:
        return "STATISTICAL_OUTLIER"

    return "UNKNOWN_ANOMALY"


def calculate_confidence(result, root_cause, spike_count, frozen, drift,
                         physical, communication, zscore_max):
    score = 0.0

    if result["iforest_flag"]:
        score += 0.30
    if result["z_flag"]:
        score += 0.25
    if spike_count > 0:
        score += 0.15
    if frozen:
        score += 0.20
    if drift:
        score += 0.20
    if physical:
        score += 0.25
    if communication:
        score += 0.25

    if zscore_max > 5:
        score += 0.15
    elif zscore_max > 3:
        score += 0.08

    if root_cause == "NORMAL":
        return 0.0

    return round(min(1.0, score), 3)


def calculate_severity(result, root_cause, zscore_max, spike_count,
                        physical, communication, frozen, drift):
    if root_cause == "NORMAL":
        return "LOW"

    if physical or communication:
        return "CRITICAL"

    if zscore_max >= 8 or spike_count >= 2:
        return "HIGH"

    if frozen or drift:
        return "HIGH"

    if result["iforest_flag"] and result["z_flag"]:
        return "HIGH"

    return "MEDIUM"


def health_status(score):
    if score >= 95:
        return "EXCELLENT"
    if score >= 85:
        return "GOOD"
    if score >= 70:
        return "DEGRADED"
    return "CRITICAL"


def build_health_summary(history_rows):
    if not history_rows:
        return "Temperature: 100% | Humidity: 100% | Pressure: 100%"

    df = pd.DataFrame(history_rows)

    values = {}
    for sensor in ["temperature", "humidity", "pressure"]:
        spike_col = f"{sensor}_spike"
        frozen_col = f"{sensor}_frozen"
        drift_col = f"{sensor}_drift"

        bad = (
            df[spike_col].astype(int)
            + df[frozen_col].astype(int)
            + df[drift_col].astype(int)
        )

        score = max(0.0, 100.0 * (1.0 - float((bad > 0).sum()) / len(df)))
        values[sensor] = score

    return (
        f"Temperature: {values['temperature']:.1f}% ({health_status(values['temperature'])}) | "
        f"Humidity: {values['humidity']:.1f}% ({health_status(values['humidity'])}) | "
        f"Pressure: {values['pressure']:.1f}% ({health_status(values['pressure'])})"
    )


def recommended_action(root_cause, severity, sensor):
    if root_cause == "STUCK_SENSOR":
        return f"Inspect {sensor} sensor for frozen/stuck output."
    if root_cause == "SENSOR_DRIFT":
        return f"Check {sensor} calibration and recent sensor trend."
    if root_cause == "OUT_OF_BOUNDS":
        return f"Immediately verify {sensor} wiring, units and sensor hardware."
    if root_cause == "DATA_DROPOUT":
        return "Check communication link, timestamp continuity and packet integrity."
    if root_cause == "PHYSICAL_INCONSISTENCY":
        return "Verify humidity/temperature calibration and physical sensor condition."
    if root_cause == "SPIKE":
        return f"Verify {sensor} reading against recent history/reference data."
    if root_cause == "STATISTICAL_OUTLIER":
        return f"Review {sensor} and compare with nearby/reference observations."
    if severity == "CRITICAL":
        return "Prioritize station inspection."
    return "Continue monitoring."


def run_pipeline(baseline_df, stream_df):
    adaptive = AdaptiveModel(baseline_df)
    roller = RollingZScore()

    sensor_history = {sensor: deque(maxlen=PERSISTENCE_WINDOW) for sensor in FEATURES}
    frozen_counts = {sensor: 0 for sensor in FEATURES}

    previous_row = None
    previous_timestamp = None
    results = []
    history_rows = []

    for _, row in stream_df.iterrows():
        row_dict = row.to_dict()

        result = run_art3_detection(
            row_dict,
            adaptive.get_model(),
            roller
        )

        z_scores = result.get("z_scores", {})

        for sensor in FEATURES:
            sensor_history[sensor].append(
                safe_float(row_dict[sensor])
            )

        deltas = calculate_sensor_deltas(row_dict, previous_row)
        spike_flags = detect_spikes(deltas)

        frozen_flags = update_frozen_counts(deltas, frozen_counts)

        drift_flags = detect_persistent_drift(
            sensor_history,
            z_scores
        )

        physical_flags = check_physical_limits(row_dict)

        dew_point = calculate_dew_point(
            row_dict["temperature_c"],
            row_dict["humidity_pct"]
        )

        dew_point_issue = check_dew_point_consistency(row_dict)

        communication_error = check_communication(
            previous_timestamp,
            row_dict["timestamp"]
        )

        root_cause = classify_root_cause(
            result,
            spike_flags,
            frozen_flags,
            drift_flags,
            physical_flags,
            dew_point_issue,
            communication_error,
            deltas,
            z_scores
        )

        spike_count = int(sum(spike_flags.values()))
        zscore_max = float(result.get("zscore_max", 0.0))

        affected_sensor = identify_sensor(
            z_scores,
            spike_flags,
            drift_flags
        )

        confidence = calculate_confidence(
            result,
            root_cause,
            spike_count,
            any(frozen_flags.values()),
            any(drift_flags.values()),
            any(physical_flags.values()) or dew_point_issue,
            communication_error,
            zscore_max
        )

        severity = calculate_severity(
            result,
            root_cause,
            zscore_max,
            spike_count,
            any(physical_flags.values()) or dew_point_issue,
            communication_error,
            any(frozen_flags.values()),
            any(drift_flags.values())
        )

        action = recommended_action(
            root_cause,
            severity,
            affected_sensor
        )

        history_rows.append({
            "temperature_spike": spike_flags["temperature_c"],
            "humidity_spike": spike_flags["humidity_pct"],
            "pressure_spike": spike_flags["pressure_hpa"],
            "temperature_frozen": frozen_flags["temperature_c"],
            "humidity_frozen": frozen_flags["humidity_pct"],
            "pressure_frozen": frozen_flags["pressure_hpa"],
            "temperature_drift": drift_flags["temperature_c"],
            "humidity_drift": drift_flags["humidity_pct"],
            "pressure_drift": drift_flags["pressure_hpa"],
        })

        sensor_health = build_health_summary(history_rows[-200:])

        diagnostic = {
            "timestamp": row_dict["timestamp"],
            "temperature_c": row_dict["temperature_c"],
            "humidity_pct": row_dict["humidity_pct"],
            "pressure_hpa": row_dict["pressure_hpa"],
            "is_anomaly": bool(result["is_anomaly"]),
            "anomaly_name": root_cause,
            "affected_sensor": affected_sensor,
            "confidence_score": confidence,
            "severity": severity,
            "iforest_flag": bool(result["iforest_flag"]),
            "iforest_score": float(result["iforest_score"]),
            "z_flag": bool(result["z_flag"]),
            "zscore_max": zscore_max,
            "temperature_change": round(deltas["temperature_c"], 4),
            "humidity_change": round(deltas["humidity_pct"], 4),
            "pressure_change": round(deltas["pressure_hpa"], 4),
            "temperature_zscore": round(float(z_scores.get("temperature_c", 0.0)), 4),
            "humidity_zscore": round(float(z_scores.get("humidity_pct", 0.0)), 4),
            "pressure_zscore": round(float(z_scores.get("pressure_hpa", 0.0)), 4),
            "temperature_spike": bool(spike_flags["temperature_c"]),
            "humidity_spike": bool(spike_flags["humidity_pct"]),
            "pressure_spike": bool(spike_flags["pressure_hpa"]),
            "temperature_frozen": bool(frozen_flags["temperature_c"]),
            "humidity_frozen": bool(frozen_flags["humidity_pct"]),
            "pressure_frozen": bool(frozen_flags["pressure_hpa"]),
            "temperature_drift": bool(drift_flags["temperature_c"]),
            "humidity_drift": bool(drift_flags["humidity_pct"]),
            "pressure_drift": bool(drift_flags["pressure_hpa"]),
            "physical_error": bool(any(physical_flags.values())),
            "dew_point_inconsistency": bool(dew_point_issue),
            "communication_error": bool(communication_error),
            "sensor_health_summary": sensor_health,
            "recommended_action": action,
        }

        results.append(diagnostic)

        previous_row = row_dict
        previous_timestamp = row_dict["timestamp"]

        if not bool(row_dict.get("is_anomaly", 0)):
            adaptive.addnrefit(row_dict)

    return pd.DataFrame(results)


def evaluate(stream_df, results_df):
    y_true = stream_df["is_anomaly"].astype(int).values
    y_pred = results_df["is_anomaly"].astype(int).values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    accuracy = float(np.mean(y_true == y_pred))

    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS root_cause_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT,
            model_name TEXT,
            baseline_rows INTEGER,
            test_rows INTEGER,
            injected_rows INTEGER,
            flagged_rows INTEGER,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            true_negatives INTEGER,
            false_positives INTEGER,
            false_negatives INTEGER,
            true_positives INTEGER,
            anomaly_breakdown TEXT
        )
    """)

    conn.commit()
    return conn


def save_run(metrics, stream_df, results_df):
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()

    anomaly_breakdown = str(
        stream_df.loc[
            stream_df["is_anomaly"] == 1,
            "anomaly_type"
        ].value_counts().to_dict()
    )

    conn = init_db()

    conn.execute("""
        INSERT INTO root_cause_runs (
            run_timestamp,
            model_name,
            baseline_rows,
            test_rows,
            injected_rows,
            flagged_rows,
            accuracy,
            precision_score,
            recall_score,
            f1_score,
            true_negatives,
            false_positives,
            false_negatives,
            true_positives,
            anomaly_breakdown
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().isoformat(timespec="seconds"),
        MODEL_NAME,
        BASELINE_ROWS,
        TEST_ROWS,
        int(stream_df["is_anomaly"].sum()),
        int(results_df["is_anomaly"].sum()),
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        int(tn),
        int(fp),
        int(fn),
        int(tp),
        anomaly_breakdown
    ))

    conn.commit()

    all_runs = pd.read_sql_query(
        "SELECT * FROM root_cause_runs ORDER BY run_id",
        conn
    )

    all_runs.to_excel(
        XLSX_PATH,
        index=False,
        sheet_name="root_cause_runs"
    )

    conn.close()


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = {"timestamp", *FEATURES}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    if len(df) < BASELINE_ROWS + TEST_ROWS:
        raise ValueError(
            f"Need at least {BASELINE_ROWS + TEST_ROWS} rows, found {len(df)}."
        )

    baseline_df = df.iloc[:BASELINE_ROWS].copy().reset_index(drop=True)
    test_df = df.iloc[BASELINE_ROWS:BASELINE_ROWS + TEST_ROWS].copy().reset_index(drop=True)

    print("\nART3 ROOT CAUSE ANALYSIS")
    print(f"Model: {MODEL_NAME}")
    print(f"CSV: {CSV_PATH}")
    print(f"Clean baseline rows: {len(baseline_df)}")
    print(f"Test/stream rows: {len(test_df)}")

    config = get_anomaly_config()
    rng = np.random.default_rng(42)

    stream_df = inject_selected_anomalies(
        test_df,
        config,
        rng
    )

    injected = int(stream_df["is_anomaly"].sum())

    print(f"\nInjected {injected} anomalous rows into {len(stream_df)} test rows.")

    if injected:
        print(
            "Breakdown:",
            stream_df.loc[
                stream_df["is_anomaly"] == 1,
                "anomaly_type"
            ].value_counts().to_dict()
        )

    print("\nRunning ART3 detection + root-cause analysis...")

    results_df = run_pipeline(
        baseline_df,
        stream_df
    )

    metrics = evaluate(
        stream_df,
        results_df
    )

    anomaly_df = results_df[
        results_df["is_anomaly"] == True
    ].copy()

    print(
        f"\nFlagged {len(anomaly_df)} rows as anomalies "
        f"out of {len(results_df)} total rows."
    )

    if injected:
        caught = int(
            np.sum(
                (stream_df["is_anomaly"].values == 1)
                & (results_df["is_anomaly"].values == 1)
            )
        )

        print(
            f"Caught {caught} out of {injected} injected anomalies "
            f"({caught / injected * 100:.1f}%)."
        )

    print("\nConfusion matrix (rows=actual, cols=predicted) [0, 1]:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 score:  {metrics['f1']:.3f}")

    print("\n--- ROOT CAUSE RESULTS ---")

    display_columns = [
        "timestamp",
        "anomaly_name",
        "affected_sensor",
        "confidence_score",
        "severity",
        "sensor_health_summary",
        "recommended_action"
    ]

    if len(anomaly_df):
        print(
            anomaly_df[display_columns].to_string(
                index=False
            )
        )
    else:
        print("No anomalies detected.")

    anomaly_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    save_run(
        metrics,
        stream_df,
        results_df
    )

    print(f"\nSaved root-cause results to:")
    print(OUTPUT_CSV)

    print(f"\nSQLite log:")
    print(DB_PATH)

    print(f"\nExcel experiment log:")
    print(XLSX_PATH)


if __name__ == "__main__":
    main()