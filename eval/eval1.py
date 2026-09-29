import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from src.data_simulator import AWSDataSimulator
from src.detector import AWSAnomalyDetector

FEATURES = ["temperature", "pressure", "humidity"]   # NOTE: this system uses plain names, no _c/_hpa/_pct suffix

BASELINE_DAYS = 3          # clean days used to train the detector's ML baseline
STREAM_DAYS = 4            # days of clean data used as the test stream, before injection
INTERVAL_MINUTES = 1       # matches the simulator's default step size


# ---------------------------------------------------------------
# Anomaly injection (column names adapted for this system: temperature/pressure/humidity)
# ---------------------------------------------------------------

def inject_spikes(df, n, rng, magnitude=10):
    out = df.copy()
    available = out.index[out["is_anomaly"] == 0]
    chosen = rng.choice(available, size=min(n, len(available)), replace=False)
    for i in chosen:
        out.loc[i, "temperature"] += magnitude * rng.choice([-1, 1])
        out.loc[i, ["is_anomaly", "anomaly_type"]] = [1, "spike"]
    return out


def inject_frozen(df, n, rng, length=12):
    out = df.copy()
    made = 0
    tries = 0
    while made < n and tries < n * 200:
        tries += 1
        start = int(rng.integers(0, max(1, len(out) - length + 1)))
        idx = list(range(start, start + length))
        if idx[-1] >= len(out):
            continue
        if (out.loc[idx, "is_anomaly"] == 0).all():
            for f in FEATURES:
                out.loc[idx, f] = out.loc[start, f]
            out.loc[idx, "is_anomaly"] = 1
            out.loc[idx, "anomaly_type"] = "frozen"
            made += 1
    return out


def inject_drift(df, n, rng, length=30, max_offset=8):
    out = df.copy()
    made = 0
    tries = 0
    while made < n and tries < n * 200:
        tries += 1
        start = int(rng.integers(0, max(1, len(out) - length + 1)))
        idx = list(range(start, start + length))
        if idx[-1] >= len(out):
            continue
        if (out.loc[idx, "is_anomaly"] == 0).all():
            direction = rng.choice([-1, 1])
            offset = np.linspace(0, max_offset * direction, length)
            out.loc[idx, "temperature"] = out.loc[idx, "temperature"].to_numpy() + offset
            out.loc[idx, "is_anomaly"] = 1
            out.loc[idx, "anomaly_type"] = "drift"
            made += 1
    return out


def inject_dropout(df, n, rng):
    out = df.copy()
    available = out.index[out["is_anomaly"] == 0]
    chosen = rng.choice(available, size=min(n, len(available)), replace=False)
    for i in chosen:
        if rng.random() < 0.5:
            out.loc[i, "humidity"] = rng.uniform(0, 5)
        else:
            out.loc[i, "humidity"] = rng.uniform(100, 130)
        out.loc[i, ["is_anomaly", "anomaly_type"]] = [1, "dropout"]
    return out


def choose_anomalies():
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

    while True:
        raw = input("\nEnter option number(s): ").strip()

        if raw == "0":
            return {}

        if raw == "5":
            selected = ["spike", "frozen", "drift", "dropout"]
            break

        try:
            nums = [int(x.strip()) for x in raw.split(",")]
            if len(nums) == len(set(nums)) and nums and all(x in [1, 2, 3, 4] for x in nums):
                names = {1: "spike", 2: "frozen", 3: "drift", 4: "dropout"}
                selected = [names[x] for x in nums]
                break
        except ValueError:
            pass

        print("Invalid choice. Use 0, 1, 2, 3, 4, 5 or combinations such as 1,3.")

    counts = {}
    for name in selected:
        while True:
            try:
                value = int(input(f"How many '{name}' anomalies/windows? "))
                if value >= 0:
                    counts[name] = value
                    break
            except ValueError:
                pass
            print("Enter a whole number >= 0.")

    return counts


def inject_anomalies(df, counts, rng):
    out = df.copy().reset_index(drop=True)
    out["is_anomaly"] = 0
    out["anomaly_type"] = "normal"

    if counts.get("spike", 0) > 0:
        out = inject_spikes(out, counts["spike"], rng)
    if counts.get("frozen", 0) > 0:
        out = inject_frozen(out, counts["frozen"], rng)
    if counts.get("drift", 0) > 0:
        out = inject_drift(out, counts["drift"], rng)
    if counts.get("dropout", 0) > 0:
        out = inject_dropout(out, counts["dropout"], rng)

    return out


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\nConfusion matrix (rows=actual, cols=predicted) [0, 1]:")
    print(cm)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate_by_type(ground_truth_df, y_pred, label_column="anomaly_type"):
    print("\n--- Catch rate by anomaly type ---")
    if label_column not in ground_truth_df.columns:
        print(f"[evaluate_by_type] No '{label_column}' column found - skipping per-type breakdown.")
        return

    for atype in ground_truth_df[label_column].unique():
        if str(atype).lower() in ("normal", "none"):
            continue
        mask = (ground_truth_df[label_column] == atype).to_numpy()
        total = int(mask.sum())
        if total == 0:
            continue
        caught = int(np.sum(mask & (y_pred == 1)))
        pct = (caught / total * 100) if total > 0 else 0.0
        print(f"{str(atype):10s}: caught {caught}/{total} ({pct:.1f}%)")


def report_detector_anomaly_type_breakdown(reports_df):
    """Shows what the DETECTOR itself labeled things as (its own anomaly_type/faulty_sensor),
    separate from the injected ground-truth type. Useful for checking root-cause accuracy,
    e.g. did it call a spike a 'SPIKE' or misclassify it as 'PHYSICAL_INCONSISTENCY'."""
    print("\n--- Detector's own classification breakdown (flagged rows only) ---")
    flagged = reports_df[reports_df["is_anomaly"] == True]
    if len(flagged) == 0:
        print("No anomalies flagged by the detector.")
        return
    print(flagged["anomaly_type"].value_counts().to_string())
    if flagged["faulty_sensor"].notna().any():
        print("\nFaulty sensor breakdown:")
        print(flagged["faulty_sensor"].value_counts(dropna=True).to_string())


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    sim = AWSDataSimulator(seed=42)

    print(f"\nGenerating {BASELINE_DAYS} clean baseline days + {STREAM_DAYS} clean stream days...")
    baseline_df = sim.generate_historical_dataset(days=BASELINE_DAYS, interval_minutes=INTERVAL_MINUTES, inject_anomalies=False)

    # Continue the simulator's clock so the stream picks up where the baseline left off
    stream_sim = AWSDataSimulator(seed=43)
    stream_df_raw = stream_sim.generate_historical_dataset(days=STREAM_DAYS, interval_minutes=INTERVAL_MINUTES, inject_anomalies=False)

    print("\nAWSAnomalyDetector (antigravity MVP) Evaluation")
    print(f"{len(baseline_df)} clean rows = initial training")
    print(f"{len(stream_df_raw)} rows = test/stream")

    counts = choose_anomalies()
    rng = np.random.default_rng()

    stream_df = inject_anomalies(stream_df_raw, counts, rng)
    n_injected = int(stream_df["is_anomaly"].sum())

    print(f"\nInjected {n_injected} anomalous rows into {len(stream_df)} stream rows.")
    if n_injected > 0:
        print("Breakdown:", stream_df.loc[stream_df["is_anomaly"] == 1, "anomaly_type"].value_counts().to_dict())
    else:
        print("No anomalies injected - running clean test.")

    detector = AWSAnomalyDetector()
    print("\nTraining detector's ML baseline on clean historical data...")
    detector.fit(baseline_df[FEATURES])

    print("Running detector over the (possibly anomaly-injected) stream...")
    reports = detector.process_dataframe(stream_df[["timestamp"] + FEATURES])

    reports_df = pd.DataFrame([{
        "timestamp": r.timestamp,
        "is_anomaly": r.is_anomaly,
        "is_weather_event": r.is_weather_event,
        "anomaly_type": r.anomaly_type,
        "faulty_sensor": r.faulty_sensor,
        "confidence": r.confidence,
        "severity": r.severity,
    } for r in reports])

    y_true = stream_df["is_anomaly"].astype(int).to_numpy()
    y_pred = reports_df["is_anomaly"].astype(int).to_numpy()

    caught = int(np.sum((y_true == 1) & (y_pred == 1)))
    flagged = int(y_pred.sum())
    catch_pct = (caught / n_injected * 100) if n_injected > 0 else 0.0

    if n_injected > 0:
        print(f"\nCaught {caught} out of {n_injected} injected anomalies ({catch_pct:.1f}%)")
    print(f"Flagged {flagged} rows as anomalies out of {len(y_pred)} total rows.")

    evaluate(y_true, y_pred)
    evaluate_by_type(stream_df, y_pred)
    report_detector_anomaly_type_breakdown(reports_df)


if __name__ == "__main__":
    main()