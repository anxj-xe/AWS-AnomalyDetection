import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ModelV2.AdaptiveRTModel import AdaptiveModel, RollingZScore, detect_anomaly
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ---------------------------------------------------------------
# Injecting Labeled Synthetic Anomalies
# ---------------------------------------------------------------

def inject_spikes(df, n_spikes=10, spike_magnitude=10, rng=None):
    df_with_spikes = df.copy()
    available = df_with_spikes[df_with_spikes["is_anomaly"] == 0].index
    n_spikes = min(n_spikes, len(available))

    if n_spikes == 0:
        return df_with_spikes

    spike_indices = rng.choice(available, size=n_spikes, replace=False)

    for idx in spike_indices:
        df_with_spikes.loc[idx, "temperature_c"] += spike_magnitude * rng.choice([-1, 1])
        df_with_spikes.loc[idx, "is_anomaly"] = 1
        df_with_spikes.loc[idx, "anomaly_type"] = "spike"

    return df_with_spikes


def inject_frozen(df, n_frozen=5, freeze_len=12, rng=None):
    df_frozen = df.copy()
    n_rows = len(df_frozen)
    placed = 0
    attempts = 0
    max_attempts = n_frozen * 50

    while placed < n_frozen and attempts < max_attempts and n_rows > freeze_len:
        attempts += 1
        start = int(rng.integers(0, n_rows - freeze_len + 1))
        window = list(range(start, start + freeze_len))

        if (df_frozen.loc[window, "is_anomaly"] == 0).all():
            frozen_temp = df_frozen.loc[start, "temperature_c"]
            frozen_hum = df_frozen.loc[start, "humidity_pct"]
            frozen_press = df_frozen.loc[start, "pressure_hpa"]

            df_frozen.loc[window, "temperature_c"] = frozen_temp
            df_frozen.loc[window, "humidity_pct"] = frozen_hum
            df_frozen.loc[window, "pressure_hpa"] = frozen_press
            df_frozen.loc[window, "is_anomaly"] = 1
            df_frozen.loc[window, "anomaly_type"] = "frozen"
            placed += 1

    if placed < n_frozen:
        print(f"[inject_frozen] Warning: could only place {placed}/{n_frozen} frozen windows.")

    return df_frozen


def inject_drift(df, n_drift=5, drift_len=30, max_offset=8, rng=None):
    df_drift = df.copy()
    n_rows = len(df_drift)
    placed = 0
    attempts = 0
    max_attempts = n_drift * 50

    while placed < n_drift and attempts < max_attempts and n_rows > drift_len:
        attempts += 1
        start = int(rng.integers(0, n_rows - drift_len + 1))
        window = list(range(start, start + drift_len))

        if (df_drift.loc[window, "is_anomaly"] == 0).all():
            direction = rng.choice([-1, 1])
            ramp = np.linspace(0, max_offset * direction, drift_len)
            df_drift.loc[window, "temperature_c"] = df_drift.loc[window, "temperature_c"].values + ramp
            df_drift.loc[window, "is_anomaly"] = 1
            df_drift.loc[window, "anomaly_type"] = "drift"
            placed += 1

    if placed < n_drift:
        print(f"[inject_drift] Warning: could only place {placed}/{n_drift} drift windows.")

    return df_drift


def inject_dropout(df, n_dropout=8, rng=None):
    df_dropout = df.copy()
    available = df_dropout[df_dropout["is_anomaly"] == 0].index
    n_dropout = min(n_dropout, len(available))

    if n_dropout == 0:
        return df_dropout

    idxs = rng.choice(available, size=n_dropout, replace=False)

    for idx in idxs:
        if rng.random() > 0.5:
            if rng.random() > 0.5:
                df_dropout.loc[idx, "humidity_pct"] = rng.uniform(0, 5)
            else:
                df_dropout.loc[idx, "humidity_pct"] = rng.uniform(100, 130)
        else:
            df_dropout.loc[idx, "pressure_hpa"] = rng.uniform(850, 900)

        df_dropout.loc[idx, "is_anomaly"] = 1
        df_dropout.loc[idx, "anomaly_type"] = "dropout"

    return df_dropout


# ---------------------------------------------------------------
# User-driven anomaly selection
# ---------------------------------------------------------------

def get_user_anomaly_config():
    print("\nSelect anomaly type(s) to inject:")
    print("0 = None")
    print("1 = Spike")
    print("2 = Frozen value")
    print("3 = Gradual drift")
    print("4 = Dropout / noise")
    print("5 = All four")
    print("Example: 1,3  -> Spike + Gradual drift")
    print("         2,4  -> Frozen + Dropout")
    print("         0     -> Clean test data")

    while True:
        raw = input("\nEnter option number(s): ").strip()

        if raw == "0":
            return {}

        if raw == "5":
            chosen = ["spike", "frozen", "drift", "dropout"]
            break

        try:
            numbers = [int(x.strip()) for x in raw.split(",")]

            if numbers and all(x in [1, 2, 3, 4] for x in numbers) and len(numbers) == len(set(numbers)):
                chosen_map = {1: "spike", 2: "frozen", 3: "drift", 4: "dropout"}
                chosen = [chosen_map[x] for x in numbers]
                break
        except ValueError:
            pass

        print("Invalid choice. Enter 0, 1, 2, 3, 4, 5 or combinations such as 1,3.")

    config = {}

    for anomaly_type in chosen:
        while True:
            qty_raw = input(f"How many '{anomaly_type}' anomalies/windows? ").strip()

            try:
                qty = int(qty_raw)

                if qty > 0:
                    config[anomaly_type] = qty
                    break
            except ValueError:
                pass

            print("Please enter a positive whole number.")

    return config


def apply_anomaly_config(df, config, rng):
    df_out = df.copy()
    df_out["is_anomaly"] = 0
    df_out["anomaly_type"] = "normal"

    if not config:
        print("No anomalies selected - running evaluation on CLEAN stream.")
        return df_out

    if "spike" in config:
        df_out = inject_spikes(df_out, config["spike"], rng=rng)

    if "frozen" in config:
        df_out = inject_frozen(df_out, config["frozen"], rng=rng)

    if "drift" in config:
        df_out = inject_drift(df_out, config["drift"], rng=rng)

    if "dropout" in config:
        df_out = inject_dropout(df_out, config["dropout"], rng=rng)

    return df_out


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\nConfusion matrix (rows=actual, cols=predicted) [0, 1]:")
    print(cm)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "AWS_Weather_5000_Clean.csv")

    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # First 1000 rows = clean baseline
    # Remaining 4000 rows = test stream
    baseline_df = df.iloc[:1000].copy().reset_index(drop=True)
    stream_df_raw = df.iloc[1000:5000].copy().reset_index(drop=True)

    rng = np.random.default_rng(42)

    config = get_user_anomaly_config()
    stream_df = apply_anomaly_config(stream_df_raw, config, rng)

    n_injected = int(stream_df["is_anomaly"].sum())

    print(f"\nInjected {n_injected} anomalous rows into {len(stream_df)} stream rows.")

    if n_injected > 0:
        print("Breakdown:", stream_df[stream_df["is_anomaly"] == 1]["anomaly_type"].value_counts().to_dict())

    # Build the original AdaptiveRTModel on clean baseline only
    adaptive = AdaptiveModel(baseline_df)
    roller = RollingZScore()

    # Same real-time processing pattern as Result1.py
    results = []

    for _, row in stream_df.iterrows():
        row_dict = row.to_dict()

        result = detect_anomaly(row_dict, adaptive.get_model(), roller)
        results.append(result)

        adaptive.addnrefit(row_dict)

    results_df = pd.DataFrame(results)

    # Compare model predictions against injected ground truth
    y_true = stream_df["is_anomaly"].astype(int).values
    y_pred = results_df["is_anomaly"].astype(int).values

    caught = int(np.sum((y_true == 1) & (y_pred == 1)))
    total_injected = int(np.sum(y_true))
    catch_pct = (caught / total_injected * 100) if total_injected > 0 else 0.0

    if total_injected > 0:
        print(f"\nCaught {caught} out of {total_injected} injected anomalies ({catch_pct:.1f}%)")

    print(f"Flagged {int(y_pred.sum())} rows as anomalies out of {len(y_pred)} total rows.")

    evaluate(y_true, y_pred)
