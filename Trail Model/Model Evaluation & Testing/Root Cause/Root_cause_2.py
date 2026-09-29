import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")

sys.path.append(PROJECT_DIR)

import pandas as pd
import numpy as np

from ModelV2.AdaptiveRTModel import (
    baseline_model,
    RollingZScore,
    detect_anomaly,
    z_threshold
)

csv_path = os.path.join(PROJECT_DIR, "ModelV2", "TestAnomalyUnlabled.csv")
df = pd.read_csv(csv_path)

window = 30

temp_history = []
humidity_history = []
pressure_history = []

model = baseline_model(df)
roller = RollingZScore()

print("FIRST TIMESTAMP:", df["timestamp"].iloc[0])
print("LAST TIMESTAMP:", df["timestamp"].iloc[-1])
print("TOTAL ROWS:", len(df))


previous_temp = None
previous_humidity = None
previous_pressure = None
previous_timestamp = None

common_shift_active = False
previous_spike_directions = None

temperature_frozen_count = 0
humidity_frozen_count = 0
pressure_frozen_count = 0

# NEW: collect every row's result so we can save + summarize at the end
results = []


for i, row in df.iterrows():

    row = row.to_dict()

    row["timestamp"] = pd.to_datetime(row["timestamp"])

    # --------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------

    result = detect_anomaly(row, model, roller)

    is_anomaly = result["is_anomaly"]

    # --------------------------------------------------
    # UPDATE HISTORY
    # --------------------------------------------------

    temp_history.append(row["temperature_c"])
    humidity_history.append(row["humidity_pct"])
    pressure_history.append(row["pressure_hpa"])

    if len(temp_history) > window:
        temp_history.pop(0)

    if len(humidity_history) > window:
        humidity_history.pop(0)

    if len(pressure_history) > window:
        pressure_history.pop(0)


    # --------------------------------------------------
    # COMMUNICATION CHECK
    # --------------------------------------------------

    if previous_timestamp is not None:
        timestamp_change = row["timestamp"] - previous_timestamp
    else:
        timestamp_change = None

    communication_gap = (
        timestamp_change is not None
        and timestamp_change > pd.Timedelta(minutes=1)
    )

    communication_duplicate = (
        timestamp_change is not None
        and timestamp_change == pd.Timedelta(0)
    )

    communication_out_of_order = (
        timestamp_change is not None
        and timestamp_change < pd.Timedelta(0)
    )

    communication_error = (
        communication_gap
        or communication_duplicate
        or communication_out_of_order
    )

    result["communication_gap"] = communication_gap
    result["communication_duplicate"] = communication_duplicate
    result["communication_out_of_order"] = communication_out_of_order
    result["communication_error"] = communication_error


    # --------------------------------------------------
    # CALCULATE SENSOR CHANGES
    # --------------------------------------------------

    if previous_temp is not None:

        temperature_change = (
            row["temperature_c"] - previous_temp
        )

        humidity_change = (
            row["humidity_pct"] - previous_humidity
        )

        pressure_change = (
            row["pressure_hpa"] - previous_pressure
        )

    else:

        temperature_change = 0
        humidity_change = 0
        pressure_change = 0


    result["temperature_change"] = temperature_change
    result["humidity_change"] = humidity_change
    result["pressure_change"] = pressure_change


    # --------------------------------------------------
    # DEW POINT  (MOVED UP: now computed before root cause,
    # and actually used, instead of being calculated and discarded)
    # --------------------------------------------------

    a = 17.27
    b = 237.7

    gamma = (
        (a * row["temperature_c"])
        / (b + row["temperature_c"])
        + np.log(row["humidity_pct"] / 100)
    )

    dew_point = (b * gamma) / (a - gamma)

    physically_impossible = (
        dew_point > row["temperature_c"]
    )

    result["dew_point"] = dew_point
    result["physically_impossible"] = physically_impossible


    # --------------------------------------------------
    # WEATHER CONSISTENCY
    # (FIXED: pressure is now included, not just temp/humidity —
    # pressure is usually the leading indicator of a real weather shift)
    # --------------------------------------------------

    weather_consistent = (
        temperature_change < -0.5
        and humidity_change > 0.2
        and pressure_change < 0
    )

    result["weather_consistent"] = weather_consistent


    # --------------------------------------------------
    # FROZEN SENSOR DETECTION
    # --------------------------------------------------

    if abs(temperature_change) < 0.001:
        temperature_frozen_count += 1
    else:
        temperature_frozen_count = 0

    if abs(humidity_change) < 0.001:
        humidity_frozen_count += 1
    else:
        humidity_frozen_count = 0

    if abs(pressure_change) < 0.001:
        pressure_frozen_count += 1
    else:
        pressure_frozen_count = 0


    temperature_frozen = temperature_frozen_count >= 5
    humidity_frozen = humidity_frozen_count >= 5
    pressure_frozen = pressure_frozen_count >= 5

    result["temperature_frozen"] = temperature_frozen
    result["humidity_frozen"] = humidity_frozen
    result["pressure_frozen"] = pressure_frozen


    # --------------------------------------------------
    # SPIKE DETECTION
    # --------------------------------------------------

    temperature_spike = abs(temperature_change) > 1.1677
    humidity_spike = abs(humidity_change) > 3.4070
    pressure_spike = abs(pressure_change) > 1.5923

    spike_count = sum([
        temperature_spike,
        humidity_spike,
        pressure_spike
    ])

    result["temperature_spike"] = temperature_spike
    result["humidity_spike"] = humidity_spike
    result["pressure_spike"] = pressure_spike
    result["spike_count"] = spike_count


    # --------------------------------------------------
    # SIMULTANEOUS SHIFT / REVERSAL
    # --------------------------------------------------

    simultaneous_shift = spike_count == 3

    if simultaneous_shift:

        current_spike_directions = (
            np.sign(temperature_change),
            np.sign(humidity_change),
            np.sign(pressure_change)
        )

    else:

        current_spike_directions = None


    simultaneous_reversal = (
        common_shift_active
        and simultaneous_shift
        and previous_spike_directions is not None
        and current_spike_directions
        == tuple(-x for x in previous_spike_directions)
    )


    result["simultaneous_reversal"] = simultaneous_reversal


    if simultaneous_reversal:

        common_shift_active = False

    elif simultaneous_shift:

        common_shift_active = True


    if simultaneous_shift:

        previous_spike_directions = current_spike_directions


    result["common_shift_active"] = common_shift_active
    result["simultaneous_shift"] = simultaneous_shift


    # --------------------------------------------------
    # ROOT CAUSE
    # (physically_impossible now checked FIRST — a physical
    # impossibility should override every other explanation)
    # --------------------------------------------------

    if physically_impossible:

        root_cause = "Physically impossible reading (likely sensor fault)"

    elif communication_error:

        root_cause = "Communication error"

    elif temperature_frozen:

        root_cause = "Temperature sensor frozen"

    elif humidity_frozen:

        root_cause = "Humidity sensor frozen"

    elif pressure_frozen:

        root_cause = "Pressure sensor frozen"

    elif simultaneous_reversal:
      root_cause = "Possible transient glitch or recalibration (rapid reversal detected)"

    elif simultaneous_shift and weather_consistent:

      root_cause = "Potential genuine rapid weather change (multi-sensor)"

    elif simultaneous_shift and not weather_consistent:

      root_cause = "Possible multi-sensor fault (e.g. power fluctuation)"    

    elif spike_count == 1:

        if temperature_spike:
            root_cause = "Temperature sensor anomaly"

        elif humidity_spike:
            root_cause = "Humidity sensor anomaly"

        elif pressure_spike:
            root_cause = "Pressure sensor anomaly"

        else:
            root_cause = "Undetermined"

    elif (
        weather_consistent
        and spike_count >= 2
        and not common_shift_active
        and not simultaneous_shift
        and not simultaneous_reversal
    ):

        root_cause = "Potential genuine weather change"

    elif is_anomaly or spike_count > 0:

       root_cause = "Anomaly detected, cause undetermined"

    else:

       root_cause = "Normal"    



    result["root_cause"] = root_cause


    # --------------------------------------------------
    # NEW: CONFIDENCE SCORE
    # How many independent signals agree that something is wrong.
    # More agreeing signals = higher confidence in the root_cause label.
    # --------------------------------------------------

    confidence_signals = sum([
        physically_impossible,
        communication_error,
        temperature_frozen or humidity_frozen or pressure_frozen,
        spike_count >= 2,
        simultaneous_reversal,
    ])

    if confidence_signals >= 2:
        confidence = "high"
    elif confidence_signals == 1:
        confidence = "medium"
    else:
        confidence = "low"

    result["confidence"] = confidence


    # --------------------------------------------------
    # NEW: SEVERITY
    # How serious the anomaly is, based on magnitude relative
    # to the spike thresholds, not just whether a threshold was crossed.
    # --------------------------------------------------

    if spike_count > 0:
        max_spike_ratio = max(
            abs(temperature_change) / 1.1677,
            abs(humidity_change) / 3.4070,
            abs(pressure_change) / 1.5923,
        )
    else:
        max_spike_ratio = 0

    if physically_impossible or communication_error:
        severity = "critical"
    elif max_spike_ratio > 2 or spike_count >= 2:
        severity = "high"
    elif spike_count == 1:
        severity = "medium"
    else:
        severity = "low"

    result["severity"] = severity
    result["max_spike_ratio"] = round(max_spike_ratio, 2)


    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------

    #print(result)

    # NEW: store this row's full result for later saving/summarizing
    results.append(result)


    # --------------------------------------------------
    # UPDATE PREVIOUS VALUES
    # --------------------------------------------------

    previous_temp = row["temperature_c"]
    previous_humidity = row["humidity_pct"]
    previous_pressure = row["pressure_hpa"]
    previous_timestamp = row["timestamp"]


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)
anomaly_df = results_df[results_df["is_anomaly"]]

print("\n--- ANOMALY RESULTS ---")
print(anomaly_df)

output_path = os.path.join(PROJECT_DIR, "ModelV2", "RootCauseResults.csv")
anomaly_df.to_csv(output_path, index=False)


# ============================================================
# NEW: SENSOR HEALTH + MAINTENANCE SUMMARY
# Aggregates recent history per sensor into a simple health score,
# useful for a dashboard "sensor health" panel.
# ============================================================

def sensor_health_summary(results_df, sensor_name, lookback=200):
    """
    sensor_name should be one of: 'temperature', 'humidity', 'pressure'
    lookback = how many recent rows to consider (default ~200 readings,
    roughly the last few hours at 1-minute intervals — adjust to your data's interval)
    """
    recent = results_df.tail(lookback)

    spike_col = f"{sensor_name}_spike"
    frozen_col = f"{sensor_name}_frozen"

    flagged = recent[spike_col].sum() + recent[frozen_col].sum()
    total = len(recent)
    health_pct = 100 * (1 - flagged / total) if total > 0 else 100

    if health_pct >= 95:
        status = "healthy"
    elif health_pct >= 85:
        status = "needs monitoring"
    else:
        status = "recommend maintenance"

    return {
        "sensor": sensor_name,
        "health_score": round(health_pct, 1),
        "status": status,
        "flagged_readings": int(flagged),
        "readings_checked": total,
    }


print("\n--- SENSOR HEALTH SUMMARY ---")
for sensor in ["temperature", "humidity", "pressure"]:
    print(sensor_health_summary(results_df, sensor))