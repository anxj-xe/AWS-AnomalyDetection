import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from ModelV2.AdaptiveRTModel import (
    baseline_model,
    RollingZScore,
    detect_anomaly,
)

INPUT_FILE = "ModelV2/TestAnomalyUnlabled.csv"
OUTPUT_FILE = "ModelV2/RootCauseResults.csv"

TEMP_SPIKE_THRESHOLD = 1.1677
HUMIDITY_SPIKE_THRESHOLD = 3.4070
PRESSURE_SPIKE_THRESHOLD = 1.5923
FROZEN_CHANGE_THRESHOLD = 0.001
FROZEN_COUNT_THRESHOLD = 5


class RootCauseAnalyzer:
    """Member 3: multivariate reasoning and root-cause diagnosis."""

    def __init__(self):
        self.temperature_frozen_count = 0
        self.humidity_frozen_count = 0
        self.pressure_frozen_count = 0
        self.common_shift_active = False
        self.previous_spike_directions = None

    def diagnose(self, row, anomaly_result, previous_temp, previous_humidity,
                 previous_pressure, previous_timestamp):
        is_anomaly = anomaly_result["is_anomaly"]

        # Communication
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

        # Sensor changes
        if previous_temp is not None:
            temperature_change = row["temperature_c"] - previous_temp
            humidity_change = row["humidity_pct"] - previous_humidity
            pressure_change = row["pressure_hpa"] - previous_pressure
        else:
            temperature_change = 0
            humidity_change = 0
            pressure_change = 0

        # Physical consistency
        a = 17.27
        b = 237.7
        gamma = (
            (a * row["temperature_c"])
            / (b + row["temperature_c"])
            + np.log(row["humidity_pct"] / 100)
        )
        dew_point = (b * gamma) / (a - gamma)
        physically_impossible = dew_point > row["temperature_c"]

        # Weather consistency — intentionally unchanged in Phase 1
        weather_consistent = (
            temperature_change < -0.5
            and humidity_change > 0.2
            and pressure_change < 0
        )

        # Frozen sensors
        if abs(temperature_change) < FROZEN_CHANGE_THRESHOLD:
            self.temperature_frozen_count += 1
        else:
            self.temperature_frozen_count = 0

        if abs(humidity_change) < FROZEN_CHANGE_THRESHOLD:
            self.humidity_frozen_count += 1
        else:
            self.humidity_frozen_count = 0

        if abs(pressure_change) < FROZEN_CHANGE_THRESHOLD:
            self.pressure_frozen_count += 1
        else:
            self.pressure_frozen_count = 0

        temperature_frozen = self.temperature_frozen_count >= FROZEN_COUNT_THRESHOLD
        humidity_frozen = self.humidity_frozen_count >= FROZEN_COUNT_THRESHOLD
        pressure_frozen = self.pressure_frozen_count >= FROZEN_COUNT_THRESHOLD

        # Spikes
        temperature_spike = abs(temperature_change) > TEMP_SPIKE_THRESHOLD
        humidity_spike = abs(humidity_change) > HUMIDITY_SPIKE_THRESHOLD
        pressure_spike = abs(pressure_change) > PRESSURE_SPIKE_THRESHOLD

        spike_count = sum([
            temperature_spike,
            humidity_spike,
            pressure_spike,
        ])

        # Simultaneous shift / reversal
        simultaneous_shift = spike_count == 3

        if simultaneous_shift:
            current_spike_directions = (
                np.sign(temperature_change),
                np.sign(humidity_change),
                np.sign(pressure_change),
            )
        else:
            current_spike_directions = None

        simultaneous_reversal = (
            self.common_shift_active
            and simultaneous_shift
            and self.previous_spike_directions is not None
            and current_spike_directions
            == tuple(-x for x in self.previous_spike_directions)
        )

        if simultaneous_reversal:
            self.common_shift_active = False
        elif simultaneous_shift:
            self.common_shift_active = True

        if simultaneous_shift:
            self.previous_spike_directions = current_spike_directions

        # Root cause — intentionally unchanged in Phase 1
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
            and not self.common_shift_active
            and not simultaneous_shift
            and not simultaneous_reversal
        ):
            root_cause = "Potential genuine weather change"
        elif is_anomaly or spike_count > 0:
            root_cause = "Anomaly detected, cause undetermined"
        else:
            root_cause = "Normal"

        # Confidence — intentionally unchanged in Phase 1
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

        # Severity — intentionally unchanged in Phase 1
        if spike_count > 0:
            max_spike_ratio = max(
                abs(temperature_change) / TEMP_SPIKE_THRESHOLD,
                abs(humidity_change) / HUMIDITY_SPIKE_THRESHOLD,
                abs(pressure_change) / PRESSURE_SPIKE_THRESHOLD,
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

        return {
            "communication_gap": communication_gap,
            "communication_duplicate": communication_duplicate,
            "communication_out_of_order": communication_out_of_order,
            "communication_error": communication_error,
            "temperature_change": temperature_change,
            "humidity_change": humidity_change,
            "pressure_change": pressure_change,
            "dew_point": dew_point,
            "physically_impossible": physically_impossible,
            "weather_consistent": weather_consistent,
            "temperature_frozen": temperature_frozen,
            "humidity_frozen": humidity_frozen,
            "pressure_frozen": pressure_frozen,
            "temperature_spike": temperature_spike,
            "humidity_spike": humidity_spike,
            "pressure_spike": pressure_spike,
            "spike_count": spike_count,
            "simultaneous_reversal": simultaneous_reversal,
            "common_shift_active": self.common_shift_active,
            "simultaneous_shift": simultaneous_shift,
            "root_cause": root_cause,
            "confidence": confidence,
            "severity": severity,
            "max_spike_ratio": round(max_spike_ratio, 2),
        }


def sensor_health_summary(results_df, sensor_name, lookback=200):
    """Phase 1: preserves the existing health calculation unchanged."""
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


# ============================================================
# MAIN PIPELINE
# ============================================================

df = pd.read_csv(INPUT_FILE)
model = baseline_model(df)
roller = RollingZScore()
diagnostics = RootCauseAnalyzer()

print("FIRST TIMESTAMP:", df["timestamp"].iloc[0])
print("LAST TIMESTAMP:", df["timestamp"].iloc[-1])
print("TOTAL ROWS:", len(df))

previous_temp = None
previous_humidity = None
previous_pressure = None
previous_timestamp = None
results = []

for i, row in df.iterrows():
    row = row.to_dict()
    row["timestamp"] = pd.to_datetime(row["timestamp"])

    # -----------------------------
    # MEMBER 2 — ANOMALY DETECTION
    # -----------------------------
    anomaly_result = detect_anomaly(row, model, roller)

    # -----------------------------
    # MEMBER 3 — MULTIVARIATE + ROOT CAUSE
    # -----------------------------
    diagnosis = diagnostics.diagnose(
        row=row,
        anomaly_result=anomaly_result,
        previous_temp=previous_temp,
        previous_humidity=previous_humidity,
        previous_pressure=previous_pressure,
        previous_timestamp=previous_timestamp,
    )

    # Combined output passed to evaluation/dashboard later
    result = {
        **anomaly_result,
        **diagnosis,
    }

    print(result)
    results.append(result)

    previous_temp = row["temperature_c"]
    previous_humidity = row["humidity_pct"]
    previous_pressure = row["pressure_hpa"]
    previous_timestamp = row["timestamp"]


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n--- ROOT CAUSE DISTRIBUTION ---")
print(results_df["root_cause"].value_counts())

print("\n--- CONFIDENCE DISTRIBUTION ---")
print(results_df["confidence"].value_counts())

print("\n--- SEVERITY DISTRIBUTION ---")
print(results_df["severity"].value_counts())

print("\n--- SENSOR HEALTH SUMMARY ---")
for sensor in ["temperature", "humidity", "pressure"]:
    print(sensor_health_summary(results_df, sensor))

  



