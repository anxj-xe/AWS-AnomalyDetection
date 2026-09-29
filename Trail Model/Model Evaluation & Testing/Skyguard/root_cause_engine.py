from collections import deque
import numpy as np
import pandas as pd

FEATURES=["temperature_c","humidity_pct","pressure_hpa"]
Z_THRESHOLD=3.0
TEMP_SPIKE_THRESHOLD=1.1677
HUMIDITY_SPIKE_THRESHOLD=3.4070
PRESSURE_SPIKE_THRESHOLD=1.5923
PERSISTENCE_WINDOW=7
PERSISTENCE_MIN_COUNT=5
DRIFT_Z_THRESHOLD=2.0
PHYSICAL_LIMITS={"temperature_c":(-80.0,60.0),"humidity_pct":(0.0,100.0),"pressure_hpa":(800.0,1100.0)}

def safe_float(value):
    """Convert a value to float, returning NaN if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def calculate_dew_point(temperature_c, humidity_pct):
    """
    Magnus dew-point approximation.

    Returns NaN when RH/temperature cannot be used safely.
    """
    temperature_c = safe_float(temperature_c)
    humidity_pct = safe_float(humidity_pct)

    if not np.isfinite(temperature_c) or not np.isfinite(humidity_pct):
        return np.nan

    if humidity_pct <= 0.0 or humidity_pct > 100.0:
        return np.nan

    a = 17.27
    b = 237.7

    gamma = (
        (a * temperature_c) / (b + temperature_c)
        + np.log(humidity_pct / 100.0)
    )

    denominator = a - gamma
    if abs(denominator) < 1e-12:
        return np.nan

    return (b * gamma) / denominator

def calculate_sensor_deltas(row, previous_row):
    """Calculate one-reading changes for the three AWS sensors."""
    if previous_row is None:
        return {f: 0.0 for f in FEATURES}

    return {
        "temperature_c": safe_float(row["temperature_c"])
        - safe_float(previous_row["temperature_c"]),
        "humidity_pct": safe_float(row["humidity_pct"])
        - safe_float(previous_row["humidity_pct"]),
        "pressure_hpa": safe_float(row["pressure_hpa"])
        - safe_float(previous_row["pressure_hpa"]),
    }

def detect_spikes(deltas):
    """
    Detect rapid one-reading changes using the thresholds from the
    original Root_cause1.py.
    """
    flags = {
        "temperature_c": abs(deltas["temperature_c"]) > TEMP_SPIKE_THRESHOLD,
        "humidity_pct": abs(deltas["humidity_pct"]) > HUMIDITY_SPIKE_THRESHOLD,
        "pressure_hpa": abs(deltas["pressure_hpa"]) > PRESSURE_SPIKE_THRESHOLD,
    }

    return flags, int(sum(flags.values()))

def check_physical_limits(row):
    """
    Check broad physical validity of each raw sensor value.
    """
    flags = {}

    for sensor, (low, high) in PHYSICAL_LIMITS.items():
        value = safe_float(row[sensor])
        flags[sensor] = (
            not np.isfinite(value)
            or value < low
            or value > high
        )

    return flags

def check_dew_point_consistency(row, dew_point):
    """
    RH <= 100% should imply dew point <= air temperature.
    This is a strong physical consistency check for the supplied variables.
    """
    rh = safe_float(row["humidity_pct"])
    temp = safe_float(row["temperature_c"])

    if not np.isfinite(dew_point) or not np.isfinite(rh):
        return False

    return rh > 100.0 or dew_point > temp + 0.2

def check_communication(previous_timestamp, current_timestamp):
    """
    Detect missing interval, duplicate timestamp, or out-of-order timestamp.
    The original code treats a gap > 1 minute as an error.
    """
    if previous_timestamp is None:
        return {
            "communication_gap": False,
            "communication_duplicate": False,
            "communication_out_of_order": False,
            "communication_error": False,
        }

    change = current_timestamp - previous_timestamp

    gap = change > pd.Timedelta(minutes=1)
    duplicate = change == pd.Timedelta(0)
    out_of_order = change < pd.Timedelta(0)

    return {
        "communication_gap": bool(gap),
        "communication_duplicate": bool(duplicate),
        "communication_out_of_order": bool(out_of_order),
        "communication_error": bool(gap or duplicate or out_of_order),
    }

def update_frozen_counts(deltas, frozen_counts):
    """
    A sensor is considered frozen after 5 consecutive readings with
    effectively no change, matching the original code.
    """
    for sensor in FEATURES:
        if abs(deltas[sensor]) < 0.001:
            frozen_counts[sensor] += 1
        else:
            frozen_counts[sensor] = 0

    flags = {
        "temperature_frozen": frozen_counts["temperature_c"] >= 5,
        "humidity_frozen": frozen_counts["humidity_pct"] >= 5,
        "pressure_frozen": frozen_counts["pressure_hpa"] >= 5,
    }

    return flags

def identify_dominant_sensor(z_scores):
    """
    Identify which sensor contributes the strongest standardized deviation.
    """
    valid = {
        sensor: abs(score)
        for sensor, score in z_scores.items()
        if np.isfinite(score)
    }

    if not valid:
        return "none"

    return max(valid, key=valid.get)

def detect_persistent_drift(sensor_history, z_scores):
    """
    Detect a persistent abnormal level/drift rather than a single spike.

    Logic:
      - current |z| >= 2
      - at least 5 of the latest 7 available readings are on the same
        side of the rolling reference (above or below)
      - requires enough history to avoid over-interpreting startup readings

    This is a diagnostic signal, not proof of hardware failure.
    """
    result = {
        "temperature_drift": False,
        "humidity_drift": False,
        "pressure_drift": False,
    }

    for sensor in FEATURES:
        history = sensor_history[sensor]

        if len(history) < PERSISTENCE_WINDOW:
            continue

        current_z = z_scores.get(sensor, 0.0)
        if not np.isfinite(current_z) or abs(current_z) < DRIFT_Z_THRESHOLD:
            continue

        recent = np.asarray(list(history)[-PERSISTENCE_WINDOW:], dtype=float)
        if len(recent) < PERSISTENCE_WINDOW:
            continue

        # Use the median of the preceding readings as a robust local reference.
        reference = float(np.median(recent[:-1]))
        current = float(recent[-1])

        if current > reference:
            same_side = sum(value > reference for value in recent)
            result[f"{sensor.split('_')[0]}_drift"] = (
                same_side >= PERSISTENCE_MIN_COUNT
            )
        elif current < reference:
            same_side = sum(value < reference for value in recent)
            result[f"{sensor.split('_')[0]}_drift"] = (
                same_side >= PERSISTENCE_MIN_COUNT
            )

    return result

def detect_common_mode_shift(deltas, spike_flags):
    """
    A common-mode shift occurs when all three sensors cross their
    one-reading spike thresholds in the same interval.
    """
    simultaneous_shift = sum(spike_flags.values()) == 3

    if not simultaneous_shift:
        return {
            "simultaneous_shift": False,
            "simultaneous_reversal": False,
            "spike_directions": None,
        }

    directions = tuple(
        int(np.sign(deltas[sensor])) for sensor in FEATURES
    )

    return {
        "simultaneous_shift": True,
        "simultaneous_reversal": False,
        "spike_directions": directions,
    }

def check_weather_consistency(deltas):
    """
    Conservative environmental consistency check.

    A falling temperature + rising RH + falling pressure is retained from
    the original code because it can represent a coherent weather transition.

    We also allow the inverse temperature/RH relationship as a supporting
    environmental signal, but do not call it a confirmed weather event.
    """
    dt = deltas["temperature_c"]
    dh = deltas["humidity_pct"]
    dp = deltas["pressure_hpa"]

    # Retain the original conservative pattern from Root_cause1.py.
    # We do NOT automatically call warming + drying + rising pressure
    # a weather event because, at one-minute resolution, that pattern
    # can also be caused by a station/sensor disturbance.
    original_pattern = dt < -0.5 and dh > 0.2 and dp < 0

    return bool(original_pattern)

def detect_sensor_exposure_issue(deltas, z_scores, spike_flags):
    """
    Temperature-only abrupt changes with stable pressure and no corresponding
    humidity response can occur because of sensor exposure, radiation-shield,
    aspiration/ventilation, or placement problems.

    This is intentionally reported as 'possible', not confirmed.
    """
    temp_z = abs(z_scores.get("temperature_c", 0.0))
    hum_z = abs(z_scores.get("humidity_pct", 0.0))
    pressure_z = abs(z_scores.get("pressure_hpa", 0.0))

    temp_spike = spike_flags["temperature_c"]

    return bool(
        temp_spike
        and temp_z >= 3.0
        and hum_z < 2.0
        and pressure_z < 2.0
        and abs(deltas["pressure_hpa"]) < PRESSURE_SPIKE_THRESHOLD
    )

def detect_humidity_condensation_issue(row, dew_point, z_scores, spike_flags):
    """
    High RH close to saturation can indicate condensation/wetting effects
    on a humidity sensor. This is a diagnostic possibility only.
    """
    rh = safe_float(row["humidity_pct"])
    temp = safe_float(row["temperature_c"])

    if not np.isfinite(rh) or not np.isfinite(temp) or not np.isfinite(dew_point):
        return False

    rh_high = rh >= 98.0
    near_dew_point = abs(temp - dew_point) <= 0.3
    humidity_abnormal = (
        abs(z_scores.get("humidity_pct", 0.0)) >= 3.0
        or spike_flags["humidity_pct"]
    )

    return bool(rh_high and near_dew_point and humidity_abnormal)

def classify_statistical_outlier(
    is_anomaly,
    iforest_flag,
    z_flag,
    z_scores,
    spike_count,
):
    """
    Give a useful label when the statistical detector finds an anomaly
    but the physical/rule-based tests do not identify a specific mechanism.
    """
    if not is_anomaly:
        return None

    if spike_count > 0:
        return None

    dominant = identify_dominant_sensor(z_scores)
    dominant_z = abs(z_scores.get(dominant, 0.0))

    if z_flag and dominant != "none":
        sensor_name = {
            "temperature_c": "temperature",
            "humidity_pct": "humidity",
            "pressure_hpa": "pressure",
        }[dominant]

        if iforest_flag:
            return (
                f"Statistical outlier with strongest {sensor_name} deviation "
                f"(possible {sensor_name} sensor drift/local environmental change)"
            )

        return (
            f"Unusual {sensor_name} reading relative to recent history "
            f"(possible sensor drift or local environmental change)"
        )

    if iforest_flag:
        return (
            "Multivariate statistical outlier "
            "(unusual combination of temperature, humidity and pressure)"
        )

    return "Local statistical anomaly in recent sensor history"

def calculate_confidence(
    physical_issue,
    communication_error,
    frozen_issue,
    spike_count,
    simultaneous_shift,
    persistent_drift,
    weather_consistent,
    iforest_flag,
    z_flag,
):
    """
    Confidence is based on independent evidence, not just the anomaly flag.
    """
    signals = [
        physical_issue,
        communication_error,
        frozen_issue,
        spike_count >= 1,
        simultaneous_shift,
        persistent_drift,
        weather_consistent,
        iforest_flag,
        z_flag,
    ]

    count = int(sum(bool(x) for x in signals))

    if count >= 3:
        return "high"
    if count >= 2:
        return "medium"
    return "low"

def calculate_severity(
    physical_issue,
    communication_error,
    max_spike_ratio,
    simultaneous_shift,
    persistent_drift,
    iforest_flag,
    z_flag,
):
    """
    Severity reflects operational seriousness, while confidence answers
    how strongly the evidence supports the diagnosis.
    """
    if physical_issue or communication_error:
        return "critical"

    if simultaneous_shift or max_spike_ratio > 2.0:
        return "high"

    if persistent_drift:
        return "high"

    if max_spike_ratio > 1.0:
        return "medium"

    if iforest_flag and z_flag:
        return "medium"

    if iforest_flag or z_flag:
        return "low"

    return "low"

def make_root_cause(
    is_anomaly,
    z_scores,
    spike_flags,
    spike_count,
    deltas,
    physical_flags,
    dew_point_issue,
    communication,
    frozen_flags,
    drift_flags,
    simultaneous_shift,
    simultaneous_reversal,
    weather_consistent,
    exposure_issue,
    condensation_issue,
    iforest_flag,
    z_flag,
):
    """
    Hierarchical root-cause diagnosis.

    The order matters: hard physical/data-integrity failures get priority
    over softer statistical explanations.
    """
    physical_issue = any(physical_flags.values()) or dew_point_issue

    if physical_issue:
        return (
            "Physically invalid/implausible reading "
            "(check sensor calibration, units or data integrity)"
        )

    if communication["communication_error"]:
        if communication["communication_gap"]:
            return "Communication/data transmission gap"
        if communication["communication_duplicate"]:
            return "Duplicate timestamp/data packet"
        if communication["communication_out_of_order"]:
            return "Out-of-order timestamp/data packet"

    if frozen_flags["temperature_frozen"]:
        return "Temperature sensor frozen/stuck"

    if frozen_flags["humidity_frozen"]:
        return "Humidity sensor frozen/stuck"

    if frozen_flags["pressure_frozen"]:
        return "Pressure sensor frozen/stuck"

    if simultaneous_reversal:
        return (
            "Possible transient multi-sensor glitch or recalibration "
            "(rapid reversal detected)"
        )

    if simultaneous_shift:
        if weather_consistent:
            return (
                "Possible genuine rapid weather transition "
                "(multi-sensor change is physically coherent)"
            )

        return (
            "Possible common-mode/multi-sensor fault "
            "(power fluctuation, electronics or station-level disturbance)"
        )

    if condensation_issue:
        return (
            "Possible humidity sensor wetting/condensation or saturation "
            "(inspect radiation shield and humidity probe)"
        )

    if exposure_issue:
        return (
            "Possible temperature sensor exposure/ventilation issue "
            "(e.g. radiation shield, aspiration or sensor placement)"
        )

    # Single-sensor rapid change.
    if spike_count == 1:
        if spike_flags["temperature_c"]:
            return "Probable temperature sensor anomaly or local thermal exposure"
        if spike_flags["humidity_pct"]:
            return "Probable humidity sensor anomaly or abrupt moisture change"
        if spike_flags["pressure_hpa"]:
            return "Probable pressure sensor anomaly or local pressure disturbance"

    # Persistent/drifting sensor.
    if drift_flags["temperature_drift"]:
        return "Possible temperature sensor drift/calibration offset"

    if drift_flags["humidity_drift"]:
        return "Possible humidity sensor drift/calibration offset"

    if drift_flags["pressure_drift"]:
        return "Possible pressure sensor drift/calibration offset"

    # Coherent environmental change.
    if weather_consistent and spike_count >= 2:
        return (
            "Possible genuine weather change "
            "(temperature/humidity/pressure pattern is coherent)"
        )

    # Statistical detector explanation.
    statistical_cause = classify_statistical_outlier(
        is_anomaly=is_anomaly,
        iforest_flag=iforest_flag,
        z_flag=z_flag,
        z_scores=z_scores,
        spike_count=spike_count,
    )

    if statistical_cause is not None:
        return statistical_cause

    # Final useful fallback. This should be rare, but is intentionally not
    # called 'undetermined'.
    if is_anomaly:
        return (
            "Possible sensor/data-quality anomaly "
            "(inspect affected sensor and recent station history)"
        )

    return "Normal"

def recommended_action(root_cause, affected_sensor, severity):
    """
    Convert diagnosis into an operational action for the department.
    """
    text = root_cause.lower()

    if "communication" in text or "timestamp" in text or "packet" in text:
        return "Check logger, telemetry link, timestamp synchronization and data transmission."

    if "frozen" in text or "stuck" in text:
        return f"Inspect {affected_sensor} sensor for stuck output, wiring, power and calibration."

    if "temperature" in text and ("sensor" in text or "exposure" in text):
        return "Inspect temperature probe, radiation shield, aspiration/ventilation, placement and calibration."

    if "humidity" in text:
        return "Inspect humidity probe, radiation shield, condensation/wetting, wiring and calibration."

    if "pressure" in text:
        return "Inspect barometer/pressure port, tubing/vent, blockage, wiring and calibration."

    if "multi-sensor" in text or "common-mode" in text:
        return "Check AWS power supply, logger, grounding, electronics and simultaneous sensor behavior."

    if "weather" in text or "environmental" in text:
        return "Cross-check nearby AWS/station observations and meteorological conditions before maintenance."

    if "statistical" in text or "outlier" in text or "data-quality" in text:
        return "Review affected sensor against recent history and neighboring/reference observations; inspect if repeated."

    if "physical" in text or "implausible" in text:
        return "Verify sensor units, calibration, wiring and data integrity immediately."

    if severity == "critical":
        return "Prioritize station inspection and verify the reading against an independent/reference source."

    return "Review sensor history and nearby/reference observations."

def build_diagnostic_evidence(
    z_scores,
    deltas,
    spike_flags,
    communication,
    frozen_flags,
    drift_flags,
    simultaneous_shift,
    weather_consistent,
    exposure_issue,
    condensation_issue,
    iforest_flag,
    z_flag,
):
    """
    Produce a human-readable evidence string so the department can see
    why the root-cause label was assigned.
    """
    evidence = []

    if iforest_flag:
        evidence.append("Isolation Forest flagged the multivariate reading")

    if z_flag:
        dominant = identify_dominant_sensor(z_scores)
        dominant_z = abs(z_scores.get(dominant, 0.0))
        if dominant != "none":
            evidence.append(
                f"{dominant.split('_')[0]} rolling z-score={dominant_z:.2f} > {Z_THRESHOLD:.1f}"
            )

    spike_sensors = [
        sensor.split("_")[0]
        for sensor in FEATURES
        if spike_flags[sensor]
    ]
    if spike_sensors:
        evidence.append(
            "rapid change in " + ", ".join(spike_sensors)
        )

    if simultaneous_shift:
        evidence.append("all three sensors crossed one-reading spike thresholds")

    if communication["communication_error"]:
        evidence.append("timestamp/data transmission irregularity")

    if frozen_flags["temperature_frozen"]:
        evidence.append("temperature unchanged for at least 5 consecutive intervals")
    if frozen_flags["humidity_frozen"]:
        evidence.append("humidity unchanged for at least 5 consecutive intervals")
    if frozen_flags["pressure_frozen"]:
        evidence.append("pressure unchanged for at least 5 consecutive intervals")

    if drift_flags["temperature_drift"]:
        evidence.append("persistent temperature deviation")
    if drift_flags["humidity_drift"]:
        evidence.append("persistent humidity deviation")
    if drift_flags["pressure_drift"]:
        evidence.append("persistent pressure deviation")

    if weather_consistent:
        evidence.append("temperature/humidity/pressure changes follow a coherent weather-transition pattern")

    if exposure_issue:
        evidence.append("temperature changed sharply while pressure/humidity remained comparatively stable")

    if condensation_issue:
        evidence.append("RH near saturation and temperature near calculated dew point")

    if not evidence:
        evidence.append(
            "anomaly detected from the multivariate statistical model without a stronger physical signature"
        )

    return "; ".join(evidence)

def determine_affected_sensor(
    z_scores,
    spike_flags,
    simultaneous_shift,
    drift_flags,
    weather_consistent,
):
    """
    Identify the sensor(s) that should be investigated first.
    """
    if simultaneous_shift:
        return "temperature, humidity, pressure"

    drifted = [
        sensor.split("_")[0]
        for sensor in FEATURES
        if drift_flags.get(f"{sensor.split('_')[0]}_drift", False)
    ]
    if drifted:
        return ", ".join(drifted)

    spiked = [
        sensor.split("_")[0]
        for sensor in FEATURES
        if spike_flags[sensor]
    ]
    if spiked:
        return ", ".join(spiked)

    dominant = identify_dominant_sensor(z_scores)
    if dominant != "none":
        return dominant.split("_")[0]

    if weather_consistent:
        return "multiple sensors"

    return "station-level / unknown"

def max_spike_ratio(deltas):
    """
    Magnitude relative to the original spike thresholds.
    """
    ratios = [
        abs(deltas["temperature_c"]) / TEMP_SPIKE_THRESHOLD,
        abs(deltas["humidity_pct"]) / HUMIDITY_SPIKE_THRESHOLD,
        abs(deltas["pressure_hpa"]) / PRESSURE_SPIKE_THRESHOLD,
    ]
    return float(max(ratios))

class RootCauseEngine:
    """Stateful adapter for the supplied Root_cause_3 diagnostic logic.

    ART3 supplies the statistical anomaly result; this layer derives the
    operational root cause, affected sensor, confidence, severity, evidence,
    and recommended action from that result plus streaming history.
    """
    def __init__(self):
        self.sensor_history={sensor:deque(maxlen=PERSISTENCE_WINDOW) for sensor in FEATURES}
        self.frozen_counts={sensor:0 for sensor in FEATURES}
        self.previous_row=None
        self.previous_timestamp=None
        self.common_shift_active=False
        self.previous_spike_directions=None

    def diagnose(self,row,ml_result):
        z_scores=ml_result.get("z_scores", {sensor:0.0 for sensor in FEATURES})
        deltas=calculate_sensor_deltas(row,self.previous_row)
        spike_flags,spike_count=detect_spikes(deltas)
        communication=check_communication(self.previous_timestamp,row["timestamp"])
        physical_flags=check_physical_limits(row)
        dew_point=calculate_dew_point(row.get("temperature_c"),row.get("humidity_pct"))
        dew_point_issue=check_dew_point_consistency(row,dew_point)
        frozen_flags=update_frozen_counts(deltas,self.frozen_counts)

        for sensor in FEATURES:
            self.sensor_history[sensor].append(safe_float(row.get(sensor)))

        drift_flags=detect_persistent_drift(self.sensor_history,z_scores)
        persistent_drift=any(drift_flags.values())
        weather_consistent=check_weather_consistency(deltas)
        common_info=detect_common_mode_shift(deltas,spike_flags)
        simultaneous_shift=common_info["simultaneous_shift"]
        current_directions=common_info["spike_directions"]
        simultaneous_reversal=bool(
            self.common_shift_active and simultaneous_shift
            and self.previous_spike_directions is not None
            and current_directions == tuple(-x for x in self.previous_spike_directions)
        )
        if simultaneous_reversal:
            self.common_shift_active=False
        elif simultaneous_shift:
            self.common_shift_active=True
        if simultaneous_shift:
            self.previous_spike_directions=current_directions

        exposure_issue=detect_sensor_exposure_issue(deltas,z_scores,spike_flags)
        condensation_issue=detect_humidity_condensation_issue(row,dew_point,z_scores,spike_flags)
        is_anomaly=bool(ml_result.get("is_anomaly",False))
        iforest_flag=bool(ml_result.get("iforest_flag",False))
        z_flag=bool(ml_result.get("z_flag",False))
        physical_issue=any(physical_flags.values()) or dew_point_issue
        frozen_issue=any(frozen_flags.values())

        root_cause=make_root_cause(
            is_anomaly,z_scores,spike_flags,spike_count,deltas,physical_flags,
            dew_point_issue,communication,frozen_flags,drift_flags,
            simultaneous_shift,simultaneous_reversal,weather_consistent,
            exposure_issue,condensation_issue,iforest_flag,z_flag
        )
        confidence=calculate_confidence(
            physical_issue,communication["communication_error"],frozen_issue,
            spike_count,simultaneous_shift,persistent_drift,weather_consistent,
            iforest_flag,z_flag
        )
        severity=calculate_severity(
            physical_issue,communication["communication_error"],max_spike_ratio(deltas),
            simultaneous_shift,persistent_drift,iforest_flag,z_flag
        )
        affected_sensor=determine_affected_sensor(
            z_scores,spike_flags,simultaneous_shift,drift_flags,weather_consistent
        )
        action=recommended_action(root_cause,affected_sensor,severity)
        evidence=build_diagnostic_evidence(
            z_scores,deltas,spike_flags,communication,frozen_flags,drift_flags,
            simultaneous_shift,weather_consistent,exposure_issue,condensation_issue,
            iforest_flag,z_flag
        )

        self.previous_row={sensor:safe_float(row.get(sensor)) for sensor in FEATURES}
        self.previous_timestamp=row.get("timestamp")

        return {
            "root_cause": root_cause,
            "affected_sensor": affected_sensor,
            "confidence": confidence,
            "severity": severity,
            "diagnostic_evidence": evidence,
            "recommended_action": action,
            "temperature_change": deltas["temperature_c"],
            "humidity_change": deltas["humidity_pct"],
            "pressure_change": deltas["pressure_hpa"],
            "temperature_spike": spike_flags["temperature_c"],
            "humidity_spike": spike_flags["humidity_pct"],
            "pressure_spike": spike_flags["pressure_hpa"],
            "dew_point_c": dew_point,
            "dew_point_inconsistency": dew_point_issue,
            "communication": communication,
            "frozen_flags": frozen_flags,
            "drift_flags": drift_flags,
            "simultaneous_shift": simultaneous_shift,
            "simultaneous_reversal": simultaneous_reversal,
            "weather_consistent": weather_consistent,
            "possible_temperature_exposure_issue": exposure_issue,
            "possible_humidity_condensation_issue": condensation_issue,
            "max_spike_ratio": max_spike_ratio(deltas),
        }

    def reset(self):
        self.sensor_history={sensor:deque(maxlen=PERSISTENCE_WINDOW) for sensor in FEATURES}
        self.frozen_counts={sensor:0 for sensor in FEATURES}
        self.previous_row=None
        self.previous_timestamp=None
        self.common_shift_active=False
        self.previous_spike_directions=None
