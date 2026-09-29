import sys
from pathlib import Path
from collections import deque

import pandas as pd
import numpy as np


# =========================================================
# FIND THE MODEL FOLDER
# =========================================================

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "ModelV2"

sys.path.insert(0, str(model_path))


# =========================================================
# IMPORT ART3 MODEL
# =========================================================

from ART3.ARTmodel import (
    AdaptiveModel,
    RollingZScore,
    FEATURES,
    detect_anomaly,
    calculate_sensor_deltas,
    detect_spikes,
    update_frozen_counts,
    detect_persistent_drift,
    check_physical_limits,
    check_dew_point_consistency,
    check_communication,
    detect_simultaneous_shift,
    check_weather_consistency,
    classify_triggers,
)

from root_cause_engine import RootCauseEngine


# =========================================================
# STREAM PROCESSOR
# =========================================================

class StreamProcessor:
    """
    Controls how weather data is sent to the ART3 model.

    ART3 combines:
    - Isolation Forest
    - Rolling Z-Score
    - Physical-limit checks
    - Dew-point consistency
    - Communication-gap checks
    - Frozen-sensor detection
    - Spike detection
    - Persistent-drift detection
    - Simultaneous sensor-shift detection
    - Adaptive Isolation Forest retraining

    The dashboard-facing interface is intentionally kept compatible
    with the previous ART1 StreamProcessor.
    """

    def __init__(
        self,
        df,
        baseline_size=200,
        start_index=0,
        end_index=None,
        batch_size=50,
        baseline_df=None
    ):

        # -------------------------------------------------
        # Prepare dataset
        # -------------------------------------------------

        self.df = (
            df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # -------------------------------------------------
        # Store settings
        # -------------------------------------------------

        self.baseline_size = int(baseline_size)
        self.start_index = int(start_index)

        if end_index is None:
            self.end_index = len(self.df)
        else:
            self.end_index = min(int(end_index), len(self.df))

        self.batch_size = int(batch_size)

        # -------------------------------------------------
        # Validate settings
        # -------------------------------------------------

        if self.baseline_size <= 0:
            raise ValueError("Baseline size must be greater than 0.")

        if self.baseline_size > len(self.df):
            raise ValueError("Baseline size cannot be larger than the dataset.")

        if baseline_df is None and self.start_index < self.baseline_size:
            raise ValueError(
                "Detection start cannot be smaller than the baseline size."
            )

        if self.start_index >= self.end_index:
            raise ValueError("Detection start must be smaller than detection end.")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")

        # -------------------------------------------------
        # Create baseline
        # -------------------------------------------------

        if baseline_df is None:
            self.baseline_df = (
                self.df.iloc[:self.baseline_size].copy()
            )
        else:
            self.baseline_df = (
                baseline_df
                .sort_values("timestamp")
                .reset_index(drop=True)
                .copy()
            )

            if self.baseline_df.empty:
                raise ValueError("The clean training dataset is empty.")

            if not set(FEATURES).issubset(self.baseline_df.columns):
                raise ValueError(
                    "Clean training dataset must contain: "
                    + ", ".join(FEATURES)
                )

            # The selected baseline size controls how much clean data
            # is used for training. The stream itself remains separate.
            self.baseline_df = self.baseline_df.iloc[:self.baseline_size].copy()

        # -------------------------------------------------
        # Create detection stream
        # -------------------------------------------------

        self.stream_df = (
            self.df.iloc[self.start_index:self.end_index].copy()
        )

        # -------------------------------------------------
        # Create ART3 components
        # -------------------------------------------------

        self.adaptive = AdaptiveModel(self.baseline_df, load_saved=False)
        self.roller = RollingZScore()

        # ART3 state that must persist across batches.
        self.sensor_history = {
            sensor: deque(maxlen=10)
            for sensor in FEATURES
        }
        self.frozen_counts = {sensor: 0 for sensor in FEATURES}
        self.previous_row = None
        self.previous_timestamp = None

        # -------------------------------------------------
        # Result storage
        # -------------------------------------------------

        self.results = []
        self.batch_history = []

        # -------------------------------------------------
        # Processing position
        # -------------------------------------------------

        self.position = 0
        self.batch_number = 0

        # -------------------------------------------------
        # Current information
        # -------------------------------------------------

        self.last_batch = None
        self.last_anomaly = None

        # Live-stream feed state. Telemetry is processed one row at a time,
        # but rows are grouped into the configured batch size for display.
        self.live_batch_number = 0
        self.live_batch = None

        # Root-cause diagnostic state mirrors the supplied Root_cause_3.py
        # and is kept separate from ART3 detection state.
        self.root_cause_engine = RootCauseEngine()

    # =====================================================
    # PROCESS NEXT SINGLE TELEMETRY READING
    # =====================================================

    def process_next_row(self):
        """Process exactly one telemetry record for live streaming.

        ART3 receives one telemetry row per call. For the dashboard feed,
        those rows are accumulated into the configured display batch size,
        so the feed shows a live-updating current batch rather than creating
        a new batch card for every simulated minute.
        """
        if self.finished():
            return None

        # Reuse the same stateful ART3 logic while advancing exactly one row.
        self.process_next(batch_size=1)

        single = self.batch_history.pop()

        # Start a new display batch when the previous one is full.
        if (
            self.live_batch is None
            or self.live_batch["readings_processed"] >= self.batch_size
        ):
            self.live_batch_number += 1
            self.live_batch = {
                "batch_number": self.live_batch_number,
                "start_position": single["start_position"],
                "end_position": single["end_position"],
                "readings_processed": 0,
                "start_timestamp": single["start_timestamp"],
                "end_timestamp": single["end_timestamp"],
                "latest_reading": single["latest_reading"],
                "anomaly_count": 0,
                "status": "NORMAL",
                "anomalies": [],
                "model_refitted": False,
                "processed_total": single["processed_total"],
                "stream_total": single["stream_total"],
            }

        batch = self.live_batch

        batch["end_position"] = single["end_position"]
        batch["readings_processed"] += single["readings_processed"]
        batch["end_timestamp"] = single["end_timestamp"]
        batch["latest_reading"] = single["latest_reading"]
        batch["anomaly_count"] += single["anomaly_count"]
        batch["anomalies"].extend(single["anomalies"])
        batch["model_refitted"] = (
            batch["model_refitted"] or single["model_refitted"]
        )
        batch["processed_total"] = single["processed_total"]
        batch["stream_total"] = single["stream_total"]
        batch["status"] = (
            "ANOMALY" if batch["anomaly_count"] > 0 else "NORMAL"
        )

        # The temporary single-row entry was removed above. Keep the
        # existing live batch in place while it is filling; append a new
        # card only when a fresh live batch starts.
        if not self.batch_history or self.batch_history[-1] is not batch:
            self.batch_history.append(batch)

        self.last_batch = batch
        return batch

    # =====================================================
    # PROCESS NEXT BATCH
    # =====================================================

    def process_next(self, batch_size=None):
        """Process one batch using the stateful ART3 pipeline."""

        if self.finished():
            return pd.DataFrame(self.results)

        if batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = int(batch_size)

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")

        # -------------------------------------------------
        # Determine batch boundaries
        # -------------------------------------------------

        batch_start_position = self.position
        end_position = min(
            self.position + batch_size,
            len(self.stream_df)
        )

        current_batch = (
            self.stream_df
            .iloc[self.position:end_position]
            .copy()
        )

        batch_results = []
        model_refitted = False

        # =================================================
        # PROCESS EACH READING
        # =================================================

        for _, row in current_batch.iterrows():
            row_dict = row.to_dict()

            # ---------------------------------------------
            # Isolation Forest prediction
            # ---------------------------------------------

            # ART3's run_art3_pipeline performs these predictions
            # before adaptive training on the current reading.
            model = self.adaptive.get_model()
            values = pd.DataFrame([row_dict])[FEATURES]

            iforest_prediction = int(model.predict(values)[0])
            iforest_score = float(model.decision_function(values)[0])
            iforest_flag = iforest_prediction == -1

            # ---------------------------------------------
            # ML anomaly detection + rolling Z-score
            # ---------------------------------------------

            ml_result = detect_anomaly(
                row_dict,
                iforest_flag,
                iforest_score,
                self.roller
            )

            # ---------------------------------------------
            # ART3 rule-based detectors
            # ---------------------------------------------

            deltas = calculate_sensor_deltas(
                row_dict,
                self.previous_row
            )

            spike_flags = detect_spikes(deltas)
            frozen_flags = update_frozen_counts(
                deltas,
                self.frozen_counts
            )

            for sensor in FEATURES:
                try:
                    value = float(row_dict[sensor])
                except (TypeError, ValueError):
                    value = np.nan
                self.sensor_history[sensor].append(value)

            drift_flags = detect_persistent_drift(
                self.sensor_history,
                ml_result["z_scores"]
            )

            physical_flags = check_physical_limits(row_dict)
            dew_point_issue = check_dew_point_consistency(row_dict)
            communication_issue = check_communication(
                self.previous_timestamp,
                row_dict["timestamp"]
            )
            simultaneous_shift = detect_simultaneous_shift(spike_flags)
            weather_consistent = check_weather_consistency(deltas)

            rule_based_anomaly = bool(
                any(physical_flags.values())
                or dew_point_issue
                or communication_issue
                or any(frozen_flags.values())
                or any(spike_flags.values())
                or any(drift_flags.values())
            )

            is_anomaly = bool(
                ml_result["is_anomaly"] or rule_based_anomaly
            )

            triggers = classify_triggers(
                ml_result,
                physical_flags,
                dew_point_issue,
                communication_issue,
                frozen_flags,
                spike_flags,
                drift_flags,
                simultaneous_shift
            )

            if is_anomaly and not triggers:
                triggers = ["final_anomaly_decision"]

            # ---------------------------------------------
            # Root-cause diagnosis
            # ---------------------------------------------
            root_cause = self.root_cause_engine.diagnose(
                row_dict,
                {
                    **ml_result,
                    "is_anomaly": is_anomaly,
                    "iforest_flag": iforest_flag,
                    "z_flag": ml_result["z_flag"],
                }
            )

            # ---------------------------------------------
            # Preserve the dashboard-compatible result shape
            # while exposing ART3's additional information.
            # ---------------------------------------------

            result = {
                "timestamp": row_dict.get("timestamp"),
                "is_anomaly": is_anomaly,
                "anomaly_type": (
                    " + ".join(triggers)
                    if is_anomaly else "normal"
                ),
                "iforest_flag": ml_result["iforest_flag"],
                "iforest_score": ml_result["iforest_score"],
                "z_flag": ml_result["z_flag"],
                "zscore_max": ml_result["zscore_max"],
                "z_scores": ml_result["z_scores"],
                "rule_based_flag": rule_based_anomaly,
                "simultaneous_shift": simultaneous_shift,
                "weather_consistent": weather_consistent,
                "physical_flags": physical_flags,
                "dew_point_issue": dew_point_issue,
                "communication_issue": communication_issue,
                "frozen_flags": frozen_flags,
                "spike_flags": spike_flags,
                "drift_flags": drift_flags,
                "triggers": triggers,
                # Root-cause diagnosis from Root_cause_3.py
                "root_cause": root_cause["root_cause"],
                "affected_sensor": root_cause["affected_sensor"],
                "confidence": root_cause["confidence"],
                "severity": root_cause["severity"],
                "diagnostic_evidence": root_cause["diagnostic_evidence"],
                "recommended_action": root_cause["recommended_action"],
                "temperature_change": root_cause["temperature_change"],
                "humidity_change": root_cause["humidity_change"],
                "pressure_change": root_cause["pressure_change"],
                "temperature_spike": root_cause["temperature_spike"],
                "humidity_spike": root_cause["humidity_spike"],
                "pressure_spike": root_cause["pressure_spike"],
                "dew_point_c": root_cause["dew_point_c"],
                "dew_point_inconsistency": root_cause["dew_point_inconsistency"],
                "communication_diagnostics": root_cause["communication"],
                "root_cause_frozen_flags": root_cause["frozen_flags"],
                "root_cause_drift_flags": root_cause["drift_flags"],
                "root_cause_simultaneous_reversal": root_cause["simultaneous_reversal"],
                "root_cause_exposure_issue": root_cause["possible_temperature_exposure_issue"],
                "root_cause_condensation_issue": root_cause["possible_humidity_condensation_issue"],
                "max_spike_ratio": root_cause["max_spike_ratio"],
            }

            self.results.append(result)
            batch_results.append(result)

            # ---------------------------------------------
            # Save latest anomaly
            # ---------------------------------------------

            if result.get("is_anomaly", False):
                self.last_anomaly = result.copy()

            # ---------------------------------------------
            # Adaptive ART3 model update
            # ---------------------------------------------

            previous_refit_count = self.adaptive.count_since_refit

            self.adaptive.addnrefit(row_dict)

            if (
                previous_refit_count > 0
                and self.adaptive.count_since_refit == 0
            ):
                model_refitted = True

            # ---------------------------------------------
            # Advance state for next reading
            # ---------------------------------------------

            self.previous_row = {
                sensor: row_dict.get(sensor)
                for sensor in FEATURES
            }
            self.previous_timestamp = row_dict.get("timestamp")

        # =================================================
        # UPDATE PROCESSING POSITION
        # =================================================

        self.position = end_position
        self.batch_number += 1

        # =================================================
        # ANALYZE THIS BATCH
        # =================================================

        result_df = pd.DataFrame(batch_results)

        if not result_df.empty:
            anomaly_count = int(result_df["is_anomaly"].sum())
        else:
            anomaly_count = 0

        latest_reading = current_batch.iloc[-1].to_dict()

        anomaly_results = [
            result
            for result in batch_results
            if result.get("is_anomaly", False)
        ]

        batch_status = "ANOMALY" if anomaly_count > 0 else "NORMAL"

        # =================================================
        # CREATE BATCH INFORMATION
        # =================================================

        batch_info = {
            "batch_number": self.batch_number,
            "start_position": batch_start_position,
            "end_position": end_position - 1,
            "readings_processed": len(current_batch),
            "start_timestamp": current_batch.iloc[0]["timestamp"],
            "end_timestamp": current_batch.iloc[-1]["timestamp"],
            "latest_reading": latest_reading,
            "anomaly_count": anomaly_count,
            "status": batch_status,
            "anomalies": anomaly_results,
            "model_refitted": model_refitted,
            "processed_total": self.position,
            "stream_total": len(self.stream_df),
        }

        self.last_batch = batch_info
        self.batch_history.append(batch_info)

        return pd.DataFrame(self.results)

    # =====================================================
    # PROCESS ALL REMAINING DATA
    # =====================================================

    def process_all(self):
        while not self.finished():
            self.process_next()

        return pd.DataFrame(self.results)

    # =====================================================
    # RESET PROCESSING
    # =====================================================

    def reset(self):
        """Reset the ART3 model and all stateful detector history."""

        self.adaptive = AdaptiveModel(self.baseline_df, load_saved=False)
        self.roller = RollingZScore()

        self.sensor_history = {
            sensor: deque(maxlen=10)
            for sensor in FEATURES
        }
        self.frozen_counts = {sensor: 0 for sensor in FEATURES}
        self.previous_row = None
        self.previous_timestamp = None

        self.results = []
        self.batch_history = []
        self.position = 0
        self.batch_number = 0
        self.last_batch = None
        self.last_anomaly = None
        self.live_batch_number = 0
        self.live_batch = None

        # Root-cause diagnostic state mirrors the supplied Root_cause_3.py
        # and is kept separate from ART3 detection state.
        self.root_cause_engine = RootCauseEngine()

    # =====================================================
    # CHECK WHETHER PROCESSING IS FINISHED
    # =====================================================

    def finished(self):
        return self.position >= len(self.stream_df)

    # =====================================================
    # PROCESSING PROGRESS
    # =====================================================

    def progress(self):
        total = len(self.stream_df)
        processed = self.position
        remaining = total - processed
        return processed, total, remaining
