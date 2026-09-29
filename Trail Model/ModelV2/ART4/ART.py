import pickle
from collections import deque
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --- Import physics.py: adjust whichever line matches your actual folder layout ---
from ModelV2.ART4.src.physics import AtmosphericPhysics          # if physics.py is in the SAME folder as this file
# from src.physics import AtmosphericPhysics    # if physics.py is inside a src/ subfolder

FEATURES=["temperature_c","humidity_pct","pressure_hpa"]
IF_CONTAMINATION=0.01
IF_RANDOM_STATE=42
ROLLING_WINDOW=30
Z_MIN_HISTORY=5
Z_THRESHOLD=3.0
TEMP_SPIKE_THRESHOLD=1.1677
HUMIDITY_SPIKE_THRESHOLD=3.4070
PRESSURE_SPIKE_THRESHOLD=1.5923
FROZEN_MIN_COUNT=5
FROZEN_TOLERANCE=0.001
DRIFT_WINDOW=10
DRIFT_MIN_COUNT=8
DRIFT_Z_THRESHOLD=2.0
DRIFT_NET_CHANGE={"temperature_c":2.0,"humidity_pct":8.0,"pressure_hpa":3.0}
PHYSICAL_LIMITS={"temperature_c":(-80.0,60.0),"humidity_pct":(0.0,100.0),"pressure_hpa":(800.0,1100.0)}
REFIT_N=200
RETRAIN_WINDOW=2000
MODEL_STATE_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"ART3_adaptive_state.pkl")

# --- New: storm-signature detection window (assumes ~1 reading per minute; adjust if your interval differs) ---
STORM_WINDOW=30

# --- New: physical coupling check thresholds ---
COUPLING_TEMP_DROP_THRESHOLD=-8.0     # a sudden temp drop at least this large...
COUPLING_MIN_HUMIDITY_AT_COLD=20.0    # ...paired with humidity this low is physically implausible


def safe_float(value):
    try:
        return float(value)
    except (TypeError,ValueError):
        return np.nan


def sensor_name(sensor):
    return sensor.replace("_c","").replace("_pct","").replace("_hpa","")


def train_isolation_forest(baseline_df):
    model=IsolationForest(contamination=IF_CONTAMINATION,random_state=IF_RANDOM_STATE,n_jobs=-1)
    model.fit(baseline_df[FEATURES])
    return model

class TrackChanges:
    def __init__(self):
        self.prev = None

    def compute_changes(self,row: dict):
        changes = {}
        for f in FEATURES:
            if self.prev is not None:
                changes[f"{f}_change"] = row[f] - self.prev[f]
            else:
                changes[f"{f}_change"] = 0.0

        self.prev = {f: row[f] for f in FEATURES}
        return changes

class TrackFrozen:
    def __init__(self,freeeze_count=4):
        self.freeze_count = freeeze_count
        self.prev_value = {f: None for f in FEATURES}
        self.repeat_count = {f: 0 for f in FEATURES}

    def check_frozen(self,row: dict):
        frozen_flags = {}
        for f in FEATURES:
            if self.prev_value is not None and row[f] == self.prev_value[f]:
                self.repeat_count[f] += 1
            else:
                self.repeat_count[f] = 0
                
            frozen_flags[f] = self.repeat_count[f]>=self.freeze_count
            self.prev_value[f]=row[f]

        any_frozen = any(frozen_flags.values())
        return any_frozen,frozen_flags

class TrackDrift:
    def __init__(self, baseline_df, window=15, slope_threshold=None):
        self.window = window
        self.buffers = {f: deque(maxlen=window) for f in FEATURES}
        # If no threshold given, estimate one from baseline's natural volatility
        if slope_threshold is None:
            self.slope_threshold = {
                f: 3.0 * baseline_df[f].diff().std() for f in FEATURES
            }
        else:
            self.slope_threshold = {f: slope_threshold for f in FEATURES}

    def check_drift(self, row: dict):
        drift_flags = {}
        for f in FEATURES:
            buf = self.buffers[f]
            buf.append(row[f])
            if len(buf) >= self.window:
                x = np.arange(len(buf))
                slope = np.polyfit(x, buf, 1)[0]   # slope of best-fit line through the window
                drift_flags[f] = abs(slope) > self.slope_threshold[f]
            else:
                drift_flags[f] = False
        any_drift = any(drift_flags.values())
        return any_drift, drift_flags

class RollingZScore:
    def __init__(self,window=ROLLING_WINDOW):
        self.window=window
        self.buffers={sensor:deque(maxlen=window) for sensor in FEATURES}

    def calculate(self,row):
        z_scores={}
        for sensor in FEATURES:
            buffer=self.buffers[sensor]
            if len(buffer)>=Z_MIN_HISTORY:
                mean=float(np.mean(buffer))
                std=float(np.std(buffer))
                if std<=1e-12:
                    std=1e-6
                z_scores[sensor]=(safe_float(row[sensor])-mean)/std
            else:
                z_scores[sensor]=0.0
        return z_scores

    def update(self,row):
        for sensor in FEATURES:
            self.buffers[sensor].append(safe_float(row[sensor]))


def detect_anomaly(row,iforest_flag,iforest_score,roller:RollingZScore,frozen: TrackFrozen,drift: TrackDrift):
    z_scores=roller.calculate(row)
    zscore_max=max(abs(value) for value in z_scores.values())
    z_flag=zscore_max>Z_THRESHOLD

    frozen_flag, frozen_details = frozen.check_frozen(row)

    drift_flag, drift_details = drift.check_drift(row)

    is_anomaly=bool((iforest_flag and z_flag) or frozen_flag or drift_flag)
    roller.update(row)
    return {"is_anomaly":is_anomaly,"iforest_flag":bool(iforest_flag),"iforest_score":float(iforest_score),"z_flag":bool(z_flag),"zscore_max":float(zscore_max),"z_scores":z_scores}


def calculate_sensor_deltas(row,previous_row):
    if previous_row is None:
        return {sensor:0.0 for sensor in FEATURES}
    return {sensor:safe_float(row[sensor])-safe_float(previous_row[sensor]) for sensor in FEATURES}


def calculate_window_deltas(sensor_history_storm):
    """New: change over STORM_WINDOW readings, for storm-signature detection (needs a real time window, not a single-step delta)."""
    deltas={}
    for sensor in FEATURES:
        history=sensor_history_storm[sensor]
        if len(history)>=STORM_WINDOW:
            deltas[sensor]=history[-1]-history[0]
        else:
            deltas[sensor]=0.0
    return deltas


def detect_spikes(deltas):
    return {"temperature_c":abs(deltas["temperature_c"])>TEMP_SPIKE_THRESHOLD,"humidity_pct":abs(deltas["humidity_pct"])>HUMIDITY_SPIKE_THRESHOLD,"pressure_hpa":abs(deltas["pressure_hpa"])>PRESSURE_SPIKE_THRESHOLD}


def update_frozen_counts(deltas,frozen_counts):
    for sensor in FEATURES:
        if abs(deltas[sensor])<=FROZEN_TOLERANCE:
            frozen_counts[sensor]+=1
        else:
            frozen_counts[sensor]=0
    return {sensor:frozen_counts[sensor]>=FROZEN_MIN_COUNT for sensor in FEATURES}


def detect_persistent_drift(sensor_history,z_scores):
    drift_flags={sensor:False for sensor in FEATURES}
    for sensor in FEATURES:
        history=sensor_history[sensor]
        if len(history)<DRIFT_WINDOW:
            continue
        values=np.asarray(list(history),dtype=float)
        changes=np.diff(values)
        positive=int(np.sum(changes>0))
        negative=int(np.sum(changes<0))
        same_direction=max(positive,negative)
        net_change=abs(values[-1]-values[0])
        z=z_scores[sensor]
        if same_direction>=DRIFT_MIN_COUNT and net_change>=DRIFT_NET_CHANGE[sensor] and np.isfinite(z) and abs(z)>=DRIFT_Z_THRESHOLD:
            drift_flags[sensor]=True
    return drift_flags


def check_physical_limits(row):
    flags={}
    for sensor,(low,high) in PHYSICAL_LIMITS.items():
        value=safe_float(row[sensor])
        flags[sensor]=bool(not np.isfinite(value) or value<low or value>high)
    return flags


# --- Dew point: now delegates to physics.py's AtmosphericPhysics instead of a local duplicate ---
def check_dew_point_consistency(row):
    dew_point=AtmosphericPhysics.dew_point(safe_float(row["temperature_c"]),safe_float(row["humidity_pct"]))
    if not np.isfinite(dew_point):
        return False
    return bool(dew_point>safe_float(row["temperature_c"])+0.2)


# --- New: multivariate physical coupling check (sudden cold + implausibly dry air together) ---
def check_physical_coupling(deltas,row):
    temp_drop=deltas["temperature_c"]<=COUPLING_TEMP_DROP_THRESHOLD
    humidity_too_low=safe_float(row["humidity_pct"])<=COUPLING_MIN_HUMIDITY_AT_COLD
    return bool(temp_drop and humidity_too_low)


def check_communication(previous_timestamp,current_timestamp):
    if previous_timestamp is None:
        return False
    difference=current_timestamp-previous_timestamp
    return bool(difference>pd.Timedelta(minutes=1) or difference<=pd.Timedelta(0))


def detect_simultaneous_shift(spike_flags):
    return bool(sum(spike_flags.values())==3)


def check_weather_consistency(deltas):
    return bool(deltas["temperature_c"]<-0.5 and deltas["humidity_pct"]>0.2 and deltas["pressure_hpa"]<0)


def classify_triggers(ml_result,physical_flags,dew_point_issue,communication_issue,frozen_flags,spike_flags,drift_flags,simultaneous_shift,coupling_issue):
    triggers=[]
    if ml_result["iforest_flag"]:
        triggers.append("IsolationForest")
    if ml_result["z_flag"]:
        triggers.append("RollingZScore")
    physical_sensors=[sensor_name(sensor) for sensor,flag in physical_flags.items() if flag]
    if physical_sensors:
        triggers.append("check_physical_limits("+",".join(physical_sensors)+")")
    if dew_point_issue:
        triggers.append("check_dew_point_consistency")
    if coupling_issue:
        triggers.append("check_physical_coupling")
    if communication_issue:
        triggers.append("check_communication")
    frozen_sensors=[sensor_name(sensor) for sensor,flag in frozen_flags.items() if flag]
    if frozen_sensors:
        triggers.append("update_frozen_counts("+",".join(frozen_sensors)+")")
    spike_sensors=[sensor_name(sensor) for sensor,flag in spike_flags.items() if flag]
    if spike_sensors:
        triggers.append("detect_spikes("+",".join(spike_sensors)+")")
    drift_sensors=[sensor_name(sensor) for sensor,flag in drift_flags.items() if flag]
    if drift_sensors:
        triggers.append("detect_persistent_drift("+",".join(drift_sensors)+")")
    if simultaneous_shift:
        triggers.append("detect_simultaneous_shift")
    return triggers


def run_art3_pipeline(df,model):
    df=df.copy().reset_index(drop=True)
    df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    iforest_predictions=model.predict(df[FEATURES])
    iforest_scores=model.decision_function(df[FEATURES])
    iforest_flags=iforest_predictions==-1
    roller=RollingZScore()
    sensor_history={sensor:deque(maxlen=DRIFT_WINDOW) for sensor in FEATURES}
    sensor_history_storm={sensor:deque(maxlen=STORM_WINDOW) for sensor in FEATURES}   # new
    frozen_counts={sensor:0 for sensor in FEATURES}
    previous_row=None
    previous_timestamp=None
    predictions=[]

    for position,series in df.iterrows():
        row=series.to_dict()
        ml_result=detect_anomaly(row,bool(iforest_flags[position]),float(iforest_scores[position]),roller,frozen,drift)
        deltas=calculate_sensor_deltas(row,previous_row)
        spike_flags=detect_spikes(deltas)
        frozen_flags=update_frozen_counts(deltas,frozen_counts)

        for sensor in FEATURES:
            sensor_history[sensor].append(safe_float(row[sensor]))
            sensor_history_storm[sensor].append(safe_float(row[sensor]))   # new

        drift_flags=detect_persistent_drift(sensor_history,ml_result["z_scores"])
        physical_flags=check_physical_limits(row)
        dew_point_issue=check_dew_point_consistency(row)
        coupling_issue=check_physical_coupling(deltas,row)   # new
        communication_issue=check_communication(previous_timestamp,row["timestamp"])
        simultaneous_shift=detect_simultaneous_shift(spike_flags)
        weather_consistent=check_weather_consistency(deltas)

        # new: storm-signature check over a real time window (not single-step delta)
        window_deltas=calculate_window_deltas(sensor_history_storm)
        is_storm,storm_conf,storm_desc=AtmosphericPhysics.detect_storm_signature(
            delta_temp=window_deltas["temperature_c"],
            delta_pressure=window_deltas["pressure_hpa"],
            delta_rh=window_deltas["humidity_pct"]
        )

        rule_based_anomaly=bool(
            any(physical_flags.values()) or dew_point_issue or communication_issue or
            any(frozen_flags.values()) or any(spike_flags.values()) or any(drift_flags.values()) or
            coupling_issue
        )
        is_anomaly=bool(ml_result["is_anomaly"] or rule_based_anomaly)

        # new: storm override — don't flag a genuine severe-weather signature as a sensor fault
        if is_storm and is_anomaly:
            is_anomaly=False

        triggers=classify_triggers(ml_result,physical_flags,dew_point_issue,communication_issue,frozen_flags,spike_flags,drift_flags,simultaneous_shift,coupling_issue)
        if is_anomaly and not triggers:
            triggers=["final_anomaly_decision"]

        predictions.append({
            "timestamp":row["timestamp"],
            "is_anomaly":is_anomaly,
            "anomaly_type":" + ".join(triggers) if is_anomaly else ("GENUINE_WEATHER_EVENT" if is_storm else "normal"),
            "iforest_flag":ml_result["iforest_flag"],
            "z_flag":ml_result["z_flag"],
            "rule_based_flag":rule_based_anomaly,
            "simultaneous_shift":simultaneous_shift,
            "weather_consistent":weather_consistent,
            "zscore_max":ml_result["zscore_max"],
            "is_storm":is_storm,               # new
            "storm_confidence":storm_conf,     # new
            "storm_description":storm_desc     # new
        })

        previous_row={sensor:safe_float(row[sensor]) for sensor in FEATURES}
        previous_timestamp=row["timestamp"]

    return pd.DataFrame(predictions)


class AdaptiveModel:
    def __init__(self,initial_df,load_saved=True):
        self.buffer=deque(maxlen=RETRAIN_WINDOW)
        self.count_since_refit=0
        self.total_training_rows=0
        self.total_refits=0

        if load_saved and os.path.isfile(MODEL_STATE_PATH):
            try:
                self.load_state()
                return
            except Exception:
                self.buffer.clear()

        clean=initial_df[FEATURES].dropna().copy()
        self.model=train_isolation_forest(clean)

        for record in clean.to_dict("records"):
            self.buffer.append(record)

        self.total_training_rows=len(clean)
        self.total_refits=1

    def addnrefit(self,row,allow_training=True):
        if not allow_training:
            return False

        self.buffer.append({f:safe_float(row[f]) for f in FEATURES})
        self.count_since_refit+=1
        self.total_training_rows+=1

        if self.count_since_refit>=REFIT_N:
            self.model=train_isolation_forest(pd.DataFrame(list(self.buffer)))
            self.count_since_refit=0
            self.total_refits+=1
            return True

        return False

    def get_model(self):
        return self.model

    def save_state(self):
        state={
            "model":self.model,
            "buffer":list(self.buffer),
            "count_since_refit":self.count_since_refit,
            "total_training_rows":self.total_training_rows,
            "total_refits":self.total_refits
        }
        with open(MODEL_STATE_PATH,"wb") as f:
            pickle.dump(state,f)

    def load_state(self):
        with open(MODEL_STATE_PATH,"rb") as f:
            state=pickle.load(f)
        self.model=state["model"]
        self.buffer=deque(state["buffer"],maxlen=RETRAIN_WINDOW)
        self.count_since_refit=state["count_since_refit"]
        self.total_training_rows=state["total_training_rows"]
        self.total_refits=state["total_refits"]

    def reset_state(self):
        if os.path.isfile(MODEL_STATE_PATH):
            os.remove(MODEL_STATE_PATH)


# # ---------------------------------------------------------------
# # New: built-in per-type evaluation helper
# # ---------------------------------------------------------------
# def evaluate_by_type(ground_truth_df, predictions_df, label_column="anomaly_type", pred_column="is_anomaly"):
#     """
#     Prints overall precision/recall/F1 plus a catch-rate breakdown per injected anomaly type.

#     ground_truth_df: must have a boolean/0-1 'is_anomaly' column AND a label_column
#                       (e.g. "spike", "frozen", "drift", "dropout", "normal") aligned
#                       row-for-row with predictions_df.
#     predictions_df:  the DataFrame returned by run_art3_pipeline().
#     """
#     from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#     y_true = ground_truth_df["is_anomaly"].astype(int).values
#     y_pred = predictions_df[pred_column].astype(int).values

#     if len(y_true) != len(y_pred):
#         print(f"[evaluate_by_type] WARNING: length mismatch — ground_truth has {len(y_true)} rows, "
#               f"predictions has {len(y_pred)} rows. Check your row alignment before trusting these numbers.")

#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
#     precision = precision_score(y_true, y_pred, zero_division=0)
#     recall = recall_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)

#     print("\nConfusion matrix (rows=actual, cols=predicted) [0, 1]:")
#     print(cm)
#     print(f"Precision: {precision:.3f}")
#     print(f"Recall:    {recall:.3f}")
#     print(f"F1 score:  {f1:.3f}")

#     print("\n--- Catch rate by anomaly type ---")
#     if label_column in ground_truth_df.columns:
#         for atype in ground_truth_df[label_column].unique():
#             if str(atype).lower() in ("normal", "none"):
#                 continue
#             mask = (ground_truth_df[label_column] == atype).values
#             total = int(mask.sum())
#             if total == 0:
#                 continue
#             caught = int(np.sum(mask & (y_pred == 1)))
#             pct = (caught / total * 100) if total > 0 else 0.0
#             print(f"{str(atype):25s}: caught {caught}/{total} ({pct:.1f}%)")
#     else:
#         print(f"[evaluate_by_type] No '{label_column}' column found in ground_truth_df — skipping per-type breakdown.")

#     return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}