import os
import pickle
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

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

def detect_anomaly(row,iforest_flag,iforest_score,roller):
    z_scores=roller.calculate(row)
    zscore_max=max(abs(value) for value in z_scores.values())
    z_flag=zscore_max>Z_THRESHOLD
    is_anomaly=bool(iforest_flag or z_flag)
    roller.update(row)
    return {"is_anomaly":is_anomaly,"iforest_flag":bool(iforest_flag),"iforest_score":float(iforest_score),"z_flag":bool(z_flag),"zscore_max":float(zscore_max),"z_scores":z_scores}

def calculate_sensor_deltas(row,previous_row):
    if previous_row is None:
        return {sensor:0.0 for sensor in FEATURES}
    return {sensor:safe_float(row[sensor])-safe_float(previous_row[sensor]) for sensor in FEATURES}

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

def calculate_dew_point(temperature_c,humidity_pct):
    temperature=safe_float(temperature_c)
    humidity=safe_float(humidity_pct)
    if not np.isfinite(temperature) or not np.isfinite(humidity) or humidity<=0 or humidity>100:
        return np.nan
    a=17.27
    b=237.7
    gamma=(a*temperature)/(b+temperature)+np.log(humidity/100.0)
    denominator=a-gamma
    if abs(denominator)<1e-12:
        return np.nan
    return (b*gamma)/denominator

def check_dew_point_consistency(row):
    dew_point=calculate_dew_point(row["temperature_c"],row["humidity_pct"])
    if not np.isfinite(dew_point):
        return False
    return bool(dew_point>safe_float(row["temperature_c"])+0.2)

def check_communication(previous_timestamp,current_timestamp):
    if previous_timestamp is None:
        return False
    difference=current_timestamp-previous_timestamp
    return bool(difference>pd.Timedelta(minutes=1) or difference<=pd.Timedelta(0))

def detect_simultaneous_shift(spike_flags):
    return bool(sum(spike_flags.values())==3)

def check_weather_consistency(deltas):
    return bool(deltas["temperature_c"]<-0.5 and deltas["humidity_pct"]>0.2 and deltas["pressure_hpa"]<0)

def classify_triggers(ml_result,physical_flags,dew_point_issue,communication_issue,frozen_flags,spike_flags,drift_flags,simultaneous_shift):
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
    frozen_counts={sensor:0 for sensor in FEATURES}
    previous_row=None
    previous_timestamp=None
    predictions=[]

    for position,series in df.iterrows():
        row=series.to_dict()
        ml_result=detect_anomaly(row,bool(iforest_flags[position]),float(iforest_scores[position]),roller)
        deltas=calculate_sensor_deltas(row,previous_row)
        spike_flags=detect_spikes(deltas)
        frozen_flags=update_frozen_counts(deltas,frozen_counts)

        for sensor in FEATURES:
            sensor_history[sensor].append(safe_float(row[sensor]))

        drift_flags=detect_persistent_drift(sensor_history,ml_result["z_scores"])
        physical_flags=check_physical_limits(row)
        dew_point_issue=check_dew_point_consistency(row)
        communication_issue=check_communication(previous_timestamp,row["timestamp"])
        simultaneous_shift=detect_simultaneous_shift(spike_flags)
        weather_consistent=check_weather_consistency(deltas)

        rule_based_anomaly=bool(any(physical_flags.values()) or dew_point_issue or communication_issue or any(frozen_flags.values()) or any(spike_flags.values()) or any(drift_flags.values()))
        is_anomaly=bool(ml_result["is_anomaly"] or rule_based_anomaly)

        triggers=classify_triggers(ml_result,physical_flags,dew_point_issue,communication_issue,frozen_flags,spike_flags,drift_flags,simultaneous_shift)
        if is_anomaly and not triggers:
            triggers=["final_anomaly_decision"]

        predictions.append({
            "timestamp":row["timestamp"],
            "is_anomaly":is_anomaly,
            "anomaly_type":" + ".join(triggers) if is_anomaly else "normal",
            "iforest_flag":ml_result["iforest_flag"],
            "z_flag":ml_result["z_flag"],
            "rule_based_flag":rule_based_anomaly,
            "simultaneous_shift":simultaneous_shift,
            "weather_consistent":weather_consistent,
            "zscore_max":ml_result["zscore_max"]
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
