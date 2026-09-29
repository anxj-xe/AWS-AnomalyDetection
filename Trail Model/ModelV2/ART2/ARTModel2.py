import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from collections import deque

normal_features = ["temperature_c","humidity_pct","pressure_hpa"]
features_change = ["temperature_c_change","humidity_pct_change","pressure_hpa_change"]
features = normal_features + features_change
contamination = 0.01
z_threshold = 3.0
rolling_window = 30
refit_n = 1000
retrain_window = 2500

def baseline_model(training_df):
    model = IsolationForest(contamination=contamination,random_state=42)
    model.fit(training_df[features])
    return model

class TrackChanges:
    def __init__(self):
        self.prev = None

    def compute_changes(self,row: dict):
        changes = {}
        for f in normal_features:
            if self.prev is not None:
                changes[f"{f}_change"] = row[f] - self.prev[f]
            else:
                changes[f"{f}_change"] = 0.0

        self.prev = {f: row[f] for f in normal_features}
        return changes

class TrackFrozen:
    def __init__(self,freeeze_count=4):
        self.freeze_count = freeeze_count
        self.prev_value = {f: None for f in normal_features}
        self.repeat_count = {f: 0 for f in normal_features}

    def check_frozen(self,row: dict):
        frozen_flags = {}
        for f in normal_features:
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
        self.buffers = {f: deque(maxlen=window) for f in normal_features}
        # If no threshold given, estimate one from baseline's natural volatility
        if slope_threshold is None:
            self.slope_threshold = {
                f: 3.0 * baseline_df[f].diff().std() for f in normal_features
            }
        else:
            self.slope_threshold = {f: slope_threshold for f in normal_features}

    def check_drift(self, row: dict):
        drift_flags = {}
        for f in normal_features:
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
    def __init__(self,window=rolling_window):
        self.window = window
        self.buffers = {f: deque(maxlen=window) for f in features}

    def updatenscore(self,row:dict):
        z_scores={}
        for f in features:
            buf = self.buffers[f]
            if(len(buf)>=5):
                mean = np.mean(buf)
                std = np.std(buf) or 1e-6
                z_scores[f] = (row[f]-mean)/std
            else:
                z_scores[f]=0.0
            buf.append(row[f])
        return z_scores

def detect_anomaly(row: dict,model,roller:RollingZScore,frozen: TrackFrozen,drift: TrackDrift):
    x = pd.DataFrame([[row[f] for f in features]], columns=features)

    iforest_pred = model.predict(x)[0]
    iforest_score = model.decision_function(x)[0]
    iforest_flag = iforest_pred == -1

    z_scores = roller.updatenscore(row)
    z_scoremax = max(abs(v) for v in z_scores.values())
    z_flag = z_scoremax > z_threshold

    frozen_flag, frozen_details = frozen.check_frozen(row)

    drift_flag, drift_details = drift.check_drift(row)

    is_anomaly = (iforest_flag and z_flag) or frozen_flag or drift_flag

    return{
        "timestamp": row.get("timestamp"),
        "is_anomaly": is_anomaly,
        "iforest_score": iforest_score,
        "iforest_flag": iforest_flag,
        "zscore_max": z_scoremax,
        "z_flag": z_flag,
        "frozen_flag": frozen_flag,
        "drift_flag": drift_flag
    }

class AdaptiveModel:
    def __init__(self,initial_df):
        self.model = baseline_model(initial_df)
        self.buffer = deque(initial_df[features].to_dict("records"),maxlen=retrain_window)
        self.count_since_refit = 0

    def addnrefit(self,row: dict):
        self.buffer.append({f: row[f] for f in features})
        self.count_since_refit += 1

        if self.count_since_refit>=refit_n:
            # self.model = baseline_model(pd.DataFrame(self.buffer))
            self.count_since_refit = 0

    def get_model(self):
        return self.model