import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from collections import deque

# RT is Real-Time btw

''' 
learn about deque just a little. I used two parameters to detect anomaly here
one is isolation forest flag (which is understandable) and
another one is zscore flag to make it more effective but tbh idk it is good or nah.

contamination is 0.065 just bcuz the testing csv was built by chatgpt and it selected that.

refit_n is after how many new readings the model will retrain &
retrain_window is on how much data the model will train on after every refit_n &
rolling_window is mainly for deque to calculate zScore.

this is just a basic RT model and it can be imported across files to get output as i did with 'Result1.py'

***Things to be added***

1. In features we can add temp,humidity & pressure change column.
2. Enhancement of getting the model know that some readings can differ across a full day.
    For example, 35`C is normal at 2pm and 25`C is also normal at 2am in the same day.For that ig we have to
    retarin the model more often but retraining everyday will make the model make no sense ?? idk

'''
features = ["temperature_c","humidity_pct","pressure_hpa"]
contamination = 0.01 
z_threshold = 3.0
rolling_window = 30
refit_n = 200
retrain_window = 1000

def baseline_model(training_df):
    model = IsolationForest(contamination=contamination,random_state=42)
    model.fit(training_df[features])
    return model

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

def detect_anomaly(row: dict,model,roller:RollingZScore):
    x = pd.DataFrame([[row[f] for f in features]], columns=features)

    iforest_pred = model.predict(x)[0]
    iforest_score = model.decision_function(x)[0]
    iforest_flag = iforest_pred == -1

    z_scores = roller.updatenscore(row)
    z_scoremax = max(abs(v) for v in z_scores.values())
    z_flag = z_scoremax > z_threshold

    is_anomaly = iforest_flag or z_flag

    return{
        "timestamp": row.get("timestamp"),
        "is_anomaly": is_anomaly,
        "iforest_score": iforest_score,
        "iforest_flag": iforest_flag,
        "zscore_max": z_scoremax,
        "z_flag": z_flag
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
            self.model = baseline_model(pd.DataFrame(self.buffer))
            self.count_since_refit = 0

    def get_model(self):
        return self.model