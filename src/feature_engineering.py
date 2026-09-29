"""
Feature Engineering for Automatic Weather Station Anomaly Detection
Extracts thermodynamic, sliding-window statistical, rate-of-change, and diurnal cyclical features.
Supports both batch DataFrame processing and real-time streaming buffer feature extraction.
"""

from typing import Dict, List, Optional, Any
import math
import numpy as np
import pandas as pd
from src.physics import AtmosphericPhysics


class AWSFeatureExtractor:
    """
    Computes rich multivariate and temporal meteorological features
    from raw Temperature, Pressure, and Relative Humidity streams.
    """

    FEATURE_NAMES = [
        "temperature", "pressure", "humidity",
        "dew_point", "dew_point_depression", "vpd", "potential_temp",
        "d_temp_1", "d_press_1", "d_rh_1",
        "d_temp_5", "d_press_5", "d_rh_5",
        "d_temp_15", "d_press_15", "d_rh_15",
        "temp_rolling_mean_15", "press_rolling_mean_15", "rh_rolling_mean_15",
        "temp_rolling_std_15", "press_rolling_std_15", "rh_rolling_std_15",
        "temp_zscore_15", "press_zscore_15", "rh_zscore_15",
        "t_rh_coupling", "barometric_tendency_30",
        "hour_sin", "hour_cos"
    ]

    def __init__(self, buffer_size: int = 60):
        self.buffer_size = buffer_size
        self.temp_buffer: List[float] = []
        self.press_buffer: List[float] = []
        self.rh_buffer: List[float] = []
        self.timestamp_buffer: List[pd.Timestamp] = []

    def reset(self):
        self.temp_buffer.clear()
        self.press_buffer.clear()
        self.rh_buffer.clear()
        self.timestamp_buffer.clear()

    def update_buffer(self, temp: float, press: float, rh: float, timestamp: Optional[pd.Timestamp] = None):
        """Add new observation to rolling historical buffer."""
        self.temp_buffer.append(temp)
        self.press_buffer.append(press)
        self.rh_buffer.append(rh)
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        self.timestamp_buffer.append(timestamp)

        if len(self.temp_buffer) > self.buffer_size:
            self.temp_buffer.pop(0)
            self.press_buffer.pop(0)
            self.rh_buffer.pop(0)
            self.timestamp_buffer.pop(0)

    def extract_streaming_features(self, temp: float, press: float, rh: float, 
                                  timestamp: Optional[pd.Timestamp] = None) -> np.ndarray:
        """
        Extract feature vector for a single streaming observation using current buffer history.
        Returns 1D numpy array matching FEATURE_NAMES.
        """
        self.update_buffer(temp, press, rh, timestamp)
        n = len(self.temp_buffer)
        t_arr = np.array(self.temp_buffer)
        p_arr = np.array(self.press_buffer)
        rh_arr = np.array(self.rh_buffer)

        # 1. Thermodynamics
        td = AtmosphericPhysics.dew_point(temp, rh)
        dd = temp - td
        vpd = AtmosphericPhysics.vapor_pressure_deficit(temp, rh)
        theta = AtmosphericPhysics.potential_temperature(temp, press)

        # 2. Lagged differences
        d_t1 = temp - t_arr[-2] if n >= 2 else 0.0
        d_p1 = press - p_arr[-2] if n >= 2 else 0.0
        d_rh1 = rh - rh_arr[-2] if n >= 2 else 0.0

        d_t5 = temp - t_arr[-6] if n >= 6 else d_t1
        d_p5 = press - p_arr[-6] if n >= 6 else d_p1
        d_rh5 = rh - rh_arr[-6] if n >= 6 else d_rh1

        d_t15 = temp - t_arr[-16] if n >= 16 else d_t5
        d_p15 = press - p_arr[-16] if n >= 16 else d_p5
        d_rh15 = rh - rh_arr[-16] if n >= 16 else d_rh5

        # 3. Rolling window statistics (up to last 15 observations)
        w_len = min(15, n)
        t_sub = t_arr[-w_len:]
        p_sub = p_arr[-w_len:]
        rh_sub = rh_arr[-w_len:]

        t_mean, t_std = float(np.mean(t_sub)), float(np.std(t_sub)) + 1e-4
        p_mean, p_std = float(np.mean(p_sub)), float(np.std(p_sub)) + 1e-4
        rh_mean, rh_std = float(np.mean(rh_sub)), float(np.std(rh_sub)) + 1e-4

        t_z = (temp - t_mean) / t_std
        p_z = (press - p_mean) / p_std
        rh_z = (rh - rh_mean) / rh_std

        # 4. Cross-parameter coupling
        # T and RH tend to change oppositely; d_temp * d_rh is typically negative
        t_rh_coupling = d_t5 * d_rh5
        # Barometric tendency over 30 samples (or available)
        p_30 = press - p_arr[0] if n >= 30 else (press - p_arr[-1])

        # 5. Diurnal cyclical features
        curr_time = self.timestamp_buffer[-1]
        hour_float = curr_time.hour + curr_time.minute / 60.0
        h_sin = math.sin(2 * math.pi * hour_float / 24.0)
        h_cos = math.cos(2 * math.pi * hour_float / 24.0)

        vec = [
            temp, press, rh,
            td, dd, vpd, theta,
            d_t1, d_p1, d_rh1,
            d_t5, d_p5, d_rh5,
            d_t15, d_p15, d_rh15,
            t_mean, p_mean, rh_mean,
            t_std, p_std, rh_std,
            t_z, p_z, rh_z,
            t_rh_coupling, p_30,
            h_sin, h_cos
        ]
        return np.array(vec, dtype=np.float64)

    @classmethod
    def extract_batch_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract full feature matrix from a historical DataFrame with columns:
        ['timestamp', 'temperature', 'pressure', 'humidity']
        """
        data = df.copy()
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            hour_float = data['timestamp'].dt.hour + data['timestamp'].dt.minute / 60.0
        else:
            hour_float = pd.Series(12.0, index=data.index)

        # Thermodynamics
        thermo = [AtmosphericPhysics.compute_all_thermodynamic_metrics(t, p, rh)
                  for t, p, rh in zip(data['temperature'], data['pressure'], data['humidity'])]
        thermo_df = pd.DataFrame(thermo, index=data.index)

        feat = pd.DataFrame(index=data.index)
        feat['temperature'] = data['temperature']
        feat['pressure'] = data['pressure']
        feat['humidity'] = data['humidity']
        feat['dew_point'] = thermo_df['dew_point_c']
        feat['dew_point_depression'] = thermo_df['dew_point_depression_c']
        feat['vpd'] = thermo_df['vpd_hpa']
        feat['potential_temp'] = thermo_df['potential_temp_c']

        # Differences
        for lag in [1, 5, 15]:
            feat[f'd_temp_{lag}'] = data['temperature'].diff(lag).fillna(0.0)
            feat[f'd_press_{lag}'] = data['pressure'].diff(lag).fillna(0.0)
            feat[f'd_rh_{lag}'] = data['humidity'].diff(lag).fillna(0.0)

        # Rolling 15-window
        feat['temp_rolling_mean_15'] = data['temperature'].rolling(15, min_periods=1).mean()
        feat['press_rolling_mean_15'] = data['pressure'].rolling(15, min_periods=1).mean()
        feat['rh_rolling_mean_15'] = data['humidity'].rolling(15, min_periods=1).mean()

        feat['temp_rolling_std_15'] = data['temperature'].rolling(15, min_periods=1).std().fillna(0.01) + 1e-4
        feat['press_rolling_std_15'] = data['pressure'].rolling(15, min_periods=1).std().fillna(0.01) + 1e-4
        feat['rh_rolling_std_15'] = data['humidity'].rolling(15, min_periods=1).std().fillna(0.01) + 1e-4

        feat['temp_zscore_15'] = (feat['temperature'] - feat['temp_rolling_mean_15']) / feat['temp_rolling_std_15']
        feat['press_zscore_15'] = (feat['pressure'] - feat['press_rolling_mean_15']) / feat['press_rolling_std_15']
        feat['rh_zscore_15'] = (feat['humidity'] - feat['rh_rolling_mean_15']) / feat['rh_rolling_std_15']

        # Coupling & Tendency
        feat['t_rh_coupling'] = feat['d_temp_5'] * feat['d_rh_5']
        feat['barometric_tendency_30'] = data['pressure'].diff(30).fillna(0.0)

        # Diurnal cyclics
        feat['hour_sin'] = np.sin(2 * np.pi * hour_float / 24.0)
        feat['hour_cos'] = np.cos(2 * np.pi * hour_float / 24.0)

        # Return aligned with FEATURE_NAMES
        return feat[cls.FEATURE_NAMES]
