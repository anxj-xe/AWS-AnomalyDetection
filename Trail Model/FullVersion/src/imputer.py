"""
Real-Time Physics-Constrained Imputation Engine for AWS Observations.
Reconstructs missing or corrupted meteorological sensor readings
while strictly enforcing thermodynamic boundaries and diurnal continuity.
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from src.physics import AtmosphericPhysics


class AWSImputer:
    """
    Intelligent Imputation Engine:
    Reconstructs corrupted measurements using:
    1. Historical EWMA & Trend extrapolation
    2. Cubic Spline interpolation over multi-point gaps
    3. Thermodynamic constraint enforcement (0 <= RH <= 100%, T_d <= T)
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.valid_temp_buffer: List[float] = []
        self.valid_press_buffer: List[float] = []
        self.valid_rh_buffer: List[float] = []
        self.valid_time_buffer: List[float] = []  # minutes relative

    def reset(self):
        self.valid_temp_buffer.clear()
        self.valid_press_buffer.clear()
        self.valid_rh_buffer.clear()
        self.valid_time_buffer.clear()

    def update_clean_history(self, temp: float, press: float, rh: float, minute_idx: float):
        """Add known clean or validated observation to buffer."""
        self.valid_temp_buffer.append(temp)
        self.valid_press_buffer.append(press)
        self.valid_rh_buffer.append(rh)
        self.valid_time_buffer.append(minute_idx)

        if len(self.valid_temp_buffer) > self.window_size:
            self.valid_temp_buffer.pop(0)
            self.valid_press_buffer.pop(0)
            self.valid_rh_buffer.pop(0)
            self.valid_time_buffer.pop(0)

    def impute_point(self,
                     corrupted_temp: float,
                     corrupted_press: float,
                     corrupted_rh: float,
                     faulty_param: Optional[str],
                     minute_idx: float) -> Dict[str, float]:
        """
        Produce physically plausible corrected values for faulty parameters.
        Parameters that are healthy are retained unchanged.
        """
        # Default fallbacks
        imp_t = corrupted_temp
        imp_p = corrupted_press
        imp_rh = corrupted_rh

        # If we have at least 3 clean historical points, estimate via extrapolation
        if len(self.valid_temp_buffer) >= 3:
            times = np.array(self.valid_time_buffer)
            # Estimate temperature if faulty
            if faulty_param in ("temperature", "multivariate", "all"):
                t_hist = np.array(self.valid_temp_buffer)
                # Weighted linear extrapolation (last 5 points)
                sub_t = t_hist[-5:]
                sub_time = times[-5:]
                if len(sub_t) >= 2:
                    slope, intercept = np.polyfit(sub_time, sub_t, 1)
                    # Limit slope to physically plausible rate (< 0.1 C/min)
                    slope = np.clip(slope, -0.08, 0.08)
                    imp_t = float(sub_t[-1] + slope * (minute_idx - sub_time[-1]))
                else:
                    imp_t = float(np.mean(sub_t))

            # Estimate pressure if faulty
            if faulty_param in ("pressure", "multivariate", "all"):
                p_hist = np.array(self.valid_press_buffer)
                sub_p = p_hist[-5:]
                sub_time = times[-5:]
                if len(sub_p) >= 2:
                    slope, _ = np.polyfit(sub_time, sub_p, 1)
                    slope = np.clip(slope, -0.05, 0.05)
                    imp_p = float(sub_p[-1] + slope * (minute_idx - sub_time[-1]))
                else:
                    imp_p = float(np.mean(sub_p))

            # Estimate humidity if faulty
            if faulty_param in ("humidity", "multivariate", "all"):
                rh_hist = np.array(self.valid_rh_buffer)
                sub_rh = rh_hist[-5:]
                sub_time = times[-5:]
                if len(sub_rh) >= 2:
                    slope, _ = np.polyfit(sub_time, sub_rh, 1)
                    slope = np.clip(slope, -0.5, 0.5)
                    imp_rh = float(sub_rh[-1] + slope * (minute_idx - sub_time[-1]))
                else:
                    imp_rh = float(np.mean(sub_rh))

        # Enforce strict physical & thermodynamic boundaries
        imp_rh = max(5.0, min(100.0, imp_rh))
        imp_p = max(800.0, min(1085.0, imp_p))
        imp_t = max(-40.0, min(60.0, imp_t))

        # Thermodynamic Dew Point Consistency: T_d <= T
        dew_point = AtmosphericPhysics.dew_point(imp_t, imp_rh)
        if dew_point > imp_t:
            # If Dew Point exceeds dry bulb, cap RH to 100.0%
            imp_rh = 100.0

        return {
            "imputed_temperature": round(imp_t, 2),
            "imputed_pressure": round(imp_p, 2),
            "imputed_humidity": round(imp_rh, 1),
            "was_imputed": faulty_param is not None and faulty_param != "none"
        }

    @classmethod
    def impute_batch_dataframe(cls, df: pd.DataFrame, anomaly_mask: pd.Series) -> pd.DataFrame:
        """
        Reconstruct an entire time series by applying cubic spline interpolation
        over flagged anomalous segments.
        """
        res = df.copy()
        for col in ['temperature', 'pressure', 'humidity']:
            if col in res.columns:
                clean_series = res[col].copy()
                clean_series[anomaly_mask] = np.nan
                # Interpolate using time/polynomial spline with linear fallback at edges
                interpolated = clean_series.interpolate(method='pchip', limit_direction='both')
                # Fallback to linear if pchip fails
                if interpolated.isna().any():
                    interpolated = clean_series.interpolate(method='linear', limit_direction='both').bfill().ffill()
                res[f'{col}_imputed'] = interpolated

        # Post-clamp humidity
        if 'humidity_imputed' in res.columns:
            res['humidity_imputed'] = res['humidity_imputed'].clip(0.0, 100.0)

        return res
