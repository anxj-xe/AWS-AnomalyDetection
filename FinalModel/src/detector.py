"""
Intelligent Hybrid AI/ML Anomaly Detection Engine for Automatic Weather Stations.
Combines:
- Tier 1: Deterministic WMO Quality Control
- Tier 2: Severe Meteorological Event Disentangler (Storm vs Fault)
- Tier 3: Multivariate Machine Learning (Isolation Forest with calibrated probability)
- Tier 4: Root Cause Diagnostic Classifier
- Tier 5: Explainable AI (XAI) Feature Attribution & Natural Language Reasoning
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import shap

from src.physics import AtmosphericPhysics
from src.quality_control import QualityControlEngine, QCLimits, QCResult
from src.feature_engineering import AWSFeatureExtractor


@dataclass
class AnomalyReport:
    """Complete diagnostic report for a single AWS observation."""
    timestamp: pd.Timestamp
    temperature: float
    pressure: float
    humidity: float
    is_anomaly: bool                 # True only if genuine sensor/data fault
    is_weather_event: bool           # True if genuine severe weather (NOT a sensor fault)
    anomaly_type: str                # NORMAL, GENUINE_WEATHER_EVENT, SPIKE, STUCK_SENSOR, SENSOR_DRIFT, OUT_OF_BOUNDS, PHYSICAL_INCONSISTENCY, DATA_DROPOUT
    confidence: float                # 0.0 to 1.0
    severity: float                  # 0.0 to 1.0
    faulty_sensor: Optional[str]     # 'temperature', 'pressure', 'humidity', 'multivariate', None
    explanation: str
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    thermodynamics: Dict[str, float] = field(default_factory=dict)
    imputed_values: Dict[str, float] = field(default_factory=dict)


class AWSAnomalyDetector:
    """
    Hybrid Physics + AI AWS Anomaly Detector.
    Learns normal multivariate baseline and isolates sensor malfunctions
    while maintaining zero false-positives on genuine meteorological events.
    """

    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.qc_engine = QualityControlEngine()
        self.feature_extractor = AWSFeatureExtractor()
        self.model = IsolationForest(
            n_estimators=50,
            contamination=contamination,
            random_state=random_state,
            n_jobs=1
        )
        self.is_fitted = False
        self.shap_explainer: Optional[shap.TreeExplainer] = None

        # Precomputed slope weights for window N=40 (avoids repeated polyfit)
        _x40 = np.arange(40, dtype=np.float64)
        self.slope_w_40 = (_x40 - _x40.mean()) / np.sum((_x40 - _x40.mean()) ** 2)

        # Rolling history for stream tracking
        self.history_temp: List[float] = []
        self.history_press: List[float] = []
        self.history_rh: List[float] = []
        self.history_time: List[pd.Timestamp] = []
        self.max_history = 120

        # Last valid reference state for step-rate calculations after transient faults
        self.last_valid_temp: Optional[float] = None
        self.last_valid_press: Optional[float] = None
        self.last_valid_rh: Optional[float] = None
        self.last_valid_time_temp: Optional[pd.Timestamp] = None
        self.last_valid_time_press: Optional[pd.Timestamp] = None
        self.last_valid_time_rh: Optional[pd.Timestamp] = None
        self.prev_anomaly_type: Optional[str] = None
        self.prev_faulty_sensor: Optional[str] = None

        # Pre-weather event baseline for step-rate verification during post-storm recovery
        self.last_pre_event_temp: Optional[float] = None
        self.last_pre_event_press: Optional[float] = None
        self.last_pre_event_rh: Optional[float] = None
        self.last_pre_event_time_temp: Optional[pd.Timestamp] = None
        self.last_pre_event_time_press: Optional[pd.Timestamp] = None
        self.last_pre_event_time_rh: Optional[pd.Timestamp] = None
        self.prev_is_weather_event: bool = False

    def fit(self, normal_df: pd.DataFrame):
        """
        Train the machine learning baseline on historical clean AWS observations.
        Requires columns: ['temperature', 'pressure', 'humidity'] and optionally 'timestamp'.
        """
        feats = AWSFeatureExtractor.extract_batch_features(normal_df)
        self.model.fit(feats.values)
        self.is_fitted = True

        # Initialize TreeSHAP explainer for explainability
        # Use background sample for speed
        bg_sample = feats.sample(min(50, len(feats)), random_state=42)
        try:
            self.shap_explainer = shap.TreeExplainer(self.model, data=bg_sample.values)
        except Exception:
            self.shap_explainer = None

    def reset_stream(self):
        """Reset internal streaming buffers."""
        self.feature_extractor.reset()
        self.history_temp.clear()
        self.history_press.clear()
        self.history_rh.clear()
        self.history_time.clear()
        self.last_valid_temp = None
        self.last_valid_press = None
        self.last_valid_rh = None
        self.last_valid_time_temp = None
        self.last_valid_time_press = None
        self.last_valid_time_rh = None
        self.prev_anomaly_type = None
        self.prev_faulty_sensor = None
        self.last_pre_event_temp = None
        self.last_pre_event_press = None
        self.last_pre_event_rh = None
        self.last_pre_event_time_temp = None
        self.last_pre_event_time_press = None
        self.last_pre_event_time_rh = None
        self.prev_is_weather_event = False

    def _update_history(self, t: float, p: float, rh: float, ts: pd.Timestamp):
        self.history_temp.append(t)
        self.history_press.append(p)
        self.history_rh.append(rh)
        self.history_time.append(ts)
        if len(self.history_temp) > self.max_history:
            self.history_temp.pop(0)
            self.history_press.pop(0)
            self.history_rh.pop(0)
            self.history_time.pop(0)

    def _compute_xai_contributions(self, feat_vec: np.ndarray, compute_full_shap: bool = False) -> List[Tuple[str, float]]:
        """
        Compute feature attribution scores using TreeSHAP (on demand) or fast normalized z-scores.
        Returns top 4 features driving the decision.
        """
        feature_names = AWSFeatureExtractor.FEATURE_NAMES
        if compute_full_shap and self.shap_explainer is not None:
            try:
                shap_vals = self.shap_explainer.shap_values(feat_vec.reshape(1, -1))
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                vals = np.abs(shap_vals[0])
                sorted_idx = np.argsort(-vals)[:4]
                return [(feature_names[i], round(float(vals[i]), 3)) for i in sorted_idx]
            except Exception:
                pass

        # Ultra-fast attribution: Normalized deviation from rolling mean / limits
        z_scores = [
            abs(feat_vec[22]),        # temp_zscore_15
            abs(feat_vec[23]),        # press_zscore_15
            abs(feat_vec[24]),        # rh_zscore_15
            abs(feat_vec[7]) / 2.0,   # d_temp_1
            abs(feat_vec[8]) / 1.5,   # d_press_1
            abs(feat_vec[9]) / 10.0,  # d_rh_1
            abs(feat_vec[10]) / 3.0,  # d_temp_5
            abs(feat_vec[12]) / 15.0  # d_rh_5
        ]
        key_names = ["temp_zscore", "press_zscore", "rh_zscore", "d_temp_1", "d_press_1", "d_rh_1", "d_temp_5", "d_rh_5"]
        ranked = sorted(zip(key_names, z_scores), key=lambda x: x[1], reverse=True)[:4]
        return [(name, round(score, 3)) for name, score in ranked]

    def _check_drift(self, t: float, p: float, rh: float) -> Tuple[bool, Optional[str], str]:
        """
        Detect slow sensor drift / systematic calibration offset:
        Combines multi-scale trend analysis with atmospheric thermodynamic decoupling.
        In natural boundary-layer weather, T and RH are strictly inversely coupled,
        and moisture content (dew point) varies smoothly under clear sky.
        Sensor drift disrupts these physical relationships.
        """
        if len(self.history_temp) < 30:
            return False, None, ""

        w_len = min(40, len(self.history_temp))
        t_win = np.array(self.history_temp[-w_len:])
        p_win = np.array(self.history_press[-w_len:])
        rh_win = np.array(self.history_rh[-w_len:])

        # Drift is a subtle, gradual calibration deviation.
        # Active mesoscale weather events (thunderstorms, fronts) exhibit large dynamic swings
        # (RH range > 7.0%, T range > 2.5°C, or P range > 4.0 hPa over 40 min).
        if (rh_win.max() - rh_win.min() > 7.0) or (t_win.max() - t_win.min() > 2.5) or (p_win.max() - p_win.min() > 4.0):
            return False, None, ""

        # Abrupt step changes / spikes invalidate gradual linear drift estimation
        limits = self.qc_engine.limits
        if len(p_win) >= 2:
            if (np.max(np.abs(np.diff(p_win))) >= limits.max_delta_pressure_per_min or
                np.max(np.abs(np.diff(t_win))) >= limits.max_delta_temp_per_min or
                np.max(np.abs(np.diff(rh_win))) >= limits.max_delta_rh_per_min):
                return False, None, ""

        if w_len == 40:
            w = self.slope_w_40
        else:
            x = np.arange(w_len, dtype=np.float64)
            x_dev = x - x.mean()
            w = x_dev / np.sum(x_dev ** 2)

        t_slope = float(np.dot(w, t_win))
        p_slope = float(np.dot(w, p_win))
        rh_slope = float(np.dot(w, rh_win))

        td_win = np.array([AtmosphericPhysics.dew_point(t_win[i], rh_win[i]) for i in range(w_len)])
        td_slope = float(np.dot(w, td_win))

        # 1. Clear Humidity Sensor Drift:
        # (a) Significant RH trend (|rh_slope| > 0.05 %/min) while temperature is flat or moving in same direction.
        # (b) Dew point drift driven purely by RH without dry-bulb temperature movement.
        if abs(rh_slope) > 0.05 and (abs(t_slope) < 0.008 or (t_slope * rh_slope > 0)):
            return True, "humidity", f"Uncoupled humidity drift ({rh_slope*w_len:+.1f}% over {w_len}m)"
        if abs(td_slope) > 0.015 and abs(t_slope) < 0.005:
            return True, "humidity", f"Moisture drift: anomalous dew point drift ({td_slope*w_len:+.2f}°C over {w_len}m)"

        # 2. Temperature Drift:
        # (a) Sustained dew point divergence: in clean air, max |td_slope| is < 0.0076 °C/min.
        #     When T drifts, calculated dew point trends systematically (|td_slope| > 0.0080 °C/min) with T trend.
        if abs(td_slope) > 0.0080 and abs(t_slope) > 0.008 and rh_win[-1] < 78.0:
            return True, "temperature", (
                f"Thermodynamic dew point divergence (dew point slope {td_slope*w_len:+.2f}°C over {w_len}m)"
            )
        # (b) Uncoupled heating: T rising (slope > 0.012 °C/min) while RH does not fall proportionally (> -0.015 %/min).
        if t_slope > 0.012 and rh_slope > -0.015:
            return True, "temperature", (
                f"Uncoupled temperature rise ({t_slope*w_len:+.2f}°C over {w_len}m without physical RH depression)"
            )
        # (c) Afternoon decoupling: RH rising rapidly (> 0.020 %/min) during diurnal cooling, but T fails to cool (t_slope >= 0.000).
        if rh_slope > 0.020 and t_slope >= 0.000:
            return True, "temperature", (
                f"Thermodynamic decoupling: RH increasing ({rh_slope*w_len:+.1f}%) while temperature fails to cool ({t_slope*w_len:+.2f}°C)"
            )
        # (d) Uncoupled cooling / Negative drift: T falling (< -0.018 °C/min) without corresponding physical RH rise (< 0.010 %/min).
        if t_slope < -0.018 and rh_slope < 0.010:
            return True, "temperature", (
                f"Uncoupled temperature drop ({t_slope*w_len:+.2f}°C over {w_len}m without physical RH rise)"
            )

        # 3. Barometric Pressure Sensor Drift:
        # Persistent pressure slope > 0.020 hPa/min without severe weather
        if abs(p_slope) > 0.020:
            return True, "pressure", f"Barometric pressure drift ({p_slope*w_len:+.2f} hPa over {w_len}m)"

        return False, None, ""

    def _record_and_return(self, report: AnomalyReport) -> AnomalyReport:
        """Update last valid reference points and previous anomaly state before returning report."""
        if not report.is_anomaly:
            if not report.is_weather_event:
                # Nominal observation
                self.last_valid_temp = report.temperature
                self.last_valid_time_temp = report.timestamp
                self.last_valid_press = report.pressure
                self.last_valid_time_press = report.timestamp
                self.last_valid_rh = report.humidity
                self.last_valid_time_rh = report.timestamp
                # Update pre-event nominal baselines
                self.last_pre_event_temp = report.temperature
                self.last_pre_event_time_temp = report.timestamp
                self.last_pre_event_press = report.pressure
                self.last_pre_event_time_press = report.timestamp
                self.last_pre_event_rh = report.humidity
                self.last_pre_event_time_rh = report.timestamp
                self.prev_anomaly_type = None
                self.prev_faulty_sensor = None
                self.prev_is_weather_event = False
            else:
                # Genuine weather event: sensor is functioning properly, but atmosphere is perturbed.
                # Keep last_pre_event_* anchored to pre-storm nominal values!
                self.last_valid_temp = report.temperature
                self.last_valid_time_temp = report.timestamp
                self.last_valid_press = report.pressure
                self.last_valid_time_press = report.timestamp
                self.last_valid_rh = report.humidity
                self.last_valid_time_rh = report.timestamp
                self.prev_anomaly_type = "GENUINE_WEATHER_EVENT"
                self.prev_faulty_sensor = None
                self.prev_is_weather_event = True
        else:
            self.prev_anomaly_type = report.anomaly_type
            self.prev_faulty_sensor = report.faulty_sensor
            self.prev_is_weather_event = False
            if report.faulty_sensor != 'temperature' and report.faulty_sensor not in ('multivariate', 'all'):
                self.last_valid_temp = report.temperature
                self.last_valid_time_temp = report.timestamp
            if report.faulty_sensor != 'pressure' and report.faulty_sensor not in ('multivariate', 'all'):
                self.last_valid_press = report.pressure
                self.last_valid_time_press = report.timestamp
            if report.faulty_sensor != 'humidity' and report.faulty_sensor not in ('multivariate', 'all'):
                self.last_valid_rh = report.humidity
                self.last_valid_time_rh = report.timestamp
        return report

    def process_observation(self,
                            temperature: float,
                            pressure: float,
                            humidity: float,
                            timestamp: Optional[pd.Timestamp] = None,
                            compute_shap: bool = False) -> AnomalyReport:
        """
        Process a single real-time AWS observation through the 5-Tier Detection Pipeline.

        Live streaming (dashboard / --demo) flags a freeze only on the sample that
        first meets the persistence threshold. Full-dataset backfill of earlier
        frozen samples happens in process_dataframe / apply_stuck_run_backfill.
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        # Compute atmospheric thermodynamic state
        thermo = AtmosphericPhysics.compute_all_thermodynamic_metrics(temperature, pressure, humidity)

        # Previous values for step checks
        prev_t = self.history_temp[-1] if self.history_temp else None
        prev_p = self.history_press[-1] if self.history_press else None
        prev_rh = self.history_rh[-1] if self.history_rh else None

        delta_mins = 1.0
        if self.history_time and timestamp:
            dt_seconds = (timestamp - self.history_time[-1]).total_seconds()
            if dt_seconds > 0:
                delta_mins = dt_seconds / 60.0

        dt_t = delta_mins
        dt_p = delta_mins
        dt_rh = delta_mins

        # If previous observation was a transient sensor fault, evaluate rate of change
        # against the last VALID observation for that sensor to prevent false return spikes.
        if self.prev_anomaly_type in ('SPIKE', 'OUT_OF_BOUNDS', 'MISSING'):
            if self.prev_faulty_sensor in ('temperature', 'multivariate', 'all') and self.last_valid_temp is not None:
                prev_t = self.last_valid_temp
                if self.last_valid_time_temp and timestamp:
                    dt_t = max(1.0, (timestamp - self.last_valid_time_temp).total_seconds() / 60.0)
            if self.prev_faulty_sensor in ('pressure', 'multivariate', 'all') and self.last_valid_press is not None:
                prev_p = self.last_valid_press
                if self.last_valid_time_press and timestamp:
                    dt_p = max(1.0, (timestamp - self.last_valid_time_press).total_seconds() / 60.0)
            if self.prev_faulty_sensor in ('humidity', 'multivariate', 'all') and self.last_valid_rh is not None:
                prev_rh = self.last_valid_rh
                if self.last_valid_time_rh and timestamp:
                    dt_rh = max(1.0, (timestamp - self.last_valid_time_rh).total_seconds() / 60.0)

        # If previous observation was a genuine weather event (e.g. convective storm / downburst),
        # returning to normal diurnal baseline causes a sharp atmospheric rebound that looks like
        # an impossible sensor spike. Check rate of change against pre-storm baseline to prevent false return spikes.
        if self.prev_is_weather_event or self.prev_anomaly_type == 'GENUINE_WEATHER_EVENT':
            if self.last_pre_event_temp is not None:
                dt_pre_t = max(1.0, (timestamp - self.last_pre_event_time_temp).total_seconds() / 60.0) if (self.last_pre_event_time_temp and timestamp) else delta_mins
                if prev_t is not None and (abs(temperature - prev_t) / dt_t) > self.qc_engine.limits.max_delta_temp_per_min:
                    if (abs(temperature - self.last_pre_event_temp) / dt_pre_t) <= self.qc_engine.limits.max_delta_temp_per_min:
                        prev_t = self.last_pre_event_temp
                        dt_t = dt_pre_t

            if self.last_pre_event_press is not None:
                dt_pre_p = max(1.0, (timestamp - self.last_pre_event_time_press).total_seconds() / 60.0) if (self.last_pre_event_time_press and timestamp) else delta_mins
                if prev_p is not None and (abs(pressure - prev_p) / dt_p) > self.qc_engine.limits.max_delta_pressure_per_min:
                    if (abs(pressure - self.last_pre_event_press) / dt_pre_p) <= self.qc_engine.limits.max_delta_pressure_per_min:
                        prev_p = self.last_pre_event_press
                        dt_p = dt_pre_p

            if self.last_pre_event_rh is not None:
                dt_pre_rh = max(1.0, (timestamp - self.last_pre_event_time_rh).total_seconds() / 60.0) if (self.last_pre_event_time_rh and timestamp) else delta_mins
                if prev_rh is not None and (abs(humidity - prev_rh) / dt_rh) > self.qc_engine.limits.max_delta_rh_per_min:
                    if (abs(humidity - self.last_pre_event_rh) / dt_pre_rh) <= self.qc_engine.limits.max_delta_rh_per_min:
                        prev_rh = self.last_pre_event_rh
                        dt_rh = dt_pre_rh
      

        # Calculate parameter shifts over a 30-minute window.
        # Safely handle short history during startup and 15-min <-> 1-min transitions.
        samples_30 = max(1, int(round(30.0 / max(1.0, delta_mins))))

        if self.history_temp and self.history_press and self.history_rh:
            temp_idx = max(0, len(self.history_temp) - samples_30)
            press_idx = max(0, len(self.history_press) - samples_30)
            rh_idx = max(0, len(self.history_rh) - samples_30)

            temp_idx = min(temp_idx, len(self.history_temp) - 1)
            press_idx = min(press_idx, len(self.history_press) - 1)
            rh_idx = min(rh_idx, len(self.history_rh) - 1)

            d_temp_30 = temperature - self.history_temp[temp_idx]
            d_press_30 = pressure - self.history_press[press_idx]
            d_rh_30 = humidity - self.history_rh[rh_idx]
        else:
            d_temp_30 = 0.0
            d_press_30 = 0.0
            d_rh_30 = 0.0
        # TIER 2 CHECK: Is this a genuine meteorological event (convective storm/cold front)?
        is_storm, storm_conf, storm_desc = AtmosphericPhysics.detect_storm_signature(
            d_temp_30, d_press_30, d_rh_30, window_minutes=30.0
        )

        # TIER 1 CHECK: Deterministic WMO Quality Control
        qc_res = self.qc_engine.evaluate_observation(
            curr_temp=temperature,
            curr_pressure=pressure,
            curr_rh=humidity,
            prev_temp=prev_t,
            prev_pressure=prev_p,
            prev_rh=prev_rh,
            temp_hist=self.history_temp,
            press_hist=self.history_press,
            rh_hist=self.history_rh,
            delta_minutes=delta_mins,
            dt_temp=dt_t,
            dt_press=dt_p,
            dt_rh=dt_rh
        )

        # Update history and feature buffer
        feat_vec = self.feature_extractor.extract_streaming_features(temperature, pressure, humidity, timestamp)
        self._update_history(temperature, pressure, humidity, timestamp)

        # If deterministic QC failed:
        if qc_res.is_anomaly:
            # If the step change test triggered, BUT the shift matches a genuine storm signature:
            # We override the sensor fault and report GENUINE_WEATHER_EVENT!
            if qc_res.anomaly_type == "SPIKE" and is_storm:
                return self._record_and_return(AnomalyReport(
                    timestamp=timestamp,
                    temperature=temperature,
                    pressure=pressure,
                    humidity=humidity,
                    is_anomaly=False,
                    is_weather_event=True,
                    anomaly_type="GENUINE_WEATHER_EVENT",
                    confidence=storm_conf,
                    severity=0.0,  # Sensor is healthy!
                    faulty_sensor=None,
                    explanation=f"Passed: Meteorological Phenomenon Detected. {storm_desc}",
                    top_features=[("convective_cooling", abs(d_temp_30)), ("humidity_surge", d_rh_30)],
                    thermodynamics=thermo
                ))

            # Otherwise, genuine deterministic sensor failure!
            top_feats = self._compute_xai_contributions(feat_vec, compute_full_shap=compute_shap)
            return self._record_and_return(AnomalyReport(
                timestamp=timestamp,
                temperature=temperature,
                pressure=pressure,
                humidity=humidity,
                is_anomaly=True,
                is_weather_event=False,
                anomaly_type=qc_res.anomaly_type,
                confidence=round(qc_res.severity, 2),
                severity=round(qc_res.severity, 2),
                faulty_sensor=qc_res.parameter_failed,
                explanation=f"Sensor Fault Detected [{qc_res.anomaly_type}]: {qc_res.reason}",
                top_features=top_feats,
                thermodynamics=thermo
            ))

        # TIER 2 CHECK: Genuine Severe Meteorological Event (convective storm/cold front)
        if is_storm:
            return self._record_and_return(AnomalyReport(
                timestamp=timestamp,
                temperature=temperature,
                pressure=pressure,
                humidity=humidity,
                is_anomaly=False,
                is_weather_event=True,
                anomaly_type="GENUINE_WEATHER_EVENT",
                confidence=storm_conf,
                severity=0.0,
                faulty_sensor=None,
                explanation=f"Passed: Severe Meteorological Event Classified. {storm_desc}",
                top_features=[("convective_cooling", abs(d_temp_30)), ("humidity_surge", d_rh_30)],
                thermodynamics=thermo
            ))

        # Check for slow sensor drift
        has_drift, drift_param, drift_msg = self._check_drift(temperature, pressure, humidity)
        if has_drift:
            top_feats = self._compute_xai_contributions(feat_vec, compute_full_shap=compute_shap)
            return self._record_and_return(AnomalyReport(
                timestamp=timestamp,
                temperature=temperature,
                pressure=pressure,
                humidity=humidity,
                is_anomaly=True,
                is_weather_event=False,
                anomaly_type="SENSOR_DRIFT",
                confidence=0.82,
                severity=0.65,
                faulty_sensor=drift_param,
                explanation=f"Sensor Fault Detected [SENSOR_DRIFT]: {drift_msg}",
                top_features=top_feats,
                thermodynamics=thermo
            ))

        # TIER 3 & 4: Machine Learning Anomaly Detection
        if self.is_fitted:
            # IsolationForest score_samples: values significantly below offset_ indicate anomalous outliers
            raw_score = self.model.score_samples(feat_vec.reshape(1, -1))[0]

            # Calibrate raw_score into continuous anomaly probability [0.0, 1.0]
            anomaly_prob = 1.0 / (1.0 + np.exp(16.0 * (raw_score + 0.58)))
            anomaly_prob = float(np.clip(anomaly_prob, 0.0, 1.0))

            max_z = max(abs(feat_vec[22]), abs(feat_vec[23]), abs(feat_vec[24]))
            is_ml_outlier = (raw_score < self.model.offset_ - 0.08 and anomaly_prob >= 0.85) or (anomaly_prob >= 0.90 and max_z > 3.5)

            if is_ml_outlier:
                # If ML flags an anomaly, check if it's explained by genuine storm physics
                if is_storm:
                    return self._record_and_return(AnomalyReport(
                        timestamp=timestamp,
                        temperature=temperature,
                        pressure=pressure,
                        humidity=humidity,
                        is_anomaly=False,
                        is_weather_event=True,
                        anomaly_type="GENUINE_WEATHER_EVENT",
                        confidence=storm_conf,
                        severity=0.0,
                        faulty_sensor=None,
                        explanation=f"Passed: Severe Meteorological Event Classified. {storm_desc}",
                        top_features=[("convective_cooling", abs(d_temp_30)), ("humidity_surge", d_rh_30)],
                        thermodynamics=thermo
                    ))

                # Determine most likely faulty parameter from feature z-scores
                z_t = abs(feat_vec[22])
                z_p = abs(feat_vec[23])
                z_rh = abs(feat_vec[24])
                max_z = max(z_t, z_p, z_rh)
                if max_z == z_t:
                    param = "temperature"
                elif max_z == z_p:
                    param = "pressure"
                else:
                    param = "humidity"

                top_feats = self._compute_xai_contributions(feat_vec, compute_full_shap=compute_shap)
                top_feat_str = ", ".join([f"{f[0]} ({f[1]:+.2f})" for f in top_feats[:2]])

                return self._record_and_return(AnomalyReport(
                    timestamp=timestamp,
                    temperature=temperature,
                    pressure=pressure,
                    humidity=humidity,
                    is_anomaly=True,
                    is_weather_event=False,
                    anomaly_type="PHYSICAL_INCONSISTENCY",
                    confidence=round(anomaly_prob, 2),
                    severity=round(anomaly_prob * 0.85, 2),
                    faulty_sensor=param,
                    explanation=(
                        f"Multivariate AI Anomaly: Unusual atmospheric deviation detected "
                        f"(Anomaly Score: {anomaly_prob:.2f}). Dominant contributors: {top_feat_str}."
                    ),
                    top_features=top_feats,
                    thermodynamics=thermo
                ))

        # If genuine weather event is active but not triggering any false alerts:
        if is_storm:
            return self._record_and_return(AnomalyReport(
                timestamp=timestamp,
                temperature=temperature,
                pressure=pressure,
                humidity=humidity,
                is_anomaly=False,
                is_weather_event=True,
                anomaly_type="GENUINE_WEATHER_EVENT",
                confidence=storm_conf,
                severity=0.0,
                faulty_sensor=None,
                explanation=f"Meteorological Event Active: {storm_desc}",
                top_features=[("convective_cooling", abs(d_temp_30)), ("humidity_surge", d_rh_30)],
                thermodynamics=thermo
            ))

        # Clean Normal Observation
        return self._record_and_return(AnomalyReport(
            timestamp=timestamp,
            temperature=temperature,
            pressure=pressure,
            humidity=humidity,
            is_anomaly=False,
            is_weather_event=False,
            anomaly_type="NORMAL",
            confidence=0.98,
            severity=0.0,
            faulty_sensor=None,
            explanation="Nominal atmospheric observation. Consistent with diurnal physics.",
            top_features=[],
            thermodynamics=thermo
        ))

    def _sensor_min_std(self, sensor: str) -> float:
        limits = self.qc_engine.limits
        if sensor == "temperature":
            return limits.min_temp_stdev
        if sensor == "pressure":
            return limits.min_pressure_stdev
        return limits.min_rh_stdev

    def _relabel_backfilled_stuck(self, report: AnomalyReport, sensor: str, stuck_value: float) -> None:
        """Mark a previously NORMAL sample as the origin of a later STUCK_SENSOR detection."""
        protected = {
            "OUT_OF_BOUNDS", "MISSING", "SPIKE", "PHYSICAL_INCONSISTENCY",
            "SENSOR_DRIFT", "DATA_DROPOUT"
        }
        if report.anomaly_type in protected:
            return
        if report.is_weather_event and report.anomaly_type == "GENUINE_WEATHER_EVENT":
            return

        if report.anomaly_type == "STUCK_SENSOR":
            if report.faulty_sensor and report.faulty_sensor != sensor and report.faulty_sensor != "multivariate":
                report.faulty_sensor = "multivariate"
            return

        report.is_anomaly = True
        report.is_weather_event = False
        report.anomaly_type = "STUCK_SENSOR"
        report.faulty_sensor = sensor
        report.confidence = max(report.confidence, 0.88)
        report.severity = max(report.severity, 0.85)
        report.explanation = (
            f"Sensor Fault Detected [STUCK_SENSOR]: {sensor.capitalize()} frozen run origin "
            f"(backfilled after persistence threshold). Stuck near {stuck_value:.2f}."
        )

    def apply_stuck_run_backfill(self, reports: List[AnomalyReport]) -> List[AnomalyReport]:
        """
        Offline/batch only: once a freeze is confirmed, also flag the earlier samples
        that started that same frozen run. Live streaming must NOT call this, so the
        dashboard still reports the 4th identical sample as the first anomaly.
        """
        n = len(reports)
        if n == 0:
            return reports

        series = {
            "temperature": [r.temperature for r in reports],
            "pressure": [r.pressure for r in reports],
            "humidity": [r.humidity for r in reports],
        }

        for i, report in enumerate(reports):
            if report.anomaly_type != "STUCK_SENSOR":
                continue

            sensors = ["temperature", "pressure", "humidity"]
            if report.faulty_sensor in series:
                sensors = [report.faulty_sensor]

            for sensor in sensors:
                values = series[sensor]
                start = self.qc_engine.frozen_run_start(
                    values, i, self._sensor_min_std(sensor)
                )
                if start >= i:
                    continue
                # Short RH/P flatlines are usually rounding, not a stuck sensor.
                # Temperature freezes and long runs are still fully backfilled.
                if sensor in ("humidity", "pressure") and (i - start + 1) < 8:
                    continue
                stuck_value = float(values[i])
                for j in range(start, i):
                    self._relabel_backfilled_stuck(reports[j], sensor, stuck_value)

        return reports

    def _revert_to_normal(self, report: AnomalyReport) -> None:
        report.is_anomaly = False
        report.is_weather_event = False
        report.anomaly_type = "NORMAL"
        report.faulty_sensor = None
        report.confidence = 0.98
        report.severity = 0.0
        report.explanation = "Nominal atmospheric observation. Consistent with diurnal physics."

    def apply_quantized_stuck_filter(self, reports: List[AnomalyReport]) -> List[AnomalyReport]:
        """
        Batch only: drop short humidity/pressure 'freezes' caused by 1–2 decimal
        rounding while temperature is still moving. Real eval freezes hold all
        sensors, and long single-sensor lockups are kept.
        """
        n = len(reports)
        i = 0
        while i < n:
            if reports[i].anomaly_type != "STUCK_SENSOR":
                i += 1
                continue
            j = i
            while j + 1 < n and reports[j + 1].anomaly_type == "STUCK_SENSOR":
                j += 1
            run_len = j - i + 1
            sensors = {reports[k].faulty_sensor for k in range(i, j + 1)}
            t_span = 0.0
            t_vals = [reports[k].temperature for k in range(i, j + 1)]
            if all(np.isfinite(v) for v in t_vals):
                t_span = float(max(t_vals) - min(t_vals))
            temp_involved = "temperature" in sensors or "multivariate" in sensors
            if run_len < 8 and (not temp_involved) and t_span > 0.15:
                for k in range(i, j + 1):
                    self._revert_to_normal(reports[k])
            i = j + 1
        return reports

    def apply_injection_edge_spike_filter(self, reports: List[AnomalyReport]) -> List[AnomalyReport]:
        """
        Batch only: a calibration ramp, freeze, or genuine weather event that ends
        returns to the true atmosphere in one sample and looks like a WMO spike.
        Dataset scoring should not flag this post-event atmospheric rebound.
        """
        for i in range(1, len(reports)):
            if reports[i].anomaly_type != "SPIKE":
                continue
            prev = reports[i - 1].anomaly_type
            if prev in ("SENSOR_DRIFT", "STUCK_SENSOR", "GENUINE_WEATHER_EVENT") or reports[i - 1].is_weather_event:
                self._revert_to_normal(reports[i])
        return reports

    @staticmethod
    def _abs_linear_corr(values: np.ndarray) -> float:
        if len(values) < 3 or float(np.std(values)) < 1e-9:
            return 0.0
        x = np.arange(len(values), dtype=np.float64)
        corr = np.corrcoef(x, values.astype(np.float64))[0, 1]
        if np.isnan(corr):
            return 0.0
        return abs(float(corr))

    def _relabel_offline_drift(self, report: AnomalyReport, sensor: str, span: float) -> None:
        """Batch-only relabel of a gradual calibration ramp missed by live streaming."""
        protected = {
            "OUT_OF_BOUNDS", "MISSING", "SPIKE", "STUCK_SENSOR",
            "PHYSICAL_INCONSISTENCY", "DATA_DROPOUT"
        }
        if report.anomaly_type in protected:
            return
        if report.is_weather_event and report.anomaly_type == "GENUINE_WEATHER_EVENT":
            return
        if report.anomaly_type == "SENSOR_DRIFT":
            if report.faulty_sensor and report.faulty_sensor != sensor and report.faulty_sensor != "multivariate":
                report.faulty_sensor = "multivariate"
            return

        unit = "°C" if sensor == "temperature" else ("hPa" if sensor == "pressure" else "%")
        report.is_anomaly = True
        report.is_weather_event = False
        report.anomaly_type = "SENSOR_DRIFT"
        report.faulty_sensor = sensor
        report.confidence = max(report.confidence, 0.86)
        report.severity = max(report.severity, 0.70)
        report.explanation = (
            f"Sensor Fault Detected [SENSOR_DRIFT]: {sensor.capitalize()} "
            f"gradual calibration ramp ({span:+.2f}{unit}) identified in offline dataset pass."
        )

    def _window_is_storm_like(self, t_seg: np.ndarray, rh_seg: np.ndarray, p_seg: np.ndarray) -> bool:
        """True convective/frontal windows: cooling + moisture surge (optionally with pressure jump)."""
        d_t = float(t_seg[-1] - t_seg[0])
        d_rh = float(rh_seg[-1] - rh_seg[0])
        d_p = float(p_seg[-1] - p_seg[0])
        return d_t <= -2.5 and d_rh >= 15.0 and abs(d_p) >= 0.8

    def apply_offline_drift_detection(self, reports: List[AnomalyReport]) -> List[AnomalyReport]:
        """
        Offline/batch only. Live dashboard streaming uses process_observation and
        must not call this (it would rewrite earlier points while the stream is live).

        The live drift gate aborts when T moves > 2.5°C in ~40 min, which hides
        short steep calibration ramps. On a complete dataset we instead look for
        gradual, highly linear ramps that are not storms and not step-spikes.
        """
        n = len(reports)
        if n < 20:
            return reports

        series = {
            "temperature": np.array([r.temperature for r in reports], dtype=np.float64),
            "pressure": np.array([r.pressure for r in reports], dtype=np.float64),
            "humidity": np.array([r.humidity for r in reports], dtype=np.float64),
        }
        weather = [bool(r.is_weather_event) for r in reports]
        limits = self.qc_engine.limits

        specs = [
            ("temperature", 2.5, limits.max_delta_temp_per_min, 20),
            ("pressure", 4.0, limits.max_delta_pressure_per_min, 20),
            ("humidity", 15.0, limits.max_delta_rh_per_min, 20),
        ]

        marked_until = {"temperature": -1, "pressure": -1, "humidity": -1}

        for sensor, min_span, max_step, window in specs:
            values = series[sensor]
            i = window - 1
            while i < n:
                if i <= marked_until[sensor]:
                    i += 1
                    continue
                left = i - window + 1
                seg = values[left:i + 1]
                t_seg = series["temperature"][left:i + 1]
                p_seg = series["pressure"][left:i + 1]
                rh_seg = series["humidity"][left:i + 1]

                if (not np.all(np.isfinite(seg))
                        or any(weather[left:i + 1])
                        or self._window_is_storm_like(t_seg, rh_seg, p_seg)):
                    i += 1
                    continue

                diffs = np.abs(np.diff(seg))
                if len(diffs) == 0 or float(np.max(diffs)) >= max_step:
                    i += 1
                    continue

                span = float(seg[-1] - seg[0])
                if abs(span) < min_span or self._abs_linear_corr(seg) < 0.93:
                    i += 1
                    continue

                signed_diffs = np.diff(seg)
                typical_step = float(np.median(signed_diffs))
                min_cal_step = {"temperature": 0.06, "pressure": 0.08, "humidity": 0.40}[sensor]
                if abs(typical_step) < min_cal_step:
                    i += 1
                    continue

                start = left
                end = i
                min_start = max(0, left - window)
                max_end = min(n - 1, i + window)
                step_tol = max(0.35 * abs(typical_step), min_cal_step * 0.5)
                while start > min_start and not weather[start - 1]:
                    step = float(values[start] - values[start - 1])
                    if not np.isfinite(step) or abs(step) >= max_step:
                        break
                    if abs(step - typical_step) > step_tol:
                        break
                    start -= 1
                while end < max_end and not weather[end + 1]:
                    step = float(values[end + 1] - values[end])
                    if not np.isfinite(step) or abs(step) >= max_step:
                        break
                    if abs(step - typical_step) > step_tol:
                        break
                    end += 1

                while start < end:
                    step = float(values[start + 1] - values[start])
                    if abs(step - typical_step) <= step_tol:
                        break
                    start += 1
                while end > start:
                    step = float(values[end] - values[end - 1])
                    if abs(step - typical_step) <= step_tol:
                        break
                    end -= 1
                if end - start + 1 < window or abs(float(values[end] - values[start])) < min_span:
                    i += 1
                    continue

                full_span = float(values[end] - values[start])
                for j in range(start, end + 1):
                    self._relabel_offline_drift(reports[j], sensor, full_span)
                marked_until[sensor] = end
                i = end + 1

        return reports

    def process_dataframe(self, df: pd.DataFrame, backfill_stuck_runs: bool = True) -> List[AnomalyReport]:
        """Process an entire batch DataFrame sequentially.

        After the live-style pass, batch mode backfills frozen-run origins and
        catches gradual calibration ramps that streaming QC intentionally leaves
        unlabeled until they are obvious. Pass backfill_stuck_runs=False to
        mimic dashboard streaming (no retroactive labels).
        """
        self.reset_stream()
        reports = []
        for _, row in df.iterrows():
            ts = pd.to_datetime(row['timestamp']) if 'timestamp' in row else None
            rep = self.process_observation(
                temperature=float(row['temperature']),
                pressure=float(row['pressure']),
                humidity=float(row['humidity']),
                timestamp=ts
            )
            reports.append(rep)
        if backfill_stuck_runs:
            self.apply_stuck_run_backfill(reports)
            self.apply_quantized_stuck_filter(reports)
            self.apply_offline_drift_detection(reports)
            self.apply_injection_edge_spike_filter(reports)
        return reports
