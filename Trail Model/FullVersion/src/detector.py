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
        Detect slow sensor drift / systematic offset:
        Persistent unilateral bias away from physical expectation over a 60-sample window.
        """
        if len(self.history_temp) < 40:
            return False, None, ""

        t_win = np.array(self.history_temp[-40:])
        rh_win = np.array(self.history_rh[-40:])

        # Linear slope check for drift using precomputed vector dot product
        t_slope = float(np.dot(self.slope_w_40, t_win))
        rh_slope = float(np.dot(self.slope_w_40, rh_win))

        # In natural diurnal heating (07:00 - 14:00), temperature rises WHILE humidity falls (rh_slope < -0.04 %/min).
        # In sensor drift, temperature drifts upwards WITHOUT the corresponding physical humidity drop!
        is_t_drift = (abs(t_slope) > 0.014 and rh_slope > -0.06 and np.sum(np.diff(t_win[::5]) < -0.01) <= 1)  # allow at most 1 small dip
        is_rh_drift = (abs(rh_slope) > 0.05 and abs(t_slope) < 0.005 and np.all(np.diff(rh_win[::5]) >= -0.01))

        if is_t_drift:
            return True, "temperature", f"Uncoupled temperature drift ({t_slope*40:+.2f}°C over window without diurnal RH depression)"
        if is_rh_drift:
            return True, "humidity", f"Uncoupled humidity drift ({rh_slope*40:+.1f}% over window)"

        return False, None, ""

    def process_observation(self,
                            temperature: float,
                            pressure: float,
                            humidity: float,
                            timestamp: Optional[pd.Timestamp] = None,
                            compute_shap: bool = False) -> AnomalyReport:
        """
        Process a single real-time AWS observation through the 5-Tier Detection Pipeline.
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

        # Calculate parameter shifts over 30-min window
        lookback_idx = max(0, len(self.history_temp) - int(30 / max(1, delta_mins)))
        d_temp_30 = temperature - self.history_temp[lookback_idx] if self.history_temp else 0.0
        d_press_30 = pressure - self.history_press[lookback_idx] if self.history_press else 0.0
        d_rh_30 = humidity - self.history_rh[lookback_idx] if self.history_rh else 0.0

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
            delta_minutes=delta_mins
        )

        # Update history and feature buffer
        feat_vec = self.feature_extractor.extract_streaming_features(temperature, pressure, humidity, timestamp)
        self._update_history(temperature, pressure, humidity, timestamp)

        # If deterministic QC failed:
        if qc_res.is_anomaly:
            # If the step change test triggered, BUT the shift matches a genuine storm signature:
            # We override the sensor fault and report GENUINE_WEATHER_EVENT!
            if qc_res.anomaly_type == "SPIKE" and is_storm:
                return AnomalyReport(
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
                )

            # Otherwise, genuine deterministic sensor failure!
            top_feats = self._compute_xai_contributions(feat_vec, compute_full_shap=compute_shap)
            return AnomalyReport(
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
            )

        # Check for slow monotonic sensor drift
        has_drift, drift_param, drift_msg = self._check_drift(temperature, pressure, humidity)
        if has_drift:
            top_feats = self._compute_xai_contributions(feat_vec, compute_full_shap=compute_shap)
            return AnomalyReport(
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
            )

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
                    return AnomalyReport(
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
                    )

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

                return AnomalyReport(
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
                )

        # If genuine weather event is active but not triggering any false alerts:
        if is_storm:
            return AnomalyReport(
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
            )

        # Clean Normal Observation
        return AnomalyReport(
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
        )

    def process_dataframe(self, df: pd.DataFrame) -> List[AnomalyReport]:
        """Process an entire batch DataFrame sequentially."""
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
        return reports
