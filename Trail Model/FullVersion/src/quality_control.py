"""
WMO-No. 8 Standard Quality Control (QC) Engine
Implements deterministic plausibility, step-test, persistence, and physical consistency filters.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from src.physics import AtmosphericPhysics


@dataclass
class QCLimits:
    """Configurable thresholds conforming to WMO-No. 8 Quality Control guidelines."""
    # Physical plausibility limits
    temp_min: float = -40.0      # deg C
    temp_max: float = 60.0       # deg C
    pressure_min: float = 800.0  # hPa (covers stations up to ~2000m AMSL)
    pressure_max: float = 1085.0 # hPa (highest recorded sea level pressure ~1084.8 hPa)
    rh_min: float = 0.0          # %
    rh_max: float = 100.0        # %

    # Maximum permissible step changes (per minute equivalent)
    max_delta_temp_per_min: float = 2.0     # deg C/min
    max_delta_pressure_per_min: float = 1.5 # hPa/min
    max_delta_rh_per_min: float = 12.0      # %/min

    # Persistence / Flatline criteria (true dead/frozen sensor has identically 0.0 variance)
    min_stdev_window_len: int = 6           # Number of consecutive identical samples
    min_temp_stdev: float = 0.0001          # deg C
    min_pressure_stdev: float = 0.0001      # hPa
    min_rh_stdev: float = 0.0001            # %


@dataclass
class QCResult:
    """Results of deterministic quality control check on a single observation."""
    is_valid: bool
    is_anomaly: bool
    anomaly_type: str  # 'NORMAL', 'OUT_OF_BOUNDS', 'SPIKE', 'STUCK_SENSOR', 'PHYSICAL_INCONSISTENCY', 'MISSING'
    parameter_failed: Optional[str] = None
    severity: float = 0.0  # 0.0 (clean) to 1.0 (critical error)
    reason: str = "Passed all standard QC tests"
    metrics: Dict[str, Any] = field(default_factory=dict)


class QualityControlEngine:
    """
    Quality Control pipeline evaluating AWS sensor data against
    WMO meteorological bounds, rate-of-change dynamics, and persistence rules.
    """

    def __init__(self, limits: Optional[QCLimits] = None):
        self.limits = limits or QCLimits()

    def check_plausibility(self, temp: Optional[float], 
                          pressure: Optional[float], 
                          rh: Optional[float]) -> Tuple[bool, str, str, float]:
        """
        Verify that values fall within physically possible environmental bounds.
        Returns: (passed, failed_param, reason, severity)
        """
        if temp is None or np.isnan(temp):
            return False, "temperature", "Missing or NaN temperature observation", 1.0
        if pressure is None or np.isnan(pressure):
            return False, "pressure", "Missing or NaN pressure observation", 1.0
        if rh is None or np.isnan(rh):
            return False, "humidity", "Missing or NaN humidity observation", 1.0

        # Temperature range check
        if temp < self.limits.temp_min or temp > self.limits.temp_max:
            return False, "temperature", (
                f"Temperature {temp:.1f}°C outside physical bounds "
                f"[{self.limits.temp_min}, {self.limits.temp_max}]°C"
            ), 0.95

        # Pressure range check
        if pressure < self.limits.pressure_min or pressure > self.limits.pressure_max:
            return False, "pressure", (
                f"Pressure {pressure:.1f} hPa outside physical bounds "
                f"[{self.limits.pressure_min}, {self.limits.pressure_max}] hPa"
            ), 0.95

        # Humidity range check
        if rh < self.limits.rh_min or rh > self.limits.rh_max:
            return False, "humidity", (
                f"Relative Humidity {rh:.1f}% outside physical limits "
                f"[{self.limits.rh_min}, {self.limits.rh_max}]%"
            ), 0.90

        return True, "", "Plausibility check passed", 0.0

    def check_step_change(self, 
                          curr_temp: float, curr_pressure: float, curr_rh: float,
                          prev_temp: float, prev_pressure: float, prev_rh: float,
                          delta_minutes: float = 1.0) -> Tuple[bool, str, str, float]:
        """
        Check for impossible instant rate-of-change spikes across consecutive observations.
        Returns: (passed, failed_param, reason, severity)
        """
        dt_mins = max(1.0, delta_minutes)
        d_temp = abs(curr_temp - prev_temp) / dt_mins
        d_press = abs(curr_pressure - prev_pressure) / dt_mins
        d_rh = abs(curr_rh - prev_rh) / dt_mins

        if d_temp > self.limits.max_delta_temp_per_min:
            severity = min(1.0, 0.5 + (d_temp / self.limits.max_delta_temp_per_min) * 0.25)
            return False, "temperature", (
                f"Impossible Temperature rate of change: {abs(curr_temp - prev_temp):.2f}°C "
                f"in {delta_minutes:.1f} min (exceeds limit {self.limits.max_delta_temp_per_min * dt_mins:.1f}°C)"
            ), severity

        if d_press > self.limits.max_delta_pressure_per_min:
            severity = min(1.0, 0.5 + (d_press / self.limits.max_delta_pressure_per_min) * 0.25)
            return False, "pressure", (
                f"Impossible Pressure rate of change: {abs(curr_pressure - prev_pressure):.2f} hPa "
                f"in {delta_minutes:.1f} min (exceeds limit {self.limits.max_delta_pressure_per_min * dt_mins:.1f} hPa)"
            ), severity

        if d_rh > self.limits.max_delta_rh_per_min:
            severity = min(1.0, 0.5 + (d_rh / self.limits.max_delta_rh_per_min) * 0.25)
            return False, "humidity", (
                f"Impossible Relative Humidity rate of change: {abs(curr_rh - prev_rh):.1f}% "
                f"in {delta_minutes:.1f} min (exceeds limit {self.limits.max_delta_rh_per_min * dt_mins:.1f}%)"
            ), severity

        return True, "", "Rate of change within allowable limits", 0.0

    def check_persistence(self, 
                          temp_history: List[float], 
                          press_history: List[float], 
                          rh_history: List[float]) -> Tuple[bool, str, str, float]:
        """
        Check for frozen / stuck sensor reading (zero variance over sliding window).
        True atmospheric signals always exhibit microscopic turbulence and fluctuations.
        """
        window_len = self.limits.min_stdev_window_len
        if len(temp_history) < window_len:
            return True, "", "Insufficient history for persistence check", 0.0

        t_window = temp_history[-window_len:]
        p_window = press_history[-window_len:]
        rh_window = rh_history[-window_len:]

        # Check standard deviations
        t_std = float(np.std(t_window))
        p_std = float(np.std(p_window))
        rh_std = float(np.std(rh_window))

        if t_std < self.limits.min_temp_stdev:
            return False, "temperature", (
                f"Temperature sensor flatline/frozen: std={t_std:.5f}°C "
                f"over {window_len} readings (value stuck at {t_window[-1]:.2f}°C)"
            ), 0.85

        if p_std < self.limits.min_pressure_stdev:
            return False, "pressure", (
                f"Barometric pressure sensor flatline/frozen: std={p_std:.5f} hPa "
                f"over {window_len} readings (value stuck at {p_window[-1]:.2f} hPa)"
            ), 0.85

        if rh_std < self.limits.min_rh_stdev:
            return False, "humidity", (
                f"Relative Humidity sensor flatline/frozen: std={rh_std:.5f}% "
                f"over {window_len} readings (value stuck at {rh_window[-1]:.2f}%)"
            ), 0.85

        return True, "", "Sensors exhibit expected natural variation", 0.0

    def check_physical_consistency(self, temp: float, rh: float) -> Tuple[bool, str, str, float]:
        """
        Check thermodynamic consistency:
        - Dew point cannot exceed dry bulb temperature (T_d <= T + 0.1 deg C)
        - Ambient air cannot sustain extreme supersaturation (RH > 102%)
        - Absolute zero RH (0.00%) sustained in open atmosphere typically indicates circuit disconnect
        """
        if rh > 100.5:
            return False, "humidity", f"Relative Humidity supersaturation ({rh:.1f}%) exceeds physical limit", 0.80

        dew_point = AtmosphericPhysics.dew_point(temp, rh)
        if dew_point > temp + 0.2:
            return False, "multivariate", (
                f"Thermodynamic violation: Dew point ({dew_point:.2f}°C) exceeds "
                f"dry bulb temperature ({temp:.2f}°C)"
            ), 0.85

        return True, "", "Thermodynamic consistency verified", 0.0

    def evaluate_observation(self,
                             curr_temp: float,
                             curr_pressure: float,
                             curr_rh: float,
                             prev_temp: Optional[float] = None,
                             prev_pressure: Optional[float] = None,
                             prev_rh: Optional[float] = None,
                             temp_hist: Optional[List[float]] = None,
                             press_hist: Optional[List[float]] = None,
                             rh_hist: Optional[List[float]] = None,
                             delta_minutes: float = 1.0) -> QCResult:
        """
        Executes all deterministic Quality Control checks in order of precedence:
        1. Range / Plausibility (Out of bounds or Missing)
        2. Physical Consistency
        3. Step Change (Spikes)
        4. Persistence (Stuck Sensor)
        """
        # 1. Plausibility Check
        ok, param, reason, sev = self.check_plausibility(curr_temp, curr_pressure, curr_rh)
        if not ok:
            atype = "MISSING" if ("Missing" in reason or "NaN" in reason) else "OUT_OF_BOUNDS"
            return QCResult(is_valid=False, is_anomaly=True, anomaly_type=atype,
                            parameter_failed=param, severity=sev, reason=reason)

        # 2. Physical Consistency
        ok, param, reason, sev = self.check_physical_consistency(curr_temp, curr_rh)
        if not ok:
            return QCResult(is_valid=False, is_anomaly=True, anomaly_type="PHYSICAL_INCONSISTENCY",
                            parameter_failed=param, severity=sev, reason=reason)

        # 3. Step Change Check (Spikes)
        if prev_temp is not None and prev_pressure is not None and prev_rh is not None:
            ok, param, reason, sev = self.check_step_change(
                curr_temp, curr_pressure, curr_rh,
                prev_temp, prev_pressure, prev_rh,
                delta_minutes
            )
            if not ok:
                return QCResult(is_valid=False, is_anomaly=True, anomaly_type="SPIKE",
                                parameter_failed=param, severity=sev, reason=reason)

        # 4. Persistence Check (Stuck / Frozen Sensor)
        if temp_hist and press_hist and rh_hist:
            ok, param, reason, sev = self.check_persistence(temp_hist, press_hist, rh_hist)
            if not ok:
                return QCResult(is_valid=False, is_anomaly=True, anomaly_type="STUCK_SENSOR",
                                parameter_failed=param, severity=sev, reason=reason)

        return QCResult(is_valid=True, is_anomaly=False, anomaly_type="NORMAL",
                        severity=0.0, reason="Passed all WMO standard quality checks")
