"""
Realistic Automatic Weather Station (AWS) Data Simulator & Anomaly Injection Studio.
Generates meteorologically authentic diurnal cycles, barometric tides, micro-turbulence,
genuine severe thunderstorm events, and controlled sensor fault injections.
"""

from typing import Dict, List, Optional, Tuple, Generator, Any
from dataclasses import dataclass
import math
import random
import numpy as np
import pandas as pd


@dataclass
class Observation:
    timestamp: pd.Timestamp
    temperature: float
    pressure: float
    humidity: float
    true_label: str       # 'NORMAL', 'GENUINE_WEATHER_EVENT', 'SPIKE', 'STUCK_SENSOR', 'SENSOR_DRIFT', 'OUT_OF_BOUNDS', 'PHYSICAL_INCONSISTENCY', 'MISSING'
    faulty_sensor: Optional[str] = None
    is_anomaly: bool = False  # True only for sensor faults


class AWSDataSimulator:
    """
    Simulates high-fidelity Automatic Weather Station telemetry.
    Physics:
    - Diurnal temperature variation (min at 06:00, max at 15:00)
    - Inverse diurnal relative humidity
    - Semi-diurnal atmospheric barometric tide (S2 solar wave)
    - Genuine convective storms (rain cooling, gust front, humidity surge)
    """

    def __init__(self,
                 base_temp: float = 28.0,
                 temp_amplitude: float = 6.0,
                 base_press: float = 1010.0,
                 base_rh: float = 65.0,
                 rh_amplitude: float = 20.0,
                 seed: int = 42):
        self.base_temp = base_temp
        self.temp_amp = temp_amplitude
        self.base_press = base_press
        self.base_rh = base_rh
        self.rh_amp = rh_amplitude
        self.random = random.Random(seed)
        np.random.seed(seed)

        # State tracking for streaming
        self.step_idx = 0
        self.drift_active = False
        self.drift_sensor: Optional[str] = None
        self.drift_offset = 0.0
        self.stuck_active = False
        self.stuck_sensor: Optional[str] = None
        self.stuck_value: float = 0.0

    def generate_point(self, timestamp: pd.Timestamp) -> Tuple[float, float, float]:
        """Generate clean, physically authentic weather measurement for a given timestamp."""
        hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0

        # Diurnal Solar Heating for Temperature (peak ~14:30, trough ~05:30)
        # Shift angle so peak is at hour 14.5
        solar_angle = 2 * math.pi * (hour - 8.5) / 24.0
        t_clean = self.base_temp + self.temp_amp * math.sin(solar_angle)
        # Micro-turbulence noise (+/- 0.08 C)
        t_clean += self.random.gauss(0, 0.05)

        # Diurnal Relative Humidity (inversely correlated to temperature)
        rh_clean = self.base_rh - self.rh_amp * math.sin(solar_angle)
        # Micro-turbulence noise (+/- 0.3 %)
        rh_clean += self.random.gauss(0, 0.2)
        rh_clean = max(10.0, min(95.0, rh_clean))

        # Barometric Atmospheric Tide (S2 solar tide: 12-hour periodicity)
        # Maxima at ~10:00 and ~22:00, minima at ~04:00 and ~16:00
        tide_angle = 4 * math.pi * (hour - 4.0) / 24.0
        p_clean = self.base_press + 1.2 * math.sin(tide_angle)
        # Synoptic slow pressure trend over multiple days
        day_offset = (timestamp.day - 1) * 24 + hour
        synoptic_wave = 2.0 * math.sin(2 * math.pi * day_offset / 96.0)
        p_clean += synoptic_wave
        # Barometer precision noise
        p_clean += self.random.gauss(0, 0.03)

        return round(t_clean, 2), round(p_clean, 2), round(rh_clean, 1)

    def generate_historical_dataset(self,
                                    days: int = 5,
                                    interval_minutes: int = 1,
                                    inject_anomalies: bool = True) -> pd.DataFrame:
        """
        Generate a complete multi-day benchmark dataset with ground-truth labels.
        Optionally injects realistic faults and genuine severe weather events.
        """
        start_time = pd.Timestamp("2026-06-01 00:00:00")
        total_steps = days * 24 * (60 // interval_minutes)

        records = []
        in_storm = False
        storm_steps_left = 0
        storm_t_delta = 0.0
        storm_rh_delta = 0.0
        storm_p_delta = 0.0

        # Anomaly schedule (step indices)
        # We place specific anomalies at known times for evaluation
        anom_schedule: Dict[int, Dict[str, Any]] = {}
        if inject_anomalies:
            # Baseline convective storm episode (Day 1, 16:00)
            storm_start = 16 * 60 // interval_minutes
            for s in range(storm_start, storm_start + (35 // interval_minutes)):
                anom_schedule[s] = {"type": "GENUINE_WEATHER_EVENT", "param": None, "start": storm_start}

            # Evaluation convective storm episode (Day 2, 18:00)
            storm_start_2 = (24 + 18) * 60 // interval_minutes
            for s in range(storm_start_2, storm_start_2 + (35 // interval_minutes)):
                anom_schedule[s] = {"type": "GENUINE_WEATHER_EVENT", "param": None, "start": storm_start_2}

            # Temperature transient spike (Day 2, 09:15)
            s_spike_t = (24 + 9) * 60 // interval_minutes + 15
            anom_schedule[s_spike_t] = {"type": "SPIKE", "param": "temperature", "val": 12.5}

            # Barometer transient spike (Day 2, 14:30)
            s_spike_p = (24 + 14) * 60 // interval_minutes + 30
            anom_schedule[s_spike_p] = {"type": "SPIKE", "param": "pressure", "val": -18.0}

            # Frozen humidity sensor (Day 3, 03:00 to 05:00)
            s_stuck_rh = (48 + 3) * 60 // interval_minutes
            for s in range(s_stuck_rh, s_stuck_rh + (120 // interval_minutes)):
                anom_schedule[s] = {"type": "STUCK_SENSOR", "param": "humidity", "val": 74.2}

            # Temperature sensor calibration drift (Day 4, 10:00 to 18:00)
            s_drift_t = (72 + 10) * 60 // interval_minutes
            for s in range(s_drift_t, s_drift_t + (480 // interval_minutes)):
                elapsed = (s - s_drift_t) * interval_minutes
                drift_amt = (elapsed / 480.0) * 8.0
                anom_schedule[s] = {"type": "SENSOR_DRIFT", "param": "temperature", "val": drift_amt}

            # Out-of-bounds hygrometer reading (Day 4, 22:00)
            s_oob = (72 + 22) * 60 // interval_minutes
            anom_schedule[s_oob] = {"type": "OUT_OF_BOUNDS", "param": "humidity", "val": 118.0}

            # Physically inconsistent thermodynamic state (Day 5, 08:00)
            s_incon = (96 + 8) * 60 // interval_minutes
            anom_schedule[s_incon] = {"type": "PHYSICAL_INCONSISTENCY", "param": "multivariate"}

            # Telemetry packet dropout / missing reading (Day 5, 12:00)
            s_drop = (96 + 12) * 60 // interval_minutes
            anom_schedule[s_drop] = {"type": "MISSING", "param": "temperature", "val": np.nan}

        for step in range(total_steps):
            ts = start_time + pd.Timedelta(minutes=step * interval_minutes)
            t, p, rh = self.generate_point(ts)

            label = "NORMAL"
            faulty_param = None
            is_anom = False

            # Check if this step has an injected condition
            if step in anom_schedule:
                action = anom_schedule[step]
                label = action["type"]
                faulty_param = action.get("param")

                if label == "GENUINE_WEATHER_EVENT":
                    # Severe Thunderstorm: sharp temperature drop (-6 C), pressure nose, RH near 100%
                    s_start = action.get("start", storm_start)
                    progress = (step - s_start) / (35 // interval_minutes)
                    t -= 6.5 * math.sin(progress * math.pi)
                    p -= 2.8 * math.sin(progress * math.pi)
                    rh = min(98.5, rh + 35.0 * math.sin(progress * math.pi))
                    is_anom = False  # Genuine weather, NOT a sensor fault!

                elif label == "SPIKE":
                    is_anom = True
                    if faulty_param == "temperature":
                        t += action["val"]
                    elif faulty_param == "pressure":
                        p += action["val"]
                    elif faulty_param == "humidity":
                        rh += action["val"]

                elif label == "STUCK_SENSOR":
                    is_anom = True
                    if faulty_param == "humidity":
                        rh = action["val"]
                    elif faulty_param == "temperature":
                        t = action["val"]
                    elif faulty_param == "pressure":
                        p = action["val"]

                elif label == "SENSOR_DRIFT":
                    is_anom = True
                    if faulty_param == "temperature":
                        t += action["val"]
                    elif faulty_param == "humidity":
                        rh += action["val"]

                elif label == "OUT_OF_BOUNDS":
                    is_anom = True
                    if faulty_param == "humidity":
                        rh = action["val"]
                    elif faulty_param == "temperature":
                        t = action["val"]

                elif label == "PHYSICAL_INCONSISTENCY":
                    is_anom = True
                    # Set impossible thermodynamic state: extreme cold with dry pressure surge
                    t -= 15.0
                    rh = 15.0

                elif label == "MISSING":
                    is_anom = True
                    if faulty_param == "temperature":
                        t = np.nan

            records.append({
                "timestamp": ts,
                "temperature": t,
                "pressure": p,
                "humidity": rh,
                "ground_truth_label": label,
                "faulty_sensor": faulty_param,
                "is_sensor_fault": is_anom
            })

        return pd.DataFrame(records)

    def stream_generator(self, interval_seconds: float = 1.0) -> Generator[Observation, None, None]:
        """Infinite generator simulating a real-time AWS telemetry stream."""
        curr_time = pd.Timestamp.now()
        while True:
            t, p, rh = self.generate_point(curr_time)

            # Apply runtime injection states if active
            label = "NORMAL"
            fault_param = None
            is_anom = False

            if self.stuck_active and self.stuck_sensor:
                label = "STUCK_SENSOR"
                fault_param = self.stuck_sensor
                is_anom = True
                if self.stuck_sensor == "temperature":
                    t = self.stuck_value
                elif self.stuck_sensor == "pressure":
                    p = self.stuck_value
                elif self.stuck_sensor == "humidity":
                    rh = self.stuck_value

            elif self.drift_active and self.drift_sensor:
                label = "SENSOR_DRIFT"
                fault_param = self.drift_sensor
                is_anom = True
                self.drift_offset += 0.05
                if self.drift_sensor == "temperature":
                    t += self.drift_offset
                elif self.drift_sensor == "humidity":
                    rh += self.drift_offset

            yield Observation(
                timestamp=curr_time,
                temperature=t,
                pressure=p,
                humidity=rh,
                true_label=label,
                faulty_sensor=fault_param,
                is_anomaly=is_anom
            )
            curr_time += pd.Timedelta(minutes=1)
