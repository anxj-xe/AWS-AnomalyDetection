"""
Sensor Degradation & Predictive Maintenance Monitor for Automatic Weather Stations.
Tracks long-term sensor stability, signal-to-noise ratio (SNR), drift accumulation,
and computes real-time Sensor Health Index (SHI: 0-100%) with actionable maintenance advice.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class SensorStatus:
    name: str
    health_score: float         # 0.0 to 100.0%
    status: str                 # 'EXCELLENT', 'GOOD', 'DEGRADED', 'CRITICAL'
    fault_count_24h: int
    drift_rate_per_day: float
    noise_level: float
    recommendation: str


@dataclass
class StationHealthReport:
    overall_health: float       # 0.0 to 100.0%
    station_status: str         # 'NOMINAL', 'ATTENTION_REQUIRED', 'CRITICAL_FAILURE'
    sensors: Dict[str, SensorStatus]
    maintenance_summary: str
    alert_level: str            # 'GREEN', 'YELLOW', 'RED'


class SensorHealthMonitor:
    """
    Predictive maintenance analyzer for AWS sensors.
    Monitors high-frequency noise, drift slope, and fault frequency over time.
    """

    def __init__(self, history_window: int = 144):  # ~2.4 hours at 1-min interval
        self.history_window = history_window
        self.fault_history: Dict[str, List[int]] = {
            "temperature": [],
            "pressure": [],
            "humidity": []
        }
        self.noise_history: Dict[str, List[float]] = {
            "temperature": [],
            "pressure": [],
            "humidity": []
        }
        self.raw_history: Dict[str, List[float]] = {
            "temperature": [],
            "pressure": [],
            "humidity": []
        }

    def record_observation(self,
                           temp: float,
                           pressure: float,
                           humidity: float,
                           faulty_sensor: Optional[str] = None):
        """Record an observation and track fault occurrence."""
        for s, val in [("temperature", temp), ("pressure", pressure), ("humidity", humidity)]:
            is_fault = (faulty_sensor == s or faulty_sensor == "multivariate")
            self.fault_history[s].append(1 if is_fault else 0)
            self.raw_history[s].append(val)

            # Trim history
            if len(self.fault_history[s]) > self.history_window:
                self.fault_history[s].pop(0)
                self.raw_history[s].pop(0)

    def evaluate_sensor_health(self, sensor_name: str) -> SensorStatus:
        """Calculate health index and predictive maintenance guidance for a specific sensor."""
        faults = self.fault_history[sensor_name]
        raws = self.raw_history[sensor_name]
        n = len(faults)

        if n < 5:
            return SensorStatus(
                name=sensor_name,
                health_score=100.0,
                status="EXCELLENT",
                fault_count_24h=0,
                drift_rate_per_day=0.0,
                noise_level=0.0,
                recommendation="Sensor initializing. Performance nominal."
            )

        # 1. Fault rate penalty
        fault_rate = sum(faults) / n  # 0.0 to 1.0
        fault_penalty = fault_rate * 60.0

        # 2. High-frequency noise level (difference variance)
        diffs = np.diff(raws)
        noise = float(np.std(diffs)) if len(diffs) > 1 else 0.0

        noise_penalty = 0.0
        if sensor_name == "temperature" and noise > 1.0:
            noise_penalty = min(25.0, (noise - 1.0) * 15.0)
        elif sensor_name == "pressure" and noise > 0.8:
            noise_penalty = min(25.0, (noise - 0.8) * 20.0)
        elif sensor_name == "humidity" and noise > 4.0:
            noise_penalty = min(25.0, (noise - 4.0) * 5.0)

        # 3. Drift penalty
        drift_rate = 0.0
        drift_penalty = 0.0
        if len(raws) >= 30:
            x = np.arange(len(raws))
            slope, _ = np.polyfit(x, raws, 1)
            drift_rate = slope * 60.0 * 24.0  # approximate change per day
            if sensor_name == "temperature" and abs(drift_rate) > 20.0:
                drift_penalty = min(25.0, abs(drift_rate) * 0.5)
            elif sensor_name == "humidity" and abs(drift_rate) > 40.0:
                drift_penalty = min(25.0, abs(drift_rate) * 0.4)

        # Final health score
        health = max(0.0, min(100.0, 100.0 - (fault_penalty + noise_penalty + drift_penalty)))

        # Status categorization & Actionable Recommendations
        if health >= 90.0:
            status = "EXCELLENT"
            rec = "Operating nominally. Continue standard periodic telemetry checks."
        elif health >= 75.0:
            status = "GOOD"
            rec = "Minor signal variance detected. Schedule routine inspection during next maintenance cycle."
        elif health >= 50.0:
            status = "DEGRADED"
            if sensor_name == "temperature":
                rec = "Elevated thermal noise / intermittent spikes. Inspect RTD wiring, shielding, and radiation screen."
            elif sensor_name == "pressure":
                rec = "Barometric signal variance high. Check barometer vent tube and static pressure port for clogging."
            else:
                rec = "Hygrometer response slowing or drifting. Clean or replace sintered filter cap; verify capacitive element."
        else:
            status = "CRITICAL"
            rec = f"Immediate field service required! {sensor_name.capitalize()} sensor exhibiting persistent failure or dead lockup."

        return SensorStatus(
            name=sensor_name,
            health_score=round(health, 1),
            status=status,
            fault_count_24h=sum(faults),
            drift_rate_per_day=round(drift_rate, 2),
            noise_level=round(noise, 3),
            recommendation=rec
        )

    def generate_station_report(self) -> StationHealthReport:
        """Produce station-wide health diagnostic report."""
        t_stat = self.evaluate_sensor_health("temperature")
        p_stat = self.evaluate_sensor_health("pressure")
        rh_stat = self.evaluate_sensor_health("humidity")

        # Weighted score: Temperature 35%, Pressure 35%, Humidity 30%
        overall = 0.35 * t_stat.health_score + 0.35 * p_stat.health_score + 0.30 * rh_stat.health_score
        overall = round(overall, 1)

        if overall >= 85.0:
            st_status = "NOMINAL"
            alert = "GREEN"
            summary = "All station sensors functioning normally within WMO operational tolerances."
        elif overall >= 60.0:
            st_status = "ATTENTION_REQUIRED"
            alert = "YELLOW"
            summary = "One or more sensors show signs of degradation. Preemptive maintenance recommended."
        else:
            st_status = "CRITICAL_FAILURE"
            alert = "RED"
            summary = "Critical sensor hardware failure in progress. Telemetry reliability compromised."

        return StationHealthReport(
            overall_health=overall,
            station_status=st_status,
            sensors={
                "temperature": t_stat,
                "pressure": p_stat,
                "humidity": rh_stat
            },
            maintenance_summary=summary,
            alert_level=alert
        )
