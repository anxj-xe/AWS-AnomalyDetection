"""
Atmospheric Physics & Meteorological Domain Logic
Implements standard thermodynamic equations and physical consistency checks
based on WMO-No. 8 standards and atmospheric science principles.
"""

import math
from typing import Dict, Tuple, Optional
import numpy as np


class AtmosphericPhysics:
    """
    Thermodynamic and atmospheric calculations for AWS sensors:
    - Temperature (deg C)
    - Atmospheric Pressure (hPa)
    - Relative Humidity (%)
    """

    # Magnus-Tetens coefficients (Sonntag 1990 / Alduchov & Eskridge 1996)
    MAGNUS_A = 6.112      # hPa
    MAGNUS_B = 17.67      # dimensionless
    MAGNUS_C = 243.5      # deg C

    # Sub-zero coefficients over ice
    MAGNUS_B_ICE = 22.46
    MAGNUS_C_ICE = 272.62

    # Standard atmospheric reference pressure
    P0_HPA = 1000.0
    KAPPA = 0.286  # R / Cp for dry air

    @staticmethod
    def saturation_vapor_pressure(temperature_c: float, over_ice: bool = False) -> float:
        """
        Calculate saturation vapor pressure e_s(T) in hPa using the Magnus-Tetens formulation.
        Valid for -40 deg C <= T <= +50 deg C.
        """
        b = AtmosphericPhysics.MAGNUS_B_ICE if (over_ice and temperature_c < 0) else AtmosphericPhysics.MAGNUS_B
        c = AtmosphericPhysics.MAGNUS_C_ICE if (over_ice and temperature_c < 0) else AtmosphericPhysics.MAGNUS_C
        
        # Guard against zero/negative division
        denom = temperature_c + c
        if abs(denom) < 1e-6:
            denom = 1e-6
        return AtmosphericPhysics.MAGNUS_A * math.exp((b * temperature_c) / denom)

    @staticmethod
    def actual_vapor_pressure(temperature_c: float, relative_humidity: float) -> float:
        """
        Calculate actual water vapor pressure e in hPa.
        e = e_s(T) * (RH / 100)
        """
        rh_clamped = max(0.0, min(relative_humidity, 100.0))
        es = AtmosphericPhysics.saturation_vapor_pressure(temperature_c)
        return es * (rh_clamped / 100.0)

    @staticmethod
    def dew_point(temperature_c: float, relative_humidity: float) -> float:
        """
        Calculate Dew Point temperature (T_d in deg C) using Magnus-Tetens inversion.
        T_d is the temperature to which air must be cooled at constant pressure to reach saturation.
        """
        rh_clamped = max(0.01, min(relative_humidity, 100.0))
        b = AtmosphericPhysics.MAGNUS_B
        c = AtmosphericPhysics.MAGNUS_C

        gamma = math.log(rh_clamped / 100.0) + (b * temperature_c) / (c + temperature_c)
        
        # Guard against singularity
        denom = b - gamma
        if abs(denom) < 1e-6:
            return temperature_c
        return (c * gamma) / denom

    @staticmethod
    def dew_point_depression(temperature_c: float, relative_humidity: float) -> float:
        """
        Calculate Dew Point Depression (T - T_d) in deg C.
        Physically, DD >= 0 in unsaturated ambient air.
        DD == 0 represents 100% saturation (fog, cloud base, or condensation).
        """
        td = AtmosphericPhysics.dew_point(temperature_c, relative_humidity)
        return temperature_c - td

    @staticmethod
    def vapor_pressure_deficit(temperature_c: float, relative_humidity: float) -> float:
        """
        Calculate Vapor Pressure Deficit (VPD in hPa).
        VPD = e_s(T) - e
        High VPD indicates dry air with high drying potential; low VPD indicates saturated air.
        """
        es = AtmosphericPhysics.saturation_vapor_pressure(temperature_c)
        e = AtmosphericPhysics.actual_vapor_pressure(temperature_c, relative_humidity)
        return max(0.0, es - e)

    @staticmethod
    def potential_temperature(temperature_c: float, pressure_hpa: float) -> float:
        """
        Calculate Potential Temperature theta in deg C (normalized to 1000 hPa).
        theta = T_kelvin * (1000 / P)^(R/Cp) - 273.15
        Conservative under dry adiabatic processes.
        """
        p_safe = max(500.0, min(pressure_hpa, 1100.0))
        t_kelvin = temperature_c + 273.15
        theta_k = t_kelvin * math.pow(AtmosphericPhysics.P0_HPA / p_safe, AtmosphericPhysics.KAPPA)
        return theta_k - 273.15

    @staticmethod
    def compute_all_thermodynamic_metrics(temperature_c: float, 
                                          pressure_hpa: float, 
                                          relative_humidity: float) -> Dict[str, float]:
        """
        Compute full thermodynamic state vector for an AWS observation.
        """
        es = AtmosphericPhysics.saturation_vapor_pressure(temperature_c)
        e = AtmosphericPhysics.actual_vapor_pressure(temperature_c, relative_humidity)
        td = AtmosphericPhysics.dew_point(temperature_c, relative_humidity)
        dd = temperature_c - td
        vpd = es - e
        theta = AtmosphericPhysics.potential_temperature(temperature_c, pressure_hpa)

        return {
            "sat_vapor_pressure_hpa": round(es, 3),
            "vapor_pressure_hpa": round(e, 3),
            "dew_point_c": round(td, 2),
            "dew_point_depression_c": round(dd, 2),
            "vpd_hpa": round(vpd, 3),
            "potential_temp_c": round(theta, 2)
        }

    @staticmethod
    def detect_storm_signature(delta_temp: float, 
                               delta_pressure: float, 
                               delta_rh: float, 
                               window_minutes: float = 30.0) -> Tuple[bool, float, str]:
        """
        Identifies whether a sudden shift is a GENUINE severe meteorological event
        (e.g., convective thunderstorm downdraft, cold front, or squall line)
        rather than a sensor malfunction.

        Meteorological physics signature of a convective downburst/cold front:
        1. Temperature drops sharply (rain-cooled air downdraft: Delta T <= -3.0 deg C in 30 min).
        2. Relative Humidity surges rapidly towards saturation (Delta RH >= +20% in 30 min, ending > 85%).
        3. Atmospheric Pressure shows a characteristic 'pressure nose' or rapid drop/jump
           (|Delta P| >= 1.2 hPa within 30 min, or rapid drop followed by rise).

        Returns:
            (is_genuine_event, confidence_score [0.0 to 1.0], explanation)
        """
        # Criteria checks
        temp_drop = delta_temp <= -2.5
        rh_surge = delta_rh >= 15.0
        pressure_active = abs(delta_pressure) >= 1.0

        score = 0.0
        reasons = []

        if temp_drop:
            # Score proportional to severity of temperature drop
            score += min(0.4, abs(delta_temp) / 10.0 * 0.4)
            reasons.append(f"Significant convective cooling ({delta_temp:+.1f}°C in {window_minutes:.0f}m)")

        if rh_surge:
            # Score proportional to RH jump
            score += min(0.35, (delta_rh / 40.0) * 0.35)
            reasons.append(f"Rapid humidity surge ({delta_rh:+.1f}% in {window_minutes:.0f}m)")

        if pressure_active:
            # Score proportional to barometric perturbation
            score += min(0.25, (abs(delta_pressure) / 4.0) * 0.25)
            reasons.append(f"Barometric pressure disturbance ({delta_pressure:+.2f} hPa in {window_minutes:.0f}m)")

        # Coupling synergy bonus: If all 3 parameters move in thermodynamic unison
        if temp_drop and rh_surge and pressure_active:
            score = min(1.0, score + 0.15)
            explanation = "Genuine Meteorological Event: Classic Convective Storm/Squall Line signature (" + "; ".join(reasons) + ")"
            return True, round(score, 3), explanation
        elif temp_drop and rh_surge:
            score = min(0.9, score + 0.1)
            explanation = "Likely Meteorological Event: Frontal passage or precipitation onset (" + "; ".join(reasons) + ")"
            return True, round(score, 3), explanation
        else:
            explanation = "No genuine storm signature detected."
            return False, round(score, 3), explanation
