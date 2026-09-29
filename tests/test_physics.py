"""
Unit tests for Atmospheric Physics & Thermodynamic Formulations.
"""

import unittest
import math
from src.physics import AtmosphericPhysics


class TestAtmosphericPhysics(unittest.TestCase):

    def test_saturation_vapor_pressure(self):
        # At 0 deg C, e_s should be ~ 6.112 hPa
        es_0 = AtmosphericPhysics.saturation_vapor_pressure(0.0)
        self.assertAlmostEqual(es_0, 6.112, places=2)

        # At 20 deg C, e_s should be ~ 23.37 hPa (standard psychrometric constant)
        es_20 = AtmosphericPhysics.saturation_vapor_pressure(20.0)
        self.assertAlmostEqual(es_20, 23.37, delta=0.5)

    def test_dew_point_calculation(self):
        # At 20 deg C and 100% RH, dew point MUST equal dry bulb temperature
        td_sat = AtmosphericPhysics.dew_point(20.0, 100.0)
        self.assertAlmostEqual(td_sat, 20.0, places=1)

        # At 20 deg C and 50% RH, dew point is approximately 9.3 deg C
        td_50 = AtmosphericPhysics.dew_point(20.0, 50.0)
        self.assertAlmostEqual(td_50, 9.3, delta=0.5)

    def test_dew_point_depression_physical_law(self):
        # Dew point depression must be non-negative in unsaturated air
        temps = [-10.0, 0.0, 15.0, 30.0, 45.0]
        rhs = [20.0, 50.0, 80.0, 99.0]
        for t in temps:
            for rh in rhs:
                dd = AtmosphericPhysics.dew_point_depression(t, rh)
                self.assertGreaterEqual(dd, -0.05, f"Violated at T={t}, RH={rh}")

    def test_storm_signature_detection(self):
        # Genuine severe convective storm downdraft:
        # Temp drops -5.5 C, RH surges +30 %, Pressure dips -2.0 hPa
        is_storm, conf, desc = AtmosphericPhysics.detect_storm_signature(
            delta_temp=-5.5, delta_pressure=-2.0, delta_rh=30.0, window_minutes=30.0
        )
        self.assertTrue(is_storm)
        self.assertGreater(conf, 0.7)
        self.assertIn("Convective Storm", desc)

        # Isolated sensor spike (not a storm!)
        is_storm_spike, _, _ = AtmosphericPhysics.detect_storm_signature(
            delta_temp=12.0, delta_pressure=0.1, delta_rh=-1.0, window_minutes=30.0
        )
        self.assertFalse(is_storm_spike)

        # Normal diurnal shift
        is_storm_norm, _, _ = AtmosphericPhysics.detect_storm_signature(
            delta_temp=0.4, delta_pressure=0.05, delta_rh=-1.2, window_minutes=30.0
        )
        self.assertFalse(is_storm_norm)


if __name__ == '__main__':
    unittest.main()
