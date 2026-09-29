"""
Unit tests for Quality Control and Multi-tier Anomaly Detection.
"""

import unittest
import pandas as pd
import numpy as np
from src.quality_control import QualityControlEngine, QCLimits
from src.detector import AWSAnomalyDetector
from src.imputer import AWSImputer
from src.health_monitor import SensorHealthMonitor


class TestQualityControlAndDetector(unittest.TestCase):

    def setUp(self):
        self.qc = QualityControlEngine()
        self.detector = AWSAnomalyDetector()
        self.imputer = AWSImputer()

    def test_out_of_bounds_detection(self):
        # Temperature out of bounds
        res = self.qc.evaluate_observation(curr_temp=75.0, curr_pressure=1013.0, curr_rh=50.0)
        self.assertTrue(res.is_anomaly)
        self.assertEqual(res.anomaly_type, "OUT_OF_BOUNDS")
        self.assertEqual(res.parameter_failed, "temperature")

        # Humidity out of bounds (> 100%)
        res_rh = self.qc.evaluate_observation(curr_temp=25.0, curr_pressure=1013.0, curr_rh=115.0)
        self.assertTrue(res_rh.is_anomaly)
        self.assertEqual(res_rh.anomaly_type, "OUT_OF_BOUNDS")
        self.assertEqual(res_rh.parameter_failed, "humidity")

    def test_step_change_spike_detection(self):
        # Sudden 10 deg C jump in 1 minute
        res = self.qc.evaluate_observation(
            curr_temp=38.0, curr_pressure=1013.0, curr_rh=50.0,
            prev_temp=28.0, prev_pressure=1013.0, prev_rh=50.0,
            delta_minutes=1.0
        )
        self.assertTrue(res.is_anomaly)
        self.assertEqual(res.anomaly_type, "SPIKE")
        self.assertEqual(res.parameter_failed, "temperature")

    def test_persistence_frozen_sensor(self):
        # Identical readings for 8 consecutive steps
        t_hist = [25.4] * 8
        p_hist = [1012.3] * 8
        rh_hist = [60.0] * 8

        res = self.qc.evaluate_observation(
            curr_temp=25.4, curr_pressure=1012.3, curr_rh=60.0,
            prev_temp=25.4, prev_pressure=1012.3, prev_rh=60.0,
            temp_hist=t_hist, press_hist=p_hist, rh_hist=rh_hist
        )
        self.assertTrue(res.is_anomaly)
        self.assertEqual(res.anomaly_type, "STUCK_SENSOR")

    def test_imputation_preserves_physical_bounds(self):
        # Seed imputer with clean data
        for i in range(10):
            self.imputer.update_clean_history(25.0 + i*0.1, 1012.0, 55.0, float(i))

        # Impute a corrupted temperature reading (spike at 45 C)
        imp = self.imputer.impute_point(corrupted_temp=45.0, corrupted_press=1012.0, corrupted_rh=55.0,
                                         faulty_param="temperature", minute_idx=11.0)
        self.assertTrue(imp["was_imputed"])
        # Reconstructed temperature should be close to trend (~26.0 C), NOT 45 C
        self.assertAlmostEqual(imp["imputed_temperature"], 26.0, delta=1.5)
        # Reconstructed humidity must be within [0, 100]
        self.assertGreaterEqual(imp["imputed_humidity"], 0.0)
        self.assertLessEqual(imp["imputed_humidity"], 100.0)

    def test_sensor_health_degradation(self):
        mon = SensorHealthMonitor()
        # Feed 20 clean observations -> should be EXCELLENT
        for _ in range(20):
            mon.record_observation(25.0, 1012.0, 60.0, faulty_sensor=None)
        rep1 = mon.generate_station_report()
        self.assertGreaterEqual(rep1.overall_health, 95.0)
        self.assertEqual(rep1.alert_level, "GREEN")

        # Now feed repeated faults on humidity sensor
        for _ in range(40):
            mon.record_observation(25.0, 1012.0, 110.0, faulty_sensor="humidity")
        rep2 = mon.generate_station_report()
        self.assertLess(rep2.sensors["humidity"].health_score, 60.0)
        self.assertTrue("Humidity" in rep2.sensors["humidity"].recommendation or "Hygrometer" in rep2.sensors["humidity"].recommendation)


if __name__ == '__main__':
    unittest.main()
