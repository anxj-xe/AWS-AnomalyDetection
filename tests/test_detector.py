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
from src.data_simulator import AWSDataSimulator


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

    def test_rapid_4sample_flatline_detection(self):
        # Exactly 4 identical readings should trigger flatline detection immediately
        t_hist = [24.12, 24.12, 24.12]  # 3 in history + 1 current = 4 total
        p_hist = [1012.0, 1012.1, 1012.2]
        rh_hist = [55.0, 55.2, 55.4]

        res = self.qc.evaluate_observation(
            curr_temp=24.12, curr_pressure=1012.3, curr_rh=55.5,
            prev_temp=24.12, prev_pressure=1012.2, prev_rh=55.4,
            temp_hist=t_hist, press_hist=p_hist, rh_hist=rh_hist
        )
        self.assertTrue(res.is_anomaly)
        self.assertEqual(res.anomaly_type, "STUCK_SENSOR")
        self.assertEqual(res.parameter_failed, "temperature")

    def test_noisy_frozen_deadband_detection(self):
        # Sensor stuck with minor ADC bit flicker (std ~ 0.001) should be caught as frozen deadband
        t_hist = [25.400, 25.402, 25.400, 25.401, 25.400]
        p_hist = [1012.0 + i*0.1 for i in range(5)]
        rh_hist = [60.0 + i*0.3 for i in range(5)]

        res = self.qc.evaluate_observation(
            curr_temp=25.401, curr_pressure=1012.5, curr_rh=61.5,
            temp_hist=t_hist, press_hist=p_hist, rh_hist=rh_hist
        )
        self.assertTrue(res.is_anomaly)
        self.assertEqual(res.anomaly_type, "STUCK_SENSOR")
        self.assertEqual(res.parameter_failed, "temperature")

    def test_uncoupled_temperature_drift_detection(self):
        # Progressive temperature rise (+0.02 C/min) without diurnal RH drop
        det = AWSAnomalyDetector()
        for i in range(40):
            t = 25.0 + i * 0.02
            p = 1013.0 + 0.05 * (i % 3 - 1)
            rh = 60.0 + 0.1 * (i % 2 - 0.5)
            rep = det.process_observation(temperature=t, pressure=p, humidity=rh)
        self.assertTrue(rep.is_anomaly)
        self.assertEqual(rep.anomaly_type, "SENSOR_DRIFT")
        self.assertEqual(rep.faulty_sensor, "temperature")

    def test_uncoupled_humidity_drift_detection(self):
        # Progressive humidity drift (+0.08 %/min) while temperature remains steady
        det = AWSAnomalyDetector()
        for i in range(40):
            t = 25.0 + 0.03 * (i % 3 - 1)
            p = 1013.0 + 0.05 * (i % 2 - 0.5)
            rh = 50.0 + i * 0.08
            rep = det.process_observation(temperature=t, pressure=p, humidity=rh)
        self.assertTrue(rep.is_anomaly)
        self.assertEqual(rep.anomaly_type, "SENSOR_DRIFT")
        self.assertEqual(rep.faulty_sensor, "humidity")

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

    def _varying_prefix(self, n: int = 12):
        rows = []
        t0 = pd.Timestamp("2024-06-01 00:00:00")
        for i in range(n):
            rows.append({
                "timestamp": t0 + pd.Timedelta(minutes=i),
                "temperature": 20.0 + i * 0.31,
                "pressure": 1010.0 + i * 0.07,
                "humidity": 48.0 + i * 0.22,
            })
        return rows, t0

    def test_streaming_stuck_flags_only_from_threshold(self):
        # Live / dashboard path: first 3 identical samples stay NORMAL; 4th+ are STUCK.
        det = AWSAnomalyDetector()
        prefix, _ = self._varying_prefix(10)
        for row in prefix:
            det.process_observation(row["temperature"], row["pressure"], row["humidity"], row["timestamp"])

        last = prefix[-1]
        stuck_t = last["temperature"] + 0.4
        freeze = []
        for i in range(6):
            freeze.append(det.process_observation(
                stuck_t, last["pressure"] + (i + 1) * 0.06, last["humidity"] + (i + 1) * 0.18,
                last["timestamp"] + pd.Timedelta(minutes=i + 1)
            ))

        self.assertFalse(freeze[0].is_anomaly)
        self.assertFalse(freeze[1].is_anomaly)
        self.assertFalse(freeze[2].is_anomaly)
        self.assertEqual(freeze[0].anomaly_type, "NORMAL")
        for rep in freeze[3:]:
            self.assertTrue(rep.is_anomaly)
            self.assertEqual(rep.anomaly_type, "STUCK_SENSOR")
            self.assertEqual(rep.faulty_sensor, "temperature")

    def test_batch_stuck_backfills_run_origin(self):
        # CLI / full-dataset path: the whole frozen run is labeled STUCK_SENSOR.
        prefix, t0 = self._varying_prefix(10)
        last = prefix[-1]
        stuck_t = last["temperature"] + 0.4
        rows = list(prefix)
        for i in range(6):
            rows.append({
                "timestamp": t0 + pd.Timedelta(minutes=10 + i),
                "temperature": stuck_t,
                "pressure": last["pressure"] + (i + 1) * 0.06,
                "humidity": last["humidity"] + (i + 1) * 0.18,
            })
        df = pd.DataFrame(rows)
        reports = AWSAnomalyDetector().process_dataframe(df)

        freeze = reports[10:16]
        for rep in freeze:
            self.assertTrue(rep.is_anomaly, msg=rep.explanation)
            self.assertEqual(rep.anomaly_type, "STUCK_SENSOR")
            self.assertEqual(rep.faulty_sensor, "temperature")
        self.assertNotEqual(reports[9].anomaly_type, "STUCK_SENSOR")

    def test_batch_can_disable_stuck_backfill(self):
        prefix, t0 = self._varying_prefix(10)
        last = prefix[-1]
        stuck_t = last["temperature"] + 0.4
        rows = list(prefix)
        for i in range(6):
            rows.append({
                "timestamp": t0 + pd.Timedelta(minutes=10 + i),
                "temperature": stuck_t,
                "pressure": last["pressure"] + (i + 1) * 0.06,
                "humidity": last["humidity"] + (i + 1) * 0.18,
            })
        df = pd.DataFrame(rows)
        reports = AWSAnomalyDetector().process_dataframe(df, backfill_stuck_runs=False)
        freeze = reports[10:16]
        self.assertFalse(freeze[0].is_anomaly)
        self.assertFalse(freeze[2].is_anomaly)
        self.assertTrue(freeze[3].is_anomaly)

    def test_batch_steep_drift_is_caught_offline(self):
        # eval1-style: +8°C over 30 min. Live gate (T range > 2.5°C) hides most of this.
        sim = AWSDataSimulator(seed=7)
        rows = []
        t0 = pd.Timestamp("2024-07-01 08:00:00")
        for i in range(80):
            ts = t0 + pd.Timedelta(minutes=i)
            t, p, rh = sim.generate_point(ts)
            rows.append({"timestamp": ts, "temperature": t, "pressure": p, "humidity": rh})
        for i in range(30):
            rows[40 + i]["temperature"] += 8.0 * i / 29.0
        df = pd.DataFrame(rows)

        live = AWSAnomalyDetector().process_dataframe(df, backfill_stuck_runs=False)
        live_catch = sum(1 for r in live[40:70] if r.is_anomaly)
        batch = AWSAnomalyDetector().process_dataframe(df)
        batch_catch = sum(1 for r in batch[40:70] if r.is_anomaly and r.anomaly_type == "SENSOR_DRIFT")

        self.assertLess(live_catch, 12)
        self.assertGreaterEqual(batch_catch, 24)
        self.assertEqual(batch[20].anomaly_type, "NORMAL")
        self.assertNotEqual(batch[39].anomaly_type, "SENSOR_DRIFT")

    def test_batch_short_rh_rounding_is_not_stuck(self):
        prefix, t0 = self._varying_prefix(12)
        rows = list(prefix)
        last = prefix[-1]
        for i in range(5):
            rows.append({
                "timestamp": t0 + pd.Timedelta(minutes=12 + i),
                "temperature": last["temperature"] + (i + 1) * 0.25,
                "pressure": last["pressure"] + (i + 1) * 0.08,
                "humidity": 81.2,
            })
        df = pd.DataFrame(rows)
        reports = AWSAnomalyDetector().process_dataframe(df)
        rh_run = reports[12:17]
        self.assertFalse(any(r.anomaly_type == "STUCK_SENSOR" for r in rh_run))

    def test_streaming_humidity_freeze_still_flags_at_fourth(self):
        det = AWSAnomalyDetector()
        prefix, _ = self._varying_prefix(10)
        for row in prefix:
            det.process_observation(row["temperature"], row["pressure"], row["humidity"], row["timestamp"])
        last = prefix[-1]
        freeze = []
        for i in range(4):
            freeze.append(det.process_observation(
                last["temperature"] + (i + 1) * 0.2,
                last["pressure"] + (i + 1) * 0.05,
                77.7,
                last["timestamp"] + pd.Timedelta(minutes=i + 1),
            ))
        self.assertFalse(freeze[2].is_anomaly)
        self.assertTrue(freeze[3].is_anomaly)
        self.assertEqual(freeze[3].anomaly_type, "STUCK_SENSOR")

    def test_batch_all_sensor_freeze_is_kept(self):
        prefix, t0 = self._varying_prefix(10)
        last = prefix[-1]
        rows = list(prefix)
        for i in range(12):
            rows.append({
                "timestamp": t0 + pd.Timedelta(minutes=10 + i),
                "temperature": last["temperature"],
                "pressure": last["pressure"],
                "humidity": last["humidity"],
            })
        df = pd.DataFrame(rows)
        reports = AWSAnomalyDetector().process_dataframe(df)
        freeze = reports[10:22]
        stuck = sum(1 for r in freeze if r.anomaly_type == "STUCK_SENSOR")
        self.assertGreaterEqual(stuck, 10)

    def test_streaming_steep_drift_does_not_rewrite_history(self):
        sim = AWSDataSimulator(seed=7)
        det = AWSAnomalyDetector()
        t0 = pd.Timestamp("2024-07-01 08:00:00")
        reports = []
        for i in range(80):
            ts = t0 + pd.Timedelta(minutes=i)
            t, p, rh = sim.generate_point(ts)
            if 40 <= i < 70:
                t += 8.0 * (i - 40) / 29.0
            reports.append(det.process_observation(t, p, rh, ts))
        # Live path never retroactively labels the start of the ramp.
        self.assertFalse(reports[40].is_anomaly)

    def test_barometer_spike_recovery_returns_to_normal(self):
        # When a single-sample barometer pressure drop occurs, it must be flagged as a SPIKE,
        # and the immediate next step (+1m) returning to normal must be NORMAL (not SPIKE or SENSOR_DRIFT).
        sim = AWSDataSimulator(seed=101)
        det = AWSAnomalyDetector()
        clean_hist = sim.generate_historical_dataset(days=2, interval_minutes=5, inject_anomalies=False)
        det.fit(clean_hist)

        base_time = pd.Timestamp("2026-06-01 12:00:00")
        for i in range(30):
            t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=i))
            det.process_observation(t, p, rh, base_time + pd.Timedelta(minutes=i))

        # Injected Barometer Spike (-16.0 hPa)
        t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=30))
        spike_rep = det.process_observation(t, p - 16.0, rh, base_time + pd.Timedelta(minutes=30))
        self.assertTrue(spike_rep.is_anomaly)
        self.assertEqual(spike_rep.anomaly_type, "SPIKE")
        self.assertEqual(spike_rep.faulty_sensor, "pressure")

        # Step +1 min (nominal observation)
        t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=31))
        step1_rep = det.process_observation(t, p, rh, base_time + pd.Timedelta(minutes=31))
        self.assertFalse(step1_rep.is_anomaly, msg=f"Expected NORMAL at +1m but got: {step1_rep.anomaly_type}, {step1_rep.explanation}")
        self.assertEqual(step1_rep.anomaly_type, "NORMAL")

        # Subsequent steps (+2m to +5m) must stay NORMAL
        for i in range(2, 6):
            t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=30 + i))
            rep = det.process_observation(t, p, rh, base_time + pd.Timedelta(minutes=30 + i))
            self.assertFalse(rep.is_anomaly, msg=f"Expected NORMAL at +{i}m but got: {rep.anomaly_type}, {rep.explanation}")
            self.assertEqual(rep.anomaly_type, "NORMAL")

    def test_genuine_weather_event_recovery_returns_to_normal(self):
        # When a genuine severe weather event occurs, it must be flagged as GENUINE_WEATHER_EVENT,
        # and the immediate next step (+1m) returning to nominal diurnal atmosphere must be NORMAL (not false SPIKE).
        sim = AWSDataSimulator(seed=101)
        det = AWSAnomalyDetector()
        clean_hist = sim.generate_historical_dataset(days=2, interval_minutes=5, inject_anomalies=False)
        det.fit(clean_hist)

        base_time = pd.Timestamp("2026-06-01 12:00:00")
        for i in range(30):
            t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=i))
            det.process_observation(t, p, rh, base_time + pd.Timedelta(minutes=i))

        # Injected Genuine Weather Event at step 30
        t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=30))
        storm_rep = det.process_observation(t - 6.2, p - 2.6, 97.5, base_time + pd.Timedelta(minutes=30))
        self.assertFalse(storm_rep.is_anomaly)
        self.assertTrue(storm_rep.is_weather_event)
        self.assertEqual(storm_rep.anomaly_type, "GENUINE_WEATHER_EVENT")

        # Step +1 min (nominal observation returning from weather event)
        t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=31))
        step1_rep = det.process_observation(t, p, rh, base_time + pd.Timedelta(minutes=31))
        self.assertFalse(step1_rep.is_anomaly, msg=f"Expected NORMAL at +1m but got: {step1_rep.anomaly_type}, {step1_rep.explanation}")
        self.assertEqual(step1_rep.anomaly_type, "NORMAL")

        # Subsequent steps (+2m to +5m) must stay NORMAL
        for i in range(2, 6):
            t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=30 + i))
            rep = det.process_observation(t, p, rh, base_time + pd.Timedelta(minutes=30 + i))
            self.assertFalse(rep.is_anomaly, msg=f"Expected NORMAL at +{i}m but got: {rep.anomaly_type}, {rep.explanation}")
            self.assertEqual(rep.anomaly_type, "NORMAL")

    def test_fifteen_minute_synoptic_interval_handling(self):
        """Verify detector handles 15-minute synoptic intervals with correct spike detection."""
        sim = AWSDataSimulator(seed=101)
        det = AWSAnomalyDetector()
        clean_hist = sim.generate_historical_dataset(days=2, interval_minutes=15, inject_anomalies=False)
        det.fit(clean_hist)

        base_time = pd.Timestamp("2026-06-01 08:00:00")
        for i in range(10):
            t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=i * 15))
            rep = det.process_observation(t, p, rh, base_time + pd.Timedelta(minutes=i * 15))
            self.assertFalse(rep.is_anomaly, msg=f"Expected NORMAL at step {i} (15m interval)")

        # Inject +12.5 C spike at step 10
        t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=10 * 15))
        spike_rep = det.process_observation(t + 12.5, p, rh, base_time + pd.Timedelta(minutes=10 * 15))
        self.assertTrue(spike_rep.is_anomaly)
        self.assertEqual(spike_rep.anomaly_type, "SPIKE")
        self.assertEqual(spike_rep.faulty_sensor, "temperature")

        # Step 11: return to normal observation at +15m
        t, p, rh = sim.generate_point(base_time + pd.Timedelta(minutes=11 * 15))
        rec_rep = det.process_observation(t, p, rh, base_time + pd.Timedelta(minutes=11 * 15))
        self.assertFalse(rec_rep.is_anomaly, msg=f"Expected recovery to NORMAL at +15m but got {rec_rep.anomaly_type}")


if __name__ == '__main__':
    unittest.main()
