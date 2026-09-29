"""
SkyGuard — Adaptive 15-min / 1-min replay with interactive Train/Test windows.

ONE 1-minute CSV is used for everything.

Workflow:
1. Load the single source CSV.
2. Ask the user for TRAINING start/end dates.
3. Ask the user for TESTING start/end dates.
4. Reject overlapping train/test windows.
5. Also require TESTING to start after TRAINING ends for a clean
   chronological time-series experiment.
6. Train only on the training window.
7. Test only on the testing window.
8. During testing:
      Normal mode = 15-minute samples
      Weather-event mode = every 1-minute sample
9. Save adaptive results and switch log.

IMPORTANT:
- The raw CSV is never modified.
- This is an adaptive replay/evaluation workflow.
- It does NOT calculate precision/recall/F1 unless ground-truth labels exist.
"""

import time
import sys
import importlib.util
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent


def _load_quality_control_in_memory():
    """Load src/quality_control.py in memory without changing the core file.

    The local core may have a slightly different check_persistence signature.
    It may also contain an accidental malformed QCResult keyword from an
    earlier edit. We repair only that keyword in the AST used for this run.
    The file on disk is never written.
    """
    import ast

    path = PROJECT_DIR / "src" / "quality_control.py"
    if not path.exists():
        raise FileNotFoundError(f"Core quality-control file not found: {path}")

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    repaired_keyword = False

    class RepairQCResultKeyword(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal repaired_keyword
            self.generic_visit(node)

            if isinstance(node.func, ast.Name) and node.func.id == "QCResult":
                for kw in node.keywords:
                    if (
                        kw.arg is not None
                        and kw.arg.startswith("is_anom")
                        and kw.arg != "is_anomaly"
                    ):
                        kw.arg = "is_anomaly"
                        repaired_keyword = True
            return node

    tree = RepairQCResultKeyword().visit(tree)
    ast.fix_missing_locations(tree)

    sys.modules.pop("src.quality_control", None)
    spec = importlib.util.spec_from_file_location("src.quality_control", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "src"
    sys.modules["src.quality_control"] = module

    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop("src.quality_control", None)
        raise

    if repaired_keyword:
        print("[IMPORT] QCResult keyword repaired IN MEMORY for this run only.")
        print("[IMPORT] src/quality_control.py on disk was NOT modified.")


def load_detector_class():
    """Load src/detector.py in memory without editing the core files.

    If detector.py contains a self-import such as
    ``from src.detector import AWSAnomalyDetector``, that import is removed
    from the IN-MEMORY AST only.
    """
    import ast

    detector_path = PROJECT_DIR / "src" / "detector.py"
    if not detector_path.exists():
        raise FileNotFoundError(f"Core detector not found: {detector_path}")

    # Load QC first so detector.py receives this run-local module instead of
    # importing the on-disk module directly.
    _load_quality_control_in_memory()

    source = detector_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(detector_path))
    removed_self_import = False
    repaired_lookback_index = False

    class RepairDetectorRuntime(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            nonlocal removed_self_import
            if node.module == "src.detector":
                removed_self_import = True
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            nonlocal repaired_lookback_index
            self.generic_visit(node)

            # The core detector assumes at least one 30-minute history index.
            # With sparse/gapped observations, int(30/delta) can become 0,
            # producing history[len(history)] and an IndexError. Clamp the
            # evaluator's in-memory copy to the last available history value.
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "lookback_idx":
                    if isinstance(node.value, ast.Call) or isinstance(node.value, ast.BinOp):
                        repaired_lookback_index = True
                        node.value = ast.parse(
                            "min(len(self.history_temp) - 1, max(0, len(self.history_temp) - int(30 / max(1, delta_mins))))"
                        ).body[0].value
            return node

    tree = RepairDetectorRuntime().visit(tree)
    ast.fix_missing_locations(tree)

    sys.modules.pop("src.detector", None)
    spec = importlib.util.spec_from_file_location("src.detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {detector_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "src"
    sys.modules["src.detector"] = module

    try:
        exec(compile(tree, str(detector_path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop("src.detector", None)
        raise

    detector_class = getattr(module, "AWSAnomalyDetector", None)
    if detector_class is None or not isinstance(detector_class, type):
        sys.modules.pop("src.detector", None)
        raise ImportError(
            "AWSAnomalyDetector class was not found after loading src/detector.py."
        )

    print("[IMPORT] Core detector loaded successfully IN MEMORY.")
    if removed_self_import:
        print("[IMPORT] Detector self-import removed IN MEMORY for this run only.")
    if repaired_lookback_index:
        print("[IMPORT] Detector lookback index guarded IN MEMORY for sparse sampling only.")
    print("[IMPORT] Core detector.py and quality_control.py on disk were NOT modified.")

    return detector_class


AWSAnomalyDetector = load_detector_class()


DEFAULT_FILE = PROJECT_DIR / "incompass_kanpur_1min.csv"

DEFAULT_OUTPUT = PROJECT_DIR / "kanpur_adaptive_train_test_results.csv"
DEFAULT_SWITCH_LOG = PROJECT_DIR / "kanpur_adaptive_train_test_switches.csv"

DATE_FORMAT = "YYYY-MM-DD HH:MM:SS"


def load_data(path):
    df = pd.read_csv(path)

    required = ["timestamp", "temperature", "humidity", "pressure"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df[required].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")

    df = df.dropna(
        subset=["timestamp", "temperature", "humidity", "pressure"]
    )

    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    return df


def median_interval(df):
    if len(df) < 2:
        return 0.0

    d = df["timestamp"].diff().dt.total_seconds() / 60.0
    d = d[d > 0]

    return float(d.median()) if len(d) else 0.0




def _consecutive_equal_run_lengths(values, tolerance):
    """Return lengths of consecutive approximately-equal runs."""
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return []

    runs = []
    run_len = 1

    for i in range(1, len(values)):
        if abs(values[i] - values[i - 1]) <= tolerance:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1

    runs.append(run_len)
    return runs


def _minimum_real_change(values):
    """Smallest non-zero observed change in the training window."""
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return None

    diffs = np.abs(np.diff(values))
    # Ignore floating-point noise only; do not impose a meteorological threshold.
    diffs = diffs[diffs > np.finfo(float).eps]

    if len(diffs) == 0:
        return None

    return float(np.min(diffs))


def _median_rolling_std(values, window=5):
    """Median rolling standard deviation used only as a learned variability descriptor."""
    series = pd.Series(values, dtype=float)
    stds = series.rolling(window).std().dropna()
    if len(stds) == 0:
        return None
    return float(stds.median())


def learn_sensor_persistence_profile(train_df):
    """
    Learn sensor-specific persistence characteristics from TRAINING DATA ONLY.

    The detector's source code is not modified. This profile is used only by
    the evaluation script to override the generic persistence decision on the
    detector instance created for this run.

    Learned values:
      - minimum real change (sensor resolution observed in training)
      - median 5-sample rolling standard deviation
      - 99th percentile of normal identical/near-identical run lengths

    The 99th-percentile run rule means a persistence alarm is reserved for a
    run longer than the longest 1% of runs observed during training, rather
    than choosing an arbitrary fixed run length.
    """
    if train_df is None or len(train_df) < 10:
        return None

    profile = {}
    columns = {
        "temperature": "temperature",
        "pressure": "pressure",
        "humidity": "humidity",
    }

    for sensor, column in columns.items():
        values = pd.to_numeric(train_df[column], errors="coerce").dropna().to_numpy(dtype=float)

        if len(values) < 6:
            return None

        min_step = _minimum_real_change(values)

        # If a sensor never changes in training, we cannot infer its resolution
        # from the data. Keep the evaluator on the main QC behaviour for it.
        if min_step is None or not np.isfinite(min_step) or min_step <= 0:
            return None

        rolling_std = _median_rolling_std(values, window=5)
        if rolling_std is None or not np.isfinite(rolling_std):
            return None

        # Tolerance is derived from observed sensor resolution, not chosen in
        # physical units. It allows tiny floating representation differences
        # while still distinguishing adjacent reported sensor levels.
        equality_tolerance = min_step * 0.25
        run_lengths = _consecutive_equal_run_lengths(values, equality_tolerance)

        if not run_lengths:
            return None

        normal_run_q99 = float(np.percentile(run_lengths, 99.0))
        learned_run_limit = max(6, int(np.ceil(normal_run_q99)) + 1)

        # A flat window has std=0. The learned variability is retained as a
        # diagnostic and used to define a conservative flatness ceiling.
        flatness_std_limit = max(
            equality_tolerance * 0.5,
            rolling_std * 0.05,
        )

        profile[sensor] = {
            "minimum_real_change": float(min_step),
            "median_rolling_5_std": float(rolling_std),
            "normal_run_q99": normal_run_q99,
            "learned_run_limit": learned_run_limit,
            "equality_tolerance": float(equality_tolerance),
            "flatness_std_limit": float(flatness_std_limit),
        }

    return profile


def print_sensor_persistence_profile(profile):
    """Print the learned training-only calibration so the experiment is auditable."""
    print()
    print("=" * 76)
    print("TRAINING-ONLY SENSOR PERSISTENCE CALIBRATION")
    print("=" * 76)
    print("These values were learned from TRAINING data only.")
    print("They override persistence handling only for this evaluation run.")
    print()

    for sensor in ("temperature", "humidity", "pressure"):
        p = profile[sensor]
        print(f"{sensor.upper()}")
        print(f"  Minimum real change       : {p['minimum_real_change']:.6g}")
        print(f"  Median 5-point std        : {p['median_rolling_5_std']:.6g}")
        print(f"  99th-percentile run      : {p['normal_run_q99']:.2f} readings")
        print(f"  Learned stuck-run limit   : {p['learned_run_limit']} readings")
        print(f"  Equality tolerance        : {p['equality_tolerance']:.6g}")
        print(f"  Flatness std ceiling      : {p['flatness_std_limit']:.6g}")
        print()


def adaptive_check_persistence(
    self,
    temp_history,
    press_history,
    rh_history,
    *args,
    **kwargs,
):
    """
    Evaluation-only replacement for QualityControlEngine.check_persistence().

    IMPORTANT:
    - Does NOT edit src/quality_control.py.
    - Does NOT alter spike/rate-of-change checks.
    - Does NOT alter plausibility or thermodynamic checks.
    - Only changes the persistence/frozen-sensor decision.
    - Extra arguments from newer core QC signatures are accepted and ignored.

    A sensor is considered stuck only when its recent values remain within the
    sensor's learned resolution for at least the learned abnormal run length.
    """
    histories = {
        "temperature": temp_history,
        "pressure": press_history,
        "humidity": rh_history,
    }

    # Need the same minimum history expected by the original QC pipeline.
    window_len = self.limits.min_stdev_window_len

    for sensor, history in histories.items():
        if len(history) < window_len:
            continue

        cfg = self._adaptive_persistence_profile.get(sensor)
        values = np.asarray(history, dtype=float)

        if cfg is None:
            # Per-sensor fallback: if this sensor could not be calibrated from
            # training data, preserve the ORIGINAL core persistence threshold
            # for that sensor only.
            recent = values[-window_len:]
            recent_std = float(np.std(recent))
            original_limit = {
                "temperature": self.limits.min_temp_stdev,
                "pressure": self.limits.min_pressure_stdev,
                "humidity": self.limits.min_rh_stdev,
            }[sensor]

            if recent_std < original_limit:
                return False, sensor, (
                    f"Default persistence fallback: {sensor} std="
                    f"{recent_std:.6g} below core threshold={original_limit:.6g}"
                ), 0.85

            continue
        tol = cfg["equality_tolerance"]
        run_limit = cfg["learned_run_limit"]
        std_limit = cfg["flatness_std_limit"]

        # Count the current approximately-equal run at the end of history.
        current_run = 1
        for i in range(len(values) - 1, 0, -1):
            if abs(values[i] - values[i - 1]) <= tol:
                current_run += 1
            else:
                break

        # Do not flag merely because six readings happen to be similar.
        # The run must exceed what was observed as normal during training.
        if current_run < run_limit:
            continue

        recent = values[-window_len:]
        recent_std = float(np.std(recent))

        if recent_std <= std_limit:
            return False, sensor, (
                f"Adaptive persistence: {sensor} remained within its learned "
                f"sensor resolution for {current_run} consecutive readings "
                f"(training-derived limit={run_limit}); "
                f"window std={recent_std:.6g}"
            ), 0.85

    return True, "", "Sensors exhibit expected data-derived variation", 0.0


def install_adaptive_persistence_override(detector, train_df):
    """
    Attach the learned persistence rule to THIS detector instance only.

    The imported src.quality_control module and all other detector instances
    remain unchanged.
    """
    profile = learn_sensor_persistence_profile(train_df)

    if profile is None:
        print()
        print("[ADAPTIVE QC] Calibration unavailable from training window.")
        print("[ADAPTIVE QC] Keeping the main model's default persistence rule.")
        return None

    detector.qc_engine._adaptive_persistence_profile = profile
    detector.qc_engine._adaptive_original_check_persistence = (
        detector.qc_engine.check_persistence
    )
    detector.qc_engine.check_persistence = MethodType(
        adaptive_check_persistence,
        detector.qc_engine,
    )

    print_sensor_persistence_profile(profile)
    print("[ADAPTIVE QC] Persistence override installed for this detector instance.")
    print("[ADAPTIVE QC] Spike/rate-of-change logic remains unchanged.")

    return profile

def first_quarter_hour_at_or_after(ts):
    """Return the next clock-aligned 15-minute timestamp."""
    ts = pd.Timestamp(ts).floor("min")

    minute = ts.minute
    remainder = minute % 15

    if remainder == 0:
        return ts

    return ts + pd.Timedelta(minutes=15 - remainder)


def ask_datetime(label):
    """Keep asking until the user enters a valid datetime."""
    while True:
        print()
        print(f"{label}")
        print(f"Enter in this format: {DATE_FORMAT}")
        value = input("> ").strip()

        try:
            return pd.Timestamp(value)
        except Exception:
            print()
            print("❌ Invalid date/time format.")
            print("Use exactly like:")
            print("   2018-07-01 00:00:00")


def ask_training_and_testing_windows(data, data_min, data_max):
    """
    Interactive window selection with leakage protection.

    Training must finish before testing starts.
    Any overlap is rejected.
    """

    print()
    print("=" * 76)
    print("SELECT TRAINING WINDOW")
    print("=" * 76)
    print(f"Available data: {data_min}  ->  {data_max}")

    while True:
        train_start = ask_datetime("Training START date/time")
        train_end = ask_datetime("Training END date/time")

        if train_start > train_end:
            print()
            print("❌ Training start cannot be after training end.")
            print("Please re-enter the training window.")
            continue

        if train_start < data_min or train_end > data_max:
            print()
            print("❌ Training window is outside the available dataset.")
            print(f"Available range: {data_min} -> {data_max}")
            continue

        train_rows = (
            (data["timestamp"] >= train_start)
            & (data["timestamp"] <= train_end)
        )

        if train_rows.sum() == 0:
            print()
            print("❌ No data exists inside this training window.")
            continue

        break

    print()
    print("=" * 76)
    print("SELECT TESTING WINDOW")
    print("=" * 76)

    while True:
        test_start = ask_datetime("Testing START date/time")
        test_end = ask_datetime("Testing END date/time")

        if test_start > test_end:
            print()
            print("❌ Testing start cannot be after testing end.")
            continue

        if test_start < data_min or test_end > data_max:
            print()
            print("❌ Testing window is outside the available dataset.")
            print(f"Available range: {data_min} -> {data_max}")
            continue

        # ------------------------------------------------------------
        # FUNDAMENTAL LEAKAGE CHECK
        # ------------------------------------------------------------
        overlap = not (
            train_end < test_start
            or test_end < train_start
        )

        if overlap:
            print()
            print("❌ DATA LEAKAGE / WINDOW COLLISION DETECTED")
            print("-" * 76)
            print("Training and testing periods overlap.")
            print()
            print(f"Training : {train_start} -> {train_end}")
            print(f"Testing  : {test_start} -> {test_end}")
            print()
            print(
                "A model must not be tested on observations that were "
                "included in its training period."
            )
            print("Please re-enter the testing period.")
            continue

        # ------------------------------------------------------------
        # CHRONOLOGICAL TIME-SERIES CHECK
        # ------------------------------------------------------------
        if test_start <= train_end:
            print()
            print("❌ INVALID TIME-SERIES ORDER")
            print("-" * 76)
            print("Testing must start AFTER the training period ends.")
            print()
            print(f"Training ends : {train_end}")
            print(f"Testing starts: {test_start}")
            print()
            print(
                "This prevents training on later observations and then "
                "testing on earlier observations."
            )
            print("Please re-enter the testing period.")
            continue

        test_rows = (
            (data["timestamp"] >= test_start)
            & (data["timestamp"] <= test_end)
        )

        if test_rows.sum() == 0:
            print()
            print("❌ No data exists inside this testing window.")
            continue

        break

    return train_start, train_end, test_start, test_end


def run_adaptive_test(
    train_df,
    test_df,
    contamination=0.03,
    storm_clear_minutes=60,
):
    """
    Train on train_df only, then replay test_df adaptively.

    The detector's generic persistence rule is overridden only on this
    detector instance when a valid training-derived profile is available.
    """

    detector = AWSAnomalyDetector(contamination=contamination)

    # ------------------------------------------------------------
    # TRAINING-ONLY ADAPTIVE PERSISTENCE CALIBRATION
    # ------------------------------------------------------------
    # The ML baseline still trains on the clock-aligned 15-minute samples.
    # Persistence calibration uses ALL observations inside the training
    # window so the sensor resolution is learned from the actual 1-minute
    # source data. Nothing from the test window is used here.
    adaptive_profile = install_adaptive_persistence_override(
        detector, train_df
    )

    # Training MUST use only the selected training period.
    train_15 = train_df[
        train_df["timestamp"].dt.minute.isin([0, 15, 30, 45])
    ].copy()

    if len(train_15) < 100:
        raise ValueError(
            f"Only {len(train_15)} clock-aligned 15-minute rows are "
            "available in the training window. Need at least 100."
        )

    print()
    print("[1/3] TRAINING")
    print("-" * 76)
    print("Training source : SAME 1-minute CSV")
    print("Training mode   : 15-minute samples")
    print(f"Training rows   : {len(train_15):,}")
    print(
        f"Training period : {train_df['timestamp'].min()} "
        f"-> {train_df['timestamp'].max()}"
    )

    t0 = time.perf_counter()
    detector.fit(train_15)
    training_time = time.perf_counter() - t0

    print(f"Training time   : {training_time:.2f} seconds")

    # IMPORTANT: reset streaming history before TESTING.
    detector.reset_stream()

    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    timestamps = test_df["timestamp"].tolist()

    mode = "15MIN"
    pos = 0

    next_15_time = first_quarter_hour_at_or_after(timestamps[0])
    last_weather_event = None

    results = []
    switches = []

    while pos < len(test_df):

        current_time = timestamps[pos]

        # ============================================================
        # NORMAL 15-MINUTE MODE
        # ============================================================
        if mode == "15MIN":

            if current_time < next_15_time:
                pos += 1
                continue

            row = test_df.iloc[pos]

            report = detector.process_observation(
                temperature=float(row["temperature"]),
                pressure=float(row["pressure"]),
                humidity=float(row["humidity"]),
                timestamp=current_time,
            )

            results.append(
                {
                    "timestamp": report.timestamp,
                    "mode": "15MIN",
                    "temperature": report.temperature,
                    "pressure": report.pressure,
                    "humidity": report.humidity,
                    "is_anomaly": bool(report.is_anomaly),
                    "is_weather_event": bool(report.is_weather_event),
                    "anomaly_type": report.anomaly_type,
                    "confidence": float(report.confidence),
                    "severity": float(report.severity),
                    "faulty_sensor": report.faulty_sensor,
                    "explanation": report.explanation,
                }
            )

            next_15_time = current_time + pd.Timedelta(minutes=15)

            # Weather event -> rapid 1-minute mode.
            if report.is_weather_event:
                mode = "1MIN"
                last_weather_event = current_time

                switches.append(
                    {
                        "timestamp": current_time,
                        "event": "15MIN_TO_1MIN",
                        "reason": report.explanation,
                        "confidence": report.confidence,
                    }
                )

                print(
                    f">>> SWITCH 15MIN -> 1MIN : {current_time}"
                )

                pos += 1
                continue

            pos += 1
            continue

        # ============================================================
        # 1-MINUTE STORM / RAPID MODE
        # ============================================================
        row = test_df.iloc[pos]

        report = detector.process_observation(
            temperature=float(row["temperature"]),
            pressure=float(row["pressure"]),
            humidity=float(row["humidity"]),
            timestamp=current_time,
        )

        results.append(
            {
                "timestamp": report.timestamp,
                "mode": "1MIN",
                "temperature": report.temperature,
                "pressure": report.pressure,
                "humidity": report.humidity,
                "is_anomaly": bool(report.is_anomaly),
                "is_weather_event": bool(report.is_weather_event),
                "anomaly_type": report.anomaly_type,
                "confidence": float(report.confidence),
                "severity": float(report.severity),
                "faulty_sensor": report.faulty_sensor,
                "explanation": report.explanation,
            }
        )

        if report.is_weather_event:
            last_weather_event = current_time

        if last_weather_event is not None:
            quiet_minutes = (
                current_time - last_weather_event
            ).total_seconds() / 60.0

            if quiet_minutes >= storm_clear_minutes:

                switches.append(
                    {
                        "timestamp": current_time,
                        "event": "1MIN_TO_15MIN",
                        "reason": (
                            f"No weather-event report for "
                            f"{storm_clear_minutes} minutes"
                        ),
                        "confidence": "",
                    }
                )

                print(
                    f"<<< SWITCH 1MIN -> 15MIN : {current_time}"
                )

                mode = "15MIN"
                last_weather_event = None

                next_15_time = first_quarter_hour_at_or_after(
                    current_time + pd.Timedelta(minutes=1)
                )

        pos += 1

    return pd.DataFrame(results), pd.DataFrame(switches)


def print_summary(
    train_start,
    train_end,
    test_start,
    test_end,
    train_15,
    test_df,
    results,
    switches,
):
    total_processed = len(results)

    anomalies = (
        int(results["is_anomaly"].sum())
        if total_processed
        else 0
    )

    weather_events = (
        int(results["is_weather_event"].sum())
        if total_processed
        else 0
    )

    modes = (
        results["mode"].value_counts().to_dict()
        if total_processed
        else {}
    )

    print()
    print("=" * 76)
    print("FINAL TRAIN / TEST SUMMARY")
    print("=" * 76)

    print()
    print("TRAINING WINDOW")
    print(f"  {train_start} -> {train_end}")
    print(f"  15-minute training rows : {len(train_15):,}")

    print()
    print("TESTING WINDOW")
    print(f"  {test_start} -> {test_end}")
    print(f"  Source rows available   : {len(test_df):,}")
    print(f"  Rows actually processed : {total_processed:,}")

    if len(test_df):
        print(
            f"  Processing load        : "
            f"{100 * total_processed / len(test_df):.2f}%"
        )

    print()
    print("ADAPTIVE SAMPLING")
    print(f"  15-minute processed     : {modes.get('15MIN', 0):,}")
    print(f"  1-minute processed      : {modes.get('1MIN', 0):,}")
    print(f"  Weather-event reports   : {weather_events:,}")
    print(f"  Adaptive switches       : {len(switches):,}")

    print()
    print("ANOMALY REPORT")
    print(f"  Sensor/data anomalies   : {anomalies:,}")

    if total_processed and anomalies:
        rows = results[results["is_anomaly"]]

        print()
        print("  Anomaly types:")

        for name, count in rows["anomaly_type"].value_counts().items():
            print(f"    {name:<23} {count:,}")

        print()
        print("  Faulty sensors:")

        for name, count in rows["faulty_sensor"].value_counts(
            dropna=True
        ).items():
            print(f"    {name:<23} {count:,}")

    print()
    print("LEAKAGE CHECK")
    print("  PASS: training and testing windows are non-overlapping.")
    print("  PASS: testing starts after training ends.")
    print("  PASS: detector was reset before testing.")
    print()
    print("NOTE")
    print(
        "These anomaly counts are detector outputs, not accuracy metrics. "
        "Ground-truth sensor-fault labels are required for precision, "
        "recall, F1, etc."
    )


def main():

    print("=" * 76)
    print("SKYGUARD — CLEAN TRAIN / TEST ADAPTIVE EVALUATION")
    print("ONE 1-MINUTE CSV -> 15-MIN NORMAL / 1-MIN RAPID")
    print("=" * 76)

    file_path = DEFAULT_FILE

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Put incompass_kanpur_1min.csv in the project folder."
        )

    print()
    print("[INFO] Loading:", file_path.name)

    df = load_data(file_path)

    data_min = df["timestamp"].min()
    data_max = df["timestamp"].max()

    print()
    print(f"[INFO] Available data : {data_min} -> {data_max}")
    print(f"[INFO] Total rows      : {len(df):,}")
    print(f"[INFO] Median gap      : {median_interval(df):.2f} min")

    # ------------------------------------------------------------
    # INTERACTIVE WINDOW SELECTION
    # ------------------------------------------------------------
    (
        train_start,
        train_end,
        test_start,
        test_end,
    ) = ask_training_and_testing_windows(df, data_min, data_max)

    train_df = df[
        (df["timestamp"] >= train_start)
        & (df["timestamp"] <= train_end)
    ].copy()

    test_df = df[
        (df["timestamp"] >= test_start)
        & (df["timestamp"] <= test_end)
    ].copy()

    train_15 = train_df[
        train_df["timestamp"].dt.minute.isin([0, 15, 30, 45])
    ]

    print()
    print("=" * 76)
    print("WINDOWS ACCEPTED")
    print("=" * 76)
    print(f"TRAIN : {train_start} -> {train_end}")
    print(f"TEST  : {test_start} -> {test_end}")
    print()
    print("No overlap detected.")
    print("Chronological order is valid.")
    print(f"Training 15-min rows available: {len(train_15):,}")
    print(f"Testing source rows available : {len(test_df):,}")

    if len(train_15) < 100:
        raise ValueError(
            f"Training window has only {len(train_15)} usable 15-minute "
            "rows. Select a longer training period."
        )

    if len(test_df) == 0:
        raise ValueError("Testing window contains no usable data.")

    # ------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------
    results, switches = run_adaptive_test(
        train_df=train_df,
        test_df=test_df,
        contamination=0.03,
        storm_clear_minutes=60,
    )

    results.to_csv(DEFAULT_OUTPUT, index=False)
    switches.to_csv(DEFAULT_SWITCH_LOG, index=False)

    print()
    print("[3/3] Results saved.")
    print_summary(
        train_start,
        train_end,
        test_start,
        test_end,
        train_15,
        test_df,
        results,
        switches,
    )

    print()
    print("Results file :", DEFAULT_OUTPUT.name)
    print("Switch file  :", DEFAULT_SWITCH_LOG.name)


if __name__ == "__main__":
    main()
