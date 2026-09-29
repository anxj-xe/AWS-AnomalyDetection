"""
Main Entry Point & Command Line Interface for SIH 2026 AWS Anomaly Detection System.

Usage:
  python main.py --demo                      # Run interactive terminal simulation stream
  python main.py --generate-data             # Generate synthetic AWS benchmark dataset
  python main.py --evaluate --file data.csv  # Process and evaluate anomalies on a CSV file
"""

import argparse
import sys
import time
import pandas as pd
import numpy as np

# Ensure Windows terminal handles UTF-8 safely
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.data_simulator import AWSDataSimulator
from src.detector import AWSAnomalyDetector
from src.imputer import AWSImputer
from src.health_monitor import SensorHealthMonitor


def run_demo_simulation(steps: int = 40, delay_sec: float = 0.2):
    """Run interactive streaming simulation in terminal."""
    print("=" * 75)
    print("  SIH 2026: AWS INTELLIGENT ANOMALY DETECTION SYSTEM (SIMULATION)")
    print("  Parameters: Temperature (°C) | Pressure (hPa) | Relative Humidity (%)")
    print("=" * 75)

    simulator = AWSDataSimulator(seed=101)
    detector = AWSAnomalyDetector()
    imputer = AWSImputer()
    health_mon = SensorHealthMonitor()

    print("\n[INFO] Pre-training ML baseline on clean historical diurnal data...")
    clean_history = simulator.generate_historical_dataset(days=3, interval_minutes=5, inject_anomalies=False)
    detector.fit(clean_history)
    print("[INFO] Model trained successfully. Starting live telemetry stream...\n")

    print(f"{'TIMESTAMP':<19} | {'T (°C)':<6} | {'P (hPa)':<7} | {'RH (%)':<6} | {'STATUS / DIAGNOSIS':<30}")
    print("-" * 75)

    stream = simulator.stream_generator()
    for idx in range(steps):
        obs = next(stream)
        t, p, rh, ts = obs.temperature, obs.pressure, obs.humidity, obs.timestamp

        # Injected test events for demonstration
        if idx == 10:
            t += 10.5  # Temperature spike fault
        elif idx >= 18 and idx <= 23:
            rh = 78.3  # Frozen humidity sensor
        elif idx == 30:
            # Genuine Severe Thunderstorm downdraft
            t -= 5.8
            rh = 96.0
            p -= 2.4

        # Process through 5-Tier Hybrid AI Pipeline
        report = detector.process_observation(t, p, rh, timestamp=ts)

        # Update Imputer and Health Monitor
        if report.is_anomaly:
            imp_res = imputer.impute_point(t, p, rh, report.faulty_sensor, idx)
            health_mon.record_observation(t, p, rh, report.faulty_sensor)
        else:
            imputer.update_clean_history(t, p, rh, idx)
            health_mon.record_observation(t, p, rh, None)

        # Display formatted log
        status_tag = report.anomaly_type
        if report.is_anomaly:
            color = "\033[91m"  # Red
            tag = f"[ANOMALY: {status_tag} ({report.faulty_sensor})]"
        elif report.is_weather_event:
            color = "\033[93m"  # Yellow/Cyan
            tag = f"[STORM: {status_tag}]"
        else:
            color = "\033[92m"  # Green
            tag = "[OK: NORMAL]"
        reset = "\033[0m"

        print(f"{ts.strftime('%Y-%m-%d %H:%M')} | {t:6.2f} | {p:7.2f} | {rh:6.1f} | {color}{tag:<30}{reset}")
        if report.is_anomaly:
            print(f"   -> XAI Rationale: {report.explanation}")
            print(f"   -> Actionable Guidance: {report.top_features}")
        elif report.is_weather_event:
            print(f"   -> Weather Notice: {report.explanation}")

        time.sleep(delay_sec)

    print("-" * 75)
    st_rep = health_mon.generate_station_report()
    print(f"\n[STATION HEALTH SUMMARY]: Overall Health: {st_rep.overall_health}% ({st_rep.station_status})")
    for s_name, s_stat in st_rep.sensors.items():
        print(f" - {s_name.capitalize():<12}: {s_stat.health_score}% [{s_stat.status}] -> {s_stat.recommendation}")
    print("=" * 75)


def generate_benchmark_csv(output_path: str = "data/aws_benchmark_dataset.csv"):
    """Generate ground-truth synthetic benchmark dataset."""
    print(f"[INFO] Generating 5-day AWS benchmark dataset with injected anomalies...")
    sim = AWSDataSimulator(seed=42)
    df = sim.generate_historical_dataset(days=5, interval_minutes=1, inject_anomalies=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Dataset saved to {output_path} ({len(df)} rows).")


def evaluate_csv_file(file_path: str):
    """Process and evaluate anomalies on any input CSV file."""
    print(f"[INFO] Loading {file_path}...")
    df = pd.read_csv(file_path)
    required = ['temperature', 'pressure', 'humidity']
    for col in required:
        if col not in df.columns:
            print(f"[ERROR] Missing required column: {col}")
            sys.exit(1)

    detector = AWSAnomalyDetector()
    print("[INFO] Training baseline on input data...")
    detector.fit(df.head(min(1000, len(df))))

    print("[INFO] Evaluating observations...")
    reports = detector.process_dataframe(df)

    anomalies = [r for r in reports if r.is_anomaly]
    storms = [r for r in reports if r.is_weather_event]

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Observations Processed: {len(reports)}")
    print(f"Sensor Anomalies Detected    : {len(anomalies)} ({len(anomalies)/len(reports)*100:.2f}%)")
    print(f"Severe Weather Events Flagged: {len(storms)} (Disentangled, not alarmed as faults)")

    breakdown = {}
    for a in anomalies:
        breakdown[a.anomaly_type] = breakdown.get(a.anomaly_type, 0) + 1
    print("\nRoot Cause Breakdown:")
    for k, v in breakdown.items():
        print(f" - {k:<25}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIH 2026 AWS Anomaly Detection System")
    parser.add_argument("--demo", action="store_true", help="Run real-time terminal streaming simulation")
    parser.add_argument("--generate-data", action="store_true", help="Generate benchmark dataset CSV")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate anomalies on a CSV file")
    parser.add_argument("--file", type=str, default="data/aws_benchmark_dataset.csv", help="CSV file path")

    args = parser.parse_args()

    if args.generate_data:
        generate_benchmark_csv(args.file)
    elif args.evaluate:
        evaluate_csv_file(args.file)
    else:
        run_demo_simulation()
