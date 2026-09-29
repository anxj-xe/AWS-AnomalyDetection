"""
Diagnose real sensor resolution / typical flatline run lengths in a CSV,
so quality_control.py's persistence thresholds can be tuned to match
your actual instrument, not guessed.

Usage: python diagnose_sensor_resolution.py incompass_kanpur_July_August_2018_1min.csv
"""
import sys
import pandas as pd
import numpy as np

def analyze(path):
    df = pd.read_csv(path)
    for col in ["temperature", "pressure", "humidity"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values

        # Effective decimal resolution: smallest nonzero gap between consecutive values
        diffs = np.abs(np.diff(vals))
        nonzero_diffs = diffs[diffs > 1e-9]
        min_step = np.min(nonzero_diffs) if len(nonzero_diffs) else float("nan")

        # Consecutive identical (bit-identical) run lengths
        runs = []
        run_len = 1
        for i in range(1, len(vals)):
            if abs(vals[i] - vals[i-1]) < 1e-4:
                run_len += 1
            else:
                runs.append(run_len)
                run_len = 1
        runs.append(run_len)
        runs = np.array(runs)

        print(f"\n=== {col} ===")
        print(f"  Smallest real step between consecutive readings: {min_step:.5f}")
        print(f"  Rolling 5-sample stdev (median across dataset):  {pd.Series(vals).rolling(5).std().median():.5f}")
        print(f"  Flatline run-length distribution (consecutive identical readings):")
        print(f"    max run length   : {runs.max()}")
        print(f"    95th percentile  : {np.percentile(runs, 95):.1f}")
        print(f"    % of runs >= 4   : {(runs >= 4).mean()*100:.1f}%")
        print(f"    % of runs >= 8   : {(runs >= 8).mean()*100:.1f}%")
        print(f"    % of runs >= 15  : {(runs >= 15).mean()*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_sensor_resolution.py <csv_path>")
        sys.exit(1)
    analyze(sys.argv[1])