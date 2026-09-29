"""
Entry point for SkyGuard AWS Anomaly Detection Dashboard.
Ensures both `streamlit run app.py` and `streamlit run index.py` work seamlessly.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "index.py"
    runpy.run_path(str(target), run_name="__main__")
