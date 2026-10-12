"""Excel-only safe entry point: audit the selected workbook and stop before modeling."""
from pathlib import Path
import subprocess
import sys
HERE = Path(__file__).resolve().parent
WORKBOOK = HERE / "dataset_2.xlsx"
raise SystemExit(subprocess.run([sys.executable, str(HERE / "data_audit.py"), str(WORKBOOK)], check=False).returncode)
