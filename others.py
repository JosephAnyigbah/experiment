"""Compatibility entry point redirected to the safe Route B workflow."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().with_name('others2.py')),run_name='__main__')
