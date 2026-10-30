"""Evaluate the current 12-joint task through the existing entry point."""
import runpy
import sys
from pathlib import Path

tools = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(tools))
runpy.run_path(str(tools / "eval_go2_biped_walk.py"), run_name="__main__")
