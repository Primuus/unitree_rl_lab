"""Evaluate the RY v20 task with the existing framework evaluator."""
import runpy
import sys
from pathlib import Path
import ry_go2_biped_walk_v20

shared_tools = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(shared_tools))
sys.argv[1:1] = ["--task", "RY-Unitree-Go2-Biped-Gait-v20"]
runpy.run_path(str(shared_tools / "eval_go2_biped_walk.py"), run_name="__main__")
