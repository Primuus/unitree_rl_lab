"""Read the saved gait-clock observation contract for checkpoint/policy replay."""
import json
from pathlib import Path


def read_gait_clock(provenance_path):
    provenance = json.loads(Path(provenance_path).read_text())
    # Before the phase-observation fix, all tasks used the command-gated clock.
    clock = provenance["layout"].get("gait_clock", {"mode": "moving_only", "period_s": 1.6})
    if clock["mode"] not in ("always", "moving_only") or clock["period_s"] <= 0:
        raise ValueError(f"Invalid saved gait clock: {clock}")
    return clock
