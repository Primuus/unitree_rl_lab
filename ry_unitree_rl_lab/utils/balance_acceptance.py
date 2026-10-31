"""Project standing acceptance shared by Isaac evaluation and MuJoCo replay."""
import math


VERSION = "stationary-v2"
THRESHOLDS = {
    "min_valid_seconds": 15.0,
    "min_upright_fraction_exclusive": 0.95,
    "max_front_contact_fraction_exclusive": 0.01,
    "max_forward_error_mps": 0.04,
    "max_mean_abs_yaw_rate_radps": 0.05,
    "max_heading_excursion_rad": math.radians(15),
    "max_horizontal_displacement_m": 0.20,
    "max_path_speed_mps": 0.04,
}


def assess_balance(values, eligible=True):
    """Assess one uninterrupted episode; caller supplies finite scalar statistics."""
    t = THRESHOLDS
    checks = {
        "trained_balance_policy": bool(eligible),
        "valid_duration": values["valid_seconds"] >= t["min_valid_seconds"],
        "no_failure": values["failures"] == 0,
        "upright": values["upright_fraction"] > t["min_upright_fraction_exclusive"],
        "front_contact": values["front_contact_fraction"] < t["max_front_contact_fraction_exclusive"],
        "forward_error": values["forward_error_mps"] <= t["max_forward_error_mps"],
        "yaw_rate": values["mean_abs_yaw_rate_radps"] <= t["max_mean_abs_yaw_rate_radps"],
        "heading_excursion": values["max_heading_excursion_rad"] <= t["max_heading_excursion_rad"],
        "horizontal_displacement": values["max_horizontal_displacement_m"] <= t["max_horizontal_displacement_m"],
        "horizontal_path": values["path_speed_mps"] <= t["max_path_speed_mps"],
    }
    return {"version": VERSION, "thresholds": dict(t), "values": values,
            "checks": checks, "failed_checks": [k for k, passed in checks.items() if not passed],
            "passed": all(checks.values())}
