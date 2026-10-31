"""Named actor layouts for the current and original policies; joint state order differs from action order."""
import numpy as np


def heading(rotation):
    forward = -rotation[:, 2]
    return np.arctan2(forward[1], forward[0])


def build_observation(rotation, linear_world, angular_body, joint_pos, joint_vel,
                      last_action, rear_height, rear_force_z, time_s, config):
    yaw = heading(rotation)
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy, vz = linear_world
    command = np.asarray(config["command"])
    moving = np.linalg.norm(command[:2]) > .04 or abs(command[2]) > .08
    phase = 2*np.pi*time_s/config["gait_period"]
    clock = np.array([np.sin(phase), np.cos(phase)]) * (config["gait_clock_mode"] == "always" or moving)
    action_names = config["action_names"]
    order = [action_names.index(name) for name in config["observation_joint_names"]]
    offset = np.asarray(config["action_offset"])
    terms = {
        "linear_velocity": [c*vx+s*vy, -s*vx+c*vy, vz],
        "angular_velocity": angular_body*.25,
        "projected_gravity": rotation.T @ np.array([0., 0., -1.]),
        "command": command, "gait_phase": clock,
        "joint_pos": (joint_pos-offset)[order], "joint_vel": joint_vel[order]*.05,
        "last_action": last_action, "heading": [s, c],
        "foot_height": np.clip((rear_height-rear_height.min())*10, 0, 1),
        "foot_load": np.clip(np.maximum(rear_force_z, 0)/(config["robot_mass"]*9.81), 0, 1.5),
    }
    names = config.get("observation_terms", list(terms))
    observation = np.concatenate([np.asarray(terms[name]) for name in names]).astype(np.float32)
    expected = config.get("observation_dim", 56)
    if observation.shape != (expected,) or not np.isfinite(observation).all():
        raise ValueError(f"Invalid {expected}-dimensional actor observation")
    return observation
