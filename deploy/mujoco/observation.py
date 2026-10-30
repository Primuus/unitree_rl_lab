"""The exact 56-value actor layout; joint state order differs from action order."""
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
    clock = np.array([np.sin(phase), np.cos(phase)]) * moving
    action_names = config["action_names"]
    order = [action_names.index(name) for name in config["observation_joint_names"]]
    offset = np.asarray(config["action_offset"])
    observation = np.concatenate((
        [c*vx+s*vy, -s*vx+c*vy, vz],
        angular_body*.25, rotation.T @ np.array([0., 0., -1.]),
        command, clock, (joint_pos-offset)[order], joint_vel[order]*.05,
        last_action, [s, c], np.clip((rear_height-rear_height.min())*10, 0, 1),
        np.clip(np.maximum(rear_force_z, 0)/(config["robot_mass"]*9.81), 0, 1.5),
    )).astype(np.float32)
    if observation.shape != (56,) or not np.isfinite(observation).all():
        raise ValueError("Invalid 56-dimensional actor observation")
    return observation
