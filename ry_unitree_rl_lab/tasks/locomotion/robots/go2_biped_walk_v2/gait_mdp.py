"""Dense load transfer and baseline-subtracted lift; no imposed joint motion."""
import math
import json
import torch
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk import walk_mdp as wm


def gait_targets(env, period=1.0, stance_ratio=0.6):
    phase = wm.phases(env, period)
    progress = ((phase - stance_ratio) / (1.0 - stance_ratio)).clamp(0.0, 1.0)
    swing = phase >= stance_ratio
    arc = torch.sin(math.pi * progress).clamp(min=0.0) * swing
    support = 1.0 - torch.sqrt(arc)
    target = support / support.sum(-1, keepdim=True).clamp(min=1e-6)
    return swing, arc, target


def foot_loads(env, sensor_cfg, robot_weight):
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (forces.clamp(min=0.0) / robot_weight).clamp(max=2.0)


def phase_load_error(env, sensor_cfg, asset_cfg, terrain_sensor_cfg, robot_weight,
                     period=1.0, command_name="base_velocity"):
    _, _, target = gait_targets(env, period)
    actual = foot_loads(env, sensor_cfg, robot_weight)
    error = (actual - target).square().sum(-1)
    step = getattr(env, "common_step_counter", -1)
    if step >= 0 and step % 2400 == 0 and getattr(env, "_gait_v2_logged", -1) != step:
        env._gait_v2_logged = step
        contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0
        z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
        velocity = wm.heading_lin_vel(env)
        print("GAIT_V2_METRICS " + json.dumps({
            "control_step": step,
            "single_support_fraction": float((contact.sum(-1) == 1).float().mean()),
            "flight_fraction": float((contact.sum(-1) == 0).float().mean()),
            "mean_relative_foot_height_m": float((z.max(-1).values-z.min(-1).values).mean()),
            "mean_forward_speed_mps": float(velocity[:, 0].mean()),
            "mean_command_mps": float(env.command_manager.get_command(command_name)[:, 0].mean()),
        }), flush=True)
    return error * wm.moving(env, command_name) * wm.posture_gate(env, terrain_sensor_cfg)


def dense_swing_clearance(env, asset_cfg, sensor_cfg, terrain_sensor_cfg,
                          robot_weight, period=1.0, clearance=0.05, std=0.025,
                          command_name="base_velocity"):
    swing, arc, _ = gait_targets(env, period)
    target = clearance * arc
    z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = z - z.min(-1, keepdim=True).values
    score = torch.exp(-((height - target) / std).square())
    baseline = torch.exp(-(target / std).square())
    # No contact Boolean on the swing foot: reward becomes available while
    # lifting begins. Continuous opposite-foot load still rejects flight.
    support = (foot_loads(env, sensor_cfg, robot_weight).flip([-1]) / 0.5).clamp(0.0, 1.0)
    value = ((score - baseline) * swing * support).sum(-1)
    return value * wm.moving(env, command_name) * wm.posture_gate(env, terrain_sensor_cfg)


def tight_velocity_tracking(env, terrain_sensor_cfg, command_name="base_velocity"):
    command = env.command_manager.get_command(command_name)[:, :2]
    velocity = wm.heading_lin_vel(env)[:, :2]
    std = (0.5 * command.norm(dim=-1)).clamp(min=0.06, max=0.12)
    error = (command - velocity).square().sum(-1)
    return torch.exp(-error / std.square()) * wm.posture_gate(env, terrain_sensor_cfg)


def forward_progress(env, terrain_sensor_cfg, command_name="base_velocity"):
    command = env.command_manager.get_command(command_name)[:, :2]
    magnitude = command.norm(dim=-1)
    direction = command / magnitude[:, None].clamp(min=0.04)
    velocity = wm.heading_lin_vel(env)[:, :2]
    fraction = ((velocity * direction).sum(-1) / magnitude.clamp(min=0.04)).clamp(-1.0, 1.0)
    return fraction * (magnitude > 0.04) * wm.posture_gate(env, terrain_sensor_cfg)


def command_curriculum(env, env_ids, ramp_steps=24000):
    # A warm-start actor already balances; start walking immediately.
    progress = min(max(env.common_step_counter / max(ramp_steps, 1), 0.0), 1.0)
    ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    ranges.lin_vel_x = (0.10, 0.16 + 0.12 * progress)
    ranges.lin_vel_y = (-0.02 * progress, 0.02 * progress)
    ranges.ang_vel_z = (-0.08 * progress, 0.08 * progress)
    return torch.tensor(progress, device=env.device)
