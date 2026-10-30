"""Heading-frame commands and phase-conditioned rewards for an upright Go2.

All vertical penalties use WORLD Z. The upright robot's local +X points up;
its horizontal forward direction is local -Z, not the standard base +X.
"""
from __future__ import annotations
import math
import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

ROBOT = SceneEntityCfg("robot")

def upright_heading(env, asset_cfg=ROBOT):
    robot = env.scene[asset_cfg.name]
    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 2] = -1.0
    forward = quat_apply(robot.data.root_quat_w, forward)
    return torch.atan2(forward[:, 1], forward[:, 0])

def heading_lin_vel(env, asset_cfg=ROBOT):
    yaw = upright_heading(env, asset_cfg)
    v = env.scene[asset_cfg.name].data.root_lin_vel_w
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]), dim=-1)

class UprightVelocityCommand(UniformVelocityCommand):
    """Commands are forward/lateral in the upright horizontal heading frame."""
    def _update_command(self):
        self.vel_command_b[self.is_standing_env] = 0.0

    def _update_metrics(self):
        horizon = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += torch.linalg.norm(
            self.vel_command_b[:, :2] - heading_lin_vel(self._env)[:, :2], dim=-1
        ) / horizon
        self.metrics["error_vel_yaw"] += torch.abs(
            self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_w[:, 2]
        ) / horizon

    def _debug_vis_callback(self, event):
        # Standard command arrows assume base +X is horizontal; suppress them.
        return

@configclass
class UprightVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UprightVelocityCommand
    heading_command: bool = False

def ground_height(env, sensor_cfg):
    hits = env.scene.sensors[sensor_cfg.name].data.ray_hits_w[..., 2]
    fallback = env.scene.env_origins[:, 2, None]
    hits = torch.where(torch.isfinite(hits), hits, fallback)
    return torch.median(hits, dim=-1).values

def root_height(env, sensor_cfg, asset_cfg=ROBOT):
    return env.scene[asset_cfg.name].data.root_pos_w[:, 2] - ground_height(env, sensor_cfg)

def upright_score(env, asset_cfg=ROBOT, std=0.4):
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    target = torch.zeros_like(gravity)
    target[:, 0] = -1.0
    return torch.exp(-torch.sum((gravity - target).square(), dim=-1) / std**2)

def posture_gate(env, sensor_cfg, asset_cfg=ROBOT):
    h = root_height(env, sensor_cfg, asset_cfg)
    return upright_score(env, asset_cfg, std=0.7) * ((h - 0.35) / 0.15).clamp(0.0, 1.0)

def track_linear(env, command_name, std, sensor_cfg, asset_cfg=ROBOT):
    error = env.command_manager.get_command(command_name)[:, :2] - heading_lin_vel(env, asset_cfg)[:, :2]
    return torch.exp(-torch.sum(error.square(), dim=-1) / std**2) * posture_gate(env, sensor_cfg, asset_cfg)

def track_yaw(env, command_name, std, sensor_cfg, asset_cfg=ROBOT):
    error = env.command_manager.get_command(command_name)[:, 2] - env.scene[asset_cfg.name].data.root_ang_vel_w[:, 2]
    return torch.exp(-error.square() / std**2) * posture_gate(env, sensor_cfg, asset_cfg)

def vertical_velocity_l2(env, asset_cfg=ROBOT):
    return env.scene[asset_cfg.name].data.root_lin_vel_w[:, 2].square()

def horizontal_angular_velocity_l2(env, asset_cfg=ROBOT):
    return env.scene[asset_cfg.name].data.root_ang_vel_w[:, :2].square().sum(dim=-1)

def height_band_l2(env, sensor_cfg, low=0.48, high=0.59, asset_cfg=ROBOT):
    height = root_height(env, sensor_cfg, asset_cfg)
    return torch.relu(low - height).square() + torch.relu(height - high).square()

def moving(env, command_name):
    cmd = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(cmd[:, :2], dim=-1) > 0.04) | (cmd[:, 2].abs() > 0.08)

def phases(env, period=1.0):
    phase = env.episode_length_buf.to(torch.float32) * env.step_dt / period
    return torch.remainder(phase[:, None] + torch.tensor([0.0, 0.5], device=env.device), 1.0)

def gait_clock(env, period=1.0, command_name="base_velocity"):
    phase = phases(env, period)[:, 0] * (2.0 * math.pi)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1) * moving(env, command_name)[:, None]

def phase_contact_reward(env, sensor_cfg, terrain_sensor_cfg, period=1.0, stance_ratio=0.6, command_name="base_velocity"):
    contacts = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0
    stance = phases(env, period) < stance_ratio
    matches = (contacts == stance).to(torch.float32).mean(dim=-1)
    return matches * moving(env, command_name) * posture_gate(env, terrain_sensor_cfg)

def swing_clearance_reward(env, asset_cfg, sensor_cfg, terrain_sensor_cfg, period=1.0,
                           stance_ratio=0.6, clearance=0.06, std=0.03, command_name="base_velocity"):
    phase = phases(env, period)
    swing_phase = ((phase - stance_ratio) / (1.0 - stance_ratio)).clamp(0.0, 1.0)
    target = clearance * torch.sin(math.pi * swing_phase)
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    # Relative to the other foot, avoids rewarding a simultaneous two-leg jump.
    relative_height = foot_z - foot_z.min(dim=1, keepdim=True).values
    contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0
    swing = (phase >= stance_ratio) & (~contact) & contact.flip(dims=[1])
    score = torch.exp(-(relative_height - target).square() / std**2)
    return (score * swing).sum(dim=-1) * moving(env, command_name) * posture_gate(env, terrain_sensor_cfg)

def feet_slide_l2(env, asset_cfg, sensor_cfg):
    sensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    velocity = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return (velocity.square().sum(dim=-1) * contacts).sum(dim=-1)

def flight_penalty(env, sensor_cfg, command_name="base_velocity"):
    contacts = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0
    return (~contacts.any(dim=1)).to(torch.float32) * moving(env, command_name)

def feet_crossing_penalty(env, asset_cfg, minimum_width=0.10):
    feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
    delta = feet[:, 0] - feet[:, 1]  # explicit RL, RR ordering
    yaw = upright_heading(env)
    width = -torch.sin(yaw) * delta[:, 0] + torch.cos(yaw) * delta[:, 1]
    return torch.relu(minimum_width - width).square()

def air_time_reward(env, sensor_cfg, terrain_sensor_cfg, command_name="base_velocity", threshold=0.3):
    # Cassie-style single-support airtime, also active at our lower command speed.
    data = env.scene.sensors[sensor_cfg.name].data
    air, contact = data.current_air_time[:, sensor_cfg.body_ids], data.current_contact_time[:, sensor_cfg.body_ids]
    support = contact > 0.0
    duration = torch.where(support, contact, air)
    reward = torch.where(support.sum(dim=-1, keepdim=True) == 1, duration, 0.0).min(dim=-1).values
    return reward.clamp(max=threshold) * moving(env, command_name) * posture_gate(env, terrain_sensor_cfg)

def bad_upright_orientation(env, limit_angle=0.95, grace_s=0.3, asset_cfg=ROBOT):
    cos = (-env.scene[asset_cfg.name].data.projected_gravity_b[:, 0]).clamp(-1, 1)
    return (cos < math.cos(limit_angle)) & (env.episode_length_buf * env.step_dt > grace_s)

def too_low(env, sensor_cfg, min_height=0.35, grace_s=0.3):
    return (root_height(env, sensor_cfg) < min_height) & (env.episode_length_buf * env.step_dt > grace_s)

def command_curriculum(env, env_ids, warmup_steps=12000, ramp_steps=48000):
    """Training-step curriculum; fresh playback disables it explicitly."""
    progress = min(max((env.common_step_counter - warmup_steps) / max(ramp_steps, 1), 0.0), 1.0)
    term = env.command_manager.get_term("base_velocity")
    if env.common_step_counter < warmup_steps:
        term.cfg.ranges.lin_vel_x = (0.0, 0.0)
        term.cfg.ranges.lin_vel_y = (0.0, 0.0)
        term.cfg.ranges.ang_vel_z = (0.0, 0.0)
    else:
        term.cfg.ranges.lin_vel_x = (0.08, 0.12 + 0.23 * progress)
        term.cfg.ranges.lin_vel_y = (-0.03 * progress, 0.03 * progress)
        term.cfg.ranges.ang_vel_z = (-0.10 * progress, 0.10 * progress)
    return torch.tensor(progress, device=env.device)

def reset_upright_root(env, env_ids, asset_cfg=ROBOT, yaw_range=(-0.15, 0.15)):
    """Apply yaw about WORLD Z, preserving base +X up for every reset."""
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
    robot = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    states = robot.data.default_root_state[env_ids].clone()
    states[:, :3] += env.scene.env_origins[env_ids]
    states[:, :2] += torch.empty((len(env_ids), 2), device=env.device).uniform_(-0.03, 0.03)
    yaw = torch.empty(len(env_ids), device=env.device).uniform_(*yaw_range)
    zeros = torch.zeros_like(yaw)
    states[:, 3:7] = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), states[:, 3:7])
    states[:, 7:] = 0.0
    robot.write_root_pose_to_sim(states[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(states[:, 7:], env_ids=env_ids)

def terrain_curriculum(env, env_ids):
    """Update difficulty only after a genuine commanded rollout, never at init."""
    terrain = env.scene.terrain
    if env.common_step_counter == 0:
        return terrain.terrain_levels.float().mean()
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_command("base_velocity")[env_ids, :2]
    active = torch.linalg.norm(cmd, dim=-1) > 0.04
    distance = torch.linalg.norm(
        env.scene["robot"].data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=-1)
    up = active & (distance > terrain.cfg.terrain_generator.size[0] * 0.5)
    down = active & (~up) & (distance < torch.linalg.norm(cmd, dim=-1) * env.max_episode_length_s * 0.5)
    terrain.update_env_origins(env_ids, up, down)
    return terrain.terrain_levels.float().mean()
