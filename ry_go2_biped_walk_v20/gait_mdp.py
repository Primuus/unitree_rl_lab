"""V7 gait terms with contact and foot-height feedback for closed-loop stepping."""
import math
import torch
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk import walk_mdp as wm


def world_velocity_tracking(env, terrain_sensor_cfg, command_name="base_velocity"):
    command = env.command_manager.get_command(command_name)[:, 0]
    velocity = env.scene["robot"].data.root_lin_vel_w[:, 0]
    std = (0.5 * command.abs()).clamp(min=0.05, max=0.10)
    return torch.exp(-((command - velocity) / std).square()) * wm.posture_gate(env, terrain_sensor_cfg)


def world_forward_progress(env, terrain_sensor_cfg, command_name="base_velocity"):
    command = env.command_manager.get_command(command_name)[:, 0]
    velocity = env.scene["robot"].data.root_lin_vel_w[:, 0]
    return (velocity / command.abs().clamp(min=0.04)).clamp(-1.0, 1.0) * (command.abs() > 0.04) * wm.posture_gate(env, terrain_sensor_cfg)


def strict_yaw_tracking(env, terrain_sensor_cfg, command_name="base_velocity"):
    yaw_rate = env.scene["robot"].data.root_ang_vel_w[:, 2]
    command = env.command_manager.get_command(command_name)[:, 2]
    error = command - yaw_rate
    return error.square().clamp(max=4.0) * wm.posture_gate(env, terrain_sensor_cfg)


def gait_targets(env, period=1.0, stance_ratio=0.5):
    phase = wm.phases(env, period)
    swing_phase = ((phase - stance_ratio) / (1.0 - stance_ratio)).clamp(0.0, 1.0)
    swing = phase >= stance_ratio
    arc = torch.sin(math.pi * swing_phase).clamp(min=0.0) * swing
    support = 1.0 - torch.sqrt(arc)
    target = support / support.sum(-1, keepdim=True).clamp(min=1e-6)
    return swing, arc, target


def phase_load(env, sensor_cfg, asset_cfg, terrain_sensor_cfg, robot_weight, period=1.0, command_name="base_velocity"):
    _, _, target = gait_targets(env, period)
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    actual = (forces.clamp(min=0.0) / robot_weight).clamp(max=2.0)
    return (actual - target).square().sum(-1) * wm.moving(env, command_name) * wm.posture_gate(env, terrain_sensor_cfg)


def swing_clearance(env, asset_cfg, sensor_cfg, terrain_sensor_cfg, robot_weight, period=1.0, clearance=0.08, std=0.04, command_name="base_velocity"):
    swing, arc, _ = gait_targets(env, period)
    z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = z - z.min(-1, keepdim=True).values
    target = clearance * arc
    score = torch.exp(-torch.abs(height - target) / std)
    loads = (env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0) / robot_weight).clamp(0.0, 2.0)
    support_load = loads.flip([-1])
    unload_gate = torch.sigmoid((0.15 - loads) * 20.0)
    support_gate = torch.sigmoid((support_load - 0.35) * 12.0)
    return (score * swing * unload_gate * support_gate).sum(-1) * wm.moving(env, command_name) * wm.posture_gate(env, terrain_sensor_cfg)


def heading_features(env):
    """Expose heading relative to the world-forward command."""
    yaw = wm.upright_heading(env)
    return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)



def rear_foot_height(env, asset_cfg):
    """Return each rear foot height relative to the lower rear foot."""
    z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    relative = z - z.min(dim=-1, keepdim=True).values
    return (relative * 10.0).clamp(0.0, 1.0)


def rear_foot_load(env, sensor_cfg, robot_weight):
    """Expose normalized vertical rear-foot loads to the policy."""
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (forces.clamp(min=0.0) / robot_weight).clamp(0.0, 1.5)


def heading_alignment(env, terrain_sensor_cfg, sigma=0.6):
    yaw = wm.upright_heading(env)
    wrapped = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return torch.exp(-(wrapped / sigma).square()) * wm.posture_gate(env, terrain_sensor_cfg)


def swing_forward_position(env, asset_cfg, sensor_cfg, terrain_sensor_cfg, robot_weight,
                           period=1.0, stride=0.10, command_name="base_velocity"):
    """Dense phase-conditioned fore-aft swing placement with support-foot gating."""
    swing, arc, _ = gait_targets(env, period)
    feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, :2]
    relative_x = feet[:, 0, 0] - feet[:, 1, 0]
    target = stride * (arc[:, 0] - arc[:, 1])
    loads = rear_foot_load(env, sensor_cfg, robot_weight)
    swing_load = torch.where(swing, loads, 0.0)
    support_load = torch.where(swing, loads.flip([-1]), 0.0)
    unload_gate = torch.sigmoid((0.35 - swing_load) * 10.0)
    support_gate = torch.sigmoid((support_load - 0.25) * 10.0)
    error = (relative_x - target).abs()
    score = torch.exp(-((error / 0.05) ** 2))
    return score * (swing * unload_gate * support_gate).sum(-1) * wm.moving(
        env, command_name
    ) * wm.posture_gate(env, terrain_sensor_cfg)


def swing_forward_speed(env, asset_cfg, sensor_cfg, terrain_sensor_cfg, robot_weight,
                        period=1.0, command_name="base_velocity"):
    """Reward forward foot motion relative to the body only during swing."""
    swing, _, _ = gait_targets(env, period)
    loads = rear_foot_load(env, sensor_cfg, robot_weight)
    swing_load = torch.where(swing, loads, 0.0)
    support_load = torch.where(swing, loads.flip([-1]), 0.0)
    unload_gate = torch.sigmoid((0.35 - swing_load) * 10.0)
    support_gate = torch.sigmoid((support_load - 0.25) * 10.0)
    foot_vx = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, 0]
    base_vx = env.scene["robot"].data.root_lin_vel_w[:, 0:1]
    relative_forward_vx = (foot_vx - base_vx).clamp(0.0, 1.0)
    reward = (relative_forward_vx * swing * unload_gate * support_gate).sum(-1)
    return reward * wm.moving(env, command_name) * wm.posture_gate(env, terrain_sensor_cfg)


def qualified_touchdown_reward(env, asset_cfg, sensor_cfg, terrain_sensor_cfg,
                               robot_weight, period=1.0, min_air=0.06,
                               min_lift=0.02, min_support=0.25,
                               command_name="base_velocity"):
    """Reward qualified landings and give a bonus only for alternating feet."""
    sensor = env.scene.sensors[sensor_cfg.name]
    data = sensor.data
    contact_time = data.current_contact_time[:, sensor_cfg.body_ids]
    contact = contact_time > 0.0
    z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = z - z.min(dim=-1, keepdim=True).values
    if not hasattr(env, "_v19_prev_contact"):
        env._v19_prev_contact = contact.clone()
        env._v19_peak_height = height.clone()
        env._v19_pending_touchdown = torch.zeros_like(contact)
        env._v19_last_touchdown_foot = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )

    first_step = env.episode_length_buf <= 1
    if first_step.any():
        env._v19_prev_contact[first_step] = contact[first_step]
        env._v19_peak_height[first_step] = height[first_step]
        env._v19_pending_touchdown[first_step] = False
        env._v19_last_touchdown_foot[first_step] = -1

    detached = ~contact
    env._v19_peak_height = torch.where(
        detached, torch.maximum(env._v19_peak_height, height), env._v19_peak_height
    )
    landing = contact & ~env._v19_prev_contact
    qualified_air = (
        (data.last_air_time[:, sensor_cfg.body_ids] >= min_air)
        & (env._v19_peak_height >= min_lift)
    )
    env._v19_pending_touchdown |= landing & qualified_air

    force = data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0)
    loads = (force / robot_weight).clamp(0.0, 2.0)
    opposite_support = loads.flip(-1) >= min_support
    event = env._v19_pending_touchdown & (contact_time >= 0.04) & opposite_support
    simultaneous = event.all(-1)
    event &= ~simultaneous[:, None]

    alternating = torch.zeros_like(event)
    for foot in range(2):
        alternating[:, foot] = (
            event[:, foot]
            & (env._v19_last_touchdown_foot >= 0)
            & (env._v19_last_touchdown_foot != foot)
        )
        env._v19_last_touchdown_foot = torch.where(
            event[:, foot],
            torch.full_like(env._v19_last_touchdown_foot, foot),
            env._v19_last_touchdown_foot,
        )
    env._v19_pending_touchdown &= ~event
    env._v19_prev_contact = contact.clone()
    env._v19_peak_height = torch.where(
        contact, torch.zeros_like(height), env._v19_peak_height
    )

    event_count = event.to(torch.float32).sum(-1)
    alternate_count = alternating.to(torch.float32).sum(-1)
    return (event_count + 3.0 * alternate_count) * wm.moving(
        env, command_name
    ) * wm.posture_gate(env, terrain_sensor_cfg)
