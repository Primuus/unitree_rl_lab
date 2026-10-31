"""Isaac Lab tensor adapter for the copied upstream reward methods."""
from types import SimpleNamespace

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk import walk_mdp as wm
from .reference_config import REFERENCE
from .reference_rewards import ReferenceRewards
from .robot_cfg import JOINT_NAMES, FOOT_NAMES
from .stepping_rewards import step_targets, clearance_score, contact_score


class RewardState(ReferenceRewards):
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.sensor = env.scene["contact_forces"]
        self.num_envs, self.device, self.num_dof = env.num_envs, env.device, 12
        self.dt = env.step_dt
        self.joint_ids, names = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        if names != JOINT_NAMES:
            raise ValueError(f"Reference joint order mismatch: {names}")
        self.feet_indices, names = self.robot.find_bodies(FOOT_NAMES, preserve_order=True)
        if names != FOOT_NAMES:
            raise ValueError(f"Reference foot order mismatch: {names}")
        self.feet_name_reward_indices = self.feet_indices[:2]
        self.contact_foot_indices = self.feet_indices[2:]
        self.sensor_ids, _ = self.sensor.find_bodies(self.robot.body_names, preserve_order=True)
        self.penalised_contact_indices, _ = self.robot.find_bodies([".*thigh", ".*calf"])
        self.cfg = SimpleNamespace(rewards=SimpleNamespace(
            **REFERENCE["reward_parameters"], tracking_lin_vel_sigma=env.cfg.tracking_lin_vel_sigma,
            tracking_ang_vel_sigma=env.cfg.tracking_ang_vel_sigma))
        self.target_gravity = torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        self.descire_joint_pos = torch.tensor(
            [REFERENCE["desired_joint_angles"][name] for name in JOINT_NAMES], device=self.device)
        self.soft_dof_pos_limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids]
        self.measured_heights = torch.zeros(self.num_envs, 1, device=self.device)
        self.previous_velocity = torch.zeros(self.num_envs, 12, device=self.device)
        self.tick = -1

    def refresh(self):
        env, data = self.env, self.robot.data
        if self.tick == env.common_step_counter:
            return
        self.tick = env.common_step_counter
        self.episode_length_buf = env.episode_length_buf
        self.root_states = data.root_state_w
        self.base_lin_vel = data.root_lin_vel_b
        self.base_ang_vel = data.root_ang_vel_b
        self.projected_gravity = data.projected_gravity_b
        self.base_euler_xyz = torch.stack(euler_xyz_from_quat(data.root_quat_w), dim=-1)
        self.base_euler_xyz = torch.where(self.base_euler_xyz > torch.pi,
                                        self.base_euler_xyz - 2 * torch.pi, self.base_euler_xyz)
        self.dof_pos = data.joint_pos[:, self.joint_ids]
        self.dof_vel = data.joint_vel[:, self.joint_ids]
        # Upstream uses the previous control-step velocity, not physics-step joint_acc.
        self.last_dof_vel = self.previous_velocity.clone()
        self.previous_velocity.copy_(self.dof_vel)
        self.torques = data.applied_torque[:, self.joint_ids]
        self.actions = env.action_manager.action
        self.last_actions = env.action_manager.prev_action
        self.commands = env.command_manager.get_command("base_velocity")
        self.rigid_state = data.body_pos_w
        self.feet_pos = self.rigid_state[:, self.feet_name_reward_indices]
        self.contact_forces = self.sensor.data.net_forces_w[:, self.sensor_ids]
        # Compute before all terms, avoiding the upstream reward-order/one-step lag.
        self._reward_base_height()


def state(env):
    if not hasattr(env, "_reference_reward_state"):
        env._reference_reward_state = RewardState(env)
    env._reference_reward_state.refresh()
    return env._reference_reward_state


def reference_reward(env, name):
    value = getattr(state(env), "_reward_" + name)()
    return torch.as_tensor(value, device=env.device).expand(env.num_envs)


def balance_zero_yaw_l2(env):
    """Leggedstand zero-turn penalty, adapted to WORLD Z and per-env height gates."""
    s = state(env)
    command = s.commands[:, 2]
    error = command - s.robot.data.root_ang_vel_w[:, 2]
    return error.square() * (s.rew_hanstand > .78) * (command.abs() < .1)


def balance_heading_alignment(env, sensor_cfg, std=0.6):
    """Original-project heading reward toward world +X (nominal reset heading)."""
    yaw = wm.upright_heading(env)
    error = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return torch.exp(-(error / std).square()) * wm.posture_gate(env, sensor_cfg)


def balance_step_reward(env, name):
    s = state(env)
    target, stance = step_targets(s._get_phase(), env.cfg.balance_step_height,
                                  env.cfg.balance_double_support_fraction)
    if name == "feet_clearance":
        data = s.robot.data
        ids = s.contact_foot_indices
        # Original Go2 URDF foot collision: sphere radius .022, local center (-.002,0,0).
        offset = data.body_pos_w[:, ids].new_tensor([-.002, 0., 0.]).expand(env.num_envs, 2, 3)
        center = data.body_pos_w[:, ids] + quat_apply(data.body_quat_w[:, ids], offset)
        height = (center[:, :, 2] - .022 - s.measured_heights.mean(-1, keepdim=True)).clamp(min=0.)
        value = clearance_score(height, target, env.cfg.balance_clearance_width)
    else:
        contact = s.contact_forces[:, s.contact_foot_indices, 2] > 1.0
        value = contact_score(contact, stance)
    return value * (s.rew_hanstand > .70)


def reset_reference_history(env, env_ids):
    if hasattr(env, "_reference_reward_state"):
        s = env._reference_reward_state
        s.previous_velocity[env_ids] = 0.0
        s.tick = -1


def joint_position_rel(env):
    robot = env.scene["robot"]
    default = robot.data.joint_pos.new_tensor(
        [REFERENCE["default_joint_angles"][name] for name in robot.joint_names])
    return robot.data.joint_pos - default


def heading_features(env):
    yaw = wm.upright_heading(env)
    return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)


def rear_foot_height(env, asset_cfg):
    z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    return ((z - z.min(dim=-1, keepdim=True).values) * 10.0).clamp(0.0, 1.0)


def rear_foot_load(env, sensor_cfg, robot_weight):
    force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (force.clamp(min=0.0) / robot_weight).clamp(0.0, 1.5)


def validate_layout(env):
    from .symmetry import mirror_data
    from .reference_config import REFERENCE
    action = env.action_manager.get_term("joint_pos")
    names = list(action._joint_names)
    dims = {group: sum(int(torch.tensor(d).prod()) for d in shape)
            for group, shape in env.observation_manager.group_obs_term_dim.items()}
    if names != JOINT_NAMES or env.action_manager.total_action_dim != 12:
        raise ValueError(f"Expected 12 ordered actions: {names}")
    if dims != {"policy": 56, "critic": 155}:
        raise ValueError(f"Unexpected observation dimensions: {dims}")
    mirror_data(env, actions=torch.zeros(env.num_envs, 12, device=env.device))
    s = state(env)
    expected = {k: v for k, v in REFERENCE["reward_scales"].items() if v != 0}
    if env.cfg.stage == "balance":
        expected["contact"] = 0.9
        expected["feet_clearance"] = 1.0
        expected["foot_slip"] = -0.5
        expected["tracking_ang_vel_zero"] = -0.2
        expected["heading_alignment"] = 1.5
    active = {name: env.reward_manager.get_term_cfg(name).weight
              for name in env.reward_manager.active_terms}
    if active != expected:
        raise ValueError(f"Reward scales differ from reference/stage: {active}, {expected}")
    return {"action_names": names, "native_joint_names": list(env.scene["robot"].joint_names),
            "observation_dims": dims, "reward_scales": active,
            "front_foot_ids": s.feet_name_reward_indices, "hind_foot_ids": s.contact_foot_indices,
            "tracking_lin_vel_sigma": s.cfg.rewards.tracking_lin_vel_sigma,
            "tracking_ang_vel_sigma": s.cfg.rewards.tracking_ang_vel_sigma,
            "balance_yaw_constraints": {
                "zero_yaw_axis": "world_z", "zero_yaw_weight": -0.2,
                "zero_command_threshold_radps": 0.1, "zero_yaw_height_gate": 0.78,
                "heading_target_rad": 0.0, "heading_direction": "world_positive_x",
                "heading_weight": active["heading_alignment"],
                "heading_std_rad": env.reward_manager.get_term_cfg("heading_alignment").params["std"],
                "heading_gate": "upright_score(std=0.7) * clamp((height-0.35)/0.15,0,1)",
            } if env.cfg.stage == "balance" else None,
            "balance_stepping": {"peak_height": env.cfg.balance_step_height,
                                 "double_support_fraction": env.cfg.balance_double_support_fraction,
                                 "clearance_width": env.cfg.balance_clearance_width,
                                 "cycle_time": s.cfg.rewards.cycle_time} if env.cfg.stage == "balance" else None}
