"""Evaluate this task without the legacy play.py terrain overrides.

Use --checkpoint for a trained policy. Without one this is a physics/MDP smoke
test with zero actions, NOT evidence of a successful walking policy.
"""
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
import argparse
import json
import hashlib
from collections.abc import Mapping
from go2_walk_metrics import GaitMetrics
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-Go2-Biped-Walk")
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--speed", type=float, default=0.2)
parser.add_argument("--video", action="store_true")
parser.add_argument("--policy_transform", choices=["none","mirror_average","share_right"], default="none")
parser.add_argument("--use_train_rewards", action="store_true")
parser.add_argument("--train_friction", action="store_true")
parser.add_argument("--train_noise", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=Path, default=Path("logs/go2_biped_walk_validation/evaluation"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.checkpoint and not Path(args.checkpoint).is_file():
    parser.error(f"Checkpoint does not exist: {args.checkpoint}")
if args.video:
    args.enable_cameras = True
import torch
import torch._inductor.utils
import torchvision
import typing_extensions
print("TORCH_IMPORT_PREFLIGHT", torch.__version__, typing_extensions.__file__, flush=True)
torch.set_num_threads(8)
if args.device.startswith("cuda"):
    torch.cuda.set_device(torch.device(args.device))
launcher = AppLauncher(dict(vars(args), multi_gpu=False))
app = launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import ry_unitree_rl_lab.tasks
from ry_unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk import walk_mdp as wm
from isaaclab.managers import SceneEntityCfg

def main():
    from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk.validation import validate_reference_frames
    validate_reference_frames()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device,
                        num_envs=args.num_envs, entry_point_key="play_env_cfg_entry_point")
    if args.use_train_rewards or args.train_friction or args.train_noise:
        train_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        if args.use_train_rewards:
            cfg.rewards = train_cfg.rewards
        if args.train_friction:
            cfg.events.friction = train_cfg.events.friction
        if args.train_noise:
            cfg.observations.policy.enable_corruption = train_cfg.observations.policy.enable_corruption
    cfg.seed = args.seed
    cfg.episode_length_s = max(cfg.episode_length_s, args.steps * cfg.sim.dt * cfg.decimation + 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.speed, args.speed)
    cfg.events.reset_root.params["yaw_range"] = (0.0, 0.0)
    print("EVAL_ENV_BUILD_START", flush=True)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    print("EVAL_ENV_BUILD_COMPLETE", flush=True)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    wrapped = RslRlVecEnvWrapper(env)
    if raw.action_manager.total_action_dim != 6:
        raise AssertionError(f"Expected 6 actions, got {raw.action_manager.total_action_dim}")
    obs = wrapped.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    policy = None
    if args.checkpoint:
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=args.device)
        if args.policy_transform != "none":
            from ry_go2_biped_walk_v20.symmetry import (
                mirror_policy, transform_joints, ACTION_NAMES, OBS_JOINT_NAMES)
            native_joints = list(robot.joint_names)
            assert native_joints == OBS_JOINT_NAMES, native_joints
            assert list(raw.action_manager.get_term("joint_pos")._joint_names) == ACTION_NAMES
            obs_names = raw.observation_manager.active_terms["policy"]
            obs_dims = [int(torch.tensor(d).prod()) for d in raw.observation_manager.group_obs_term_dim["policy"]]
            base_policy = policy
            def symmetry_policy(observation):
                original = base_policy(observation)
                reflected_obs = mirror_policy(observation, obs_names, obs_dims, native_joints)
                reflected = transform_joints(base_policy(reflected_obs), ACTION_NAMES)
                if args.policy_transform == "mirror_average":
                    return (original + reflected) * 0.5
                combined = original.clone()
                combined[:, :3] = reflected[:, :3]
                return combined
            policy = symmetry_policy
            print("POLICY_TRANSFORM", args.policy_transform, flush=True)
        print("EVAL_CHECKPOINT_LOADED", flush=True)
    foot_ids, foot_names = robot.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)
    sensor = raw.scene.sensors["contact_forces"]
    sensor_ids, _ = sensor.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)
    scanner = SceneEntityCfg("height_scanner")
    dt = raw.step_dt
    metrics = GaitMetrics(raw, robot, foot_ids, sensor, sensor_ids, wm, scanner)
    finite = True
    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(args.out.with_suffix(".mp4")), fps=25, codec="libx264", macro_block_size=None)
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                action = policy(obs) if policy else torch.zeros(args.num_envs, 6, device=raw.device)
                obs, reward, done, extras = wrapped.step(action)
                if isinstance(obs, tuple):
                    obs = obs[0]
                tensors = list(obs.values()) if isinstance(obs, Mapping) else [obs]
                finite &= all(bool(torch.isfinite(t).all()) for t in tensors)
                finite &= bool(torch.isfinite(reward).all() and torch.isfinite(robot.data.root_state_w).all())
                if not finite:
                    raise AssertionError(f"Non-finite simulation value at step {step}")
                metrics.update(step, done)
                if writer is not None and step % 2 == 0:
                    p = robot.data.root_pos_w[0].detach().cpu().tolist()
                    raw.sim.set_camera_view(
                        eye=(p[0]+1.7,p[1]-2.0,p[2]+0.7),
                        target=(p[0],p[1],p[2]+0.05))
                    writer.append_data(env.render())
    finally:
        if writer is not None:
            writer.close()
    result = metrics.result(policy is not None)
    result["policy_transform"] = args.policy_transform
    result["config_flags"] = {"use_train_rewards": args.use_train_rewards,
                              "train_friction": args.train_friction, "train_noise": args.train_noise}
    result.update(checkpoint=args.checkpoint, task=args.task, seed=args.seed,
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
        mode="trained_policy" if policy else "zero_action_smoke", steps=args.steps,
        seconds_per_env=args.steps*dt, num_envs=args.num_envs, finite=finite, hind_feet=foot_names, action_dim=6)
    args.out.with_suffix(".trace.json").write_text(json.dumps(dict(samples=metrics.trace, first_env_landings=metrics.landings),indent=2))
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print("GO2_WALK_EVALUATION " + json.dumps(result), flush=True)
    wrapped.close()

try:
    main()
except BaseException:
    import traceback, sys, os
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    app.close()
