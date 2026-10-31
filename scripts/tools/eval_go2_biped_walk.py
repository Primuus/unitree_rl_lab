"""Evaluate the current full-Go2 task; zero actions only validate the execution path."""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
faulthandler.enable()

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="RY-Unitree-Go2-Biped-Gait-v20")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--stage", choices=["balance", "walk"], default="walk")
parser.add_argument("--gait_clock", choices=["auto", "always", "moving_only"], default="auto",
                    help="auto reads the checkpoint observation contract; overrides are diagnostic")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--speed", type=float, default=0.2)
parser.add_argument("--video", action="store_true")
parser.add_argument("--policy_transform", choices=["none", "mirror_average"], default="none")
parser.add_argument("--train_friction", action="store_true")
parser.add_argument("--train_noise", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", type=Path, default=Path("logs/evaluation/current"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.checkpoint and not args.checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {args.checkpoint}")
if args.video:
    args.enable_cameras = True

import torch
import torch._inductor.utils
import torchvision
import typing_extensions
torch.set_num_threads(8)
if args.device.startswith("cuda"):
    torch.cuda.set_device(torch.device(args.device))
launcher = AppLauncher(dict(vars(args), multi_gpu=False))
app = launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from ry_unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from ry_unitree_rl_lab.utils.phase_contract import read_gait_clock
from isaaclab_tasks.utils import load_cfg_from_registry
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk import walk_mdp as wm
from isaaclab.managers import SceneEntityCfg
import ry_go2_biped_walk_v20
from ry_go2_biped_walk_v20.gait_env_cfg import configure_stage
from ry_go2_biped_walk_v20.gait_mdp import validate_layout
from ry_go2_biped_walk_v20.symmetry import mirror_data, transform_joints, ACTION_NAMES
from go2_walk_metrics import GaitMetrics


def main():
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs,
                        entry_point_key="play_env_cfg_entry_point")
    configure_stage(cfg, args.stage)
    if args.gait_clock != "auto":
        cfg.balance_gait_clock = args.gait_clock
    elif args.checkpoint:
        clock = read_gait_clock(args.checkpoint.parent/"params/provenance.json")
        cfg.balance_gait_clock = clock["mode"]
        if clock["period_s"] != cfg.observations.policy.gait_phase.params["period"]:
            raise ValueError("Checkpoint gait period differs from current evaluation configuration")
    cfg.seed = args.seed
    cfg.episode_length_s = max(cfg.episode_length_s, args.steps * cfg.sim.dt * cfg.decimation + 1)
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.speed, args.speed) if args.stage == "walk" else (0., 0.)
    # Evaluate each command separately; never inherit the training mixture.
    cfg.commands.base_velocity.rel_standing_envs = 1.0 if args.stage == "balance" else 0.0
    cfg.events.reset_root.params["yaw_range"] = (0.0, 0.0)
    if args.train_friction:
        training = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        cfg.events.friction = training.events.friction
    cfg.observations.policy.enable_corruption = args.train_noise
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    raw = env.unwrapped
    layout = validate_layout(raw)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    obs, _ = wrapped.get_observations()
    policy = None
    if args.checkpoint:
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        if args.policy_transform == "mirror_average":
            base_policy = policy
            def policy(observation):
                mirrored = mirror_data(raw, obs=observation, obs_type="policy")[0][raw.num_envs:]
                return 0.5 * (base_policy(observation) +
                              transform_joints(base_policy(mirrored), ACTION_NAMES))
    robot = raw.scene["robot"]
    foot_ids, foot_names = robot.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)
    sensor = raw.scene["contact_forces"]
    sensor_ids, _ = sensor.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)
    metrics = GaitMetrics(raw, robot, foot_ids, sensor, sensor_ids, wm, SceneEntityCfg("height_scanner"))
    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(args.out.with_suffix(".mp4")), fps=25, codec="libx264", macro_block_size=None)
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                action = policy(obs) if policy else torch.zeros(raw.num_envs, 12, device=raw.device)
                obs, reward, done, extras = wrapped.step(action)
                if not all(bool(torch.isfinite(t).all()) for t in [obs, reward, robot.data.root_state_w]):
                    raise AssertionError(f"Non-finite simulation value at step {step}")
                metrics.update(step, done)
                if writer and step % 2 == 0:
                    p = robot.data.root_pos_w[0].tolist()
                    raw.sim.set_camera_view(eye=(p[0]+1.7,p[1]-2.,p[2]+.7), target=tuple(p))
                    writer.append_data(env.render())
    finally:
        if writer:
            writer.close()
    result = metrics.result(policy is not None)
    result.update(task=args.task, stage=args.stage, checkpoint=str(args.checkpoint) if args.checkpoint else None,
                  gait_clock_override=args.gait_clock,
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
                  mode="trained_policy" if policy else "zero_action_smoke", finite=True,
                  steps=args.steps, seconds_per_env=args.steps*raw.step_dt, num_envs=raw.num_envs,
                  action_dim=12, hind_feet=foot_names, layout=layout)
    args.out.with_suffix(".trace.json").write_text(json.dumps({"samples": metrics.trace, "first_env_landings": metrics.landings}, indent=2))
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print("GO2_WALK_EVALUATION", json.dumps(result), flush=True)
    wrapped.close()


try:
    main()
except BaseException:
    import traceback, sys
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    app.close()
