"""Train the current 12-joint task in place; legacy filename remains the entry point."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone
import faulthandler
faulthandler.enable()

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=15000)
parser.add_argument("--save_interval", type=int, help="Checkpoint interval in PPO iterations")
parser.add_argument("--run_name", default="handstand")
parser.add_argument("--stage", choices=["balance", "walk"], default="balance")
parser.add_argument("--resume", type=Path)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.resume and not args.resume.is_file():
    parser.error(f"Checkpoint does not exist: {args.resume}")

import torch
import torch._inductor.utils
import torchvision
import typing_extensions
torch.set_num_threads(8)
if args.device.startswith("cuda"):
    torch.cuda.set_device(torch.device(args.device))
    torch.zeros(1, device=args.device)
    torch.cuda.synchronize(args.device)
launcher = AppLauncher(dict(vars(args), multi_gpu=False))
app = launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry
from ry_unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from ry_unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
import ry_go2_biped_walk_v20
from ry_go2_biped_walk_v20.gait_env_cfg import configure_stage
from ry_go2_biped_walk_v20.gait_mdp import validate_layout

TASK = "RY-Unitree-Go2-Biped-Gait-v20"


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    configure_stage(cfg, args.stage)
    cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.seed, agent_cfg.device = args.seed, args.device
    agent_cfg.max_iterations = args.max_iterations
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs/rsl_rl") / agent_cfg.experiment_name / f"{stamp}_{args.stage}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    print("TRAIN_LOG_DIR", str(log_dir.resolve()), flush=True)
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    layout = validate_layout(env.unwrapped)
    print("CURRENT_TASK_LAYOUT", json.dumps(layout), flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=args.device)
    if args.resume:
        runner.load(str(args.resume), load_optimizer=True)
    policy = runner.alg.policy
    params = log_dir / "params"
    params.mkdir(exist_ok=True)
    dump_yaml(str(params / "env.yaml"), cfg)
    dump_yaml(str(params / "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, str(log_dir))
    project = Path(__file__).resolve().parents[1]
    source_files = list((project / "ry_go2_biped_walk_v20").glob("*.py"))
    source_files += [Path(__file__).resolve(), project / "doc/reference_port_manifest.json"]
    source_files += list((project / "third_party/My_unitree_go2_gym").glob("*"))
    hashes = {}
    for source in source_files:
        relative = source.relative_to(project)
        target = params / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (params / "provenance.json").write_text(json.dumps({
        "task": TASK, "implementation": "12-joint-reference-handstand",
        "stage": args.stage, "resume": str(args.resume) if args.resume else None,
        "initialization": "resume" if args.resume else "scratch",
        "layout": layout, "seed": args.seed, "source_sha256": hashes,
    }, indent=2))
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    export_dir = str(log_dir / "exported")
    export_policy_as_jit(policy, normalizer=None, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy, normalizer=None, path=export_dir, filename="policy.onnx")
    print("POLICY_EXPORTED", export_dir, flush=True)
    env.close()
    print("TRAINING_COMPLETE", flush=True)


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
