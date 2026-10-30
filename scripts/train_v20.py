"""Train v20 with a heading-aware actor warm start and fresh critic/optimizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
import faulthandler
faulthandler.enable()

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=4000)
parser.add_argument("--run_name", default="reference_walk_v20")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--warm_start_actor", type=Path, required=True)
args = parser.parse_args()

import torch
import torch._inductor.utils
import torchvision
import typing_extensions
print("TORCH_IMPORT_PREFLIGHT", torch.__version__, typing_extensions.__file__, flush=True)
torch.set_num_threads(8)
if args.device.startswith("cuda"):
    torch.cuda.set_device(torch.device(args.device))
    torch.zeros(1, device=args.device)
    torch.cuda.synchronize(args.device)
    print("CUDA_READY", args.device, torch.cuda.get_device_name(args.device), flush=True)
launch_args = vars(args).copy()
launch_args["multi_gpu"] = False
launcher = AppLauncher(launch_args)
app = launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from ry_unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry
from ry_unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import ry_go2_biped_walk_v20 as v20_task
import ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk_v2 as shared_v2


def main():
    task = "RY-Unitree-Go2-Biped-Gait-v20"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.max_iterations = args.max_iterations
    assert not agent_cfg.empirical_normalization, "Actor-only warm start needs matching observation normalization."
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs/rsl_rl") / agent_cfg.experiment_name / f"{timestamp}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    print("V20_LOG_DIR", log_dir.resolve(), flush=True)
    print("V20_ENV_BUILD_START", flush=True)
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    print("V20_ENV_BUILD_COMPLETE", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=args.device)
    policy = getattr(runner.alg, "actor_critic", None)
    if policy is None:
        policy = getattr(runner.alg, "policy", None)
    if policy is None or not hasattr(policy, "actor"):
        raise RuntimeError("Unsupported RSL-RL actor layout; refusing an ambiguous warm start.")
    checkpoint = torch.load(args.warm_start_actor, map_location=args.device, weights_only=False)
    actor = {k[len("actor."):]: v for k, v in checkpoint["model_state_dict"].items() if k.startswith("actor.")}
    if not actor:
        raise RuntimeError("Checkpoint has no actor.* weights.")
    target = policy.actor.state_dict()
    if set(actor) != set(target):
        raise RuntimeError(f"Warm-start actor keys differ: source-only={set(actor) - set(target)}, target-only={set(target) - set(actor)}")
    expanded_columns = []
    for name, source in actor.items():
        destination = target[name]
        if source.shape != destination.shape:
            if (name == "0.weight" and source.ndim == 2 and destination.ndim == 2
                    and source.shape == (destination.shape[0], 34)
                    and destination.shape == (source.shape[0], 38)):
                value = destination.clone()
                value[:, :34].copy_(source)
                value[:, 34:].zero_()
                source = value
                expanded_columns.append(name)
            else:
                raise RuntimeError(f"Warm-start shape mismatch for {name}: {tuple(source.shape)} vs {tuple(destination.shape)}")
        destination.copy_(source)
    if expanded_columns not in ([], ["0.weight"]):
        raise RuntimeError(f"Expected exactly one 34-to-38 input expansion, got {expanded_columns}")
    policy.actor.load_state_dict(target, strict=True)
    print("ACTOR_WARM_START_EXPANDED", expanded_columns, "input_dim", target["0.weight"].shape[1], flush=True)
    # The critic and optimizer remain fresh; std comes from the v4 runner config.
    print("ACTOR_WARM_START_COMPLETE", args.warm_start_actor, flush=True)
    params = log_dir / "params"
    params.mkdir(exist_ok=True)
    shutil.copy2(__file__, params / "train_go2_biped_v20.py")
    dump_yaml(str(params / "env.yaml"), cfg)
    dump_yaml(str(params / "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, str(log_dir))
    source = Path(v20_task.__file__).parent
    shutil.copytree(source, params / "v20_sources", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(Path(shared_v2.__file__).parent, params / "v2_sources", ignore=shutil.ignore_patterns("__pycache__"))
    base = Path(shared_v2.__file__).parent.parent / "go2_biped_walk"
    shutil.copytree(base, params / "base_sources", ignore=shutil.ignore_patterns("__pycache__"))
    manifest = {str(p.relative_to(params)): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in [params / "v20_sources", params / "v2_sources", params / "base_sources"]
                for p in folder.rglob("*") if p.is_file()}
    (params / "provenance.json").write_text(json.dumps({
        "task": task, "warm_start_actor": str(args.warm_start_actor.resolve()),
        "warm_start_sha256": hashlib.sha256(args.warm_start_actor.read_bytes()).hexdigest(),
        "fresh_critic": True, "fresh_optimizer": True, "curriculum_control_step": 0,
        "actor_input_expansion": {"source_dim": int(actor["0.weight"].shape[1]), "target_dim": int(target["0.weight"].shape[1]), "new_columns_zeroed": list(range(int(actor["0.weight"].shape[1]), int(target["0.weight"].shape[1])))},
        "seed": args.seed, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "device": args.device, "source_sha256": manifest,
    }, indent=2))
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    # Reuse the original framework exporters; empirical normalization is disabled above.
    export_dir = str(log_dir / "exported")
    export_policy_as_jit(policy, normalizer=None, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy, normalizer=None, path=export_dir, filename="policy.onnx")
    print("V20_POLICY_EXPORTED", export_dir, flush=True)
    env.close()
    print("V20_TRAINING_COMPLETE", flush=True)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    app.close()
