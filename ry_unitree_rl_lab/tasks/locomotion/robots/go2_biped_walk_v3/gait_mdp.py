"""Balance-first velocity curriculum for v3."""
import torch


def command_curriculum(env, env_ids, balance_steps=3000, ramp_steps=10000):
    progress = min(max((env.common_step_counter - balance_steps) / max(ramp_steps, 1), 0.0), 1.0)
    ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    ranges.lin_vel_x = (0.10 * progress, 0.16 * progress)
    ranges.lin_vel_y = (0.0, 0.0)
    ranges.ang_vel_z = (0.0, 0.0)
    return torch.tensor(progress, device=env.device)
