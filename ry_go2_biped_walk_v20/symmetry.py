"""Name-based 12-joint reflection, independent of simulator storage order."""
import torch

ACTION_NAMES = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
                for part in ("hip", "thigh", "calf")]
POLICY_NAMES = ["linear_velocity", "angular_velocity", "projected_gravity", "command",
                "gait_phase", "joint_pos", "joint_vel", "last_action", "heading",
                "foot_height", "foot_load"]
POLICY_DIMS = [3, 3, 3, 3, 2, 12, 12, 12, 2, 2, 2]


def transform_joints(x, names):
    pairs = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
    swapped = [pairs[name[:2]] + name[2:] for name in names]
    indices = [names.index(name) for name in swapped]
    signs = x.new_tensor([-1.0 if "_hip_" in name else 1.0 for name in names])
    return x[..., indices] * signs


def mirror_policy(obs, names, dims, joints):
    if names != POLICY_NAMES or dims != POLICY_DIMS or obs.shape[-1] != 56:
        raise ValueError(f"Unexpected 12-joint observation layout: {names}, {dims}, {obs.shape}")
    result = obs.clone()
    offset = 0
    for name, size in zip(names, dims):
        term = obs[..., offset:offset + size]
        if name in ("linear_velocity", "projected_gravity"):
            term = term * term.new_tensor([1., -1., 1.])
        elif name == "angular_velocity":
            term = term * term.new_tensor([-1., 1., -1.])
        elif name == "command":
            term = term * term.new_tensor([1., -1., -1.])
        elif name == "gait_phase":
            term = -term
        elif name in ("joint_pos", "joint_vel"):
            term = transform_joints(term, joints)
        elif name == "last_action":
            term = transform_joints(term, ACTION_NAMES)
        elif name == "heading":
            term = term * term.new_tensor([-1., 1.])
        elif name in ("foot_height", "foot_load"):
            term = term.flip(-1)
        result[..., offset:offset + size] = term
        offset += size
    return result


def mirror_critic(obs, names, dims, joints):
    if names[8] != "height_scan" or dims[8] != 99 or obs.shape[-1] != 155:
        raise ValueError(f"Unexpected critic layout: {names}, {dims}, {obs.shape}")
    policy = torch.cat((obs[..., :50], obs[..., 149:]), dim=-1)
    mirrored = mirror_policy(policy, names[:8] + names[9:], dims[:8] + dims[9:], joints)
    scan = obs[..., 50:149].reshape(*obs.shape[:-1], 9, 11).flip(-2).flatten(-2)
    return torch.cat((mirrored[..., :50], scan, mirrored[..., 50:]), dim=-1)


def mirror_data(env, obs=None, actions=None, obs_type="policy"):
    raw = env.unwrapped
    joints = list(raw.scene["robot"].joint_names)
    actual = list(raw.action_manager.get_term("joint_pos")._joint_names)
    if actual != ACTION_NAMES or set(joints) != set(ACTION_NAMES):
        raise ValueError(f"Unexpected joint mappings: {joints}, {actual}")
    aug_obs = None
    if obs is not None:
        names = raw.observation_manager.active_terms[obs_type]
        dims = [int(torch.tensor(d).prod()) for d in raw.observation_manager.group_obs_term_dim[obs_type]]
        transform = {"policy": mirror_policy, "critic": mirror_critic}[obs_type]
        aug_obs = torch.cat((obs, transform(obs, names, dims, joints)), dim=0)
    aug_actions = None if actions is None else torch.cat(
        (actions, transform_joints(actions, ACTION_NAMES)), dim=0)
    return aug_obs, aug_actions
