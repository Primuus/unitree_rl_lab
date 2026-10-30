"""Policy mirror regularization for the upright Go2. No critic augmentation."""
import math
import torch

ACTION_NAMES = ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
OBS_JOINT_NAMES = ["RL_hip_joint", "RR_hip_joint", "RL_thigh_joint",
                   "RR_thigh_joint", "RL_calf_joint", "RR_calf_joint"]

def joint_map(names):
    swapped = [n.replace("RL_", "TMP_").replace("RR_", "RL_").replace("TMP_", "RR_") for n in names]
    if set(swapped) != set(names):
        raise ValueError(f"Non-symmetric joint layout: {names}")
    indices = [names.index(n) for n in swapped]
    signs = [-1.0 if "_hip_" in n else 1.0 for n in names]
    return indices, signs

def transform_joints(x, names):
    indices, signs = joint_map(names)
    if x.shape[-1] != len(names):
        raise ValueError("Joint dimension mismatch")
    return x[..., indices] * x.new_tensor(signs)

def mirror_policy(obs, names, dims, joints):
    expected = ["linear_velocity", "angular_velocity", "projected_gravity", "command",
                "gait_phase", "joint_pos", "joint_vel", "last_action", "heading",
                "foot_height", "foot_load"]
    expected_dims = [3,3,3,3,2,6,6,6,2,2,2]
    if names != expected or dims != expected_dims or obs.shape[-1] != 38:
        raise ValueError(f"Unverified policy layout: {names}, {dims}, {obs.shape}")
    result = obs.clone()
    offset = 0
    for name, size in zip(names, dims):
        term = obs[..., offset:offset+size]
        if name in ("linear_velocity", "projected_gravity"):
            term = term * term.new_tensor([1.,-1.,1.])
        elif name == "angular_velocity":
            term = term * term.new_tensor([-1.,1.,-1.])
        elif name == "command":
            term = term * term.new_tensor([1.,-1.,-1.])
        elif name == "gait_phase":
            term = -term  # half-cycle shift exchanges left and right phases
        elif name in ("joint_pos", "joint_vel"):
            term = transform_joints(term, OBS_JOINT_NAMES)
        elif name == "last_action":
            term = transform_joints(term, ACTION_NAMES)
        elif name == "heading":
            term = term * term.new_tensor([-1.,1.])
        elif name in ("foot_height", "foot_load"):
            term = term.flip(-1)
        result[..., offset:offset+size] = term
        offset += size
    return result

def mirror_critic(obs, names, dims, joints):
    expected = ["linear_velocity", "angular_velocity", "projected_gravity", "command",
                "gait_phase", "joint_pos", "joint_vel", "last_action", "height_scan",
                "heading", "foot_height", "foot_load"]
    expected_dims = [3,3,3,3,2,6,6,6,99,2,2,2]
    if names != expected or dims != expected_dims or obs.shape[-1] != 137:
        raise ValueError(f"Unverified critic layout: {names}, {dims}, {obs.shape}")
    result = obs.clone()
    offset = 0
    for name, size in zip(names, dims):
        term = obs[..., offset:offset+size]
        if name in ("linear_velocity", "projected_gravity"):
            term = term * term.new_tensor([1., -1., 1.])
        elif name == "angular_velocity":
            term = term * term.new_tensor([-1., 1., -1.])
        elif name == "command":
            term = term * term.new_tensor([1., -1., -1.])
        elif name == "gait_phase":
            term = -term
        elif name in ("joint_pos", "joint_vel"):
            term = transform_joints(term, OBS_JOINT_NAMES)
        elif name == "last_action":
            term = transform_joints(term, ACTION_NAMES)
        elif name == "height_scan":
            term = term.reshape(term.shape[0], 9, 11).flip(-2).reshape(term.shape[0], 99)
        elif name == "heading":
            term = term * term.new_tensor([-1., 1.])
        elif name in ("foot_height", "foot_load"):
            term = term.flip(-1)
        result[..., offset:offset+size] = term
        offset += size
    return result

def mirror_data(env, obs=None, actions=None, obs_type="policy"):
    raw = env.unwrapped
    joints = list(raw.scene["robot"].joint_names)
    if joints != OBS_JOINT_NAMES:
        raise ValueError(f"Unexpected native joint observation order: {joints}")
    if list(raw.action_manager.get_term("joint_pos")._joint_names) != ACTION_NAMES:
        raise ValueError("Action joint order differs from verified layout")
    if obs_type == "policy":
        names = raw.observation_manager.active_terms["policy"]
        dims = [math.prod(d) for d in raw.observation_manager.group_obs_term_dim["policy"]]
        mirrored = mirror_policy(obs, names, dims, joints) if obs is not None else None
    elif obs_type == "critic":
        names = raw.observation_manager.active_terms["critic"]
        dims = [math.prod(d) for d in raw.observation_manager.group_obs_term_dim["critic"]]
        mirrored = mirror_critic(obs, names, dims, joints) if obs is not None else None
    else:
        raise ValueError(f"Unsupported observation group: {obs_type}")
    aug_obs = torch.cat((obs, mirrored), dim=0) if obs is not None else None
    aug_actions = torch.cat((actions, transform_joints(actions, ACTION_NAMES)), dim=0) if actions is not None else None
    return aug_obs, aug_actions
