"""Upright Go2 hind-leg walking task with folded, fixed front legs."""
import gymnasium as gym

gym.register(
    id="Unitree-Go2-Biped-Walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:Go2BipedWalkEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.walk_env_cfg:Go2BipedWalkPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.walk_ppo_cfg:Go2BipedWalkPPORunnerCfg",
    },
)
