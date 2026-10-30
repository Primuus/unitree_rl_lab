"""Versioned Go2 hind-leg walking task with dense load-transfer rewards."""
import gymnasium as gym

gym.register(
    id="Unitree-Go2-Biped-Gait-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gait_env_cfg:Go2BipedGaitEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gait_env_cfg:Go2BipedGaitPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.gait_env_cfg:Go2BipedGaitRunnerCfg",
    },
)
