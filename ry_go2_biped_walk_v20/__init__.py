"""Go2 hind-leg walking with world-axis progress and explicit alternating gait."""
import gymnasium as gym

gym.register(
    id="RY-Unitree-Go2-Biped-Gait-v20",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gait_env_cfg:Go2BipedGaitV20EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gait_env_cfg:Go2BipedGaitV20PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.gait_env_cfg:Go2BipedGaitV20RunnerCfg",
    },
)
