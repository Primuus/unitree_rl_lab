"""Go2 hind-leg walking with a balance-first command curriculum."""
import gymnasium as gym

gym.register(
    id="Unitree-Go2-Biped-Gait-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gait_env_cfg:Go2BipedGaitV3EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gait_env_cfg:Go2BipedGaitV3PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.gait_env_cfg:Go2BipedGaitV3RunnerCfg",
    },
)
