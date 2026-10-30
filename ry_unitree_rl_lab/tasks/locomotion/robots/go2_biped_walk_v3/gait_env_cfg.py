"""V3 preserves v2 rewards and adds a balance-first velocity curriculum."""
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk_v2.gait_env_cfg import (
    GaitCurriculumCfg,
    Go2BipedGaitEnvCfg,
    Go2BipedGaitPlayEnvCfg,
    Go2BipedGaitRunnerCfg,
)
from . import gait_mdp


@configclass
class BalanceFirstCurriculumCfg(GaitCurriculumCfg):
    commands = CurrTerm(func=gait_mdp.command_curriculum, params={"balance_steps": 3000, "ramp_steps": 10000})


@configclass
class Go2BipedGaitV3EnvCfg(Go2BipedGaitEnvCfg):
    curriculum: BalanceFirstCurriculumCfg = BalanceFirstCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0


@configclass
class Go2BipedGaitV3PlayEnvCfg(Go2BipedGaitPlayEnvCfg):
    pass


@configclass
class Go2BipedGaitV3RunnerCfg(Go2BipedGaitRunnerCfg):
    experiment_name = "unitree_go2_biped_walk_v3"

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.25
