"""V20 combines frame alignment and gait target gradients based on v3 evidence."""
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk_v2.gait_env_cfg import (
    GaitRewardsCfg,
    Go2BipedGaitEnvCfg,
    Go2BipedGaitPlayEnvCfg,
    Go2BipedGaitRunnerCfg,
    REAR_BODIES,
    REAR_CONTACTS,
    ROBOT_WEIGHT,
    SCANNER,
)
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk.walk_env_cfg import GAIT_PERIOD, WalkObservationsCfg
from . import gait_mdp as gm
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk_v2.gait_env_cfg import GaitCurriculumCfg
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk_v3.gait_mdp import command_curriculum


@configclass
class V20RewardsCfg(GaitRewardsCfg):
    track_linear = RewTerm(func=gm.world_velocity_tracking, weight=2.5, params={"terrain_sensor_cfg": SCANNER})
    track_yaw = RewTerm(func=gm.strict_yaw_tracking, weight=-0.5, params={"terrain_sensor_cfg": SCANNER})
    progress = RewTerm(func=gm.world_forward_progress, weight=2.0, params={"terrain_sensor_cfg": SCANNER})
    phase_load = RewTerm(func=gm.phase_load, weight=-2.5, params={"sensor_cfg": REAR_CONTACTS, "asset_cfg": REAR_BODIES, "terrain_sensor_cfg": SCANNER, "robot_weight": ROBOT_WEIGHT, "period": 1.0})
    swing_clearance = RewTerm(func=gm.swing_clearance, weight=5.0, params={"asset_cfg": REAR_BODIES, "sensor_cfg": REAR_CONTACTS, "terrain_sensor_cfg": SCANNER, "robot_weight": ROBOT_WEIGHT, "period": 1.0, "clearance": 0.06, "std": 0.04})
    swing_forward_position = RewTerm(func=gm.swing_forward_position, weight=3.0,
        params={"asset_cfg": REAR_BODIES, "sensor_cfg": REAR_CONTACTS, "terrain_sensor_cfg": SCANNER,
                "robot_weight": ROBOT_WEIGHT, "period": GAIT_PERIOD, "stride": 0.06})
    swing_forward_speed = RewTerm(func=gm.swing_forward_speed, weight=0.4,
        params={"asset_cfg": REAR_BODIES, "sensor_cfg": REAR_CONTACTS, "terrain_sensor_cfg": SCANNER,
                "robot_weight": ROBOT_WEIGHT, "period": GAIT_PERIOD})
    heading_alignment = RewTerm(func=gm.heading_alignment, weight=1.5, params={"terrain_sensor_cfg": SCANNER})
    qualified_touchdown = RewTerm(func=gm.qualified_touchdown_reward, weight=12.0,
        params={"asset_cfg": REAR_BODIES, "sensor_cfg": REAR_CONTACTS,
                "terrain_sensor_cfg": SCANNER, "period": GAIT_PERIOD,
                "robot_weight": ROBOT_WEIGHT, "min_air": 0.06,
                "min_lift": 0.02, "min_support": 0.25})



@configclass
class V20ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        linear_velocity = ObsTerm(func=gm.wm.heading_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        angular_velocity = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = ObsTerm(func=gm.wm.gait_clock, params={"period": GAIT_PERIOD})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)
        heading = ObsTerm(func=gm.heading_features)
        foot_height = ObsTerm(func=gm.rear_foot_height, params={"asset_cfg": REAR_BODIES})
        foot_load = ObsTerm(func=gm.rear_foot_load, params={"sensor_cfg": REAR_CONTACTS, "robot_weight": ROBOT_WEIGHT})
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        linear_velocity = ObsTerm(func=gm.wm.heading_lin_vel)
        angular_velocity = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = ObsTerm(func=gm.wm.gait_clock, params={"period": GAIT_PERIOD})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SCANNER}, clip=(-1.0, 1.0))
        heading = ObsTerm(func=gm.heading_features)
        foot_height = ObsTerm(func=gm.rear_foot_height, params={"asset_cfg": REAR_BODIES})
        foot_load = ObsTerm(func=gm.rear_foot_load, params={"sensor_cfg": REAR_CONTACTS, "robot_weight": ROBOT_WEIGHT})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class V20BalanceCurriculumCfg(GaitCurriculumCfg):
    commands = CurrTerm(func=command_curriculum, params={"balance_steps": 3000, "ramp_steps": 10000})


@configclass
class Go2BipedGaitV20EnvCfg(Go2BipedGaitEnvCfg):
    observations: V20ObservationsCfg = V20ObservationsCfg()
    curriculum: V20BalanceCurriculumCfg = V20BalanceCurriculumCfg()
    rewards: V20RewardsCfg = V20RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The v7 actor already balances while moving; expose the gait objective immediately.
        self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.12)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.curriculum.commands = None
        self.actions.joint_pos.scale = 0.4


@configclass
class Go2BipedGaitV20PlayEnvCfg(Go2BipedGaitPlayEnvCfg):
    observations: V20ObservationsCfg = V20ObservationsCfg()
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = 0.4


@configclass
class Go2BipedGaitV20RunnerCfg(Go2BipedGaitRunnerCfg):
    experiment_name = "ry_unitree_go2_biped_walk_v20"
    save_interval = 20

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.08
        self.algorithm.learning_rate = 3e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, use_mirror_loss=False, mirror_loss_coeff=0.0,
            data_augmentation_func="ry_go2_biped_walk_v20.symmetry:mirror_data")
