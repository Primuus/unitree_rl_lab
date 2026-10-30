"""Current task: full Go2 with the reference handstand reward set."""
import isaaclab.terrains as terrain_gen
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg)
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk.walk_env_cfg import (
    Go2BipedWalkEnvCfg, WalkSceneCfg, SCANNER, REAR_BODIES, REAR_CONTACTS)
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk import walk_mdp as wm
from . import gait_mdp as gm
from .reference_config import REFERENCE
from .robot_cfg import ROBOT_CFG, ROBOT_WEIGHT, JOINT_NAMES

GAIT_PERIOD = REFERENCE["reward_parameters"]["cycle_time"]


@configclass
class ReferenceSceneCfg(WalkSceneCfg):
    robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ReferenceActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=JOINT_NAMES, preserve_order=True,
        scale=REFERENCE["control"]["action_scale"], use_default_offset=False,
        offset=REFERENCE["default_joint_angles"])


@configclass
class V20ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        linear_velocity = ObsTerm(func=wm.heading_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        angular_velocity = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = ObsTerm(func=wm.gait_clock, params={"period": GAIT_PERIOD})
        joint_pos = ObsTerm(func=gm.joint_position_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
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
        linear_velocity = ObsTerm(func=wm.heading_lin_vel)
        angular_velocity = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = ObsTerm(func=wm.gait_clock, params={"period": GAIT_PERIOD})
        joint_pos = ObsTerm(func=gm.joint_position_rel)
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


# Every nonzero upstream scale is registered; zero/absent terms stay disabled.
ReferenceRewardsCfg = configclass(type("ReferenceRewardsCfg", (), {
    name: RewTerm(func=gm.reference_reward, weight=weight, params={"name": name})
    for name, weight in REFERENCE["reward_scales"].items() if weight != 0
}))


@configclass
class ReferenceTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(func=mdp.illegal_contact, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0})


def configure_stage(cfg, stage):
    if stage not in ("balance", "walk"):
        raise ValueError(f"Unknown stage: {stage}")
    cfg.stage = stage
    cfg.rewards = ReferenceRewardsCfg()
    if stage == "balance":
        cfg.rewards.contact = None
        cfg.rewards.feet_clearance = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0) if stage == "balance" else (0.2, 0.2)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 1.0 if stage == "balance" else 0.0


@configclass
class Go2BipedGaitV20EnvCfg(Go2BipedWalkEnvCfg):
    stage: str = "balance"
    scene: ReferenceSceneCfg = ReferenceSceneCfg(num_envs=4096, env_spacing=3.0)
    actions: ReferenceActionsCfg = ReferenceActionsCfg()
    observations: V20ObservationsCfg = V20ObservationsCfg()
    rewards: ReferenceRewardsCfg = ReferenceRewardsCfg()
    terminations: ReferenceTerminationsCfg = ReferenceTerminationsCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=5.0, num_rows=1, num_cols=1,
            curriculum=False, use_cache=False,
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)})
        self.scene.terrain.max_init_terrain_level = 0
        self.events.friction.params.update(
            static_friction_range=(0.2, 1.2), dynamic_friction_range=(0.2, 1.2),
            restitution_range=(0.0, 0.3))
        self.events.reference_history = EventTerm(func=gm.reset_reference_history, mode="reset")
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        configure_stage(self, self.stage)


@configclass
class Go2BipedGaitV20PlayEnvCfg(Go2BipedGaitV20EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.friction = None
        self.observations.policy.enable_corruption = False


@configclass
class Go2BipedGaitV20RunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "ry_unitree_go2_handstand"
    empirical_normalization = False
    num_steps_per_env = REFERENCE["ppo_runner"]["num_steps_per_env"]
    max_iterations = REFERENCE["ppo_runner"]["max_iterations"]
    save_interval = REFERENCE["ppo_runner"]["save_interval"]
    clip_actions = 100.0
    policy = RslRlPpoActorCriticCfg(**REFERENCE["ppo_policy"])
    algorithm = RslRlPpoAlgorithmCfg(**{
        key: value for key, value in REFERENCE["ppo_algorithm"].items()
        if key not in ("sym_loss", "obs_permutation", "act_permutation", "frame_stack", "sym_coef")},
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False, use_mirror_loss=True, mirror_loss_coeff=1.0,
            data_augmentation_func="ry_go2_biped_walk_v20.symmetry:mirror_data"))
