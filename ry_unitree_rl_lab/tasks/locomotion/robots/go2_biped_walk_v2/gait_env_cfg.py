"""Walking refinement from a balance policy, with a separately registered task."""
import xml.etree.ElementTree as ET
import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk import walk_mdp as wm
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk.robot_cfg import RESOURCE_PATH
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk.walk_env_cfg import (
    Go2BipedWalkEnvCfg, WalkRewardsCfg, WalkCurriculumCfg,
    SCANNER, REAR_BODIES, REAR_CONTACTS, GAIT_PERIOD,
)
from ry_unitree_rl_lab.tasks.locomotion.robots.go2_biped_walk.walk_ppo_cfg import Go2BipedWalkPPORunnerCfg
from . import gait_mdp as gm

ROBOT_WEIGHT = sum(float(x.get("value")) for x in ET.parse(RESOURCE_PATH).findall("link/inertial/mass")) * 9.81


@configclass
class GaitRewardsCfg(WalkRewardsCfg):
    upright = RewTerm(func=wm.upright_score, weight=2.0, params={"std": 0.55})
    alive = RewTerm(func=wm.upright_score, weight=0.3, params={"std": 0.7})
    track_linear = RewTerm(func=gm.tight_velocity_tracking, weight=2.5,
                          params={"terrain_sensor_cfg": SCANNER})
    progress = RewTerm(func=gm.forward_progress, weight=1.0,
                       params={"terrain_sensor_cfg": SCANNER})
    phase_contacts = None
    phase_load = RewTerm(func=gm.phase_load_error, weight=-2.5,
        params={"sensor_cfg": REAR_CONTACTS, "asset_cfg": REAR_BODIES,
                "terrain_sensor_cfg": SCANNER, "robot_weight": ROBOT_WEIGHT,
                "period": GAIT_PERIOD})
    swing_clearance = RewTerm(func=gm.dense_swing_clearance, weight=3.0,
        params={"asset_cfg": REAR_BODIES, "sensor_cfg": REAR_CONTACTS,
                "terrain_sensor_cfg": SCANNER, "robot_weight": ROBOT_WEIGHT,
                "period": GAIT_PERIOD, "clearance": 0.05, "std": 0.025})
    feet_air_time = RewTerm(func=wm.air_time_reward, weight=2.0,
        params={"sensor_cfg": REAR_CONTACTS, "terrain_sensor_cfg": SCANNER, "threshold": 0.3})
    flight = RewTerm(func=wm.flight_penalty, weight=-2.0, params={"sensor_cfg": REAR_CONTACTS})


@configclass
class GaitCurriculumCfg(WalkCurriculumCfg):
    commands = CurrTerm(func=gm.command_curriculum, params={"ramp_steps": 24000})


@configclass
class Go2BipedGaitEnvCfg(Go2BipedWalkEnvCfg):
    rewards: GaitRewardsCfg = GaitRewardsCfg()
    curriculum: GaitCurriculumCfg = GaitCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.16)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=5.0, num_rows=4, num_cols=10,
            horizontal_scale=0.1, vertical_scale=0.005,
            difficulty_range=(0.0, 0.5), curriculum=True, use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.9),
                "small_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.1, noise_range=(0.005, 0.015),
                    noise_step=0.005, border_width=0.5),
            },
        )


@configclass
class Go2BipedGaitPlayEnvCfg(Go2BipedGaitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=2.0, num_rows=1, num_cols=1,
            curriculum=False, use_cache=False,
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
        )
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.commands = None
        self.curriculum.terrain = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.events.friction = None
        self.observations.policy.enable_corruption = False


@configclass
class Go2BipedGaitRunnerCfg(Go2BipedWalkPPORunnerCfg):
    experiment_name = "unitree_go2_biped_walk_v2"
    max_iterations = 4000
    save_interval = 100

    def __post_init__(self):
        self.policy.init_noise_std = 0.5
        self.algorithm.entropy_coef = 0.015
        self.algorithm.learning_rate = 2e-4
