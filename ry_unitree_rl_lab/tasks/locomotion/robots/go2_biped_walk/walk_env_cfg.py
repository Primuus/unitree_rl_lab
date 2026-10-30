"""A separate reference-video-oriented task; independent of legacy biped configs."""
import math
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm, EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from ry_unitree_rl_lab.tasks.locomotion import mdp
from . import walk_mdp as wm
from .robot_cfg import GO2_BIPED_WALK_CFG, HIND_JOINTS, HIND_FEET

GAIT_PERIOD = 1.0
SCANNER = SceneEntityCfg("height_scanner")
REAR_BODIES = SceneEntityCfg("robot", body_names=HIND_FEET, preserve_order=True)
REAR_CONTACTS = SceneEntityCfg("contact_forces", body_names=HIND_FEET, preserve_order=True)

EASY_TERRAINS = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=10.0, num_rows=4, num_cols=10,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    difficulty_range=(0.0, 0.5), curriculum=True, use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),
        "small_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.005, 0.02), noise_step=0.005, border_width=0.5,
        ),
        "small_boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.01, 0.035), platform_width=2.0,
        ),
    },
)

@configclass
class WalkSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator",
        terrain_generator=EASY_TERRAINS, max_init_terrain_level=0, collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.43, 0.46), roughness=1.0),
        debug_vis=False,
    )
    robot: ArticulationCfg = GO2_BIPED_WALK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.0, 0.8)),
        mesh_prim_paths=["/World/ground"], debug_vis=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=1600.0),
    )

@configclass
class WalkCommandsCfg:
    base_velocity = wm.UprightVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.1, heading_command=False, debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0),
        ),
    )

@configclass
class WalkActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=HIND_JOINTS, preserve_order=True,
        scale=0.4, use_default_offset=True, clip={"R[LR]_hip_joint": (-0.65, 0.65), "R[LR]_thigh_joint": (1.35, 2.95), "R[LR]_calf_joint": (-2.5, -0.9)},
    )

@configclass
class WalkObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        linear_velocity = ObsTerm(func=wm.heading_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        angular_velocity = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = ObsTerm(func=wm.gait_clock, params={"period": GAIT_PERIOD})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)
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
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SCANNER}, clip=(-1.0, 1.0))
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class WalkRewardsCfg:
    upright = RewTerm(func=wm.upright_score, weight=3.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    track_linear = RewTerm(
        func=wm.track_linear, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25, "sensor_cfg": SCANNER},
    )
    track_yaw = RewTerm(
        func=wm.track_yaw, weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5, "sensor_cfg": SCANNER},
    )
    height = RewTerm(
        func=wm.height_band_l2, weight=-20.0,
        params={"sensor_cfg": SCANNER, "low": 0.48, "high": 0.59},
    )
    vertical_velocity = RewTerm(func=wm.vertical_velocity_l2, weight=-2.5)
    tilt_velocity = RewTerm(func=wm.horizontal_angular_velocity_l2, weight=-0.05)
    phase_contacts = RewTerm(
        func=wm.phase_contact_reward, weight=1.5,
        params={"sensor_cfg": REAR_CONTACTS, "terrain_sensor_cfg": SCANNER, "period": GAIT_PERIOD},
    )
    swing_clearance = RewTerm(
        func=wm.swing_clearance_reward, weight=1.5,
        params={
            "asset_cfg": REAR_BODIES, "sensor_cfg": REAR_CONTACTS,
            "terrain_sensor_cfg": SCANNER, "period": GAIT_PERIOD, "clearance": 0.06,
        },
    )
    feet_air_time = RewTerm(
        func=wm.air_time_reward, weight=2.5,
        params={"sensor_cfg": REAR_CONTACTS, "terrain_sensor_cfg": SCANNER, "threshold": 0.3},
    )
    flight = RewTerm(func=wm.flight_penalty, weight=-1.0, params={"sensor_cfg": REAR_CONTACTS})
    foot_slip = RewTerm(
        func=wm.feet_slide_l2, weight=-0.5, params={"asset_cfg": REAR_BODIES, "sensor_cfg": REAR_CONTACTS},
    )
    crossing = RewTerm(func=wm.feet_crossing_penalty, weight=-10.0, params={"asset_cfg": REAR_BODIES})
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-3.0)
    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="R[LR]_hip_joint")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)

@configclass
class WalkEventsCfg:
    reset_root = EventTerm(func=wm.reset_upright_root, mode="reset")
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.03, 0.03), "velocity_range": (-0.05, 0.05)},
    )
    friction = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.1), "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.0), "num_buckets": 32,
        },
    )

@configclass
class WalkTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=wm.bad_upright_orientation, params={"limit_angle": 0.95})
    low_base = DoneTerm(func=wm.too_low, params={"sensor_cfg": SCANNER, "min_height": 0.35})
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "R[LR]_(hip|thigh|calf)"]),
            "threshold": 5.0,
        },
    )

@configclass
class WalkCurriculumCfg:
    commands = CurrTerm(func=wm.command_curriculum, params={"warmup_steps": 12000, "ramp_steps": 48000})
    terrain = CurrTerm(func=wm.terrain_curriculum)

@configclass
class Go2BipedWalkEnvCfg(ManagerBasedRLEnvCfg):
    scene: WalkSceneCfg = WalkSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: WalkObservationsCfg = WalkObservationsCfg()
    actions: WalkActionsCfg = WalkActionsCfg()
    commands: WalkCommandsCfg = WalkCommandsCfg()
    rewards: WalkRewardsCfg = WalkRewardsCfg()
    events: WalkEventsCfg = WalkEventsCfg()
    terminations: WalkTerminationsCfg = WalkTerminationsCfg()
    curriculum: WalkCurriculumCfg = WalkCurriculumCfg()
    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.step_dt if hasattr(self, "step_dt") else self.decimation*self.sim.dt
        self.viewer.eye = (2.5, -2.5, 1.8)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.viewer.resolution = (960, 720)

@configclass
class Go2BipedWalkPlayEnvCfg(Go2BipedWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # Local mesh avoids waiting for the default remote Nucleus ground USD.
        self.scene.terrain.terrain_type = "generator"
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
