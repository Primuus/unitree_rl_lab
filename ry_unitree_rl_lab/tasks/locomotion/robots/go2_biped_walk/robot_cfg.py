"""Go2 with six hind-leg actuators; original foot shape and inertias retained."""
import math
from pathlib import Path

from isaaclab.assets import ArticulationCfg
from ry_unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg
from ry_unitree_rl_lab.assets.robots import unitree_actuators

RESOURCE_PATH = Path(__file__).parent / "resources" / "go2_biped_walk.urdf"
HIND_JOINTS = [
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
HIND_FEET = ["RL_foot", "RR_foot"]
CALF_ANGLE = -1.3
THIGH_ANGLE = math.pi / 2 - CALF_ANGLE / 2
STANCE_HEIGHT = 0.1934 + 2 * 0.213 * math.cos(CALF_ANGLE / 2) + 0.022

GO2_BIPED_WALK_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(asset_path=str(RESOURCE_PATH)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, STANCE_HEIGHT),
        rot=(math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0),
        joint_pos={
            "R[LR]_hip_joint": 0.0,
            "R[LR]_thigh_joint": THIGH_ANGLE,
            "R[LR]_calf_joint": CALF_ANGLE,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hind_legs": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=["R[LR]_.*_joint"],
            stiffness=60.0, damping=2.5, friction=0.01,
        )
    },
    joint_sdk_names=HIND_JOINTS,
    soft_joint_pos_limit_factor=0.95,
)
