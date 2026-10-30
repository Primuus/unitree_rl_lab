"""Complete Go2; PD/default angles come from the reference handstand task."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from ry_unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
from .reference_config import REFERENCE

JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
               for joint in ("hip", "thigh", "calf")]
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
URDF_PATH = Path(UNITREE_GO2_CFG.spawn.asset_path)
_tree = ET.parse(URDF_PATH).getroot()
_limits = {j.get("name"): j.find("limit") for j in _tree.findall("joint")
           if j.get("type") == "revolute"}
ROBOT_WEIGHT = sum(float(m.get("value")) for m in _tree.findall("link/inertial/mass")) * 9.81

ROBOT_CFG = UNITREE_GO2_CFG.copy()
ROBOT_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, REFERENCE["reward_parameters"]["base_height_target"]),
    rot=(math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0),
    joint_pos=REFERENCE["desired_joint_angles"],
    joint_vel={".*": 0.0},
)
ROBOT_CFG.actuators = {
    "legs": IdealPDActuatorCfg(
        joint_names_expr=JOINT_NAMES,
        stiffness=REFERENCE["control"]["stiffness"]["joint"],
        damping=REFERENCE["control"]["damping"]["joint"],
        effort_limit={name: float(_limits[name].get("effort")) for name in JOINT_NAMES},
        velocity_limit={name: float(_limits[name].get("velocity")) for name in JOINT_NAMES},
    )
}
ROBOT_CFG.soft_joint_pos_limit_factor = REFERENCE["reward_parameters"]["soft_dof_pos_limit"]
