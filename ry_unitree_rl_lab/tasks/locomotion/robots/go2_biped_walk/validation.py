"""Math and asset checks, imported by the headless evaluation script."""
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from . import walk_mdp as wm

def validate_reference_frames():
    n = 2
    zeros = torch.zeros(n)
    upright = torch.tensor([[math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]]).repeat(n, 1)
    quats = quat_mul(quat_from_euler_xyz(zeros, zeros, torch.tensor([0.0, math.pi/2])), upright)
    data = SimpleNamespace(
        root_quat_w=quats,
        root_lin_vel_w=torch.tensor([[0.2, 0.0, 0.1], [0.0, 0.2, 0.1]]),
        root_ang_vel_w=torch.zeros(n, 3),
        root_pos_w=torch.tensor([[0., 0., 0.55], [0., 0., 0.55]]),
        projected_gravity_b=torch.tensor([[-1., 0., 0.], [-1., 0., 0.]]),
        body_pos_w=torch.tensor([[[0., 0., 0.022], [0., 0., 0.022]]]).repeat(n, 1, 1),
    )
    class Scene(dict):
        pass
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.env_origins = torch.zeros(n, 3)
    contact = SimpleNamespace(current_contact_time=torch.tensor([[0., 0.1], [0., 0.1]]))
    scene.sensors = {
        "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=torch.zeros(n, 3, 3))),
        "contact_forces": SimpleNamespace(data=contact),
    }
    command = torch.tensor([[0.2, 0., 0.], [0.2, 0., 0.]])
    env = SimpleNamespace(
        scene=scene, num_envs=n, device="cpu", step_dt=0.02,
        episode_length_buf=torch.full((n,), 35),
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    assert torch.allclose(wm.heading_lin_vel(env)[:, 0], torch.full((n,), 0.2), atol=1e-6)
    assert torch.allclose(wm.vertical_velocity_l2(env), torch.full((n,), 0.01), atol=1e-6)
    assert torch.allclose(wm.upright_score(env), torch.ones(n))
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
    feet_cfg = SimpleNamespace(name="robot", body_ids=[0, 1])
    scanner_cfg = SceneEntityCfg("height_scanner")
    # Mid-swing of RL: RR supports. A simultaneous jump may never get lift reward.
    data.body_pos_w[:, 0, 2] += 0.06 * math.sin(math.pi * 0.25)
    lift = wm.swing_clearance_reward(env, feet_cfg, sensor_cfg, scanner_cfg)
    assert bool((lift > 0.99).all()), lift
    contact.current_contact_time.zero_()
    assert bool((wm.swing_clearance_reward(env, feet_cfg, sensor_cfg, scanner_cfg) == 0).all())
    # Each environment is gated separately; a fallen neighbor cannot gate the batch.
    data.projected_gravity_b[1] = torch.tensor([0., 0., -1.])
    gate = wm.posture_gate(env, scanner_cfg)
    assert gate[0] > 0.99 and gate[1] < 0.02
    command.zero_()
    assert bool((wm.gait_clock(env) == 0).all())
    root = ET.parse(Path(__file__).parent / "resources/go2_biped_walk.urdf").getroot()
    source = ET.parse("/workspace/unitree_ros/robots/go2_description/urdf/go2_description.urdf").getroot()
    movable = [j for j in root.findall("joint") if j.get("type") == "revolute"]
    assert len(movable) == 6 and all(j.get("name").startswith(("RL_", "RR_")) for j in movable)
    for side in ("RL", "RR"):
        a=root.find(f"link[@name='{side}_foot']/collision")
        b=source.find(f"link[@name='{side}_foot']/collision")
        assert ET.tostring(a) == ET.tostring(b)
    mass=lambda r:sum(float(m.get("value")) for m in r.findall("link/inertial/mass"))
    assert abs(mass(root)-mass(source)) < 1e-9
    print("REFERENCE_FRAME_ASSET_CHECKS_PASSED", flush=True)
