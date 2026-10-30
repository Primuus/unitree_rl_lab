"""Contract checks independent of whether the trained policy balances successfully."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from deploy import Simulation
from model import load_model
from observation import build_observation
from policy import Policy

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/"config.json").read_text())


class DeploymentTests(unittest.TestCase):
    def test_observation_frames_and_joint_orders(self):
        # Upright local X points up, local -Z points world +Y.
        rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
        offsets = np.array(CFG["action_offset"])
        q = offsets+np.arange(12)/10
        obs = build_observation(rotation, np.array([0., 1., 0.]), np.array([4., 8., 12.]),
                                q, np.arange(12)*20., np.arange(12),
                                np.array([.02, .05]), np.array([.4, .6])*16.087*9.81, 1., CFG)
        np.testing.assert_allclose(obs[:3], [1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(obs[3:9], [1, 2, 3, -1, 0, 0])
        np.testing.assert_array_equal(obs[9:14], np.zeros(5))
        order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        np.testing.assert_allclose(obs[14:26], np.array(order)/10)
        np.testing.assert_allclose(obs[26:38], order)
        np.testing.assert_allclose(obs[38:50], np.arange(12))
        np.testing.assert_allclose(obs[50:], [1, 0, 0, .3, .4, .6], atol=1e-7)

    def test_contact_force_world_sign_and_weight(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/>
        <worldbody><geom type="plane" size="1 1 .1"/>
        <body pos="0 0 .1"><freejoint/><geom name="foot" type="sphere" size=".1" mass="1"/>
        </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        for _ in range(500):
            mujoco.mj_step(model, data)
        state = SimpleNamespace(model=model, data=data, foot_geoms={model.geom("foot").id: 2},
                                contact_force=np.zeros(6))
        forces = Simulation.forces(state)
        np.testing.assert_allclose(forces[2], [0, 0, 9.81], atol=.01)

    def test_source_mass_limits_and_fixed_body_fusion(self):
        model = load_model(CFG, ROOT)
        self.assertEqual(model.nu, 12)
        self.assertEqual(model.nq, 19)
        self.assertEqual(model.nv, 18)
        self.assertAlmostEqual(model.body_mass.sum(), 16.087)
        # Root includes head + four fixed hip rotors, as merge_fixed_joints in Isaac.
        self.assertAlmostEqual(model.body("base").mass[0], 6.921+.002+4*.089)
        source = ET.parse(ROOT/"resources/go2_description.urdf").getroot()
        for joint in source.findall("joint"):
            if joint.get("type") != "revolute":
                continue
            name, limit = joint.get("name"), joint.find("limit")
            np.testing.assert_allclose(model.joint(name).range,
                                       [float(limit.get("lower")), float(limit.get("upper"))])
            np.testing.assert_allclose(model.actuator(name).ctrlrange,
                                       [-float(limit.get("effort")), float(limit.get("effort"))])

    def test_exported_policy_backends(self):
        pt = Policy(ROOT/"models/stage1_balance/exported/policy.pt")
        onnx = Policy(ROOT/"models/stage1_balance/exported/policy.onnx")
        inputs = np.random.default_rng(1).normal(size=(16, 56)).astype(np.float32)
        for obs in inputs:
            np.testing.assert_allclose(pt(obs), onnx(obs), rtol=1e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
