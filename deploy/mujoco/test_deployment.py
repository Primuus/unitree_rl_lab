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
from ry_unitree_rl_lab.utils.phase_contract import read_gait_clock

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
                                np.array([.02, .05]), np.array([.4, .6])*16.087*9.81, 1.,
                                {**CFG, 'gait_clock_mode':'moving_only', 'command':[0.,0.,0.]})
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
        visuals = model.geom_group == 2
        self.assertTrue(visuals.any())
        self.assertTrue(np.all(model.geom_type[visuals] == mujoco.mjtGeom.mjGEOM_MESH))
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
            self.assertAlmostEqual(model.jnt_user[model.joint(name).id, 0],
                                   float(limit.get("velocity")))

    def test_first_step_velocity_limit_and_position_increment(self):
        sim = Simulation(CFG, ROOT/CFG["policy"])
        before = sim.data.qpos[sim.qids].copy()
        # Exercise projection independently of the current policy and timestep.
        sim.data.qvel[sim.vids] = 2 * sim.velocity_limits
        sim.step()
        self.assertGreater(sim.velocity_limit_applications, 0)
        self.assertTrue(np.all(np.abs(sim.data.qvel[sim.vids]) <= sim.velocity_limits))
        self.assertTrue(np.all(np.abs(sim.data.qpos[sim.qids]-before)
                               <= sim.velocity_limits*sim.model.opt.timestep+1e-12))

    def test_original_six_joint_model_and_observation_contract(self):
        cfg = dict(CFG)
        cfg.update(
            urdf="../../ry_unitree_rl_lab/tasks/locomotion/robots/go2_biped_walk/resources/go2_biped_walk.urdf",
            action_names=[f"{leg}_{joint}_joint" for leg in ("RL","RR") for joint in ("hip","thigh","calf")],
            observation_joint_names=[f"{leg}_{joint}_joint" for joint in ("hip","thigh","calf") for leg in ("RL","RR")],
            action_offset=[0.,2.2207963,-1.3,0.,2.2207963,-1.3],
            observation_terms=["linear_velocity","angular_velocity","projected_gravity","command","gait_phase",
                               "joint_pos","joint_vel","last_action","heading"],
            observation_dim=34, gait_clock_mode="moving_only", gait_period=1.0)
        model = load_model(cfg, ROOT)
        self.assertEqual(model.nu, 6)
        self.assertEqual(model.nq, 13)
        names = [model.joint(i).name for i in range(model.njnt)]
        self.assertFalse(any(n.startswith(("FL_","FR_")) for n in names))
        np.testing.assert_allclose(model.body_mass.sum(), CFG["robot_mass"])
        q = np.asarray(cfg["action_offset"])+np.arange(6)/10
        obs = build_observation(np.eye(3),np.zeros(3),np.zeros(3),q,np.arange(6)*20,
                                np.arange(6),np.array([0.,.03]),np.array([0.,100.]),0.,cfg)
        self.assertEqual(obs.shape, (34,))
        np.testing.assert_allclose(obs[14:20],np.array([0,3,1,4,2,5])/10)
        np.testing.assert_allclose(obs[20:26],[0,3,1,4,2,5])
        np.testing.assert_allclose(obs[26:32],np.arange(6))
        cfg["observation_terms"] += ["foot_height","foot_load"]
        cfg["observation_dim"] = 38
        extended = build_observation(np.eye(3),np.zeros(3),np.zeros(3),q,np.arange(6)*20,
                                    np.arange(6),np.array([0.,.03]),np.array([0.,100.]),0.,cfg)
        np.testing.assert_allclose(extended[:34],obs)
        np.testing.assert_allclose(extended[34:],[0.,.3,0.,100/(cfg["robot_mass"]*9.81)],atol=1e-7)

    def test_original_fixed_body_com_and_velocity_observation(self):
        cfg = json.loads((ROOT/"original_stand.json").read_text())
        sim = Simulation(cfg, ROOT/cfg["policy"])
        # This symmetric source must retain zero lateral COM after compiling.
        np.testing.assert_allclose(sim.data.subtree_com[sim.base, 1], 0., atol=1e-10)
        self.assertEqual(sim.model.nv, 12)
        sim.data.qvel[:6] = [1.,2.,3.,.1,.2,.3]
        mujoco.mj_forward(sim.model, sim.data)
        # Independent point Jacobian at the training COM, expressed in world frame.
        point = sim.data.xpos[sim.base] + sim.data.xmat[sim.base].reshape(3,3) @ cfg["observation_com"]
        jac = np.zeros((3,sim.model.nv)); rot = np.zeros_like(jac)
        mujoco.mj_jac(sim.model,sim.data,jac,rot,point,sim.base)
        np.testing.assert_allclose(sim.observation()[:3], jac @ sim.data.qvel, atol=2e-6)
        sim.data.time = .02
        sim.records = [dict(time=0.,position=[0.,0.,.2],upright_cos=1.,forward_speed=0.,
                            heading_yaw_rad=0.,yaw_rate_world_radps=0.,base_contact=False,front_contact=False)]
        self.assertEqual(sim.result()["balance_acceptance"]["values"]["failures"],1)
        pt = Policy(ROOT/cfg["policy"],34,6)
        onnx = Policy((ROOT/cfg["policy"]).with_suffix(".onnx"),34,6)
        for obs in np.random.default_rng(2).normal(size=(16,34)).astype(np.float32):
            np.testing.assert_allclose(pt(obs),onnx(obs),rtol=1e-4,atol=2e-5)

    def test_exported_policy_backends(self):
        policy_path = ROOT/CFG["policy"]
        pt = Policy(policy_path.with_suffix(".pt"))
        onnx = Policy(policy_path.with_suffix(".onnx"))
        inputs = np.random.default_rng(1).normal(size=(16, 56)).astype(np.float32)
        for obs in inputs:
            np.testing.assert_allclose(pt(obs), onnx(obs), rtol=1e-4, atol=2e-5)

    def test_saved_clock_contract_and_legacy_policy_replay(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'provenance.json'
            path.write_text(json.dumps({'layout':{}}))
            self.assertEqual(read_gait_clock(path), {'mode':'moving_only','period_s':1.6})
            path.write_text(json.dumps({'layout':{'gait_clock':{'mode':'always','period_s':1.6}}}))
            self.assertEqual(read_gait_clock(path), {'mode':'always','period_s':1.6})
        with patch('deploy.read_gait_clock', return_value={'mode':'moving_only','period_s':1.6}):
            sim = Simulation({**CFG, 'command':[0.,0.,0.]}, ROOT/CFG['policy'])
        self.assertEqual(sim.cfg['gait_clock_mode'], 'moving_only')
        self.assertEqual(CFG['gait_clock_mode'], 'auto')
        np.testing.assert_array_equal(sim.observation()[12:14], [0.,0.])
        with patch('deploy.read_gait_clock', return_value={'mode':'always','period_s':1.6}):
            modern = Simulation(CFG, ROOT/CFG['policy'])
        self.assertEqual(modern.cfg['gait_clock_mode'], 'always')
        for t, expected in [(0.,[0.,1.]),(.4,[1.,0.]),(.8,[0.,-1.]),(1.2,[-1.,0.])]:
            modern.data.time = t
            np.testing.assert_allclose(modern.observation()[12:14], expected, atol=1e-6)
        modern.reset()
        np.testing.assert_array_equal(modern.observation()[12:14], [0.,1.])

    def test_acceptance_rejects_turning_and_excludes_post_failure_motion(self):
        sim = Simulation({**CFG, 'stage':'balance', 'command':[0.,0.,0.]}, ROOT/CFG['policy'])
        sim.data.time = 20.
        sim.records = [dict(time=i*.02, position=[0., 0., .52], upright_cos=1.,
                            forward_speed=0., front_contact=False, base_contact=False,
                            heading_yaw_rad=float(np.arctan2(np.sin(i*.01), np.cos(i*.01))),
                            yaw_rate_world_radps=.5) for i in range(1000)]
        result = sim.result()
        self.assertFalse(result['balance_acceptance_passed'])
        self.assertAlmostEqual(result['net_heading_change_rad'], 9.99)
        self.assertEqual(result['balance_acceptance']['failed_checks'], ['yaw_rate', 'heading_excursion'])
        sim.records[800]['base_contact'] = True
        for record in sim.records[800:]:
            record['position'] = [100., 100., 0.]
        result = sim.result()
        self.assertEqual(result['horizontal_path_length_m'], 0.)
        self.assertEqual(result['balance_acceptance']['values']['valid_seconds'], 16.)
        self.assertAlmostEqual(result['net_heading_change_rad'], 7.99)
        self.assertIn('no_failure', result['balance_acceptance']['failed_checks'])


if __name__ == "__main__":
    unittest.main()
