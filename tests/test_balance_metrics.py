"""Verify small-step evidence and exclusion of post-failure motion."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

spec = importlib.util.spec_from_file_location("metrics", Path(__file__).resolve().parents[1] / "scripts/tools/go2_walk_metrics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(stage):
    force = torch.zeros(1, 4, 3)
    force[:, :, 2] = torch.tensor([0., 0., 20., 20.])
    sensor = NS(data=NS(net_forces_w=force), find_bodies=lambda *a, **k: ([0, 1], []))
    data = NS(root_pos_w=torch.tensor([[0., 0., .52]]), root_lin_vel_w=torch.zeros(1, 3),
              root_ang_vel_w=torch.zeros(1, 3), projected_gravity_b=torch.tensor([[-1., 0., 0.]]),
              body_pos_w=torch.zeros(1, 4, 3), body_lin_vel_w=torch.zeros(1, 4, 3))
    raw = NS(num_envs=1, device="cpu", step_dt=.02, cfg=NS(stage=stage), yaw=torch.zeros(1),
             termination_manager=NS(terminated=torch.tensor([False]), active_terms=[]),
             command_manager=NS(get_command=lambda _: torch.zeros(1, 3)))
    wm = NS(upright_heading=lambda _: raw.yaw, heading_lin_vel=lambda _: data.root_lin_vel_w,
            root_height=lambda *a: data.root_pos_w[:, 2])
    return module.GaitMetrics(raw, NS(data=data), [2, 3], sensor, [2, 3], wm, None)


class BalanceMetricsTests(unittest.TestCase):
    def test_real_small_lift_counts_but_contact_chatter_does_not(self):
        for stage, lift, expected in [("balance", .015, 1), ("balance", .001, 0), ("walk", .015, 0)]:
            with self.subTest(stage=stage, lift=lift):
                m = fixture(stage)
                for step in range(20):
                    airborne = 1 <= step <= 8
                    m.sensor.data.net_forces_w[0, 2, 2] = 0. if airborne else 20.
                    m.robot.data.body_pos_w[0, 2, 2] = lift if airborne else 0.
                    m.update(step, torch.tensor([False]))
                self.assertEqual(m.touch[0, 0].item(), expected)

    def test_position_and_cycles_stop_at_first_failure(self):
        m = fixture("balance")
        m.cycle_steps = 2
        for step, x in enumerate([.01, .02, 100., 200.]):
            m.robot.data.root_pos_w[0, 0] = x
            m.raw.yaw[:] = [0.01, 0.02, 2., -2.][step]
            m.raw.termination_manager.terminated[:] = step == 2
            m.update(step, torch.tensor([step == 2]))
        result = m.result(True)
        self.assertAlmostEqual(result['net_horizontal_displacement_m'][0], .02)
        self.assertAlmostEqual(result['max_horizontal_displacement_m'][0], .02)
        self.assertEqual(result['cycle_displacements'][0]['valid'], [True])
        self.assertEqual(result['cycle_displacements'][1]['valid'], [False])
        self.assertEqual(result['falls'], 1)
        self.assertAlmostEqual(result['horizontal_path_length_m'][0], .02)
        self.assertAlmostEqual(result['net_heading_change_rad'][0], .02)
        self.assertAlmostEqual(result['absolute_heading_travel_rad'][0], .02)

    def test_stationary_policy_passes_but_smoke_and_short_rollout_do_not(self):
        m = fixture('balance')
        m.update(0, torch.tensor([False]))
        self.assertIn('valid_duration', m.result(True)['balance_acceptance_reports'][0]['failed_checks'])
        for step in range(1, 1000):
            m.update(step, torch.tensor([False]))
        self.assertTrue(m.result(True)['balance_acceptance_passed'])
        self.assertFalse(m.result(False)['balance_acceptance_passed'])

    def test_full_turn_is_not_hidden_by_angle_wrapping(self):
        m = fixture('balance')
        for step in range(1000):
            angle = torch.tensor((step+1)*2*torch.pi/1000)
            m.raw.yaw[:] = torch.atan2(angle.sin(), angle.cos())
            m.robot.data.root_ang_vel_w[0, 2] = 2*torch.pi/20
            m.update(step, torch.tensor([False]))
        r = m.result(True)
        self.assertAlmostEqual(r['net_heading_change_rad'][0], 2*torch.pi, places=4)
        self.assertIn('heading_excursion', r['balance_acceptance_reports'][0]['failed_checks'])
        self.assertIn('yaw_rate', r['balance_acceptance_reports'][0]['failed_checks'])

    def test_small_crossing_of_pi_does_not_look_like_full_turn(self):
        m = fixture('balance')
        m.previous_yaw[:] = 3.13
        m.raw.yaw[:] = -3.13
        m.update(0, torch.tensor([False]))
        self.assertAlmostEqual(m.result(True)['net_heading_change_rad'][0], 2*torch.pi-6.26, places=5)

    def test_back_and_forth_motion_cannot_hide_in_net_displacement(self):
        m = fixture('balance')
        for step in range(1000):
            m.robot.data.root_pos_w[0, 1] = .1*torch.sin(torch.tensor((step+1)*2*torch.pi/100))
            m.update(step, torch.tensor([False]))
        r = m.result(True)
        self.assertLess(r['net_horizontal_displacement_m'][0], .001)
        self.assertLess(r['max_horizontal_displacement_m'][0], .11)
        self.assertGreater(r['horizontal_path_length_m'][0], 3.9)
        self.assertEqual(r['balance_acceptance_reports'][0]['failed_checks'], ['horizontal_path'])

    def test_slow_drift_and_turn_fail_even_with_small_mean_velocities(self):
        m = fixture('balance')
        for step in range(1000):
            m.raw.yaw[:] = .02*(step+1)*.02
            m.robot.data.root_ang_vel_w[0, 2] = .02
            m.robot.data.root_pos_w[0, 0] = .015*(step+1)*.02
            m.robot.data.root_lin_vel_w[0, 0] = .015
            m.update(step, torch.tensor([False]))
        r = m.result(True)['balance_acceptance_reports'][0]
        self.assertEqual(r['failed_checks'], ['heading_excursion', 'horizontal_displacement'])


if __name__ == '__main__':
    unittest.main()
