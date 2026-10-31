"""Regression tests against the vendored upstream, without launching Isaac Sim."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


reference = module("reference_config", "ry_go2_biped_walk_v20/reference_config.py").REFERENCE
ported = module("ported_rewards", "ry_go2_biped_walk_v20/reference_rewards.py").ReferenceRewards
symmetry = module("current_symmetry", "ry_go2_biped_walk_v20/symmetry.py")
stepping = module("stepping_rewards", "ry_go2_biped_walk_v20/stepping_rewards.py")
source = (ROOT / "third_party/My_unitree_go2_gym/Go2_Handstand.py").read_text()
original_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
original_methods = {n.name: n for n in original_class.body if isinstance(n, ast.FunctionDef)}
names = ["_get_phase", "_get_gait_phase"] + ["_reward_" + k for k, v in reference["reward_scales"].items() if v != 0]
oracle_ast = ast.ClassDef(name="Oracle", bases=[], keywords=[],
                          body=[copy.deepcopy(original_methods[n]) for n in names], decorator_list=[])
namespace = {"torch": torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[oracle_ast], type_ignores=[])), "<upstream>", "exec"), namespace)
Oracle = namespace["Oracle"]


def sample(cls):
    obj = cls()
    obj.num_envs, obj.num_dof, obj.device, obj.dt = 8, 12, "cpu", .02
    obj.cfg = SimpleNamespace(rewards=SimpleNamespace(
        **reference["reward_parameters"], tracking_lin_vel_sigma=reference["reward_parameters"]["tracking_sigma"],
        tracking_ang_vel_sigma=reference["reward_parameters"]["tracking_sigma"]))
    obj.episode_length_buf = torch.tensor([1, 7, 13, 21, 31, 51, 61, 71])
    obj.root_states = torch.zeros(8, 13)
    obj.root_states[:, 2] = .52
    obj.measured_heights = torch.zeros(8, 1)
    obj.base_lin_vel = torch.randn(8, 3)
    obj.base_ang_vel = torch.randn(8, 3)
    obj.commands = torch.randn(8, 3)
    obj.projected_gravity = torch.randn(8, 3)
    obj.target_gravity = torch.tensor([-1., 0., 0.])
    obj.torques = torch.randn(8, 12)
    obj.dof_vel, obj.last_dof_vel = torch.randn(8, 12), torch.randn(8, 12)
    obj.actions, obj.last_actions = torch.randn(8, 12), torch.randn(8, 12)
    obj.dof_pos, obj.descire_joint_pos = torch.randn(8, 12), torch.randn(12)
    obj.soft_dof_pos_limits = torch.tensor([[-1., 1.]]).repeat(12, 1)
    obj.contact_forces = torch.randn(8, 6, 3)
    obj.rigid_state = torch.randn(8, 6, 13)
    obj.feet_pos = obj.rigid_state[:, :2, :3]
    obj.feet_indices = [0, 1, 2, 3]
    obj.feet_name_reward_indices = [0, 1]
    obj.contact_foot_indices = [2, 3]
    obj.penalised_contact_indices = [4, 5]
    obj.base_euler_xyz = torch.randn(8, 3)
    obj._reward_base_height()
    return obj


class ReferencePortTests(unittest.TestCase):
    def test_reward_constants_are_extracted_from_source(self):
        tree = ast.parse((ROOT / "third_party/My_unitree_go2_gym/Go2_Handstand_Config.py").read_text())
        cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2_Handstand_Cfg_Yu")
        rewards = next(n for n in cfg.body if isinstance(n, ast.ClassDef) and n.name == "rewards")
        scales = next(n for n in rewards.body if isinstance(n, ast.ClassDef) and n.name == "scales")
        expected = {n.targets[0].id: ast.literal_eval(n.value) for n in scales.body if isinstance(n, ast.Assign)}
        self.assertEqual(expected, reference["reward_scales"])
        self.assertEqual(23, sum(v != 0 for v in expected.values()))
        self.assertNotIn("feet_air_time", expected)

    def test_only_documented_formula_changes(self):
        methods = {n.name: n for n in ast.parse((ROOT / "ry_go2_biped_walk_v20/reference_rewards.py").read_text()).body[-1].body
                   if isinstance(n, ast.FunctionDef)}
        for name in names:
            text = ast.get_source_segment(source, original_methods[name])
            text = text.replace("phase>0.5", "phase>=0.5")
            text = text.replace("torch.mean(torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target)*5))",
                                "torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target)*5)")
            text = text.replace("torch.mean(self.rew_hanstand)", "self.rew_hanstand")
            if name == "_reward_tracking_lin_vel":
                text = text.replace("self.cfg.rewards.tracking_sigma", "self.cfg.rewards.tracking_lin_vel_sigma")
            if name == "_reward_tracking_ang_vel":
                text = text.replace("self.cfg.rewards.tracking_sigma", "self.cfg.rewards.tracking_ang_vel_sigma")
            self.assertEqual(ast.dump(ast.parse(text).body[0]), ast.dump(methods[name]), name)

    def test_all_23_rewards_match_upstream_on_same_state(self):
        torch.manual_seed(7)
        original = sample(Oracle)
        current = ported()
        current.__dict__.update(copy.deepcopy(original.__dict__))
        current._reward_base_height()
        for name in names[2:]:
            expected = torch.as_tensor(getattr(original, name)())
            actual = torch.as_tensor(getattr(current, name)())
            torch.testing.assert_close(actual, expected, msg=name)

    def test_height_gate_is_independent_between_environments(self):
        obj = sample(ported)
        obj.base_lin_vel.zero_()
        obj.commands.zero_()
        obj.root_states[1:, 2] = 0.
        obj._reward_base_height()
        reward = obj._reward_tracking_lin_vel()
        self.assertEqual(1., reward[0].item())
        self.assertTrue(torch.equal(reward[1:], torch.zeros(7)))

    def test_balance_linear_width_preserves_angular_reward_and_height_gate(self):
        obj = sample(ported)
        obj.commands.zero_()
        obj.base_lin_vel.zero_()
        obj.base_ang_vel.fill_(0.2)
        obj.base_lin_vel[:, 2] = torch.tensor([0., .05, .1, .2, -.2, .3, 0., .2])
        original_angular = obj._reward_tracking_ang_vel().clone()
        for width, values in [
            (.04, [1., .939413, .778801, .367879, .367879, .105399, 1., .367879]),
            (.01, [1., .778801, .367879, .018316, .018316, .00012341, 1., .018316]),
            (.0025, [1., .367879, .018316, 1.12535e-7, 1.12535e-7, 2.31952e-16, 1., 1.12535e-7]),
        ]:
            with self.subTest(width=width):
                obj.cfg.rewards.tracking_lin_vel_sigma = width
                torch.testing.assert_close(obj._reward_tracking_lin_vel(), torch.tensor(values), atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(obj._reward_tracking_ang_vel(), original_angular)
        obj.root_states[-1, 2] = 0.
        obj._reward_base_height()
        self.assertEqual(obj._reward_tracking_lin_vel()[-1].item(), 0.)

    def test_balance_angular_width_preserves_linear_reward_axis_and_height_gate(self):
        obj = sample(ported)
        obj.commands.zero_()
        obj.base_ang_vel.zero_()
        obj.base_ang_vel[:, 0] = torch.tensor([0., .1, .2, .3, .4, .5, -.2, .1])
        obj.cfg.rewards.tracking_lin_vel_sigma = .0025
        original_linear = obj._reward_tracking_lin_vel().clone()
        obj.cfg.rewards.tracking_ang_vel_sigma = .04
        expected = torch.tensor([1., .778801, .367879, .105399, .018316, .00193045, .367879, .778801])
        torch.testing.assert_close(obj._reward_tracking_ang_vel(), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), original_linear)
        obj.base_ang_vel[:, 1:] = 10.
        torch.testing.assert_close(obj._reward_tracking_ang_vel(), expected, atol=1e-6, rtol=1e-5)
        obj.root_states[-1, 2] = 0.
        obj._reward_base_height()
        self.assertEqual(obj._reward_tracking_ang_vel()[-1].item(), 0.)

    def test_stage_switch_restores_tracking_widths(self):
        tree = ast.parse((ROOT / "ry_go2_biped_walk_v20/gait_env_cfg.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure_stage")
        namespace = {"REFERENCE": reference, "ReferenceRewardsCfg": lambda: SimpleNamespace(
                         **{name: SimpleNamespace(weight=weight) for name, weight in reference["reward_scales"].items() if weight != 0}),
                     "RewTerm": SimpleNamespace, "wm": SimpleNamespace(feet_slide_l2=object()),
                     "gm": SimpleNamespace(balance_step_reward=object(), balance_zero_yaw_l2=object(),
                                           balance_heading_alignment=object()),
                     "REAR_BODIES": object(), "REAR_CONTACTS": object(), "SCANNER": object()}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<configure_stage>", "exec"), namespace)
        cfg = SimpleNamespace(commands=SimpleNamespace(base_velocity=SimpleNamespace(ranges=SimpleNamespace())))
        for stage, linear, angular in [("balance", .0025, .04), ("walk", .25, .25), ("balance", .0025, .04)]:
            namespace["configure_stage"](cfg, stage)
            self.assertEqual((cfg.tracking_lin_vel_sigma, cfg.tracking_ang_vel_sigma), (linear, angular))
            if stage == "balance":
                for name in ("contact", "feet_clearance"):
                    term = getattr(cfg.rewards, name)
                    self.assertIs(term.func, namespace["gm"].balance_step_reward)
                    self.assertEqual(term.weight, .9 if name == "contact" else 1.0)
                term = cfg.rewards.foot_slip
                self.assertEqual(term.weight, -0.5)
                self.assertIs(term.func, namespace["wm"].feet_slide_l2)
                self.assertIs(term.params["asset_cfg"], namespace["REAR_BODIES"])
                self.assertIs(term.params["sensor_cfg"], namespace["REAR_CONTACTS"])
                self.assertEqual(cfg.rewards.tracking_ang_vel_zero.weight, -.2)
                self.assertIs(cfg.rewards.tracking_ang_vel_zero.func, namespace['gm'].balance_zero_yaw_l2)
                self.assertEqual(cfg.rewards.heading_alignment.weight, 1.5)
                self.assertIs(cfg.rewards.heading_alignment.func, namespace['gm'].balance_heading_alignment)
                self.assertEqual(cfg.rewards.heading_alignment.params['std'], .6)
                self.assertIs(cfg.rewards.heading_alignment.params['sensor_cfg'], namespace['SCANNER'])
                self.assertEqual(len(vars(cfg.rewards)), 26)
            else:
                self.assertFalse(hasattr(cfg.rewards, "foot_slip"))
                self.assertFalse(hasattr(cfg.rewards, "tracking_ang_vel_zero"))
                self.assertFalse(hasattr(cfg.rewards, "heading_alignment"))
                self.assertEqual(len(vars(cfg.rewards)), 23)
                self.assertEqual(cfg.rewards.contact.weight, .3)
                self.assertEqual(cfg.rewards.feet_clearance.weight, .4)

    def test_zero_yaw_penalty_world_axis_command_and_per_environment_gate(self):
        path = ROOT / 'ry_go2_biped_walk_v20/gait_mdp.py'
        fn = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'balance_zero_yaw_l2')
        obj = SimpleNamespace(commands=torch.zeros(8, 3), rew_hanstand=torch.ones(8),
                              robot=SimpleNamespace(data=SimpleNamespace(root_ang_vel_w=torch.zeros(8, 3))))
        obj.robot.data.root_ang_vel_w[:, 2] = torch.tensor([0., .5, -.5, 1., 2., .5, .5, .5])
        obj.commands[5, 2] = .1  # strict boundary: nonzero-turn penalty disabled
        obj.commands[6, 2] = .05
        obj.rew_hanstand[7] = .78  # strict per-env height gate
        namespace = {'state': lambda _: obj}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        reward = namespace['balance_zero_yaw_l2'](None)
        torch.testing.assert_close(reward, torch.tensor([0., .25, .25, 1., 4., 0., .2025, 0.]))
        obj.robot.data.root_ang_vel_w[:, :2] = 100.
        torch.testing.assert_close(namespace['balance_zero_yaw_l2'](None), reward)
        # Mirror the axial yaw velocity and command: penalty must stay unchanged.
        obj.robot.data.root_ang_vel_w[:, 2] *= -1
        obj.commands[:, 2] *= -1
        torch.testing.assert_close(namespace['balance_zero_yaw_l2'](None), reward)

    def test_heading_alignment_wraps_and_preserves_original_posture_gate(self):
        path = ROOT / 'ry_go2_biped_walk_v20/gait_mdp.py'
        fn = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'balance_heading_alignment')
        yaw = torch.tensor([0., .6, -.6, 2*torch.pi+.6, torch.pi, -torch.pi, 0.])
        gate = torch.tensor([1., 1., 1., 1., 1., 1., .25])
        scanner = object()
        def posture_gate(env, sensor):
            self.assertIs(sensor, scanner)
            return gate
        namespace = {'torch': torch, 'wm': SimpleNamespace(
            upright_heading=lambda _: yaw, posture_gate=posture_gate)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        reward = namespace['balance_heading_alignment'](None, scanner)
        torch.testing.assert_close(reward, torch.tensor([1., .36787944, .36787944, .36787944,
                                                        1.240424e-12, 1.240424e-12, .25]))
        yaw *= -1
        torch.testing.assert_close(namespace['balance_heading_alignment'](None, scanner), reward)

    def test_hind_foot_slip_uses_contact_and_horizontal_velocity(self):
        path = ROOT / "ry_unitree_rl_lab/tasks/locomotion/robots/go2_biped_walk/walk_mdp.py"
        fn = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == "feet_slide_l2")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
        class Scene(dict):
            pass
        # Robot and contact sensor body orders intentionally differ.
        velocity = torch.full((4, 4, 3), 100.)
        velocity[:, 1, :2] = torch.tensor([3., 4.])
        velocity[:, 3, :2] = torch.tensor([6., 8.])
        contact_time = torch.ones(4, 4)
        contact_time[:, [2, 0]] = torch.tensor([[0., 0.], [.02, 0.], [0., .02], [.02, .02]])
        scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=velocity)))
        scene.sensors = {"contact": SimpleNamespace(data=SimpleNamespace(current_contact_time=contact_time))}
        value = namespace["feet_slide_l2"](
            SimpleNamespace(scene=scene), SimpleNamespace(name="robot", body_ids=[1, 3]),
            SimpleNamespace(name="contact", body_ids=[2, 0]))
        # Airborne feet, front feet and vertical speed must not contribute.
        torch.testing.assert_close(value, torch.tensor([0., 25., 100., 125.]))

    def test_phase_boundary_always_has_one_stance(self):
        obj = sample(ported)
        obj.episode_length_buf = torch.tensor([0, 40, 80, 120, 160, 200, 240, 280])
        torch.testing.assert_close(obj._get_gait_phase().sum(-1), torch.ones(8))

    def test_small_steps_alternate_with_double_support_and_no_flight_target(self):
        phase = torch.tensor([0., .025, .25, .475, .5, .525, .75, .975])
        height, stance = stepping.step_targets(phase)
        torch.testing.assert_close(height[[2, 6]], torch.tensor([[.06, 0.], [0., .06]]))
        self.assertTrue(stance[[0, 1, 3, 4, 5, 7]].all())
        self.assertTrue(stance.any(-1).all())
        torch.testing.assert_close(stepping.clearance_score(height, height), torch.ones(8))
        self.assertTrue((stepping.clearance_score(torch.zeros_like(height), height)[[2, 6]] < .6).all())
        torch.testing.assert_close(stepping.contact_score(stance, stance), torch.ones(8))
        # Keeping only one leg on the ground cannot satisfy both swing phases.
        fixed_right = torch.tensor([[False, True]]).expand(8, 2)
        torch.testing.assert_close(stepping.contact_score(fixed_right, stance)[[2, 6]], torch.tensor([1., 0.]))

    def test_small_steps_repeat_and_mirror_after_half_cycle(self):
        phase = torch.arange(80) / 80
        height, stance = stepping.step_targets(phase)
        h2, s2 = stepping.step_targets(phase + .5)
        torch.testing.assert_close(h2, height.flip(-1), atol=1e-7, rtol=1e-5)
        # Avoid floating-point comparisons exactly on phase boundaries.
        interior = ((phase * 100).round().long() % 5 != 0)
        self.assertTrue(torch.equal(s2[interior], stance.flip(-1)[interior]))

    def test_reflection_round_trip_and_native_joint_permutation(self):
        joints = symmetry.ACTION_NAMES[::3] + symmetry.ACTION_NAMES[1::3] + symmetry.ACTION_NAMES[2::3]
        values = torch.arange(12.).unsqueeze(0)
        mirrored = symmetry.transform_joints(values, joints)
        self.assertEqual(-1., mirrored[0, 0].item())
        torch.testing.assert_close(symmetry.transform_joints(mirrored, joints), values)
        tagged = torch.arange(56.).unsqueeze(0)
        tagged_mirror = symmetry.mirror_policy(tagged, symmetry.POLICY_NAMES, symmetry.POLICY_DIMS, joints)
        self.assertEqual(-15., tagged_mirror[0, 14].item())  # FL hip observation <- native FR hip
        self.assertEqual(19., tagged_mirror[0, 18].item())   # FL thigh observation <- native FR thigh
        self.assertEqual(-41., tagged_mirror[0, 38].item())  # FL action <- action-order FR hip
        obs = torch.randn(8, 56)
        mirror = symmetry.mirror_policy(obs, symmetry.POLICY_NAMES, symmetry.POLICY_DIMS, joints)
        torch.testing.assert_close(symmetry.mirror_policy(mirror, symmetry.POLICY_NAMES, symmetry.POLICY_DIMS, joints), obs)
        names = symmetry.POLICY_NAMES[:8] + ["height_scan"] + symmetry.POLICY_NAMES[8:]
        dims = symmetry.POLICY_DIMS[:8] + [99] + symmetry.POLICY_DIMS[8:]
        critic = torch.randn(8, 155)
        mirror = symmetry.mirror_critic(critic, names, dims, joints)
        torch.testing.assert_close(symmetry.mirror_critic(mirror, names, dims, joints), critic)


if __name__ == "__main__":
    unittest.main()
