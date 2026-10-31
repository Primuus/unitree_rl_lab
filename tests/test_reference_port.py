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
        tracking_ang_vel_sigma=reference["reward_parameters"]["tracking_sigma"],
        base_height_reward_width=.05, task_upright_cos_min=.94))
    obj.episode_length_buf = torch.tensor([1, 7, 13, 21, 31, 51, 61, 71])
    obj.root_states = torch.zeros(8, 13)
    obj.root_states[:, 2] = .52
    obj.measured_heights = torch.zeros(8, 1)
    obj.base_lin_vel = torch.randn(8, 3)
    obj.base_ang_vel = torch.randn(8, 3)
    obj.commands = torch.randn(8, 3)
    obj.projected_gravity = torch.tensor([-1., 0., 0.]).expand(8, 3).clone()
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
            if name in ("_reward_tracking_lin_vel", "_reward_tracking_ang_vel"):
                text = text.replace("(self.rew_hanstand>0.70)", "self._task_upright_gate()")
            if name == "_reward_base_height":
                text = text.replace(
                    "return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target)*5)",
                    "return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target)"
                    "/ self.cfg.rewards.base_height_reward_width)")
            if name == "_reward_default_pos":
                text = text.replace("self.dof_pos - self.descire_joint_pos",
                                    "self.dof_pos[:, :6] - self.descire_joint_pos[:6]")
            if name == "_reward_symmetric_joints":
                text = "".join(line for line in text.splitlines(keepends=True)
                               if "dof[:,3,0] *= -1" not in line)
                text = text.replace(" + torch.sum(torch.abs(dof[:,2,:] - dof[:,3,:]),axis=1)", "")
            self.assertEqual(ast.dump(ast.parse(text).body[0]), ast.dump(methods[name]), name)

    def test_unmodified_rewards_match_upstream_on_same_state(self):
        torch.manual_seed(7)
        original = sample(Oracle)
        current = ported()
        current.__dict__.update(copy.deepcopy(original.__dict__))
        current._reward_base_height()
        for name in names[2:]:
            if name in ("_reward_default_pos", "_reward_symmetric_joints"):
                continue  # Front-only adaptations are covered by behavioral tests below.
            expected = torch.as_tensor(getattr(original, name)())
            actual = torch.as_tensor(getattr(current, name)())
            torch.testing.assert_close(actual, expected, msg=name)

    def test_hind_alternation_does_not_change_front_posture_or_symmetry_penalties(self):
        obj = sample(ported)
        obj.dof_pos = obj.descire_joint_pos.expand(8, -1).clone()
        posture = obj._reward_default_pos().clone()
        symmetry_cost = obj._reward_symmetric_joints().clone()
        for command in (0., .075):
            obj.commands.fill_(0.)
            obj.commands[:, 0] = command
            # Unequal hind thigh/calf excursions and mirrored hip deviations.
            obj.dof_pos[:, 6:] += torch.tensor([.2, .7, -.5, -.1, -.3, .4])
            torch.testing.assert_close(obj._reward_default_pos(), posture)
            torch.testing.assert_close(obj._reward_symmetric_joints(), symmetry_cost)
        # A front joint deviation must still change both penalties.
        obj.dof_pos[:, 1] += .25
        torch.testing.assert_close(obj._reward_default_pos(), torch.full((8,), .25))
        self.assertFalse(torch.allclose(obj._reward_symmetric_joints(), symmetry_cost))
        obj.rew_hanstand[0] = .69
        self.assertEqual(obj._reward_symmetric_joints()[0].item(), 0.)

    def test_front_symmetry_hip_sign_and_hind_hip_regularization_remain(self):
        obj = sample(ported)
        obj.dof_pos.zero_()
        obj.dof_pos[:, :6] = torch.tensor([.2, .8, -1.5, -.2, .8, -1.5])
        torch.testing.assert_close(obj._reward_symmetric_joints(), torch.zeros(8))
        before = obj._reward_default_hip_pos().clone()
        obj.dof_pos[:, 6] = .3
        torch.testing.assert_close(obj._reward_default_hip_pos(), before + .3)
        torch.testing.assert_close(obj._reward_symmetric_joints(), torch.zeros(8))
        obj.dof_pos[:, 3] = .2
        torch.testing.assert_close(obj._reward_symmetric_joints(), torch.full((8,), .4))

    def test_tracking_orientation_gate_is_independent_of_height_and_other_environments(self):
        obj = sample(ported)
        obj.base_lin_vel.zero_()
        obj.base_ang_vel.zero_()
        obj.commands.zero_()
        obj.root_states[:, 2] = torch.tensor([.52, .425, .35, .6, .425, .425, .425, .425])
        obj.projected_gravity[4:, 0] = torch.tensor([-.94, -.939, 0., 1.])
        obj._reward_base_height()
        expected = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.])
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), expected)
        torch.testing.assert_close(obj._reward_tracking_ang_vel(), expected)
        # Tightening the height reward must not tighten task or legacy gates.
        legacy_score = obj.rew_hanstand.clone()
        obj.cfg.rewards.base_height_reward_width = .01
        obj._reward_base_height()
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), expected)
        torch.testing.assert_close(obj.rew_hanstand, legacy_score)

    def test_height_reward_is_narrow_and_separate_from_legacy_height_score(self):
        obj = sample(ported)
        obj.root_states[:, 2] = torch.tensor([.52, .48, .425, .56, .615, .52, .52, .52])
        expected = torch.tensor([1., .44932896, .14956862, .44932896, .14956862, 1., 1., 1.])
        torch.testing.assert_close(obj._reward_base_height(), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(obj.rew_hanstand,
                                  torch.exp(-5*(obj.root_states[:, 2]-.52).abs()))
        self.assertGreater(obj.rew_hanstand[2].item(), .6)
        self.assertLess(expected[2].item(), .15)

    def test_balance_linear_width_preserves_angular_reward_and_orientation_gate(self):
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
        obj.projected_gravity[-1] = torch.tensor([0., 0., -1.])
        self.assertEqual(obj._reward_tracking_lin_vel()[-1].item(), 0.)

    def test_balance_angular_width_preserves_linear_reward_axis_and_orientation_gate(self):
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
        obj.projected_gravity[-1] = torch.tensor([0., 0., -1.])
        self.assertEqual(obj._reward_tracking_ang_vel()[-1].item(), 0.)

    def test_stage_switch_preserves_shared_rewards_and_changes_command_distribution(self):
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
        for stage, linear, angular in [("balance", .0025, .04), ("walk", .0025, .04), ("balance", .0025, .04)]:
            namespace["configure_stage"](cfg, stage)
            self.assertEqual((cfg.tracking_lin_vel_sigma, cfg.tracking_ang_vel_sigma), (linear, angular))
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
            self.assertEqual(cfg.rewards.feet_clearance_deficit.weight, -1.)
            self.assertEqual(cfg.rewards.feet_clearance_deficit.params, {'name': 'feet_clearance_deficit'})
            self.assertIs(cfg.rewards.feet_clearance_deficit.func, namespace['gm'].balance_step_reward)
            self.assertEqual(len(vars(cfg.rewards)), 27)
            command = cfg.commands.base_velocity
            self.assertEqual(command.ranges.lin_vel_x, (.05, .10) if stage == "balance" else (.2, .2))
            self.assertEqual(command.rel_standing_envs, .5 if stage == "balance" else 0.)
            self.assertEqual(command.ranges.lin_vel_y, (0., 0.))
            self.assertEqual(command.ranges.ang_vel_z, (0., 0.))
            self.assertFalse(command.heading_command)
            self.assertEqual(command.resampling_time_range, (10., 10.))
            self.assertEqual(cfg.balance_gait_clock, "always")

    def test_zero_yaw_penalty_world_axis_command_and_no_height_gate(self):
        path = ROOT / 'ry_go2_biped_walk_v20/gait_mdp.py'
        fn = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'balance_zero_yaw_l2')
        obj = SimpleNamespace(commands=torch.zeros(8, 3), rew_hanstand=torch.ones(8),
                              robot=SimpleNamespace(data=SimpleNamespace(root_ang_vel_w=torch.zeros(8, 3))))
        obj.robot.data.root_ang_vel_w[:, 2] = torch.tensor([0., .5, -.5, 1., 2., .5, .5, .5])
        obj.commands[5, 2] = .1  # strict boundary: nonzero-turn penalty disabled
        obj.commands[6, 2] = .05
        obj.rew_hanstand[7] = 0.  # Lowering the body must not disable this penalty.
        namespace = {'state': lambda _: obj}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        reward = namespace['balance_zero_yaw_l2'](None)
        torch.testing.assert_close(reward, torch.tensor([0., .25, .25, 1., 4., 0., .2025, .25]))
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

    def test_swing_shortfall_decreases_with_lift_and_ignores_support(self):
        phase = torch.tensor([.25, .25, .25, .25, .75, 0., .5])
        target, stance = stepping.step_targets(phase)
        height = torch.tensor([[0., 0.], [.03, .2], [.06, 0.], [.08, 0.],
                               [.2, .03], [0., 0.], [0., 0.]])
        cost = stepping.clearance_deficit(height, target, ~stance)
        torch.testing.assert_close(cost, torch.tensor([1., .25, 0., 0., .25, 0., 0.]))
        target2, stance2 = stepping.step_targets(phase+.5)
        torch.testing.assert_close(stepping.clearance_deficit(height.flip(-1), target2, ~stance2), cost)

    def test_low_stance_keeps_step_signals_and_deficit_survives_orientation_gate(self):
        path = ROOT/'ry_go2_biped_walk_v20/gait_mdp.py'
        fn = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'balance_step_reward')
        positions = torch.zeros(3,4,3)
        positions[:,:,2] = .022
        positions[1,2,2] += .03
        s = SimpleNamespace(_get_phase=lambda:torch.full((3,),.25),num_envs=3,contact_foot_indices=[2,3],
                            measured_heights=torch.zeros(3,1),rew_hanstand=torch.full((3,),.6),
                            root_states=torch.tensor([[0.,0.,.425],[0.,0.,.46],[0.,0.,.50]]),
                            _task_upright_gate=lambda:torch.tensor([True,True,False]),
                            contact_forces=torch.tensor([[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,10.]]]).expand(3,4,3),
                            robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=positions,
                                                                       body_quat_w=torch.zeros(3,4,4))))
        namespace = {'state':lambda _:s,'step_targets':stepping.step_targets,
                     'clearance_score':stepping.clearance_score,'clearance_deficit':stepping.clearance_deficit,
                     'contact_score':stepping.contact_score,'body_height_factor':stepping.body_height_factor,
                     'quat_apply':lambda quat,offset:offset}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
        env = SimpleNamespace(num_envs=3,cfg=SimpleNamespace(balance_step_height=.06,balance_double_support_fraction=.2,
                                                   balance_clearance_width=.01, step_reward_height_low=.43,
                                                   step_reward_height_full=.50, step_reward_height_min_factor=.3))
        torch.testing.assert_close(namespace['balance_step_reward'](env,'feet_clearance_deficit'),
                                   torch.tensor([1.,.25,1.]))
        expected = torch.tensor([(1.+torch.exp(torch.tensor(-6.)))/2,
                                 (1.+torch.exp(torch.tensor(-3.)))/2,0.])
        torch.testing.assert_close(namespace['balance_step_reward'](env,'feet_clearance'),expected*torch.tensor([.3,.6,1.]))
        torch.testing.assert_close(namespace['balance_step_reward'](env,'contact'),
                                   torch.tensor([.3,.6,0.]))

    def test_step_height_factor_endpoints_monotonicity_and_batch_independence(self):
        height = torch.tensor([.3, .425, .43, .445, .46, .48, .5, .6])
        factor = stepping.body_height_factor(height)
        torch.testing.assert_close(factor, torch.tensor([.3, .3, .3, .45, .6, .8, 1., 1.]))
        self.assertTrue((factor[1:] >= factor[:-1]).all())
        for i in range(len(height)):
            torch.testing.assert_close(stepping.body_height_factor(height[i:i+1]), factor[i:i+1])

    def test_shared_clock_survives_stop_start_and_supports_legacy_replay(self):
        import math
        original_path = ROOT/'ry_unitree_rl_lab/tasks/locomotion/robots/go2_biped_walk/walk_mdp.py'
        nodes = [n for n in ast.parse(original_path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in ('moving', 'phases', 'gait_clock')]
        original = {'torch': torch, 'math': math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(original_path), 'exec'), original)
        path = ROOT/'ry_go2_biped_walk_v20/gait_mdp.py'
        fn = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'gait_clock')
        namespace = {'torch': torch, 'wm': SimpleNamespace(gait_clock=original['gait_clock'])}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        command = torch.zeros(4, 3)
        env = SimpleNamespace(cfg=SimpleNamespace(stage='balance', balance_gait_clock='always'),
                              episode_length_buf=torch.tensor([0, 20, 40, 60]),step_dt=.02,
                              num_envs=4,device='cpu',
                              command_manager=SimpleNamespace(get_command=lambda _: command))
        clock = namespace['gait_clock'](env)
        torch.testing.assert_close(clock, torch.tensor([[0.,1.],[1.,0.],[0.,-1.],[-1.,0.]]),atol=1e-6,rtol=1e-5)
        target, _ = stepping.step_targets(env.episode_length_buf*.02/1.6)
        self.assertGreater(target[1,0], 0.)
        self.assertGreater(target[3,1], 0.)
        obs = torch.zeros(4,56); obs[:,12:14] = clock
        mirrored = symmetry.mirror_policy(obs, symmetry.POLICY_NAMES, symmetry.POLICY_DIMS, symmetry.ACTION_NAMES)
        env.episode_length_buf += 40
        torch.testing.assert_close(mirrored[:,12:14], namespace['gait_clock'](env),atol=1e-6,rtol=1e-5)
        env.episode_length_buf[:] = 0
        torch.testing.assert_close(namespace['gait_clock'](env),torch.tensor([[0.,1.]]).expand(4,2))
        env.cfg.balance_gait_clock = 'moving_only'
        torch.testing.assert_close(namespace['gait_clock'](env),torch.zeros(4,2))
        env.cfg.stage = 'walk'; env.cfg.balance_gait_clock = 'always'
        torch.testing.assert_close(namespace['gait_clock'](env),torch.tensor([[0.,1.]]).expand(4,2))
        command[:,0] = .2
        torch.testing.assert_close(namespace['gait_clock'](env),torch.tensor([[0.,1.]]).expand(4,2))

    def test_tracking_uses_each_environment_command_in_mixed_batch(self):
        obj = sample(ported)
        obj.cfg.rewards.tracking_lin_vel_sigma = .0025
        obj.commands.zero_()
        obj.commands[:, 0] = torch.tensor([0., .05, .075, .1, 0., .05, .075, .1])
        obj.base_lin_vel.zero_()
        obj.base_lin_vel[:, 2] = -obj.commands[:, 0]
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), torch.ones(8))
        obj.base_lin_vel.zero_()
        expected = torch.exp(-obj.commands[:, 0].square() / .0025)
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), expected)
        self.assertLess(expected[3].item(), .02)

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
