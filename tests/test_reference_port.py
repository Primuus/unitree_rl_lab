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
    obj.cfg = SimpleNamespace(rewards=SimpleNamespace(**reference["reward_parameters"]))
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

    def test_reward_methods_are_identical_to_pinned_upstream(self):
        tree = ast.parse((ROOT / "ry_go2_biped_walk_v20/reference_rewards.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(methods), set(names))
        for name in names:
            self.assertEqual(ast.dump(methods[name]), ast.dump(original_methods[name]), name)

    def test_all_rewards_match_upstream_for_high_low_and_mixed_batches(self):
        for heights in ([.52]*8, [.425]*8, [.52]*7+[.2], [.3,.4,.425,.48,.52,.56,.6,.7]):
            with self.subTest(heights=heights):
                torch.manual_seed(7)
                original = sample(Oracle)
                original.root_states[:, 2] = torch.tensor(heights)
                original._reward_base_height()
                current = ported()
                current.__dict__.update(copy.deepcopy(original.__dict__))
                for name in names:
                    expected = torch.as_tensor(getattr(original, name)())
                    actual = torch.as_tensor(getattr(current, name)())
                    torch.testing.assert_close(actual, expected, msg=name)

    def test_height_score_and_batch_gate_match_reference(self):
        obj = sample(ported)
        obj.commands.zero_()
        obj.base_lin_vel.zero_()
        obj.base_ang_vel.zero_()
        obj.root_states[:, 2] = .425
        torch.testing.assert_close(obj._reward_base_height(), torch.full((8,), .6218851))
        self.assertEqual(obj.rew_hanstand.ndim, 0)
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), torch.zeros(8))
        # Seven tall environments enable the shared gate even for one low robot.
        obj.root_states[:7, 2] = .52
        obj._reward_base_height()
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), torch.ones(8))
        torch.testing.assert_close(obj._reward_tracking_ang_vel(), torch.ones(8))
        # Original gate is strict and does not check orientation.
        obj.rew_hanstand = torch.tensor(.70)
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), torch.zeros(8))

    def test_reference_tracking_width_and_body_axes(self):
        obj = sample(ported)
        obj.commands.zero_()
        obj.base_lin_vel.zero_()
        obj.base_ang_vel.zero_()
        obj.base_lin_vel[:, 2] = .2
        obj.base_ang_vel[:, 0] = .2
        expected = torch.full((8,), .85214379)
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), expected)
        torch.testing.assert_close(obj._reward_tracking_ang_vel(), expected)
        obj.commands[:, 0] = -.2
        obj.commands[:, 2] = .2
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), torch.ones(8))
        torch.testing.assert_close(obj._reward_tracking_ang_vel(), torch.ones(8))

    def test_hind_joints_again_contribute_to_posture_and_symmetry(self):
        obj = sample(ported)
        obj.descire_joint_pos = torch.tensor([reference["desired_joint_angles"][n]
                                             for n in symmetry.ACTION_NAMES])
        obj.dof_pos = obj.descire_joint_pos.expand(8, -1).clone()
        posture = obj._reward_default_pos()
        symmetric = obj._reward_symmetric_joints()
        obj.dof_pos[:, 7] += .3
        torch.testing.assert_close(obj._reward_default_pos(), posture+.3)
        torch.testing.assert_close(obj._reward_symmetric_joints(), symmetric+.3)

    def test_reference_clearance_scores_only_swing_foot(self):
        obj = sample(ported)
        obj.episode_length_buf[:] = 20  # phase .25: left support, right swing
        obj.rigid_state[:, :4, 2] = .02
        obj.rigid_state[:, 3, 2] = .08
        torch.testing.assert_close(obj._reward_feet_clearance(), torch.ones(8))
        obj.rigid_state[:, 2, 2] = .5  # stance-foot height cannot add clearance reward
        torch.testing.assert_close(obj._reward_feet_clearance(), torch.ones(8))
        obj.rigid_state[:, 3, 2] = .02
        torch.testing.assert_close(obj._reward_feet_clearance(), torch.full((8,), .54881164))

    def test_contact_counts_one_foot_without_requiring_alternation(self):
        obj = sample(ported)
        obj.contact_forces.zero_()
        obj.contact_forces[:4, 2, 2] = 2.
        obj.contact_forces[4:6, 2:4, 2] = 2.
        expected = torch.tensor([1.,1.,1.,1.,0.,0.,0.,0.])
        torch.testing.assert_close(obj._reward_contact().float(), expected)
        obj.episode_length_buf += 40
        torch.testing.assert_close(obj._reward_contact().float(), expected)

    def test_original_half_cycle_boundary_is_preserved(self):
        obj = sample(ported)
        obj._get_phase = lambda: torch.tensor([0., .25, .5, .75, 0., .25, .5, .75])
        torch.testing.assert_close(obj._get_gait_phase(),
                                   torch.tensor([[1.,0.],[1.,0.],[0.,0.],[0.,1.]]).repeat(2,1))

    def test_stage_configuration_uses_only_reference_terms(self):
        path = ROOT / "ry_go2_biped_walk_v20/gait_env_cfg.py"
        fn = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == "configure_stage")
        expected = {k:v for k,v in reference["reward_scales"].items() if v}
        namespace = {"ReferenceRewardsCfg": lambda: SimpleNamespace(
            **{k: SimpleNamespace(weight=v) for k,v in expected.items()})}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
        cfg = SimpleNamespace(commands=SimpleNamespace(base_velocity=SimpleNamespace(ranges=SimpleNamespace())))
        for stage in ("balance", "walk", "balance"):
            namespace["configure_stage"](cfg, stage)
            self.assertEqual({k:v.weight for k,v in vars(cfg.rewards).items()}, expected)
            self.assertEqual(len(vars(cfg.rewards)), 23)
            cmd = cfg.commands.base_velocity
            self.assertEqual(cmd.ranges.lin_vel_x, (.05,.10) if stage=="balance" else (.2,.2))
            self.assertEqual(cmd.rel_standing_envs, .5 if stage=="balance" else 0.)
            self.assertEqual(cmd.resampling_time_range, (10.,10.))
            self.assertFalse(cmd.heading_command)
            self.assertEqual(cfg.balance_gait_clock, "always")

    def test_tracking_uses_each_environment_command(self):
        obj = sample(ported)
        obj.commands.zero_()
        obj.commands[:, 0] = torch.tensor([0., .05, .075, .1, 0., .05, .075, .1])
        obj.base_lin_vel.zero_()
        obj.base_lin_vel[:, 2] = -obj.commands[:, 0]
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), torch.ones(8))
        obj.base_lin_vel.zero_()
        torch.testing.assert_close(obj._reward_tracking_lin_vel(),
                                   torch.exp(-obj.commands[:, 0].square() / .25))

    def test_reward_constants_are_extracted_from_source(self):
        tree = ast.parse((ROOT / "third_party/My_unitree_go2_gym/Go2_Handstand_Config.py").read_text())
        cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2_Handstand_Cfg_Yu")
        rewards = next(n for n in cfg.body if isinstance(n, ast.ClassDef) and n.name == "rewards")
        scales = next(n for n in rewards.body if isinstance(n, ast.ClassDef) and n.name == "scales")
        expected = {n.targets[0].id: ast.literal_eval(n.value) for n in scales.body if isinstance(n, ast.Assign)}
        self.assertEqual(expected, reference["reward_scales"])
        self.assertEqual(23, sum(v != 0 for v in expected.values()))
        self.assertNotIn("feet_air_time", expected)

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
