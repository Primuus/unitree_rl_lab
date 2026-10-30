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
    obj.cfg = SimpleNamespace(rewards=SimpleNamespace(
        **reference["reward_parameters"], tracking_lin_vel_sigma=reference["reward_parameters"]["tracking_sigma"]))
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
        obj.cfg.rewards.tracking_lin_vel_sigma = .04
        expected = torch.tensor([1., .939413, .778801, .367879, .367879, .105399, 1., .367879])
        torch.testing.assert_close(obj._reward_tracking_lin_vel(), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(obj._reward_tracking_ang_vel(), original_angular)
        obj.root_states[-1, 2] = 0.
        obj._reward_base_height()
        self.assertEqual(obj._reward_tracking_lin_vel()[-1].item(), 0.)

    def test_phase_boundary_always_has_one_stance(self):
        obj = sample(ported)
        obj.episode_length_buf = torch.tensor([0, 40, 80, 120, 160, 200, 240, 280])
        torch.testing.assert_close(obj._get_gait_phase().sum(-1), torch.ones(8))

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
