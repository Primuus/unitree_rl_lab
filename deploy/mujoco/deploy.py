"""View the stage1 Go2 balance actor in MuJoCo; R resets, Space pauses."""
import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import threading
import time
import sys

import mujoco
import numpy as np

from model import load_model
from observation import build_observation, heading
from policy import Policy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from ry_unitree_rl_lab.utils.balance_acceptance import assess_balance
from ry_unitree_rl_lab.utils.phase_contract import read_gait_clock


@contextmanager
def open_viewer(model, data, key_callback):
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    try:
        yield viewer
    finally:
        # Same lifecycle handling as quadruped: close() only signals the render thread.
        viewer.close()
        deadline = time.monotonic()+5
        while viewer._sim() is not None:
            if time.monotonic() >= deadline:
                raise RuntimeError("MuJoCo viewer did not finish closing")
            time.sleep(.005)


class Simulation:
    def __init__(self, config, policy_path):
        config = dict(config)
        if config["gait_clock_mode"] == "auto":
            clock = read_gait_clock(Path(policy_path).resolve().parents[1]/"params/provenance.json")
            config["gait_clock_mode"], config["gait_period"] = clock["mode"], clock["period_s"]
        if config["gait_clock_mode"] not in ("always", "moving_only"):
            raise ValueError("gait_clock_mode must be auto, always or moving_only")
        self.cfg = config
        self.model = load_model(config, ROOT)
        self.data = mujoco.MjData(self.model)
        self.policy = Policy(policy_path)
        self.base = self.model.body("base").id
        names = config["action_names"]
        self.jids = np.array([self.model.joint(n).id for n in names])
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in names])
        self.vids = np.array([self.model.joint(n).dofadr[0] for n in names])
        self.velocity_limits = self.model.jnt_user[self.jids, 0]
        self.aids = np.array([self.model.actuator(n).id for n in names])
        self.feet = [self.model.site(n+"_foot").id for n in ["FL", "FR", "RL", "RR"]]
        self.foot_geoms = {self.model.geom(n+"_foot_collision_0").id: i
                           for i, n in enumerate(["FL", "FR", "RL", "RR"])}
        self.contact_force = np.zeros(6)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.cfg["initial_position"]
        self.data.qpos[3:7] = self.cfg["initial_quaternion"]
        self.data.qpos[self.qids] = self.cfg["initial_joint_pos"]
        self.last_action.fill(0)
        self.target = np.asarray(self.cfg["initial_joint_pos"])
        self.steps = 0
        self.velocity_limit_applications = 0
        self.records = []
        mujoco.mj_forward(self.model, self.data)

    def forces(self):
        forces = np.zeros((4, 3))
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 not in self.foot_geoms and contact.geom2 not in self.foot_geoms:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, self.contact_force)
            world = contact.frame.reshape(3, 3).T @ self.contact_force[:3]
            if contact.geom1 in self.foot_geoms:
                forces[self.foot_geoms[contact.geom1]] -= world
            if contact.geom2 in self.foot_geoms:
                forces[self.foot_geoms[contact.geom2]] += world
        return forces

    def observation(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        velocity = np.zeros(6)
        # BODY velocity is centered at the body's COM, as Isaac Lab root_lin_vel_w.
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.base, velocity, 0)
        return build_observation(
            rotation, velocity[3:], rotation.T @ velocity[:3],
            self.data.qpos[self.qids], self.data.qvel[self.vids], self.last_action,
            self.data.site_xpos[self.feet[2:], 2], self.forces()[2:, 2], self.data.time, self.cfg)

    def base_contact(self):
        """Match the training base-contact failure threshold in world coordinates."""
        total = np.zeros(3)
        force = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            first = self.model.geom_bodyid[contact.geom1] == self.base
            second = self.model.geom_bodyid[contact.geom2] == self.base
            if first or second:
                mujoco.mj_contactForce(self.model, self.data, i, force)
                total += (int(second)-int(first)) * (contact.frame.reshape(3, 3).T @ force[:3])
        return bool(np.linalg.norm(total) > 1.0)

    def step(self):
        if self.steps % self.cfg["decimation"] == 0:
            obs = self.observation()
            clip = self.cfg["clip_actions"]
            self.last_action = np.clip(self.policy(obs), -clip, clip)
            self.target = np.asarray(self.cfg["action_offset"]) + self.cfg["action_scale"]*self.last_action
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                    self.base, velocity, 0)
            self.records.append({"time": float(self.data.time),
                                 "position": self.data.qpos[:3].tolist(),
                                 "upright_cos": float(-obs[6]),
                                 "forward_speed": float(obs[0]),
                                 "heading_yaw_rad": float(heading(self.data.xmat[self.base].reshape(3, 3))),
                                 "yaw_rate_world_radps": float(velocity[2]),
                                 "base_contact": self.base_contact(),
                                 "rear_load": obs[54:56].tolist(),
                                 "front_contact": bool((np.linalg.norm(self.forces()[:2], axis=1)>1).any())})
        torque = (self.target-self.data.qpos[self.qids])*self.cfg["kp"] - self.data.qvel[self.vids]*self.cfg["kd"]
        limits = self.model.actuator_ctrlrange[self.aids]
        self.data.ctrl[self.aids] = np.clip(torque, limits[:, 0], limits[:, 1])
        mujoco.mj_step(self.model, self.data)
        # Approximate the URDF velocity bounds enforced by PhysX during training.
        # implicitfast integrates position with the new velocity: project both
        # joint velocity and its position increment, leaving the free base alone.
        velocity = self.data.qvel[self.vids]
        bounded = np.clip(velocity, -self.velocity_limits, self.velocity_limits)
        self.velocity_limit_applications += int(np.count_nonzero(velocity != bounded))
        self.data.qpos[self.qids] += (bounded-velocity)*self.model.opt.timestep
        self.data.qvel[self.vids] = bounded
        # Refresh state-derived values at the integrated qpos/qvel for the next policy call.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Non-finite MuJoCo state")

    def result(self):
        r = self.records
        upright = [v["upright_cos"] > .94 and v["position"][2] > .43 for v in r]
        # Viewer continues after failure, but acceptance stops at the first base contact.
        failure = next((i for i, v in enumerate(r) if v["base_contact"]), len(r))
        valid = r[:failure]
        seconds = min(len(valid)*self.cfg["physics_dt"]*self.cfg["decimation"], float(self.data.time))
        yaw = np.unwrap([v["heading_yaw_rad"] for v in valid])
        xy = np.array([v["position"][:2] for v in valid])
        path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(valid)>1 else 0.
        excursion = float(np.max(np.abs(yaw-yaw[0]))) if valid else 0.
        distance = float(np.linalg.norm(xy-xy[0], axis=1).max()) if valid else 0.
        values = dict(valid_seconds=seconds,failures=int(failure<len(r)),
            upright_fraction=float(np.mean(upright[:failure])) if valid else 0.,
            front_contact_fraction=float(np.mean([v["front_contact"] for v in valid])) if valid else 0.,
            forward_error_mps=float(np.mean([abs(v["forward_speed"]-self.cfg["command"][0]) for v in valid])) if valid else 0.,
            mean_abs_yaw_rate_radps=float(np.mean([abs(v["yaw_rate_world_radps"]) for v in valid])) if valid else 0.,
            max_heading_excursion_rad=excursion,max_horizontal_displacement_m=distance,
            path_speed_mps=path/seconds if seconds else 0.)
        acceptance = assess_balance(values, self.cfg["stage"] == "balance" and not any(self.cfg["command"]))
        return {"simulation_seconds": float(self.data.time), "policy_steps": len(r),
                "gait_clock_mode": self.cfg["gait_clock_mode"], "gait_period_s": self.cfg["gait_period"],
                "balance_acceptance_passed": acceptance["passed"], "balance_acceptance": acceptance,
                "horizontal_path_length_m": path, "max_horizontal_displacement_m": distance,
                "net_heading_change_rad": float(yaw[-1]-yaw[0]) if valid else 0.,
                "max_heading_excursion_rad": excursion,
                "absolute_heading_travel_rad": float(np.abs(np.diff(yaw)).sum()),
                "joint_velocity_limit_mode": "project_velocity_and_position",
                "velocity_limit_applications": self.velocity_limit_applications,
                "upright_fraction": float(np.mean(upright)) if r else 0.,
                "front_contact_fraction": float(np.mean([v["front_contact"] for v in r])) if r else 0.,
                "mean_abs_forward_speed": float(np.mean([abs(v["forward_speed"]) for v in r])) if r else 0.,
                "final_position": self.data.qpos[:3].tolist(), "finite": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0, help="Simulation seconds; 0 runs until closed")
    parser.add_argument("--output", type=Path, help="Save summary and trace JSON")
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("duration must be nonnegative")
    cfg = json.loads(args.config.read_text())
    policy = args.policy or ROOT/cfg["policy"]
    sim = Simulation(cfg, policy)
    reset, pause = threading.Event(), threading.Event()

    def key(code):
        if code in (ord('R'), ord('r')):
            reset.set()
        elif code == 32:
            pause.clear() if pause.is_set() else pause.set()

    context = nullcontext(None)
    if not args.headless:
        context = open_viewer(sim.model, sim.data, key)
    print(f"policy={policy}\n56 observations / 12 actions; PD={cfg['kp']}/{cfg['kd']}; "
          f"control_dt={cfg['physics_dt']*cfg['decimation']}s; mass={sim.model.body_mass.sum():.3f}kg; "
          f"gait_clock={sim.cfg['gait_clock_mode']}", flush=True)
    start = time.monotonic()
    next_display = start
    next_status = 1.
    try:
        with context as viewer:
            if viewer:
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -15
                viewer.cam.lookat[:] = [0, 0, .5]
            while (args.duration == 0 or sim.data.time < args.duration-1e-9) and (viewer is None or viewer.is_running()):
                if reset.is_set():
                    sim.reset()
                    reset.clear()
                    start, next_status = time.monotonic(), 1.
                if pause.is_set():
                    start = time.monotonic()-sim.data.time
                    if viewer:
                        viewer.sync()
                    time.sleep(.02)
                    continue
                sim.step()
                now = time.monotonic()
                if viewer and now >= next_display:
                    viewer.cam.lookat[:] = sim.data.qpos[:3]
                    viewer.sync()
                    next_display = now + 1/60
                if sim.data.time >= next_status:
                    r = sim.records[-1]
                    print(f"t={sim.data.time:.1f}s height={sim.data.qpos[2]:.3f}m "
                          f"upright={r['upright_cos']:.3f} vx={r['forward_speed']:.3f}m/s", flush=True)
                    next_status += 1
                if viewer:
                    time.sleep(max(0, start+sim.data.time-time.monotonic()))
    except KeyboardInterrupt:
        pass
    result = sim.result()
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": result, "trace": sim.records}, indent=2))


if __name__ == "__main__":
    main()
