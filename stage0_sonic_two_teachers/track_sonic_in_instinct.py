"""Stage 0 / check 1: run the released SONIC tracker (PyTorch re-implementation, sonic_net.py) inside the
InstinctLab (Isaac Lab 2.1) simulator on flat ground and measure tracking error on the 13 deploy example motions.

One env per motion, all played from frame 0 in parallel, robot initialised to the reference frame-0 state.

Robot variants (--robot):
  instinct : teacher's robot — torso-rooted G1 with shoes + beyondmimic delayed actuators (the target setup)
  noshoe   : same URDF without shoes, same delayed actuators
Actuator override (--actuators):
  robot    : keep the variant's actuators
  sonic    : SONIC's implicit PD values per joint (differs from InstinctLab only at hip pitch) and no delay

Metrics per motion (until failure):
  mpjpe_l_mm : mean over frames/bodies of || (p_body - p_pelvis) rotated into own pelvis heading frame ||, robot vs ref, 14 bodies
  jpe_deg    : mean abs joint angle error
  failed     : |pelvis_z(robot) - pelvis_z(ref)| > 0.25 m at any time (BeyondMimic-style anchor-height termination)

Run from the InstinctLab repo root via run_instinct.ps1:
  python track_sonic_in_instinct.py --robot instinct --actuators robot --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=["instinct", "noshoe"], default="instinct")
parser.add_argument("--actuators", choices=["robot", "sonic"], default="robot")
parser.add_argument("--mode", choices=["policy", "replay"], default="policy",
                    help="replay = open-loop PD on the reference joint angles (sanity check of the setup)")
parser.add_argument("--ref_joint_order", choices=["bfs", "dfs"], default="bfs")
parser.add_argument("--waist_sign", type=float, nargs=3, default=[-1.0, -1.0, -1.0],
                    help="sign of (waist_yaw, waist_roll, waist_pitch) robot joint value relative to SONIC convention; the torso-rooted InstinctLab URDF reverses the waist chain, so all three are flipped (verified: reset MPJPE-L 70mm -> 1.5mm)")
parser.add_argument("--probe_signs", action="store_true", help="try all 8 waist sign combos at reset and exit")
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--ref_dir", default=r"D:\robot paper\sonic_student\ref_example")
parser.add_argument("--weights", default=r"D:\robot paper\sonic_student\sonic_release_policy_sd.pt")
parser.add_argument("--out", default=r"D:\robot paper\sonic_student\results_track")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json
import os
import sys

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from instinctlab.tasks.parkour.config.g1 import g1_parkour_target_amp_cfg as g1cfg
from instinctlab.assets import unitree_g1 as ug1

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sonic_net import (  # noqa: E402
    DFS_JOINTS, FUTURE_STEP, NUM_FUTURE, REF_BODIES, SONIC_JOINTS, ProprioHistory, SonicActor, build_g1_encoder_input,
    quat_inv, quat_mul, quat_rotate_inverse,
)

SHOE_DZ = 0.058  # g1 with shoe: feet_at_plane height_offset in the InstinctLab config
DEV = "cuda:0"
DT = 0.02
DECIM = 4


def load_motion(d):
    rd = lambda n: np.loadtxt(os.path.join(d, n), delimiter=",", skiprows=1, dtype=np.float32)
    jp, jv = rd("joint_pos.csv"), rd("joint_vel.csv")
    T = jp.shape[0]
    return dict(
        jp=jp, jv=jv, bpos=rd("body_pos.csv").reshape(T, 14, 3), bquat=rd("body_quat.csv").reshape(T, 14, 4),
        blin=rd("body_lin_vel.csv").reshape(T, 14, 3), bang=rd("body_ang_vel.csv").reshape(T, 14, 3), T=T,
    )


def sonic_actuators():
    s = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "act_sonic.json")))
    import re
    vel = {}
    for a in ug1.beyondmimic_g1_29dof_delayed_actuators.values():  # keep InstinctLab velocity limits
        v = a.velocity_limit_sim
        for j in SONIC_JOINTS:
            if any(re.fullmatch(e, j) for e in a.joint_names_expr):
                vel[j] = v if not isinstance(v, dict) else next(val for e, val in v.items() if re.fullmatch(e, j))
    return {"all": ImplicitActuatorCfg(
        joint_names_expr=list(SONIC_JOINTS),
        stiffness={j: s[j]["stiffness"] for j in SONIC_JOINTS},
        damping={j: s[j]["damping"] for j in SONIC_JOINTS},
        armature={j: s[j]["armature"] for j in SONIC_JOINTS},
        effort_limit_sim={j: s[j]["effort_limit_sim"] for j in SONIC_JOINTS},
        velocity_limit_sim=vel,
    )}


def main():
    names = sorted(d for d in os.listdir(args.ref_dir) if os.path.isdir(os.path.join(args.ref_dir, d)))
    motions = [load_motion(os.path.join(args.ref_dir, n)) for n in names]
    N, Tmax = len(names), max(m["T"] for m in motions)

    robot_cfg = (g1cfg.G1_with_shoe_CFG if args.robot == "instinct" else g1cfg.G1_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.actuators = sonic_actuators() if args.actuators == "sonic" else ug1.beyondmimic_g1_29dof_delayed_actuators
    dz = SHOE_DZ if args.robot == "instinct" else 0.0
    terrain_mat = g1cfg.G1ParkourEnvCfg().scene.terrain.physics_material

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", physics_material=terrain_mat)
        robot = robot_cfg
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=750.0))

    sim_cfg = sim_utils.SimulationCfg(dt=DT / DECIM, render_interval=DECIM, device=DEV, physics_material=terrain_mat)
    sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=N, env_spacing=4.0))
    sim.reset()
    robot = scene["robot"]

    # ---- index maps
    perm_r2s = torch.tensor([robot.joint_names.index(j) for j in SONIC_JOINTS], device=DEV)  # robot -> sonic order
    perm_s2r = torch.tensor([SONIC_JOINTS.index(j) for j in robot.joint_names], device=DEV)  # sonic -> robot order
    sgn_s = torch.ones(29, device=DEV)  # robot_value = sgn * sonic_value (per SONIC joint)
    for j, v in zip(("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"), args.waist_sign):
        sgn_s[SONIC_JOINTS.index(j)] = v
    to_robot = lambda x_s: (x_s * sgn_s)[..., perm_s2r]
    to_sonic = lambda x_r: x_r[..., perm_r2s] * sgn_s
    bidx = torch.tensor([robot.body_names.index(b) for b in REF_BODIES], device=DEV)
    pelvis = robot.body_names.index("pelvis")
    torso_ref = REF_BODIES.index("torso_link")
    default_s = to_sonic(robot.data.default_joint_pos)
    s_act = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "act_sonic.json")))
    # SONIC was trained with its own action scale; the policy output is always mapped with SONIC's scale
    scale_s = torch.tensor([s_act[j]["scale"] for j in SONIC_JOINTS], device=DEV)

    # ---- padded reference tensors [N, Tmax, ...] (pad with last frame)
    def pad(key):
        arr = np.stack([np.concatenate([m[key], np.repeat(m[key][-1:], Tmax - m["T"], 0)]) for m in motions])
        return torch.tensor(arr, device=DEV)
    R = {k: pad(k) for k in ("jp", "jv", "bpos", "bquat", "blin", "bang")}
    if args.ref_joint_order == "dfs":  # convert reference joints to SONIC (BFS) order
        p = torch.tensor([DFS_JOINTS.index(j) for j in SONIC_JOINTS], device=DEV)
        R["jp"], R["jv"] = R["jp"][..., p], R["jv"][..., p]
    lengths = torch.tensor([m["T"] for m in motions], device=DEV)
    origins = scene.env_origins
    # move each reference so that its frame-0 pelvis xy sits on the env origin
    R["bpos"] = R["bpos"] - torch.cat([R["bpos"][:, :1, :1, :2], torch.zeros_like(R["bpos"][:, :1, :1, :1])], -1) \
        + torch.cat([origins[:, None, None, :2], torch.zeros(N, 1, 1, 1, device=DEV)], -1)

    # ---- reset robot to reference frame 0 (root = torso_link in this URDF)
    ids = torch.arange(N, device=DEV)
    root_pose = torch.cat([R["bpos"][:, 0, torso_ref] + torch.tensor([0, 0, dz], device=DEV), R["bquat"][:, 0, torso_ref]], -1)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.cat([R["blin"][:, 0, torso_ref], R["bang"][:, 0, torso_ref]], -1))
    robot.write_joint_state_to_sim(to_robot(R["jp"][:, 0]), to_robot(R["jv"][:, 0]))
    robot.reset()
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)

    net = SonicActor.from_release(args.weights, DEV)
    hist = ProprioHistory(N, DEV)
    down = torch.tensor([0.0, 0.0, -1.0], device=DEV).expand(N, 3)

    def state():
        q = robot.data.body_quat_w[:, pelvis]
        w = robot.data.body_ang_vel_w[:, pelvis]
        return (quat_rotate_inverse(q, w), to_sonic(robot.data.joint_pos) - default_s,
                to_sonic(robot.data.joint_vel), quat_rotate_inverse(q, down), q)

    av, jp, jv, g, _ = state()
    hist.reset(ids, av, jp, jv, g)
    last_act = torch.zeros(N, 29, device=DEV)

    def heading_quat(q):
        w, x, y, z = q.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)

    def loc(b, q):  # bodies relative to pelvis, in the pelvis heading frame
        return quat_rotate_inverse(heading_quat(q)[:, None].expand(-1, b.shape[1], -1), b - b[:, :1])

    err_sum = torch.zeros(N, device=DEV); jerr_sum = torch.zeros(N, device=DEV); cnt = torch.zeros(N, device=DEV)
    failed = torch.zeros(N, dtype=torch.bool, device=DEV); fail_t = torch.full((N,), -1, device=DEV)
    future = torch.arange(NUM_FUTURE, device=DEV) * FUTURE_STEP

    def frame_err(k):
        rb = R["bpos"][ids, k]; rq = R["bquat"][ids, k, 0]
        sb = robot.data.body_pos_w[:, bidx] - torch.tensor([0, 0, dz], device=DEV); sq = robot.data.body_quat_w[:, pelvis]
        e = (loc(sb, sq) - loc(rb, rq)).norm(dim=-1).mean(-1)
        je = (to_sonic(robot.data.joint_pos) - R["jp"][ids, k]).abs().mean(-1)
        return rb, sb, e, je

    if args.probe_signs:
        import itertools
        for combo in itertools.product([1.0, -1.0], repeat=3):
            for j, v in zip(("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"), combo):
                sgn_s[SONIC_JOINTS.index(j)] = v
            robot.write_root_pose_to_sim(root_pose)
            robot.write_joint_state_to_sim(to_robot(R["jp"][:, 0]), to_robot(R["jv"][:, 0]))
            scene.write_data_to_sim(); sim.forward(); scene.update(0.0)
            rb, sb, e, je = frame_err(torch.zeros(N, dtype=torch.long, device=DEV))
            pb = (sb - rb).norm(dim=-1).mean(0) * 1000
            print(f"[PROBE] waist sign (yaw,roll,pitch)={combo}: mpjpe_l {1000 * e.mean():.1f} mm  pelvis {pb[0]:.1f}  ankles {pb[3]:.1f}/{pb[6]:.1f} mm", flush=True)
        return
    rb, sb, e, je = frame_err(torch.zeros(N, dtype=torch.long, device=DEV))
    pb = (sb - rb).norm(dim=-1).mean(0) * 1000
    print("[TRACK] per-body world pos err after reset (mm): " + ", ".join(f"{b.replace('_link','')} {v:.0f}" for b, v in zip(REF_BODIES, pb.tolist())), flush=True)
    print(f"[TRACK] after reset: mpjpe_l {1000 * e.mean():.1f} mm (max {1000 * e.max():.1f}), joint err {np.degrees(float(je.mean())):.2f} deg, "
          f"pelvis z err {1000 * (sb[:, 0, 2] - rb[:, 0, 2]).abs().max():.1f} mm, pelvis xy err {1000 * (sb[:, 0, :2] - rb[:, 0, :2]).norm(dim=-1).max():.1f} mm", flush=True)
    with torch.inference_mode():
        for t in range(min(Tmax - 1, args.max_steps or Tmax)):
            av, jp, jv, g, pq = state()
            if t > 0:
                hist.push(av, jp, jv, last_act, g)
            fi = torch.minimum(t + future[None, :], lengths[:, None] - 1)  # [N, 10]
            gi = lambda x: x[ids[:, None], fi]
            enc_in = build_g1_encoder_input(gi(R["jp"]), gi(R["jv"]), gi(R["bquat"])[:, :, 0], pq)
            if args.mode == "policy":
                act = net(enc_in, hist.proprio())
                target_s = default_s + act * scale_s
            else:
                act = torch.zeros(N, 29, device=DEV)
                target_s = R["jp"][ids, torch.clamp(torch.full((N,), t + 1, device=DEV), max=lengths - 1)]
            last_act = act
            robot.set_joint_position_target(to_robot(target_s))
            for _ in range(DECIM):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(DT / DECIM)
            # ---- metrics against reference frame t+1
            k = torch.clamp(torch.full((N,), t + 1, device=DEV), max=lengths - 1)
            active = (t + 1 < lengths) & ~failed
            rb, sb, e, je = frame_err(k)
            if t < 5 or t % 50 == 0:
                print(f"[TRACK]   t={t + 1}: mpjpe_l {1000 * e.mean():.1f} mm, joint err {np.degrees(float(je.mean())):.2f} deg, "
                      f"pelvis z err mean {1000 * (sb[:, 0, 2] - rb[:, 0, 2]).abs().mean():.1f} mm, gravity_z {float(g[:, 2].mean()):.3f}", flush=True)
            fall = (sb[:, 0, 2] - rb[:, 0, 2]).abs() > 0.25
            newly = fall & active
            failed |= newly; fail_t[newly] = t + 1
            act_m = active & ~newly
            err_sum += e * act_m; jerr_sum += je * act_m; cnt += act_m
            if t % 100 == 0:
                print(f"[TRACK] step {t}/{Tmax}  mean mpjpe_l {1000 * (err_sum / cnt.clamp(min=1)).mean():.1f} mm  failed {int(failed.sum())}", flush=True)

    res = {}
    for i, n in enumerate(names):
        res[n] = {"frames": int(lengths[i]), "mpjpe_l_mm": float(1000 * err_sum[i] / max(cnt[i], 1)),
                  "jpe_deg": float(np.degrees(float(jerr_sum[i] / max(cnt[i], 1)))),
                  "failed": bool(failed[i]), "fail_frame": int(fail_t[i])}
    tag = f"{args.robot}_{args.actuators}_{args.mode}"
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, f"track_{tag}.json"), "w"), indent=1)
    print(f"[TRACK] RESULT {tag}")
    for n, r in res.items():
        print(f"[TRACK] {n:40s} frames {r['frames']:4d}  MPJPE-L {r['mpjpe_l_mm']:6.1f} mm  joint err {r['jpe_deg']:5.2f} deg  {'FAIL@' + str(r['fail_frame']) if r['failed'] else 'ok'}")
    print(f"[TRACK] mean MPJPE-L {np.mean([r['mpjpe_l_mm'] for r in res.values()]):.1f} mm, success {sum(not r['failed'] for r in res.values())}/{len(res)}")


if __name__ == "__main__":
    main()
    os._exit(0)  # app.close() hangs on this machine after headless runs
