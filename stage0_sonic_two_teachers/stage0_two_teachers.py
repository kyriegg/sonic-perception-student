"""Stage 0 / check 2: both teachers in ONE InstinctLab env (Isaac Lab 2.1).

  - terrain block : envs pinned to each paper terrain at a fixed level, driven by the InstinctLab parkour teacher
                    (its own observations: proprio history + depth camera)
  - flat block    : one env per SONIC reference motion on an extra flat tile, driven by SONIC (sonic_net.py)
                    through bridge.py (joint order + waist sign + SONIC action scale -> InstinctLab action space)

Both go through the same env.step(); the env keeps all its managers (events, terminations, command) — for flat-block
envs that the env resets (any termination / timeout) the robot is re-initialised to the motion's frame 0.

Compact terrain (requested sub-terrains + "flat", 1 column each, 10 rows) so it fits on an 8 GB laptop GPU.

Run from the InstinctLab repo root via run_instinct.ps1:
  python stage0_two_teachers.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run> --checkpoint=model_30000.pt \
      --envs_per_terrain 8 --steps 1000 --headless
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

PAPER_TERRAINS = [
    "perlin_rough", "square_gaps", "pyramid_stairs", "pyramid_stairs_high", "pyramid_stairs_inv",
    "pyramid_stairs_inv_high", "boxes", "mesh_boxes", "hf_pyramid_slope_inv",
]
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Instinct-Parkour-Target-Amp-G1-v0")
parser.add_argument("--terrains", nargs="+", default=PAPER_TERRAINS)
parser.add_argument("--envs_per_terrain", type=int, default=8)
parser.add_argument("--level", type=int, default=-1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--ref_dir", default=r"D:\robot paper\sonic_student\ref_example")
parser.add_argument("--sonic_weights", default=r"D:\robot paper\sonic_student\sonic_release_policy_sd.pt")
parser.add_argument("--out", default=r"D:\robot paper\sonic_student\results_stage0")
parser.add_argument("--video", action="store_true", help="record one mp4; the camera follows each --video_targets entry in turn")
parser.add_argument("--video_targets", nargs="+", default=[
    "sonic:walking_quip_360_R_002__A428", "sonic:dance_in_da_party_001__A464", "sonic:tired_one_leg_jumping_R_001__A359",
    "terrain:pyramid_stairs_inv", "terrain:square_gaps", "terrain:mesh_boxes", "terrain:hf_pyramid_slope_inv"])
parser.add_argument("--segment_steps", type=int, default=500, help="steps per camera target (50 Hz)")
cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.steps = len(args_cli.video_targets) * args_cli.segment_steps
app = AppLauncher(args_cli).app

import copy
import json

import gymnasium as gym
import numpy as np
import torch

from instinct_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge import SHOE_DZ, Bridge, MotionPlayer, load_motions  # noqa: E402
from sonic_net import ProprioHistory, SonicActor  # noqa: E402

FALL_TERMS = ("base_contact", "bad_orientation", "root_height")


def column_to_subterrain(gen_cfg):
    names = list(gen_cfg.sub_terrains.keys())
    props = np.array([c.proportion for c in gen_cfg.sub_terrains.values()], dtype=float)
    cum = np.cumsum(props / props.sum())
    return [names[int(np.min(np.where(j / gen_cfg.num_cols + 0.001 < cum)[0]))] for j in range(gen_cfg.num_cols)]


def main():
    motions = load_motions(args_cli.ref_dir)
    n_terr, E, M = len(args_cli.terrains), args_cli.envs_per_terrain, len(motions)
    num_envs = n_terr * E + M
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.curriculum.terrain_levels = None

    # ---- compact terrain: requested sub-terrains + one flat tile type (zero-amplitude perlin, no walls)
    gen = env_cfg.scene.terrain.terrain_generator
    flat = copy.deepcopy(gen.sub_terrains["perlin_rough"])
    flat.noise_scale = [0.0, 0.0]
    flat.wall_prob = [0.0, 0.0, 0.0, 0.0]
    gen.sub_terrains = {**{n: gen.sub_terrains[n] for n in args_cli.terrains}, "flat": flat}
    for c in gen.sub_terrains.values():
        c.proportion = 1.0
    if args_cli.video:  # like the upstream Play config: no 5 m perimeter walls, they block the follow camera
        for c in gen.sub_terrains.values():
            if hasattr(c, "wall_prob"):
                c.wall_prob = [0.0, 0.0, 0.0, 0.0]
    n_flat_cols = (M + gen.num_rows - 1) // gen.num_rows  # one flat tile per SONIC env (rows x flat columns)
    flat.proportion = float(n_flat_cols)
    gen.num_cols = n_terr + n_flat_cols
    cmd_cfg = env_cfg.commands.base_velocity
    vr = dict(cmd_cfg.velocity_ranges)
    vr["flat"] = vr["perlin_rough"]
    cmd_cfg.velocity_ranges = {k: v for k, v in vr.items() if k in gen.sub_terrains}
    if cmd_cfg.random_velocity_terrain is not None:
        cmd_cfg.random_velocity_terrain = [k for k in cmd_cfg.random_velocity_terrain if k in gen.sub_terrains]
    col_names = column_to_subterrain(gen)
    level = gen.num_rows - 1 if args_cli.level < 0 else args_cli.level

    agent_cfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    run_dir = args_cli.load_run
    resume_path = get_checkpoint_path(os.path.dirname(run_dir), os.path.basename(run_dir), agent_cfg.load_checkpoint)
    if args_cli.video:
        from isaaclab.envs import ViewerCfg
        env_cfg.viewer = ViewerCfg(eye=(2.6, 2.0, 2.0), lookat=(0.0, 0.0, -0.5), origin_type="asset_root",
                                   asset_name="robot", env_index=0, resolution=(1280, 720))
        genv = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        os.makedirs(os.path.join(args_cli.out, "videos"), exist_ok=True)
        genv = gym.wrappers.RecordVideo(genv, video_folder=os.path.join(args_cli.out, "videos"), name_prefix="stage0_two_teachers",
                                        step_trigger=lambda st: st == 0, video_length=args_cli.steps, disable_logger=True)
        env = InstinctRlVecEnvWrapper(genv)
    else:
        env = InstinctRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    teacher = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    dev = u.device
    terrain, robot, term_mgr = u.scene.terrain, u.scene["robot"], u.termination_manager
    cmd = u.command_manager.get_term("base_velocity")
    act_term = u.action_manager.get_term("joint_pos")

    # ---- pin envs: [terrain blocks ...][flat block (one per motion)]
    block = torch.arange(num_envs, device=dev) // E
    block[n_terr * E:] = n_terr
    names_all = list(args_cli.terrains) + ["flat"]
    types = torch.zeros(num_envs, dtype=torch.long, device=dev)
    for i, name in enumerate(names_all):
        cols = torch.tensor([j for j, n in enumerate(col_names) if n == name], device=dev)
        ids = (block == i).nonzero().flatten()
        types[ids] = cols[torch.arange(len(ids), device=dev) % len(cols)]
        v = cmd_cfg.velocity_ranges[name]
        cmd.lin_vel_x_range[ids] = torch.tensor(v["lin_vel_x"], device=dev)
        cmd.lin_vel_y_range[ids] = torch.tensor(v["lin_vel_y"], device=dev)
        cmd.ang_vel_z_range[ids] = torch.tensor(v["ang_vel_z"], device=dev)
        cmd.random_velocity_indices[ids] = name in (cmd_cfg.random_velocity_terrain or [])
    terrain.terrain_levels[:] = level
    # flat block: spread over all rows/columns of the flat tiles so every SONIC robot has its own tile
    flat_cols = torch.tensor([j for j, n in enumerate(col_names) if n == "flat"], device=dev)
    kk = torch.arange(M, device=dev)
    terrain.terrain_levels[n_terr * E:] = kk % gen.num_rows
    types[n_terr * E:] = flat_cols[kk // gen.num_rows]
    terrain.terrain_types[:] = types
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    sonic_ids = torch.arange(n_terr * E, num_envs, device=dev)
    terr_ids = torch.arange(n_terr * E, device=dev)
    print(f"[S0] {n_terr}x{E} teacher envs on {args_cli.terrains} level {level}; {M} SONIC envs on flat", flush=True)

    obs, _ = env.reset()

    # ---- SONIC side
    br = Bridge(robot, dev)
    player = MotionPlayer(motions, sonic_ids, terrain.env_origins[sonic_ids], dev)
    net = SonicActor.from_release(args_cli.sonic_weights, dev)
    hist = ProprioHistory(M, dev)
    last_act = torch.zeros(M, 29, device=dev)
    allM = torch.arange(M, device=dev)

    def sonic_reset(local):
        player.t[local] = 0
        player.write_reset(local, br, SHOE_DZ)
        u.scene.write_data_to_sim()
        u.sim.forward()
        u.scene.update(0.0)
        s = br.sonic_state(sonic_ids)
        hist.reset(local, s["ang_vel"], s["jpos"], s["jvel"], s["grav"])
        last_act[local] = 0.0

    sonic_reset(allM)

    # ---- bookkeeping
    stats = {n: {"episodes": 0, "falls": 0} for n in args_cli.terrains}
    s_err = torch.zeros(M, device=dev); s_cnt = torch.zeros(M, device=dev)
    s_done = torch.zeros(M, device=dev); s_fail = torch.zeros(M, device=dev); s_envreset = torch.zeros(M, device=dev)
    max_target_err = 0.0

    seg_log = []
    with torch.inference_mode():
        for step in range(args_cli.steps):
            if args_cli.video and step % args_cli.segment_steps == 0:
                kind, name = args_cli.video_targets[step // args_cli.segment_steps].split(":", 1)
                if kind == "sonic":
                    li = player.names.index(name)
                    idx = int(sonic_ids[li])
                    sonic_reset(torch.tensor([li], device=dev))
                else:
                    idx = int((block == args_cli.terrains.index(name)).nonzero()[0])
                u.viewport_camera_controller.set_view_env_index(idx)
                # terrain tiles at high levels are pits/walls: look from almost straight above there
                u.viewport_camera_controller.update_view_location(
                    eye=(2.6, 2.0, 2.0) if kind == "sonic" else (1.0, 0.8, 3.2), lookat=(0.0, 0.0, -0.5 if kind == "sonic" else -0.8))
                seg_log.append(f"{step * u.step_dt:6.1f}s - {(step + args_cli.segment_steps) * u.step_dt:6.1f}s  {kind:7s} {name}  (env {idx})")
                print("[S0] video segment " + seg_log[-1], flush=True)
            actions = teacher(obs).clone()
            # SONIC actions for the flat block
            s = br.sonic_state(sonic_ids)
            hist.push(s["ang_vel"], s["jpos"], s["jvel"], last_act, s["grav"])
            a_s = net(player.encoder_input(s["pelvis_quat"]), hist.proprio())
            last_act = a_s
            target_s = br.sonic_target(a_s)
            actions[sonic_ids] = br.instinct_action(target_s, act_term, sonic_ids)

            obs, _, dones, _ = env.step(actions)
            dones = dones.bool()

            # pathway check: the env's processed joint targets must equal the SONIC targets
            max_target_err = max(max_target_err, float((br.to_sonic(act_term.processed_actions[sonic_ids]) - target_s).abs().max()))

            # teacher block stats
            fell = torch.zeros(num_envs, dtype=torch.bool, device=dev)
            for n in FALL_TERMS:
                if n in term_mgr.active_terms:
                    fell |= term_mgr.get_term(n).bool()
            for i, name in enumerate(args_cli.terrains):
                m = (block == i) & dones
                stats[name]["episodes"] += int(m.sum()); stats[name]["falls"] += int((m & fell).sum())

            # SONIC block: advance clock, metrics, restarts
            player.t += 1
            sd = dones[sonic_ids]
            alive = ~sd & (player.t < player.len)
            loc, pz = br.body_pos_local(sonic_ids, SHOE_DZ)
            k = torch.clamp(player.t, max=player.len - 1)
            rloc, rz = player.ref_local(k)
            e = (loc - rloc).norm(dim=-1).mean(-1)
            fail = alive & ((pz - rz).abs() > 0.25)
            ok = alive & ~fail
            s_err += e * ok; s_cnt += ok
            finished = alive & (player.t >= player.len - 1)
            s_done += finished; s_fail += fail; s_envreset += sd
            restart = (sd | fail | finished | (player.t >= player.len)).nonzero().flatten()
            if len(restart):
                sonic_reset(restart)
            if step % 100 == 0:
                tf = sum(v["falls"] for v in stats.values()); te = sum(v["episodes"] for v in stats.values())
                print(f"[S0] step {step}: teacher episodes {te} falls {tf} | SONIC mpjpe_l {1000 * float((s_err / s_cnt.clamp(min=1)).mean()):.1f} mm "
                      f"completed {int(s_done.sum())} fails {int(s_fail.sum())} env-resets {int(s_envreset.sum())} | target err {max_target_err:.1e}", flush=True)

    res = {"checkpoint": resume_path, "level": level, "steps": args_cli.steps, "envs_per_terrain": E,
           "action_pathway_max_target_err": max_target_err,
           "teacher": {n: {**v, "fall_rate": v["falls"] / max(v["episodes"], 1)} for n, v in stats.items()},
           "sonic": {player.names[i]: {"mpjpe_l_mm": float(1000 * s_err[i] / max(s_cnt[i], 1)), "completed": int(s_done[i]),
                                       "fails": int(s_fail[i]), "env_resets": int(s_envreset[i])} for i in range(M)}}
    os.makedirs(args_cli.out, exist_ok=True)
    fn = os.path.join(args_cli.out, f"stage0_two_teachers_L{level}.json")
    json.dump(res, open(fn, "w"), indent=1)
    print("[S0] saved " + fn, flush=True)
    if args_cli.video:
        with open(os.path.join(args_cli.out, "videos", "segments.txt"), "w", encoding="utf-8") as f:
            f.write(chr(10).join(seg_log) + chr(10))
        try:
            env.close()  # flush the RecordVideo writer
        except Exception as e:  # upstream shutdown error (points_visualizer) after the video is written
            print(f"[S0] env.close: {e!r}", flush=True)
    print("[S0] RESULT " + json.dumps(res), flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    os._exit(0)
