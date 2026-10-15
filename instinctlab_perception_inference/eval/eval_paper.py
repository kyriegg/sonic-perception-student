"""Paper-style per-terrain evaluation (Hiking in the Wild, Table I / II protocol as far as the paper specifies it).

Each listed terrain type gets its own block of robots (default 1000), all pinned to one difficulty level
(default: hardest row) with the terrain curriculum disabled, run for a fixed number of steps (default 10000).
Everything else is the unmodified training task config.

Metrics per terrain:
  - episode success rate = 1 - (episodes ending in base_contact / bad_orientation / root_height) / episodes
  - robot success rate   = robots that never fell during the whole run / robots
  - goal reach rate      = targets reached (within target_dis_threshold) before being replaced / targets
  - mean reaching time   = seconds from target assignment to first arrival, over reached targets
  (standing-command targets and targets already within the threshold at assignment are excluded from reach stats)

Run from the InstinctLab repo root:
  python eval_paper.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=/abs/run --checkpoint=model_30000.pt \
      --terrains boxes mesh_boxes hf_pyramid_slope_inv --envs_per_terrain 1000 --steps 10000 --headless
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Paper-style per-terrain evaluation.")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--terrains", nargs="+", required=True, help="Sub-terrain names (keys of the terrain generator).")
parser.add_argument("--envs_per_terrain", type=int, default=1000)
parser.add_argument("--level", type=int, default=-1, help="Terrain row/difficulty; -1 = hardest.")
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--out", type=str, default=None)
cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import gymnasium as gym
import torch

from instinct_rl.runners import OnPolicyRunner

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

FALL_TERMS = ("base_contact", "bad_orientation", "root_height")


def column_to_subterrain(gen_cfg):
    """Same column -> sub-terrain mapping as IsaacLab TerrainGenerator (curriculum mode) and PoseVelocityCommand."""
    names = list(gen_cfg.sub_terrains.keys())
    props = np.array([c.proportion for c in gen_cfg.sub_terrains.values()], dtype=float)
    props /= props.sum()
    cum = np.cumsum(props)
    return [names[int(np.min(np.where(j / gen_cfg.num_cols + 0.001 < cum)[0]))] for j in range(gen_cfg.num_cols)]


def main():
    n_terr = len(args_cli.terrains)
    num_envs = n_terr * args_cli.envs_per_terrain
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.curriculum.terrain_levels = None  # pin robots to the chosen level
    gen_cfg = env_cfg.scene.terrain.terrain_generator
    col_names = column_to_subterrain(gen_cfg)
    for name in args_cli.terrains:
        assert name in col_names, f"unknown terrain {name}; available: {sorted(set(col_names))}"
    level = gen_cfg.num_rows - 1 if args_cli.level < 0 else args_cli.level

    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    run_dir = args_cli.load_run
    resume_path = get_checkpoint_path(os.path.dirname(run_dir), os.path.basename(run_dir), agent_cfg.load_checkpoint)
    out_dir = args_cli.out or os.path.join(run_dir, "eval_paper")
    os.makedirs(out_dir, exist_ok=True)

    env = InstinctRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[EVAL] loading {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    dev = uenv.device
    terrain = uenv.scene.terrain
    robot = uenv.scene["robot"]
    term_mgr = uenv.termination_manager
    cmd = uenv.command_manager.get_term("base_velocity")
    dt = uenv.step_dt

    # ---- pin each block of robots to one terrain type (spread over that type's columns) at `level`
    block = torch.arange(num_envs, device=dev) // args_cli.envs_per_terrain  # terrain index per env
    types = torch.zeros(num_envs, dtype=torch.long, device=dev)
    for i, name in enumerate(args_cli.terrains):
        cols = torch.tensor([j for j, n in enumerate(col_names) if n == name], device=dev)
        ids = (block == i).nonzero().flatten()
        types[ids] = cols[torch.arange(len(ids), device=dev) % len(cols)]
        # velocity ranges are assigned once in the command's __init__ from the initial terrain type: redo them
        vr = cmd.cfg.velocity_ranges[name]
        cmd.lin_vel_x_range[ids] = torch.tensor(vr["lin_vel_x"], device=dev)
        cmd.lin_vel_y_range[ids] = torch.tensor(vr["lin_vel_y"], device=dev)
        cmd.ang_vel_z_range[ids] = torch.tensor(vr["ang_vel_z"], device=dev)
        cmd.random_velocity_indices[ids] = name in (cmd.cfg.random_velocity_terrain or [])
    terrain.terrain_levels[:] = level
    terrain.terrain_types[:] = types
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    print(f"[EVAL] terrains {args_cli.terrains}, level {level}, {args_cli.envs_per_terrain} robots each, dt {dt}")

    obs, _ = env.reset()
    thr = cmd.cfg.target_dis_threshold
    # per-env goal tracking
    goal = cmd.pos_command_w[:, :2].clone()
    goal_t0 = torch.zeros(num_envs, dtype=torch.long, device=dev)
    goal_d0 = (robot.data.root_pos_w[:, :2] - goal).norm(dim=1)
    goal_stand = cmd.is_standing_env.clone()
    reached = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    reach_t = torch.zeros(num_envs, dtype=torch.long, device=dev)
    ever_fell = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    ep = {k: [] for k in ("terrain", "fell", "reason_fall", "length")}
    gl = {k: [] for k in ("terrain", "reached", "time_s", "d0", "stand", "ended_by_fall")}
    ep_len_prev = uenv.episode_length_buf.clone()

    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs, _, dones, _ = env.step(policy(obs))
            dones = dones.bool()
            fell = torch.zeros(num_envs, dtype=torch.bool, device=dev)
            for name in FALL_TERMS:
                if name in term_mgr.active_terms:
                    fell |= term_mgr.get_term(name).bool()
            fell &= dones
            ever_fell |= fell

            # arrival check for live envs (post-step pose of reset envs is the new spawn, so skip them)
            live = ~dones
            dist = (robot.data.root_pos_w[:, :2] - goal).norm(dim=1)
            newly = live & ~reached & (dist < thr)
            reached |= newly
            reach_t[newly] = step + 1 - goal_t0[newly]

            # goal finished: replaced by the command resampler, or episode ended
            new_goal = cmd.pos_command_w[:, :2]
            finished = dones | (new_goal - goal).abs().sum(dim=1).gt(1e-6)
            fid = finished.nonzero().flatten()
            if len(fid):
                gl["terrain"] += block[fid].tolist()
                gl["reached"] += reached[fid].tolist()
                gl["time_s"] += (reach_t[fid].float() * dt).tolist()
                gl["d0"] += goal_d0[fid].tolist()
                gl["stand"] += goal_stand[fid].tolist()
                gl["ended_by_fall"] += fell[fid].tolist()
                goal[fid] = new_goal[fid]
                goal_t0[fid] = step + 1
                goal_d0[fid] = (robot.data.root_pos_w[fid, :2] - goal[fid]).norm(dim=1)
                goal_stand[fid] = cmd.is_standing_env[fid]
                reached[fid] = False
                reach_t[fid] = 0

            did = dones.nonzero().flatten()
            if len(did):
                ep["terrain"] += block[did].tolist()
                ep["fell"] += fell[did].tolist()
                ep["length"] += (ep_len_prev[did] + 1).tolist()
            ep_len_prev = uenv.episode_length_buf.clone()
            if step % 1000 == 0:
                print(f"[EVAL] step {step}/{args_cli.steps} episodes {len(ep['fell'])} goals {len(gl['reached'])}", flush=True)

    ep = {k: np.array(v) for k, v in ep.items() if k != "reason_fall"}
    gl = {k: np.array(v) for k, v in gl.items()}
    ever_fell = ever_fell.cpu().numpy()
    block_np = block.cpu().numpy()
    summary = {
        "checkpoint": resume_path,
        "task": args_cli.task,
        "level": level,
        "envs_per_terrain": args_cli.envs_per_terrain,
        "steps": args_cli.steps,
        "step_dt": dt,
        "target_dis_threshold": thr,
        "terrains": {},
    }
    for i, name in enumerate(args_cli.terrains):
        e = ep["terrain"] == i
        g = (gl["terrain"] == i) & ~gl["stand"] & (gl["d0"] >= thr)
        r = g & gl["reached"]
        summary["terrains"][name] = {
            "episodes": int(e.sum()),
            "episode_success_rate": float(1 - ep["fell"][e].mean()) if e.any() else None,
            "robot_success_rate": float(1 - ever_fell[block_np == i].mean()),
            "goals": int(g.sum()),
            "goal_reach_rate": float(r.sum() / max(g.sum(), 1)),
            "mean_reaching_time_s": float(gl["time_s"][r].mean()) if r.any() else None,
        }
    tag = os.path.splitext(os.path.basename(resume_path))[0] + "_" + "-".join(args_cli.terrains)
    with open(os.path.join(out_dir, f"evalpaper_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(
        os.path.join(out_dir, f"evalpaper_{tag}_raw.npz"),
        **{f"ep_{k}": v for k, v in ep.items()},
        **{f"goal_{k}": v for k, v in gl.items()},
        robot_ever_fell=ever_fell,
        robot_terrain=block_np,
        terrain_names=np.array(args_cli.terrains),
    )
    print("[EVAL] SUMMARY " + json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
