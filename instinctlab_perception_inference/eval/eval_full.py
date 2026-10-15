"""Headless evaluation of a trained checkpoint on the ORIGINAL training task (no rendering).

Uses the unmodified task config (full 10x20 terrain grid, walls, curriculum, all terminations);
only num_envs can be overridden. Per finished episode it records: sub-terrain type, terrain level,
termination reason, episode length, and distance from the spawn origin.

Run from the InstinctLab repo root (same as play.py):
  python /path/to/eval_full.py --task=Instinct-Parkour-Target-Amp-G1-v0 \
      --load_run=/abs/path/to/run --checkpoint=model_30000.pt --num_envs=4096 --steps=3000 --headless
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an Instinct-RL checkpoint on the original task.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--steps", type=int, default=3000, help="Env steps to run (1000 steps = one 20s episode).")
parser.add_argument("--out", type=str, default=None, help="Output directory (default: <run>/eval).")
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
    """Replicates IsaacLab TerrainGenerator._generate_curriculum_terrains column -> sub-terrain mapping."""
    names = list(gen_cfg.sub_terrains.keys())
    props = np.array([c.proportion for c in gen_cfg.sub_terrains.values()], dtype=float)
    props /= props.sum()
    cum = np.cumsum(props)
    return [names[int(np.min(np.where(j / gen_cfg.num_cols + 0.001 < cum)[0]))] for j in range(gen_cfg.num_cols)]


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    run_dir = args_cli.load_run
    resume_path = get_checkpoint_path(os.path.dirname(run_dir), os.path.basename(run_dir), agent_cfg.load_checkpoint)
    out_dir = args_cli.out or os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    env = InstinctRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[EVAL] loading {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    terrain = uenv.scene.terrain
    robot = uenv.scene["robot"]
    term_mgr = uenv.termination_manager
    term_names = list(term_mgr.active_terms)
    col_names = column_to_subterrain(env_cfg.scene.terrain.terrain_generator)
    print(f"[EVAL] terminations: {term_names}")
    print(f"[EVAL] column -> sub-terrain: {col_names}")

    rec = {k: [] for k in ("level", "col", "length", "dist", "reason")}
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            # snapshot pre-step state: reset envs get new origins/levels inside env.step
            level = terrain.terrain_levels.clone()
            col = terrain.terrain_types.clone()
            origin = terrain.env_origins[:, :2].clone()
            pos = robot.data.root_pos_w[:, :2].clone()
            ep_len = uenv.episode_length_buf.clone() + 1

            obs, _, dones, _ = env.step(policy(obs))
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids) == 0:
                continue
            reason = ["?"] * len(done_ids)
            for name in term_names:  # first matching term wins; falls listed after time-outs in cfg
                hit = term_mgr.get_term(name)[done_ids].cpu().numpy()
                for i in np.nonzero(hit)[0]:
                    if reason[i] == "?" or name in FALL_TERMS:
                        reason[i] = name
            rec["level"] += level[done_ids].tolist()
            rec["col"] += col[done_ids].tolist()
            rec["length"] += ep_len[done_ids].tolist()
            rec["dist"] += (pos[done_ids] - origin[done_ids]).norm(dim=1).tolist()
            rec["reason"] += reason
            if step % 200 == 0:
                print(f"[EVAL] step {step}/{args_cli.steps}  episodes so far {len(rec['reason'])}", flush=True)

    reason = np.array(rec["reason"])
    fell = np.isin(reason, FALL_TERMS)
    col = np.array(rec["col"])
    level = np.array(rec["level"])
    dist = np.array(rec["dist"])
    summary = {
        "checkpoint": resume_path,
        "task": args_cli.task,
        "num_envs": uenv.num_envs,
        "steps": args_cli.steps,
        "episodes": int(len(reason)),
        "fall_rate": float(fell.mean()) if len(reason) else None,
        "reasons": {r: int((reason == r).sum()) for r in sorted(set(reason.tolist()))},
        "mean_dist_m": float(dist.mean()) if len(dist) else None,
        "mean_terrain_level_final": float(terrain.terrain_levels.float().mean()),
        "per_subterrain": {},
        "per_level": {},
    }
    for j, name in enumerate(col_names):
        m = col == j
        if m.any():
            d = summary["per_subterrain"].setdefault(name, {"episodes": 0, "falls": 0, "dist_sum": 0.0})
            d["episodes"] += int(m.sum())
            d["falls"] += int(fell[m].sum())
            d["dist_sum"] += float(dist[m].sum())
    for d in summary["per_subterrain"].values():
        d["fall_rate"] = d["falls"] / d["episodes"]
        d["mean_dist_m"] = d.pop("dist_sum") / d["episodes"]
    for lv in sorted(set(level.tolist())):
        m = level == lv
        summary["per_level"][int(lv)] = {"episodes": int(m.sum()), "fall_rate": float(fell[m].mean())}

    tag = os.path.splitext(os.path.basename(resume_path))[0]
    with open(os.path.join(out_dir, f"eval_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(out_dir, f"eval_{tag}_episodes.npz"), **{k: np.array(v) for k, v in rec.items()})
    print("[EVAL] SUMMARY " + json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
