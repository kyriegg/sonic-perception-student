"""Record joint motion data of a trained policy on chosen terrains (no rendering needed).

Same setup as eval_paper.py (original task config, curriculum off, each block of robots pinned to one
sub-terrain at a fixed difficulty level), but instead of success statistics it logs per-step data:

  joint_pos, joint_vel, joint_pos_target, applied_torque   [T, N, J]
  actions (raw policy output)                              [T, N, A]
  root_pos_w, root_quat_w (w,x,y,z), root_lin_vel_b, root_ang_vel_b
  command (pos_command_w), done, fell, episode_id           [T, N, ...]

Frame t is the state AFTER env step t. For an env with done[t]=True, frame t is already the new spawn
pose of the next episode, so split trajectories on done (episode_id does this for you).

Output (one file per terrain):  <out>/joints_<ckpt>_<terrain>_L<level>.npz
Optional --csv: also writes <out>/csv/<terrain>_robot<k>.csv for the first --csv_robots robots.

Run from the InstinctLab repo root (like eval_paper.py):
  python record_joints.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=/abs/run --checkpoint=model_30000.pt \
      --envs_per_terrain 10 --steps 1000 --level -1 --headless --csv
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

PAPER_TERRAINS = [
    "perlin_rough", "square_gaps", "pyramid_stairs", "pyramid_stairs_high", "pyramid_stairs_inv",
    "pyramid_stairs_inv_high", "boxes", "mesh_boxes", "hf_pyramid_slope_inv",
]

parser = argparse.ArgumentParser(description="Record joint trajectories per terrain.")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--terrains", nargs="+", default=PAPER_TERRAINS)
parser.add_argument("--envs_per_terrain", type=int, default=10)
parser.add_argument("--level", type=int, default=-1, help="Terrain row/difficulty; -1 = hardest.")
parser.add_argument("--steps", type=int, default=1000, help="Policy steps (step_dt = 0.02 s -> 1000 = 20 s).")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--csv", action="store_true", help="Also write per-robot CSV files.")
parser.add_argument("--csv_robots", type=int, default=1, help="How many robots per terrain to export as CSV.")
parser.add_argument("--full_terrain", action="store_true",
                    help="Keep the original 10x20 terrain grid (needs ~20GB VRAM). Default: compact grid with only the "
                         "requested sub-terrains (same num_rows/difficulty formula, 1 column each) so it fits on 8GB.")
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
    names = list(gen_cfg.sub_terrains.keys())
    props = np.array([c.proportion for c in gen_cfg.sub_terrains.values()], dtype=float)
    props /= props.sum()
    cum = np.cumsum(props)
    return [names[int(np.min(np.where(j / gen_cfg.num_cols + 0.001 < cum)[0]))] for j in range(gen_cfg.num_cols)]


def main():
    n_terr = len(args_cli.terrains)
    num_envs = n_terr * args_cli.envs_per_terrain
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.curriculum.terrain_levels = None
    gen_cfg = env_cfg.scene.terrain.terrain_generator
    if not args_cli.full_terrain:
        # recording-only change: drop unused sub-terrains; tile geometry / difficulty per row are unchanged
        missing = [n for n in args_cli.terrains if n not in gen_cfg.sub_terrains]
        assert not missing, f"unknown terrain {missing}; available: {list(gen_cfg.sub_terrains)}"
        gen_cfg.sub_terrains = {n: gen_cfg.sub_terrains[n] for n in args_cli.terrains}
        for c in gen_cfg.sub_terrains.values():
            c.proportion = 1.0
        gen_cfg.num_cols = n_terr
        cmd_cfg = env_cfg.commands.base_velocity  # its per-terrain tables must only name existing sub-terrains
        cmd_cfg.velocity_ranges = {k: v for k, v in cmd_cfg.velocity_ranges.items() if k in gen_cfg.sub_terrains}
        if cmd_cfg.random_velocity_terrain is not None:
            cmd_cfg.random_velocity_terrain = [k for k in cmd_cfg.random_velocity_terrain if k in gen_cfg.sub_terrains]
    col_names = column_to_subterrain(gen_cfg)
    for name in args_cli.terrains:
        assert name in col_names, f"unknown terrain {name}; available: {sorted(set(col_names))}"
    level = gen_cfg.num_rows - 1 if args_cli.level < 0 else args_cli.level

    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    run_dir = args_cli.load_run
    resume_path = get_checkpoint_path(os.path.dirname(run_dir), os.path.basename(run_dir), agent_cfg.load_checkpoint)
    out_dir = args_cli.out or os.path.join(run_dir, "joint_data")
    os.makedirs(out_dir, exist_ok=True)

    env = InstinctRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[REC] loading {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    dev = uenv.device
    terrain = uenv.scene.terrain
    robot = uenv.scene["robot"]
    term_mgr = uenv.termination_manager
    cmd = uenv.command_manager.get_term("base_velocity")
    dt = uenv.step_dt

    block = torch.arange(num_envs, device=dev) // args_cli.envs_per_terrain
    types = torch.zeros(num_envs, dtype=torch.long, device=dev)
    for i, name in enumerate(args_cli.terrains):
        cols = torch.tensor([j for j, n in enumerate(col_names) if n == name], device=dev)
        ids = (block == i).nonzero().flatten()
        types[ids] = cols[torch.arange(len(ids), device=dev) % len(cols)]
        vr = cmd.cfg.velocity_ranges[name]
        cmd.lin_vel_x_range[ids] = torch.tensor(vr["lin_vel_x"], device=dev)
        cmd.lin_vel_y_range[ids] = torch.tensor(vr["lin_vel_y"], device=dev)
        cmd.ang_vel_z_range[ids] = torch.tensor(vr["ang_vel_z"], device=dev)
        cmd.random_velocity_indices[ids] = name in (cmd.cfg.random_velocity_terrain or [])
    terrain.terrain_levels[:] = level
    terrain.terrain_types[:] = types
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    print(f"[REC] terrains {args_cli.terrains}, level {level}, {args_cli.envs_per_terrain} robots each, dt {dt}")

    obs, _ = env.reset()
    T, N, J = args_cli.steps, num_envs, robot.num_joints
    A = env.num_actions
    f32 = dict(dtype=torch.float32, device=dev)
    buf = {
        "joint_pos": torch.zeros(T, N, J, **f32),
        "joint_vel": torch.zeros(T, N, J, **f32),
        "joint_pos_target": torch.zeros(T, N, J, **f32),
        "applied_torque": torch.zeros(T, N, J, **f32),
        "actions": torch.zeros(T, N, A, **f32),
        "root_pos_w": torch.zeros(T, N, 3, **f32),
        "root_quat_w": torch.zeros(T, N, 4, **f32),
        "root_lin_vel_b": torch.zeros(T, N, 3, **f32),
        "root_ang_vel_b": torch.zeros(T, N, 3, **f32),
        "command": torch.zeros(T, N, cmd.pos_command_w.shape[1], **f32),
        "done": torch.zeros(T, N, dtype=torch.bool, device=dev),
        "fell": torch.zeros(T, N, dtype=torch.bool, device=dev),
    }

    with torch.inference_mode():
        for t in range(T):
            act = policy(obs)
            obs, _, dones, _ = env.step(act)
            dones = dones.bool()
            fell = torch.zeros(N, dtype=torch.bool, device=dev)
            for name in FALL_TERMS:
                if name in term_mgr.active_terms:
                    fell |= term_mgr.get_term(name).bool()
            d = robot.data
            buf["joint_pos"][t] = d.joint_pos
            buf["joint_vel"][t] = d.joint_vel
            buf["joint_pos_target"][t] = d.joint_pos_target
            buf["applied_torque"][t] = d.applied_torque
            buf["actions"][t] = act
            buf["root_pos_w"][t] = d.root_pos_w
            buf["root_quat_w"][t] = d.root_quat_w
            buf["root_lin_vel_b"][t] = d.root_lin_vel_b
            buf["root_ang_vel_b"][t] = d.root_ang_vel_b
            buf["command"][t] = cmd.pos_command_w
            buf["done"][t] = dones
            buf["fell"][t] = fell & dones
            if t % 200 == 0:
                print(f"[REC] step {t}/{T}", flush=True)

    data = {k: v.cpu().numpy() for k, v in buf.items()}
    # episode id per env: increments after each done
    ep_id = np.zeros((T, N), dtype=np.int32)
    ep_id[1:] = np.cumsum(data["done"][:-1], axis=0)
    data["episode_id"] = ep_id
    data["env_origin"] = terrain.env_origins.cpu().numpy()
    data["default_joint_pos"] = robot.data.default_joint_pos.cpu().numpy()
    block_np = block.cpu().numpy()
    joint_names = list(robot.joint_names)
    ckpt = os.path.splitext(os.path.basename(resume_path))[0]
    meta = {"checkpoint": resume_path, "task": args_cli.task, "level": level, "full_terrain": args_cli.full_terrain, "step_dt": dt, "steps": T,
            "envs_per_terrain": args_cli.envs_per_terrain, "joint_names": joint_names, "files": {}}

    for i, name in enumerate(args_cli.terrains):
        m = block_np == i
        sub = {k: (v[:, m] if v.ndim >= 2 and v.shape[1] == N else v[m]) for k, v in data.items()}
        fn = os.path.join(out_dir, f"joints_{ckpt}_{name}_L{level}.npz")
        np.savez_compressed(fn, **sub, joint_names=np.array(joint_names), time_s=np.arange(1, T + 1) * dt,
                            terrain=name, level=level)
        meta["files"][name] = {"npz": fn, "episodes_ended": int(sub["done"].sum()), "falls": int(sub["fell"].sum())}
        print(f"[REC] {name}: saved {fn}  falls {int(sub['fell'].sum())}", flush=True)
        if args_cli.csv:
            csv_dir = os.path.join(out_dir, "csv")
            os.makedirs(csv_dir, exist_ok=True)
            hdr = (["time_s", "episode_id", "done", "fell"]
                   + [f"root_{c}" for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
                   + [f"q_{j}" for j in joint_names] + [f"dq_{j}" for j in joint_names]
                   + [f"tau_{j}" for j in joint_names])
            for k in range(min(args_cli.csv_robots, int(m.sum()))):
                rows = np.concatenate([
                    (np.arange(1, T + 1) * dt)[:, None], sub["episode_id"][:, k, None], sub["done"][:, k, None],
                    sub["fell"][:, k, None], sub["root_pos_w"][:, k], sub["root_quat_w"][:, k],
                    sub["joint_pos"][:, k], sub["joint_vel"][:, k], sub["applied_torque"][:, k]], axis=1)
                np.savetxt(os.path.join(csv_dir, f"{name}_L{level}_robot{k}.csv"), rows, delimiter=",",
                           header=",".join(hdr), comments="", fmt="%.6g")

    with open(os.path.join(out_dir, f"joints_{ckpt}_L{level}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[REC] DONE " + json.dumps(meta["files"], indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
