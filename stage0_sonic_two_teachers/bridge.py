"""SONIC <-> InstinctLab bridge: conventions verified in stage 0 (track_sonic_in_instinct.py).

- joint order: InstinctLab robot.joint_names (BFS of the torso-rooted URDF) vs SONIC_JOINTS (BFS of pelvis-rooted G1)
- waist sign: the torso-rooted URDF reverses the waist chain -> waist_yaw/roll/pitch values are negated
  (reset MPJPE-L 70 mm -> 1.5 mm when flipped)
- SONIC actions always use SONIC's own action scale (act_sonic.json); the joint target is then expressed in
  InstinctLab's action space a = (target - offset) / scale of the env's JointPositionAction term.
- reference motions: GEAR-SONIC deploy CSV format (50 Hz, SONIC joint order, 14 bodies, quats w,x,y,z)
"""

import json
import os

import numpy as np
import torch

from sonic_net import (
    FUTURE_STEP, NUM_FUTURE, REF_BODIES, SONIC_JOINTS, ProprioHistory, SonicActor, build_g1_encoder_input,
    quat_rotate_inverse,
)

HERE = os.path.dirname(os.path.abspath(__file__))
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
SHOE_DZ = 0.058


def load_motions(ref_dir, names=None):
    names = names or sorted(d for d in os.listdir(ref_dir) if os.path.isdir(os.path.join(ref_dir, d)))
    out = []
    for n in names:
        d = os.path.join(ref_dir, n)
        rd = lambda f: np.loadtxt(os.path.join(d, f), delimiter=",", skiprows=1, dtype=np.float32)
        jp = rd("joint_pos.csv"); T = jp.shape[0]
        out.append(dict(name=n, T=T, jp=jp, jv=rd("joint_vel.csv"), bpos=rd("body_pos.csv").reshape(T, 14, 3),
                        bquat=rd("body_quat.csv").reshape(T, 14, 4), blin=rd("body_lin_vel.csv").reshape(T, 14, 3),
                        bang=rd("body_ang_vel.csv").reshape(T, 14, 3)))
    return out


def heading_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], -1)


class Bridge:
    def __init__(self, robot, device, waist_sign=(-1.0, -1.0, -1.0)):
        self.robot, self.dev = robot, device
        self.perm_r2s = torch.tensor([robot.joint_names.index(j) for j in SONIC_JOINTS], device=device)
        self.perm_s2r = torch.tensor([SONIC_JOINTS.index(j) for j in robot.joint_names], device=device)
        self.sgn = torch.ones(29, device=device)
        for j, v in zip(WAIST, waist_sign):
            self.sgn[SONIC_JOINTS.index(j)] = v
        self.bidx = torch.tensor([robot.body_names.index(b) for b in REF_BODIES], device=device)
        self.pelvis = robot.body_names.index("pelvis")
        self.torso_ref = REF_BODIES.index("torso_link")
        s_act = json.load(open(os.path.join(HERE, "act_sonic.json")))
        self.scale_s = torch.tensor([s_act[j]["scale"] for j in SONIC_JOINTS], device=device)
        self.default_s = self.to_sonic(robot.data.default_joint_pos[0])

    def to_robot(self, x_s):
        return (x_s * self.sgn)[..., self.perm_s2r]

    def to_sonic(self, x_r):
        return x_r[..., self.perm_r2s] * self.sgn

    # ---- SONIC observations from the simulated robot
    def sonic_state(self, ids=slice(None)):
        d = self.robot.data
        q = d.body_quat_w[ids, self.pelvis]
        w = d.body_ang_vel_w[ids, self.pelvis]
        down = torch.tensor([0.0, 0.0, -1.0], device=self.dev).expand(q.shape[0], 3)
        return dict(ang_vel=quat_rotate_inverse(q, w), jpos=self.to_sonic(d.joint_pos[ids]) - self.default_s,
                    jvel=self.to_sonic(d.joint_vel[ids]), grav=quat_rotate_inverse(q, down), pelvis_quat=q)

    def sonic_target(self, act_s):
        """SONIC action -> joint position target in SONIC order/sign."""
        return self.default_s + act_s * self.scale_s

    def instinct_action(self, target_s, action_term, env_ids):
        """Joint target (SONIC convention) -> action vector of InstinctLab's JointPositionAction term."""
        sel = lambda x: x[env_ids] if torch.is_tensor(x) else x
        return (self.to_robot(target_s) - sel(action_term._offset)) / sel(action_term._scale)

    def instinct_target(self, act_r, action_term, env_ids):
        """InstinctLab action -> joint target (SONIC convention). Inverse of instinct_action."""
        sel = lambda x: x[env_ids] if torch.is_tensor(x) else x
        return self.to_sonic(act_r * sel(action_term._scale) + sel(action_term._offset))

    def body_pos_local(self, ids, dz=0.0):
        """Tracked bodies relative to pelvis in pelvis heading frame (robot), shoe offset removed."""
        d = self.robot.data
        b = d.body_pos_w[ids][:, self.bidx]
        q = d.body_quat_w[ids, self.pelvis]
        return quat_rotate_inverse(heading_quat(q)[:, None].expand(-1, b.shape[1], -1), b - b[:, :1]), b[:, 0, 2] - dz


class MotionPlayer:
    """Plays reference motions (one per env) and provides SONIC encoder inputs / metrics."""

    def __init__(self, motions, env_ids, origins, device):
        self.dev, self.ids = device, env_ids
        self.n = len(env_ids)
        Tmax = max(m["T"] for m in motions)

        def pad(key):
            return torch.tensor(np.stack([np.concatenate([m[key], np.repeat(m[key][-1:], Tmax - m["T"], 0)]) for m in motions]), device=device)

        self.R = {k: pad(k) for k in ("jp", "jv", "bpos", "bquat", "blin", "bang")}
        self.len = torch.tensor([m["T"] for m in motions], device=device)
        self.names = [m["name"] for m in motions]
        # place every motion's frame-0 pelvis xy on its env origin, z relative to the origin height
        p0 = self.R["bpos"][:, :1, :1].clone(); p0[..., 2] = 0
        self.R["bpos"] = self.R["bpos"] - p0 + origins[:, None, None, :]
        self.t = torch.zeros(self.n, dtype=torch.long, device=device)
        self.ar = torch.arange(self.n, device=device)
        self.future = torch.arange(NUM_FUTURE, device=device) * FUTURE_STEP

    def frame(self, key, k=None):
        return self.R[key][self.ar, self.t if k is None else k]

    def encoder_input(self, pelvis_quat):
        fi = torch.minimum(self.t[:, None] + self.future[None], self.len[:, None] - 1)
        g = lambda key: self.R[key][self.ar[:, None], fi]
        return build_g1_encoder_input(g("jp"), g("jv"), g("bquat")[:, :, 0], pelvis_quat)

    def write_reset(self, local, bridge, dz):
        """Put robots of local indices `local` at their reference frame t (normally 0)."""
        if len(local) == 0:
            return
        r = bridge.robot
        env = self.ids[local]
        t = self.t[local]
        tr = bridge.torso_ref
        pose = torch.cat([self.R["bpos"][local, t, tr] + torch.tensor([0, 0, dz], device=self.dev), self.R["bquat"][local, t, tr]], -1)
        vel = torch.cat([self.R["blin"][local, t, tr], self.R["bang"][local, t, tr]], -1)
        r.write_root_pose_to_sim(pose, env_ids=env)
        r.write_root_velocity_to_sim(vel, env_ids=env)
        r.write_joint_state_to_sim(bridge.to_robot(self.R["jp"][local, t]), bridge.to_robot(self.R["jv"][local, t]), env_ids=env)
        r.reset(env)

    def ref_local(self, k):
        b = self.R["bpos"][self.ar, k]
        q = self.R["bquat"][self.ar, k, 0]
        return quat_rotate_inverse(heading_quat(q)[:, None].expand(-1, b.shape[1], -1), b - b[:, :1]), b[:, 0, 2]
