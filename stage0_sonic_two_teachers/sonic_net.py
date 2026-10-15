"""Standalone PyTorch re-implementation of the released SONIC actor (g1 encoder + FSQ + g1_dyn decoder).

Loads `sonic_release_policy_sd.pt` (the `policy_state_dict` of GEAR-SONIC `sonic_release/last.pt`)
without needing the gear_sonic / trl / Isaac Lab 2.3 stack.

Conventions (from gear_sonic/envs/manager_env/mdp + trl/modules, sonic_release/config.yaml):
  - joint order: SONIC_JOINTS (Isaac Lab BFS order of the pelvis-rooted G1)
  - g1 encoder input [B, 640]:
        cmd = cat([ref_joint_pos[B,10,29].flatten(1), ref_joint_vel[B,10,29].flatten(1)], 1).view(B,10,58)
        ori = 6D (first two columns of rotation matrix, row-major) of quat_inv(robot_pelvis_quat) * ref_root_quat,
              [B,10,6]
        x   = cat([cmd, ori], -1).view(B, 640)
    future frames are 0.1 s apart (5 steps at 50 Hz), frame 0 = current reference frame.
  - FSQ: 32 dims x 32 levels per token, 2 tokens -> token_flattened [B, 64]
  - decoder input [B, 994] = cat([token_flattened, proprio]), proprio [B, 930] =
        base_ang_vel(pelvis, body frame)[10x3] | joint_pos - default[10x29] | joint_vel[10x29]
        | last_action[10x29] | gravity_dir(pelvis frame)[10x3]          (each oldest -> newest)
  - action = decoder output; joint target = default_pos + action * G1_ACTION_SCALE (0.25*effort/stiffness)
"""

import math

import torch
import torch.nn as nn

SONIC_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
# hardware / MuJoCo (DFS) joint order
DFS_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# bodies stored in the deploy reference CSVs (body_pos.csv / body_quat.csv), in that order
REF_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]

NUM_FUTURE = 10
FUTURE_STEP = 5  # 0.1 s at 50 Hz
HIST = 10


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class FSQ(nn.Module):
    """vector_quantize_pytorch FSQ.bound (eps=1e-3, tanh) with uniform levels."""

    def __init__(self, levels=32, dim=32, eps=1e-3):
        super().__init__()
        half_l = (levels - 1) * (1 + eps) / 2
        offset = 0.5 if levels % 2 == 0 else 0.0
        self.half_l, self.offset = half_l, offset
        self.shift = math.atanh(offset / half_l)
        self.half_width = levels // 2

    def forward(self, z):
        b = torch.tanh(z + self.shift) * self.half_l - self.offset
        q = b + (torch.round(b) - b).detach()  # straight-through
        return q / self.half_width


class SonicActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_g1 = mlp([640, 2048, 1024, 512, 512, 64])
        self.dec = mlp([994, 2048, 2048, 1024, 1024, 512, 512, 29])
        self.fsq = FSQ()

    @classmethod
    def from_release(cls, path, device="cpu"):
        sd = torch.load(path, map_location="cpu")
        m = cls()
        m.enc_g1.load_state_dict({k.split("encoders.g1.module.")[1]: v for k, v in sd.items() if "encoders.g1.module." in k})
        m.dec.load_state_dict({k.split("decoders.g1_dyn.module.")[1]: v for k, v in sd.items() if "decoders.g1_dyn.module." in k})
        return m.to(device).eval()

    def encode_g1(self, enc_in):
        """enc_in [B, 640] -> token_flattened [B, 64]"""
        z = self.enc_g1(enc_in).view(-1, 2, 32)
        return self.fsq(z).view(-1, 64)

    def decode(self, token_flat, proprio):
        return self.dec(torch.cat([token_flat, proprio], dim=-1))

    def forward(self, enc_in, proprio):
        return self.decode(self.encode_g1(enc_in), proprio)


# ---------------------------------------------------------------- helpers (quaternions are w, x, y, z)
def quat_inv(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).view(*q.shape[:-1], 3, 3)


def quat_rotate_inverse(q, v):
    return (quat_to_mat(q).transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)


def build_g1_encoder_input(ref_jpos, ref_jvel, ref_root_quat, robot_pelvis_quat):
    """ref_* : [B, 10, 29] / [B, 10, 4]; robot_pelvis_quat [B, 4]. Returns [B, 640]."""
    B = ref_jpos.shape[0]
    cmd = torch.cat([ref_jpos.reshape(B, -1), ref_jvel.reshape(B, -1)], dim=1).view(B, NUM_FUTURE, -1)
    dif = quat_mul(quat_inv(robot_pelvis_quat).unsqueeze(1).expand(-1, NUM_FUTURE, -1), ref_root_quat)
    ori = quat_to_mat(dif)[..., :2].reshape(B, NUM_FUTURE, 6)
    return torch.cat([cmd, ori], dim=-1).reshape(B, -1)


class ProprioHistory:
    """10-frame history buffers for the decoder proprioception (oldest -> newest)."""

    def __init__(self, num_envs, device):
        self.buf = {
            "ang_vel": torch.zeros(num_envs, HIST, 3, device=device),
            "jpos": torch.zeros(num_envs, HIST, 29, device=device),
            "jvel": torch.zeros(num_envs, HIST, 29, device=device),
            "act": torch.zeros(num_envs, HIST, 29, device=device),
            "grav": torch.zeros(num_envs, HIST, 3, device=device),
        }

    def reset(self, ids, ang_vel, jpos, jvel, grav):
        """Fill the whole history with the current state (Isaac Lab CircularBuffer behaviour after reset)."""
        for k, v in (("ang_vel", ang_vel), ("jpos", jpos), ("jvel", jvel), ("grav", grav)):
            self.buf[k][ids] = v[ids].unsqueeze(1).expand(-1, HIST, -1)
        self.buf["act"][ids] = 0.0

    def push(self, ang_vel, jpos, jvel, act, grav):
        for k, v in (("ang_vel", ang_vel), ("jpos", jpos), ("jvel", jvel), ("act", act), ("grav", grav)):
            self.buf[k] = torch.cat([self.buf[k][:, 1:], v.unsqueeze(1)], dim=1)

    def proprio(self):
        B = self.buf["jpos"].shape[0]
        return torch.cat([self.buf[k].reshape(B, -1) for k in ("ang_vel", "jpos", "jvel", "act", "grav")], dim=-1)
