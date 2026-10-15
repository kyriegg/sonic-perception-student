# SONIC + Perception Student (G1)

Unitree G1 上的双老师蒸馏实验：学生以 NVIDIA SONIC 为底座，从 SONIC（动作跟踪）和 InstinctLab 感知策略（复杂地形）两个老师学习。

## 目录

| 目录 | 内容 |
|---|---|
| [`instinctlab_perception_inference/`](instinctlab_perception_inference/) | **纯 InstinctLab 感知策略的推理**（复现好的 `model_30000.pt`，不涉及 SONIC）：play 回放视频（`videos/`，3000 与 30000 次迭代对比）和 play 日志（`play_logs/`）；原版地形评估（`eval/eval_full.py`，`eval/results_3k/`、`eval/results_15k/`）；按论文方式的 9 种地形评估（`eval/eval_paper.py`、`eval/results_paper/`）；最高难度下 9 种地形的关节运动数据录制（`eval/record_joints.py`、`joint_data/`）；本地运行包装脚本 `run_instinct.ps1` |
| [`stage0_sonic_two_teachers/`](stage0_sonic_two_teachers/) | 初步实验（第零阶段）：PyTorch 重写 SONIC 并与官方 ONNX 比对、SONIC 搬进 InstinctLab 仿真器（含腰部关节符号修正）、两个老师在同一环境中的验证与回放视频 |

详细说明见各目录下的 README：
- `instinctlab_perception_inference/README_2026-09-24_play与评估.md`
- `stage0_sonic_two_teachers/README_2026-10-10_关节数据与第零阶段.md`

## 视频

### InstinctLab 感知策略 `model_30000.pt`（Play 版地形，1 个机器人，10 s）

![InstinctLab model_30000 play](media/instinctlab_model_30000_play.gif)

原视频（1280×720，50 fps）：[`instinctlab_perception_inference/videos/model_30000_play_1env.mp4`](instinctlab_perception_inference/videos/model_30000_play_1env.mp4)

### 第零阶段：两个老师在同一个 InstinctLab 环境里（70 s，GIF 为 4 倍速）

0–30 s 由 SONIC 控制（边走边转身 / 跳舞 / 单腿跳），30–70 s 由感知老师控制（上台阶 / 缺口 / Small Box / Ramp，最高难度）。

![Stage 0 two teachers](media/stage0_two_teachers_4x.gif)

原视频（1280×720，50 fps）：[`stage0_sonic_two_teachers/results_stage0/video_final/videos/stage0_two_teachers-step-0.mp4`](stage0_sonic_two_teachers/results_stage0/video_final/videos/stage0_two_teachers-step-0.mp4)

## 关节数据格式

`instinctlab_perception_inference/joint_data/model_30000_L9/`：9 种地形 × 10 个机器人 × 1000 步（20 s，50 Hz），第 9 级难度。
- `joints_model_30000_<地形>_L9.npz`：形状 `[1000 步, 10 机器人, ...]`
- `joints_model_30000_L9_meta.json`：关节名、任务、难度、步长等元信息
- `csv/<地形>_L9_robot0.csv`：每种地形 1 个机器人的 CSV（时间、根位姿、q、dq、力矩）

## 未包含的文件

- SONIC 官方动作网络权重 `sonic_release_policy_sd.pt`（约 104 MB，超过 GitHub 单文件上限，且属于 NVIDIA 官方发布）：从 SONIC 官方仓库的 Hugging Face 下载脚本获取（`python download_from_hf.py --training --no-smpl` 得到 `sonic_release/last.pt`），再转换成 state_dict。
- 感知策略 checkpoint `model_30000.pt` 及 InstinctLab / IsaacLab 源码。

## 依赖

- InstinctLab（Isaac Lab 2.1.0 / Isaac Sim 4.5）
- SONIC（NVIDIA GEAR）
