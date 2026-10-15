# 2026-09-24 ~ 09-25 工作记录：录 play 视频 + 原版地形评估

训练本身在 2026-09-20 ~ 09-22 已在 AutoDL 上跑完（任务 `Instinct-Parkour-Target-Amp-G1-v0`，4096 环境，30000 次迭代，
唯一偏差是 `save_interval=1000`）。这两天做的是：**把训练好的策略看一眼（视频）+ 在原版地形上量化评估**。

---

## 1. 结论速览

- 最终策略 `model_30000.pt` **在原版训练地形上工作正常**：61612 个回合，总摔倒率 **1.9%**。
  评估结束时平均地形等级 **6.18**，与训练结束时的 **6.03** 基本一致。
- `model_3000.pt`（10% 训练量）动作明显发抖；`model_30000.pt` 明显更稳。
- **按论文方式评估（每种地形 1000 机器人、最高难度、10000 步）**：起伏、缺口、上台阶、上平台 ≥99.76%，基本复现；
  下平台 99.42%、下台阶 98.36% 略低；**Large Box 95.85%、Small Box 79.52%、Ramp 84.49%，明显低于论文的 100 / 99.09 / 100**。
  到达时间和论文一个量级（约 5 s）。论文没写清成功定义和评估难度，详见第 3 节。
- **AutoDL 上 Isaac Sim 4.5 无法渲染**（段错误），录视频只能在本地笔记本做。不开渲染的训练和评估不受影响。

---

## 2. 视频（本地笔记本录制）

| 文件 | 内容 |
|---|---|
| `videos/model_3000_play_1env.mp4` | 第 3000 次迭代，1 个机器人，10 秒（500 步，50fps，1280×720），明显发抖 |
| `videos/model_30000_play_1env.mp4` | 最终模型，同上设置，明显更稳 |

原始输出也在 `logs_server/20260920_011638/videos/play/`（play.py 默认写到这里）。

**注意：视频用的是 Play 版任务 `Instinct-Parkour-Target-Amp-G1-Play-v0`，不是训练时的原版地形。**
Play 版与原版的区别（源码 `InstinctLab/.../config/g1/g1_parkour_target_amp_cfg.py`）：

| | 原版（训练用） | Play 版（视频） |
|---|---|---|
| 地形网格 | 10 × 20 = 200 块 | 4 × 10 = 40 块 |
| 墙 | 有（`wall_prob=0.3`） | 全部去掉 |
| `root_height` 摔倒判定 | 开 | 关 |
| 回合长度 | 20 s | 10 s |
| 摩擦力随机化、关节初始扰动 | 开 | 关 |

原版 200 块地形在本地会爆显存（8GB，深度相机 + 高度扫描的 Warp BVH 放不下），所以视频只能用 Play 版。

**本地录视频命令**（PowerShell，在 `InstinctLab` 目录下）：
```powershell
$icd = "C:\Windows\System32\DriverStore\FileRepository\nvlti.inf_amd64_923ecf596c1d55f9\nv-vk64.json"
$env:VK_ICD_FILENAMES = $icd; $env:VK_DRIVER_FILES = $icd   # 笔记本核显+独显，指定 NVIDIA 的 Vulkan
& "D:\robot paper\instinct_repro\run_instinct.ps1" "source\instinctlab\instinctlab\tasks\parkour\scripts\play.py" `
  --task=Instinct-Parkour-Target-Amp-G1-Play-v0 `
  --load_run="D:\robot paper\instinct_repro\logs_server\20260920_011638" `
  --checkpoint=model_30000.pt --num_envs=1 --headless --video --video_length=500
```
- 约占 3.9GB 显存，一段几分钟。
- 结尾的 `FileNotFoundError` 和随后的 access violation 可以忽略：play.py 录完后调用 `code`（VS Code）打开视频，本机 PATH 里没有。视频在这之前已写完。

---

## 3. 原版地形评估（AutoDL，无渲染）

### 做法
- 脚本：`eval/eval_full.py`（新写的，**不改任何上游代码和配置**）。
- 加载原版任务 `Instinct-Parkour-Target-Amp-G1-v0` 的默认配置：200 块地形、带墙、课程学习开、全部终止条件开，
  只把 `num_envs` 设为 4096（和训练一样）。策略用确定性推理（`get_inference_policy`，和 play.py 一致）。
- 每个回合结束时记录：子地形类型、难度等级、终止原因、回合长度、离出生点的距离。
- 摔倒 = `base_contact` / `bad_orientation` / `root_height`；`time_out` / `terrain_out_bound` 算正常结束。
- 跑了两次：
  - **3000 步**（3 个回合）：课程学习从 ≤5 级起步，爬不上去，只覆盖到 0–5 级 → 摔倒率 1.3%，**不能代表最难地形**。
  - **15000 步**（15 个回合）：课程学习爬到和训练结束时相当的水平，覆盖 0–9 级。**以这次为准。**

服务器上的运行命令（`/root/autodl-tmp/run_eval.sh`）：
```bash
source /root/autodl-tmp/server_setup/env.sh && activate_env && cd $SRC_DIR/InstinctLab
python /root/autodl-tmp/eval_full.py --task=Instinct-Parkour-Target-Amp-G1-v0 \
  --load_run=$LOG_ROOT/20260920_011638 --checkpoint=model_30000.pt \
  --num_envs=4096 --steps=15000 --headless --out=$LOG_ROOT/20260920_011638/eval_15k
```
4090 上约 40 分钟，显存约 20GB。结尾的 `AttributeError: points_visualizer` 是上游关闭时的已知报错，训练时也有，不影响结果。

### 15000 步结果（`model_30000.pt`）

总计 61612 回合，摔倒 1181 次（**1.9%**）：`root_height` 864、`bad_orientation` 234、`base_contact` 83；
`time_out` 60430、`terrain_out_bound` 1。评估结束时平均地形等级 6.18（训练结束时 6.03）。

**按子地形**（全部难度 / 最难两级 8–9 级）：

| 子地形 | 全部等级摔倒率 | 8–9 级摔倒率 | 平均位移 (m) |
|---|---|---|---|
| perlin_rough（起伏地面） | 0.13% | 0.3% | 2.69 |
| perlin_rough_stand（原地站立指令） | 0% | 0% | 0.34 |
| square_gaps（方形缺口） | 0.42% | 0.1% | 3.65 |
| pyramid_stairs（上台阶） | 0.27% | 0.9% | 3.64 |
| pyramid_stairs_high（高台阶） | 0.10% | 0.2% | 3.65 |
| pyramid_stairs_inv（下台阶） | 0.02% | 0% | 3.66 |
| pyramid_stairs_inv_high（下高台阶） | 0.05% | 0% | 3.64 |
| boxes（离散方块） | 3.1% | 6.2% | 2.62 |
| **mesh_boxes（随机叠放方块）** | **10.8%** | **15.2%** | 2.52 |
| **hf_pyramid_slope_inv（倒金字塔坡）** | **4.1%** | **10.5%** | 2.47 |

**按难度等级**：0–5 级摔倒率 0.8%–1.7%，6–7 级约 1.8%，**8 级 3.3%，9 级 3.9%**。

说明：
- 平均位移是"回合结束时离出生点的直线距离"。这个任务是走向地块内的目标点，受 8m×8m 地块大小限制，不是总行走距离。
- 子地形由地形列号推出，映射方式和 IsaacLab `TerrainGenerator` 的课程模式一致（按 proportion 分配 20 列）。
### 与论文对比

论文：`D:\robot paper\instinct_extract\paper\hiking-in-the-wild.pdf`，Table I（第 8 页，"Ours" 行）、Table II，附录 Table VI（地形定义）。
附录 Table VI 的地形顺序和比例与代码里的子地形一一对应（5/5/10/15/10/15/10/10/10/10%），
代码默认配置也和论文一致（MoE 4 专家、AMP、步行动作数据），所以按下表对照。
我们的"成功率"= 1 − 摔倒率。

| 论文地形名 | 代码子地形 | 论文成功率 | 我们（全部等级） | 我们（8–9 级） |
|---|---|---|---|---|
| Rough | perlin_rough | 100.0% | 99.87% | 99.7% |
| Gap | square_gaps | 100.0% | 99.58% | 99.9% |
| Stair down | pyramid_stairs | 99.95% | 99.73% | 99.1% |
| Platform down | pyramid_stairs_high | 100.0% | 99.90% | 99.8% |
| Stair up | pyramid_stairs_inv | 100.0% | 99.98% | 100.0% |
| Platform up | pyramid_stairs_inv_high | 100.0% | 99.95% | 100.0% |
| Large Box | boxes | 100.0% | **96.9%** | **93.8%** |
| Small Box | mesh_boxes | 99.09% | **89.2%** | **84.8%** |
| Ramp | hf_pyramid_slope_inv | 100.0% | **95.9%** | **89.5%** |

**结论**：
- 台阶、平台、缺口、起伏地面 6 类**基本复现**（≥99.6%，与论文差 <0.5 个百分点）。
- **Large Box、Small Box、Ramp 三类明显低于论文**（差 3–10 个百分点，最难等级差得更多）。
  趋势一致：论文里 Small Box 也是最难的一类（唯一不到 100% 的方块类，消融实验里一去掉组件就崩）。

**口径差异（所以不能直接判定"没复现出来"）**：
1. 论文没写清"成功"的定义（可能是到达目标，也可能是没摔倒），也没写评估用哪个难度等级。
   论文是**每种地形单独**用 1000 个机器人测（Table II 写的是 10000 步 × 1000 机器人）；
   我们是 10 种地形混在一起、课程学习在 0–9 级之间动态调整。
2. `Large Box`/`Small Box` 对应 `boxes`/`mesh_boxes` 是按附录顺序和几何参数推断的（Discrete Boxes 宽 0.8–1.5 m → Large）。
3. 我们没测"平均到达时间"（论文 Table I 右半部分），当前脚本不记录目标点。
4. 只训练了一个种子；论文结果可能是挑过的或多种子最好的。

### 按论文方式的评估（2026-09-25，以此为准）

**做法**：脚本 `eval/eval_paper.py`。每种地形单独放 **1000 个机器人**，全部固定在**最高难度（第 9 级）**，
关闭地形课程学习，跑 **10000 步**（200 s，约 10 个回合）；其余仍是原版任务配置。
分三批跑（每批 3 种地形 × 1000 = 3000 环境，每批约 25 分钟，4090 显存约 17GB），服务器命令见 `/root/autodl-tmp/run_evalpaper.sh`。
为了让机器人留在指定地形，脚本改了 `terrain_levels` / `terrain_types` / `env_origins`，
并按新地形重新设置了速度指令范围（命令项只在初始化时按地形设置一次）——都是评估用设置，不改训练配置。

记录的指标：
- **回合成功率** = 1 − 以摔倒（`base_contact` / `bad_orientation` / `root_height`）结束的回合比例
- **机器人成功率** = 10000 步内一次都没摔过的机器人比例（更严格）
- **平均到达时间** = 从分配目标点到第一次进入 0.4 m 内的时间（只算到达了的目标；站立指令和出生时已在 0.4 m 内的目标不计）

| 论文地形 | 代码子地形 | 论文成功率 | **回合成功率** | 机器人成功率 | 论文到达时间 (s) | 我们到达时间 (s) |
|---|---|---|---|---|---|---|
| Rough | perlin_rough | 100.0 | **99.95** | 99.5 | 5.40 | 5.13 |
| Gap | square_gaps | 100.0 | **99.76** | 98.2 | 5.43 | 4.59 |
| Stair down | pyramid_stairs | 99.95 | **98.36** | 86.4 | 4.45 | 4.96 |
| Platform down | pyramid_stairs_high | 100.0 | **99.42** | 94.6 | 5.48 | 5.17 |
| Stair up | pyramid_stairs_inv | 100.0 | **99.99** | 99.9 | 5.35 | 5.65 |
| Platform up | pyramid_stairs_inv_high | 100.0 | **99.97** | 99.7 | 5.45 | 5.13 |
| Large Box | boxes | 100.0 | **95.85** | 66.8 | 5.50 | 5.15 |
| Small Box | mesh_boxes | 99.09 | **79.52** | 12.9 | 8.03 | 5.27 |
| Ramp | hf_pyramid_slope_inv | 100.0 | **84.49** | 22.8 | 5.84 | 5.49 |

**结论**：
- **起伏、缺口、上台阶、上平台 4 类复现得很好**（≥99.76%，与论文差 ≤0.24 个百分点）；
  下平台 99.42%、下台阶 98.36% 略低于论文（差 0.6–1.6 个百分点）。
- **Large Box / Small Box / Ramp 在最高难度下明显低于论文**：95.85 / 79.52 / 84.49 vs 100 / 99.09 / 100。
  这和课程评估（0–9 级混合，96.9 / 89.2 / 95.9）的结论方向一致，只是固定在最难一级时差距更大。
- **到达时间整体和论文一个量级（4.5–5.7 s vs 4.45–5.84 s）**，说明走得快慢基本对得上。
  Small Box 我们 5.27 s 比论文 8.03 s 短，但我们只统计了到达的目标，这一类大量机器人中途摔倒，有幸存者偏差。
- "机器人成功率"（200 s 内一次不摔）远低于回合成功率，是更严格的口径，论文应该不是这个定义（否则它的 100% 很难做到）。

**仍然存在的口径不确定**（论文没写清）：
1. "成功"的定义：我们主口径用"回合没摔倒"。
2. 评估用的难度等级：我们用最高级；如果论文用的是较低难度或随机难度，Box / Ramp 的差距会小很多（参考课程评估结果）。
3. 训练只有一个种子，也没做任何挑选。

**可能的原因（未验证）**：方块和坡是最依赖深度感知的地形；只训练一个种子，且论文可能是训练后又做了 fine-tune
（论文提到 "the base policy is post-trained for high-speed running and some specific terrains"，这部分代码/配置没有公开的迹象）。

---

## 4. 为什么视频不在 AutoDL 上录（排查记录）

1. 前两台新机器（`westb` 驱动 535、`bjb1` 驱动 580）上 Vulkan 起不来：`Could not get 'vkCreateInstance' ... for ICD libGLX_nvidia.so.0`。
   **原因：容器里没有 X 服务器，GLX 入口不能用；把 ICD 改成 `libEGL_nvidia.so.0` 就能识别 4090。**
   已修进 `server_setup/setup_server.sh`（优先写 EGL 入口）。
2. 在原训练机（`bjb3`，驱动 595，Vulkan 正常）上 play 开渲染直接段错误。gdb 调用栈显示崩在 Isaac Sim 自己的
   `omni.hydra.rtx/.../librtx.scenedb.plugin.so`，时间点是创建场景、初始化 RTX 渲染器时。
   - 试过 `--kit_args="--/rtx-transient/scenedb/forceMaxTLASInstancesLimit=..."`（16M / 1M），无效。
   - 试过强制 EGL 入口，无效。
   - **IsaacLab 自带空场景 `scripts/tutorials/00_sim/create_empty.py --headless --enable_cameras` 同样崩溃**，
     所以与 InstinctLab / 模型 / 地形无关，是 Isaac Sim 4.5 渲染器在该容器环境下起不来。
   - 是驱动太新（Isaac Sim 4.5 推荐 535.129，AutoDL 基本都是 570/580/590+）还是容器缺东西，没能区分（找不到 535–550 的机器）。
3. 结论：AutoDL 只用来跑无渲染的训练和评估；视频在本地录。若将来一定要原版地形的视频，可选：
   服务器记录轨迹 + 本地回放渲染，或租带官方 Isaac Sim 镜像的完整虚拟机（如 AWS g5/g6）。

---

## 5. 文件位置汇总（都在 `D:\robot paper\instinct_repro\` 下）

| 路径 | 说明 |
|---|---|
| `README_2026-09-24_play与评估.md` | 本文件 |
| `logs_server/20260920_011638/` | 从服务器同步回来的训练结果：`model_1000.pt` … `model_30000.pt`（30 个）、tfevents、`params/` |
| `logs_server/20260920_011638/videos/play/` | play.py 输出的原始视频 |
| `videos/model_3000_play_1env.mp4` | 3000 次迭代的视频 |
| `videos/model_30000_play_1env.mp4` | 最终模型的视频 |
| `eval/eval_full.py` | 原版地形评估脚本 |
| `eval/results_15k/eval_model_30000.json` | **15000 步评估汇总（主结果）** |
| `eval/results_15k/eval_model_30000_episodes.npz` | 15000 步每回合明细（`level`、`col`、`length`、`dist`、`reason`） |
| `eval/results_3k/eval_model_30000.json`、`..._episodes.npz` | 3000 步评估（只覆盖 0–5 级，仅供参考） |
| `eval/eval_paper.py` | 按论文方式的评估脚本（每种地形 1000 机器人、固定难度、10000 步、记录到达时间） |
| `eval/results_paper/evalpaper_model_30000_*.json` | **按论文方式的评估结果（3 个文件，每个 3 种地形）** |
| `eval/results_paper/evalpaper_model_30000_*_raw.npz` | 对应的每回合 / 每目标点明细 |
| `eval/results_paper/evalpaper.log` | 三批评估的服务器日志 |
| `eval/eval15k.log`、`eval/eval.log` | 两次评估的服务器运行日志 |
| `play_local_play1.log`、`play_local_30000.log` | 本地成功录视频的两次日志（3000 / 30000） |
| `play_model3000*.log`（v1–v3） | 09-24 早些时候本地用原版任务 play 爆显存的失败日志 |
| `server_setup/setup_server.sh` | 已改：Vulkan ICD 优先用 `libEGL_nvidia.so.0` |
| `server_setup/README.md` | 已补充：EGL 入口、AutoDL 无法渲染两条常见问题 |
| `run_instinct.ps1` | 本地运行包装脚本（清掉 user-site，避免导入到别的 isaaclab） |

服务器（AutoDL 云服务器，`/root/autodl-tmp/`）上相应位置：
`logs/20260920_011638/eval/`（3k）、`logs/20260920_011638/eval_15k/`（15k）、`logs/20260920_011638/eval_paper/`（论文方式）、
`eval_full.py`、`eval_paper.py`、`run_eval.sh`、`run_evalpaper.sh`、`run_play.sh`、`gdb.log`。
以上结果都已下载到本地，服务器可以关机或释放。
