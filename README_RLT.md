# RL Token (RLT) — Reproduction on Evo-RL / LeRobot

> **基于 PI0.5 复现 PI 团队最新工作《Learning to Reinforce with Latent Representations》（RL Token）**  
> 在 [Evo-RL](https://github.com/EvoRL) 修改的 [LeRobot](https://github.com/huggingface/lerobot) 框架下实现。

---

## 目录

- [方法概述](#方法概述)
- [整体架构](#整体架构)
- [新增文件总览](#新增文件总览)
- [安装](#安装)
- [完整训练与部署流程](#完整训练与部署流程)
  - [Step 1: 数据收集（遥操作）](#step-1-数据收集遥操作)
  - [Step 2: VLA + RL Token 联合训练](#step-2-vla--rl-token-联合训练)
  - [Step 3: Online RL 训练](#step-3-online-rl-训练)
  - [Step 4（可选）: Mode Switch MLP 训练](#step-4可选-mode-switch-mlp-训练)
  - [Step 5: 策略部署](#step-5-策略部署)
- [脚本参数详解](#脚本参数详解)
- [关键设计说明](#关键设计说明)
- [论文参数参考](#论文参数参考)

---

## 方法概述

RL Token 的核心思路：**让 Actor-Critic 网络高效利用 VLA（Vision-Language-Action）模型中的高维视觉信息**。

传统 Actor-Critic 难以直接消费像素或高维图像特征。RL Token 方法通过在 VLA 骨干网络上附加一个 Encoder-Decoder 结构：

1. **Encoder** `g_φ`：将 VLM 最后一层的图像 embedding `z_{1:M}` 压缩成单一 RL Token `z_rl`（维度 2048）。
2. **Decoder** `d_φ + h_φ`：从 `z_rl` 自回归重建 `z_{1:M}`，通过最小化重建误差 `L_ro` 保证 RL Token 包含足够信息。
3. **Actor-Critic**：以 `x = {z_rl, s^p}` 作为输入（`s^p` 为本体感知状态），使用 TD3 风格的 Double Q Critic 和 Gaussian Actor。Actor 以 VLA 动作 `ã` 作为参考，通过 Reference Dropout 防止过拟合。
4. **Mode Switch MLP**（可选）：自动判断当前是否需要 RL Actor 接管。

整体联合损失：
```
L_total = L_ro + α × L_vla
```

---

## 整体架构

```
                ┌──────────────────────────────────────────────────┐
                │                PI0.5 (PI05)                       │
                │  PaliGemma (VLM) + Gemma Action Expert            │
                │  Flow Matching → action chunk ã_{1:C}             │
                └──────────────────┬───────────────────────────────┘
                                   │ z_{1:M}  (final-layer image embeddings)
                    ┌──────────────▼──────────────┐
                    │  RL Token Encoder  g_φ       │  4-layer Transformer
                    │  z_{1:M} + e_rl → z_rl      │  hidden=512, heads=8
                    └──────────────┬──────────────┘
                                   │ z_rl  (B, 2048)
              ┌────────────────────┼──────────────────────────┐
              │ Decoder (训练时)    │                          │ Actor-Critic (在线RL)
              │ d_φ + h_φ          │                          │
              │ 2-layer Transformer│ x = {z_rl, s^p}         │
              │ L_ro = MSE(pred,z̄)│                          │
              └────────────────────┘    ┌───────────────────────────────┐
                                        │  Double Q Critic  Q_ψ / Q_ψ' │ 3-layer MLP
                                        │  Gaussian Actor  π_θ          │ hidden=512
                                        │  [Optional] Mode Switch MLP  │ 2-layer MLP
                                        └───────────────────────────────┘
```

---

## 新增文件总览

| 文件路径                                                  | 说明                                                                                              |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `src/lerobot/policies/pi05_rlt/__init__.py`               | 包初始化                                                                                          |
| `src/lerobot/policies/pi05_rlt/configuration_pi05_rlt.py` | `PI05RLTConfig`：继承 `PI05Config`，包含所有 RLT 超参数                                           |
| `src/lerobot/policies/pi05_rlt/modeling_pi05_rlt.py`      | 核心模型：`RLTokenEncoder`、`RLTokenDecoder`、`PI05RLTPytorch`、`PI05RLTPolicy`                   |
| `src/lerobot/rl/rlt_actor_critic.py`                      | Actor-Critic 网络：`QNetwork`、`DoubleCritic`、`GaussianActor`、`ModeSwitchMLP`、`RLTActorCritic` |
| `src/lerobot/rl/rlt_buffer.py`                            | 在线 RL Replay Buffer：`RLTReplayBuffer`、`RLTDataManager`（统一数据目录管理）                    |
| `src/lerobot/scripts/lerobot_train_rlt.py`                | VLA + RL Token 联合训练脚本                                                                       |
| `src/lerobot/scripts/lerobot_online_rl.py`                | Online RL 训练脚本（键盘控制、机器人交互）                                                        |
| `src/lerobot/scripts/lerobot_train_mode_switch.py`        | Mode Switch MLP 训练脚本                                                                          |
| `src/lerobot/scripts/lerobot_deploy_rlt.py`               | 策略部署脚本（手动/自动模式切换）                                                                 |
| `RLT_params.md`                                           | 所有参数的详细说明文档（调参指南）                                                                |

**修改的文件：**

| 文件路径                          | 修改内容                                                          |
| --------------------------------- | ----------------------------------------------------------------- |
| `src/lerobot/policies/factory.py` | 注册 `pi05_rlt` 到 `get_policy_class()` 和 `make_policy_config()` |
| `pyproject.toml`                  | 新增 `pi05_rlt` 依赖组和 4 个脚本入口点                           |

---

## 安装

### 1. 基础依赖安装

```bash
# 克隆并安装（包含 pi05_rlt 依赖）
pip install -e ".[pi05_rlt]"
```

`pi05_rlt` 依赖组包含：
- `transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi`（PI0.5 专用 transformers）
- `scipy>=1.10.1,<1.15`
- `pynput>=1.7.0,<2.0.0`（键盘监听，用于 Online RL 和部署脚本）

### 2. 验证安装

```bash
# 验证新脚本可以访问
lerobot-train-rlt --help
lerobot-online-rl --help
lerobot-train-mode-switch --help
lerobot-deploy-rlt --help
```

---

## 完整训练与部署流程

### Step 1: 数据收集（遥操作）

使用现有的遥操作脚本收集机器人演示数据（此脚本无需修改）：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyUSB0 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyUSB1 \
  --dataset.repo_id=my_org/my_task_demos \
  --dataset.task="pick and place the red cube"
```

---

### Step 2: VLA + RL Token 联合训练

从 PI0.5 基础模型出发，联合训练 VLA 微调和 RL Token Encoder/Decoder：

```bash
lerobot-train-rlt \
  --policy.type=pi05_rlt \
  --policy.pretrained_name_or_path=physical-intelligence/pi05-base \
  --dataset.repo_id=my_org/my_task_demos \
  --output_dir=outputs/rlt_joint \
  --training.num_grad_steps=5000 \
  --training.batch_size=32 \
  --policy.rlt_alpha=0.5
```

**可选训练模式（通过 `freeze_vla` / `freeze_rlt` 控制）：**

```bash
# 模式 A：仅训练 RL Token（冻结 VLA）
lerobot-train-rlt \
  --policy.type=pi05_rlt \
  --freeze_vla=true \
  --output_dir=outputs/rlt_only ...

# 模式 B：仅微调 VLA（冻结 RL Token 部分）
lerobot-train-rlt \
  --policy.type=pi05_rlt \
  --freeze_rlt=true \
  --output_dir=outputs/vla_only ...

# 模式 C（默认）：联合训练
lerobot-train-rlt \
  --policy.type=pi05_rlt \
  --output_dir=outputs/rlt_joint ...
```

训练脚本将在 `output_dir` 下保存：
- `last_checkpoint/`：最新 checkpoint
- `best_checkpoint/`：验证集上最优 checkpoint
- `wandb/`（如已配置）：训练曲线

---

### Step 3: Online RL 训练

在真实机器人上进行 Online RL 训练。需要机器人硬件连接。

```bash
lerobot-online-rl \
  --vla_checkpoint=outputs/rlt_joint/last_checkpoint \
  --output_dir=outputs/online_rl \
  --data_path=outputs/online_rl/data \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyUSB0 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyUSB1 \
  --task="pick and place the red cube" \
  --chunk_size_rl=10 \
  --warmup_steps=500 \
  --subsample_stride=2 \
  --reset_duration_s=3.0 \
  --save_mode_switch_data=true
```

**键盘控制说明：**

| 按键           | 功能                                                                  |
| -------------- | --------------------------------------------------------------------- |
| `SPACE`        | 标记本轮 episode **成功**，自动复位手臂并开始下一轮（稀疏奖励 r=1）   |
| `f`            | 标记本轮 episode **失败**，自动复位手臂并开始下一轮（稀疏奖励 r=0）   |
| `r`            | 切换 **RL 阶段**（ON/OFF）；OFF 时策略由 VLA 执行，ON 时由 Actor 执行 |
| `i`            | 切换**人工干预模式**（teleop 接管机器人控制）                         |
| `q` / `Ctrl+C` | 退出                                                                  |

**训练阶段说明：**

1. **Warmup 阶段**（前 `warmup_steps` 步）：由 VLA 执行动作，填充 Replay Buffer
2. **VLA 阶段**（RL 阶段 OFF）：由 VLA 执行动作，Mode Switch 数据标签 = 0
3. **RL 阶段**（按 `r` 切入）：由 Actor `π_θ` 执行动作，Mode Switch 数据标签 = 1；以 `subsample_stride` 步长对 replay buffer 进行降采样
4. **干预时**（按 `i`）：人类 teleop 接管，其动作记录为 transition 的 `ref_action`；teleop 镜像机器人运动
5. **每步更新**：执行 G=`updates_per_step` 次 Critic + Actor 更新
6. **episode 结束**：按 `SPACE`（成功）或 `f`（失败）或超时（失败）→ 机器人自动插值复位到初始姿态

**统一数据目录** (`data_path/`)：

```
data_path/
├── meta.json          ← 训练元信息（步数、episode数、buffer大小等）
├── play_buffer.pkl    ← RL Replay Buffer
└── mode_switch.pkl    ← Mode Switch 训练数据（每帧记录，含 label 0/1）
```

中断后可通过 `--data_path` 恢复训练：

```bash
lerobot-online-rl \
  --data_path=outputs/online_rl/data \
  --actor_critic_checkpoint=outputs/online_rl/actor_critic_step_500.pth \
  ...（其他参数不变）
```

---

### Step 4（可选）: Mode Switch MLP 训练

在收集了足够的 Mode Switch 数据后（Online RL 中设置 `--save_mode_switch_data=true`），训练一个自动模式切换分类器：

```bash
lerobot-train-mode-switch \
  --data_path=outputs/online_rl/data \
  --output_dir=outputs/mode_switch \
  --rl_token_dim=2048 \
  --state_dim=14 \
  --action_dim=7 \
  --chunk_size_rl=10 \
  --hidden_dim=256 \
  --num_layers=2 \
  --steps=2000 \
  --lr=3e-4
```

> **说明**：`--data_path` 指向 Online RL 的统一数据目录，脚本将自动读取其中的 `mode_switch.pkl`。
> 该文件中每帧均有标签：`1` = RL 阶段，`0` = VLA 阶段。

训练完成后将保存：
- `mode_switch_best.pth`：验证集最优模型
- `mode_switch_final.pth`：最终模型

---

### Step 5: 策略部署

部署训练好的 RLT 策略进行测试：

**手动模式**（通过按键手动切换 VLA/RL）：

```bash
lerobot-deploy-rlt \
  --vla_checkpoint=outputs/rlt_joint/last_checkpoint \
  --actor_critic_checkpoint=outputs/online_rl/actor_critic_best.pth \
  --mode=manual \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyUSB0 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --task="pick and place the red cube"
```

**自动模式**（需要先完成 Mode Switch MLP 训练）：

```bash
lerobot-deploy-rlt \
  --vla_checkpoint=outputs/rlt_joint/last_checkpoint \
  --actor_critic_checkpoint=outputs/online_rl/actor_critic_best.pth \
  --mode_switch_checkpoint=outputs/mode_switch/mode_switch_best.pth \
  --mode=auto \
  ...（其他参数）
```

**部署键盘控制（手动模式）：**

| 按键 | 功能                          |
| ---- | ----------------------------- |
| `m`  | 切换 VLA 模式 ↔ RL Actor 模式 |
| `r`  | 复位机器人                    |
| `q`  | 退出                          |

---

## 脚本参数详解

### `lerobot-train-rlt`

| 参数                               | 默认值     | 说明                                  |
| ---------------------------------- | ---------- | ------------------------------------- |
| `--policy.type`                    | `pi05_rlt` | 策略类型，必须为 `pi05_rlt`           |
| `--policy.pretrained_name_or_path` | -          | PI0.5 预训练模型路径或 HuggingFace ID |
| `--policy.rlt_alpha`               | `0.5`      | 联合损失中 VLA 损失的权重 α           |
| `--policy.rlt_encoder_layers`      | `4`        | Encoder Transformer 层数              |
| `--policy.rlt_decoder_layers`      | `2`        | Decoder Transformer 层数              |
| `--policy.rl_token_dim`            | `2048`     | RL Token 的维度                       |
| `--freeze_vla`                     | `false`    | 是否冻结 VLA 参数（只训练 RLT）       |
| `--freeze_rlt`                     | `false`    | 是否冻结 RLT 参数（只微调 VLA）       |

### `lerobot-online-rl`

| 参数                        | 默认值                   | 说明                                                                   |
| --------------------------- | ------------------------ | ---------------------------------------------------------------------- |
| `--vla_checkpoint`          | -                        | PI05-RLT 模型 checkpoint 路径                                          |
| `--actor_critic_checkpoint` | `None`                   | Actor-Critic checkpoint（None=从头开始）                               |
| `--data_path`               | `outputs/online_rl/data` | 统一数据目录（含 replay buffer、mode switch data、meta.json）          |
| `--task`                    | -                        | 任务描述语句                                                           |
| `--chunk_size_rl`           | `10`                     | RL 动作 chunk 长度 C                                                   |
| `--warmup_steps`            | `500`                    | Warmup 步数 N_warmup                                                   |
| `--subsample_stride`        | `2`                      | RL 阶段写入 replay buffer 的步长（每 N 个 chunk 写一次，减少冗余数据） |
| `--reset_duration_s`        | `3.0`                    | episode 结束后机器人插值复位所需的时间（秒）                           |
| `--updates_per_step`        | `5`                      | 每步更新次数 G                                                         |
| `--gamma`                   | `0.99`                   | 折扣因子 γ                                                             |
| `--beta`                    | `0.5`                    | Actor 参考正则化权重 β                                                 |
| `--ref_dropout`             | `0.5`                    | Reference Dropout 概率                                                 |
| `--actor_sigma`             | `0.1`                    | Gaussian Actor 固定标准差 σ                                            |
| `--save_mode_switch_data`   | `false`                  | 是否收集 Mode Switch 训练数据（每帧均记录，label=0/1）                 |
| `--wandb_project`           | `None`                   | WandB 项目名（None=不启用 WandB 日志）                                 |
| `--wandb_entity`            | `None`                   | WandB entity（组织/用户名）                                            |
| `--wandb_run_name`          | `None`                   | WandB run 名称                                                         |
| `--debug`                   | `false`                  | Debug 模式（不连接机器人，打印详细信息）                               |

### `lerobot-deploy-rlt`

| 参数                        | 默认值   | 说明                                        |
| --------------------------- | -------- | ------------------------------------------- |
| `--mode`                    | `manual` | 模式切换方式：`manual` 或 `auto`            |
| `--vla_checkpoint`          | -        | PI05-RLT checkpoint 路径                    |
| `--actor_critic_checkpoint` | -        | Actor-Critic checkpoint 路径                |
| `--mode_switch_checkpoint`  | `None`   | Mode Switch MLP checkpoint（auto 模式必需） |

---

## 关键设计说明

### 1. PI05RLTPolicy 继承结构

```
nn.Module
  └── PreTrainedPolicy
        └── PI05Policy          (src/lerobot/policies/pi05/)
              └── PI05RLTPolicy (src/lerobot/policies/pi05_rlt/)
```

`PI05RLTPolicy` 覆盖 `forward()` 方法来计算联合损失，并增加 `extract_rl_token()` 方法供 Online RL 使用。内部的 `PI05RLTPytorch` 在 `PI05Pytorch` 基础上增加了 `rlt_encoder` 和 `rlt_decoder` 模块。

### 2. RL Token 的提取流程

```
输入观测 (images, language)
     │
     ▼
embed_prefix()          # PaliGemma embedding 层
     │
     ▼
paligemma LM forward    # 前向通过 VLM Transformer 层
     │
     ▼
z_image = prefix_out[:, :M, :]   # 取出图像 token 的最后层输出
     │
     ▼
rlt_encoder(z_image)    # 4-layer Transformer → z_rl (B, 2048)
```

### 3. Stop-Gradient 机制

重建损失中对目标 `z̄_i` 使用 `.detach()` 实现 stop-gradient，防止梯度通过 Decoder 反向传播到 `z_image`，从而分离 RL Token 压缩目标与 VLM 微调目标：

```python
z_image_sg = z_image.detach()  # stop-gradient
predictions = rlt_decoder(rl_token, z_image_sg)
recon_loss = F.mse_loss(predictions, z_image_sg)
```

### 4. Double Critic（TD3 风格）

使用两个独立的 Q 网络 `Q1, Q2`（各有对应 target network `Q1', Q2'`）。Target Q 值取 `min(Q1', Q2')` 防止高估：

```python
target_q = min(Q1'(x', a'), Q2'(x', a'))
Q_hat = Σ γ^t r_t + γ^C × target_q
```

### 5. Reference Dropout

训练 Actor 时，以概率 `ref_dropout=0.5` 将整个参考动作 `ã` 置零，防止 Actor 完全复制 VLA 的动作：

```python
mask = (torch.rand(B) > ref_dropout).float()
a_ref_masked = a_ref * mask.unsqueeze(-1).unsqueeze(-1)
```

---

## 论文参数参考

以下参数直接来自原论文或基于论文估计：

| 参数                   | 值         | 来源     |
| ---------------------- | ---------- | -------- |
| RL Token 维度          | 2048       | 论文原文 |
| RL Token 数量          | 1          | 论文原文 |
| 动作 Chunk C           | 10         | 论文原文 |
| Actor MLP 层数         | 3          | 论文原文 |
| Actor/Critic 隐藏维度  | 512        | 论文原文 |
| Reference Dropout      | 50%        | 论文原文 |
| Warmup 步数 N          | 500        | 论文估计 |
| Update-to-data ratio G | 5          | 论文估计 |
| 联合训练 α             | 0.5        | 经验值   |
| Actor 参考正则化 β     | 0.5        | 经验值   |
| Encoder: 层数/heads    | 4 / 8      | 经验值   |
| Decoder: 层数/heads    | 2 / 8      | 经验值   |
| Mode Switch MLP 层数   | 2          | 经验值   |
| Mode Switch 隐藏维度   | 256        | 经验值   |
| 联合训练步数           | 2000–10000 | 论文原文 |
| 折扣因子 γ             | 0.99       | 标准 RL  |
| Target 网络软更新 τ    | 0.005      | TD3 默认 |

---

## 详细参数文档

所有脚本的完整参数说明、取值范围、调参建议以及常见问题排查，请参见：

**[RLT_params.md](RLT_params.md)**

该文档覆盖：
- `PI05RLTConfig`（模型超参数）
- `OnlineRLConfig`（在线 RL 参数）
- `ModeSwitchTrainConfig`（Mode Switch 训练参数）
- 统一数据目录结构与 `meta.json` 字段说明
- WandB 指标索引（train/\* 和 episode/\* 全部字段）
- 常见调参问题与建议

---

## 引用

如果您使用了本仓库的代码，请同时引用原始论文：

```bibtex
@article{rlt2025,
  title={Learning to Reinforce with Latent Representations},
  author={Physical Intelligence},
  year={2025}
}
```

以及 LeRobot 框架：

```bibtex
@misc{lerobot,
  title={LeRobot: State-of-the-art Machine Learning for Real-World Robotics},
  author={HuggingFace},
  year={2024},
  url={https://github.com/huggingface/lerobot}
}
```
