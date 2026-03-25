# RLT 参数说明文档

本文档解释 RL Token 复现中每个模块的关键参数，帮助调试和调优。

---

## 1. `PI05RLTConfig` — 模型配置

文件：`src/lerobot/policies/pi05_rlt/configuration_pi05_rlt.py`

继承自 `PI05Config`（PI0.5 的所有参数），在此基础上新增：

### 1.1 RL Token Encoder（`g_φ`）

| 参数                     | 默认值 | 说明                                                                                                                                      |
| ------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `rlt_encoder_layers`     | `4`    | Transformer 编码器层数。层数越多容量越大，训练越慢。论文中为 4 层。                                                                       |
| `rlt_encoder_hidden_dim` | `512`  | 编码器 Transformer 的隐藏维度。VLM embedding 先被投影到这个维度再编码。                                                                   |
| `rlt_encoder_nheads`     | `8`    | 多头注意力的头数。需整除 `rlt_encoder_hidden_dim`（512/8=64 ✓）。                                                                         |
| `rlt_encoder_dropout`    | `0.1`  | Transformer 内的 dropout。训练初期可以减小（0.05）以加快收敛。                                                                            |
| `rl_token_dim`           | `2048` | RL Token 的输出维度（即 `z_rl` 的维度）。论文指定为 2048，与 `gemma_2b` 的 VLM 维度一致。若使用 `gemma_300m`（width=1024），可改为 1024。 |

### 1.2 RL Token Decoder（`d_φ + h_φ`）

| 参数                     | 默认值 | 说明                          |
| ------------------------ | ------ | ----------------------------- |
| `rlt_decoder_layers`     | `2`    | 因果 Transformer 解码器层数。 |
| `rlt_decoder_hidden_dim` | `512`  | 解码器隐藏维度。              |
| `rlt_decoder_nheads`     | `8`    | 多头注意力头数。              |
| `rlt_decoder_dropout`    | `0.1`  | 解码器 dropout。              |

### 1.3 联合训练损失

| 参数        | 默认值 | 说明                                                                                                                                                                            |
| ----------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rlt_alpha` | `0.5`  | 联合损失权重：`L_total = L_ro + α * L_vla`。`α=0` 表示只训练 RL Token 不微调 VLA；`α=1` 两部分等权重；`α>1` 更强调 VLA 微调。初期建议 0.5，如果 `loss_recon` 下降过慢可以减小。 |

### 1.4 Actor-Critic 架构参数（供 `OnlineRLConfig` 参考）

| 参数                | 默认值    | 说明                                                                                |
| ------------------- | --------- | ----------------------------------------------------------------------------------- |
| `chunk_size_rl`     | `10`      | RL 动作 chunk 长度 C。论文原文 C=10。Critic 计算 Q 值时的折扣跨度为 `γ^C`。         |
| `actor_hidden_dim`  | `512`     | Actor MLP 每层隐藏维度。                                                            |
| `actor_num_layers`  | `3`       | Actor MLP 层数（不含输入/输出层）。论文为 3 层。                                    |
| `critic_hidden_dim` | `512`     | Critic MLP 每层隐藏维度。                                                           |
| `critic_num_layers` | `3`       | Critic MLP 层数。                                                                   |
| `actor_sigma`       | `0.1`     | Gaussian Actor 的固定标准差 σ。控制探索幅度；σ 越大探索越多但越不稳定。             |
| `beta`              | `0.5`     | Actor 损失中参考动作正则化系数：`L_π = -Q + β·                                      |  | a - ã |  | ²`。过大会使 actor 退化为 VLA；过小则失去 VLA 先验。 |
| `ref_dropout`       | `0.5`     | 参考动作的 dropout 概率。每个样本以此概率将整个 `ã` 置零，防止 actor 完全复制 VLA。 |
| `gamma`             | `0.99`    | RL 折扣因子。                                                                       |
| `tau`               | `0.005`   | Target 网络软更新系数：`θ' ← τθ + (1-τ)θ'`。过大目标网络更新太快，不稳定。          |
| `warmup_steps`      | `500`     | Online RL 开始前的 VLA warmup 步数（chunk 步数）。                                  |
| `buffer_capacity`   | `100_000` | Replay Buffer 最大容量（transition 数）。                                           |
| `rl_batch_size`     | `256`     | 每次更新采样的 batch 大小。                                                         |
| `updates_per_step`  | `5`       | 每个环境 chunk step 后的 gradient update 次数（update-to-data ratio G）。           |

### 1.5 Mode Switch MLP

| 参数                     | 默认值 | 说明                         |
| ------------------------ | ------ | ---------------------------- |
| `mode_switch_hidden_dim` | `256`  | Mode Switch MLP 隐藏层维度。 |
| `mode_switch_num_layers` | `2`    | Mode Switch MLP 层数。       |

---

## 2. `OnlineRLConfig` — Online RL 训练配置

文件：`src/lerobot/scripts/lerobot_online_rl.py`

| 参数                      | 默认值                                | 说明                                                                                                                |
| ------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `vla_checkpoint`          | `"outputs/rlt_joint/last_checkpoint"` | 已训练好的 PI05-RLT 模型路径。                                                                                      |
| `actor_critic_checkpoint` | `None`                                | Actor-Critic checkpoint 路径。`None` 表示从随机初始化开始。                                                         |
| `data_path`               | `"outputs/online_rl/data"`            | **统一数据目录**，包含 `meta.json`, `play_buffer.pkl`, `mode_switch.pkl`。                                          |
| `output_dir`              | `"outputs/online_rl"`                 | checkpoint 保存目录。                                                                                               |
| `task`                    | -                                     | 机器人任务的自然语言描述，用于 VLA 推理。                                                                           |
| `chunk_size_rl`           | `10`                                  | RL 动作 chunk 长度 C，应与训练时保持一致。                                                                          |
| `warmup_steps`            | `500`                                 | 前 N 个 chunk step 只跑 VLA 填充 buffer，不进行 RL 更新。                                                           |
| `subsample_stride`        | `2`                                   | Replay Buffer 采样步长。在 RL 阶段，每隔 `stride` 个 chunk 才存入一条 transition。`stride=2` 即 (0,C), (2,C+2), ... |
| `gamma`                   | `0.99`                                | 折扣因子。                                                                                                          |
| `tau`                     | `0.005`                               | Target 网络软更新率。                                                                                               |
| `beta`                    | `0.5`                                 | Actor 参考正则化权重。                                                                                              |
| `ref_dropout`             | `0.5`                                 | 参考动作 dropout 率。                                                                                               |
| `actor_sigma`             | `0.1`                                 | Actor 高斯分布的标准差（固定值）。                                                                                  |
| `rl_batch_size`           | `256`                                 | 每次更新的采样批量大小。                                                                                            |
| `updates_per_step`        | `5`                                   | 每个环境步的 gradient update 次数（G）。                                                                            |
| `buffer_capacity`         | `100_000`                             | Replay Buffer 容量。                                                                                                |
| `actor_lr` / `critic_lr`  | `3e-4`                                | Actor / Critic Adam 学习率。                                                                                        |
| `fps`                     | `10`                                  | 机器人控制频率（Hz）。                                                                                              |
| `max_episodes`            | `1000`                                | 最大训练 episode 数。                                                                                               |
| `episode_timeout_s`       | `30.0`                                | 单个 episode 超时时间（秒），超时自动标记为 failure。                                                               |
| `reset_duration_s`        | `3.0`                                 | episode 结束后机械臂复位的插值时长（秒）。                                                                          |
| `save_freq_episodes`      | `10`                                  | 每隔多少个 episode 保存一次数据和 checkpoint。                                                                      |
| `log_freq`                | `10`                                  | 每隔多少个 chunk 步打印一次日志。                                                                                   |
| `wandb_project`           | `None`                                | WandB 项目名（`None` 表示不上传）。                                                                                 |
| `save_mode_switch_data`   | `False`                               | 是否收集 Mode Switch MLP 训练数据（每帧记录 label 0/1）。                                                           |
| `debug`                   | `False`                               | 开启后每个 chunk step 都打印详细日志。                                                                              |

---

## 3. `TrainPipelineConfig` （联合训练脚本额外参数）

文件：`src/lerobot/scripts/lerobot_train_rlt.py`（通过 CLI 传入）

| 额外 CLI 参数  | 默认值  | 说明                                                                   |
| -------------- | ------- | ---------------------------------------------------------------------- |
| `--freeze_vla` | `false` | 冻结 VLA 参数，只训练 RL Token Encoder/Decoder（适合先单独训练 RLT）。 |
| `--freeze_rlt` | `false` | 冻结 RL Token 参数，只微调 VLA（标准 PI0.5 微调）。                    |

> 两者均为 `false` 时进行联合训练。不可同时为 `true`。

---

## 4. `ModeSwitchTrainConfig` — Mode Switch 训练配置

文件：`src/lerobot/scripts/lerobot_train_mode_switch.py`

| 参数            | 默认值 | 说明                                                                             |
| --------------- | ------ | -------------------------------------------------------------------------------- |
| `data_path`     | -      | 统一数据目录路径（含 `mode_switch.pkl`）。                                       |
| `rl_token_dim`  | `2048` | RL Token 维度，需与 Online RL 保持一致。                                         |
| `state_dim`     | `14`   | 本体感知状态维度（双臂 7+7=14）。                                                |
| `action_dim`    | `7`    | 单臂动作维度。                                                                   |
| `chunk_size_rl` | `10`   | 动作 chunk 长度。                                                                |
| `hidden_dim`    | `256`  | Mode Switch MLP 隐藏维度。                                                       |
| `num_layers`    | `2`    | Mode Switch MLP 层数。                                                           |
| `num_epochs`    | `100`  | 训练轮数。                                                                       |
| `lr`            | `1e-3` | Adam 学习率。                                                                    |
| `val_split`     | `0.1`  | 验证集比例。                                                                     |
| `batch_size`    | `256`  | 训练批量大小。                                                                   |
| `pos_weight`    | `1.0`  | BCE 损失中正样本权重（如正负样本不均衡可调整，例如 RL phase 比例低时增大此值）。 |

---

## 5. 数据目录结构说明

Online RL 运行后，`data_path/` 目录包含：

```
data_path/
├── meta.json           # 元数据
├── play_buffer.pkl     # Replay Buffer（RL transitions）
└── mode_switch.pkl     # Mode Switch 训练数据（可选）
```

### `meta.json` 字段说明

| 字段                  | 说明                                     |
| --------------------- | ---------------------------------------- |
| `robot_type`          | 机器人类型字符串，如 `"so101_follower"`  |
| `buffer_size`         | Replay Buffer 当前存储的 transition 数量 |
| `buffer_capacity`     | Replay Buffer 最大容量                   |
| `rl_token_dim`        | RL Token 维度                            |
| `state_dim`           | 本体感知状态维度                         |
| `chunk_size`          | 动作 chunk 长度                          |
| `action_dim`          | 单步动作维度                             |
| `mode_switch_samples` | Mode Switch 数据条数                     |
| `total_episodes`      | 已完成的 episode 总数                    |
| `total_env_steps`     | 已执行的环境步（chunk 步）总数           |
| `total_updates`       | 已进行的 gradient update 总次数          |

---

## 6. WandB 监控指标

### 联合训练脚本（`lerobot-train-rlt`）

| WandB Key                | 说明                                                       |
| ------------------------ | ---------------------------------------------------------- |
| `train/loss_total`       | 总损失 `L_ro + α * L_vla`                                  |
| `train/loss_vla`         | VLA flow-matching 损失 `L_vla`                             |
| `train/loss_recon`       | RL Token 重建损失 `L_ro`                                   |
| `train/loss_flow_dim{i}` | 第 i 个动作维度的 flow-matching 损失（诊断各关节训练效果） |
| `train/grad_norm`        | 梯度范数（监控梯度爆炸）                                   |
| `train/lr`               | 当前学习率                                                 |
| `train/update_s`         | 每步更新耗时（秒）                                         |

### Online RL 脚本（`lerobot-online-rl`）

| WandB Key              | 说明                                            |
| ---------------------- | ----------------------------------------------- |
| `train/critic_loss`    | Critic 总损失 `loss_q1 + loss_q2`               |
| `train/critic_loss_q1` | Q1 网络损失                                     |
| `train/critic_loss_q2` | Q2 网络损失                                     |
| `train/q1_mean`        | Q1 预测均值（监控 Q 值尺度）                    |
| `train/q_target_mean`  | Target Q 均值                                   |
| `train/actor_loss`     | Actor 总损失                                    |
| `train/actor_loss_q`   | Actor 损失中 `-Q` 项                            |
| `train/actor_loss_ref` | Actor 损失中参考正则化项 `β·                    |  | a-ã |  | ²` |
| `train/actor_q_mean`   | Actor 更新时 Q 均值                             |
| `episode/reward`       | 本 episode 奖励（0 或 1）                       |
| `episode/avg10_reward` | 最近 10 个 episode 的平均奖励（成功率近似）     |
| `episode/total`        | 已完成 episode 总数                             |
| `buffer/size`          | Replay Buffer 当前大小                          |
| `env/mode`             | 当前执行模式（`warmup/vla/actor/intervention`） |

---

## 7. 常见问题与调参建议

### `loss_recon` 不下降
- 尝试减小 `rlt_alpha`（如 0.2）让 encoder-decoder 有更多梯度
- 检查 VLM image token 数量 `M` 是否正确提取（可在 `debug` 模式下打印）
- 尝试减小 encoder/decoder dropout

### `loss_vla` 不下降
- 增大 `rlt_alpha`（如 0.8）
- 检查 VLA 参数是否被正确解冻（非 `freeze_vla` 模式）

### Actor 不离开 VLA 参考
- 减小 `beta`（从 0.5 降到 0.1）
- 减小 `ref_dropout` 则更依赖参考，增大则更独立

### Q 值发散（过大）
- 减小 `gamma`（如 0.95）
- 减小 `tau`（如 0.001）让 target 网络更新更慢
- 减小 `actor_lr`

### 机械臂复位失败
- 确认机器人关节名称包含 `.pos` 后缀（`init_pose` 通过此过滤）
- 检查 `reset_duration_s` 是否足够长
