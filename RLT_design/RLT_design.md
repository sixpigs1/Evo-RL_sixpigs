# RL Token Design

这是复现 RL Token 时的设计文档，这里会记录论文中的关键信息，算法结构和需要完成的工作。

## 整体架构

由两个主要部分组成：VLA with RL Token 和 Actor - Critic Network。具体细节将在 组件细节 部分介绍。

### VLA with RL Token

- 基于 Pi 0.5 Model (论文原文中使用 Pi0.6, 但是 Pi0.6 未开源所以我们采用 Pi 0.5 Model, 整体结构应该是基本一样的), 在此基础上增加了一个 Encoder - Decoder 模块。 
  - Encoder 接受 VLA 中的 VLM 的 final layer 的 image embeddings, 输出 RL Token。
  - Decoder 接受 RL Token, 还原成 Encoder 输入的 image embeddings, 并通过与原本的 embedding 进行对比从而最小化 RK Token 的压缩信息损失。

### Actor - Critic Network

- 采用 TD3 的结构 (Double Q Network, use smaller Q value)
  - Critic Network 来预测 Q Value
  - Actor Network 来给出动作微调（拟合一个高斯分布）

## 方法和组件细节

### Chunk-Level RL 

- 在每一个 episode 结束时进行 sparse binary reward. $r_T = 1$ if success, $r_T = 0$ else.
- Chunked policy: $\pi(\textbf{a}_{t:t+C-1}|\textbf{s}_t)$ where $C < H$
- Chunked action value function: $$Q^\pi(\textbf{s}_t,\textbf{a}_{t:t+C-1}) = \sum_{t^\prime=t}^{t+C-1}\gamma^{t^\prime-t}r_{t^\prime} + \gamma^C\mathbb{E}_{a^\prime\sim\pi|s_{t+C}}[Q^\pi(\textbf{s}_{t+C},\textbf{a}^\prime)]$$

### RL Token 

- 改进 VLA 结构来从中提取 vision 等关键高维信息为 RL Token 用于 Actor-Critic Network (解决了 actor-critic 难以高效利用高维信息的问题) 
- 采用 encoder - decoder 结构
  - Encoder 是 Transformer $g_\phi$，学习从 image embeddings $\textbf{z}_{1:M} = f(s,l,\theta_{\text{vla}})$ 和 learned embedding $\textbf{e}_{rl} = \textbf{e}_\phi(<rl>)$ 提炼 RL Token $\textbf{z}_{rl}$
  - Decoder 是 Transformer $d_\phi$ 和 linear output projection $h_\phi$ 来将 RL Token 还原为 $\textbf{z}_{1:M}$
- 学习目标 autoregressive reconstruction objective 是最小化 Decoder 还原的结果和原先 image embedding 的差距，从而最小化压缩损失
  - Loss: $$\mathcal{L}_{ro} = \mathbb{E}_{\mathcal{D}}\big[ \sum_{i=1}^M\| h_\phi(d_\phi([\textbf{z}_{rl},\bar{\textbf{z}}_{1:i-1}])) - \bar{\textbf{z}}_i \|^2 \big]$$
    - $\bar{\textbf{z}}_i = \text{sg}(\textbf{z}_i)$ 表示 stop-gradient (不进行梯度传播，作为常量处理)
  - VLA finetuning 和 RL Token training 共同训练: $$\phi,\theta_{vla} = \arg\min_{\phi,\theta_{\text{vla}}} \mathcal{L}_{ro}(\phi) + \alpha \mathcal{L}_{\text{vla}}(\theta_{\text{vla}})$$

### Actor - Critic Network

- Pre Definition: input state $\textbf{x} = \{\textbf{z}_{rl}, \textbf{s}^p\}$, 其中 $\textbf{z}_{rl}$ 是 RL Token, $\textbf{s}^p$ 是 proprioceptive state information
- Critic Network $Q_\psi$ and Target Network $Q_{\psi^\prime}$
  - Loss: $$\mathcal{L}_Q = \mathbb{E}_{(\textbf{x},\textbf{a}_{1:C},\textbf{}^\prime)\sim\mathcal{B}} \big[ \big(\hat{Q} - Q_\psi(\textbf{x},\textbf{a}_{1:C})\big)^2 \big]$$
  - Target Q (use smaller Q in Double Q strategy): $$\hat{Q} = \sum_{t^\prime=1}^C \gamma^{t^\prime-1}r_{t^\prime} + \gamma^C \mathbb{E}_{\textbf{a}^\prime\sim\pi_\theta}\big[ Q_{\psi^\prime}(\textbf{x}^\prime,\textbf{a}^\prime) \big]$$
- Actor Network: $\pi_\theta(\cdot|\textbf{x},\tilde{\textbf{a}}_{1:C}) = \mathcal{N}\big( \mu_\theta(\textbf{x},\tilde{\textbf{a}}_{1:C}),\sigma^2 \textbf{I} \big)$ 拟合一个高斯分布来对 VLA 输出的 action 进行纠偏
  - Loss: $$\mathcal{L}_\pi(\theta) = \mathbb{E}_{\textbf{s}\sim\mathcal{B}, \textbf{a}_{1:C}\sim\pi_\theta}\big[ -Q_\psi(\textbf{x},\textbf{a}_{1:C}) + \beta\| \textbf{a}_{1:C} - \tilde{\textbf{a}}_{1:C} \|_2^2 \big], \quad \tilde{\textbf{a}}_{1:C} \sim \pi_{\text{vla}}(\cdot|\textbf{s},l)$$ Loss 结合 Q value 并且尽量不要过于偏离 VLA 选择采取的动作
  - Reference drop out: 每个 reference action $\tilde{a}$ 有随机概率被 mask 成 0 以避免 actor 完全仿照 VLA 动作

### 完整系统的其他细节

- Warmup: 初期先不进行 online RL, 先跑一些 deploy 来 prefill RL Token
- Rollout: 采集 $\textbf{a}$时有 wramup/rollout/intervention 三种状态
- Subsampling action chunks: 以 step stride 2 来采集数据，增加数据丰富度
- Update: 异步进行 rollout 和 learning, update-to-data ratio 5
- Targeted improvement of critical phases: 对关键部分进行特定改进，只对需要精细操作的部分实施 online RL, 可以由 VLA 进行自动切换

## 算法流程与结构

### Data Collection with Teleoperation

由 src/lerobot/scripts/lerobot_teleoperate.py 脚本实现。应该基本不需要修改。我们的测试将首先在 SO101 机械臂上进行。

### VLA Fintuning & RL Token Training

这是关键部分一。

你需要首先实现一个新的 VLA with RL Token 模型(基于 Pi0.5 Model)。新的模型应该在 src/lerobot/policies 中实现（你可以参考这个文件夹下其他模型的实现情况，尤其是Pi0.5本身）。另外你也可以阅读 Pi0.5 在 lerobot 框架下实现的文档，在 docs/source/pi05.mdx 中有相关介绍。

再实现这个模型对应的训练脚本，来进行 VLA finetuning 和 RL Token training 的联合训练。src/lerobot/scripts/lerobot_train.py 是原本模型的训练脚本，你可以查看参考。你需要实现一个新的训练脚本来进行训练。也要提供选项来尝试先训练 VLA 再进行 RL Token 训练(即可以冻结那一个部分进行训练)。我们将从 Pi0.5 base model 和 随机初始化参数的 encoder-decoder 开始训练。

下面是训练算法：
- Train $\phi$ using $\textbf{z}_i = f_i(s,l,\theta_{\text{vla}})$, $\textbf{rl}=g_\phi(\textbf{z}_{1:M},\textbf{e}_{rl})_{M+1}$ and $\theta_{\text{VLA}}$ (only if $\alpha>0$)
  - $$\mathcal{L}_{ro} = \mathbb{E}_{\mathcal{D}}\big[ \sum_{i=1}^M\| h_\phi(d_\phi([\textbf{z}_{rl},\bar{\textbf{z}}_{1:i-1}])) - \bar{\textbf{z}}_i \|^2 \big]$$
- $$\phi,\theta_{vla} = \arg\min_{\phi,\theta_{\text{vla}}} \mathcal{L}_{ro}(\phi) + \alpha \mathcal{L}_{\text{vla}}(\theta_{\text{vla}})$$

### Online RL (Actor-Critic Network Training)

这是关键部分二。

你需要首先实现一个新的 Actor-Critic Network 模型，来进行 online RL。Critic 需要使用 Double Q Network 结构, 每个 Q Network 都有一个 Target Network。Actor Network 需要拟合一个高斯分布来对 VLA 输出的 action 进行纠偏。现在这个框架下已有 src/lerobot/rl/actor.py, src/lerobot/rl/buffer.py 等文件，你可以看一下这些文件是否适合使用，如果不适合，你可以自己创建新的文件。

其次你需要实现一个新的训练脚本，来进行 Online RL 训练。训练实现脚本写到 src/lerobot/scripts/lerobot_online_RL.py 中。人类可以通过手动按下一个按钮(例如，空格)来进入 online RL 学习状态。

应当将 Replay Buffer 保存下来以备进行 online RL 继续恢复训练。

同时你也应当保存每一帧的 $(\textbf{x}_t, \tilde{\textbf{a}})$ 数据用来训练 Mode Switch Network。是否保存这个数据是 optional 的。

Online RL 算法具体步骤如下：
- Initialize critic networks $Q_\psi$ and RL Policy $\pi_\theta$
- for environment steps t = C, 2C .. do
  - Sample VLA reference chunk $\tilde{\textbf{a}}_{t:t+C-1} \sim \pi_{\text{vla}}(s_t)$.
  - From RL state $\textbf{x} = (\textbf{z}_{rl}(\textbf{s}_t), \textbf{s}_t^p)$
  - $\textbf{a}_{t:t+C-1} \leftarrow 1. \textbf{a}^{\text{human}} \text{if intervention; } 2. \tilde{\textbf{a}}_{t:t+C-1} \text{if } t < N_{\text{warmup}}; 3. \sim \pi_\theta(\cdot|\textbf{x}_t,\tilde{\textbf{a}}) \text{ortherwise}$ 
  - Execute $\textbf{a}_{t:t+C-1}$ and observe $r_t, \textbf{s}_{t+1}, \textbf{s}_{t+1}^p$
  - $\tilde{\textbf{a}}_{t:t+C-1} \leftarrow \textbf{a}^{\text{human}}$ if intervention
  - Store transition in $\mathcal{B}$: $<\textbf{x}_t, \textbf{a}_{t:t+C-1}, \tilde{\textbf{a}}, r_t, \textbf{x}_{t+1}>$
  - for g = 1, .. ,G do
    - Sample batch of data $b \sim \mathcal{B}$
    - Compute target Q value (choose smaller Q after computing target): $$\hat{Q} = \sum_{t^\prime=1}^C \gamma^{t^\prime-1}r_{t^\prime} + \gamma^C \mathbb{E}_{\textbf{a}^\prime\sim\pi_\theta}\big[ Q_{\psi^\prime}(\textbf{x}^\prime,\textbf{a}^\prime) \big]$$
    - Train Critic with TD backup $$\mathcal{L}_Q(\psi) = \mathbb{E}_{(\textbf{x},\textbf{a}_{1:C},\textbf{}^\prime)\sim\mathcal{B}} \big[ \big(\hat{Q} - Q_\psi(\textbf{x},\textbf{a}_{1:C})\big)^2 \big]$$
    - Train policy $\textbf{a} \sim \pi_\theta(\cdot|\textbf{s}, \tilde{\textbf{a}})$: $$\mathcal{L}_\pi(\theta) = \mathbb{E}_{\textbf{s}\sim\mathcal{B}, \textbf{a}_{1:C}\sim\pi_\theta}\big[ -Q_\psi(\textbf{x},\textbf{a}_{1:C}) + \beta\| \textbf{a}_{1:C} - \tilde{\textbf{a}}_{1:C} \|_2^2 \big]$$
  - end for
- end for

### (Optional) Mode Switch Network Training

这个用来训练一个额外的 MLP 从 RL Token, MLP 输入与 actor/critic network 的输入相同都是 $(\textbf{x}_t, \tilde{\textbf{a}})$, 输出是一个 binary (0/1) 的 mode 选择。

你需要完成对应的 MLP 模型的视线和训练脚本。

### Policy Deployment with RL

你需要实现一个脚本，来进行 policy deployment。这个脚本用来部署测试 online RL 结束后的 policy。可以通过手动或自动两种方式来切换哪个部分需要进行 RL 精细操控。

手动控制通过一个按钮来切换，自动控制需要完成上一步 Mode Switch Network 的训练。在 VLA 完成推理后 actor 推理和 Mode Switch MLP 的推理同步进行。如果 MLP 输出 1 则机械臂执行 actor 纠正过的动作，如果为 0 则执行 VLA 输出的动作。

## 你需要完成的任务整理

大部分工作在算法和流程结构中已经有叙述，这里再总结一下

1. 仔细阅读 RLT_design.md 文件, 查看论文和插图, 熟悉我们需要完成的工作
2. 项目相对比较复杂，查看整体文件结构，尤其是 scripts, policy, rl 等关键文件夹下的内容
3. 检查 src/lerobot/scripts/lerobot_teleoperate.py 脚本是否可以正常进行数据收集
4. 实现 VLA with RL Token 模型架构，即基于 Pi0.5 整体结构增加 encoder-decoder 结构
5. 实现 VLA finetuning 和 RL Token training 联合训练脚本
6. 实现 Actor-Critic Network 模型，包括 Critic 网络(Double Q) 和 Actor 网络
7. 实现 Online RL 训练脚本，包括 Replay Buffer 保存，Mode Switch Network 训练
8. 实现 Mode Switch MLP 模型和对应的训练脚本
9. 实现 Policy Deployment 脚本，包括手动/自动模式切换，RL 精细操控
10. 检查项目的依赖和安装问题是否正常处理
11. 撰写 README_RLT 来总结你的代码应当如何运行。你可以参考现有的 README.md

## 其他注意事项

- 如果现在已有的代码和我们的算法设计不兼容，你可以根据文件结构在对应的地方重写一个代码，尽量不要修改原来的代码格式。
- 你需要查看 requirements-ubuntu.txt 等文件来查看这个项目的依赖，写代码的时候需要注意需要尽量兼容这里面的版本
- 你也需要查看 setup.py 和 pyproject.toml 等文件，确保我们的依赖可以通过 `pip install -e .` 来全部安装，需要运行的脚本也可以
- 由于是复现，可能会遇到各种参数设置不合理，代码有问题等情况。请尽可能记录可能对于参数设置有帮助的信息，以便后续的调试和优化。

### 参考参数

论文中给出了一些实现细节的参数，你可以作为 default parameters
  - Base VLA finetuning / RL token Training: 2000 - 10000 gradient steps
  - MLP scale: Three-layer, hidden dimension 512 for screw
  - Reference drop out mask ratio: 50%
  - RL action chunk: C = 10
  - Sparse +1 reward is provided by operator after RL tesk completed
  - RL Token dim: 1 * 2048

其他一些参数论文中没有给出，我简单想了一下，你可以根据我的想法和你对此的思考来调整
  - joint training 中的 $\alphs$: 0.5
  - actor training 中的 $\beta$: 0.5
  - Encoder Transformer: 4 layers, hidden dimension 512, attention heads 8
  - Decoder Transformer: 2 layers, hidden dimension 512
  - Mode Switch MLP: 2 layers, hidden dimension 256

-----------------------------------------

## 可能还需要修改的地方

我查看了你的代码，完成的很好，但是可以考虑再去检查这些地方：

- 因为这个复现中很多参数都尚不清楚，代码也可能有问题。所以需要进行广泛的测试，请检查你的代码确保训练过程等关键数据被记录下来并可以上传到 wandb 等平台进行查看。例如有些 loss 由两部分组成的时候(例如: joint training of VLA and encoder-decoder; actor training)你可以在原有总和 loss 的基础上分别记录两个部分的 loss 来帮助我检查每个部分的训练效果。另外，其他代码也可以提供 --debug 模式来输出一些必要的细节。
- 为了方便我调试等，请你写一个 .md 文档来解释每个模型的参数的情况，简要解释每个参数的作用。
- online RL 是整个流程的核心部分，请具体检查下面这些问题
  - 你需要注意：并不是整一个执行阶段都是需要 RL 的。所以你需要添加一个按键来控制进入和退出 需要RL 阶段。
  - 你在 online RL 阶段似乎加载了 mode switch MLP 的参数, 你有进行训练吗？这个训练似乎是不必要的，因为我们安排了单独对这个模块进行训练的脚本。
  - 为了方便 intervention 的进行，即使还没有处于 intervention 状态, teleoperator 应该跟随 robot 一起动, RECAP 中的 src/lerobot/scripts/lerobot_human_inloop_record.py 可能有相关实现，你可以去查看一下
  - 在按下 space 按键以后是会直接结束 episode, 并标记为 success(reward +1) 吗？可能还需要考虑 1. 结束 episodes 以后让机械臂恢复到操控的初始位置；2. 提供一个按键来结束 episode 并标记为 failure (reward 0)。另外，如果超时自动标记为 failure。
  - 结束 episode 的时候最好可以自动将 teleoperators 和 robots 恢复到初始位置。
  - 我们的 online RL 需要兼容双臂，请检查。
- 关于 online RL 的数据收集，你需要注意：
  - 对于 mode switch MLP training 的数据，你需要在整个 episode 记录每一帧的数据，如果现在正处于 RL 数据收集阶段，则 是否RL 标记为 1，否则标记为 0。这个 0/1 是 mode switch MLP 的学习目标。 
  - 对于放入 replay buffer 的数据，你需要进行 subsample。即在 需要RL 阶段中，你应该每隔 2(这应当是一个可以控制的参数) 个时间戳记录一个数据，i.e. (0, C), (2, C+2), (4, C+4), ...
  - 格式：我注意到在保存 mode switch 数据时候也添加了后缀，但是这似乎是没有必要的。这些数据和 checkpoint 不一样，应当直
  - 0.接一起存储。在保存数据的时候现在比较混乱，我们可以采取一个统一的格式来保存。例如：
    data_path
    - meta.json - 放一些基本的信息，比如使用了什么机械臂，各种数据的 dim，数据量 等。由你来定义这个 .json 数据
    - play_buffer.pkl
    - mode_switch.pkl
- 完成所有更新以后别忘了更新 README_RLT.md 文件中对应的地方。