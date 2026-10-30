# 当前实现：12 关节控制与后腿直立行走

2026-10-30 按用户最新要求实施：直接覆盖原 V20，后续也在当前实现上修改，不建立并存的版本目录或旧代码副本。现有任务 ID、包名和训练/评估入口保留，用于保持调用路径稳定。

## 来源与移植范围

主要来源为 [My_unitree_go2_gym/go2_handstand](https://github.com/yusongmin1/My_unitree_go2_gym/tree/main/legged_gym/envs/Go2_Stand/Go2_Handstand)，对应用户提供的 [演示视频](https://www.bilibili.com/video/BV19fToz9EZ5/)。

上游原文件及 MIT 许可保存在工程的 `third_party/My_unitree_go2_gym/`。活动奖励函数从上游源码提取到 `reference_rewards.py`，参数字面值提取到 `reference_config.py`，以原文件为对照运行正式回归测试。来源文件 Git blob、SHA256、启用权重与适配差异见 [移植清单](reference_port_manifest.json)。

本次没有新增自拟奖励或自行调整上游权重。早期方案中的混合奖励、额外 Air Time、较小 PPO 学习率等候选设计由本页的实际配置替代。

## 当前模型与控制

- 完整模型来自 `/workspace/unitree_ros/robots/go2_description/urdf/go2_description.urdf`，原始资源不修改。
- 全部 12 个关节由策略控制；动作顺序为 FL、FR、RL、RR，每条腿依次 hip、thigh、calf。
- 显式 PD 使用上游 `stiffness=40`、`damping=1`，动作缩放为 `0.25`；力矩、速度限值读取现有 URDF。
- 关节目标为 `0.25 * action + 上游 default_joint_angles`，初始直立姿态使用上游 `descire_joint_angles`。两套角度各有用途，不混用。
- 前腿初始 thigh/calf 为 `0.8 / -1.5 rad`，后腿为 `2.25 / -1.75 rad`；这替代早期方案中的前腿初始角候选。
- 控制周期 `0.02 s`，步态周期 `1.6 s`，机身高度奖励目标 `0.52 m`、前脚高度目标 `0.67 m`、后脚抬脚目标 `0.06 m`，均来自上游。
- 使用已有观测结构扩展到 Actor `56`、Critic `155` 维，保留相位、航向和后脚状态；这与上游的 `45/86` 观测布局不同。
- 镜像按实际关节名映射，独立处理原生观测顺序与动作顺序。包括前腿的左右交换与 hip 符号。

前脚离地、前脚高度、默认姿态、左右关节对称等约束均采用上游已有项，没有额外设计前腿摆动目标。

## 奖励与权重

行走阶段注册上游所有 23 项非零奖励。原来 V20 的载荷分配、额外前摆、合格落脚等奖励已退出当前任务。上游未启用的 `feet_air_time` 不加入当前配置。

| 上游项 | 权重 |
| --- | --- |
| `tracking_lin_vel` | 2.5 |
| `tracking_ang_vel` | 2.5 |
| `lin_vel_z` | 0.2 |
| `ang_vel_xy` | 0.2 |
| `handstand_orientation` | -1 |
| `torques` | -0.0002 |
| `dof_acc` | -2.5e-7 |
| `base_height` | 1.5 |
| `handstand_feet_on_air` | 0.4 |
| `collision` | -2 |
| `action_rate` | -0.05 |
| `default_pos` | -0.1 |
| `default_hip_pos` | -0.1 |
| `feet_clearance` | 0.4 |
| `ang_xz` | -0.5 |
| `contact` | 0.3 |
| `symmetric_joints` | -0.1 |
| `orientation_symmetry` | -0.5 |
| `feet_height_symmetry` | -0.2 |
| `handstand_feet_height_exp` | 5 |
| `default_pos_reward` | 0.5 |
| `dof_pos_limits` | -2 |
| `alive` | 1 |

保留上游的力矩 L1、控制步间速度差计算的关节加速度、前腿姿态及前后腿对称公式。不会因某项与之前优化建议不同而在本轮自行更改它。

为执行已确认的 V21 修正，显式列出三项语义差异：

1. 高度姿态门控按环境独立计算，替代上游跨环境平均值；门控阈值仍为上游的 `0.70`。
2. 右腿支撑相位包含 `phase=0.5`，使相位边界不漏掉支撑腿。
3. 每个控制步先更新当前高度门控，再计算各奖励，消除上游奖励执行顺序导致的旧门控值。

其余变化是接口适配：通过名字重排关节与脚端数据、将标量 alive 扩成每环境值、按环境重置上一控制步关节速度。每步只更新一次速度历史，避免 23 项奖励重复推进状态。

## PPO 与训练阶段

PPO 使用上游隐藏层 `[512, 256, 128]`、ELU、初始标准差 `1.0`、学习率 `1e-3`、adaptive schedule、entropy coefficient `0.01`、每环境 rollout `24` 步。其余 PPO 参数来自同一上游配置。使用当前 RSL-RL 的镜像损失接口，系数 `1.0`；不同时启用镜像数据增强。

当前保留两个阶段；按用户要求停止混合采样改动，Walk 固定速度调整为 `0.2 m/s`。奖励权重保持原值。经用户确认，站立阶段线速度奖励的指数分母 `tracking_lin_vel_sigma` 从 `0.25` 收紧到 `0.04`；角速度和行走阶段线速度仍用 `0.25`。这里的参数直接作为平方误差的分母，不再平方。高度门控、PPO、控制和其他奖励不变。该参数是当前任务的优化值，不是上游原值。

| 阶段 | 命令 | 奖励差异 |
| --- | --- | --- |
| `balance` | 零速度 | 关闭 `contact`、`feet_clearance`，其余 21 项沿用上游 |
| `walk` | 前向 `0.2 m/s`、零横向和转向 | 完整的 23 项上游奖励 |

阶段划分来自 V21 方案。初始状态位于直立附近；首版不实现四足起立课程。地形为平地，材质摩擦与恢复系数随机化采用上游范围；上游其他质量、质心、增益、延迟和推扰随机化尚未移植。这是奖励和控制的移植，不宣称复现上游全部训练条件。

任务使用与上游相同的机身坐标速度奖励：直立时机身 `-Z` 对应水平前进，机身 `+X` 对应竖直方向。源码公式保留，没有替换成自拟的世界速度奖励。阶段命令固定无转向，尚不能据此认定停止、转向或起立能力。

## 运行入口

在目标容器中执行：

```bash
cd /workspace/ry_unitree_rl_lab
./ry_python.sh scripts/train_v20.py --stage balance \
  --num_envs 4096 --max_iterations 15000 --run_name balance \
  --headless --device cuda:0
```

以上为正式训练命令示例；本次验证只进行少量迭代。训练完成后，在同一入口恢复当前 12 动作权重并切换阶段：

```bash
./ry_python.sh scripts/train_v20.py --stage walk \
  --resume logs/rsl_rl/ry_unitree_go2_handstand/<平衡训练目录>/model_<迭代>.pt \
  --num_envs 4096 --max_iterations 15000 --run_name walk \
  --headless --device cuda:0
```

旧的 6 动作权重与新网络不兼容，不再要求 `--warm_start_actor`。已有权重与日志作为实验记录保留，未删除，也不作为当前任务依赖。

评估使用同一阶段和当前权重：

```bash
./ry_python.sh scripts/eval_v20.py --stage walk \
  --checkpoint logs/rsl_rl/ry_unitree_go2_handstand/<训练目录>/model_<迭代>.pt \
  --num_envs 16 --steps 1000 --speed 0.2 \
  --out logs/evaluation/walk --headless --device cuda:0
```

不传 checkpoint 时为零动作执行检查，不算训练策略验收。评估增加前脚触地与速度误差统计，前进速度按命令误差验收；正常超时与异常终止分开记录。原有独立合格落脚、交替率和滑移诊断仍保留，没有为了匹配奖励而放宽这些证据。

## 代码位置与验证

- `ry_go2_biped_walk_v20/`：当前唯一任务实现，历史路径名保留。
- `reference_config.py`、`reference_rewards.py`：从上游提取的配置与公式。
- `gait_mdp.py`：Isaac Lab 数据适配、状态历史和观测。
- `robot_cfg.py`、`gait_env_cfg.py`、`symmetry.py`：完整模型、阶段配置和镜像。
- `scripts/train_v20.py`、`scripts/eval_v20.py`：当前训练与评估入口。
- `tests/test_reference_port.py`：上游公式/参数、门控、相位、镜像回归测试。

验证记录见 [当前实现验证报告](V21_实现与验证.md)。短训练、导出和有限值检查确认执行链路；只有充分训练后通过独立步态验收，才认定策略达到目标效果。
